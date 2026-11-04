#!/usr/bin/env python3
"""Read ONLY saved JSON; separate correct-anchor soft harm from wrong-anchor cases.

No torch, CUDA, image decode, checkpoint tensor, LPIPS recomputation or training.
Wrong-anchor cases still confound choice and mixing until predicted-anchor hard
outputs exist. Never label their entire deficit as a causal classification cost.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parents[1]
ACTIVE_ROOT = '/inspire/hdd/project/multimodal-discrete-diffusion/liuxiaohong-25080/zhenhaowang'
RUN = 'qualification_mixed_driver_4x2_v1'
REPORT = 'reports/foundir_frozen_expert_router_terminal_2026-11-04.json'
REPORT_SHA = '71b718d83997916f0060727728f8525cf4d887db41de35db8e6cdf42aa53c191'
OUTPUT = 'reports/foundir_frozen_router_error_breakdown_2026-11-04.json'
ORDER = ('blur', 'noise', 'haze', 'lowlight', 'rain')
ARMS = ('known_class_control', 'router_step400', 'router_step408')
METRICS = ('psnr', 'lpips')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    path = Path(path)
    require(path.suffix == '.json', 'diagnosis only reads JSON, never tensor/image files')
    require(path.stat().st_size < 16*1024**2, 'unexpectedly large JSON')
    raw = path.read_bytes()
    require(len(raw) < 16*1024**2, 'unexpectedly large JSON')
    def reject(value):
        raise ValueError('nonfinite JSON: '+value)
    return json.loads(raw, parse_constant=reject), hashlib.sha256(raw).hexdigest()


def number(value):
    return type(value) in (int, float) and math.isfinite(value)


def mean(values):
    values = list(values)
    return math.fsum(values)/len(values) if values else None


def check_close(actual, expected, label):
    require(number(actual) and number(expected) and math.isclose(actual, expected, abs_tol=1e-9, rel_tol=1e-10),
            'saved metric/count mismatch: '+label)


def check_route(route, task, *, control, prompts):
    matrix = route['weights']
    require(isinstance(matrix, list) and len(matrix) == 1 and len(matrix[0]) == 5, 'expected five image-level weights')
    weights = matrix[0]
    require(all(number(v) and 0 <= v <= 1 for v in weights) and abs(sum(weights)-1) <= 2e-6, 'invalid probabilities')
    index = max(range(5), key=lambda i: weights[i])  # First index breaks ties, like torch.argmax.
    anchor = ORDER[index]
    require(route['anchor'] == anchor, 'anchor does not match saved probabilities')
    expected = {'prompt': prompts[anchor], 'alpha': (.75, .30, 1., .925, 1.)[index],
        'second_quantization': anchor not in ('haze', 'rain'), 'schedule_steps': 50,
        'model_calls': 26 if anchor == 'rain' else 50,
        'readout': 'endpoint_at_call25' if anchor == 'rain' else 'final_flow',
        'rain_after_call25': 'zero_and_renormalize_nonrain'}
    require(all(route.get(k) == v for k, v in expected.items()), 'prompt/seed-independent protocol mismatch')
    if control:
        require(weights == [float(t == task) for t in ORDER], 'known-class control is not the matching one-hot expert')
    ranked = sorted(weights, reverse=True)
    return {'anchor': anchor, 'correct': anchor == task, 'weights': weights,
            'primary_weight': weights[index], 'other_experts_total_weight': 1-weights[index],
            'true_class_weight': weights[ORDER.index(task)], 'top_two_margin': ranked[0]-ranked[1]}


def gaps(current, baseline):
    # Positive = worse, negative = improvement, for BOTH of these fields.
    # dB differences are not percentages of image error.
    return {'psnr_loss_db': baseline['psnr']-current['psnr'],
            'lpips_increase': current['lpips']-baseline['lpips']}


def group_stats(rows, stage, population):
    key = 'step'+str(stage)
    return {'count': len(rows), 'mean_baseline': {m: mean(r['baseline'][m] for r in rows) for m in METRICS},
        'mean_router': {m: mean(r[key]['metrics'][m] for r in rows) for m in METRICS},
        'mean_primary_weight': mean(r[key]['primary_weight'] for r in rows),
        'mean_other_experts_total_weight': mean(r[key]['other_experts_total_weight'] for r in rows),
        'mean_true_class_weight': mean(r[key]['true_class_weight'] for r in rows),
        'deficit': {m: {'conditional_mean': mean(r[key]['gap_vs_baseline'][m] for r in rows),
            'contribution_to_full_population_mean': math.fsum(r[key]['gap_vs_baseline'][m] for r in rows)/population,
            'positive_harm_sum': math.fsum(max(0., r[key]['gap_vs_baseline'][m]) for r in rows),
            'improvement_sum': math.fsum(max(0., -r[key]['gap_vs_baseline'][m]) for r in rows),
            'harmed_images': sum(r[key]['gap_vs_baseline'][m] > 0 for r in rows)}
            for m in ('psnr_loss_db', 'lpips_increase')}}


def split_groups(rows, stage):
    key = 'step'+str(stage); total = len(rows)
    correct = [r for r in rows if r[key]['correct']]
    wrong = [r for r in rows if not r[key]['correct']]
    groups = {'all': group_stats(rows, stage, total),
              'correct_anchor_soft_vs_same_anchor_onehot': group_stats(correct, stage, total),
              'wrong_anchor_choice_and_mixing_combined': group_stats(wrong, stage, total)}
    for m in ('psnr_loss_db', 'lpips_increase'):
        parts = [groups[k]['deficit'][m]['contribution_to_full_population_mean'] for k in groups if k != 'all']
        check_close(sum(parts), groups['all']['deficit'][m]['conditional_mean'], 'partition identity '+m)
    groups['wrong_to_predicted_confusion'] = {task: dict(Counter(r[key]['anchor'] for r in wrong if r['task'] == task)) for task in ORDER}
    return groups


def analyze(items, records, result, receipt_sha, hashes, prompts):
    require(len(items) == 40 and len({i['key'] for i in items}) == 40, 'exactly 40 unique development items required')
    require(Counter(i['task'] for i in items) == Counter({t: 8 for t in ORDER}), 'eight items per task required')
    require(all(i['split'] == 'validation' and i['row']['split'] == 'validation' for i in items), 'only declared development metadata')
    require(set(records) == set(ARMS), 'missing/extra arm')
    keys = {i['key'] for i in items}
    require(all(set(records[a]) == keys for a in ARMS), 'missing, duplicate or extra evaluation records')
    rows = []
    for item in items:
        row = {'key': item['key'], 'task': item['task'], 'record_id': item['row']['record_id'],
               'source_group': item['row']['source_group'], 'crop_xy': item['row']['crop_xy'], 'seed': item['seed']}
        for arm in ARMS:
            entry = records[arm][item['key']]
            control = arm == 'known_class_control'
            ident = {'receipt_sha256': receipt_sha, 'item': item, 'arm': 'evaluation/'+arm}
            if not control:
                ident['router_hash'] = hashes[int(arm.removeprefix('router_step'))]
            require(entry['identity'] == ident, 'input, checkpoint or run identity changed')
            require(entry['known_class_control_not_deployed'] is control, 'control mislabelled as deployment')
            metrics = entry['metrics']
            require(set(metrics) == set(METRICS) and all(number(v) for v in metrics.values()), 'nonfinite/invalid metrics')
            route = check_route(entry['route'], item['task'], control=control, prompts=prompts)
            if control:
                row['baseline'] = metrics
            else:
                row[arm.removeprefix('router_')] = {**route, 'metrics': metrics, 'gap_vs_baseline': gaps(metrics, row['baseline'])}
        row['change_408_minus_400'] = {m: row['step408']['metrics'][m]-row['step400']['metrics'][m] for m in METRICS}
        rows.append(row)
    # Reconcile every class/arm mean, correct-selection count and regression
    # count against the hash-pinned terminal result, not just a grand mean.
    for stage in (400, 408):
        field = 'before_quality' if stage == 400 else 'after_quality'
        for task in ORDER:
            selected = [r for r in rows if r['task'] == task]
            saved = result[field]['groups'][task]
            require(saved['count'] == 8, 'wrong terminal sample count')
            for arm in ('known_class_control', 'router_step'+str(stage)):
                control = arm == 'known_class_control'; key = 'step'+str(stage)
                for m in METRICS:
                    check_close(mean(r['baseline'][m] if control else r[key]['metrics'][m] for r in selected),
                                saved['means'][arm][m], task+'/'+arm+'/'+m)
                require(saved['router_anchor_correct'][arm] == (8 if control else sum(r[key]['correct'] for r in selected)),
                        'terminal correctness count mismatch')
                if not control:
                    for m, gap in [('psnr', 'psnr_loss_db'), ('lpips', 'lpips_increase')]:
                        require(saved['regressions_vs_known_class'][arm][m] == sum(r[key]['gap_vs_baseline'][gap] > 0 for r in selected),
                                'terminal regression count mismatch')
    transitions = {}
    for name, before, after in [('kept_correct', True, True), ('became_correct', False, True),
                                 ('became_wrong', True, False), ('stayed_wrong', False, False)]:
        selected = [r for r in rows if r['step400']['correct'] == before and r['step408']['correct'] == after]
        transitions[name] = {'count': len(selected), 'keys': [r['key'] for r in selected],
            'mean_change_408_minus_400': {m: mean(r['change_408_minus_400'][m] for r in selected) for m in METRICS}}
    # Only these cases lack the hard-selected version of the model's OWN anchor.
    # Do not force the GT class or search for the best output using GT metrics.
    missing = [{'key': r['key'], 'predicted_anchor': r['step408']['anchor'],
        'hard_weights_from_model_argmax': [float(t == r['step408']['anchor']) for t in ORDER],
        'seed': r['seed'], 'crop_xy': r['crop_xy']} for r in rows if not r['step408']['correct']]
    return {'status': 'saved_json_error_breakdown_complete', 'run_id': RUN,
        'stage400': split_groups(rows, 400), 'stage408': split_groups(rows, 408),
        'per_task': {t: {str(s): split_groups([r for r in rows if r['task'] == t], s) for s in (400, 408)} for t in ORDER},
        'transitions': transitions, 'per_image': rows,
        'regressions_step408_vs400': {'psnr': sum(r['change_408_minus_400']['psnr'] < 0 for r in rows),
                                     'lpips': sum(r['change_408_minus_400']['lpips'] > 0 for r in rows)},
        'missing_counterfactual': {'count': len(missing), 'items': missing, 'launched': False,
            'needed_for': 'Separate wrong-anchor hard-choice deficit from soft-mixture change on those same cases only',
            'GT_used_to_choose_counterfactual_anchor': False},
        'interpretation': ['Correct-anchor group already compares soft mixture to the same-anchor one-hot under matched protocol.',
            'Wrong-anchor group combines choice, protocol and mixture; do not call its entire deficit classification cost.',
            'Signed contributions add to the reported mean gap, but are not causal percentages or percentages of dB.',
            'Step400 versus408 also changes optimizer exposure; this is not a matched VLM causal ablation.',
            'Development singleton diagnosis does not establish official-test or mixed-degradation performance.'],
        'images_read': False, 'checkpoint_tensors_read': False, 'GPU_used': False,
        'formal_training_started': False, 'automatic_counterfactual_launch': False}


def aggregate_only(terminal):
    groups = terminal['result']['after_quality']['groups']
    rows = {}
    for task in ORDER:
        g = groups[task]; base, before, after = (g['means'][a] for a in ARMS)
        rows[task] = {'count': g['count'], 'correct_anchor_final': g['router_anchor_correct']['router_step408'],
            'gap_final_vs_onehot': gaps(after, base),
            'change_408_minus_400': {m: after[m]-before[m] for m in METRICS},
            'all_correct_group_means_isolate_mixing': g['router_anchor_correct']['router_step408'] == g['count']}
    return {'status': 'aggregate_only_per_image_breakdown_pending', 'per_task': rows,
            'missing_final_predicted_hard_outputs': sum(g['count']-g['router_anchor_correct']['router_step408'] for g in groups.values()),
            'warning': 'Cannot infer per-image probabilities, which images switched, or wrong-group error contribution from these class means.'}


def diagnose(root):
    root = Path(root).resolve()
    terminal, digest = read_json(root/REPORT)
    require(digest == REPORT_SHA and terminal['run_id'] == RUN, 'wrong terminal report')
    require(terminal['status'] == 'completed_all_four_nodes_status0' and len(terminal['attempts']) == 4
            and {r['node'] for r in terminal['attempts']} == set(range(4))
            and all(r['status'] == 0 for r in terminal['attempts']), 'run incomplete')
    output = root/'checkpoints/foundir_frozen_expert_router_v1'/RUN
    receipt, receipt_sha = read_json(output/'receipt.json')
    result, result_sha = read_json(output/'result.json')
    require(receipt_sha == terminal['result']['receipt_sha256'] and result_sha == terminal['result_sha256'], 'wrong raw receipt/result')
    require(result == terminal['result'] and result['step'] == 408 and result['frozen_experts'] is True, 'terminal copy mismatch')
    items = [i for i in receipt['items'] if i['split'] == 'validation']
    hashes = {}; input_files = {REPORT: digest}
    for step in (400, 408):
        path = output/'steps'/f'step_{step:06d}.json'
        marker, sha = read_json(path)
        require(marker['step'] == step and marker['receipt_hash'] == receipt_sha, 'wrong checkpoint metadata')
        if step == 408:
            require(marker == terminal['final_checkpoint'], 'final checkpoint marker mismatch')
        hashes[step] = marker['parameter_hash']; input_files[path.relative_to(root).as_posix()] = sha
    records = {}
    for arm in ARMS:
        folder = output/'evaluation'/arm
        files = list(folder.glob('*.json'))
        require({p.stem for p in files} == {i['key'] for i in items}, 'missing/extra saved JSON')
        records[arm] = {}
        for path in files:
            value, sha = read_json(path)
            records[arm][path.stem] = value; input_files[path.relative_to(root).as_posix()] = sha
    prompts = {task: receipt['parent']['experts'][task]['prompt'] for task in ORDER}
    report = analyze(items, records, result, receipt_sha, hashes, prompts)
    require(report['regressions_step408_vs400'] == terminal['regressions_step408_vs400'], 'cross-stage regressions disagree')
    return {**report, 'receipt_sha256': receipt_sha, 'result_sha256': result_sha, 'input_json_sha256': input_files,
            'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def write_report(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n').encode()
    if path.exists():
        require(path.read_bytes() == raw, 'existing diagnosis differs; preserve it, do not overwrite')
        return
    fd, temporary = tempfile.mkstemp(prefix='.pending-router-analysis-', suffix='.json', dir=path.parent)
    with os.fdopen(fd, 'wb') as f:
        f.write(raw); f.flush(); os.fsync(f.fileno())
    try:
        os.link(temporary, path)
    except FileExistsError:
        require(path.read_bytes() == raw, 'concurrent report differs')
    else:
        os.unlink(temporary)
    fd = os.open(path.parent, os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary-only', action='store_true', help='stdout: facts already available from GitHub aggregate report only')
    args = parser.parse_args()
    if args.summary_only:
        terminal, sha = read_json(ROOT/REPORT)
        require(sha == REPORT_SHA, 'wrong report')
        print(json.dumps(aggregate_only(terminal), indent=2, allow_nan=False)); return
    require(str(ROOT) == ACTIVE_ROOT, 'run this JSON-only diagnosis in the destination workspace')
    value = diagnose(ROOT)
    write_report(ROOT/OUTPUT, value)
    print('FROZEN_ROUTER_ERROR_BREAKDOWN_PASS pixels_read=false tensors_read=false GPU_used=false')
    print(json.dumps({'output': OUTPUT, 'final_partition': value['stage408'],
        'transitions': value['transitions'], 'missing_counterfactual_count': value['missing_counterfactual']['count']}, allow_nan=False))


if __name__ == '__main__':
    main()
