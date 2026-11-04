#!/usr/bin/env python3
"""CPU-only VLM score repair; preserve correct image routes, retain corrections.

No new restoration model. No old holdout feature reads. Calibration is development,
not blind evidence. A fresh metadata-only confirmation set is reserved, NOT run.
"""
import argparse
from collections import Counter
import csv
import math
import os
from pathlib import Path
import platform
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from smoke import repair_foundir_frozen_image_router_v2 as parent
from smoke import frozen_image_router_state_v2 as state

c, require, ORDER = parent.c, parent.require, parent.ORDER
RUN = 'preserve_correct_v3'
OUTPUT = 'checkpoints/foundir_vlm_preserve_router_v3/' + RUN
REPORT = 'reports/foundir_vlm_preserve_router_v3_terminal_2026-11-04.json'
STEPS, CHECKS = 1600, (400, 800, 1200, 1600)
PRESERVE_WEIGHT, MARGIN_CAP = 1.0, 0.2
RESERVE_PER_CLASS = 20
EXPECTED_RESERVE_SHA = '57596103ccf992e6f47dbcdb0bce0c1c0d3a8b7ee02e78837fe43b929b6043a7'
SOURCES = ['smoke/repair_foundir_vlm_preserve_router_v3.py',
           'smoke/run_foundir_vlm_preserve_router_v3_cpu.sh',
           'smoke/start_foundir_vlm_preserve_router_v3_tmux.sh']
RECIPE = {'steps': STEPS, 'checks': list(CHECKS), 'batch': 32, 'seed': parent.SEED,
          'lr': .001, 'weight_decay': .01, 'label_smoothing': .05, 'grad_clip': 1.,
          'preserve_weight': PRESERVE_WEIGHT, 'margin_cap': MARGIN_CAP,
          'loss': 'CE + mean_over_batch(image_correct * relu(min(image_margin,.2)-final_margin))',
          'initialization': 'fresh_zero_output_VLM_same_seed_as_V2',
          'trainable': 'VLM_MLP_only', 'fit': 2560, 'calibration': 400,
          'GPU_training': False, 'CPU_cap_seconds': 3600}


def reserve_confirmation(rows, previous, original, per_class=RESERVE_PER_CLASS):
    """Select train-collection singleton metadata, globally disjoint from ALL old roles.

    No quota relaxation, no fallback to previously examined samples. Fewer than
    requested sources is reported as a blocker for confirmation, not for fitting.
    """
    rows = list(rows)
    fields = ('source_group', 'gt_sha256', 'lq_sha256')
    excluded = {f: {i['row'][f] for i in previous + original} |
                {r[f] for r in rows if r['split'] != 'train'} for f in fields}
    categories = {v: k for k, v in parent.audit.CATEGORIES.items()}
    pools = {t: {} for t in ORDER}
    for row in rows:
        if (row['split'] != 'train' or row['category'] not in categories or
                int(row['composition_size']) != 1 or min(int(row['width']), int(row['height'])) < 512 or
                any(row[f] in excluded[f] for f in fields)):
            continue
        t = categories[row['category']]
        source = row['source_group']
        old = pools[t].get(source)
        if old is None or c.stable(RUN, 'record', row['record_id']) < c.stable(RUN, 'record', old['record_id']):
            pools[t][source] = row
    available = {t: len(pools[t]) for t in ORDER}
    items, used = [], {f: set() for f in fields}
    for t in sorted(ORDER, key=lambda t: (available[t], ORDER.index(t))):
        count = 0
        for source, row in sorted(pools[t].items(), key=lambda kv: c.stable(RUN, 'source', kv[0])):
            if count == per_class:
                break
            if any(row[f] in used[f] for f in fields):
                continue
            r = dict(row)
            require(re.fullmatch(r'[A-Za-z0-9_-]+', r['record_id']) is not None, 'unsafe reserved key')
            for side in ('lq_path', 'gt_path'):
                require(r[side].startswith('data/foundir/MillionIRData/Train/' + r['category'] + '/')
                        and '..' not in Path(r[side]).parts, 'reserved path outside training collection')
            raw = bytes.fromhex(c.stable(RUN, 'crop', r['record_id']))
            r['crop_xy'] = [int.from_bytes(raw[:8], 'big') % (int(r['width']) - 511),
                            int.from_bytes(raw[8:16], 'big') % (int(r['height']) - 511)]
            items.append({'key': 'confirm_' + r['record_id'], 'task': t, 'role': 'future_confirmation',
                          'split': 'train', 'seed': int.from_bytes(raw[16:20], 'big'), 'row': r})
            for f in fields:
                used[f].add(r[f])
            count += 1
    items.sort(key=lambda i: (ORDER.index(i['task']), i['key']))
    counts = dict(Counter(i['task'] for i in items))
    return {'items': items, 'items_sha256': c.digest(items), 'available_before_cross_class_allocation': available,
            'counts': counts, 'ready': all(counts.get(t, 0) == per_class for t in ORDER),
            'requested_per_class': per_class, 'pixels_or_features_read': False,
            'automatic_extraction': False, 'whole_system_unseen_claim': False}


def prepare():
    parent.verify()  # Historical JSON/artifact integrity only, no old holdout tensors.
    b = parent.receipt()
    index = parent.feature_index(b)
    active = [i for i in b['items'] if i['role'] in ('fit', 'calibration')]
    index = {i['key']: index[i['key']] for i in active}
    manifest = c.path(ROOT, b['original']['config']['manifest'])
    require(c.sha(manifest) == b['original']['config']['manifest_sha256'], 'manifest changed')
    with manifest.open(newline='') as f:
        reserved = reserve_confirmation(csv.DictReader(f), b['items'], parent.s.load_bound()['items'])
    require(reserved['ready'] and reserved['items_sha256'] == EXPECTED_RESERVE_SHA,
            'future confirmation differs from verified metadata plan; do not substitute old images')
    value = {'run_id': RUN, 'recipe': RECIPE, 'parent_receipt_sha256': c.sha(ROOT / parent.OUTPUT / 'receipt.json'),
             'parent_result_sha256': c.sha(ROOT / parent.REPORT), 'items': active, 'features': index,
             'sources': {**b['sources'], **{n: c.sha(ROOT / n) for n in SOURCES},
                         'smoke/frozen_image_router_state_v2.py': c.sha(ROOT / 'smoke/frozen_image_router_state_v2.py')},
             'future_confirmation': reserved, 'image_checkpoint': b['image_checkpoint'],
             'old_holdout_features_allowed': False, 'automatic_followup': False}
    c.write_new(ROOT / OUTPUT / 'receipt.json', value)
    print(f'VLM_PRESERVE_V3_PREPARED fit=2560 calibration=400 future_confirmation_ready={reserved["ready"]}', flush=True)


def receipt():
    b = c.read(ROOT / OUTPUT / 'receipt.json')
    require(b['run_id'] == RUN and b['recipe'] == RECIPE, 'wrong run/recipe')
    old = parent.receipt()
    require(c.sha(ROOT / parent.OUTPUT / 'receipt.json') == b['parent_receipt_sha256'] and
            c.sha(ROOT / parent.REPORT) == b['parent_result_sha256'], 'parent changed')
    require(b['items'] == [i for i in old['items'] if i['role'] in ('fit', 'calibration')], 'training population changed')
    require(set(b['features']) == {i['key'] for i in b['items']}, 'features must be fit/calibration only')
    require(b['image_checkpoint'] == old['image_checkpoint'], 'frozen image changed')
    require(b['old_holdout_features_allowed'] is False and b['automatic_followup'] is False, 'scope changed')
    for n, sha in b['sources'].items():
        require(c.sha(ROOT / n) == sha, 'bound source changed: ' + n)
    reserve = b['future_confirmation']
    require(c.digest(reserve['items']) == reserve['items_sha256'] == EXPECTED_RESERVE_SHA and reserve['ready'],
            'confirmation reservation changed')
    for field in ('source_group', 'gt_sha256', 'lq_sha256'):
        ids = [i['row'][field] for i in reserve['items']]
        require(len(ids) == len(set(ids)) and not set(ids) & {i['row'][field] for i in old['items']},
                'confirmation overlap')
    return b


def load_role(b, role):
    import torch
    from safetensors import safe_open
    require(role in ('fit', 'calibration'), 'old holdout and reserved confirmation must not be loaded')
    items = [i for i in b['items'] if i['role'] == role]
    require(Counter(i['task'] for i in items) == Counter({t: 512 if role == 'fit' else 80 for t in ORDER}),
            'role quotas changed')
    xs, vs = [], []
    for i in items:
        spec = b['features'][i['key']]
        file = c.path(ROOT, spec['path'])
        require(c.sha(file) == spec['sha256'], 'feature hash changed')
        with safe_open(str(file), framework='pt', device='cpu') as f:
            x, v = f.get_tensor('image'), f.get_tensor('vlm')
        require(x.shape == (1, 6144) and v.shape == (1, 3584) and torch.isfinite(x).all() and torch.isfinite(v).all(),
                'invalid frozen features')
        xs.append(x.float())
        vs.append(v.float())
    return items, torch.cat(xs), torch.cat(vs), torch.tensor([ORDER.index(i['task']) for i in items])


def objective(logits, image_logits, labels):
    import torch
    import torch.nn.functional as F
    # Detach the reference even if a caller accidentally leaves its grad enabled.
    image_logits = image_logits.detach()
    mask = F.one_hot(labels, logits.shape[-1]).bool()
    final_margin = logits.gather(1, labels[:, None]).squeeze(1) - logits.masked_fill(mask, -torch.inf).max(1).values
    image_margin = image_logits.gather(1, labels[:, None]).squeeze(1) - image_logits.masked_fill(mask, -torch.inf).max(1).values
    correct = image_logits.argmax(-1) == labels
    # Cap the inherited margin: do not copy confidence or enforce a wrong image route.
    target_margin = image_margin.clamp(min=0., max=MARGIN_CAP)
    penalty = (correct * F.relu(target_margin - final_margin)).mean()
    ce = F.cross_entropy(logits, labels, label_smoothing=.05)
    return ce + PRESERVE_WEIGHT * penalty, ce, penalty, correct


def update(m, optimizer, x, v, labels):
    import torch
    optimizer.zero_grad(set_to_none=True)
    logits, _, im, _ = m(x, v)
    loss, ce, penalty, protected = objective(logits, im, labels)
    require(bool(torch.isfinite(loss)), 'nonfinite loss')
    loss.backward()
    norm = float(torch.nn.utils.clip_grad_norm_(m.vlm.parameters(), 1., error_if_nonfinite=True))
    optimizer.step()
    require(all(p.grad is None for p in m.image.parameters()), 'frozen image received gradients')
    require(all(torch.isfinite(p).all() for p in m.parameters()), 'nonfinite parameters')
    return {'loss': float(loss.detach()), 'ce': float(ce.detach()), 'preserve_penalty': float(penalty.detach()),
            'protected_training_rows': int(protected.sum()),
            'harmful_training_overrides': int((protected & (logits.argmax(-1) != labels)).sum()),
            'vlm_grad_norm_before_clip': norm}


def transitions(base, corrected):
    a = {r['key']: r for r in base['rows']}
    z = {r['key']: r for r in corrected['rows']}
    require(len(a) == len(base['rows']) and len(z) == len(corrected['rows']) and a.keys() == z.keys(), 'paired population mismatch')
    count = Counter()
    per = {t: Counter() for t in ORDER}
    for key, r in a.items():
        q = z[key]
        require((r['task_for_reporting_only'], r['seed'], r['crop_xy'], r['image_logits']) ==
                (q['task_for_reporting_only'], q['seed'], q['crop_xy'], q['image_logits']), 'paired image reference changed')
        truth = r['task_for_reporting_only']
        before, after = r['predicted_anchor'] == truth, q['predicted_anchor'] == truth
        kind = 'retained_correct' if before and after else 'harm' if before else 'fix' if after else 'still_wrong'
        count[kind] += 1
        per[truth][kind] += 1
    kinds = ('fix', 'harm', 'retained_correct', 'still_wrong')
    return {**{k: count[k] for k in kinds}, 'per_task': {t: {k: per[t][k] for k in kinds} for t in ORDER}}


def eligible(base, old, new):
    previous, current = transitions(base, old), transitions(base, new)
    return (new['correct'] > base['correct'] and new['ce'] <= base['ce'] and
            all(new['per_task'][t]['correct'] >= base['per_task'][t]['correct'] for t in ORDER) and
            current['harm'] < previous['harm'] and current['fix'] >= previous['fix'])


def choose(base, old, milestones):
    allowed = [step for step in CHECKS if eligible(base, old, milestones[step])]
    step = min(allowed or list(CHECKS), key=lambda k: (milestones[k]['ce'], k))
    return {'step': step, 'calibration_screen_pass': bool(allowed),
            'eligible_steps': allowed, 'criterion': 'minimum_CE_among_safe_correction_candidates_else_diagnostic_minimum_CE',
            'deploy_or_restoration_authorized': False}


def train():
    import torch
    import shutil
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized(), 'CPU-only training')
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    b = receipt()
    out = ROOT / OUTPUT
    runtime = {'torch': str(torch.__version__), 'build': c.digest(torch.__config__.show()),
               'python': platform.python_version(), 'threads': 8, 'interop': 1, 'deterministic': True,
               'cpu': next(l for l in Path('/proc/cpuinfo').read_text().splitlines() if l.startswith('model name'))}
    c.write_new(out / 'cpu_runtime.json', runtime)
    if (ROOT / REPORT).exists():
        verify()
        print('VLM_PRESERVE_V3_ALREADY_COMPLETE no_training_or_evaluation_rerun=true', flush=True)
        return
    if not (out / 'steps/step_000000.json').exists():
        require(shutil.disk_usage(out).free >= 30 * 1024**3, 'need 30GiB for durable per-step state')
    fit, cal = load_role(b, 'fit'), load_role(b, 'calibration')
    m, optimizer = parent.model(b)
    image_hash = state.io.parameter_hash(dict(m.image.named_parameters()))
    identity = c.digest({'receipt': c.sha(out / 'receipt.json'), 'runtime': runtime, 'image': image_hash})
    saved = state.load(out, identity)
    start, history = 0, []
    if saved:
        state.restore(m, optimizer, saved)
        start, history = saved['step'], saved['history']
    else:
        state.save(out, m, optimizer, 0, history, identity)
    base = parent.evaluate(m, cal, False)
    historical = c.read(ROOT / parent.REPORT)['calibration']
    # Same actual frozen image predictions, not an internal branch of a joint model.
    require(base == historical['image_only'], 'parent calibration image predictions differ; do not silently migrate runtime')
    c.write_new(out / 'calibration_image_only.json', base)
    if start in CHECKS and not (out / f'calibration_{start}.json').exists():
        c.write_new(out / f'calibration_{start}.json', parent.evaluate(m, cal, True))
    for step in range(start, STEPS):
        ids = torch.tensor(parent.batch_indices(fit[0], step))
        stats = update(m, optimizer, fit[1][ids], fit[2][ids], fit[3][ids])
        history.append(stats)
        state.save(out, m, optimizer, step + 1, history, identity)
        if (step + 1) % 100 == 0:
            print(f'VLM_PRESERVE_V3_STEP step={step+1} loss={stats["loss"]} preserve={stats["preserve_penalty"]}', flush=True)
        if step + 1 in CHECKS:
            c.write_new(out / f'calibration_{step+1}.json', parent.evaluate(m, cal, True))
    require(state.io.parameter_hash(dict(m.image.named_parameters())) == image_hash, 'image branch changed')
    milestones = {step: c.read(out / f'calibration_{step}.json') for step in CHECKS}
    selection = {**choose(base, historical['corrected'], milestones),
                 'calibration_sha256': {str(step): c.sha(out / f'calibration_{step}.json') for step in CHECKS},
                 'receipt_sha256': c.sha(out / 'receipt.json')}
    c.write_new(out / 'selection.json', selection)
    chosen = selection['step']
    state.load_step(out, chosen, identity)
    value = {'status': 'vlm_preserve_router_training_complete', 'run_id': RUN,
             'receipt_sha256': c.sha(out / 'receipt.json'), 'runtime': runtime, 'frozen_image_hash': image_hash,
             'selected_checkpoint': c.read(out / f'steps/step_{chosen:06d}.json'),
             'selection': selection, 'selection_sha256': c.sha(out / 'selection.json'),
             'calibration': {'image_only': base, 'previous_v2': historical['corrected'], 'corrected': milestones[chosen]},
             'transitions': {'previous_v2': transitions(base, historical['corrected']),
                             'corrected': transitions(base, milestones[chosen])},
             'milestones': {str(k): {a: v for a, v in x.items() if a != 'rows'} for k, x in milestones.items()},
             'future_confirmation': {k: v for k, v in b['future_confirmation'].items() if k != 'items'},
             'old_holdout_features_accessed': False, 'fresh_confirmation_evaluated': False,
             'restoration_metrics_measured': False, 'automatic_followup': False, 'GPU_training': False,
             'limitations': ['Calibration was used for selection; not an independent success result.',
                            'Correction sees2560; frozen image was trained on640. Not a matched VLM ablation.',
                            'Training protection is not a guarantee of no harm on new inputs.',
                            'Singleton top1 only; no mixed-degradation/PSNR/LPIPS/official gate claim.']}
    c.write_new(ROOT / REPORT, value)
    print(f'VLM_PRESERVE_V3_TRAIN_COMPLETE selected={chosen} calibration_screen={selection["calibration_screen_pass"]}', flush=True)


def verify_summary(summary, items):
    rows = summary['rows']
    require(summary['count'] == len(rows) == 400 and len({r['key'] for r in rows}) == 400 and
            {r['key'] for r in rows} == set(items), 'evaluation population changed')
    counts, losses = Counter(), []
    for row in rows:
        item = items[row['key']]
        z, im, delta = row['logits'], row['image_logits'], row['vlm_delta']
        require(all(len(v) == 5 and all(math.isfinite(x) for x in v) for v in (z, im, delta)), 'invalid logits')
        require(all(abs(a - b - d) < 1e-4 for a, b, d in zip(z, im, delta)), 'logit components differ')
        pred = ORDER[max(range(5), key=lambda j: z[j])]
        require((row['task_for_reporting_only'], row['seed'], row['crop_xy'], row['predicted_anchor']) ==
                (item['task'], item['seed'], item['row']['crop_xy'], pred), 'prediction or input mismatch')
        counts[item['task']] += int(pred == item['task'])
        mx = max(z)
        losses.append(mx + math.log(sum(math.exp(v - mx) for v in z)) - z[ORDER.index(item['task'])])
    require(summary['correct'] == sum(counts.values()) and abs(summary['ce'] - sum(losses) / 400) < 2e-5,
            'aggregate mismatch')
    require(summary['per_task'] == {t: {'count': 80, 'correct': counts[t]} for t in ORDER}, 'per-class mismatch')


def verify():
    b, out = receipt(), ROOT / OUTPUT
    r = c.read(ROOT / REPORT)
    require(r['status'] == 'vlm_preserve_router_training_complete' and r['run_id'] == RUN and
            r['receipt_sha256'] == c.sha(out / 'receipt.json') and r['runtime'] == c.read(out / 'cpu_runtime.json'), 'result identity mismatch')
    require(all(r[k] is False for k in ('old_holdout_features_accessed', 'fresh_confirmation_evaluated',
                                       'restoration_metrics_measured', 'automatic_followup', 'GPU_training')), 'scope claim mismatch')
    items = {i['key']: i for i in b['items'] if i['role'] == 'calibration'}
    old = c.read(ROOT / parent.REPORT)['calibration']
    base, previous = r['calibration']['image_only'], r['calibration']['previous_v2']
    require(base == old['image_only'] and previous == old['corrected'], 'historical comparison changed')
    summaries = {k: c.read(out / f'calibration_{k}.json') for k in CHECKS}
    for summary in [base, previous, *summaries.values()]:
        verify_summary(summary, items)
    selected = c.read(out / 'selection.json')
    require(selected == {**choose(base, previous, summaries),
                         'calibration_sha256': {str(k): c.sha(out / f'calibration_{k}.json') for k in CHECKS},
                         'receipt_sha256': c.sha(out / 'receipt.json')}, 'selection rule changed')
    require(selected == r['selection'] and c.sha(out / 'selection.json') == r['selection_sha256'], 'selection receipt mismatch')
    require(r['calibration']['corrected'] == summaries[selected['step']] and
            r['transitions'] == {'previous_v2': transitions(base, previous),
                                 'corrected': transitions(base, summaries[selected['step']])}, 'correction accounting mismatch')
    require(r['milestones'] == {str(k): {a: v for a, v in x.items() if a != 'rows'} for k, x in summaries.items()}, 'milestone report mismatch')
    require(r['future_confirmation'] == {k: v for k, v in b['future_confirmation'].items() if k != 'items'}, 'reservation report mismatch')
    files = sorted((out / 'steps').glob('step_*.json'))
    require([p.name for p in files] == [f'step_{k:06d}.json' for k in range(STEPS + 1)], 'missing saved steps')
    identity = c.digest({'receipt': c.sha(out / 'receipt.json'), 'runtime': r['runtime'], 'image': r['frozen_image_hash']})
    for k in {0, selected['step'], STEPS}:
        marker = c.read(out / f'steps/step_{k:06d}.json')
        require(marker['step'] == k and marker['identity'] == identity and c.sha(c.path(out, marker['file'])) == marker['sha256'],
                'checkpoint integrity mismatch')
    require(r['selected_checkpoint'] == c.read(out / f'steps/step_{selected["step"]:06d}.json'), 'selected checkpoint mismatch')
    print('VLM_PRESERVE_V3_RESULT_VERIFIED calibration_only=true no_new_holdout=true', flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('phase', choices=['prepare', 'train', 'verify'])
    args = ap.parse_args()
    require(str(ROOT) == parent.s.analysis.ACTIVE_ROOT, 'destination workspace only')
    {'prepare': prepare, 'train': train, 'verify': verify}[args.phase]()


if __name__ == '__main__':
    main()
