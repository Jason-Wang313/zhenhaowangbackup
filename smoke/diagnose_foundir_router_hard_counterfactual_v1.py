#!/usr/bin/env python3
"""Four saved-argmax one-hot renders only; never train or select an expert with GT."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from smoke import analyze_foundir_frozen_router_errors_v1 as analysis

BREAKDOWN = analysis.OUTPUT
BREAKDOWN_SHA = '911c411710d96498dcdf6d8af2e1abe772c52b95bc3b79d948b485aba5bbf6c5'
RUN = 'predicted_hard_four_v1'
OUTPUT = 'checkpoints/foundir_router_hard_counterfactual_v1/' + RUN
REPORT = 'reports/foundir_router_hard_counterfactual_terminal_2026-11-04.json'
SOURCES = [
    'smoke/diagnose_foundir_router_hard_counterfactual_v1.py',
    'smoke/run_foundir_router_hard_counterfactual_v1_gpu.sh',
    'smoke/start_foundir_router_hard_counterfactual_v1_tmux.sh',
    'smoke/analyze_foundir_frozen_router_errors_v1.py',
]
EXPECTED_KEYS = {
    'validation_foundir_train_0144583', 'validation_foundir_train_0189233',
    'validation_foundir_train_0508782', 'validation_foundir_train_0479048',
}
require = analysis.require


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def select(report):
    require(report['status'] == 'saved_json_error_breakdown_complete' and report['run_id'] == analysis.RUN,
            'wrong diagnostic parent')
    rows = report['per_image']
    require(len(rows) == 40 and len({r['key'] for r in rows}) == 40, 'need complete unique parent rows')
    selected = [r for r in rows if not r['step408']['correct']]
    require(len(selected) == 4 and {r['key'] for r in selected} == EXPECTED_KEYS, 'four fixed cases only')
    plans = []
    for row in selected:
        weights = row['step408']['weights']
        require(len(weights) == 5 and all(analysis.number(w) and 0 <= w <= 1 for w in weights)
                and abs(sum(weights)-1) < 2e-6, 'invalid saved probabilities')
        anchor = analysis.ORDER[max(range(5), key=lambda k: weights[k])]
        require(anchor == row['step408']['anchor'] and anchor != row['task'], 'changed argmax')
        plans.append({'key': row['key'], 'predicted_anchor': anchor,
                      'hard_weights_from_model_argmax': [float(t == anchor) for t in analysis.ORDER],
                      'seed': row['seed'], 'crop_xy': row['crop_xy']})
    require(plans == report['missing_counterfactual']['items'] and report['missing_counterfactual']['count'] == 4,
            'counterfactual list changed')
    return plans


def decompose(row, hard):
    require(set(hard) == {'psnr', 'lpips'} and all(analysis.number(v) for v in hard.values()),
            'nonfinite/wrong hard metrics')
    baseline = row['baseline']; soft = row['step408']['metrics']
    selection = analysis.gaps(hard, baseline)
    mixing = analysis.gaps(soft, hard)
    total = analysis.gaps(soft, baseline)
    for key in total:
        analysis.check_close(selection[key] + mixing[key], total[key], 'decomposition/'+key)
    return {'key': row['key'], 'true_task_for_reporting_only': row['task'],
            'predicted_anchor': row['step408']['anchor'], 'baseline': baseline, 'hard': hard, 'soft': soft,
            'choice_and_native_protocol_deficit': selection, 'soft_mixture_incremental_deficit': mixing,
            'total_soft_deficit': total}


def prepare():
    # Standard-library only: no image decode, torch or GPU initialization here.
    report, digest = analysis.read_json(ROOT / BREAKDOWN)
    require(digest == BREAKDOWN_SHA, 'pinned CPU report changed')
    require(analysis.diagnose(ROOT) == report, 'destination inputs differ from published CPU diagnosis')
    plans = select(report)
    parent = ROOT/'checkpoints/foundir_frozen_expert_router_v1'/analysis.RUN
    bound, receipt_sha = analysis.read_json(parent/'receipt.json')
    require(receipt_sha == report['receipt_sha256'], 'parent receipt changed')
    for name, digest in bound['sources'].items():
        require(sha(ROOT/name) == digest, 'bound source changed: '+name)
    for name, stat in bound['asset_stats'].items():
        actual = (ROOT/name).stat()
        require([actual.st_size, actual.st_mtime_ns] == stat, 'asset metadata changed: '+name)
    items = {i['key']: i for i in bound['items'] if i['split'] == 'validation'}
    for plan in plans:
        item = items[plan['key']]
        require(item['seed'] == plan['seed'] and item['row']['crop_xy'] == plan['crop_xy'], 'crop/seed changed')
    value = {'run_id': RUN, 'parent_run': analysis.RUN, 'parent_receipt_sha256': receipt_sha,
             'breakdown_sha256': BREAKDOWN_SHA, 'plans': plans,
             'sources': {name: sha(ROOT/name) for name in SOURCES},
             'local_workers': 2, 'nodes': 1, 'training': False, 'GT_selects_expert': False,
             'new_render_count': 4, 'wall_cap_seconds': 1800, 'automatic_followup': False}
    analysis.write_report(ROOT/OUTPUT/'receipt.json', value)
    print('ROUTER_HARD_PREFLIGHT_PASS renders=4 training=false pixels_read=false', flush=True)


def bound_inputs():
    receipt, _ = analysis.read_json(ROOT/OUTPUT/'receipt.json')
    require(receipt['run_id'] == RUN and receipt['breakdown_sha256'] == BREAKDOWN_SHA, 'wrong new receipt')
    for name, digest in receipt['sources'].items():
        require(sha(ROOT/name) == digest, 'new bound source changed: '+name)
    report, digest = analysis.read_json(ROOT/BREAKDOWN)
    require(digest == BREAKDOWN_SHA and select(report) == receipt['plans'], 'changed plan')
    parent = ROOT/'checkpoints/foundir_frozen_expert_router_v1'/analysis.RUN
    require(sha(parent/'receipt.json') == receipt['parent_receipt_sha256'], 'parent receipt changed')
    return receipt, report, parent


def runtime_check(idle):
    from smoke import train_foundir_frozen_expert_router_mixed_driver_v1 as impl
    from smoke.frozen_router_runtime_policy_v1 import execution_signature
    _, _, parent = bound_inputs()
    bound = impl.contract.load_receipt(parent)
    actual = impl.native.hopper.hardware(idle)
    actual['metrics'] = impl.parent.four_impl.metric_runtime({'metric': bound['parent']['metric']})
    actual['memory_policy'] = {'inherited': bound['parent']['memory_policy'], 'router_explicit_gate_checkpoint': True}
    for node in range(4):
        old = impl.c.read(parent/f'runtime_node{node}.json')
        require(execution_signature(actual) == execution_signature(old), 'numerical runtime differs from parent')
    # The inherited checker describes its old topology. Explicitly record the actual new topology.
    value = {'inherited_comparison_signature': actual, 'actual_nodes': 1, 'actual_gpus': 2,
             'training': False, 'parent_gpu_names': [impl.c.read(parent/f'runtime_node{n}.json')['gpu_name'] for n in range(4)],
             'driver_difference_allowed': True}
    file = ROOT/OUTPUT/'runtime.json'
    if idle:
        analysis.write_report(file, value)
    else:
        require(analysis.read_json(file)[0] == value, 'runtime changed during/resuming diagnostic')
    return impl, bound


def saved_result(plan):
    output = ROOT/OUTPUT
    path = output/'hard'/(plan['key']+'.json')
    if not path.exists():
        return None
    row, _ = analysis.read_json(path)
    expected = {'plan': plan, 'receipt_sha256': sha(output/'receipt.json'), 'runtime_sha256': sha(output/'runtime.json')}
    require(row['identity'] == expected, 'saved hard identity changed')
    file = (output/row['tensor']).resolve()
    file.relative_to(output.resolve())
    require(row['files'] == {row['tensor']: sha(file)}, 'saved hard tensor changed')
    require(row['route']['anchor'] == plan['predicted_anchor']
            and row['route']['weights'] == [plan['hard_weights_from_model_argmax']], 'saved route changed')
    return row


def run():
    import torch
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    from lowlight_restoration.five_expert_flux import restore_five_expert
    rank = int(os.environ['LOCAL_RANK'])
    require(rank in (0, 1) and int(os.environ['WORLD_SIZE']) == 2
            and int(os.environ['LOCAL_WORLD_SIZE']) == 2 and int(os.environ['RANK']) == rank,
            'single node, two inference workers only')
    receipt, report, parent = bound_inputs()
    impl, bound = runtime_check(False)
    torch.cuda.set_device(rank)
    device = torch.device('cuda', rank)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    os.environ.update(KLEIN_4B_BASE_MODEL_PATH=str(impl.c.path(ROOT, bound['parent']['model']['flow_weights'])),
                      AE_MODEL_PATH=str(impl.c.path(ROOT, bound['parent']['model']['autoencoder_weights'])))
    pending = [p for p in receipt['plans'][rank::2] if saved_result(p) is None]
    if not pending:
        print(f'ROUTER_HARD_WORKER_COMPLETE rank={rank} reused=2', flush=True); return
    model, gates, rules, ae, prompts = impl.load_bank(bound, device)
    metric = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True).to(device).eval().requires_grad_(False)
    items = {i['key']: i for i in bound['items'] if i['split'] == 'validation'}
    output = ROOT/OUTPUT
    for plan in pending:
        start = time.monotonic(); item = items[plan['key']]
        weights = torch.tensor([plan['hard_weights_from_model_argmax']], device=device)
        with torch.no_grad():
            lq = impl.pixels(item, 'lq', device)
            raw, u8, route = restore_five_expert(model, gates, rules, weights, ae, lq, prompts, seed=plan['seed'])
            require(route['anchor'] == plan['predicted_anchor'], 'wrong predicted primary')
            # Ground truth is loaded only after the one predetermined output exists.
            gt = impl.pixels(item, 'gt', device)
            image, target = u8.float()/255., (gt+1)*.5
            mse = float((image-target).square().mean())
            metrics = {'psnr': 100. if mse <= 1e-12 else -10*math.log10(mse), 'lpips': float(metric(image, target))}
            metric.reset()
        require(all(analysis.number(v) for v in metrics.values()), 'nonfinite metric')
        tensor, digest = impl.native.replay.save_tensors(output, 'hard_'+plan['key'], {'uint8': u8})
        result = {'identity': {'plan': plan, 'receipt_sha256': sha(output/'receipt.json'),
                              'runtime_sha256': sha(output/'runtime.json')},
                  'tensor': tensor, 'files': {tensor: digest}, 'route': route, 'metrics': metrics,
                  'seconds': time.monotonic()-start, 'rank': rank,
                  'peak_allocated_gib': torch.cuda.max_memory_allocated(device)/1024**3,
                  'GT_used_for_generation_or_selection': False}
        analysis.write_report(output/'hard'/(plan['key']+'.json'), result)
        print('ROUTER_HARD_IMAGE_COMPLETE '+json.dumps({'key': plan['key'], 'metrics': metrics}), flush=True)
        del raw, u8, gt, image, target, lq
    print(f'ROUTER_HARD_WORKER_COMPLETE rank={rank}', flush=True)


def summarize():
    receipt, report, parent = bound_inputs()
    rows = {r['key']: r for r in report['per_image']}
    results = []; artifacts = {}
    require({p.stem for p in (ROOT/OUTPUT/'hard').glob('*.json')} == EXPECTED_KEYS, 'four complete outputs required')
    prompts = {task: analysis.read_json(parent/'receipt.json')[0]['parent']['experts'][task]['prompt'] for task in analysis.ORDER}
    for plan in receipt['plans']:
        record = saved_result(plan)
        require(record is not None and record['GT_used_for_generation_or_selection'] is False, 'missing/invalid output')
        analysis.check_route(record['route'], plan['predicted_anchor'], control=True, prompts=prompts)
        results.append({**decompose(rows[plan['key']], record['metrics']),
                        'seconds': record['seconds'], 'peak_allocated_gib': record['peak_allocated_gib']})
        artifacts[plan['key']] = {'record_sha256': sha(ROOT/OUTPUT/'hard'/(plan['key']+'.json')), 'files': record['files']}
    means = {part: {metric: analysis.mean(r[part][metric] for r in results)
                    for metric in ('psnr_loss_db', 'lpips_increase')}
             for part in ('choice_and_native_protocol_deficit', 'soft_mixture_incremental_deficit', 'total_soft_deficit')}
    runtime = analysis.read_json(ROOT/OUTPUT/'runtime.json')[0]
    value = {'status': 'four_predicted_hard_outputs_verified', 'run_id': RUN, 'count': 4,
             'receipt_sha256': sha(ROOT/OUTPUT/'receipt.json'), 'breakdown_sha256': BREAKDOWN_SHA,
             'per_image': results, 'four_case_mean': means, 'runtime': runtime, 'artifacts': artifacts,
             'training': False, 'GT_selects_expert': False, 'automatic_followup': False,
             'interpretation': ['Positive deficits mean worse for both metrics; signed parts sum to total.',
                'Choice component includes anchor-dependent prompt/alpha/readout, not classification alone.',
                'Four preselected error cases are diagnosis, not a new overall performance estimate.',
                'This is not a VLM contribution ablation or mixed-degradation success.',
                'Driver/Hopper model differences are recorded; no bitwise cross-hardware equivalence claimed.']}
    analysis.write_report(ROOT/REPORT, value)
    print('ROUTER_HARD_RESULT_VERIFIED '+json.dumps(means), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('phase', choices=['prepare', 'hardware', 'run', 'summarize'])
    args = ap.parse_args()
    require(str(ROOT) == analysis.ACTIVE_ROOT, 'destination workspace only')
    if args.phase == 'prepare': prepare()
    elif args.phase == 'hardware': runtime_check(True)
    elif args.phase == 'run': run()
    else: summarize()


if __name__ == '__main__':
    main()
