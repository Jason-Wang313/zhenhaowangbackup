#!/usr/bin/env python3
"""Read-only, per-task comparison of completed matched Rain/Blur arms."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def compare(moe, dense):
    for result, arm in ((moe, 'moe'), (dense, 'dense')):
        if not (result['status'] == 'complete' and result['mode'] == 'formal'
                and result['arm'] == arm and result['steps'] == 3000
                and result['engineering_pass'] and result['initial_rain_parity_pass']
                and not result['protected_pixels_accessed']):
            raise RuntimeError('requires two completed formal arms')
    if (moe['config_sha256'] != dense['config_sha256'] or moe['data'] != dense['data']
            or moe['source_sha256'] != dense['source_sha256']):
        raise RuntimeError('unmatched protocol/code/data')
    summary = {}
    for split in ('rain', 'blur_full'):
        a, b = moe['history'][-1][split], dense['history'][-1][split]
        aa = {r['record_id']: r for r in a['rows']}
        bb = {r['record_id']: r for r in b['rows']}
        if aa.keys() != bb.keys():
            raise RuntimeError('unpaired records')
        if any(aa[k]['source_group'] != bb[k]['source_group'] or
               any(abs(aa[k]['input_' + m] - bb[k]['input_' + m]) > 1e-6 for m in ('psnr', 'lpips')) for k in aa):
            raise RuntimeError('unmatched crops/inputs')
        for entry in (a, b):
            for metric in ('psnr', 'lpips'):
                mean = sum(r['output_' + metric] for r in entry['rows']) / len(entry['rows'])
                if not math.isclose(mean, entry[metric], abs_tol=1e-12):
                    raise RuntimeError('aggregate metric mismatch')
        summary[split] = {'records': len(aa), 'moe_psnr': a['psnr'], 'dense_psnr': b['psnr'],
            'moe_lpips': a['lpips'], 'dense_lpips': b['lpips'],
            'delta_psnr': a['psnr'] - b['psnr'], 'delta_lpips': a['lpips'] - b['lpips'],
            'psnr_win_records': sum(aa[k]['output_psnr'] > bb[k]['output_psnr'] for k in aa),
            'lpips_win_records': sum(aa[k]['output_lpips'] < bb[k]['output_lpips'] for k in aa),
            'mean_expert_weights': a['mean_expert_weights']}
    return {'status': 'matched_formal_comparison', 'tasks': summary,
            'moe_both_task_gates_pass': moe['both_task_gates_pass'],
            'moe_improves_both_metrics_on_both_tasks': all(v['delta_psnr'] > 0 and v['delta_lpips'] < 0 for v in summary.values()),
            'trainable_parameters': {'moe': moe['trainable_parameters'], 'dense': dense['trainable_parameters']},
            'allocated_gpu_hours': {'moe': moe['allocated_gpu_hours'], 'dense': dense['allocated_gpu_hours']},
            'capacity_matched': False, 'flops_matched': False,
            'caveat': 'Same data/steps/objective; MoE has more parameters and compute. This is not a routing-only causal ablation.',
            'constant_expert_ablations_require_inspection': True, 'automatic_next_launch': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--moe', required=True, type=Path)
    parser.add_argument('--dense', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(compare(json.loads(args.moe.read_text()), json.loads(args.dense.read_text())), indent=2))


if __name__ == '__main__':
    main()
