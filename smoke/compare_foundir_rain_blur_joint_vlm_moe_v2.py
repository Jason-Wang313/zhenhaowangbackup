#!/usr/bin/env python3
"""Read-only paired final comparison; never select a checkpoint or tune a gate."""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


def paired_summary(left, right):
    a, b = {r['record_id']: r for r in left['rows']}, {r['record_id']: r for r in right['rows']}
    if len(a) != len(left['rows']) or len(b) != len(right['rows']) or a.keys() != b.keys():
        raise ValueError('paired record mismatch')
    grouped = defaultdict(list)
    for key in sorted(a):
        x, y = a[key], b[key]
        if x['source_group'] != y['source_group']:
            raise ValueError('source-group mismatch')
        for metric in ('psnr', 'lpips'):
            if not all(math.isfinite(z[side + '_' + metric]) for z in (x, y) for side in ('input', 'output')):
                raise ValueError('nonfinite metrics')
            if abs(x['input_' + metric] - y['input_' + metric]) > 1e-6:
                raise ValueError('input/crop/metric protocol mismatch')
        grouped[x['source_group']].append([y['output_psnr']-x['output_psnr'], y['output_lpips']-x['output_lpips']])
    blocks = [np.asarray(v) for _, v in sorted(grouped.items())]
    differences = np.concatenate(blocks)
    sums = np.stack([v.sum(0) for v in blocks]); counts = np.asarray([len(v) for v in blocks])
    rng = np.random.default_rng(2026102603)
    boot = []
    for _ in range(2000):
        idx = rng.integers(len(blocks), size=len(blocks))
        boot.append(sums[idx].sum(0) / counts[idx].sum())
    intervals = np.quantile(boot, [.025, .975], axis=0)
    # Recompute means from rows instead of trusting a cached scalar summary.
    return {'records': len(a), 'source_groups': len(blocks),
        'image_only': {m: float(np.mean([r['output_' + m] for r in a.values()])) for m in ('psnr', 'lpips')},
        'vlm_assisted': {m: float(np.mean([r['output_' + m] for r in b.values()])) for m in ('psnr', 'lpips')},
        'VLM_minus_image_PSNR': float(differences[:, 0].mean()),
        'VLM_minus_image_LPIPS': float(differences[:, 1].mean()),
        'PSNR_source_bootstrap_95pct': intervals[:, 0].tolist(),
        'LPIPS_source_bootstrap_95pct': intervals[:, 1].tolist(),
        'PSNR_win_fraction': float((differences[:, 0] > 0).mean()),
        'LPIPS_win_fraction': float((differences[:, 1] < 0).mean()),
        'interval_caveat': 'paired_source_bootstrap_for_one_seed_not_training_seed_uncertainty'}


def compare(a, b):
    if (a['arm'], b['arm']) != ('image_only', 'vlm_assisted'):
        raise ValueError('arm order must be image_only, vlm_assisted')
    for q in (a, b):
        if not (q['status'] == 'complete' and q['mode'] == 'formal' and q['steps'] == 10000
                and q['engineering_pass'] and not q['protected_pixels_accessed']):
            raise ValueError('requires complete qualified formal results')
    for key in ('config_sha256', 'source_sha256', 'data', 'sample_presentations'):
        if a[key] != b[key]:
            raise ValueError(f'matched protocol differs: {key}')
    summaries = {}
    for split in ('rain', 'blur_full'):
        left, right = a['history'][-1][split], b['history'][-1][split]
        if left['step'] != 10000 or right['step'] != 10000 or left['route'] != 'learned' or right['route'] != 'learned':
            raise ValueError('not the frozen final deployed route')
        summaries[split] = paired_summary(left, right)
    gates = {'rain': (34.2, .08), 'blur_full': (25.9, .18)}
    absolute = {arm: all(summaries[s][arm]['psnr'] >= gates[s][0] and summaries[s][arm]['lpips'] <= gates[s][1]
                         for s in gates) for arm in ('image_only', 'vlm_assisted')}
    return {'status': 'matched_final_comparison', 'per_task': summaries,
        'absolute_quality_gates': absolute,
        'macro_PSNR_delta': float(np.mean([r['VLM_minus_image_PSNR'] for r in summaries.values()])),
        'macro_LPIPS_delta': float(np.mean([r['VLM_minus_image_LPIPS'] for r in summaries.values()])),
        'VLM_both_metrics_improve_each_task': all(r['VLM_minus_image_PSNR'] > 0 and r['VLM_minus_image_LPIPS'] < 0 for r in summaries.values()),
        'paper_novelty_or_SOTA_established': False,
        'frozen_VLM_diagnosis_precompute_cost_excluded_from_restorer_runtime': True,
        'protected_pixels_accessed': False}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('image_only', type=Path)
    p.add_argument('vlm_assisted', type=Path)
    args = p.parse_args()
    print(json.dumps(compare(json.loads(args.image_only.read_text()), json.loads(args.vlm_assisted.read_text())), indent=2))
