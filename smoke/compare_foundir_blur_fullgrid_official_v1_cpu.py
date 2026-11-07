#!/usr/bin/env python3
"""Compare complete fixed official outputs; never select checkpoints or images."""
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from smoke.existing_flux_replay_contract_v1 import read, sha, write_new

INPUTS = {
    'step200': 'reports/foundir_blur_coverage_official_v2_result_2026-11-06.json',
    'step1280': 'reports/foundir_blur_exposure_official_2x2_v2_result_2026-11-07.json',
    'step1792': 'reports/foundir_blur_multiview_official_v1_result_2026-11-07.json',
    'step9472': 'reports/foundir_blur_fullgrid_official_v1_result_2026-11-07.json',
}
SAME_PROTOCOL = ('population', 'inference', 'model_calls_per_tile', 'tile', 'overlap',
                 'stitch', 'padding', 'metrics', 'route', 'gate')
METRICS = ('psnr', 'lpips', 'ssim')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def summarize(results):
    require(set(results) == set(INPUTS), 'four fixed complete results required')
    maps = {}
    for name, result in results.items():
        require(result['status'] == 'complete_official_Blur150' and result['exit_code'] == 0
                and result['GT_used_in_training_or_selection'] is False, 'unfinished or GT-selected result')
        require(result['recipe']['selected_step'] == int(name.removeprefix('step')), 'wrong fixed checkpoint')
        records = result['records']
        by_id = {r['item']['row']['record_id']: r for r in records}
        require(len(records) == len(by_id) == 150, 'all150 unique records required')
        for record in records:
            require(record['GT_input_or_selection'] is False, 'per-image output selected using GT')
            require(set(record['metrics']) == {'lq', 'old50step', 'candidate'}, 'metric arms changed')
            for values in record['metrics'].values():
                require(set(values) == set(METRICS) and all(math.isfinite(v) for v in values.values()), 'invalid metrics')
        for arm in ('lq', 'old50step', 'candidate'):
            for metric in METRICS:
                mean = math.fsum(r['metrics'][arm][metric] for r in records)/150
                require(abs(mean-result['means'][arm][metric]) < 1e-12, 'reported mean differs from150 records')
        maps[name] = by_id
    parent = results['step1280']
    for name, result in results.items():
        require(all(result['recipe'][k] == parent['recipe'][k] for k in SAME_PROTOCOL), 'evaluation protocol changed')
        require(set(maps[name]) == set(maps['step1280']), 'official population changed')
        for rid, record in maps[name].items():
            old = maps['step1280'][rid]
            require(record['item'] == old['item'], 'input images/route/baseline identity changed')
            for arm in ('lq', 'old50step'):
                for metric in METRICS:
                    require(abs(record['metrics'][arm][metric]-old['metrics'][arm][metric]) <= 1e-6,
                            'matched baseline metrics changed')
    records = []
    for rid in sorted(maps['step1280']):
        original = maps['step1280'][rid]
        metrics = {'lq': original['metrics']['lq'], 'original_FLUX': original['metrics']['old50step']}
        metrics.update({name: by_id[rid]['metrics']['candidate'] for name, by_id in maps.items()})
        records.append({'record_id': rid, 'metrics': metrics})
    means = {arm: {k: math.fsum(r['metrics'][arm][k] for r in records)/150 for k in METRICS}
             for arm in records[0]['metrics']}
    current = means['step9472']
    comparisons = {}
    for arm in ('lq', 'original_FLUX', 'step200', 'step1280', 'step1792'):
        def better(r):
            a, b = r['metrics']['step9472'], r['metrics'][arm]
            return a['psnr'] > b['psnr'] and a['lpips'] < b['lpips']
        def worse(r):
            a, b = r['metrics']['step9472'], r['metrics'][arm]
            return a['psnr'] < b['psnr'] and a['lpips'] > b['lpips']
        comparisons[arm] = {
            'PSNR_delta_dB': current['psnr']-means[arm]['psnr'],
            'LPIPS_relative_reduction_percent': 100*(1-current['lpips']/means[arm]['lpips']),
            'both_means_improve': current['psnr'] > means[arm]['psnr'] and current['lpips'] < means[arm]['lpips'],
            'per_image_both_better': sum(better(r) for r in records),
            'per_image_both_worse': sum(worse(r) for r in records),
        }
    gate = {'psnr': current['psnr'] >= 25.9, 'lpips': current['lpips'] <= .18}
    require(results['step9472']['gate_checks'] == gate
            and results['step9472']['official_Blur_pass'] == all(gate.values()), 'gate report mismatch')
    return {'status': 'complete_fixed_official150_cross_round_comparison', 'record_count': 150,
            'means': means, 'latest_vs': comparisons, 'gate_checks': gate,
            'official_Blur_pass': all(gate.values()), 'PSNR_shortfall_dB': max(0., 25.9-current['psnr']),
            'LPIPS_excess': max(0., current['lpips']-.18), 'records': records,
            'mean_recomputed_from_all150': True, 'matched_baseline_absolute_tolerance': 1e-6,
            'checkpoint_or_per_image_selection': False, 'independent_blind_test': False,
            'official_used_for_iterative_development': True}


def main():
    start = time.monotonic()
    result = summarize({name: read(ROOT/path) for name, path in INPUTS.items()})
    result.update(inputs={name: {'path': path, 'sha256': sha(ROOT/path)} for name, path in INPUTS.items()},
                  seconds=time.monotonic()-start, exit_code=0, GPU_hours=0)
    stem = ROOT/'reports/foundir_blur_fullgrid_v1_official_comparison_2026-11-07'
    write_new(stem.with_suffix('.json'), result)
    rows = ['| 固定方案 | PSNR ↑ | LPIPS ↓ |', '|---|---:|---:|']
    for arm, metrics in result['means'].items():
        rows.append(f"| {arm} | {metrics['psnr']:.6f} | {metrics['lpips']:.6f} |")
    text = '# Blur完整官方150：固定终点跨轮比较\n\n'+'\n'.join(rows)+'\n\n'
    text += '最新9472双门槛25.9/.18通过='+str(result['official_Blur_pass'])+'。\n\n'
    text += '全部150逐图均值独立重算，输入/路由/原基线一致；不拼接不同方案最好指标，不按GT选图或选权重。官方已参与多轮开发，不是独立盲测。\n'
    if stem.with_suffix('.md').exists():
        require(stem.with_suffix('.md').read_text() == text, 'existing comparison text differs')
    else:
        with stem.with_suffix('.md').open('x') as handle: handle.write(text)
    print('BLUR_FULLGRID_OFFICIAL_COMPARISON_VERIFIED '+json.dumps({k:v for k,v in result.items() if k != 'records'}), flush=True)


if __name__ == '__main__':
    main()
