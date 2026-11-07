#!/usr/bin/env python3
"""Describe fixed official outputs after multiview training; no fitting or selection."""
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from smoke import analyze_foundir_blur_coverage_direction_v2_cpu as base
from smoke.existing_flux_replay_contract_v1 import read, sha, write_new as write

RESULT = ROOT / 'reports/foundir_blur_multiview_official_v1_result_2026-11-07.json'
PRIOR = ROOT / 'reports/foundir_blur_exposure_official_2x2_v2_result_2026-11-07.json'
DIAG = ROOT / 'reports/foundir_blur_exposure_direction_v1_CPU_2026-11-07.json'
REPORT = ROOT / 'reports/foundir_blur_multiview_direction_v1_CPU_2026-11-07.json'


def main():
    start = time.monotonic()
    current, prior, diagnostic = read(RESULT), read(PRIOR), read(DIAG)
    assert current['status'] == prior['status'] == 'complete_official_Blur150'
    assert diagnostic['result_sha256'] == sha(PRIOR), 'previous diagnostic binding changed'
    a = {r['item']['row']['record_id']: r for r in prior['records']}
    b = {r['item']['row']['record_id']: r for r in current['records']}
    assert len(a) == len(b) == len(current['records']) == len(prior['records']) == 150
    assert a.keys() == b.keys()
    for rid in a:
        assert a[rid]['item'] == b[rid]['item'], 'row/route/baseline changed'
        assert a[rid]['metrics']['lq'] == b[rid]['metrics']['lq'], 'LQ metric path changed'
        assert b[rid]['GT_input_or_selection'] is False
    # Reuse the established pixel/SHA/PSNR/energy-identity checks unchanged.
    # This changes only this process's output-root variable, not historical files.
    base.OUT = ROOT / 'checkpoints/foundir_blur_multiview_official_v1'
    with ThreadPoolExecutor(max_workers=2) as pool:
        records = list(pool.map(base.describe, current['records']))
    keys = ('MSE', 'LQ_MSE', 'residual_MSE_norm', 'residual_target_dot_mean',
            'error_DC_MSE', 'error_DC_fraction')
    summary = {k: math.fsum(r[k] for r in records) / 150 for k in keys}
    cosines = [r['residual_target_cosine'] for r in records if r['residual_target_cosine'] is not None]
    summary.update(cosine_mean=math.fsum(cosines) / len(cosines),
                   MSE_worse_than_LQ_count=sum(r['MSE_change_from_LQ'] > 0 for r in records),
                   nonpositive_direction_count=sum(r['nonpositive_direction'] for r in records),
                   aligned_but_excess_norm_count=sum(r['aligned_but_excess_norm'] for r in records))
    paired = []
    for rid in sorted(a):
        old, new = a[rid]['metrics']['candidate'], b[rid]['metrics']['candidate']
        paired.append({'record_id': rid, 'step1280': old, 'step1792': new,
                       'PSNR_delta': new['psnr'] - old['psnr'],
                       'LPIPS_delta': new['lpips'] - old['lpips']})
    value = {'status': 'fixed_official150_post_generation_description_complete',
             'summary': {'step1792': summary, 'step1280': diagnostic['summary']['step1280'],
                         'step200': diagnostic['summary']['step200'],
                         'old50step': diagnostic['summary']['old50step']},
             'records': records, 'paired_metrics': paired,
             'joint_better_than_step1280': sum(r['PSNR_delta'] > 0 and r['LPIPS_delta'] < 0 for r in paired),
             'joint_worse_than_step1280': sum(r['PSNR_delta'] < 0 and r['LPIPS_delta'] > 0 for r in paired),
             'same_official_rows_routes_baselines': True,
             'result_sha256': sha(RESULT), 'prior_result_sha256': sha(PRIOR),
             'prior_diagnostic_sha256': sha(DIAG),
             'sources': {str(Path(p).relative_to(ROOT)): sha(Path(p))
                         for p in (__file__, base.__file__)},
             'GT_use': 'post_generation_descriptive_statistics_only',
             'parameters_fitted': False, 'new_predictions_generated': False,
             'GT_output_selection': False, 'new_quality_gate': False,
             'note': 'Direction/excess-energy labels are descriptive, not a deployment selector or fitted scale.',
             'seconds': time.monotonic() - start, 'GPU_hours': 0, 'exit_code': 0}
    write(REPORT, value)
    print('BLUR_MULTIVIEW_DIRECTION_END ' + json.dumps({k: v for k, v in value.items()
          if k not in ('records', 'paired_metrics')}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
