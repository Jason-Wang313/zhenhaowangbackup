#!/usr/bin/env python3
"""Explain fixed Blur residuals on train/development caches; never deploy GT fits."""
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from smoke import run_foundir_fullstack_frequency_repair_v1 as experiment
from lowlight_restoration import flux_frequency_repair_v1 as spectral
import numpy as np
from scipy.fft import dctn

REPORT = ROOT / 'reports/foundir_blur_frequency_evidence_v1_CPU_2026-11-06.json'


def psnr(prediction, target):
    mse = float(np.mean((spectral.image(prediction) - spectral.image(target)) ** 2))
    return -10 * math.log10(max(mse, 1e-12))


def main():
    if REPORT.exists():
        raise ValueError('preserve existing diagnostic report')
    start = time.monotonic()
    bound = experiment.bound('blur')
    experiment.verify_cache('blur', bound)
    receipt_sha = experiment.sha(experiment.out('blur') / 'receipt.json')
    coefficients_path = experiment.out('blur') / 'coefficients.json'
    frozen = experiment.read(coefficients_path)
    experiment.need(frozen['receipt_sha256'] == receipt_sha, 'coefficient receipt mismatch')
    rows = []
    bands = spectral.band_map(512, 512)
    for item in bound['rows']:
        _, images = experiment.cached('blur', item, receipt_sha)
        lq, gt = experiment.crop(item, 'lq'), experiment.crop(item, 'gt')
        raw, baseline = images['raw'], images['baseline']
        stats = spectral.statistics(lq, raw, gt)
        desired = dctn(spectral.image(gt) - spectral.image(lq), axes=(0, 1), norm='ortho')
        desired_energy = np.array([np.square(desired[bands == k]).sum() / desired.size for k in range(6)])
        energy, cross = np.array(stats['energy']), np.array(stats['cross'])
        # Per-image GT fit is ONLY an explanatory control, never a deployable policy.
        oracle_weights = np.clip(np.divide(cross, energy, out=np.zeros_like(cross), where=energy > 0), 0, 1)
        oracle = spectral.apply(lq, raw, oracle_weights)
        candidate = spectral.apply(lq, raw, frozen['weights'])
        alignment = np.divide(cross, np.sqrt(energy * desired_energy),
                              out=np.zeros_like(cross), where=energy * desired_energy > 0)
        exact_projection_error = desired_energy - 2 * oracle_weights * cross + oracle_weights ** 2 * energy
        experiment.need(np.all(exact_projection_error > -1e-12), 'quadratic energy inconsistent')
        rows.append({'record_id': item['row']['record_id'], 'role': item['role'],
                     'cache_marker_sha256': experiment.sha(experiment.out('blur') / 'cache' / (experiment.key(item) + '.json')),
                     'PSNR': {name: psnr(image, gt) for name, image in
                              [('lq', lq), ('raw', raw), ('baseline', baseline),
                               ('frozen_candidate', candidate), ('GT_per_image_diagnostic', oracle)]},
                     'residual_energy': energy.tolist(), 'desired_energy': desired_energy.tolist(),
                     'cross': cross.tolist(), 'alignment_cosine': alignment.tolist(),
                     'GT_per_image_diagnostic_weights': oracle_weights.tolist(),
                     'unclipped_quadratic_residual_error': exact_projection_error.tolist()})
    summary = {}
    for role in ('train', 'development'):
        selected = [row for row in rows if row['role'] == role]
        by_input_psnr = {}
        for label, low, high in [('below20', -math.inf, 20), ('20to25', 20, 25),
                                 ('25to30', 25, 30), ('30plus', 30, math.inf)]:
            subset = [row for row in selected if low <= row['PSNR']['lq'] < high]
            by_input_psnr[label] = {'count': len(subset), 'mean_PSNR':
                {arm: math.fsum(row['PSNR'][arm] for row in subset) / len(subset)
                 for arm in selected[0]['PSNR']} if subset else None}
        summary[role] = {
            'count': len(selected),
            'mean_PSNR': {arm: math.fsum(row['PSNR'][arm] for row in selected) / len(selected)
                          for arm in selected[0]['PSNR']},
            'mean_alignment_cosine_per_band': np.mean([row['alignment_cosine'] for row in selected], axis=0).tolist(),
            'negative_alignment_count_per_band': np.sum(np.array([row['cross'] for row in selected]) < 0, axis=0).tolist(),
            'mean_desired_error_energy_per_band': np.mean([row['desired_energy'] for row in selected], axis=0).tolist(),
            'baseline_PSNR_below_LQ_count': sum(row['PSNR']['baseline'] < row['PSNR']['lq'] for row in selected),
            'descriptive_input_PSNR_strata_not_a_deployment_selector': by_input_psnr,
        }
    result = {'status': 'CPU_explanatory_diagnostic_complete', 'receipt_sha256': receipt_sha,
              'coefficients_sha256': experiment.sha(coefficients_path),
              'source_sha256': experiment.sha(Path(__file__)), 'bands': list(spectral.BANDS),
              'summary': summary, 'rows': rows, 'official_pixels_read': False,
              'GT_diagnostic_is_not_deployable': True,
              'GT_diagnostic_changes_frozen_candidate': False,
              'GT_diagnostic_is_not_rigorous_bound_after_clip_and_quantization': True,
              'no_LPIPS_inference_from_PSNR_diagnostic': True,
              'GPU_hours': 0, 'seconds': time.monotonic() - start, 'exit_code': 0}
    experiment.write(REPORT, result)
    print('BLUR_FREQUENCY_EVIDENCE_COMPLETE ' + json.dumps({'summary': summary, 'seconds': result['seconds']}), flush=True)


if __name__ == '__main__':
    main()
