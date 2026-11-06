#!/usr/bin/env python3
"""CPU-only summary of existing error records; never fit or generate images."""
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / 'checkpoints/foundir_fullstack_round1_cpu_audit/records'
PARENT = ROOT / 'checkpoints/foundir_official_vlm_router_v4/official_vlm_v4_frozen4800_v1_4x2'
REPORT = ROOT / 'reports/foundir_fullstack_haze_priority_v1_2026-11-06.json'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    start = time.monotonic()
    receipt = json.loads((PARENT / 'receipt.json').read_text())
    rows = [r for r in receipt['rows'] if r['category'] == '08Haze']
    assert len(rows) == 200
    records, hashes = [], {}
    for row in rows:
        path = AUDIT / (row['record_id'] + '.json')
        r = json.loads(path.read_text())
        assert r['record_id'] == row['record_id'] and r['category'] == '08Haze'
        assert r['identity']['receipt_sha256'] == sha(PARENT / 'receipt.json')
        saved = json.loads((PARENT / 'results' / (row['record_id'] + '.json')).read_text())
        assert saved['decisions']['conditional_v4']['anchor'] == 'haze'
        assert saved['metrics']['haze']['psnr'] == r['saved_known_class_psnr']
        assert saved['metrics']['haze']['lpips'] == r['known_class_lpips']
        assert all(math.isfinite(v) for v in r['mean_rgb_error'])
        records.append(r)
        hashes[row['record_id']] = sha(path)
    mean = lambda xs: math.fsum(xs) / len(xs)
    fractions = []
    for r in records:
        mse = 10 ** (-r['known_class_psnr_diagnostic'] / 10) - 1e-8
        bias_energy = mean([v * v for v in r['mean_rgb_error']])
        assert mse > 0 and 0 <= bias_energy <= mse + 1e-8
        fractions.append(bias_energy / mse)
    psnr = mean([r['saved_known_class_psnr'] for r in records])
    result = {
        'status': 'complete', 'scope': 'existing Haze200 error statistics only',
        'source_sha256': sha(Path(__file__)),
        'parent_receipt_sha256': sha(PARENT / 'receipt.json'),
        'record_hashes': hashes, 'records': 200,
        'psnr': psnr, 'lpips': mean([r['known_class_lpips'] for r in records]),
        'psnr_gap_db': 22.1 - psnr,
        'geometric_mean_MSE_reduction_required_fraction': 1 - 10 ** (-(22.1 - psnr) / 10),
        'PSNR_improved_over_input_count': sum(r['known_class_psnr_gain_db'] > 0 for r in records),
        'mean_RGB_error': [mean([r['mean_rgb_error'][c] for r in records]) for c in range(3)],
        'positive_RGB_error_counts': [sum(r['mean_rgb_error'][c] > 0 for r in records) for c in range(3)],
        'mean_fraction_error_energy_in_per_image_RGB_mean': mean(fractions),
        'median_fraction_error_energy_in_per_image_RGB_mean': statistics.median(fractions),
        'mean_tile_edge_to_interior_MSE_ratio': mean([
            r['regions']['tile_edge_bands']['output_mse'] / r['regions']['interior']['output_mse']
            for r in records]),
        'gpus': 0, 'training': False, 'parameter_fitting': False,
        'new_restorations': False, 'GT_used_only_for_diagnostics': True,
        'independent_blind_test': False,
        'decision': 'Prioritize Haze; inspect current-checkpoint legal training residuals before a bounded change. Do not fit correction parameters on official GT.',
        'limitations': [
            'Small mean gap does not guarantee an easy or reliable improvement.',
            'RGB-mean error energy is a diagnostic decomposition, not a deployable correction or achievable gain.',
            'Mixed bias signs do not justify a constant brightness shift.',
            'Historical step400 scalar and global-affine attempts failed; do not replay them as new evidence.',
            'Edge/interior association does not establish a stitching cause.',
        ],
        'seconds': time.monotonic() - start, 'exit_code': 0,
    }
    # Preserve a completed audit rather than overwrite its original runtime.
    if REPORT.exists():
        old = json.loads(REPORT.read_text())
        assert all(old[k] == v for k, v in result.items() if k != 'seconds')
        print('HAZE_PRIORITY_AUDIT_REUSED', REPORT)
    else:
        with REPORT.open('x') as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2)
            stream.write('\n')
        print('HAZE_PRIORITY_AUDIT_COMPLETE', result['seconds'], REPORT)


if __name__ == '__main__':
    main()
