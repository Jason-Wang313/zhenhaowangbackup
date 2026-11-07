#!/usr/bin/env python3
"""Fixed-weight blending error identity on existing training8 outputs; no tuning."""
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from smoke import official_vlm_router_v4_contract as x
from smoke.analyze_foundir_blur_delta_official_errors_v1_cpu import image
from smoke.blur_context_geometry_v1 import crop
from smoke.existing_flux_replay_contract_v1 import read, sha, write_new as write
OUT = ROOT / 'checkpoints/foundir_blur_full_context_v1'
RESULT = ROOT / 'reports/foundir_blur_full_context_v1_result_2026-11-07.json'
REPORT = ROOT / 'reports/foundir_blur_context_blending_v1_CPU_2026-11-07.json'


def main():
    import numpy as np
    start = time.monotonic()
    result = read(RESULT)
    assert result['status'] == 'training8_full_context_diagnostic_complete'
    assert len(result['records']) == 8
    records = []
    for record in result['records']:
        item, row = record['item'], record['item']['row']
        gt = image(ROOT / row['gt_path'], row['gt_sha256'])
        full = image(OUT / record['output']['path'], record['output']['sha256'])
        control = record['uncached_crop']['output']
        standalone = image(OUT / control['path'], control['sha256'])
        total = np.zeros_like(gt)
        square_error = np.zeros_like(gt)
        norm = np.zeros(gt.shape[:2], dtype=np.float64)
        native_tiles = {}
        for tile in x.tiles(row):
            marker = read(OUT / 'tiles' / (tile['key'] + '.json'))
            assert marker['identity']['item'] == item and marker['identity']['tile'] == tile
            assert marker['GT_input'] is False
            spec = marker['png']
            prediction = image(OUT / spec['path'], spec['sha256'])
            native_tiles[tile['key']] = np.rint(prediction * 255).astype(np.uint8)
            tx, ty = tile['x'], tile['y']
            h, w = min(512, gt.shape[0] - ty), min(512, gt.shape[1] - tx)
            vy = np.hanning(h).astype(np.float32) if h > 2 else np.ones(h, np.float32)
            vx = np.hanning(w).astype(np.float32) if w > 2 else np.ones(w, np.float32)
            weight = np.maximum(np.sqrt(np.outer(vy, vx)), np.float32(1e-3)).astype(np.float64)
            delta = prediction[:h, :w] - gt[ty:ty + h, tx:tx + w]
            total[ty:ty + h, tx:tx + w] += prediction[:h, :w] * weight[..., None]
            square_error[ty:ty + h, tx:tx + w] += delta ** 2 * weight[..., None]
            norm[ty:ty + h, tx:tx + w] += weight
        # Actual native float32 stitch must reproduce the committed PNG exactly.
        np.testing.assert_array_equal(x.stitch(row, native_tiles), np.rint(full * 255).astype(np.uint8))
        mean = total / norm[..., None]
        weighted_error = square_error / norm[..., None]
        blend_error = (mean - gt) ** 2
        variance = weighted_error - blend_error
        assert float(variance.min()) > -1e-12, 'convex blending identity violated'
        quant_error = (full - gt) ** 2
        # Rounding and float32 accumulation must stay within half a gray level plus tolerance.
        assert float(np.abs(full - mean).max()) <= .5 / 255 + 1e-6
        summary = {}
        for region, margin in (('trained_crop512', 0), ('trained_center384', 64)):
            target = crop(gt, item, margin)
            standalone_region = standalone if margin == 0 else standalone[margin:-margin, margin:-margin]
            values = {'standalone_MSE': float(np.mean((standalone_region - target) ** 2)),
                      'weighted_individual_tile_MSE': float(crop(weighted_error, item, margin).mean()),
                      'blended_float_MSE': float(crop(blend_error, item, margin).mean()),
                      'committed_uint8_MSE': float(crop(quant_error, item, margin).mean()),
                      'variance_removed_by_blending': float(crop(variance, item, margin).mean())}
            assert abs(values['weighted_individual_tile_MSE'] - values['blended_float_MSE']
                       - values['variance_removed_by_blending']) < 1e-12
            values['PSNR_standalone'] = -10 * math.log10(values['standalone_MSE'])
            values['PSNR_committed_uint8'] = -10 * math.log10(values['committed_uint8_MSE'])
            summary[region] = values
        assert abs(-10 * math.log10(float(quant_error.mean()))
                   - record['metrics_training_fit_only']['full_prediction']['psnr']) < 5e-5
        records.append({'record_id': row['record_id'], 'regions': summary,
                        'full_weighted_individual_tile_MSE': float(weighted_error.mean()),
                        'full_blended_float_MSE': float(blend_error.mean()),
                        'full_committed_uint8_MSE': float(quant_error.mean()),
                        'rounding_and_float32_max_abs_RGB01': float(np.abs(full - mean).max()),
                        'native_stitch_exactly_reproduced': True})
    means = {region: {key: math.fsum(r['regions'][region][key] for r in records) / 8
                      for key in records[0]['regions'][region]}
             for region in ('trained_crop512', 'trained_center384')}
    value = {'status': 'existing_training8_fixed_blend_identity_verified', 'records': records,
             'means': means, 'result_sha256': sha(RESULT), 'script_sha256': sha(Path(__file__)),
             'all_native_stitches_exactly_reproduced': True,
             'GT_use': 'post_generation_error_identity_only', 'parameters_fitted': False,
             'new_predictions_generated': False, 'GT_output_selection': False,
             'note': 'Fixed-weight blending reduces MSE versus the weighted constituent tile errors. '
                     'This is not a guarantee versus a different standalone crop or proof of a unique cause.',
             'new_quality_gate': False, 'seconds': time.monotonic() - start, 'GPU_hours': 0, 'exit_code': 0}
    write(REPORT, value)
    print('BLUR_CONTEXT_BLEND_IDENTITY_END ' + json.dumps({k: v for k, v in value.items() if k != 'records'}), flush=True)


if __name__ == '__main__':
    main()
