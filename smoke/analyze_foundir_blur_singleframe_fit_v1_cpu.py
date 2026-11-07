#!/usr/bin/env python3
"""Describe all fixed single-frame snapshots; no output selection or new inference."""
import json
import math
from pathlib import Path
import sys
import time
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from smoke.existing_flux_replay_contract_v1 import read, sha, write_new
from smoke.blur_singleframe_contract_v1 import key

OUT = ROOT/'checkpoints/foundir_blur_singleframe_fit_v1'
RESULT = ROOT/'reports/foundir_blur_singleframe_fit_v1_result_2026-11-07.json'
REPORT = ROOT/'reports/foundir_blur_singleframe_fit_v1_interpretation_2026-11-07.json'


def pixels(file, expected):
    if sha(file) != expected: raise ValueError('input/output SHA mismatch')
    with Image.open(file) as image:
        return np.array(image.convert('RGB'), copy=True).astype(np.float64)/255.


def direction(lq, gt, pred):
    correction, desired = pred-lq, gt-lq
    norm2, desired2 = float(np.sum(correction**2)), float(np.sum(desired**2))
    dot = float(np.sum(correction*desired))
    return {'correction_MSE': norm2/lq.size,
            'cosine': dot/math.sqrt(norm2*desired2) if norm2*desired2 else None,
            'zero_correction': norm2 == 0, 'nonpositive_direction': norm2 > 0 and dot <= 0,
            'positive_but_excessive': dot > 0 and norm2 > 2*dot}


def overlap_disagreement(arrays, items):
    total, elements, pairs = 0., 0, 0
    for i, a in enumerate(items[:15]):
        ax, ay = a['crop_xy']
        for j in range(i+1, 15):
            bx, by = items[j]['crop_xy']
            x0, x1 = max(ax, bx), min(ax+512, bx+512)
            y0, y1 = max(ay, by), min(ay+512, by+512)
            if x1 <= x0 or y1 <= y0: continue
            left = arrays[i][y0-ay:y1-ay, x0-ax:x1-ax]
            right = arrays[j][y0-by:y1-by, x0-bx:x1-bx]
            total += float(np.sum((left-right)**2)); elements += left.size; pairs += 1
    return {'overlap_pairs': pairs, 'compared_RGB_elements_including_repeated_pixels': elements,
            'same_pixel_prediction_MSE': total/elements}


def main():
    begun = time.monotonic(); b = read(OUT/'receipt.json'); result = read(RESULT)
    if (result['status'] != 'singleframe16_context_fixed200_training_fit_complete'
            or result['receipt_sha256'] != sha(OUT/'receipt.json')
            or result['scope'] != 'training_fit_only' or result['may_seed_formal']):
        raise ValueError('complete diagnostic required')
    row = b['items'][0]['row']
    if row['split'] != 'train' or row['category'] != '01Blur': raise ValueError('training only')
    lq = pixels(ROOT/row['lq_path'], row['lq_sha256']); gt = pixels(ROOT/row['gt_path'], row['gt_sha256'])
    by_step = {}; baseline = None
    for step in (1792, 1842, 1892, 1992):
        records = sorted([r for r in result['records'] if r['step'] == step], key=lambda r: r['index'])
        if len(records) != 16 or [r['index'] for r in records] != list(range(16)):
            raise ValueError('complete16 required')
        arrays, views = [], []
        for item, r in zip(b['items'], records):
            if item != r['item']: raise ValueError('view changed')
            spec = r['output']; pred = pixels(OUT/spec['path'], spec['sha256']); arrays.append(pred)
            x, y = item['crop_xy']; cut_lq, cut_gt = lq[y:y+512, x:x+512], gt[y:y+512, x:x+512]
            views.append({'key': key(item), 'metrics': r['metrics'], 'direction': direction(cut_lq, cut_gt, pred)})
        if baseline is None: baseline = views
        changes = [{k: v['metrics']['candidate'][k]-old['metrics']['candidate'][k] for k in ('psnr','lpips')}
                   for v, old in zip(views, baseline)]
        snapshot = result['snapshots'][str(step)]; spec = snapshot['full']['output']
        full = pixels(OUT/spec['path'], spec['sha256'])
        raw_mse = float(np.mean((full-gt)**2))
        psnr = -10*math.log10(raw_mse)
        if abs(psnr-snapshot['full']['candidate']['psnr']) > 1e-4:
            raise ValueError('independent full image PSNR differs')
        crop_item = b['items'][15]; x,y = crop_item['crop_xy']
        full_crop_psnr = -10*math.log10(float(np.mean((full[y:y+512,x:x+512]-gt[y:y+512,x:x+512])**2)))
        by_step[str(step)] = {'full_metrics': snapshot['full'], 'view_means': snapshot['means'],
                             'views': views, 'changes_vs1792': changes,
                             'both_better_vs1792': sum(c['psnr']>0 and c['lpips']<0 for c in changes),
                             'both_worse_vs1792': sum(c['psnr']<0 and c['lpips']>0 for c in changes),
                             'both_better_vs_LQ': snapshot['both_improve_count'],
                             'full_direction': direction(lq, gt, full),
                             'overlap_disagreement': overlap_disagreement(arrays,b['items']),
                             'original_crop_from_full_PSNR_float64': full_crop_psnr,
                             'original_crop_standalone_PSNR': views[15]['metrics']['candidate']['psnr'],
                             'independent_full_PSNR_float64': psnr}
    value = {'status': 'fixed_snapshots_training_context_analysis_complete', 'scope': 'training_fit_only',
             'source_sha256': sha(Path(__file__)), 'result_sha256': sha(RESULT), 'snapshots': by_step,
             'official_Blur_pass': False, 'new_quality_gate': False, 'new_predictions': False,
             'GT_fitted_parameters_or_selection': False, 'official_pixels_read': False,
             'seconds': time.monotonic()-begun, 'GPU_hours': 0, 'exit_code': 0}
    write_new(REPORT, value)
    print('BLUR_SINGLEFRAME_ANALYSIS_VERIFIED '+json.dumps({s: {k: v[k] for k in
          ('both_better_vs1792','both_worse_vs1792','both_better_vs_LQ','overlap_disagreement')}
          for s, v in by_step.items()}), flush=True)


if __name__ == '__main__': main()
