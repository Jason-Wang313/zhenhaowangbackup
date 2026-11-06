#!/usr/bin/env python3
"""Descriptive error identities only: no fit, candidate selection, or new output."""
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from smoke.existing_flux_replay_contract_v1 import read, sha, write_new as write
OUT = ROOT/'checkpoints/foundir_blur_delta_official_v1'
RESULT = ROOT/'reports/foundir_blur_delta_official_v1_result_2026-11-06.json'
REPORT = ROOT/'reports/foundir_blur_delta_official_error_identity_v1_CPU_2026-11-06.json'
PARENT = ROOT/'checkpoints/foundir_official_vlm_router_v4/official_vlm_v4_frozen4800_v1_4x2'


def image(path, digest):
    import numpy as np
    from PIL import Image
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != digest: raise ValueError('image SHA changed')
    with Image.open(io.BytesIO(data)) as im: return np.array(im.convert('RGB'), dtype=np.float64)/255


def error_identity(lq, gt, pred):
    import numpy as np
    from scipy.fft import dctn
    target = gt-lq; residual = pred-lq; error = pred-gt
    g2 = float(np.mean(target**2)); r2 = float(np.mean(residual**2)); dot = float(np.mean(target*residual))
    mse = float(np.mean(error**2)); dc = float(np.mean(np.mean(error, axis=(0,1))**2))
    reconstructed = g2+r2-2*dot
    if abs(mse-reconstructed) > 1e-12: raise ValueError('error identity mismatch')
    h,w,_ = lq.shape
    # Orthogonal full-image cosine basis. Fixed pixel-frequency bins, no search.
    fy = np.arange(h)[:,None]/(2*h); fx = np.arange(w)[None,:]/(2*w)
    frequency = np.sqrt(fy**2+fx**2)
    edges = (0., 1/128, 1/32, 1/8, float('inf'))
    masks = [(frequency>=a)&(frequency<b) for a,b in zip(edges[:-1],edges[1:])]
    bands = np.zeros(4, dtype=np.float64)
    for channel in range(3):
        coeff = dctn(error[:,:,channel], norm='ortho', workers=2)
        for i,mask in enumerate(masks): bands[i] += np.sum(coeff[mask]**2)/(h*w*3)
    if abs(float(bands.sum())-mse) > 1e-12: raise ValueError('DCT Parseval mismatch')
    return {'MSE': mse, 'PSNR_RGB01': -10*math.log10(mse), 'LQ_MSE': g2,
            'residual_MSE_norm': r2, 'residual_target_dot_mean': dot,
            'residual_target_cosine': dot/math.sqrt(g2*r2) if g2*r2 else None,
            'residual_target_RMS_ratio': math.sqrt(r2/g2) if g2 else None,
            'MSE_change_from_LQ': r2-2*dot, 'opposite_or_orthogonal_correction': bool(r2>0 and dot<=0),
            'aligned_but_excess_norm_for_observed_direction': bool(dot>0 and r2>2*dot),
            'error_DC_MSE': dc, 'error_DC_fraction': dc/mse,
            'orthogonal_error_band_MSE': bands.tolist()}


def main():
    start = time.monotonic(); result = read(RESULT)
    if result['status'] != 'complete_official_Blur150' or len(result['records']) != 150:
        raise ValueError('complete official results required first')
    rows = []
    for record in result['records']:
        item = record['item']; row = item['row']; rid = row['record_id']
        oldpath = PARENT/'results'/(rid+'.json')
        if sha(oldpath) != item['parent_record_sha256']: raise ValueError('old record changed')
        old = read(oldpath)['outputs']['blur']
        lq, gt = [image(ROOT/row[k+'_path'], row[k+'_sha256']) for k in ('lq','gt')]
        arms = {}
        for name,root,spec in [('old50step',PARENT,old),('candidate',OUT,record['output'])]:
            pred = image(root/spec['path'], spec['sha256'])
            if pred.shape != lq.shape or gt.shape != lq.shape: raise ValueError('image geometry changed')
            arms[name] = error_identity(lq,gt,pred)
            if abs(arms[name]['PSNR_RGB01']-record['metrics'][name]['psnr']) > 5e-5:
                raise ValueError('independent PSNR differs from frozen pyiqa')
        rows.append({'record_id':rid, 'lq_PSNR':record['metrics']['lq']['psnr'], 'arms':arms})
        if len(rows)%25 == 0: print(f'BLUR_DELTA_ERROR_IDENTITY records={len(rows)}/150', flush=True)
    summary = {}
    for name in ('old50step','candidate'):
        values = [r['arms'][name] for r in rows]
        means = {key: math.fsum(v[key] for v in values)/150 for key in
                 ('MSE','LQ_MSE','residual_MSE_norm','residual_target_dot_mean','error_DC_MSE','error_DC_fraction')}
        means['MSE_worse_than_LQ_count'] = sum(v['MSE_change_from_LQ']>0 for v in values)
        means['nonpositive_direction_count'] = sum(v['opposite_or_orthogonal_correction'] for v in values)
        means['aligned_but_excess_norm_count'] = sum(v['aligned_but_excess_norm_for_observed_direction'] for v in values)
        means['cosine_mean'] = math.fsum(v['residual_target_cosine'] for v in values if v['residual_target_cosine'] is not None)/sum(v['residual_target_cosine'] is not None for v in values)
        means['orthogonal_error_band_MSE'] = [math.fsum(v['orthogonal_error_band_MSE'][i] for v in values)/150 for i in range(4)]
        summary[name] = means
    write(REPORT, {'status':'descriptive_complete', 'result_sha256':sha(RESULT), 'records':rows, 'summary':summary,
                   'frequency_edges_cycles_per_pixel':[0.,1/128,1/32,1/8,'infinity'],
                   'GT_use':'post_generation_error_statistics_only', 'parameters_fitted':False,
                   'new_predictions_or_candidates_generated':False, 'GT_output_selection':False,
                   'is_acceptance_gate':False, 'official_only':True,
                   'note':'Excess-norm classification uses GT only for diagnosis, is not a deployable rejector; no optimal scaling is fitted.',
                   'seconds':time.monotonic()-start, 'exit_code':0, 'GPU_hours':0})
    print('BLUR_DELTA_ERROR_IDENTITY_END '+json.dumps(summary), flush=True)


if __name__ == '__main__': main()
