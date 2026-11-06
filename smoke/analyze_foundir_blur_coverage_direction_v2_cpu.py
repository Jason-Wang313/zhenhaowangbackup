#!/usr/bin/env python3
"""Describe correction energy/alignment on fixed official outputs; never fit alpha or choose output."""
import json
import math
from pathlib import Path
import sys
import time
from concurrent.futures import ThreadPoolExecutor
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from smoke.analyze_foundir_blur_delta_official_errors_v1_cpu import image
from smoke.existing_flux_replay_contract_v1 import read,sha,write_new as write
RESULT=ROOT/'reports/foundir_blur_coverage_official_v2_result_2026-11-06.json'
OLD=ROOT/'reports/foundir_blur_delta_official_error_identity_v1_CPU_2026-11-06.json'
OUT=ROOT/'checkpoints/foundir_blur_coverage_official_v2'
REPORT=ROOT/'reports/foundir_blur_coverage_direction_v2_CPU_2026-11-06.json'


def describe(record):
    import numpy as np
    row=record['item']['row'];spec=record['output']
    lq,gt=[image(ROOT/row[k+'_path'],row[k+'_sha256']) for k in ('lq','gt')]
    output=image(OUT/spec['path'],spec['sha256'])
    if output.shape!=lq.shape or gt.shape!=lq.shape:raise ValueError('geometry mismatch')
    target=gt-lq;residual=output-lq
    g2=float(np.mean(target*target));r2=float(np.mean(residual*residual));dot=float(np.mean(target*residual))
    error=output-gt;mse=float(np.mean(error*error));dc=float(np.mean(np.mean(error,axis=(0,1))**2))
    if abs(mse-(g2+r2-2*dot))>1e-12:raise ValueError('MSE identity failed')
    psnr=-10*math.log10(mse)
    if abs(psnr-record['metrics']['candidate']['psnr'])>5e-5:raise ValueError('independent PSNR differs')
    return {'record_id':row['record_id'],'MSE':mse,'LQ_MSE':g2,'residual_MSE_norm':r2,
      'residual_target_dot_mean':dot,'residual_target_cosine':dot/math.sqrt(g2*r2) if g2*r2 else None,
      'MSE_change_from_LQ':r2-2*dot,'nonpositive_direction':bool(r2>0 and dot<=0),
      'aligned_but_excess_norm':bool(dot>0 and r2>2*dot),'error_DC_MSE':dc,'error_DC_fraction':dc/mse,
      'independent_PSNR_RGB01':psnr,'output_SHA256':spec['sha256']}


def main():
    start=time.monotonic();result=read(RESULT);old=read(OLD)
    if result['status']!='complete_official_Blur150' or len(result['records'])!=150:raise ValueError('full150 required')
    if old['result_sha256']!=sha(ROOT/'reports/foundir_blur_delta_official_v1_result_2026-11-06.json'):raise ValueError('old diagnostic source changed')
    with ThreadPoolExecutor(max_workers=2) as pool:records=list(pool.map(describe,result['records']))
    if len({r['record_id'] for r in records})!=150:raise ValueError('duplicate IDs')
    summary={k:math.fsum(r[k] for r in records)/150 for k in ('MSE','LQ_MSE','residual_MSE_norm','residual_target_dot_mean','error_DC_MSE','error_DC_fraction')}
    cosines=[r['residual_target_cosine'] for r in records if r['residual_target_cosine'] is not None]
    summary.update(cosine_mean=math.fsum(cosines)/len(cosines),
      MSE_worse_than_LQ_count=sum(r['MSE_change_from_LQ']>0 for r in records),
      nonpositive_direction_count=sum(r['nonpositive_direction'] for r in records),
      aligned_but_excess_norm_count=sum(r['aligned_but_excess_norm'] for r in records))
    value={'status':'post_generation_description_complete','summary':{'delta1024':summary,
      'delta128':{k:old['summary']['candidate'][k] for k in summary},
      'old50step':{k:old['summary']['old50step'][k] for k in summary}},'records':records,
      'result_sha256':sha(RESULT),'old_diagnostic_sha256':sha(OLD),
      'GT_used_only_for_post_generation_statistics':True,'parameters_fitted':False,
      'new_predictions_generated':False,'GT_output_selection':False,'new_quality_gate':False,
      'note':'Residual correlation and excess-energy labels use GT for interpretation only, never as a deployable gate or alpha fit.',
      'seconds':time.monotonic()-start,'GPU_hours':0,'exit_code':0}
    write(REPORT,value);print('BLUR_COVERAGE_DIRECTION_END '+json.dumps(value['summary']),flush=True)


if __name__=='__main__':main()
