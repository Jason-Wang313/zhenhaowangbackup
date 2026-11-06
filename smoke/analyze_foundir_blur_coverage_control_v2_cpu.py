#!/usr/bin/env python3
"""Paired description of fixed-budget Blur data control; no candidate or GT fitting."""
import argparse
import json
import math
from pathlib import Path
import time
import hashlib
ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'checkpoints/foundir_blur_delta_velocity_v1'
NEW=ROOT/'checkpoints/foundir_blur_delta_coverage_v2'
OLD_OFF=ROOT/'checkpoints/foundir_blur_delta_official_v1'
NEW_OFF=ROOT/'checkpoints/foundir_blur_coverage_official_v2'
RESULT_OLD=ROOT/'reports/foundir_blur_delta_official_v1_result_2026-11-06.json'
RESULT_NEW=ROOT/'reports/foundir_blur_coverage_official_v2_result_2026-11-06.json'
REPORT=ROOT/'reports/foundir_blur_coverage_control_v2_comparison_2026-11-06.json'


def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def need(ok,message):
    if not ok:raise ValueError(message)


def plan():
    a,b=read(OLD/'receipt.json'),read(NEW/'receipt.json')
    fixed=('train_steps','global_batch','learning_rate','weight_decay','clip_grad','seed','loss','inference','native_CFG',
      'reference_time','model_calls_per_image','vae','trainable','world','node','crop','sample_order','checkpoint')
    need(all(a['recipe'][k]==b['recipe'][k] for k in fixed),'non-data recipe changed')
    need(all(a[k]==b[k] for k in ('assets','model','lora','prompt','metric_identity')),'initial weights/numerics changed')
    pa,pb=read(OLD_OFF/'plan.json'),read(NEW_OFF/'plan.json')
    need(pa['rows']==pb['rows'] and len(pb['rows'])==150,'official populations/old metrics/routes differ')
    need(pa['metric_identity']==pb['metric_identity']==a['metric_identity'],'metric identity mismatch')
    initial_a,initial_b=[read(p/'pilot/steps/step_000000.json') for p in (OLD,NEW)]
    need(initial_a['parameter_hash']==initial_b['parameter_hash'],'initial LoRA differs')
    return {'only_changed_recipe_component':'training_data_population1080p_and_source_count_jointly; effects_not_separable',
      'fixed_recipe_fields':{k:b['recipe'][k] for k in fixed},'training_example_exposures_each':1600,
      'initial_parameter_hash_equal':True,'initial_parameter_hash':initial_b['parameter_hash'],
      'same_official150_rows_routes_and_parent_metrics':True,'same_metric_identity':True,
      'new_quality_gate':False,'GT_fitting_or_output_selection':False}


def finish():
    start=time.monotonic();comparison=plan();a,b=read(RESULT_OLD),read(RESULT_NEW)
    need(a['status']==b['status']=='complete_official_Blur150','complete fixed official results required')
    def index(result):
        records=result['records'];mapped={r['item']['row']['record_id']:r for r in records}
        need(len(records)==len(mapped)==150,'missing or duplicate official records')
        need(all(math.isfinite(v) for r in records for m in r['metrics'].values() for v in m.values()),'nonfinite metric')
        return mapped
    left,right=index(a),index(b);need(left.keys()==right.keys(),'different official IDs')
    records=[]
    for key in sorted(left):
        p,q=left[key],right[key];need(p['item']==q['item'],'changed row/route/baseline')
        need(p['GT_input_or_selection'] is False and q['GT_input_or_selection'] is False,'GT deployment selection')
        need(p['metrics']['lq']==q['metrics']['lq'],'LQ scoring path changed')
        records.append({'record_id':key,'LQ':q['metrics']['lq'],'old50step':q['metrics']['old50step'],
          'delta128':p['metrics']['candidate'],'delta1024':q['metrics']['candidate'],
          'PSNR_delta_1024_minus_128':q['metrics']['candidate']['psnr']-p['metrics']['candidate']['psnr'],
          'LPIPS_delta_1024_minus_128':q['metrics']['candidate']['lpips']-p['metrics']['candidate']['lpips'],
          'output128':p['output'],'output1024':q['output']})
    means={name:{k:math.fsum(r[name][k] for r in records)/150 for k in ('psnr','lpips','ssim')}
      for name in ('LQ','old50step','delta128','delta1024')}
    best=means['delta1024'];old=means['delta128']
    value={'status':'complete_paired_official150_fixed_budget_control','comparison_contract':comparison,
      'means':means,'records':records,'official_Blur_pass':best['psnr']>=25.9 and best['lpips']<=.18,
      'PSNR_delta_vs_delta128':best['psnr']-old['psnr'],
      'LPIPS_relative_reduction_vs_delta128_percent':100*(1-best['lpips']/old['lpips']),
      'joint_better_than_delta128':sum(r['PSNR_delta_1024_minus_128']>0 and r['LPIPS_delta_1024_minus_128']<0 for r in records),
      'joint_worse_than_delta128':sum(r['PSNR_delta_1024_minus_128']<0 and r['LPIPS_delta_1024_minus_128']>0 for r in records),
      'parent_result_SHA256':{'delta128':sha(RESULT_OLD),'delta1024':sha(RESULT_NEW)},
      'official_is_adaptive_not_independent_blind_test':True,'other_evaluation_set_used':False,
      'candidate_averaging_or_selection':False,'seconds':time.monotonic()-start,'GPU_hours':0,'exit_code':0}
    need(not REPORT.exists(),'comparison report already exists; preserve')
    REPORT.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')
    print('BLUR_COVERAGE_CONTROL_COMPARED '+json.dumps({k:v for k,v in value.items() if k not in ('records','comparison_contract')}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['plan','finish']);a=p.parse_args()
    if a.phase=='plan':print(json.dumps(plan(),indent=2))
    else:finish()
