#!/usr/bin/env python3
"""Calibrate fixed loss scale on existing32 training probes, no GPU or optimization."""
import json
import math
from pathlib import Path
import statistics
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from smoke import diagnose_foundir_fullstack_rain_prefix_cross_v1 as prior
from lowlight_restoration.rain_logmse_objective_v1 import per_image_terms,EPSILON_RGB_MINUS1_1
REPORT=ROOT/'reports/foundir_rain_logmse_objective_v1_CPU_2026-11-06.json'
c=prior.c;train=prior.train


def main():
    start=time.monotonic();need=prior.need
    need(not REPORT.exists(),'preserve completed CPU report')
    value,receipt,cache=train.bound();mse={s:[] for s in (800,900,1000)};lpips={s:[] for s in mse};source_hashes={};pairs=[]
    for index in train.PROBES:
        row=receipt['rows'][index]
        need(row['split']=='train' and row['usage']=='train_update' and row['source_group'] not in value['excluded_source_groups'],'illegal training reference')
        data=train.cached(value,receipt,cache,index,'cpu');gt=data['gt'];predictions={}
        for step in mse:
            r,u8=prior.old_probe(index,step);prediction=u8.float()/127.5-1
            error=(prediction-gt).square().mean();psnr=-10*math.log10(float(error)/4)
            need(abs(psnr-r['metrics']['psnr'])<1e-5,'RGB MSE does not reproduce fixed probe PSNR')
            predictions[step]=prediction
            mse[step].append(float(error));lpips[step].append(r['metrics']['lpips'])
            p=train.OUT/'probes'/f'step_{step:06d}_image_{index}.json';source_hashes[str(p.relative_to(ROOT))]=c.sha(p)
        pairs.append((index,gt,predictions))
    reference=statistics.median(mse[800]);summaries={};weights={}
    for step in mse:
        values=[];effective=[]
        for offset,(index,gt,predictions) in enumerate(pairs):
            loss,terms=per_image_terms(predictions[step],gt,torch.tensor([lpips[step][offset]]),reference)
            values.append(float(loss));effective.append(float(terms['effective_mse_weight']))
        summaries[str(step)]={'quantized_mean_MSE_minus1_1':statistics.mean(mse[step]),
           'mean_PSNR':statistics.mean(-10*math.log10(x/4) for x in mse[step]),'mean_LPIPS_reused_from_verified_probe':statistics.mean(lpips[step]),
           'old_objective_on_quantized_probes':10*statistics.mean(mse[step])+.05*statistics.mean(lpips[step]),
           'candidate_objective_on_quantized_probes':statistics.mean(values)}
        weights[str(step)]={'min':min(effective),'median':statistics.median(effective),'max':max(effective)}
    result={'status':'CPU_objective_implementation_and_training_reference_checked_not_GPU_qualified',
       'source_sha256':{p:c.sha(ROOT/p) for p in ['lowlight_restoration/rain_logmse_objective_v1.py','smoke/check_foundir_rain_logmse_objective_v1_cpu.py']},
       'training_receipt_sha256':c.sha(train.OUT/'receipt.json'),'source_probe_markers':source_hashes,
       'training_reference_indices':train.PROBES,'reference_mse':reference,'reference_definition':'median32_predeclared_step800_training_probe_quantized_MSE_minus1_1',
       'epsilon':EPSILON_RGB_MINUS1_1,'MSE_coefficient':10.,'LPIPS_coefficient':.05,
       'formula':'10*(reference+eps)*mean_i(log((MSE_i+eps)/(reference+eps))) + .05*mean_i(LPIPS_i)',
       'per_image_effective_MSE_weights':weights,'summaries':summaries,
       'source_loss_vs_quantized_probe_distinction':'future training would use continuous decoded output; this CPU audit evaluates already-saved uint8 predictions only',
       'full32_step1000_old_objective_worse':summaries['1000']['old_objective_on_quantized_probes']>summaries['800']['old_objective_on_quantized_probes'],
       'conclusion':'Eight-crop mean-MSE/mean-PSNR conflict is real, but full32 also worsens under original loss. New loss is a candidate, not a proven complete root-cause fix.',
       'reference_gradient_matches_old_at_median':True,'single_gray_level_floor_limits_near_zero_weight':True,
       'LPIPS_recomputed':False,'new_training':False,'new_GPU':False,'official_or_development_pixels_read':False,
       'no_baseline_preservation_guarantee':True,'no_new_RGB_network':True,
       'GPU_pilot_training_recipe_or_launcher_ready':False,'automatic_training':False,
       'seconds':time.monotonic()-start,'GPU_hours':0,'exit_code':0}
    c.write_new(REPORT,result);print('RAIN_LOGMSE_CPU_COMPLETE '+json.dumps({k:result[k] for k in ('reference_mse','summaries','seconds')}),flush=True)

if __name__=='__main__':torch.set_num_threads(2);main()
