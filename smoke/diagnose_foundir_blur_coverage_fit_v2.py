#!/usr/bin/env python3
"""Fixed first64 training inputs: diagnose fit/direction only; no quality gate or updates."""
import argparse
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'third_party/flux2/src'))
from smoke import run_foundir_fullstack_blur_coverage_v2 as train
from smoke import evaluate_foundir_blur_coverage_official_v2 as official
from smoke.existing_flux_replay_contract_v1 import read,sha,write_new as write
OUT=ROOT/'checkpoints/foundir_blur_coverage_fit_v2'
REPORT=ROOT/'reports/foundir_blur_coverage_fit_v2_result_2026-11-06.json'
SOURCES=['smoke/diagnose_foundir_blur_coverage_fit_v2.py','smoke/run_foundir_fullstack_blur_coverage_fit_v2.sh']
need=train.need


def prepare():
    start=time.monotonic();b=train.bound();p,e=official.bound();result=read(official.REPORT)
    need(result['status']=='complete_official_Blur150' and result['official_Blur_pass'] is False,'failed complete official control required')
    items=b['rows'][:64];need(len(items)==64 and all(i['role']=='train' for i in items),'fixed64 training only')
    for i in items:need(train.cached(i,'cpu') is not None,'training cache missing')
    value={'items':items,'train_receipt_sha256':sha(train.OUT/'receipt.json'),'official_receipt_sha256':sha(official.OUT/'receipt.json'),
      'official_result_sha256':sha(official.REPORT),'weight':e['weight'],'parameter_hash':e['parameter_hash'],
      'sources':{n:sha(ROOT/n) for n in sorted(set(p['sources'])|set(SOURCES))},'new_training':False,
      'quality_gate':False,'population':'fixed_first64_training_inputs_used_in_current_training','official_Blur_pass':False,'node':3,'world':2}
    write(OUT/'receipt.json',value)
    print('BLUR_COVERAGE_FIT_PREFLIGHT_PASS '+json.dumps({'inputs':64,'seconds':time.monotonic()-start,'GPU_hours':0}),flush=True)


def bound():
    b=read(OUT/'receipt.json')
    need(sha(train.OUT/'receipt.json')==b['train_receipt_sha256'] and sha(official.OUT/'receipt.json')==b['official_receipt_sha256']
      and sha(official.REPORT)==b['official_result_sha256'],'parent changed')
    need(b['items']==train.bound()['rows'][:64],'fixed training scope changed')
    for n,d in b['sources'].items():need(sha(ROOT/n)==d,'source changed: '+n)
    need(sha(ROOT/b['weight']['path'])==b['weight']['sha256'],'checkpoint changed')
    return b


def run():
    import torch
    from lowlight_restoration.lora import load_lora_checkpoint,lora_state_dict
    from smoke.official_vlm_v4_metric_identity_fix import checked_metric_models,producer
    start=time.monotonic();b=bound();training=train.bound();rank,device=train.setup(training)
    model=train.load_model(training,device).eval();load_lora_checkpoint(model,ROOT/b['weight']['path']);model.requires_grad_(False)
    initial=train.state.parameter_hash(lora_state_dict(model));need(initial==b['parameter_hash'],'actual GPU model differs')
    ae,prompt=train.load_vae_prompt(training,device);metrics,_=checked_metric_models(device,training['metric_identity'])
    for item in b['items'][rank::2]:
        key=train.parent.key(item);data=train.cached(item,device)
        with torch.no_grad():pred=train.prediction(model,ae,prompt,data);array=train.core.uint8_output(pred)
        file=OUT/'png'/(key+'.png');producer.png_once(file,array[0].permute(1,2,0).cpu().numpy())
        # Commit the only output before reading the training GT. No selection/fallback.
        gt=(train.rgb(item,'gt',device)+1)*.5;lq=train.core.uint8_output(data['lq']).float()/255;candidate=array.float()/255
        with torch.no_grad():values={name:{k:float(m(image,gt).mean()) for k,m in metrics.items()} for name,image in [('lq',lq),('candidate',candidate)]}
        residual=candidate.double()-lq.double();target=gt.double()-lq.double();r2=float(residual.square().mean());g2=float(target.square().mean());dot=float((residual*target).mean())
        record={'item':item,'rank':rank,'receipt_sha256':sha(OUT/'receipt.json'),'parameter_hash':initial,
          'metrics_training_fit_only':values,'residual_MSE_norm':r2,'target_MSE_norm':g2,'residual_target_dot':dot,
          'cosine':dot/(r2*g2)**.5 if r2*g2 else None,'zero_residual':r2==0,'zero_target':g2==0,'nonpositive_direction':r2>0 and dot<=0,'aligned_but_excess_norm':dot>0 and r2>2*dot,
          'output':{'path':str(file.relative_to(OUT)),'sha256':sha(file)},'GT_input_or_selection':False}
        write(OUT/'records'/(key+'.json'),record)
        print('BLUR_COVERAGE_FIT_RECORD '+json.dumps({'key':key,'rank':rank,'metrics':values,'cosine':record['cosine']}),flush=True)
        del data,pred,array,gt,lq,candidate,residual,target
    final=train.state.parameter_hash(lora_state_dict(model));need(initial==final,'read-only diagnosis changed weights')
    write(OUT/f'worker{rank}.json',{'rank':rank,'keys':[train.parent.key(i) for i in b['items'][rank::2]],
      'receipt_sha256':sha(OUT/'receipt.json'),'parameter_hash_before':initial,'parameter_hash_after':final,
      'seconds':time.monotonic()-start,'peak_allocated_GiB':torch.cuda.max_memory_allocated(device)/1024**3})
    print(f'BLUR_COVERAGE_FIT_END rank={rank} status=0',flush=True)


def finish():
    import math
    start=time.monotonic();b=bound();records=[]
    for rank in (0,1):
        w=read(OUT/f'worker{rank}.json');need(w['rank']==rank and w['receipt_sha256']==sha(OUT/'receipt.json')
          and w['keys']==[train.parent.key(i) for i in b['items'][rank::2]]
          and w['parameter_hash_before']==w['parameter_hash_after']==b['parameter_hash'],'worker incomplete/changed')
        for item in b['items'][rank::2]:
            r=read(OUT/'records'/(train.parent.key(item)+'.json'));need(r['item']==item and r['rank']==rank
              and r['receipt_sha256']==sha(OUT/'receipt.json') and r['parameter_hash']==b['parameter_hash']
              and r['GT_input_or_selection'] is False,'record binding mismatch')
            need(sha(OUT/r['output']['path'])==r['output']['sha256'],'output changed');records.append(r)
    summary={name:{k:math.fsum(r['metrics_training_fit_only'][name][k] for r in records)/64 for k in ('psnr','lpips','ssim')} for name in ('lq','candidate')}
    value={'status':'training_fit_diagnostic_complete','summary':summary,'records':records,'receipt_sha256':sha(OUT/'receipt.json'),
      'nonpositive_direction_count':sum(r['nonpositive_direction'] for r in records),
      'aligned_but_excess_norm_count':sum(r['aligned_but_excess_norm'] for r in records),'cosine_mean':math.fsum(r['cosine'] for r in records if r['cosine'] is not None)/max(1,sum(r['cosine'] is not None for r in records)),
      'zero_residual_count':sum(r['zero_residual'] for r in records),'zero_target_count':sum(r['zero_target'] for r in records),
      'PSNR_worse_than_LQ_count':sum(r['metrics_training_fit_only']['candidate']['psnr']<r['metrics_training_fit_only']['lq']['psnr'] for r in records),
      'quality_gate':False,'new_training':False,'official_Blur_pass':False,'new_evaluation_population':False,
      'seconds':time.monotonic()-start,'exit_code':0,'GPU_hours':0}
    write(REPORT,value);print('BLUR_COVERAGE_FIT_VERIFIED '+json.dumps({k:v for k,v in value.items() if k!='records'}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','run','finish']);globals()[p.parse_args().phase]()
