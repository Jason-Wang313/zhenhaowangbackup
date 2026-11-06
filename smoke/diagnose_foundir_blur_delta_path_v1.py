#!/usr/bin/env python3
"""Eight fixed training inputs: cached/fresh/evaluation/gradient-mode consistency."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'third_party/flux2/src'))
from smoke import run_foundir_fullstack_blur_delta_v1 as t
from smoke import evaluate_foundir_blur_delta_official_v1 as official
from smoke.existing_flux_replay_contract_v1 import read,sha,write_new as write
OUT=ROOT/'checkpoints/foundir_blur_delta_path_v1'
SOURCES=['smoke/diagnose_foundir_blur_delta_path_v1.py','smoke/run_foundir_fullstack_blur_delta_path_v1.sh']


def prepare():
    begin=time.monotonic();b=t.bound();plan,final=official.bound()
    result=read(official.REPORT)
    t.need(result['status']=='complete_official_Blur150','complete official result required')
    items=[i for i in b['rows'] if i['role']=='train'][:8]
    t.need(len(items)==8,'fixed first8 training inputs required')
    for item in items:t.need(t.cached(item,'cpu') is not None,'training cache unavailable')
    value={'items':items,'train_receipt_sha256':sha(t.OUT/'receipt.json'),
           'official_receipt_sha256':sha(official.OUT/'receipt.json'),'official_result_sha256':sha(official.REPORT),
           'sources':{n:sha(ROOT/n) for n in sorted(set(SOURCES)|set(plan['sources']))},
           'trained_weight':final['weight'],'parameter_hash':final['parameter_hash'],'diagnostic_only':True,'quality_gate':False,
           'new_training':False,'new_evaluation_population':False,'node':3,'world':2}
    write(OUT/'receipt.json',value)
    print('BLUR_DELTA_PATH_PREFLIGHT_PASS '+json.dumps({'seconds':time.monotonic()-begin,'inputs':8,'training_only':True}),flush=True)


def bound():
    b=read(OUT/'receipt.json')
    t.need(b['train_receipt_sha256']==sha(t.OUT/'receipt.json') and b['official_receipt_sha256']==sha(official.OUT/'receipt.json')
           and b['official_result_sha256']==sha(official.REPORT),'parent binding changed')
    t.need(b['items']==[i for i in t.bound()['rows'] if i['role']=='train'][:8],'probe scope changed')
    for n,d in b['sources'].items():t.need(sha(ROOT/n)==d,'source changed: '+n)
    t.need(sha(ROOT/b['trained_weight']['path'])==b['trained_weight']['sha256'],'trained checkpoint changed')
    return b


def run():
    import torch
    from flux2.sampling import batched_prc_img
    from lowlight_restoration.lora import load_lora_checkpoint,lora_state_dict
    from smoke.official_vlm_v4_metric_identity_fix import checked_metric_models,producer
    begin=time.monotonic();b=bound();training=t.bound();rank,device=t.setup(training)
    model=t.load_model(training,device).eval();load_lora_checkpoint(model,ROOT/b['trained_weight']['path'])
    names=[n for n,p in model.named_parameters() if p.requires_grad]
    before=t.state.parameter_hash(lora_state_dict(model))
    t.need(before==b['parameter_hash'],'GPU loaded a different trained model')
    reference=t.load_model(training,device).eval().requires_grad_(False)
    ae,prompt=t.load_vae_prompt(training,device);metrics,_=checked_metric_models(device,training['metric_identity'])
    for item in b['items'][rank::2]:
        key=t.parent.key(item);cached=t.cached(item,device);lq=t.rgb(item,'lq',device)
        with torch.no_grad():
            latent=ae.encode(lq).float()
            velocity=t.core.native_velocity(reference,latent,prompt,batched_prc_img)
            decode=ae.decode(latent).float()
            fresh={'lq':lq,'latent':latent,'reference_velocity':velocity,'reference_decode':decode}
            cache_pred=t.prediction(model,ae,prompt,cached)
            fresh_pred=t.prediction(model,ae,prompt,fresh)
            model.requires_grad_(False)
            official_pred=t.prediction(model,ae,prompt,fresh)
        for name,param in model.named_parameters():param.requires_grad_(name in names)
        model.train()
        gradient_pred=t.prediction(model,ae,prompt,cached)
        t.need(gradient_pred.requires_grad,'training-mode graph missing')
        gradient_pred=gradient_pred.detach();model.eval()
        predictions={'cached_eval':cache_pred,'fresh_eval':fresh_pred,'official_mode':official_pred,'training_mode':gradient_pred}
        diffs={}
        for name,pred in predictions.items():
            diff=t.core.uint8_output(pred).int()-t.core.uint8_output(cache_pred).int()
            diffs[name]={'max_float_difference':float((pred-cache_pred).abs().max()),
                         'max_uint8_difference':int(diff.abs().max()),'uint8_different_values':int(torch.count_nonzero(diff))}
        cache_diffs={k:float((cached[k]-fresh[k]).abs().max()) for k in cached}
        outputs={}
        for name,pred in predictions.items():
            file=OUT/'png'/(key+'_'+name+'.png')
            producer.png_once(file,t.core.uint8_output(pred)[0].permute(1,2,0).cpu().numpy())
            outputs[name]={'path':str(file.relative_to(OUT)),'sha256':sha(file)}
        # No GT supplied to any of the four generation paths above.
        gt=(t.rgb(item,'gt',device)+1)*.5
        _,old=t.parent.cached('blur',item,training['parent_receipt_sha256'])
        old_tensor=torch.from_numpy(old['baseline']).permute(2,0,1).unsqueeze(0).to(device).float()/255
        values={}
        for name,pred in [('lq',(lq+1)*.5),('old50step',old_tensor),('candidate',t.core.uint8_output(cache_pred).float()/255)]:
            with torch.no_grad():values[name]={k:float(metric(pred,gt).mean()) for k,metric in metrics.items()}
        write(OUT/'records'/(key+'.json'),{'item':item,'rank':rank,'receipt_sha256':sha(OUT/'receipt.json'),
               'parameter_hash':before,'cached_vs_fresh_tensors_max_difference':cache_diffs,
               'path_differences':diffs,'metrics_training_fit_only':values,'outputs':outputs,'GT_input':False})
        print('BLUR_DELTA_PATH_RECORD '+json.dumps({'key':key,'rank':rank,'differences':diffs,'cache':cache_diffs}),flush=True)
        del predictions,cache_pred,fresh_pred,official_pred,gradient_pred,cached,fresh,latent,velocity,decode,lq,gt,old_tensor
    after=t.state.parameter_hash(lora_state_dict(model));t.need(after==before,'diagnosis changed weights')
    write(OUT/f'worker{rank}.json',{'rank':rank,'keys':[t.parent.key(i) for i in b['items'][rank::2]],
          'receipt_sha256':sha(OUT/'receipt.json'),'parameter_hash_before':before,'parameter_hash_after':after,
          'seconds':time.monotonic()-begin,'peak_allocated_GiB':torch.cuda.max_memory_allocated(device)/1024**3})
    print(f'BLUR_DELTA_PATH_END rank={rank} status=0',flush=True)


def finish():
    begin=time.monotonic();b=bound();records=[]
    for rank in (0,1):
        worker=read(OUT/f'worker{rank}.json')
        t.need(worker['receipt_sha256']==sha(OUT/'receipt.json') and worker['keys']==[t.parent.key(i) for i in b['items'][rank::2]]
               and worker['parameter_hash_before']==worker['parameter_hash_after']==b['parameter_hash'],'worker incomplete')
        for item in b['items'][rank::2]:
            r=read(OUT/'records'/(t.parent.key(item)+'.json'))
            t.need(r['item']==item and r['rank']==rank and r['receipt_sha256']==sha(OUT/'receipt.json')
                   and r['parameter_hash']==b['parameter_hash'] and r['GT_input'] is False,'record mismatch')
            for spec in r['outputs'].values():t.need(sha(OUT/spec['path'])==spec['sha256'],'probe PNG changed')
            records.append(r)
    summary={name:{m:sum(r['metrics_training_fit_only'][name][m] for r in records)/8 for m in ('psnr','lpips','ssim')}
             for name in ('lq','old50step','candidate')}
    identical=all(d['max_uint8_difference']==0 for r in records for d in r['path_differences'].values())
    result={'status':'diagnostic_complete','receipt_sha256':sha(OUT/'receipt.json'),'records':records,
            'all_four_paths_uint8_identical_on_fixed8':identical,'training_fit_only_summary':summary,
            'new_training':False,'quality_gate':False,'official_Blur_pass':False,
            'seconds':time.monotonic()-begin,'exit_code':0,'GPU_hours':0}
    write(ROOT/'reports/foundir_blur_delta_path_v1_result_2026-11-06.json',result)
    print('BLUR_DELTA_PATH_VERIFIED '+json.dumps({'four_paths_identical':identical,'training_fit_only':summary}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','run','finish']);globals()[p.parse_args().phase]()
