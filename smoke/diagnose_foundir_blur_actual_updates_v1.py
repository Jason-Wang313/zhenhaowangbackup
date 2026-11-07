#!/usr/bin/env python3
"""Replay two already-saved updates without an optimizer or new trained parameters."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'third_party/flux2/src'))
from smoke import run_foundir_fullstack_blur_coverage_v2 as t
from smoke import diagnose_foundir_blur_loss_gradients_v1 as prior
from smoke.blur_actual_update_tools_v1 import load_exact_step,original_batch,tensor_dot,replay_check
from smoke.existing_flux_replay_contract_v1 import read,sha,write_new as write
OUT=ROOT/'checkpoints/foundir_blur_actual_updates_v1'
REPORT=ROOT/'reports/foundir_blur_actual_updates_v1_result_2026-11-07.json'
SOURCES=['smoke/diagnose_foundir_blur_actual_updates_v1.py','smoke/blur_actual_update_tools_v1.py',
         'smoke/run_foundir_fullstack_blur_actual_updates_v1.sh']
need=t.need


def export_state(step,value):
    import torch
    from safetensors.torch import save_file,load_file
    path=OUT/'weights'/f'step{step}_{value["parameter_hash"]}.safetensors';path.parent.mkdir(parents=True,exist_ok=True)
    if not path.exists():
        temp=path.with_name(path.name+f'.pending.{time.time_ns()}');save_file(value['model'],str(temp))
        with temp.open('rb') as handle:os.fsync(handle.fileno())
        os.link(temp,path);temp.unlink()
    need(t.state.parameter_hash(load_file(str(path)))==value['parameter_hash'],'LoRA export mismatch')
    rng=OUT/'rng'/f'step{step}_{time.time_ns()}.pt';rng.parent.mkdir(parents=True,exist_ok=True)
    with rng.open('xb') as handle:
        torch.save(value['rank_rng'],handle);handle.flush();os.fsync(handle.fileno())
    return {'step':step,'parameter_hash':value['parameter_hash'],
      'weight':{'path':str(path.relative_to(ROOT)),'sha256':sha(path)},
      'rng':{'path':str(rng.relative_to(ROOT)),'sha256':sha(rng)}}


def prepare():
    start=time.monotonic();b=t.bound();old=prior.bound();result=read(prior.REPORT)
    need(result['status']=='fixed8_two_checkpoint_gradient_diagnostic_complete','previous gradient diagnosis required')
    receipt=sha(t.OUT/'receipt.json');states={};history={};order=None;groups=None
    for step in (0,1,199,200):
        marker,value=load_exact_step(t.OUT/'pilot',step,receipt)
        if order is None:order=value['parameter_order'];groups=value['optimizer']['param_groups']
        need(value['parameter_order']==order and value['optimizer']['param_groups']==groups,'optimizer/schema changed')
        states[str(step)]={**export_state(step,value),'original_step_receipt':marker}
        if step in (1,200):history[str(step-1)]=value['history'][step-1]
        del value
    transitions={}
    for before in (0,199):
        batch=original_batch(b['rows'],before)
        for items in batch['items_by_rank'].values():
            for item in items:need(t.cached(item,'cpu') is not None,'actual batch cache missing')
        transitions[f'{before}_to_{before+1}']={**batch,'before':before,'after':before+1,'logged_before_update':history[str(before)]}
    value={'states':states,'transitions':transitions,'parameter_order':order,'train_receipt_sha256':receipt,
      'prior_receipt_sha256':sha(prior.OUT/'receipt.json'),'prior_result_sha256':sha(prior.REPORT),
      'sources':{n:sha(ROOT/n) for n in sorted(set(old['sources'])|set(SOURCES))},
      'new_optimizer_steps':0,'world':2,'global_batch':8,'quality_gate':False,'official_GT_pixels':False,
      'replay_tolerances':{'loss_absolute':1e-7,'gradient_norm_normalized':3e-4}}
    write(OUT/'receipt.json',value)
    print('BLUR_ACTUAL_UPDATE_PREFLIGHT_PASS '+json.dumps({'seconds':time.monotonic()-start,'historical_checkpoints_checked':4,'actual_global_batches':2,'GPU_hours':0}),flush=True)


def bound():
    b=read(OUT/'receipt.json');train=t.bound()
    need(b['train_receipt_sha256']==sha(t.OUT/'receipt.json') and b['prior_receipt_sha256']==sha(prior.OUT/'receipt.json')
      and b['prior_result_sha256']==sha(prior.REPORT),'parent identity changed')
    for label,s in b['transitions'].items():
        batch=original_batch(train['rows'],s['before'])
        need(s['after']==s['before']+1 and label==f'{s["before"]}_to_{s["after"]}'
          and s['indices_by_rank']==batch['indices_by_rank'] and s['items_by_rank']==batch['items_by_rank'],'actual batch changed')
    for n,d in b['sources'].items():need(sha(ROOT/n)==d,'source changed: '+n)
    for s in b['states'].values():
        for name in ('weight','rng'):need(sha(ROOT/s[name]['path'])==s[name]['sha256'],'export changed')
    return b


def score(prediction,gt,metrics):
    import torch
    loss,terms=t.core.objective(prediction,gt,metrics['lpips'])
    with torch.no_grad():
        quantized=t.core.uint8_output(prediction)
        psnr=float(metrics['psnr'](quantized.float()/255,(gt+1)*.5).mean())
        digest=hashlib.sha256(quantized.cpu().numpy().tobytes()).hexdigest()
    values={'loss':float(loss.detach()),'mse':float(terms['mse'].detach()),'lpips':float(terms['lpips'].detach()),'psnr_uint8':psnr}
    need(all(math.isfinite(v) for v in values.values()),'nonfinite loss/score')
    return loss,terms,values,digest


def run():
    import torch
    import torch.distributed as dist
    from datetime import timedelta
    from safetensors.torch import load_file
    from lowlight_restoration.lora import load_lora_checkpoint,lora_state_dict
    from smoke.official_vlm_v4_metric_identity_fix import checked_metric_models
    begin=time.monotonic();b=bound();training=t.bound();rank,device=t.setup(training)
    dist.init_process_group('nccl',timeout=timedelta(minutes=10));need(dist.get_world_size()==2,'two ranks required')
    model=t.load_model(training,device);ae,prompt=t.load_vae_prompt(training,device)
    metrics,_=checked_metric_models(device,training['metric_identity']);metrics['lpips'].as_loss=True
    named=[(n,p) for n,p in model.named_parameters() if p.requires_grad];names=[n for n,p in named];parameters=[p for n,p in named]
    need(names==b['parameter_order'] and len(names)==100,'parameter order changed')
    phase_hashes={}
    for label in ('0_to_1','199_to_200'):
        stage=b['transitions'][label];before=b['states'][str(stage['before'])];after=b['states'][str(stage['after'])]
        load_lora_checkpoint(model,ROOT/before['weight']['path']);model.train()
        need(t.state.parameter_hash(lora_state_dict(model))==before['parameter_hash'],'wrong before model')
        saved_rng=torch.load(ROOT/before['rng']['path'],map_location='cpu',weights_only=True);t.state.restore_rng(saved_rng[rank],device)
        future=load_file(str(ROOT/after['weight']['path']))
        delta=[future[n].to(device)-p.detach() for n,p in named];del future
        need(tensor_dot(delta,delta)>0,'actual saved update is zero')
        sum_a=[torch.zeros_like(p) for p in parameters];sum_b=[torch.zeros_like(p) for p in parameters]
        for p in parameters:p.grad=None
        local={};totals=torch.zeros(4,device=device,dtype=torch.float64)
        for item in stage['items_by_rank'][str(rank)]:
            key=t.parent.key(item);data=t.cached(item,device);pred=t.prediction(model,ae,prompt,data)
            need(torch.isfinite(pred).all() and pred.requires_grad,'invalid before graph')
            gt=t.rgb(item,'gt',device);loss,terms,values,digest=score(pred,gt,metrics)
            a=torch.autograd.grad(10*terms['mse']/8,parameters,retain_graph=True)
            z=torch.autograd.grad(.1*terms['lpips']/8,parameters,retain_graph=True)
            # This backward follows the actual original combined-loss path/scaling.
            (loss/8).backward()
            for dest,x in zip(sum_a,a):dest.add_(x)
            for dest,x in zip(sum_b,z):dest.add_(x)
            local[key]={'item':item,'before':values,'before_quantized_SHA256':digest,
              'per_image_first_order_weighted_MSE_dot_actual_delta':8*tensor_dot(a,delta),
              'per_image_first_order_weighted_LPIPS_dot_actual_delta':8*tensor_dot(z,delta)}
            totals+=torch.tensor([values[k] for k in ('loss','mse','lpips','psnr_uint8')],device=device,dtype=torch.float64)/8
            del data,pred,gt,loss,terms,a,z
        need(all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters),'missing/nonfinite combined gradient')
        for a,z,p in zip(sum_a,sum_b,parameters):dist.all_reduce(a);dist.all_reduce(z);dist.all_reduce(p.grad)
        dist.all_reduce(totals)
        combined=[p.grad for p in parameters]
        norm=float(torch.linalg.vector_norm(torch.stack([g.norm(2) for g in combined]),2))
        values_before=dict(zip(('loss','mse','lpips','psnr_uint8'),totals.cpu().tolist()))
        check=replay_check(values_before,norm,stage['logged_before_update'])
        write(OUT/f'{label}_replay_rank{rank}.json',check)
        need(check['passed'],'original training did not reproduce: '+json.dumps(check))
        difference=[c-a-z for c,a,z in zip(combined,sum_a,sum_b)]
        derivative={'weighted_MSE_dot_actual_delta':tensor_dot(sum_a,delta),'weighted_LPIPS_dot_actual_delta':tensor_dot(sum_b,delta),
          'combined_dot_actual_delta':tensor_dot(combined,delta),'actual_delta_norm':math.sqrt(tensor_dot(delta,delta)),
          'separate_sum_vs_combined_relative_norm_error':math.sqrt(tensor_dot(difference,difference)/max(tensor_dot(combined,combined),1e-300)),
          'combined_gradient_norm':norm}
        need(t.state.parameter_hash(lora_state_dict(model))==before['parameter_hash'],'before model mutated during backward')
        del sum_a,sum_b,combined,difference,delta
        for p in parameters:p.grad=None
        load_lora_checkpoint(model,ROOT/after['weight']['path'])
        need(t.state.parameter_hash(lora_state_dict(model))==after['parameter_hash'],'wrong after model')
        totals_after=torch.zeros(4,device=device,dtype=torch.float64)
        for item in stage['items_by_rank'][str(rank)]:
            key=t.parent.key(item);data=t.cached(item,device)
            with torch.no_grad():
                pred=t.prediction(model,ae,prompt,data);need(torch.isfinite(pred).all(),'nonfinite after output')
                gt=t.rgb(item,'gt',device);loss,terms,values,digest=score(pred,gt,metrics)
            r={**local[key],'after':values,'after_quantized_SHA256':digest,'transition':label,'rank':rank,
              'receipt_sha256':sha(OUT/'receipt.json'),'new_optimizer_step':False,
              'before_parameter_hash':before['parameter_hash'],'after_parameter_hash':after['parameter_hash']}
            write(OUT/'records'/(label+'_'+key+'.json'),r)
            totals_after+=torch.tensor([values[k] for k in ('loss','mse','lpips','psnr_uint8')],device=device,dtype=torch.float64)/8
            del data,pred,gt,loss,terms
        dist.all_reduce(totals_after);values_after=dict(zip(('loss','mse','lpips','psnr_uint8'),totals_after.cpu().tolist()))
        need(t.state.parameter_hash(lora_state_dict(model))==after['parameter_hash'],'after model mutated')
        phase_hashes[label]={'before':before['parameter_hash'],'after':after['parameter_hash']}
        result={'transition':label,'rank':rank,'receipt_sha256':sha(OUT/'receipt.json'),'before':values_before,'after':values_after,
          'finite_changes':{k:values_after[k]-values_before[k] for k in values_before},'first_order_actual_update':derivative,
          'original_training_reproduced':True,'new_optimizer_steps':0,'phase_hashes':phase_hashes[label]}
        write(OUT/f'{label}_batch_rank{rank}.json',result)
        print('BLUR_ACTUAL_UPDATE_BATCH '+json.dumps(result),flush=True);dist.barrier()
    write(OUT/f'worker{rank}.json',{'rank':rank,'receipt_sha256':sha(OUT/'receipt.json'),'phase_hashes':phase_hashes,
      'seconds':time.monotonic()-begin,'peak_allocated_GiB':torch.cuda.max_memory_allocated(device)/1024**3})
    dist.barrier();dist.destroy_process_group();print(f'BLUR_ACTUAL_UPDATE_END rank={rank} status=0',flush=True)


def finish():
    begin=time.monotonic();b=bound();records=[];summaries={}
    for rank in (0,1):
        w=read(OUT/f'worker{rank}.json');need(w['rank']==rank and w['receipt_sha256']==sha(OUT/'receipt.json'),'worker incomplete')
        for label,stage in b['transitions'].items():
            pair={'before':b['states'][str(stage['before'])]['parameter_hash'],'after':b['states'][str(stage['after'])]['parameter_hash']}
            need(w['phase_hashes'][label]==pair,'worker checkpoint identity changed')
            check=read(OUT/f'{label}_replay_rank{rank}.json');need(check['passed'],'original replay failed')
            for item in stage['items_by_rank'][str(rank)]:
                r=read(OUT/'records'/(label+'_'+t.parent.key(item)+'.json'))
                need(r['item']==item and r['rank']==rank and r['transition']==label and r['receipt_sha256']==sha(OUT/'receipt.json')
                  and r['before_parameter_hash']==pair['before'] and r['after_parameter_hash']==pair['after']
                  and r['new_optimizer_step'] is False,'record identity differs');records.append(r)
    for label in b['transitions']:
        a,z=[read(OUT/f'{label}_batch_rank{r}.json') for r in (0,1)]
        need({k:v for k,v in a.items() if k!='rank'}=={k:v for k,v in z.items() if k!='rank'},'global rank summaries differ')
        subset=[r for r in records if r['transition']==label];need(len(subset)==8,'global8 incomplete')
        for phase in ('before','after'):
            for key in ('loss','mse','lpips','psnr_uint8'):
                need(abs(math.fsum(r[phase][key] for r in subset)/8-a[phase][key])<1e-10,'independent mean differs')
        summaries[label]={**a,'MSE_worse_after_count':sum(r['after']['mse']>r['before']['mse'] for r in subset),
          'LPIPS_worse_after_count':sum(r['after']['lpips']>r['before']['lpips'] for r in subset)}
    value={'status':'two_actual_historical_updates_reproduced','records':records,'summary':summaries,
      'receipt_sha256':sha(OUT/'receipt.json'),'new_optimizer_steps':0,'official_Blur_pass':False,'quality_gate':False,
      'official_pixels_accessed':False,'GT_scope':'two_original_training_global8_batches_only',
      'seconds':time.monotonic()-begin,'GPU_hours':0,'exit_code':0}
    write(REPORT,value);print('BLUR_ACTUAL_UPDATE_VERIFIED '+json.dumps(summaries),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','run','finish']);globals()[p.parse_args().phase]()
