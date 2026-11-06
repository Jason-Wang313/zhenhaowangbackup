#!/usr/bin/env python3
"""Two frozen checkpoints, first8 training inputs, real weighted LoRA gradient geometry."""
import argparse
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'third_party/flux2/src'))
from smoke import run_foundir_fullstack_blur_coverage_v2 as t
from smoke import diagnose_foundir_blur_coverage_fit_v2 as fit
from smoke import evaluate_foundir_blur_coverage_official_v2 as official
from smoke.blur_loss_gradient_geometry_v1 import geometry
from smoke.existing_flux_replay_contract_v1 import read,sha,write_new as write
OUT=ROOT/'checkpoints/foundir_blur_loss_gradients_v1'
REPORT=ROOT/'reports/foundir_blur_loss_gradients_v1_result_2026-11-06.json'
SOURCES=['smoke/diagnose_foundir_blur_loss_gradients_v1.py','smoke/blur_loss_gradient_geometry_v1.py',
         'smoke/run_foundir_fullstack_blur_loss_gradients_v1.sh']
need=t.need


def prepare():
    start=time.monotonic();parent=fit.bound();previous=read(fit.REPORT);training=t.bound()
    need(previous['status']=='training_fit_diagnostic_complete','64-fit diagnosis required')
    items=training['rows'][:8]
    for i in items:need(t.cached(i,'cpu') is not None and i['role']=='train','fixed training cache required')
    stages={'initial':{'weight':training['lora'],'parameter_hash':read(t.OUT/'pilot/steps/step_000000.json')['parameter_hash']},
      'step200':{'weight':parent['weight'],'parameter_hash':parent['parameter_hash']}}
    for stage in stages.values():need(sha(ROOT/stage['weight']['path'])==stage['weight']['sha256'],'weight changed')
    value={'items':items,'stages':stages,'sources':{n:sha(ROOT/n) for n in sorted(set(parent['sources'])|set(SOURCES)|{'smoke/diagnose_foundir_blur_coverage_fit_v2.py'})},
      'fit_receipt_sha256':sha(fit.OUT/'receipt.json'),'fit_result_sha256':sha(fit.REPORT),
      'train_receipt_sha256':sha(t.OUT/'receipt.json'),'weighted_objective':'10*MSE_RGB01 + .1*LPIPS',
      'parameter_updates':False,'optimizer_steps':0,'world':2,'global_batch':8,'quality_gate':False,
      'SGD_direction_diagnostic_is_not_Adam_update_evidence':True}
    write(OUT/'receipt.json',value)
    print('BLUR_LOSS_GRADIENT_PREFLIGHT_PASS '+json.dumps({'seconds':time.monotonic()-start,'images':8,'checkpoints':2,'GPU_hours':0}),flush=True)


def bound():
    b=read(OUT/'receipt.json')
    need(b['items']==t.bound()['rows'][:8] and b['train_receipt_sha256']==sha(t.OUT/'receipt.json')
      and b['fit_receipt_sha256']==sha(fit.OUT/'receipt.json') and b['fit_result_sha256']==sha(fit.REPORT),'parent/scope changed')
    for n,d in b['sources'].items():need(sha(ROOT/n)==d,'source changed: '+n)
    for stage in b['stages'].values():need(sha(ROOT/stage['weight']['path'])==stage['weight']['sha256'],'checkpoint changed')
    return b


def run():
    import torch
    import torch.distributed as dist
    from datetime import timedelta
    from lowlight_restoration.lora import load_lora_checkpoint,lora_state_dict
    from smoke.official_vlm_v4_metric_identity_fix import checked_metric_models
    start=time.monotonic();b=bound();training=t.bound();rank,device=t.setup(training)
    dist.init_process_group('nccl',timeout=timedelta(minutes=10));need(dist.get_world_size()==2,'exact two ranks required')
    model=t.load_model(training,device);ae,prompt=t.load_vae_prompt(training,device)
    metrics,_=checked_metric_models(device,training['metric_identity']);metrics['lpips'].as_loss=True
    named=[(n,p) for n,p in model.named_parameters() if p.requires_grad];names=[n for n,_ in named];parameters=[p for _,p in named]
    need(len(parameters)==100 and sum(p.numel() for p in parameters)==32440320,'LoRA parameters changed')
    phase_hashes={}
    for stage in ('initial','step200'):
        load_lora_checkpoint(model,ROOT/b['stages'][stage]['weight']['path']);model.train()
        before=t.state.parameter_hash(lora_state_dict(model));need(before==b['stages'][stage]['parameter_hash'],'loaded model differs')
        sum_a=[torch.zeros_like(p) for p in parameters];sum_b=[torch.zeros_like(p) for p in parameters]
        for item in b['items'][rank::2]:
            key=t.parent.key(item);data=t.cached(item,device);prediction=t.prediction(model,ae,prompt,data)
            need(torch.isfinite(prediction).all() and prediction.requires_grad,'invalid prediction graph')
            image=t.core.uint8_output(prediction)[0].permute(1,2,0).cpu().numpy()
            if stage=='initial':need(torch.equal(prediction.detach(),data['lq']),'initialization not exact LQ')
            else:
                import numpy as np
                record=read(fit.OUT/'records'/(key+'.json'));spec=record['output']
                need(record['parameter_hash']==before and record['receipt_sha256']==b['fit_receipt_sha256'],'prior fixed output binding differs')
                previous=official.png(fit.OUT/spec['path'],spec['sha256'])
                need(np.array_equal(image,previous),'gradient-mode output differs from fixed inference')
            # Only the eight existing training GTs are used to form loss derivatives.
            gt=t.rgb(item,'gt',device);loss,terms=t.core.objective(prediction,gt,metrics['lpips'])
            a=torch.autograd.grad(10*terms['mse'],(prediction,*parameters),retain_graph=True)
            z=torch.autograd.grad(.1*terms['lpips'],(prediction,*parameters))
            rgb=geometry(a[:1],z[:1],['RGB'])['all'];parameter=geometry(a[1:],z[1:],names)
            for dest,x in zip(sum_a,a[1:]):dest.add_(x,alpha=1/8)
            for dest,x in zip(sum_b,z[1:]):dest.add_(x,alpha=1/8)
            need(all(p.grad is None for p in parameters),'autograd diagnostic accumulated .grad')
            record={'stage':stage,'item':item,'rank':rank,'receipt_sha256':sha(OUT/'receipt.json'),
              'parameter_hash':before,'weighted_RGB_gradient_geometry':rgb,'weighted_parameter_gradient_geometry':parameter,
              'continuous_training_losses':{'mse':float(terms['mse'].detach()),'lpips':float(terms['lpips'].detach())},
              'prediction_matches_prior_path':True,'parameter_updates':False,'optimizer_step_executed':False}
            write(OUT/'records'/(stage+'_'+key+'.json'),record)
            print('BLUR_LOSS_GRADIENT_RECORD '+json.dumps({'stage':stage,'key':key,'rank':rank,'parameters':parameter['all']}),flush=True)
            del data,prediction,image,gt,loss,terms,a,z
        for a,z in zip(sum_a,sum_b):dist.all_reduce(a);dist.all_reduce(z)
        batch=geometry(sum_a,sum_b,names)
        write(OUT/f'{stage}_batch_rank{rank}.json',{'stage':stage,'rank':rank,'receipt_sha256':sha(OUT/'receipt.json'),
          'global_batch':8,'parameter_hash':before,'weighted_parameter_gradient_geometry':batch})
        after=t.state.parameter_hash(lora_state_dict(model));need(after==before,'diagnostic modified checkpoint')
        phase_hashes[stage]={'before':before,'after':after};del sum_a,sum_b
        dist.barrier()
    write(OUT/f'worker{rank}.json',{'rank':rank,'receipt_sha256':sha(OUT/'receipt.json'),
      'keys':[t.parent.key(i) for i in b['items'][rank::2]],'phase_hashes':phase_hashes,
      'seconds':time.monotonic()-start,'peak_allocated_GiB':torch.cuda.max_memory_allocated(device)/1024**3})
    dist.barrier();dist.destroy_process_group();print(f'BLUR_LOSS_GRADIENT_END rank={rank} status=0',flush=True)


def finish():
    import statistics
    start=time.monotonic();b=bound();records=[];batches={};summary={}
    for rank in (0,1):
        w=read(OUT/f'worker{rank}.json');need(w['rank']==rank and w['receipt_sha256']==sha(OUT/'receipt.json')
          and w['keys']==[t.parent.key(i) for i in b['items'][rank::2]],'rank incomplete')
        for stage in ('initial','step200'):
            need(w['phase_hashes'][stage]['before']==w['phase_hashes'][stage]['after']==b['stages'][stage]['parameter_hash'],'checkpoint changed')
            for item in b['items'][rank::2]:
                r=read(OUT/'records'/(stage+'_'+t.parent.key(item)+'.json'))
                need(r['item']==item and r['rank']==rank and r['stage']==stage and r['receipt_sha256']==sha(OUT/'receipt.json')
                  and r['parameter_hash']==b['stages'][stage]['parameter_hash'] and r['parameter_updates'] is False
                  and r['prediction_matches_prior_path'] is True,'record mismatch')
                records.append(r)
    for stage in ('initial','step200'):
        a,z=[read(OUT/f'{stage}_batch_rank{r}.json') for r in (0,1)]
        for rank,v in enumerate((a,z)):
            need(v['rank']==rank and v['stage']==stage and v['receipt_sha256']==sha(OUT/'receipt.json')
              and v['parameter_hash']==b['stages'][stage]['parameter_hash'] and v['global_batch']==8,'batch identity mismatch')
        need(a['weighted_parameter_gradient_geometry']==z['weighted_parameter_gradient_geometry'],'global reduced gradients differ')
        batches[stage]=a['weighted_parameter_gradient_geometry'];subset=[r['weighted_parameter_gradient_geometry']['all'] for r in records if r['stage']==stage]
        ratios=[r['LPIPS_to_MSE_norm_ratio'] for r in subset if r['LPIPS_to_MSE_norm_ratio'] is not None]
        summary[stage]={'per_image_ratio_min':min(ratios) if ratios else None,'per_image_ratio_median':statistics.median(ratios) if ratios else None,
          'per_image_ratio_max':max(ratios) if ratios else None,'per_image_negative_cosine':sum(r['cosine'] is not None and r['cosine']<0 for r in subset),
          'per_image_mixed_raw_SGD_descent_increases_MSE':sum(r['raw_SGD_mixed_descent_first_order_increases_MSE'] for r in subset),
          'global_batch':batches[stage]['all']}
    value={'status':'fixed8_two_checkpoint_gradient_diagnostic_complete','records':records,'global_batches':batches,'summary':summary,
      'receipt_sha256':sha(OUT/'receipt.json'),'parameter_updates':False,'optimizer_steps':0,'quality_gate':False,
      'official_Blur_pass':False,'GT_scope':'fixed8_training_inputs_only','Adam_update_or_finite_improvement_proven':False,
      'seconds':time.monotonic()-start,'GPU_hours':0,'exit_code':0}
    write(REPORT,value);print('BLUR_LOSS_GRADIENT_VERIFIED '+json.dumps(summary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','run','finish']);globals()[p.parse_args().phase]()
