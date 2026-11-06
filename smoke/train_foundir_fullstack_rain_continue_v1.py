#!/usr/bin/env python3
"""Bounded Rain step800->1000 continuation, global32 across three two-GPU nodes."""
import argparse
from datetime import timedelta
import gc
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'third_party/flux2/src'))
from smoke import train_foundir_rain_onpolicy_continue_v1 as old
from smoke import rain_native50_continue_state_v1 as state_io
from lowlight_restoration import rain_global_batch_v1 as batch_logic
c=old.c;parent=old.parent;logic=old.logic
PARENT=ROOT/'checkpoints/foundir_rain_onpolicy_continue_v1/continue600_to800_4x2_v1'
OUT=ROOT/'checkpoints/foundir_fullstack_rain_continue_3x2_v1'
REPORT=ROOT/'reports/foundir_fullstack_rain_continue_v1_result_2026-11-06.json'
PREFLIGHT=ROOT/'reports/foundir_fullstack_rain_continue_v1_preflight_2026-11-06.json'
PARENT_SHA='6f2c0e162e3d868b427308534836f90cbbc72f8d1c1aca4c043183f23fc259c0'
WORLD=6;START=800;END=1000;PROBES=list(range(12800,12832))
SOURCES=['smoke/train_foundir_fullstack_rain_continue_v1.py',
         'smoke/run_foundir_fullstack_rain_continue_v1.sh',
         'lowlight_restoration/rain_global_batch_v1.py',
         'smoke/expanded_vlm_h100_runtime_v4.py', 'smoke/rain_native50_continue_state_v1.py',
         'reports/foundir_fullstack_rain_official_overlap_audit_v1_2026-11-06.json',
         'reports/foundir_fullstack_rain_overlap_exposure_verified_2026-11-06.json']
RECIPE={'start_step':START,'end_step':END,'new_updates':200,'global_batch':32,
        'learning_rate':1e-5,'loss_weights':{'gt_rgb_mse':10.,'gt_lpips':.05},
        'readout_call_index':25,'prefix_calls_without_grad':25,'schedule_steps':50,
        'world':6,'nodes':3,'local_gpus':2,'migration_seed':2026110607,
        'migration':'copy exact model and Adam, preserve next_global_slot, reseed six ranks; not bitwise eight-rank resume',
        'logical_training_slots':[25600,32000],'training_probe_slots':[12800,12832],
        'cache_selection':'first6400 indices >=25600 excluding every source_group with official LQ/GT exact-file overlap',
        'legacy_checkpoint_has_confirmed_official_GT_source_overlap':True,
        'training_probe_steps':[800,900,1000],'official_or_development_training':False,
        'official_or_development_pixels_in_this_entry':False,
        'vae_policy':'vae_native_conv_fp32_query_chunk256_v1',
        'max_training_wall_seconds_per_attempt':10800,'automatic_extension':False}


def need(ok,message):
    if not ok:raise ValueError(message)


def prepare():
    start=time.monotonic();prior,receipt,_=old.load_receipt(PARENT)
    cache=ROOT/prior['cache_source'];record=c.read(PARENT/'steps/step_000800.json')
    need(record['sha256']==PARENT_SHA and record['step']==800,'wrong parent step800')
    need(c.sha(PARENT/record['file'])==PARENT_SHA,'parent state hash changed')
    need(c.sha(PARENT/'receipt.json')==record['receipt_hash'],'parent receipt changed')
    final=c.read(PARENT/'final_lora.json')
    need(final['parameter_hash']==record['parameter_hash'],'export and complete state differ')
    official=c.read(ROOT/'checkpoints/foundir_official_vlm_router_v4/official_vlm_v4_frozen4800_v1_4x2/receipt.json')
    protected={r[k] for r in official['rows'] for k in ('lq_sha256','gt_sha256')}
    excluded={r['source_group'] for r in receipt['rows'][:64000]
              if r['lq_sha256'] in protected or r['gt_sha256'] in protected}
    plan=[i for i in range(25600,64000) if receipt['rows'][i]['source_group'] not in excluded
          and all(receipt['rows'][i][k] not in protected for k in ('lq_sha256','gt_sha256'))][:6400]
    need(len(plan)==len(set(plan))==6400,'not enough eligible cached training presentations')
    indices=[*PROBES,*plan];hashes={}
    for index in indices:
        row=receipt['rows'][index]
        need(row['usage']=='train_update' and row['split']=='train' and row['category']=='10Rain','nontraining row')
        need(all(row[k].startswith('data/foundir/MillionIRData/Train/10Rain/') for k in ('lq_path','gt_path')),'illegal training path')
        need(row['lq_sha256'] not in protected and row['gt_sha256'] not in protected,'official exact-file overlap')
        need(row['source_group'] not in excluded,'related source group must also be excluded')
        marker=parent.cache_marker(cache,index);entry=c.read(marker)
        need(entry['identity']['row']==row and entry['identity']['index']==index,'cache metadata changed')
        need(c.path(cache,entry['tensor']).is_file(),'cache tensor missing')
        hashes[str(index)]=c.sha(marker)
    need(shutil.disk_usage(ROOT).free>100*1024**3,'need 100GiB; never delete old assets')
    sources={**prior['sources'],**{name:c.sha(ROOT/name) for name in SOURCES}}
    value={'recipe':RECIPE,'parent_checkpoint':record,'parent_receipt_sha256':c.sha(PARENT/'receipt.json'),
           'parent_result_sha256':c.sha(PARENT/'result.json'),'cache_source':str(cache.relative_to(ROOT)),
           'cache_receipt_sha256':c.sha(cache/'receipt.json'),'cache_markers':hashes,
           'training_cache_indices':plan,'excluded_source_groups':sorted(excluded),
           'selection_digest':c.digest([receipt['rows'][i] for i in indices]),'sources':sources}
    c.write_new(OUT/'receipt.json',value)
    report={'status':'CPU_metadata_preflight_pass','receipt_sha256':c.sha(OUT/'receipt.json'),
            'parent_state_SHA_verified':True,'new_training_presentations':6400,
            'local_sample_counts':[len(batch_logic.positions(800,r,WORLD)) for r in range(WORLD)],
            'training_probe_records':32,'official_exact_file_hashes_disjoint':True,
            'excluded_source_groups':len(excluded),'last_selected_cache_index':plan[-1],
            'legacy_exposure_is_not_erased_by_this_filter':True,
            'no_claim_of_semantic_source_disjointness_from_hashes_alone':True,
            'seconds':time.monotonic()-start,'GPU_started':False,'training_pixels_read':False}
    c.write_new(PREFLIGHT,report);print('RAIN_FULLSTACK_PREFLIGHT_PASS '+json.dumps(report),flush=True)


def bound():
    value=c.read(OUT/'receipt.json');need(value['recipe']==RECIPE,'recipe changed')
    for name,digest in value['sources'].items():need(c.sha(ROOT/name)==digest,'source changed: '+name)
    need(c.sha(PARENT/'receipt.json')==value['parent_receipt_sha256'] and
         c.sha(PARENT/'result.json')==value['parent_result_sha256'],'parent metadata changed')
    cache=ROOT/value['cache_source'];need(c.sha(cache/'receipt.json')==value['cache_receipt_sha256'],'cache receipt changed')
    receipt=c.read(cache/'receipt.json');indices=[*PROBES,*value['training_cache_indices']]
    need(c.digest([receipt['rows'][i] for i in indices])==value['selection_digest'],'data selection changed')
    return value,receipt,cache


def cached(value,receipt,cache,index,device,training=False):
    import torch
    from safetensors import safe_open
    need(index in (value['training_cache_indices'] if training else PROBES),'unbound train/probe pixel access')
    marker=parent.cache_marker(cache,index);need(c.sha(marker)==value['cache_markers'][str(index)],'cache marker changed')
    entry=c.read(marker);c.validate_record(entry,parent.identity(cache,receipt['rows'][index],index),cache)
    # Only these two tensors are loaded. Cached teacher/latent do not enter the objective.
    with safe_open(str(c.path(cache,entry['tensor'])),framework='pt',device=str(device)) as f:
        data={k:f.get_tensor(k) for k in ('lq','gt')}
    need(all(t.dtype==torch.float32 and t.shape==(1,3,512,512) and torch.isfinite(t).all()
             and (t.abs()<=1).all() for t in data.values()),'invalid training RGB')
    return data


def validate_state(saved,record,receipt_hash):
    import torch
    step=saved['step']
    need(800<=step<=1000 and record['step']==step and saved['world']==WORLD and len(saved['rank_rng'])==WORLD,
         'wrong continuation step/world')
    need(saved['receipt_hash']==record['receipt_hash']==receipt_hash and saved['next_global_slot']==32*step
         and len(saved['history'])==step,'checkpoint identity/progress mismatch')
    need(saved['parameter_order']==list(saved['model']) and
         state_io.parameter_hash(saved['model'])==record['parameter_hash']==saved['parameter_hash'],'model schema/hash changed')
    need(all(v.dtype==torch.float32 and torch.isfinite(v).all() for v in saved['model'].values()),'nonfinite/nonFP32 LoRA')
    states=saved['optimizer']['state'];groups=saved['optimizer']['param_groups']
    need(len(groups)==1 and groups[0]['lr']==1e-5 and groups[0]['weight_decay']==0 and len(states)==len(saved['model']),
         'Adam schema changed')
    for state in states.values():
        need(float(state['step'])==step and all(state[k].dtype==torch.float32 and torch.isfinite(state[k]).all()
             for k in ('exp_avg','exp_avg_sq')),'invalid Adam state')


def latest(value):
    import torch
    markers=sorted((OUT/'steps').glob('step_*.json'))
    if not markers:return None,None
    records=[c.read(p) for p in markers]
    need([r['step'] for r in records]==list(range(800,800+len(records))),'checkpoint gap; no silent rollback')
    record=records[-1];file=c.path(OUT,record['file'])
    need(file.parent==(OUT/'steps').resolve() and c.sha(file)==record['sha256'],'state hash/path changed')
    saved=torch.load(file,map_location='cpu',weights_only=True)
    validate_state(saved,record,c.sha(OUT/'receipt.json'))
    return saved,record


def probe(model,ae,prompt,metric,value,receipt,cache,step,rank,device):
    import torch
    rng=state_io.rng_state(device);model.eval()
    try:
        for index in PROBES[rank::WORLD]:
            marker=OUT/'probes'/f'step_{step:06d}_image_{index}.json'
            identity={'receipt_sha256':c.sha(OUT/'receipt.json'),'step':step,'index':index,
                      'checkpoint_receipt_sha256':c.sha(OUT/'steps'/f'step_{step:06d}.json'),
                      'cache_marker_sha256':value['cache_markers'][str(index)]}
            if marker.exists():c.validate_record(c.read(marker),identity,OUT);continue
            data=cached(value,receipt,cache,index,device)
            seed=receipt['config']['seed']+1000003*(index+1)
            rgb=old.previous.inference(model,ae,data['lq'],prompt,seed)
            image=parent.quantize(rgb);metrics=parent.scores(metric,image,data['gt'])
            name,digest=parent.replay.save_tensors(OUT,f'probe_{step}_{index}',{'uint8':image})
            c.write_new(marker,{'identity':identity,'index':index,'metrics':metrics,'tensor':name,
                               'files':{name:digest},'training_fit_only':True,'GT_used_for_output_selection':False})
            print(f'RAIN_FULLSTACK_PROBE rank={rank} step={step} index={index} metrics={metrics}',flush=True)
    finally:state_io.restore_rng(rng,device)


def milestone(step):
    rows=[]
    for index in PROBES:
        row=c.read(OUT/'probes'/f'step_{step:06d}_image_{index}.json')
        need(row['training_fit_only'] is True and row['index']==index,'invalid probe')
        c.validate_record(row,{'receipt_sha256':c.sha(OUT/'receipt.json'),'step':step,'index':index,
            'checkpoint_receipt_sha256':c.sha(OUT/'steps'/f'step_{step:06d}.json'),
            'cache_marker_sha256':c.read(OUT/'receipt.json')['cache_markers'][str(index)]},OUT)
        rows.append(row)
    means={k:math.fsum(r['metrics'][k] for r in rows)/32 for k in ('psnr','lpips')}
    oldrows=[c.read(PARENT/'probes'/f'step_000800_image_{i}.json') for i in PROBES]
    historical={k:math.fsum(r['metrics'][k] for r in oldrows)/32 for k in ('psnr','lpips')}
    reference=means if step==800 else c.read(OUT/'milestones/step_000800.json')['means']
    result={'step':step,'records':32,'training_fit_only':True,'means':means,
            'delta_vs_matched800':{k:means[k]-reference[k] for k in means},
            'historical800':historical,'official_gate_claim':False}
    c.write_new(OUT/'milestones'/f'step_{step:06d}.json',result)
    print('RAIN_FULLSTACK_MILESTONE '+json.dumps(result),flush=True)
    return result


def run(node):
    import torch
    import torch.distributed as dist
    from flux2.util import load_ae
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    from lowlight_restoration.rain_flow_adaptation import enable_native_checkpointing
    from lowlight_restoration.flux2_endpoint_precision import assert_fp32_optimizer
    from smoke.expanded_vlm_h100_runtime_v4 import bounded_vae,POLICY
    rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK'])
    need(int(os.environ['WORLD_SIZE'])==6 and int(os.environ['LOCAL_WORLD_SIZE'])==2 and
         node in range(3) and rank==2*node+local,'wrong three-node topology')
    value,receipt,cache=bound();torch.cuda.set_device(local);device=torch.device('cuda',local)
    need(torch.cuda.device_count()==2 and torch.cuda.get_device_properties(local).total_memory>=75*1024**3,'need two 75GiB Hopper GPUs')
    need(any(n in torch.cuda.get_device_name(local) for n in ('H100','H200')),'Hopper required')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    os.environ.update(KLEIN_4B_BASE_MODEL_PATH=str(ROOT/receipt['model']['flow_weights']),
                      AE_MODEL_PATH=str(ROOT/receipt['model']['autoencoder_weights']))
    for key in ('flow_weights','autoencoder_weights'):
        need(c.sha(ROOT/receipt['model'][key])==receipt['model'][key+'_sha256'],'base weight changed')
    dist.init_process_group('nccl',device_id=device,timeout=timedelta(minutes=10))
    try:
        ping=torch.tensor([rank+1.],device=device);dist.all_reduce(ping);need(ping.item()==21,'NCCL collective failed')
        print(f'RAIN_FULLSTACK_NCCL_VERIFIED rank={rank} world=6',flush=True)
        runtime={'torch':str(torch.__version__),'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),
                 'gpu':torch.cuda.get_device_name(local),'vae_policy':POLICY,'tf32':False,
                 'metrics':parent.metric_impl.metric_runtime({'metric':receipt['metric']})}
        need(runtime['metrics']==c.read(PARENT/'runtime_node0.json')['metrics'],'training metric identity changed')
        signatures=[None]*WORLD;dist.all_gather_object(signatures,runtime)
        need(all(s==signatures[0] for s in signatures),'numerical runtimes differ')
        c.write_new(OUT/f'runtime_rank{rank}.json',runtime)
        print(f'RAIN_FULLSTACK_LOADING rank={rank}',flush=True)
        model=parent.new_model(receipt,device);enable_native_checkpointing(model)
        ae=bounded_vae(parent.hopper.configure_vae(load_ae('flux.2-klein-base-4b',device=device).eval().requires_grad_(False)))
        prompt=parent.load_prompt(receipt,device)
        params=[p for p in model.parameters() if p.requires_grad]
        optimizer=torch.optim.AdamW(params,lr=1e-5,weight_decay=0.,fused=False)
        metric=LearnedPerceptualImagePatchSimilarity(net_type='alex',normalize=True).to(device).eval().requires_grad_(False)
        saved,_=latest(value);fresh=saved is None
        if fresh:
            record=value['parent_checkpoint'];need(c.sha(PARENT/record['file'])==PARENT_SHA,'parent state changed')
            saved=torch.load(PARENT/record['file'],map_location='cpu',weights_only=True)
            old.state_io.validate_state(saved,record,value['parent_receipt_sha256'],world=8)
        starts=[None]*WORLD;dist.all_gather_object(starts,saved['step']);need(len(set(starts))==1,'different starts')
        state_io.restore(model,optimizer,saved,rank,device)
        if fresh:
            random.seed(RECIPE['migration_seed']+rank);torch.manual_seed(RECIPE['migration_seed']+rank)
        start=saved['step'];history=saved['history'];del saved
        receipt_hash=c.sha(OUT/'receipt.json');begun=time.monotonic()
        print(f'RAIN_FULLSTACK_RESTORED rank={rank} step={start} Adam_restored=true world_migration_not_bitwise={fresh}',flush=True)
        def save(step):
            rngs=[None]*WORLD;dist.all_gather_object(rngs,state_io.rng_state(device))
            if step in (800,801,900,1000):
                hashes=[None]*WORLD;dist.all_gather_object(hashes,state_io.parameter_hash(state_io.trainable_state(model)))
                need(len(set(hashes))==1,'rank parameters differ')
            if rank==0:
                state_io.save_step(OUT,model,optimizer,step=step,world=WORLD,rank_rng=rngs,
                                   receipt_hash=receipt_hash,history=history)
                print(f'RAIN_FULLSTACK_CHECKPOINT step={step}',flush=True)
            dist.barrier()
        def evaluate(step):
            gc.collect();torch.cuda.empty_cache();probe(model,ae,prompt,metric,value,receipt,cache,step,rank,device)
            dist.barrier()
            if rank==0:milestone(step)
            dist.barrier()
        if fresh:save(800)
        if start in (800,900,1000):evaluate(start)
        for step in range(start,1000):
            need(time.monotonic()-begun<RECIPE['max_training_wall_seconds_per_attempt'],'bounded training wall cap; preserve checkpoint')
            step_start=time.monotonic();optimizer.zero_grad(set_to_none=True)
            totals={'total':0.,'gt_rgb_mse':0.,'gt_lpips':0.};prefix_seconds=0.
            for micro,position in enumerate(batch_logic.positions(step,rank,WORLD)):
                cache_index=value['training_cache_indices'][position-25600]
                data=cached(value,receipt,cache,cache_index,device,training=True)
                seed=receipt['config']['seed']+1000003*(cache_index+1);model.eval();t=time.monotonic()
                inputs=logic.generated_inputs(model,ae,data['lq'],prompt,seed);torch.cuda.synchronize(device)
                prefix_seconds+=time.monotonic()-t;model.train()
                with parent.hopper.offload_activations():
                    loss,terms=logic.objective(model,ae,inputs,data['gt'],metric,RECIPE['loss_weights'])
                    need(torch.isfinite(loss),'nonfinite objective');(loss/32).backward()
                metric.reset()
                for k,v in {'total':loss,**terms}.items():totals[k]+=float(v.detach())/32
                del data,inputs,loss,terms
                if step==800:print(f'RAIN_FULLSTACK_FIRST_MICRO rank={rank} sample={micro+1}',flush=True)
            batch_logic.sum_gradients(params)
            norm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
            need(float(norm)>0,'zero global gradient')
            optimizer.step();assert_fp32_optimizer(optimizer);optimizer.zero_grad(set_to_none=True)
            need(all(torch.isfinite(p).all() for p in params),'nonfinite LoRA')
            v=torch.tensor(list(totals.values()),dtype=torch.float64,device=device);dist.all_reduce(v)
            stats=torch.tensor([float(norm),time.monotonic()-step_start,prefix_seconds,
                                torch.cuda.max_memory_allocated(device)/1024**3],device=device)
            dist.all_reduce(stats,op=dist.ReduceOp.MAX)
            log={'step':step+1,'new_updates':step+1-800,'losses':dict(zip(totals,v.tolist())),
                 'max_rank_grad_norm':float(stats[0]),'max_rank_step_seconds_excluding_checkpoint':float(stats[1]),
                 'max_rank_prefix_seconds':float(stats[2]),'peak_allocated_gib':float(stats[3]),
                 'next_global_slot':32*(step+1),'world':6,'global_batch':32,
                 'last_selected_cache_index':value['training_cache_indices'][32*(step+1)-25601],
                 'objective':'current_policy_prefix25_detached_last_call_GT_RGB_MSE_LPIPS'}
            history.append(log)
            if rank==0:print('RAIN_FULLSTACK_TRAIN '+json.dumps(log),flush=True)
            save(step+1)
            if step+1 in (900,1000):evaluate(step+1)
        if rank==0:
            final=state_io.trainable_state(model);name,digest=parent.replay.save_tensors(OUT,'rain_fullstack_step1000',final)
            c.write_new(OUT/'final_lora.json',{'step':1000,'tensor':name,'files':{name:digest},
                        'receipt_hash':receipt_hash,'parameter_hash':state_io.parameter_hash(final)})
        dist.barrier()
        c.write_new(OUT/f'worker{rank}.json',{'rank':rank,'receipt_sha256':receipt_hash,'step':1000,
                    'seconds':time.monotonic()-begun,'peak_allocated_gib':torch.cuda.max_memory_allocated(device)/1024**3})
        print(f'RAIN_FULLSTACK_WORKER_COMPLETE rank={rank}',flush=True)
    finally:dist.destroy_process_group()


def finish():
    value,receipt,cache=bound();saved,record=latest(value)
    need(saved is not None and saved['step']==1000,'training incomplete')
    for rank in range(WORLD):
        m=c.read(OUT/f'worker{rank}.json');need(m['step']==1000 and m['receipt_sha256']==c.sha(OUT/'receipt.json'),'worker incomplete')
    final=c.read(OUT/'final_lora.json');need(final['parameter_hash']==saved['parameter_hash'],'export mismatch')
    for name,digest in final['files'].items():need(c.sha(OUT/name)==digest,'export hash changed')
    milestones={str(s):milestone(s) for s in (800,900,1000)}
    result={'status':'bounded_training_complete_official_Rain_not_yet_evaluated','last_step':1000,'new_updates':200,
            'new_presentations':6400,'receipt_sha256':c.sha(OUT/'receipt.json'),'checkpoint':record,'final_lora':final,
            'milestones':milestones,'history_new_updates':saved['history'][800:],
            'world_migration_bitwise_equivalent':False,'model_and_Adam_preserved_at_migration':True,
            'official_pixels_used_for_training':False,'development_pixels_read':False,
            'legacy_checkpoint_has_confirmed_official_GT_source_overlap':True,
            'new_training_excludes_all_detected_official_source_groups':True,
            'training_fit_only':True,'automatic_more_training':False}
    c.write_new(REPORT,result);print('RAIN_FULLSTACK_TRAIN_RESULT_VERIFIED',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','run','finish']);p.add_argument('--node',type=int,default=0)
    a=p.parse_args()
    if a.phase=='run':run(a.node)
    else:globals()[a.phase]()
