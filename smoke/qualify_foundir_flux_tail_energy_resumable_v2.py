#!/usr/bin/env python3
"""Same scientific qualification, eight GPUs by default, reusable cache + durable steps."""
from __future__ import annotations
import argparse
from contextlib import nullcontext
import copy
import json
import os
from pathlib import Path
import re
import signal
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from smoke import flux_tail_energy_contract_v1 as parent
from smoke import qualify_foundir_flux_tail_energy_v1 as core
from smoke import flux_tail_checkpoint_v2 as saves
from smoke import flux_energy_qualification_contract_v1 as common
from smoke import diagnose_foundir_flux_tail_oracle_v1 as base

CONFIG=ROOT/'configs/foundir_flux_tail_energy_resumable_v2.json'
EXECUTION_FILES=[str(CONFIG.relative_to(ROOT)), 'smoke/flux_tail_checkpoint_v2.py',
    'smoke/qualify_foundir_flux_tail_energy_resumable_v2.py',
    'smoke/manage_flux_tail_energy_jobs_v2.py',
    'smoke/run_foundir_flux_tail_energy_resumable_v2_gpu.sh',
    'smoke/start_foundir_flux_tail_energy_resumable_v2_tmux.sh']


def config():
    spec=json.loads(CONFIG.read_text())
    if (spec['training_steps']!=25 or spec['checkpoint_every_optimizer_step']!=1
            or spec['allowed_gpu_counts']!=[4,8] or spec['default_gpu_count']!=8
            or spec['auto_formal'] or spec['qualification_may_seed_formal'] or spec['auto_restart']
            or spec['budget_wall_seconds_by_gpus']!={'4':10800,'8':5400}):
        raise RuntimeError('execution scope changed')
    return spec


def binding():
    return parent.binding() | {name:common.sha(ROOT/name) for name in EXECUTION_FILES}


def import_cache_metadata(meta, source_hashes):
    result=copy.deepcopy(meta)
    result['cache_producer_source_sha256']=result['source_sha256']
    result['source_sha256']=source_hashes
    result['cache_reused_without_regeneration']=True
    return result


def prepare(output,world):
    execution=config(); parent.assigned([],0,world)
    cache_root=common.path(execution['cache_root'])
    common.bound(str(cache_root/'preflight.json'),execution['cache_preflight_sha256'])
    common.bound(str(cache_root/'cache_complete.json'),execution['cache_complete_sha256'])
    producer=parent.load(cache_root,4)
    done=json.loads((cache_root/'cache_complete.json').read_text())
    if (done['records']!=32 or done['train_sources']!=24 or done['evaluation_sources']!=8
            or not done['baseline_parity_pass'] or done['source_sha256']!=producer['source_sha256']
            or done['config_sha256']!=producer['config_sha256']):
        raise RuntimeError('parent cache incomplete/inconsistent')
    if (cache_root/'energy_step_000025.pt').exists():
        raise RuntimeError('legacy training checkpoint now exists; preserve it and review migration, not fresh replay')
    sources=binding()
    receipt=copy.deepcopy(producer)
    receipt.update(world_size=world,source_sha256=sources,execution_spec=execution,
        execution_config_sha256=common.sha(CONFIG),cache_producer_world_size=4,
        cache_producer_preflight_sha256=execution['cache_preflight_sha256'],
        legacy_unsaved_steps_not_resumed=True,cache_regenerated=False)
    if (output/'preflight.json').exists():
        if json.loads((output/'preflight.json').read_text())!=receipt:
            raise RuntimeError('existing resumable run differs; do not overwrite')
    output.mkdir(parents=True,exist_ok=True)
    for i in range(32):
        role=producer['selected'][i]['role']
        meta=parent.cache_meta(cache_root,producer,i,role)
        destination=output/meta['file']; source=cache_root/meta['file']
        if destination.exists() or destination.is_symlink():
            if not destination.is_symlink() or destination.resolve()!=source.resolve():
                raise RuntimeError('cache alias conflict')
        else: destination.symlink_to(source)
        imported=import_cache_metadata(meta,sources)
        imported['cache_producer_metadata_sha256']=common.sha(cache_root/f'cache_{i:03d}.json')
        meta_path=output/f'cache_{i:03d}.json'
        if meta_path.exists():
            if json.loads(meta_path.read_text())!=imported: raise RuntimeError('cache metadata conflict')
        else: saves.atomic_json(meta_path,imported)
    imported_done=done | dict(source_sha256=sources,cache_regenerated=False,
                             cache_producer_complete_sha256=execution['cache_complete_sha256'])
    for name,value in [('cache_complete.json',imported_done),('preflight.json',receipt)]:
        destination=output/name
        if destination.exists():
            if json.loads(destination.read_text())!=value: raise RuntimeError('receipt conflict: '+name)
        else: saves.atomic_json(destination,value)
    print('TAIL_ENERGY_V2_PREPARED '+json.dumps(dict(gpus=world,cache_records_reused=32,
        cache_regenerated=False,legacy_memory_checkpoint_available=False,training_steps=25)),flush=True)
    return receipt


def load(output,world):
    receipt=json.loads((output/'preflight.json').read_text())
    parent.validate_spec(receipt['spec'])
    if (receipt['world_size']!=world or receipt['source_sha256']!=binding()
            or receipt['execution_spec']!=config() or receipt['execution_config_sha256']!=common.sha(CONFIG)):
        raise RuntimeError('run code/config/world changed; no silent migration')
    return receipt


def train(output,receipt):
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from lowlight_restoration.flux_tail_learned_energy import TailQualityEnergy,quality_targets,learning_loss
    world=receipt['world_size'];rank,device=base.setup_ddp(world);spec=receipt['spec']
    stop=[False]
    def request_stop(*_): stop[0]=True
    handlers={s:signal.signal(s,request_stop) for s in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP)}
    try:
        metadata=[parent.cache_meta(output,receipt,i,'train') for i in range(24)]
        print(f'TAIL_ENERGY_V2_TRAIN_READY rank={rank} cache_verified=true',flush=True)
        torch.manual_seed(spec['seed']);torch.cuda.manual_seed_all(spec['seed'])
        model=TailQualityEnergy(**spec['energy']).to(device)
        wrapped=DDP(model,device_ids=[device.index],broadcast_buffers=False,gradient_as_bucket_view=False)
        optimizer=torch.optim.AdamW(model.parameters(),lr=spec['learning_rate'],weight_decay=spec['weight_decay'])
        perceptual=core.perceptual_model(device)
        state,record=saves.load_latest(output,receipt,world)
        log=[];initial_hash=core.parameter_hash(model);elapsed_before=0.;peak_before=[0.]*world;first=1
        if state is not None:
            model.load_state_dict(state['model'],strict=True);optimizer.load_state_dict(state['optimizer'])
            if core.parameter_hash(model)!=state['parameter_sha256']: raise RuntimeError('restore mismatch')
            log=state['training_log'];first=state['step']+1;initial_hash=state['initial_parameter_sha256']
            elapsed_before=state['elapsed_training_seconds'];peak_before=state['peak_memory_gib']
            saves.restore_rng(state['rank_rng'][rank],device)
            if rank==0: print(f'TAIL_ENERGY_V2_RESUME step={first-1} optimizer_restored=true next_global_slot={(first-1)*32}',flush=True)
        started=time.monotonic();torch.cuda.reset_peak_memory_stats(device)
        def checkpoint(step):
            local=dict(rng=saves.rng_state(device),parameter_sha256=core.parameter_hash(model),
                       seconds=elapsed_before+time.monotonic()-started,
                       peak=max(peak_before[rank],torch.cuda.max_memory_allocated(device)/1024**3))
            parts=[None]*world;dist.all_gather_object(parts,local)
            if len({v['parameter_sha256'] for v in parts})!=1: raise RuntimeError('DDP parameters diverged')
            record=None
            if rank==0:
                record=saves.commit_step(output,model,optimizer,step=step,world=world,
                    rank_rng=[p['rng'] for p in parts],receipt=receipt,log=log,initial_hash=initial_hash,
                    elapsed_seconds=max(p['seconds'] for p in parts),peak_memory_gib=[p['peak'] for p in parts])
                print('TAIL_ENERGY_V2_CHECKPOINT '+json.dumps(record),flush=True)
            records=[record];dist.broadcast_object_list(records,src=0)
            return records[0]
        if state is None: record=checkpoint(0)
        for step in range(first,26):
            tick=time.monotonic();optimizer.zero_grad(set_to_none=True)
            lr=spec['learning_rate']*min(step/spec['warmup_steps'],1.)
            for group in optimizer.param_groups:group['lr']=lr
            slots=parent.training_slots(step,rank,world);stats={}
            for micro,slot in enumerate(slots):
                x,p,target,candidate,condition,timestep=core.load_pair(output,receipt,metadata,slot,device)
                labels=quality_targets(candidate,target,perceptual)
                with wrapped.no_sync() if micro<len(slots)-1 else nullcontext():
                    prediction=wrapped(x,p,candidate,condition,timestep)
                    loss,diagnostics=learning_loss(prediction,labels,**spec['loss'])
                    if not bool(loss.isfinite()):raise RuntimeError('nonfinite loss')
                    (loss/len(slots)).backward()
                for key,value in {'loss':loss,**diagnostics}.items():
                    stats[key]=stats.get(key,0.)+float(value.detach())/len(slots)
                del prediction,labels,loss,diagnostics
                if rank==0:print(f'TAIL_ENERGY_V2_MICRO step={step} completed={micro+1}/{len(slots)}',flush=True)
            if any(p.grad is None or not bool(p.grad.isfinite().all()) for p in model.parameters()):
                raise RuntimeError('missing/nonfinite second-order parameter gradient')
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),spec['gradient_clip'],error_if_nonfinite=True)
            optimizer.step();torch.cuda.synchronize()
            names=sorted(stats);values=torch.tensor([stats[k] for k in names],device=device,dtype=torch.float64)
            dist.all_reduce(values);values/=world
            log.append(dict(step=step,lr=lr,rank0_seconds=time.monotonic()-tick,rank0_grad_norm=float(norm),
                            global_mean_stats=dict(zip(names,values.tolist(),strict=True))))
            record=checkpoint(step)
            if rank==0:print('TAIL_ENERGY_V2_TRAIN '+json.dumps(log[-1]|dict(checkpoint_saved=True)),flush=True)
            flag=torch.tensor(int(stop[0] or (output/'STOP_AFTER_STEP').exists()),device=device)
            dist.all_reduce(flag,op=dist.ReduceOp.MAX)
            if int(flag) and step<25:
                if rank==0:print(f'TAIL_ENERGY_V2_SAFE_STOP saved_step={step}',flush=True)
                return False
        if rank==0:
            state,record=saves.load_latest(output,receipt,world)
            if state['step']!=25:raise RuntimeError('final checkpoint absent')
            final=output/'energy_step_000025.pt'
            if not final.exists():os.link(output/record['file'],final);saves.sync_directory(output)
            if common.sha(final)!=record['sha256']:raise RuntimeError('final checkpoint mismatch')
            if not (output/'training_complete.json').exists():
                saves.atomic_json(output/'training_complete.json',dict(steps=25,checkpoint_sha256=record['sha256'],
                    parameter_count=sum(p.numel() for p in model.parameters()),
                    rank_parameter_sha256=[state['parameter_sha256']]*world,
                    initial_parameter_sha256=initial_hash,peak_memory_gib=state['peak_memory_gib'],
                    rank0_mean_step_seconds=sum(r['rank0_seconds'] for r in log)/25,
                    seconds=state['elapsed_training_seconds'],training_log=log,may_seed_formal=False,
                    config_sha256=receipt['config_sha256'],source_sha256=receipt['source_sha256'],
                    checkpoints_saved_each_step=True,legacy_unsaved_steps_replayed=True))
            print('TAIL_ENERGY_V2_TRAIN_COMPLETE steps=25 checkpoint_verified=true',flush=True)
        dist.barrier();return True
    finally:
        for s,h in handlers.items():signal.signal(s,h)
        dist.destroy_process_group()


def evaluate(output,receipt):
    import torch
    import torch.distributed as dist
    from safetensors import safe_open
    from safetensors.torch import save_file
    from lowlight_restoration.flux_tail_learned_energy import TailQualityEnergy
    from smoke.flux_tail_energy_runtime_v1 import FrozenFlow,learned_guidance,measured_metrics
    world=receipt['world_size'];rank,device=base.setup_ddp(world);started=time.monotonic()
    try:
        training=json.loads((output/'training_complete.json').read_text())
        checkpoint=common.bound(str(output/'energy_step_000025.pt'),training['checkpoint_sha256'])
        state=torch.load(checkpoint,map_location='cpu',weights_only=True)
        if state['source_sha256']!=receipt['source_sha256'] or state['config_sha256']!=receipt['config_sha256']:
            raise RuntimeError('evaluation checkpoint identity mismatch')
        energy=TailQualityEnergy(**receipt['spec']['energy']).to(device)
        energy.load_state_dict(state['model'],strict=True);energy.eval().requires_grad_(False)
        if core.parameter_hash(energy)!=training['rank_parameter_sha256'][0]:raise RuntimeError('wrong evaluation weights')
        if rank==0:
            for name,sha in receipt['artifacts'].items():common.bound(name,sha)
        dist.barrier()
        pending=[]
        for index in parent.assigned(range(24,32),rank,world):
            result_file=output/f'evaluation_{index:03d}.json'
            if result_file.exists():
                row=json.loads(result_file.read_text())
                if (row['identity']!=receipt['selected'][index] or row['rank']!=rank
                        or row['source_sha256']!=receipt['source_sha256'] or row['gt_used_for_guidance']
                        or not row['gt_loaded_after_guidance']):raise RuntimeError('bad completed evaluation')
                common.bound(str(output/row['outputs_file']),row['outputs_sha256'])
                print(f'TAIL_ENERGY_V2_EVAL_REUSED index={index}',flush=True)
            else:pending.append(index)
        if pending:flow=FrozenFlow(receipt,device);perceptual=core.perceptual_model(device)
        for index in pending:
            tick=time.monotonic();before=dict(flow.calls);torch.cuda.reset_peak_memory_stats(device)
            meta=parent.cache_meta(output,receipt,index,'engineering_evaluation')
            with safe_open(str(output/meta['file']),framework='pt',device='cpu') as h:
                observation=h.get_tensor('observation');proposal=h.get_tensor('proposal')
            session=flow.session(observation,meta['identity']['seed'],expected_proposal=proposal)
            condition=torch.tensor([common.descriptor(meta['identity']['row'])],device=device)
            guidance=learned_guidance(session,energy,condition,receipt['spec']);flow.assert_frozen()
            if any(p.grad is not None or p.requires_grad for p in energy.parameters()):raise RuntimeError('energy not frozen')
            with safe_open(str(output/meta['file']),framework='pt',device='cpu') as h:
                target=h.get_tensor('target')[None].to(device).float()/127.5-1
            images={'historical_bf16':proposal[None],'fp32_control':guidance['control_uint8']}
            images.update({f'energy_step{k}':v for k,v in guidance['milestones'].items()})
            metrics={k:measured_metrics(v.to(device).float()/127.5-1,target,perceptual) for k,v in images.items()}
            filename=output/f'outputs_{index:03d}_{time.time_ns()}.safetensors'
            # Unique name: interrupted outputs remain preserved, not overwritten.
            save_file({k:v.contiguous().clone() for k,v in images.items()},str(filename))
            with filename.open('rb') as h:os.fsync(h.fileno())
            row=dict(index=index,identity=meta['identity'],rank=rank,metrics=metrics,
                predicted_baseline=guidance['predicted_baseline'],trajectory=guidance['history'],
                accepted_updates=guidance['accepted_updates'],stop_reason=guidance['stop_reason'],
                gt_used_for_guidance=False,gt_loaded_after_guidance=True,baseline_parity_pass=True,
                outputs_file=filename.name,outputs_sha256=common.sha(filename),
                seconds=time.monotonic()-tick,peak_memory_gib=torch.cuda.max_memory_allocated(device)/1024**3,
                calls={k:flow.calls[k]-before[k] for k in flow.calls},config_sha256=receipt['config_sha256'],
                source_sha256=receipt['source_sha256'])
            saves.atomic_json(output/f'evaluation_{index:03d}.json',row)
            print('TAIL_ENERGY_V2_EVAL '+json.dumps(dict(index=index,metrics=metrics,accepted_updates=row['accepted_updates'])),flush=True)
        dist.barrier()
        if rank==0:
            rows=[json.loads((output/f'evaluation_{i:03d}.json').read_text()) for i in range(24,32)]
            result=core.aggregate(rows,receipt)
            result.update(checkpoint_sha256=training['checkpoint_sha256'],training=training,
                evaluation_seconds_sum_records=sum(r['seconds'] for r in rows),
                latest_evaluation_attempt_seconds=time.monotonic()-started,
                cache_reused=True,execution_config_sha256=receipt['execution_config_sha256'])
            if (output/'result.json').exists():
                prior=json.loads((output/'result.json').read_text())
                if prior['rows']!=rows or prior['means']!=result['means']:raise RuntimeError('existing result differs')
            else:saves.atomic_json(output/'result.json',result)
        dist.barrier()
    finally:dist.destroy_process_group()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=['prepare','hardware','train','evaluate','verify'])
    p.add_argument('--run-id',default='qualification_8gpu_v2');p.add_argument('--gpus',type=int,choices=[4,8],default=8)
    a=p.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+',a.run_id):raise ValueError('bad run ID')
    output=common.path(config()['output_root'])/a.run_id
    if a.phase=='hardware':base.hardware(a.gpus)
    elif a.phase=='prepare':prepare(output,a.gpus)
    else:
        receipt=load(output,a.gpus)
        if a.phase=='train':
            if not train(output,receipt):raise SystemExit(75)
        elif a.phase=='evaluate':evaluate(output,receipt)
        else:
            core.verify(output,receipt)
            state,_=saves.load_latest(output,receipt,a.gpus)
            if state is None or state['step']!=25:raise RuntimeError('missing durable terminal checkpoint')
            print('TAIL_ENERGY_V2_RESULT_VERIFIED',flush=True)


if __name__=='__main__':main()
