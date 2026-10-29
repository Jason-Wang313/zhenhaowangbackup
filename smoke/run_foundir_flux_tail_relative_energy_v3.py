#!/usr/bin/env python3
"""Train-only learning repair: relative quality, self-mined negatives, one-step probe.

No calibration/evaluation images are read. Generalization/calibration is a later
stage, after this training-fit question is answered, not an implied paper result.
"""
from __future__ import annotations
import argparse
from contextlib import nullcontext
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from smoke import flux_energy_qualification_contract_v1 as common
from smoke import flux_tail_energy_contract_v1 as old_contract
from smoke import qualify_foundir_flux_tail_energy_resumable_v2 as parent
from smoke import qualify_foundir_flux_tail_energy_v1 as core
from smoke import flux_tail_checkpoint_v2 as atomic
from smoke import diagnose_foundir_flux_tail_oracle_v1 as base

CONFIG=ROOT/'configs/foundir_flux_tail_relative_energy_v3.json'
FILES=['configs/foundir_flux_tail_relative_energy_v3.json',
    'lowlight_restoration/flux_tail_relative_energy_v3.py',
    'smoke/flux_tail_relative_checkpoint_v3.py',
    'smoke/run_foundir_flux_tail_relative_energy_v3.py',
    'smoke/run_foundir_flux_tail_relative_energy_v3_gpu.sh',
    'smoke/start_foundir_flux_tail_relative_energy_v3_tmux.sh']


def spec():
    value=json.loads(CONFIG.read_text())
    if (value['schema']!=3 or value['train_indices']!=list(range(24))
        or value['probe_indices']!=list(range(8)) or value['crop']!=512
        or value['global_batch_pairs']!=32 or value['default_gpu_count']!=8
        or value['allowed_gpu_counts']!=[4,8] or value['mining_updates']!=4
        or value['probe_updates']!=1 or value['auto_formal'] or value['auto_trainfit']
        or value['read_calibration_or_evaluation_pixels'] or value['extra_restoration_network']
        or value['qualification_may_seed_trainfit']
        or value['modes']['qualification']['steps']!=8
        or value['modes']['trainfit']['steps']!=200):
        raise RuntimeError('scope changed')
    return value


def binding():
    return parent.binding()|{p:common.sha(ROOT/p) for p in FILES}


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def immutable_json(path,value):
    if path.exists():
        if json.loads(path.read_text())!=value:raise RuntimeError('immutable metadata conflict: '+str(path))
    else:atomic.atomic_json(path,value)


def prepare(output,mode,world,qualification_id='qualification_v1'):
    s=spec();old_contract.assigned([],0,world)
    root=common.path(s['parent_run'])
    common.bound(str(root/'result.json'),s['parent_result_sha256'])
    common.bound(str(root/'energy_step_000025.pt'),s['negative_generator_sha256'])
    old=parent.load(root,8)
    rows=old['selected'][:24]
    if len(rows)!=24 or any(v['role']!='train' or v['row']['split']!='train'
                            or v['row']['category']!='05Noise' for v in rows):
        raise RuntimeError('only original twenty-four training sources are allowed')
    if len({v['row']['source_group'] for v in rows})!=24:raise RuntimeError('source duplicates')
    for i in range(24):old_contract.cache_meta(root,old,i,'train')
    sources=binding()
    cache=common.path(s['output_root'])/'cache_v1';cache.mkdir(parents=True,exist_ok=True)
    cache_contract=dict(spec=s,source_sha256=sources,parent_preflight_sha256=common.sha(root/'preflight.json'),
                        identities=rows,protected_pixels_accessed=False,evaluation_pixels_accessed=False)
    immutable_json(cache/'contract.json',cache_contract)
    qualification_result_sha256=None
    if mode=='trainfit':
        if not re.fullmatch(r'[A-Za-z0-9_-]+',qualification_id):raise ValueError('invalid qualification run ID')
        qualification=common.path(s['output_root'])/qualification_id/'result.json'
        if not qualification.exists():raise RuntimeError('qualification result required before train-fit')
        q=json.loads(qualification.read_text())
        if not q['engineering_pass'] or q['source_sha256']!=sources or q['mode']!='qualification':
            raise RuntimeError('qualification binding/pass mismatch')
        qualification_result_sha256=common.sha(qualification)
    receipt=dict(spec=s,mode=mode,world_size=world,source_sha256=sources,
        parent_preflight_sha256=cache_contract['parent_preflight_sha256'],cache_contract_sha256=common.sha(cache/'contract.json'),
        cache_root=str(cache.relative_to(ROOT)),selected=rows,
        qualification_result_sha256=qualification_result_sha256,
        qualification_checkpoint_loaded=False,protected_pixels_accessed=False,evaluation_pixels_accessed=False)
    receipt['protocol_sha256']=digest(receipt)
    output.mkdir(parents=True,exist_ok=True);immutable_json(output/'preflight.json',receipt)
    print('TAIL_RELATIVE_V3_PREPARED '+json.dumps(dict(mode=mode,gpus=world,train_sources=24,
        steps=s['modes'][mode]['steps'],evaluation_pixels_accessed=False)),flush=True)
    return receipt


def load_receipt(output,world):
    r=json.loads((output/'preflight.json').read_text())
    check=dict(r);check.pop('protocol_sha256')
    if (r['spec']!=spec() or r['source_sha256']!=binding() or r['world_size']!=world
        or r['protocol_sha256']!=digest(check)):
        raise RuntimeError('run source/config/world changed')
    common.bound(str(common.path(r['cache_root'])/'contract.json'),r['cache_contract_sha256'])
    return r


def old_receipt(r):
    root=common.path(r['spec']['parent_run'])
    common.bound(str(root/'preflight.json'),r['parent_preflight_sha256'])
    return json.loads((root/'preflight.json').read_text())


def tensor_file(path,tensors):
    from safetensors.torch import save
    payload=save({k:v.detach().cpu().contiguous().clone() for k,v in tensors.items()})
    atomic.atomic_write(path,lambda h:h.write(payload))
    return common.sha(path)


def verify_meta(directory,index,receipt):
    meta=json.loads((directory/f'record_{index:03d}.json').read_text())
    if (meta['identity']!=receipt['selected'][index] or meta['identity']['role']!='train'
        or meta['source_sha256']!=receipt['source_sha256']):raise RuntimeError('training cache identity changed')
    common.bound(str(directory/meta['file']),meta['sha256'])
    return meta


def labeled_frames(candidates,target,perceptual,device):
    import torch
    from lowlight_restoration.flux_tail_learned_energy import quality_targets
    scores,gm,gp=[],[],[]
    for candidate in candidates:
        labels,grads=quality_targets(candidate[None].to(device),target,perceptual)
        if not bool(candidate.isfinite().all() and labels.isfinite().all()
                    and grads[0].isfinite().all() and grads[1].isfinite().all()):
            raise RuntimeError('nonfinite candidate/score/target derivative')
        scores.append(labels[0].cpu());gm.append(grads[0][0].cpu());gp.append(grads[1][0].cpu())
    return dict(candidates=candidates.cpu(),scores=torch.stack(scores),
                gm=torch.stack(gm),gp=torch.stack(gp))


def cache_base(output,r):
    import torch
    import torch.distributed as dist
    from safetensors import safe_open
    from lowlight_restoration.flux_tail_relative_energy_v3 import edit_stats,trust_limits
    world=r['world_size'];rank,device=base.setup_ddp(world)
    directory=common.path(r['cache_root'])/'base';directory.mkdir(exist_ok=True)
    try:
        previous=old_receipt(r);source=common.path(r['spec']['parent_run'])
        perceptual=core.perceptual_model(device)
        for i in old_contract.assigned(range(24),rank,world):
            if (directory/f'record_{i:03d}.json').exists():
                verify_meta(directory,i,r);continue
            meta=old_contract.cache_meta(source,previous,i,'train')
            with safe_open(str(source/meta['file']),framework='pt',device='cpu') as h:
                x,p,gt=[h.get_tensor(k) for k in ('observation','proposal','target')]
                continuous=h.get_tensor('frames_continuous').float()
                quantized=h.get_tensor('frames_quantized').float()/127.5-1
            candidates=torch.stack((continuous,quantized),1).flatten(0,1)
            labeled=labeled_frames(candidates,gt[None].to(device).float()/127.5-1,perceptual,device)
            labeled.update(observation=x,proposal=p,target=gt)
            file=f'record_{i:03d}_{time.time_ns()}.safetensors';sha=tensor_file(directory/file,labeled)
            kinds=meta['frame_kinds'];proposal_index=kinds.index('historical_proposal')*2+1
            teachers=[j*2+1 for j,k in enumerate(kinds) if k.startswith('teacher_step')]
            first=kinds.index('teacher_step1')*2+1 if 'teacher_step1' in kinds else None
            edit=None if first is None else edit_stats(candidates[first:first+1],p[None].float()/127.5-1)
            info=dict(identity=r['selected'][i],file=file,sha256=sha,frames=len(candidates),
                proposal_index=proposal_index,teacher_indices=teachers,first_teacher_edit=edit,
                timestep=meta['timestep'],source_sha256=r['source_sha256'])
            atomic.atomic_json(directory/f'record_{i:03d}.json',info)
            print(f'TAIL_RELATIVE_V3_BASE_CACHE index={i} frames={len(candidates)}',flush=True)
        dist.barrier()
        if rank==0:
            metadata=[verify_meta(directory,i,r) for i in range(24)]
            limits=trust_limits([m['first_teacher_edit'] for m in metadata if m['first_teacher_edit'] is not None],r['spec'])
            immutable_json(directory/'complete.json',dict(records=24,frames=sum(m['frames'] for m in metadata),
                record_metadata_sha256=[common.sha(directory/f'record_{i:03d}.json') for i in range(24)],
                trust_limits=limits,trust_is_calibrated=False,source_sha256=r['source_sha256']))
            print('TAIL_RELATIVE_V3_BASE_COMPLETE '+json.dumps(dict(records=24,limits=limits)),flush=True)
        dist.barrier()
    finally:dist.destroy_process_group()


def mining_directory(output,r,generation):
    return (common.path(r['cache_root'])/'bootstrap' if generation==0
            else output/'mining'/f'step_{generation:06d}')


def mine(output,r,generation):
    import torch
    import torch.distributed as dist
    from safetensors import safe_open
    from lowlight_restoration.flux_tail_learned_energy import TailQualityEnergy
    from smoke.flux_tail_energy_runtime_v1 import FrozenFlow,learned_guidance
    from smoke import flux_tail_relative_checkpoint_v3 as checkpoints
    rank,device=base.setup_ddp(r['world_size']);directory=mining_directory(output,r,generation)
    directory.mkdir(parents=True,exist_ok=True)
    try:
        previous=old_receipt(r)
        if generation==0:
            checkpoint=common.bound(str(common.path(r['spec']['parent_run'])/'energy_step_000025.pt'),
                                    r['spec']['negative_generator_sha256'])
            state=torch.load(checkpoint,map_location='cpu',weights_only=True);model_sha=common.sha(checkpoint)
        else:
            state,record=checkpoints.load(output,r,generation)
            if state is None or state['step']!=generation:raise RuntimeError('mining requires exact current checkpoint')
            model_sha=record['sha256']
        energy=TailQualityEnergy(**r['spec']['energy']).to(device)
        energy.load_state_dict(state['model'],strict=True);energy.eval().requires_grad_(False)
        pending=[]
        for i in old_contract.assigned(range(24),rank,r['world_size']):
            if (directory/f'record_{i:03d}.json').exists():
                m=verify_meta(directory,i,r)
                if m['generator_sha256']!=model_sha:raise RuntimeError('mining generator changed')
            else:pending.append(i)
        if pending:
            if rank==0:
                for path,sha in previous['artifacts'].items():common.bound(path,sha)
            # All ranks enter the barrier, including when their shard is cached.
        needs=torch.tensor(int(bool(pending)),device=device);dist.all_reduce(needs)
        if int(needs):
            if rank==0 and not pending:
                for path,sha in previous['artifacts'].items():common.bound(path,sha)
            dist.barrier()
        if pending:flow=FrozenFlow(previous,device);perceptual=core.perceptual_model(device)
        for i in pending:
            source=common.path(r['spec']['parent_run']);meta=old_contract.cache_meta(source,previous,i,'train')
            with safe_open(str(source/meta['file']),framework='pt',device='cpu') as h:
                x=h.get_tensor('observation');p=h.get_tensor('proposal');gt=h.get_tensor('target')
            session=flow.session(x,meta['identity']['seed'],expected_proposal=p)
            descriptor=torch.tensor([common.descriptor(meta['identity']['row'])],device=device)
            stress=copy.deepcopy(previous['spec']);stress['guidance_max_updates']=r['spec']['mining_updates']
            stress['milestones']=list(range(1,r['spec']['mining_updates']+1))
            # No GT reaches learned_guidance. Collect both continuous and uint8
            # frames from each actually selected model-generated state.
            rendered=[];original_render=session.render
            def capture(state):
                value=original_render(state);rendered.append((value[0].detach().cpu(),value[1].detach().cpu()))
                return value
            session.render=capture
            guidance=learned_guidance(session,energy,descriptor,stress)
            session.render=original_render
            selected=[];offset=1
            for row in guidance['history']:
                if row.get('chosen_trial') is not None:
                    continuous,quantized=rendered[offset+row['chosen_trial']]
                    selected.extend([continuous[0],quantized[0].float()/127.5-1])
                offset+=len(row['trials'])
            if not selected:
                continuous,quantized=rendered[0]
                selected=[continuous[0],quantized[0].float()/127.5-1]
            # GT is used only now, to LABEL training-source generated images.
            candidates=torch.stack(selected)
            labeled=labeled_frames(candidates,gt[None].to(device).float()/127.5-1,perceptual,device)
            file=f'record_{i:03d}_{time.time_ns()}.safetensors';sha=tensor_file(directory/file,labeled)
            atomic.atomic_json(directory/f'record_{i:03d}.json',dict(identity=r['selected'][i],file=file,
                sha256=sha,frames=len(candidates),generator_sha256=model_sha,generation=generation,
                generated_without_gt=True,labels_use_training_gt=True,accepted=guidance['accepted_updates'],
                source_sha256=r['source_sha256']))
            flow.assert_frozen()
            print(f'TAIL_RELATIVE_V3_MINED generation={generation} index={i} frames={len(candidates)}',flush=True)
            del session,rendered,candidates,labeled,guidance
        dist.barrier()
        if rank==0:
            rows=[verify_meta(directory,i,r) for i in range(24)]
            immutable_json(directory/'complete.json',dict(records=24,generation=generation,
                generator_sha256=model_sha,record_metadata_sha256=[common.sha(directory/f'record_{i:03d}.json') for i in range(24)],
                source_sha256=r['source_sha256'],frames=sum(v['frames'] for v in rows)))
            print(f'TAIL_RELATIVE_V3_MINING_COMPLETE generation={generation}',flush=True)
        dist.barrier()
    finally:dist.destroy_process_group()


def training_metadata(output,r,generation):
    root=common.path(r['cache_root'])/'base';mined=mining_directory(output,r,generation)
    for directory in (root,mined):
        complete=json.loads((directory/'complete.json').read_text())
        if complete['records']!=24 or complete['source_sha256']!=r['source_sha256']:
            raise RuntimeError('training candidates incomplete or changed')
        for i,sha in enumerate(complete['record_metadata_sha256']):
            common.bound(str(directory/f'record_{i:03d}.json'),sha)
    return root,mined,[verify_meta(root,i,r) for i in range(24)],[verify_meta(mined,i,r) for i in range(24)]


def load_pair(r,metadata,slot,device):
    import torch
    from safetensors import safe_open
    from lowlight_restoration.flux_tail_relative_energy_v3 import pair_plan
    root,mined,bases,negatives=metadata
    index,choices=pair_plan(slot,r['spec']['modes'][r['mode']]['seed'],
        [v['frames'] for v in bases],[v['frames'] for v in negatives],
        [v['teacher_indices'] for v in bases],[v['proposal_index'] for v in bases])
    identity=r['selected'][index]
    if identity['role']!='train' or identity['row']['split']!='train':raise RuntimeError('not a training source')
    with safe_open(str(root/bases[index]['file']),framework='pt',device='cpu') as h:
        x,p=[h.get_tensor(k)[None].float()/127.5-1 for k in ('observation','proposal')]
    values={k:[] for k in ('candidates','scores','gm','gp')}
    for kind,frame in choices:
        directory,meta=(root,bases[index]) if kind=='base' else (mined,negatives[index])
        with safe_open(str(directory/meta['file']),framework='pt',device='cpu') as h:
            for key in values:values[key].append(h.get_slice(key)[frame].float())
    values={k:torch.stack(v).to(device) for k,v in values.items()}
    descriptor=torch.tensor([common.descriptor(identity['row'])]*2,device=device)
    timestep=torch.full((2,1),bases[index]['timestep'],device=device)
    return (x.repeat(2,1,1,1).to(device),p.repeat(2,1,1,1).to(device),
            values['candidates'],descriptor,timestep),(values['scores'],(values['gm'],values['gp']))


def train(output,r,until):
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from lowlight_restoration.flux_tail_learned_energy import TailQualityEnergy
    from lowlight_restoration.flux_tail_relative_energy_v3 import relative_loss,generation_for_step,parameter_gradient_audit
    from smoke import flux_tail_relative_checkpoint_v3 as checkpoints
    mode=r['spec']['modes'][r['mode']];world=r['world_size'];rank,device=base.setup_ddp(world)
    stop=[False]
    def mark_stop(*_):stop[0]=True
    handlers={s:signal.signal(s,mark_stop) for s in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP)}
    try:
        if until not in mode['refresh_steps']+[mode['steps']]:raise ValueError('undeclared training checkpoint')
        torch.manual_seed(mode['seed']);torch.cuda.manual_seed_all(mode['seed'])
        model=TailQualityEnergy(**r['spec']['energy']).to(device)
        optimizer=torch.optim.AdamW(model.parameters(),lr=r['spec']['learning_rate'],weight_decay=r['spec']['weight_decay'])
        state,record=checkpoints.load(output,r);first=1;log=[];elapsed_before=0.;peaks_before=[0.]*world
        if state is not None:
            model.load_state_dict(state['model'],strict=True);optimizer.load_state_dict(state['optimizer'])
            first=state['step']+1;log=state['training_log'];elapsed_before=state['elapsed_training_seconds']
            peaks_before=[v['peak_gib'] for v in state['rank_states']]
            atomic.restore_rng(state['rank_states'][rank]['rng'],device)
            if rank==0:print(f'TAIL_RELATIVE_V3_RESUME step={first-1} Adam_RNG_sampler_restored=true',flush=True)
        if first>until:return True
        generation=generation_for_step(first,mode['refresh_steps'])
        if generation_for_step(until,mode['refresh_steps'])!=generation:
            raise RuntimeError('must refresh self-generated candidates at declared boundary')
        metadata=training_metadata(output,r,generation)
        if rank==0:print(f'TAIL_RELATIVE_V3_TRAIN_READY world={world} generation={generation} next_step={first}',flush=True)
        audit_file=output/f'parameter_gradient_audit_before_{first:06d}.json'
        if rank==0 and not audit_file.exists():
            inputs,labels=load_pair(r,metadata,(first-1)*32,device)
            audit_loss,diagnostics,parts=relative_loss(model(*inputs),labels,r['spec']['loss'])
            norms=parameter_gradient_audit(model,parts,r['spec']['loss'])
            atomic.atomic_json(audit_file,dict(before_step=first,weighted_parameter_gradient_norms=norms,
                loss_terms={k:float(v) for k,v in diagnostics.items()},protocol_sha256=r['protocol_sha256']))
            print('TAIL_RELATIVE_V3_PARAMETER_GRADIENT_AUDIT '+json.dumps(norms),flush=True)
            del inputs,labels,parts,diagnostics,audit_loss
        dist.barrier()
        wrapped=DDP(model,device_ids=[device.index],broadcast_buffers=False,gradient_as_bucket_view=False)
        started=time.monotonic();torch.cuda.reset_peak_memory_stats(device)
        def save(step):
            local=dict(rng=atomic.rng_state(device),parameter_sha256=core.parameter_hash(model),
                peak_gib=max(peaks_before[rank],torch.cuda.max_memory_allocated(device)/1024**3),
                elapsed=elapsed_before+time.monotonic()-started)
            gathered=[None]*world;dist.all_gather_object(gathered,local)
            result=None
            if rank==0:
                result=checkpoints.commit(output,model,optimizer,step,r,gathered,log,max(v['elapsed'] for v in gathered))
                print('TAIL_RELATIVE_V3_CHECKPOINT '+json.dumps(result),flush=True)
            dist.barrier()
        if state is None:save(0)
        for step in range(first,until+1):
            tick=time.monotonic();optimizer.zero_grad(set_to_none=True)
            lr=r['spec']['learning_rate']*min(step/r['spec']['warmup_steps'],1.)
            for group in optimizer.param_groups:group['lr']=lr
            slots=old_contract.training_slots(step,rank,world);statistics={}
            for micro,slot in enumerate(slots):
                inputs,labels=load_pair(r,metadata,slot,device)
                with wrapped.no_sync() if micro<len(slots)-1 else nullcontext():
                    loss,diagnostics,parts=relative_loss(wrapped(*inputs),labels,r['spec']['loss'])
                    if not bool(loss.isfinite()):raise RuntimeError('nonfinite loss')
                    (loss/len(slots)).backward()
                for key,value in dict(loss=loss.detach(),**diagnostics).items():
                    statistics[key]=statistics.get(key,0.)+float(value)/len(slots)
                del inputs,labels,loss,diagnostics,parts
                if rank==0:print(f'TAIL_RELATIVE_V3_MICRO step={step} done={micro+1}/{len(slots)}',flush=True)
            if any(p.grad is None or not bool(p.grad.isfinite().all()) for p in model.parameters()):
                raise RuntimeError('missing/nonfinite second-order parameter gradient')
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),r['spec']['gradient_clip'],error_if_nonfinite=True)
            optimizer.step();torch.cuda.synchronize()
            names=sorted(statistics);values=torch.tensor([statistics[k] for k in names],device=device,dtype=torch.float64)
            dist.all_reduce(values);values/=world
            row=dict(step=step,lr=lr,generation=generation,rank0_seconds=time.monotonic()-tick,
                     gradient_norm=float(norm),global_mean_stats=dict(zip(names,values.tolist(),strict=True)))
            log.append(row);save(step)
            if rank==0:print('TAIL_RELATIVE_V3_TRAIN '+json.dumps(row|dict(saved=True)),flush=True)
            flag=torch.tensor(int(stop[0] or (output/'STOP_AFTER_STEP').exists()),device=device)
            dist.all_reduce(flag,op=dist.ReduceOp.MAX)
            if int(flag) and step<mode['steps']:
                if rank==0:print(f'TAIL_RELATIVE_V3_SAFE_STOP step={step}',flush=True)
                return False
        if until==mode['steps'] and rank==0:
            final,record=checkpoints.load(output,r)
            immutable_json(output/'training_complete.json',dict(steps=until,checkpoint=record,
                mean_step_seconds=sum(v['rank0_seconds'] for v in log)/len(log),
                rank_states_peak_gib=[v['peak_gib'] for v in final['rank_states']],
                protocol_sha256=r['protocol_sha256'],source_sha256=r['source_sha256'],may_seed_other_run=False))
            print(f'TAIL_RELATIVE_V3_TRAIN_COMPLETE steps={until}',flush=True)
        dist.barrier();return True
    finally:
        for s,h in handlers.items():signal.signal(s,h)
        dist.destroy_process_group()


def probe(output,r):
    import torch
    import torch.distributed as dist
    from safetensors import safe_open
    from lowlight_restoration.flux_tail_learned_energy import TailQualityEnergy
    from lowlight_restoration.flux_tail_relative_energy_v3 import bounded_one_step
    from smoke.flux_tail_energy_runtime_v1 import FrozenFlow,measured_metrics
    from smoke import flux_tail_relative_checkpoint_v3 as checkpoints
    rank,device=base.setup_ddp(r['world_size'])
    try:
        state,record=checkpoints.load(output,r)
        if state is None or state['step']!=r['spec']['modes'][r['mode']]['steps']:
            raise RuntimeError('probe requires final predeclared checkpoint')
        energy=TailQualityEnergy(**r['spec']['energy']).to(device)
        energy.load_state_dict(state['model'],strict=True);energy.eval().requires_grad_(False)
        previous=old_receipt(r);source=common.path(r['spec']['parent_run'])
        limits=json.loads((common.path(r['cache_root'])/'base'/'complete.json').read_text())['trust_limits']
        pending=[]
        for i in old_contract.assigned(r['spec']['probe_indices'],rank,r['world_size']):
            if (output/f'probe_{i:03d}.json').exists():
                row=json.loads((output/f'probe_{i:03d}.json').read_text())
                if row['checkpoint_sha256']!=record['sha256'] or row['identity']!=r['selected'][i]:
                    raise RuntimeError('prior probe identity changed')
                common.bound(str(output/row['outputs_file']),row['outputs_sha256'])
            else:pending.append(i)
        if rank==0:
            for path,sha in previous['artifacts'].items():common.bound(path,sha)
        dist.barrier()
        if pending:flow=FrozenFlow(previous,device);perceptual=core.perceptual_model(device)
        for i in pending:
            meta=old_contract.cache_meta(source,previous,i,'train')
            with safe_open(str(source/meta['file']),framework='pt',device='cpu') as h:
                x=h.get_tensor('observation');p=h.get_tensor('proposal')
            session=flow.session(x,meta['identity']['seed'],expected_proposal=p)
            condition=torch.tensor([common.descriptor(meta['identity']['row'])],device=device)
            result=bounded_one_step(session,energy,condition,previous['spec'],limits)
            flow.assert_frozen()
            # These are training-fit probes; still do not expose GT to selection.
            with safe_open(str(source/meta['file']),framework='pt',device='cpu') as h:
                target=h.get_tensor('target')[None].to(device).float()/127.5-1
            metrics={k:measured_metrics(v.to(device).float()/127.5-1,target,perceptual)
                     for k,v in result['outputs'].items()}
            file=f'probe_outputs_{i:03d}_{time.time_ns()}.safetensors'
            sha=tensor_file(output/file,result.pop('outputs'))
            row=dict(index=i,identity=r['selected'][i],metrics=metrics,outputs_file=file,outputs_sha256=sha,
                checkpoint_sha256=record['sha256'],protocol_sha256=r['protocol_sha256'],
                gt_loaded_after_selection=True,probe_role=r['spec']['probe_role'],**result)
            atomic.atomic_json(output/f'probe_{i:03d}.json',row)
            print('TAIL_RELATIVE_V3_PROBE '+json.dumps(dict(index=i,metrics=metrics,bounded_choice=row['bounded_choice'])),flush=True)
        dist.barrier()
        if rank==0:
            rows=[json.loads((output/f'probe_{i:03d}.json').read_text()) for i in r['spec']['probe_indices']]
            means={name:{metric:sum(row['metrics'][name][metric] for row in rows)/len(rows)
                         for metric in ('psnr','lpips')} for name in rows[0]['metrics']}
            baseline=means['original']
            for name,value in means.items():
                value['psnr_gain']=value['psnr']-baseline['psnr']
                value['lpips_relative_reduction']=1-value['lpips']/baseline['lpips']
                value['joint_target']=value['psnr_gain']>=r['spec']['psnr_gain_target'] and value['lpips_relative_reduction']>=r['spec']['lpips_relative_reduction_target']
            result=dict(status='training_fit_probe_complete_not_generalization',mode=r['mode'],
                engineering_pass=True,steps=state['step'],world_size=r['world_size'],means=means,
                bounded_accepted_images=sum(row['bounded_choice'] is not None for row in rows),
                raw_accepted_images=sum(row['raw_choice'] is not None for row in rows),
                rows=rows,trust_limits=limits,trust_is_calibrated=False,
                checkpoint_sha256=record['sha256'],protocol_sha256=r['protocol_sha256'],source_sha256=r['source_sha256'],
                protected_pixels_accessed=False,evaluation_pixels_accessed=False,gt_used_for_selection=False,
                formal_authorized=False,auto_trainfit=False,generalization_established=False)
            immutable_json(output/'result.json',result)
            print('TAIL_RELATIVE_V3_PROBE_COMPLETE '+json.dumps(means),flush=True)
        dist.barrier()
    finally:dist.destroy_process_group()


def verify(output,r):
    from smoke import flux_tail_relative_checkpoint_v3 as checkpoints
    result=json.loads((output/'result.json').read_text());state,record=checkpoints.load(output,r)
    if (state['step']!=r['spec']['modes'][r['mode']]['steps']
        or result['checkpoint_sha256']!=record['sha256']
        or result['source_sha256']!=r['source_sha256'] or result['protocol_sha256']!=r['protocol_sha256']
        or result['gt_used_for_selection'] or result['evaluation_pixels_accessed']
        or result['protected_pixels_accessed'] or result['generalization_established']):
        raise RuntimeError('result protocol violation')
    if [row['index'] for row in result['rows']]!=list(range(8)):raise RuntimeError('missing/duplicate probes')
    for row in result['rows']:
        i=row['index']
        if (row['identity']!=r['selected'][i] or row['identity']['role']!='train'
            or row['gt_used_for_selection'] or not row['gt_loaded_after_selection']
            or row['checkpoint_sha256']!=record['sha256']):raise RuntimeError('probe protocol violation')
        common.bound(str(output/row['outputs_file']),row['outputs_sha256'])
    for name,metrics in result['means'].items():
        for key in ('psnr','lpips'):
            actual=sum(row['metrics'][name][key] for row in result['rows'])/8
            if not math.isfinite(actual) or abs(actual-metrics[key])>1e-10:raise RuntimeError('aggregate mismatch')
    print('TAIL_RELATIVE_V3_RESULT_VERIFIED engineering_only_no_auto_training',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=['hardware','prepare','cache','mine','train','probe','verify'])
    parser.add_argument('--mode',choices=['qualification','trainfit'],default='qualification')
    parser.add_argument('--run-id',default='qualification_v1');parser.add_argument('--gpus',type=int,choices=[4,8],default=8)
    parser.add_argument('--qualification-run-id',default='qualification_v1')
    parser.add_argument('--generation',type=int,default=0);parser.add_argument('--until',type=int)
    args=parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+',args.run_id):raise ValueError('invalid run ID')
    output=common.path(spec()['output_root'])/args.run_id
    if args.phase=='hardware':base.hardware(args.gpus);return
    if args.phase=='prepare':prepare(output,args.mode,args.gpus,args.qualification_run_id);return
    r=load_receipt(output,args.gpus)
    if r['mode']!=args.mode:raise RuntimeError('mode changed for existing run')
    if args.phase=='cache':cache_base(output,r)
    elif args.phase=='mine':
        if args.generation not in [0]+r['spec']['modes'][r['mode']]['refresh_steps']:raise ValueError('undeclared refresh')
        mine(output,r,args.generation)
    elif args.phase=='train':
        if not train(output,r,args.until):raise SystemExit(75)
    elif args.phase=='probe':probe(output,r)
    else:verify(output,r)


if __name__=='__main__':main()
