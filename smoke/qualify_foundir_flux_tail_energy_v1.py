#!/usr/bin/env python3
"""25-step learning qualification: cached real tails -> energy -> GT-free tail guidance."""
from __future__ import annotations
import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import random
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from smoke import flux_tail_energy_contract_v1 as contract
from smoke import flux_energy_qualification_contract_v1 as common
from smoke import diagnose_foundir_flux_tail_oracle_v1 as base


def perceptual_model(device):
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    return LearnedPerceptualImagePatchSimilarity(net_type='alex').to(device).eval().requires_grad_(False).net


def cache(output, receipt):
    import torch
    import torch.distributed as dist
    from safetensors.torch import save_file
    from smoke.qualify_foundir_flux_energy_v1 import image_crop
    from smoke.flux_tail_energy_runtime_v1 import FrozenFlow, cache_training_frames
    world = receipt['world_size']; rank, device = base.setup_ddp(world); started = time.monotonic()
    try:
        if rank == 0:
            for name, sha in receipt['artifacts'].items(): common.bound(name, sha)
        base.parent_receipt(receipt['parent_spec'])
        dist.barrier()
        flow, perceptual = FrozenFlow(receipt, device), perceptual_model(device)
        for index in contract.assigned(range(32), rank, world):
            tick = time.monotonic(); before = dict(flow.calls)
            identity = receipt['selected'][index]
            images = image_crop(identity['row'])
            session = flow.session(images['observation'], identity['seed'])
            images['proposal'] = ((session.proposal[0].detach().cpu()+1)*127.5).round().to(torch.uint8)
            kinds, history = [], []
            if identity['role'] == 'train':
                target = images['target'][None].to(device).float()/127.5-1
                frames, kinds, history = cache_training_frames(session, target, perceptual, receipt['spec'], identity['seed'])
                images.update(frames)
            elif identity['role'] != 'engineering_evaluation':
                raise RuntimeError('unexpected source role')
            flow.assert_frozen()
            filename = output/f'cache_{index:03d}.safetensors'
            if filename.exists(): raise FileExistsError(filename)
            save_file(images, str(filename))
            common.write_new(output/f'cache_{index:03d}.json', dict(identity=identity,
                file=filename.name, sha256=common.sha(filename), frame_kinds=kinds, teacher_history=history,
                timestep=session.timestep, baseline_parity_pass=True,
                config_sha256=receipt['config_sha256'], source_sha256=receipt['source_sha256'],
                seconds=time.monotonic()-tick, calls={k: flow.calls[k]-before[k] for k in flow.calls}))
            print('TAIL_ENERGY_CACHE '+json.dumps(dict(rank=rank,index=index,role=identity['role'],
                  continuous_frames=len(kinds),quantized_frames=len(kinds),seconds=time.monotonic()-tick)), flush=True)
            del session, images
        dist.barrier()
        if rank == 0:
            items = [contract.cache_meta(output, receipt, i, receipt['selected'][i]['role']) for i in range(32)]
            common.write_new(output/'cache_complete.json', dict(records=32, train_sources=24, evaluation_sources=8,
                seconds=time.monotonic()-started, config_sha256=receipt['config_sha256'],
                source_sha256=receipt['source_sha256'], train_frames=sum(len(i['frame_kinds'])*2 for i in items),
                baseline_parity_pass=all(i['baseline_parity_pass'] for i in items), protected_pixels_accessed=False))
            print('TAIL_ENERGY_CACHE_COMPLETE records=32 train=24 evaluation=8', flush=True)
        dist.barrier()
    finally:
        dist.destroy_process_group()


def parameter_hash(model):
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode()); digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def pair_choice(slot, seed, sizes):
    rng = random.Random(int(common.digest(seed, 'global_training_slot', slot)[:16], 16))
    index = rng.randrange(24)
    # Choose two different states. Each may be continuous or actually quantized.
    first, second = rng.sample(range(sizes[index]), 2)
    return index, [(first, rng.randrange(2)), (second, rng.randrange(2))]


def load_pair(output, receipt, metadata, slot, device):
    import torch
    from safetensors import safe_open
    index, choices = pair_choice(slot, receipt['spec']['seed'], [len(m['frame_kinds']) for m in metadata])
    meta = metadata[index]
    if meta['identity']['role'] != 'train': raise RuntimeError('evaluation source entered training')
    with safe_open(str(output/meta['file']), framework='pt', device='cpu') as h:
        observation, proposal, target = [h.get_tensor(k)[None].float()/127.5-1 for k in ('observation','proposal','target')]
        candidates = []
        for frame, quantized in choices:
            key = 'frames_quantized' if quantized else 'frames_continuous'
            value = h.get_slice(key)[frame].float()
            candidates.append(value/127.5-1 if quantized else value)
    values = [v.repeat(2,1,1,1).to(device) for v in (observation,proposal,target)]
    condition = torch.tensor([common.descriptor(meta['identity']['row'])]*2, device=device)
    timestep = torch.full((2,1), meta['timestep'], device=device)
    return (*values, torch.stack(candidates).to(device), condition, timestep)


def train(output, receipt):
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from lowlight_restoration.flux_tail_learned_energy import TailQualityEnergy, quality_targets, learning_loss
    world=receipt['world_size']; rank,device=base.setup_ddp(world); started=time.monotonic(); spec=receipt['spec']
    try:
        done=json.loads((output/'cache_complete.json').read_text())
        if done['config_sha256'] != receipt['config_sha256'] or done['source_sha256'] != receipt['source_sha256']:
            raise RuntimeError('cache source identity changed')
        metadata=[contract.cache_meta(output,receipt,i,'train') for i in range(24)]
        torch.manual_seed(spec['seed']); torch.cuda.manual_seed_all(spec['seed'])
        model=TailQualityEnergy(**spec['energy']).to(device)
        initial_hash=parameter_hash(model)
        wrapped=DDP(model,device_ids=[device.index],broadcast_buffers=False,gradient_as_bucket_view=False)
        optimizer=torch.optim.AdamW(model.parameters(),lr=spec['learning_rate'],weight_decay=spec['weight_decay'])
        perceptual=perceptual_model(device); timings=[]; all_stats=[]
        torch.cuda.reset_peak_memory_stats(device)
        for step in range(1,spec['training_steps']+1):
            tick=time.monotonic(); optimizer.zero_grad(set_to_none=True)
            lr=spec['learning_rate']*min(step/spec['warmup_steps'],1.)
            for group in optimizer.param_groups: group['lr']=lr
            slots=contract.training_slots(step,rank,world); stats={}
            for micro,slot in enumerate(slots):
                observation,proposal,target,candidate,condition,timestep=load_pair(output,receipt,metadata,slot,device)
                labels=quality_targets(candidate,target,perceptual)
                with wrapped.no_sync() if micro<len(slots)-1 else nullcontext():
                    prediction=wrapped(observation,proposal,candidate,condition,timestep)
                    loss,diagnostics=learning_loss(prediction,labels,**spec['loss'])
                    if not bool(loss.isfinite()): raise RuntimeError('nonfinite training loss')
                    (loss/len(slots)).backward()
                for key,value in {'loss':loss,**diagnostics}.items():
                    stats[key]=stats.get(key,0.)+float(value.detach())/len(slots)
                del prediction,labels,loss,diagnostics
            if any(p.grad is None or not bool(p.grad.isfinite().all()) for p in model.parameters()):
                raise RuntimeError('missing/nonfinite second-order parameter gradient')
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),spec['gradient_clip'],error_if_nonfinite=True)
            optimizer.step(); torch.cuda.synchronize(); timings.append(time.monotonic()-tick)
            names=sorted(stats); values=torch.tensor([stats[k] for k in names],device=device,dtype=torch.float64)
            dist.all_reduce(values); values/=world
            row=dict(step=step,lr=lr,rank0_seconds=timings[-1],rank0_grad_norm=float(norm),
                     global_mean_stats=dict(zip(names,values.tolist(),strict=True)))
            all_stats.append(row)
            if rank==0: print('TAIL_ENERGY_TRAIN '+json.dumps(row),flush=True)
        final_hash=parameter_hash(model); hashes=[None]*world; dist.all_gather_object(hashes,final_hash)
        if len(set(hashes))!=1 or final_hash==initial_hash: raise RuntimeError('DDP divergence/no model update')
        peaks=[None]*world; dist.all_gather_object(peaks,torch.cuda.max_memory_allocated(device)/1024**3)
        if rank==0:
            checkpoint=output/'energy_step_000025.pt'
            if checkpoint.exists(): raise FileExistsError(checkpoint)
            torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),step=25,
                config_sha256=receipt['config_sha256'],source_sha256=receipt['source_sha256'],
                next_global_slot=800,may_seed_formal=False),checkpoint)
            saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
            for key,value in model.state_dict().items():
                if not torch.equal(value.detach().cpu(),saved['model'][key]): raise RuntimeError('checkpoint roundtrip failed')
            for state in saved['optimizer']['state'].values():
                if int(state['step'])!=25 or any(not bool(state[k].isfinite().all()) for k in ('exp_avg','exp_avg_sq')):
                    raise RuntimeError('incomplete/nonfinite saved Adam state')
            common.write_new(output/'training_complete.json',dict(steps=25,checkpoint_sha256=common.sha(checkpoint),
                parameter_count=sum(p.numel() for p in model.parameters()),rank_parameter_sha256=hashes,
                initial_parameter_sha256=initial_hash,peak_memory_gib=peaks,rank0_mean_step_seconds=sum(timings)/25,
                seconds=time.monotonic()-started,training_log=all_stats,may_seed_formal=False,
                config_sha256=receipt['config_sha256'],source_sha256=receipt['source_sha256']))
            print('TAIL_ENERGY_TRAIN_COMPLETE steps=25 checkpoint_verified=true',flush=True)
        dist.barrier()
    finally:
        dist.destroy_process_group()


def evaluate(output,receipt):
    import torch
    import torch.distributed as dist
    from safetensors import safe_open
    from safetensors.torch import save_file
    from lowlight_restoration.flux_tail_learned_energy import TailQualityEnergy
    from smoke.flux_tail_energy_runtime_v1 import FrozenFlow, learned_guidance, measured_metrics
    world=receipt['world_size'];rank,device=base.setup_ddp(world);started=time.monotonic()
    try:
        training=json.loads((output/'training_complete.json').read_text())
        checkpoint=common.bound(str(output/'energy_step_000025.pt'),training['checkpoint_sha256'])
        state=torch.load(checkpoint,map_location='cpu',weights_only=True)
        if state['config_sha256']!=receipt['config_sha256'] or state['source_sha256']!=receipt['source_sha256']:
            raise RuntimeError('checkpoint protocol mismatch')
        energy=TailQualityEnergy(**receipt['spec']['energy']).to(device)
        energy.load_state_dict(state['model'],strict=True);energy.eval().requires_grad_(False)
        if parameter_hash(energy)!=training['rank_parameter_sha256'][0]: raise RuntimeError('evaluation weights mismatch')
        flow=FrozenFlow(receipt,device);perceptual=perceptual_model(device)
        for index in contract.assigned(range(24,32),rank,world):
            tick=time.monotonic();before=dict(flow.calls);torch.cuda.reset_peak_memory_stats(device)
            meta=contract.cache_meta(output,receipt,index,'engineering_evaluation')
            # Only observation/proposal are read before inference. GT is absent
            # from both the session object and learned_guidance's interface.
            with safe_open(str(output/meta['file']),framework='pt',device='cpu') as h:
                observation=h.get_tensor('observation');proposal=h.get_tensor('proposal')
            session=flow.session(observation,meta['identity']['seed'],expected_proposal=proposal)
            condition=torch.tensor([common.descriptor(meta['identity']['row'])],device=device)
            guidance=learned_guidance(session,energy,condition,receipt['spec'])
            flow.assert_frozen()
            if any(p.grad is not None or p.requires_grad for p in energy.parameters()):
                raise RuntimeError('energy weights must be frozen during guidance')
            # GT first loaded after all directions, acceptance and stopping.
            with safe_open(str(output/meta['file']),framework='pt',device='cpu') as h:
                target=h.get_tensor('target')[None].to(device).float()/127.5-1
            images={'historical_bf16':proposal[None], 'fp32_control':guidance['control_uint8']}
            images.update({f'energy_step{k}':v for k,v in guidance['milestones'].items()})
            measured={k:measured_metrics(v.to(device).float()/127.5-1,target,perceptual) for k,v in images.items()}
            filename=output/f'outputs_{index:03d}.safetensors'
            if filename.exists(): raise FileExistsError(filename)
            save_file({k:v.contiguous() for k,v in images.items()},str(filename))
            row=dict(index=index,identity=meta['identity'],rank=rank,metrics=measured,
                predicted_baseline=guidance['predicted_baseline'],trajectory=guidance['history'],
                accepted_updates=guidance['accepted_updates'],stop_reason=guidance['stop_reason'],
                gt_used_for_guidance=False,gt_loaded_after_guidance=True,baseline_parity_pass=True,
                outputs_file=filename.name,outputs_sha256=common.sha(filename),
                seconds=time.monotonic()-tick,peak_memory_gib=torch.cuda.max_memory_allocated(device)/1024**3,
                calls={k:flow.calls[k]-before[k] for k in flow.calls},config_sha256=receipt['config_sha256'],
                source_sha256=receipt['source_sha256'])
            common.write_new(output/f'evaluation_{index:03d}.json',row)
            print('TAIL_ENERGY_EVAL '+json.dumps(dict(index=index,metrics=measured,
                  accepted_updates=row['accepted_updates'],seconds=row['seconds'])),flush=True)
        dist.barrier()
        if rank==0:
            rows=[json.loads((output/f'evaluation_{i:03d}.json').read_text()) for i in range(24,32)]
            result=aggregate(rows,receipt)
            result.update(checkpoint_sha256=training['checkpoint_sha256'],training=training,
                          evaluation_seconds=time.monotonic()-started)
            common.write_new(output/'result.json',result)
            print('TAIL_ENERGY_QUALIFICATION_COMPLETE '+json.dumps(dict(engineering_pass=True,
                measured_primary=result['means']['energy_step16'],generalization_success_established=False)),flush=True)
        dist.barrier()
    finally:
        dist.destroy_process_group()


def aggregate(rows,receipt):
    import math
    if sorted(r['index'] for r in rows)!=list(range(24,32)):
        raise RuntimeError('missing/duplicate evaluation records')
    names={'historical_bf16','fp32_control','energy_step1','energy_step2','energy_step4','energy_step8','energy_step16'}
    for row in rows:
        if (row['identity']!=receipt['selected'][row['index']] or row['identity']['role']!='engineering_evaluation'
                or row['rank']!=(row['index']-24)%receipt['world_size'] or set(row['metrics'])!=names
                or row['gt_used_for_guidance'] or not row['gt_loaded_after_guidance'] or not row['baseline_parity_pass']
                or row['source_sha256']!=receipt['source_sha256'] or row['config_sha256']!=receipt['config_sha256']):
            raise RuntimeError('invalid evaluation identity/GT boundary')
        if any(not math.isfinite(v) for m in row['metrics'].values() for v in m.values()):
            raise RuntimeError('nonfinite evaluation metric')
    means={k:{m:sum(r['metrics'][k][m] for r in rows)/8 for m in ('psnr','lpips')} for k in sorted(names)}
    baseline=dict(means['historical_bf16'])
    for value in means.values():
        value.update(psnr_gain=value['psnr']-baseline['psnr'],lpips_reduction=1-value['lpips']/max(baseline['lpips'],1e-12))
        value['joint_target']=value['psnr_gain']>=.3 and value['lpips_reduction']>=receipt['spec']['lpips_relative_reduction_target']
    return dict(status='tail_energy_engineering_complete',engineering_pass=True,steps=25,rows=rows,means=means,
        primary_endpoint='energy_step16',world_size=receipt['world_size'],train_sources=24,evaluation_sources=8,
        evaluation_role='source_disjoint_training_pool_engineering_check',energy_trained=True,
        gt_used_for_guidance=False,predicted_safety_is_guaranteed=False,scientific_gate_evaluated=False,
        generalization_success_established=False,qualification_may_seed_formal=False,auto_formal=False,
        protected_pixels_accessed=False,config_sha256=receipt['config_sha256'],source_sha256=receipt['source_sha256'])


def verify(output,receipt):
    from lowlight_restoration.flux_tail_budget import choose_trial
    result=json.loads((output/'result.json').read_text())
    rebuilt=aggregate(result['rows'],receipt)
    for k,v in rebuilt.items():
        if result[k]!=v: raise RuntimeError('result mismatch: '+k)
    common.bound(str(output/'energy_step_000025.pt'),result['checkpoint_sha256'])
    for row in result['rows']:
        if json.loads((output/f'evaluation_{row["index"]:03d}.json').read_text())!=row:
            raise RuntimeError('evaluation file mismatch')
        common.bound(str(output/row['outputs_file']),row['outputs_sha256'])
        history=row['trajectory'];accepted=0
        if not 1<=len(history)<=16: raise RuntimeError('invalid trajectory length')
        current=history[0]['current_before']
        for step,item in enumerate(history,1):
            if item['step']!=step or item['current_before']!=current: raise RuntimeError('trajectory chain mismatch')
            if item['stop_reason']=='negligible_learned_direction':
                if item['trials'] or item['chosen_trial'] is not None: raise RuntimeError('invalid zero-gradient stop')
            else:
                chosen,decisions=choose_trial(item['trials'],current,row['predicted_baseline'],'psnr_budget',.3)
                if chosen!=item['chosen_trial'] or decisions!=item['decision']: raise RuntimeError('predicted selection mismatch')
                if chosen is not None: current=item['trials'][chosen]['metrics'];accepted+=1
            if item['accepted_updates']!=accepted or item['current_after']!=current: raise RuntimeError('accepted count mismatch')
            if item['chosen_trial'] is None and step!=len(history): raise RuntimeError('did not stop after rejection')
        if len(history)<16 and history[-1]['chosen_trial'] is not None: raise RuntimeError('truncated trajectory')
        if row['accepted_updates']!=accepted: raise RuntimeError('final accepted count mismatch')
    print('TAIL_ENERGY_RESULT_VERIFIED engineering_only_no_formal_authorized',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=('hardware','preflight','prepare','cache','train','evaluate','verify'))
    p.add_argument('--run-id',default='qualification_4gpu_v1')
    p.add_argument('--gpus',type=int,choices=(4,8),default=4)
    args=p.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+',args.run_id): raise ValueError('invalid fresh run ID')
    spec=json.loads(contract.CONFIG.read_text());contract.validate_spec(spec)
    output=common.path(spec['output_root'])/args.run_id
    if args.phase=='hardware': base.hardware(args.gpus)
    elif args.phase in ('preflight','prepare'): contract.prepare(output,args.gpus,args.phase=='prepare')
    else:
        receipt=contract.load(output,args.gpus)
        if args.phase=='cache' and any(p.name!='preflight.json' for p in output.iterdir()):
            raise FileExistsError('preserve partial cache; use fresh ID')
        if args.phase=='train' and (output/'energy_step_000025.pt').exists():
            raise FileExistsError('qualification cannot resume or overwrite weights')
        if args.phase=='evaluate' and any(output.glob('evaluation_*.json')):
            raise FileExistsError('preserve prior/partial evaluation')
        globals()[args.phase](output,receipt)


if __name__=='__main__': main()
