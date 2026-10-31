#!/usr/bin/env python3
"""Real FLUX-only 25-step, full-three-update engineering qualification.

No automatic formal run. Per-step durable adapters/Adam/RNG/stream resume.
The eight probes are training sources, not independent generalization evidence.
"""
from __future__ import annotations

import argparse
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import random
import socket
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'third_party/flux2/src'))
from smoke import flux_native_energy_contract_v1 as contract


def hardware(world):
    import torch
    _,local_gpus=contract.topology(world)
    if not torch.cuda.is_available() or torch.cuda.device_count()!=local_gpus:
        raise RuntimeError(f'expose exactly {local_gpus} local GPUs for total world={world}')
    devices=[torch.cuda.get_device_properties(i) for i in range(local_gpus)]
    if len({p.name for p in devices})!=1 or any(not any(k in p.name for k in ('H100','H200'))
                                              or p.total_memory<75*1024**3 for p in devices):
        raise RuntimeError('requires homogeneous H100/H200 devices, at least 75 GiB each')
    if len({str(p.uuid) for p in devices})!=local_gpus:
        raise RuntimeError('duplicate CUDA device identities')
    for i in range(local_gpus):
        free,total=torch.cuda.mem_get_info(i)
        if total-free>2*1024**3:
            raise RuntimeError(f'GPU {i} is busy; do not overlap or kill another job')
    print('NATIVE_ENERGY_HARDWARE_PASS '+json.dumps([p.name for p in devices]),flush=True)


def prepare(output,world,resume,node_rank=0):
    from smoke.flux_native_energy_runtime_v1 import atomic_json
    nodes,_=contract.topology(world)
    if not 0<=node_rank<nodes:raise ValueError('node rank outside layout')
    receipt=contract.preflight(world)
    if node_rank:
        deadline=time.monotonic()+900
        print(f'NATIVE_ENERGY_WAIT_NODE0_RECEIPT node_rank={node_rank}',flush=True)
        while not (output/'preflight.json').exists():
            if time.monotonic()>=deadline:raise TimeoutError('node0 receipt not visible; check shared root/node0 log')
            time.sleep(1)
        if json.loads((output/'preflight.json').read_text())!=receipt:
            raise RuntimeError('node source/config/assets/topology differ from node0')
        if (output/'result.json').exists():raise RuntimeError('completed qualification must not be relaunched')
        print(f'NATIVE_ENERGY_PREPARED node_rank={node_rank} '+receipt['protocol_sha256'],flush=True)
        return
    if output.exists():
        if not resume: raise FileExistsError('use a fresh run ID or explicit resume')
        if json.loads((output/'preflight.json').read_text())!=receipt:
            raise RuntimeError('resume source/config/assets/topology changed')
        if (output/'result.json').exists():raise RuntimeError('completed qualification must not be relaunched')
    else:
        if resume:raise FileNotFoundError('cannot resume a nonexistent run')
        output.mkdir(parents=True)
        atomic_json(output/'preflight.json',receipt)
    print('NATIVE_ENERGY_PREPARED '+receipt['protocol_sha256'],flush=True)


def cpu_tree(value):
    import torch
    if isinstance(value,torch.Tensor):return value.detach().cpu().clone()
    if isinstance(value,dict):return {k:cpu_tree(v) for k,v in value.items()}
    if isinstance(value,list):return [cpu_tree(v) for v in value]
    if isinstance(value,tuple):return tuple(cpu_tree(v) for v in value)
    return value


def parameter_hash(state):
    digest=hashlib.sha256()
    for name,value in sorted(state.items()):
        digest.update(name.encode());digest.update(value.contiguous().numpy().tobytes())
    return digest.hexdigest()


def sample_index(slot,seed):
    # A fresh deterministic permutation every 24 global samples, independent
    # of rank count. Actual world size remains bound for exact RNG resume.
    epoch,position=divmod(slot,24)
    indices=list(range(24));random.Random(f'{seed}:{epoch}').shuffle(indices)
    return indices[position]


def action_pair(slot):
    # Every action is actually seen, including wrong expert and repeated FLUX;
    # GT positive is training-only and never enters inference selection.
    options=[(0,i) for i in (1,2,3,4,5,6,7)]+[(2,i) for i in (1,3,4,5,6,7)]
    return options[slot%len(options)]


def sample_slots(step,rank,world):
    """Exactly eight samples, rotating the two extra slots for world=6."""
    if step<1 or not 0<=rank<world or world not in (2,4,6,8):
        raise ValueError('invalid sample assignment')
    logical=(rank+step-1)%world
    return range((step-1)*8+logical,step*8,world)


def average_gradients(parameters,world):
    """Each local sample loss is scaled world/8, so this is global sample mean."""
    import torch
    import torch.distributed as dist
    for parameter in parameters:
        if parameter.grad is None:parameter.grad=torch.zeros_like(parameter)
        dist.all_reduce(parameter.grad);parameter.grad.div_(world)


def validate_execution(devices,world):
    nodes,local_gpus=contract.topology(world)
    if len(devices)!=world or [d['rank'] for d in devices]!=list(range(world)):
        raise RuntimeError('missing/duplicate global ranks')
    if len({d['uuid'] for d in devices})!=world:
        raise RuntimeError('same GPU assigned more than once')
    if len({d['host'] for d in devices})!=nodes:
        raise RuntimeError('expected distinct node hostnames for the requested layout')
    for node in range(nodes):
        subset=devices[node*local_gpus:(node+1)*local_gpus]
        if (len({d['host'] for d in subset})!=1 or len({d['name'] for d in subset})!=1
            or [d['local_rank'] for d in subset]!=list(range(local_gpus))
            or any(d['node_rank']!=node for d in subset)):
            raise RuntimeError('per-node rank/GPU family mapping mismatch')
    # H100/H200 may differ across nodes, but the software and code root may not.
    for field in ('torch','cuda','python','packages','root','transport','launch_mode'):
        if any(d[field]!=devices[0][field] for d in devices):
            raise RuntimeError('cross-node environment mismatch: '+field)


def shared_storage_check(output,receipt,rank,world):
    import uuid
    import torch.distributed as dist
    from smoke.flux_native_energy_runtime_v1 import atomic_json
    nonce=[uuid.uuid4().hex if rank==0 else None]
    dist.broadcast_object_list(nonce,src=0)
    folder=output/'coordination'/nonce[0];folder.mkdir(parents=True,exist_ok=True)
    atomic_json(folder/f'rank_{rank}.json',dict(rank=rank,nonce=nonce[0],protocol=receipt['protocol_sha256']))
    dist.barrier()
    for peer in range(world):
        if json.loads((folder/f'rank_{peer}.json').read_text())!=dict(rank=peer,nonce=nonce[0],protocol=receipt['protocol_sha256']):
            raise RuntimeError('ranks do not see the same shared run directory')
    dist.barrier()
    if rank==0:print(f'NATIVE_ENERGY_SHARED_STORAGE_AND_COLLECTIVES_PASS ranks={world}',flush=True)


def save_step(output,model,optimizer,step,receipt,rank,world,device,elapsed,loss_log):
    import torch
    import torch.distributed as dist
    from smoke.flux_native_energy_runtime_v1 import atomic_write,atomic_json
    rng=dict(python=random.getstate(),cpu=torch.get_rng_state(),
             cuda=torch.cuda.get_rng_state(device) if torch.device(device).type=='cuda' else None)
    gathered=[None]*world;dist.all_gather_object(gathered,rng)
    if rank==0:
        state=dict(schema=1,model=model.trainable_state(),optimizer=cpu_tree(optimizer.state_dict()),
            step=step,next_slot=step*receipt['spec']['global_batch'],rank_rng=gathered,world=world,
            protocol_sha256=receipt['protocol_sha256'],elapsed=elapsed,loss_log=loss_log,
            execution_sha256=contract.sha(output/'execution.json'),may_seed_trainfit=False)
        state['model_sha256']=parameter_hash(state['model'])
        name=f'step_{step:06d}_{time.time_ns()}.pt'
        directory=output/'steps';directory.mkdir(exist_ok=True)
        atomic_write(directory/name,lambda h:torch.save(state,h))
        record=dict(step=step,file='steps/'+name,sha256=contract.sha(directory/name),model_sha256=state['model_sha256'])
        atomic_json(directory/f'step_{step:06d}.json',record)
        atomic_json(output/'latest.json',record,replace=True)
        print(f'NATIVE_ENERGY_CHECKPOINT_SAVED step={step} sha256={record["sha256"]}',flush=True)
    dist.barrier()


def load_step(output,model,optimizer,receipt,rank,world,device):
    import torch
    files=sorted((output/'steps').glob('step_*.json'))
    if not files:return 0,0.,[]
    info=json.loads(files[-1].read_text());path=output/info['file']
    if path.resolve().parent!=(output/'steps').resolve() or contract.sha(path)!=info['sha256']:
        raise RuntimeError('checkpoint receipt/hash mismatch; no silent rollback')
    state=torch.load(path,map_location='cpu',weights_only=True)
    step=state['step']
    if (state['schema']!=1 or not 0<=step<=25 or step!=info['step'] or state['world']!=world
        or state['protocol_sha256']!=receipt['protocol_sha256'] or state['next_slot']!=step*8
        or state['execution_sha256']!=contract.sha(output/'execution.json') or state['may_seed_trainfit']
        or len(state['rank_rng'])!=world or len(state['loss_log'])!=step
        or parameter_hash(state['model'])!=info['model_sha256']):
        raise RuntimeError('checkpoint execution/progress/parameters changed')
    model.load_trainable_state(state['model']);optimizer.load_state_dict(state['optimizer'])
    for item in optimizer.state.values():
        if int(item['step'])!=step or not bool(item['exp_avg'].isfinite().all() and item['exp_avg_sq'].isfinite().all()):
            raise RuntimeError('Adam is incomplete/nonfinite')
    if step and not optimizer.state:raise RuntimeError('missing Adam state')
    rng=state['rank_rng'][rank]
    random.setstate(rng['python']);torch.set_rng_state(rng['cpu'])
    if rng['cuda'] is not None:torch.cuda.set_rng_state(rng['cuda'],device)
    print(f'NATIVE_ENERGY_STATE_RESTORED rank={rank} step={step} next_slot={state["next_slot"]}',flush=True)
    return step,state['elapsed'],state['loss_log']


def probe(output,receipt,model,ae,perceptual,contexts,rank,world,device):
    import torch
    import torch.distributed as dist
    from flux2.sampling import get_schedule
    from lowlight_restoration.flux_native_energy import unroll,select_without_target
    from smoke import flux_native_energy_runtime_v1 as runtime
    rows=[]
    for index in receipt['spec']['probe_indices'][rank::world]:
        record=runtime.load_record(output,receipt,index,device)
        t=get_schedule(50,record['initial'].shape[1])[48]
        args=runtime.energy_arguments(record,contexts,t)
        energy=lambda z:model(z,**args)
        spec=receipt['spec']
        states,scores=unroll(energy,record['initial'].float(),training=False,
            updates=spec['updates'],step_size=spec['step_size'],radius=spec['latent_rms_radius'])
        choice=select_without_target(scores,spec['selection_energy_margin'])
        measured=[runtime.metrics(record['proposal'],record['target'],perceptual)]
        with torch.no_grad():
            _,zero=runtime.deployed(record['observation'],runtime.render_tail(model,ae,record,contexts,states[0]))
            control=runtime.metrics(zero,record['target'],perceptual)
            for state in states[1:]:
                _,image=runtime.deployed(record['observation'],runtime.render_tail(model,ae,record,contexts,state))
                measured.append(runtime.metrics(image,record['target'],perceptual))
        rows.append(dict(index=index,source_group=receipt['selected'][index]['row']['source_group'],
            metrics=measured,fp32_zero_control=control,selected_index=choice,selected=measured[choice],
            energy=[float(v.detach()) for v in scores],target_used_for_selection=False))
        print('NATIVE_ENERGY_TRAIN_PROBE '+json.dumps(rows[-1]),flush=True)
    gathered=[None]*world;dist.all_gather_object(gathered,rows)
    return sorted([row for group in gathered for row in group],key=lambda x:x['index'])


def run(output,world,resume=False):
    import torch
    import torch.distributed as dist
    from flux2.sampling import get_schedule
    from lowlight_restoration.flux_native_energy import FluxNativeEnergy,unroll,direct_quality_loss,pareto_ranking_loss
    from smoke import flux_native_energy_runtime_v1 as runtime
    nodes,local_gpus=contract.topology(world)
    if int(os.environ['WORLD_SIZE'])!=world or int(os.environ['LOCAL_WORLD_SIZE'])!=local_gpus:
        raise RuntimeError('torchrun topology differs from frozen layout')
    local=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(local);device=torch.device('cuda',local)
    dist.init_process_group('nccl',device_id=device,timeout=timedelta(minutes=45));rank=dist.get_rank()
    receipt=json.loads((output/'preflight.json').read_text())
    unsigned=dict(receipt);unsigned.pop('protocol_sha256')
    if (receipt['sources']!=contract.binding() or receipt['config_sha256']!=contract.sha(contract.CONFIG)
        or receipt['spec']!=contract.load_spec() or receipt['world_size']!=world
        or receipt['topology']!=[nodes,local_gpus]
        or hashlib.sha256(json.dumps(unsigned,sort_keys=True).encode()).hexdigest()!=receipt['protocol_sha256']):
        raise RuntimeError('implementation changed after preflight')
    spec=receipt['spec'];start=time.monotonic()
    try:
        props=torch.cuda.get_device_properties(device)
        from importlib.metadata import version
        identity=dict(host=socket.gethostname(),rank=rank,local_rank=local,node_rank=rank//local_gpus,
                      name=props.name,uuid=str(props.uuid),root=str(ROOT),launch_mode='resume' if resume else 'fresh',
                      torch=torch.__version__,cuda=torch.version.cuda,python=sys.version,
                      packages={k:version(k) for k in ('torchmetrics','safetensors','transformers','torchvision')},
                      transport={k:os.environ.get(k,'') for k in ('NCCL_IB_DISABLE','NCCL_NVLS_ENABLE',
                          'NCCL_SOCKET_IFNAME','GLOO_SOCKET_IFNAME')})
        hosts=[None]*world;dist.all_gather_object(hosts,identity)
        validate_execution(hosts,world)
        shared_storage_check(output,receipt,rank,world)
        execution=dict(protocol_sha256=receipt['protocol_sha256'],
                       devices=[{k:v for k,v in h.items() if k!='launch_mode'} for h in hosts])
        if rank==0:
            if (output/'execution.json').exists():
                if json.loads((output/'execution.json').read_text())!=execution:
                    raise RuntimeError('exact resume requires same device/runtime mapping')
            else:runtime.atomic_json(output/'execution.json',execution)
        dist.barrier()
        flow,ae,perceptual=runtime.load_models(receipt,device)
        runtime.build_cache(output,receipt,rank,world,flow,ae,perceptual,device);dist.barrier()
        torch.manual_seed(spec['seed']);random.seed(spec['seed'])
        # One real bound state proves adding adapters does not alter G.
        check=runtime.load_record(output,receipt,rank,device);contexts=runtime.prompt(receipt,'noise',device)
        time48=get_schedule(50,check['initial'].shape[1])[48]
        with torch.no_grad():
            before=runtime.velocity(flow,check['initial'],check['ids'],check['reference'],check['reference_ids'],
                contexts['cfg_context'],contexts['cfg_ids'],time48)
        model=FluxNativeEnergy(flow,spec['energy_rank'],spec['energy_alpha'],expected_adapters=50)
        with torch.no_grad():
            after=runtime.velocity(model.generation,check['initial'],check['ids'],check['reference'],check['reference_ids'],
                contexts['cfg_context'],contexts['cfg_ids'],time48)
        if not torch.equal(before,after):raise RuntimeError('energy-off generator changed after adapter insertion')
        del before,after,check
        parameters=[p for p in model.parameters() if p.requires_grad]
        for p in parameters:dist.broadcast(p.data,src=0)
        architecture=model.architecture_receipt()
        if rank==0:
            if not (output/'architecture.json').exists():runtime.atomic_json(output/'architecture.json',architecture)
            print('NATIVE_ENERGY_ARCHITECTURE '+json.dumps(architecture),flush=True)
        optimizer=torch.optim.AdamW(parameters,lr=spec['learning_rate'],weight_decay=spec['weight_decay'])
        step,elapsed_before,loss_log=load_step(output,model,optimizer,receipt,rank,world,device)
        if not (output/'steps'/'step_000000.json').exists():
            save_step(output,model,optimizer,0,receipt,rank,world,device,0.,loss_log)
        training_start=time.monotonic(); initial_hash=parameter_hash(model.trainable_state())
        for next_step in range(step+1,spec['steps']+1):
            tick=time.monotonic();optimizer.zero_grad(set_to_none=True)
            learning_rate=spec['learning_rate']*min(1.,next_step/spec['warmup_steps'])
            for group in optimizer.param_groups:group['lr']=learning_rate
            local_loss=0.
            for slot in sample_slots(next_step,rank,world):
                index=sample_index(slot,spec['seed']);record=runtime.load_record(output,receipt,index,device)
                t=get_schedule(50,record['initial'].shape[1])[48]
                args=runtime.energy_arguments(record,contexts,t)
                states,scores=unroll(lambda z:model(z,**args),record['initial'].float(),training=True,
                    updates=3,step_size=spec['step_size'],radius=spec['latent_rms_radius'])
                images=[runtime.continuous_output(record,runtime.render_tail(model,ae,record,contexts,z)) for z in states[1:]]
                direct,diagnostics=direct_quality_loss(images,record['target'],record['proposal'],perceptual)
                with torch.no_grad():
                    path_costs=torch.stack([runtime.quality(record['proposal'],record['target'],perceptual)] +
                        [runtime.quality(image,record['target'],perceptual) for image in images])
                trajectory_rank=pareto_ranking_loss(torch.cat(scores),path_costs,margin=spec['ranking_margin'])
                pair=action_pair(slot);endpoint_args=runtime.energy_arguments(record,contexts,0.)
                action_scores=torch.cat([model(record['actions'][k:k+1],**endpoint_args) for k in pair])
                action_rank=pareto_ranking_loss(action_scores,record['costs'][list(pair)],margin=spec['ranking_margin'])
                loss=direct+spec['ranking_weight']*(trajectory_rank+action_rank)
                if not bool(loss.isfinite()):raise RuntimeError('nonfinite training loss')
                (loss*(world/8)).backward();local_loss+=float(loss.detach())*(world/8)
                del states,scores,images,direct,loss,action_scores,trajectory_rank,action_rank,record
            # Explicit reduction, not DDP's unsupported autograd.grad hook path.
            # Every parameter participates in identical order even for a zero rank loss.
            average_gradients(parameters,world)
            if any(not bool(p.grad.isfinite().all()) for p in parameters):raise RuntimeError('nonfinite parameter gradient')
            lora_norm=sum(float(p.grad.square().sum()) for n,p in model.named_parameters()
                          if '.energy_b.' in n and p.grad is not None)**.5
            if lora_norm<=0:raise RuntimeError('no gradient reached in-FLUX energy LoRA')
            norm=float(torch.nn.utils.clip_grad_norm_(parameters,spec['gradient_clip']))
            optimizer.step();model.architecture_receipt()
            value=torch.tensor(local_loss,device=device);dist.all_reduce(value);value/=world
            step_time=torch.tensor(time.monotonic()-tick,device=device)
            peak=torch.tensor(torch.cuda.max_memory_allocated(device)/1024**3,device=device)
            dist.all_reduce(step_time,op=dist.ReduceOp.MAX);dist.all_reduce(peak,op=dist.ReduceOp.MAX)
            log=dict(step=next_step,loss=float(value),gradient_norm=norm,energy_lora_gradient_norm=lora_norm,
                     learning_rate=learning_rate,step_seconds=time.monotonic()-tick,
                     slowest_rank_compute_seconds=float(step_time),
                     samples_per_rank=[len(sample_slots(next_step,r,world)) for r in range(world)],
                     peak_allocated_gib=float(peak))
            loss_log.append(log)
            elapsed=elapsed_before+time.monotonic()-training_start
            save_step(output,model,optimizer,next_step,receipt,rank,world,device,elapsed,loss_log)
            if rank==0:print('NATIVE_ENERGY_TRAIN '+json.dumps(log),flush=True)
        if step<25 and parameter_hash(model.trainable_state())==initial_hash:
            raise RuntimeError('optimizer did not change trainable FLUX parameters')
        final_hash=parameter_hash(model.trainable_state())
        restored_step,_,_=load_step(output,model,optimizer,receipt,rank,world,device)
        if restored_step!=25 or parameter_hash(model.trainable_state())!=final_hash:
            raise RuntimeError('final real-device checkpoint reload mismatch')
        optimizer.zero_grad(set_to_none=True)
        rows=probe(output,receipt,model,ae,perceptual,contexts,rank,world,device)
        attempt_duration=torch.tensor(time.monotonic()-start,device=device)
        attempt_durations=[torch.zeros_like(attempt_duration) for _ in range(world)]
        dist.all_gather(attempt_durations,attempt_duration)
        if rank==0:
            if len(rows)!=8 or [r['index'] for r in rows]!=list(range(8)):raise RuntimeError('incomplete probes')
            arms={}
            for arm in ('baseline','step1','step2','step3','selected','fp32_zero_control'):
                values=[r['selected'] if arm=='selected' else r['fp32_zero_control'] if arm=='fp32_zero_control'
                        else r['metrics'][0 if arm=='baseline' else int(arm[-1])] for r in rows]
                arms[arm]={k:sum(v[k] for v in values)/8 for k in ('psnr','lpips')}
            base,selected=arms['baseline'],arms['selected']
            gain=selected['psnr']-base['psnr'];reduction=1-selected['lpips']/base['lpips']
            result=dict(status='qualification_complete',engineering_pass=True,steps=25,world_size=world,
                protocol_sha256=receipt['protocol_sha256'],architecture=architecture,arms=arms,rows=rows,
                selected_psnr_gain=gain,selected_lpips_relative_reduction=reduction,
                training_probe_joint_gate=gain>=.3 and reduction>=spec['lpips_relative_reduction_target'],
                generalization_established=False,probe_role=spec['probe_role'],auto_trainfit=False,
                final_checkpoint_reloaded=True,
                protected_pixels_accessed=False,cnn_or_restormer_used=False,
                topology=[nodes,local_gpus],attempt_wall_seconds=max(float(v) for v in attempt_durations),
                attempt_gpu_hours=sum(float(v) for v in attempt_durations)/3600,
                train_log=loss_log,checkpoint=json.loads((output/'latest.json').read_text()))
            runtime.atomic_json(output/'result.json',result)
            print('NATIVE_ENERGY_QUALIFICATION_COMPLETE '+json.dumps(arms),flush=True)
        dist.barrier()
    finally:
        dist.destroy_process_group()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id',default='qualification_3x2_v1')
    parser.add_argument('--world-size',type=int,choices=(2,4,6,8),default=6)
    parser.add_argument('--node-rank',type=int,default=0)
    parser.add_argument('--hardware',action='store_true')
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    if args.hardware:hardware(args.world_size);return
    output=contract.run_path(args.run_id)
    if args.prepare:prepare(output,args.world_size,args.resume,args.node_rank)
    else:run(output,args.world_size,args.resume)


if __name__=='__main__':main()
