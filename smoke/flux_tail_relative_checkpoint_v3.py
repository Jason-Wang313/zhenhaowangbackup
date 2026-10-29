"""Immutable per-step resume for the separate V3 qualification/train-fit runs."""
from pathlib import Path
import json
import time
import torch
from smoke.flux_tail_checkpoint_v2 import atomic_write,atomic_json,cpu_tree,sha
from smoke.qualify_foundir_flux_tail_energy_v1 import parameter_hash


def commit(output,model,optimizer,step,receipt,rank_states,log,elapsed_seconds):
    directory=Path(output)/'steps';directory.mkdir(exist_ok=True)
    if len(log)!=step or len(rank_states)!=receipt['world_size']:
        raise ValueError('incomplete log/rank states')
    hashes={r['parameter_sha256'] for r in rank_states}
    if hashes!={parameter_hash(model)}:raise RuntimeError('DDP parameter mismatch')
    state=dict(format_version=3,step=step,next_global_slot=step*32,
        model=cpu_tree(model.state_dict()),optimizer=cpu_tree(optimizer.state_dict()),
        rank_states=cpu_tree(rank_states),training_log=log,elapsed_training_seconds=elapsed_seconds,
        protocol_sha256=receipt['protocol_sha256'],world_size=receipt['world_size'],
        parameter_sha256=parameter_hash(model),mode=receipt['mode'],
        scheduler=dict(warmup_steps=5,last_step=step,base_lr=receipt['spec']['learning_rate']),
        may_seed_other_run=False)
    file=f'steps/energy_{step:06d}_{time.time_ns()}.pt'
    atomic_write(Path(output)/file,lambda h:torch.save(state,h))
    record=dict(step=step,file=file,sha256=sha(Path(output)/file),
                protocol_sha256=receipt['protocol_sha256'])
    atomic_json(directory/f'step_{step:06d}.json',record)
    atomic_json(Path(output)/'latest.json',record,replace=True)
    return record


def load(output,receipt,requested_step=None):
    directory=Path(output)/'steps';files=sorted(directory.glob('step_*.json'))
    if requested_step is not None:
        files=[directory/f'step_{requested_step:06d}.json']
        if not files[0].exists():raise RuntimeError('requested durable step missing')
    if not files:return None,None
    record=json.loads(files[-1].read_text());file=Path(output)/record['file']
    if file.resolve().parent!=directory.resolve() or sha(file)!=record['sha256']:
        raise RuntimeError('invalid checkpoint path/hash; no silent fallback')
    state=torch.load(file,map_location='cpu',weights_only=True)
    step=state['step'];spec=receipt['spec'];world=receipt['world_size']
    if (state['format_version']!=3 or not 0<=step<=spec['modes'][receipt['mode']]['steps']
        or state['step']!=record['step'] or state['next_global_slot']!=step*32
        or state['mode']!=receipt['mode'] or state['world_size']!=world
        or state['protocol_sha256']!=receipt['protocol_sha256']
        or record['protocol_sha256']!=receipt['protocol_sha256']
        or state['may_seed_other_run'] or len(state['training_log'])!=step
        or len(state['rank_states'])!=world
        or state['scheduler']!=dict(warmup_steps=5,last_step=step,base_lr=spec['learning_rate'])):
        raise RuntimeError('checkpoint run/world/progress mismatch')
    import hashlib
    digest=hashlib.sha256()
    for name,tensor in state['model'].items():
        if not bool(tensor.isfinite().all()):raise RuntimeError('nonfinite model')
        digest.update(name.encode());digest.update(tensor.contiguous().numpy().tobytes())
    if digest.hexdigest()!=state['parameter_sha256']:raise RuntimeError('parameter hash mismatch')
    if any(r['parameter_sha256']!=state['parameter_sha256'] for r in state['rank_states']):
        raise RuntimeError('saved rank parameter mismatch')
    if step and not state['optimizer']['state']:raise RuntimeError('missing Adam')
    for item in state['optimizer']['state'].values():
        if int(item['step'])!=step or any(not bool(item[k].isfinite().all()) for k in ('exp_avg','exp_avg_sq')):
            raise RuntimeError('invalid Adam state')
    return state,record
