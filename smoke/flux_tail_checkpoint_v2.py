"""Crash-consistent per-step snapshots; immutable old steps remain recoverable."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile
import time


def sha(path):
    with Path(path).open('rb') as h:
        return hashlib.file_digest(h, 'sha256').hexdigest()


def sync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try: os.fsync(fd)
    finally: os.close(fd)


def atomic_write(path, writer, *, replace=False):
    """Write+fsync before publication. Never overwrite an immutable artifact."""
    path = Path(path)
    fd, tmp = tempfile.mkstemp(prefix='.'+path.name+'.', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as h:
            writer(h)
            h.flush(); os.fsync(h.fileno())
        if replace:
            os.replace(tmp, path)
        else:
            os.link(tmp, path)
            os.unlink(tmp)
        sync_directory(path.parent)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def atomic_json(path, data, *, replace=False):
    payload = (json.dumps(data, indent=2, sort_keys=True, allow_nan=False)+'\n').encode()
    atomic_write(path, lambda h: h.write(payload), replace=replace)


def cpu_tree(value):
    import torch
    if isinstance(value, torch.Tensor): return value.detach().cpu().clone()
    if isinstance(value, dict): return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list): return [cpu_tree(v) for v in value]
    if isinstance(value, tuple): return tuple(cpu_tree(v) for v in value)
    return value


def rng_state(device):
    import torch
    return dict(python=random.getstate(), torch_cpu=torch.get_rng_state(),
                torch_cuda=torch.cuda.get_rng_state(device) if torch.device(device).type=='cuda' else None)


def restore_rng(state, device):
    import torch
    random.setstate(state['python']); torch.set_rng_state(state['torch_cpu'])
    if state['torch_cuda'] is not None: torch.cuda.set_rng_state(state['torch_cuda'], device)


def commit_step(output, model, optimizer, *, step, world, rank_rng, receipt, log,
                initial_hash, elapsed_seconds, peak_memory_gib):
    import torch
    from smoke.qualify_foundir_flux_tail_energy_v1 import parameter_hash
    output = Path(output); directory = output/'steps'; directory.mkdir(exist_ok=True)
    if len(log)!=step or len(rank_rng)!=world: raise ValueError('incomplete progress/RNG')
    name = f'energy_step_{step:06d}_{time.time_ns()}.pt'
    state = dict(format_version=2, model=cpu_tree(model.state_dict()),
        optimizer=cpu_tree(optimizer.state_dict()), step=step, next_global_slot=step*32,
        scheduler=dict(kind='explicit_linear_warmup',warmup_steps=5,last_step=step,
                       base_lr=receipt['spec']['learning_rate']), world_size=world,
        rank_rng=cpu_tree(rank_rng), training_log=log, initial_parameter_sha256=initial_hash,
        parameter_sha256=parameter_hash(model), elapsed_training_seconds=elapsed_seconds,
        peak_memory_gib=peak_memory_gib, may_seed_formal=False,
        config_sha256=receipt['config_sha256'], source_sha256=receipt['source_sha256'])
    atomic_write(directory/name, lambda h: torch.save(state,h))
    record = dict(step=step,file='steps/'+name,sha256=sha(directory/name),
                  parameter_sha256=state['parameter_sha256'],world_size=world,
                  next_global_slot=step*32,config_sha256=receipt['config_sha256'])
    # A committed receipt, not a possibly stale latest pointer, defines durability.
    atomic_json(directory/f'step_{step:06d}.json', record)
    atomic_json(output/'latest.json', record, replace=True)
    return record


def load_latest(output, receipt, world):
    import torch
    output = Path(output); files = sorted((output/'steps').glob('step_*.json'))
    if not files: return None, None
    record = json.loads(files[-1].read_text())
    file = output/record['file']
    if file.resolve().parent != (output/'steps').resolve() or sha(file)!=record['sha256']:
        raise RuntimeError('checkpoint path/hash invalid; do not silently roll back')
    state = torch.load(file, map_location='cpu', weights_only=True)
    step = state['step']
    if (state['format_version']!=2 or not 0<=step<=25 or record['step']!=step
            or state['world_size']!=world or record['world_size']!=world
            or state['next_global_slot']!=32*step or record['next_global_slot']!=32*step
            or state['config_sha256']!=receipt['config_sha256']
            or state['source_sha256']!=receipt['source_sha256']
            or state['may_seed_formal'] or len(state['training_log'])!=step
            or len(state['rank_rng'])!=world
            or state['scheduler']!=dict(kind='explicit_linear_warmup',warmup_steps=5,
                last_step=step,base_lr=receipt['spec']['learning_rate'])):
        raise RuntimeError('checkpoint protocol/progress/world mismatch')
    digest=hashlib.sha256()
    for name,tensor in state['model'].items():
        digest.update(name.encode()); digest.update(tensor.contiguous().numpy().tobytes())
    if digest.hexdigest()!=record['parameter_sha256'] or digest.hexdigest()!=state['parameter_sha256']:
        raise RuntimeError('saved parameters do not match receipt')
    for item in state['optimizer']['state'].values():
        if int(item['step'])!=step or any(not bool(item[k].isfinite().all()) for k in ('exp_avg','exp_avg_sq')):
            raise RuntimeError('incomplete/nonfinite Adam state')
    if step and not state['optimizer']['state']: raise RuntimeError('missing Adam state')
    return state, record
