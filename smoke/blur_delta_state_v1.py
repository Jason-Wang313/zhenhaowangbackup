"""Immutable fresh-step checkpoints; all Adam/RNG state retained, no fallback."""
import os
from pathlib import Path
import time
import torch
from smoke.existing_flux_replay_contract_v1 import read, sha, write_new as write
from smoke.rain_native50_continue_state_v1 import (
    cpu_tree, parameter_hash, rng_state, restore_rng, trainable_state)


def save(output, model, optimizer, *, step, world, rank_rng, receipt_sha, history):
    if len(history) != step or len(rank_rng) != world or step < 0:
        raise ValueError('incomplete checkpoint')
    directory = Path(output) / 'steps'
    directory.mkdir(parents=True, exist_ok=True)
    state = trainable_state(model)
    payload = {'step': step, 'world': world, 'next_global_slot': step * 8,
               'model': state, 'parameter_order': list(state), 'optimizer': cpu_tree(optimizer.state_dict()),
               'rank_rng': cpu_tree(rank_rng), 'history': history, 'receipt_sha256': receipt_sha,
               'parameter_hash': parameter_hash(state)}
    file = directory / f'state_{step:06d}_{time.time_ns()}.pt'
    with file.open('xb') as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    record = {k: payload[k] for k in ('step', 'world', 'next_global_slot', 'receipt_sha256', 'parameter_hash')}
    record.update(file=str(file.relative_to(output)), sha256=sha(file))
    write(directory / f'step_{step:06d}.json', record)
    return record


def load(output, receipt_sha, world):
    directory = Path(output) / 'steps'
    records = [read(p) for p in sorted(directory.glob('step_*.json'))]
    if not records:
        return None
    if [r['step'] for r in records] != list(range(len(records))):
        raise ValueError('committed checkpoint gap')
    record = records[-1]
    file = Path(output) / record['file']
    if file.resolve().parent != directory.resolve() or sha(file) != record['sha256']:
        raise ValueError('corrupt checkpoint; no silent fallback')
    value = torch.load(file, map_location='cpu', weights_only=True)
    for k in ('step', 'world', 'next_global_slot', 'receipt_sha256', 'parameter_hash'):
        if value[k] != record[k]:
            raise ValueError('checkpoint receipt mismatch')
    step = value['step']
    if (value['world'] != world or value['receipt_sha256'] != receipt_sha
            or value['next_global_slot'] != step*8 or len(value['history']) != step
            or len(value['rank_rng']) != world or parameter_hash(value['model']) != value['parameter_hash']):
        raise ValueError('checkpoint scope changed')
    if any(t.dtype != torch.float32 or not torch.isfinite(t).all() for t in value['model'].values()):
        raise ValueError('invalid model state')
    states = value['optimizer']['state']
    if step and len(states) != len(value['model']):
        raise ValueError('missing Adam states')
    for state in states.values():
        if (float(state['step']) != step or any(state[k].dtype != torch.float32
                or not torch.isfinite(state[k]).all() for k in ('exp_avg', 'exp_avg_sq'))):
            raise ValueError('invalid Adam state')
    return value


def restore(model, optimizer, value, rank, device):
    parameters = {n: p for n, p in model.named_parameters() if p.requires_grad}
    if list(parameters) != value['parameter_order']:
        raise ValueError('parameter order changed')
    with torch.no_grad():
        for name, parameter in parameters.items():
            tensor = value['model'][name]
            if tensor.shape != parameter.shape or tensor.dtype != parameter.dtype:
                raise ValueError('parameter shape/dtype changed')
            parameter.copy_(tensor.to(parameter.device))
    optimizer.load_state_dict(value['optimizer'])
    restore_rng(value['rank_rng'][rank], device)
