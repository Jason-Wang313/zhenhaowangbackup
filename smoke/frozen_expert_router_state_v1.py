"""Immutable router-only Adam/RNG snapshots, every completed optimizer step."""
from pathlib import Path
import os
import time
from smoke import rain_native50_continue_state_v1 as io
c = io.c


def model_state(model):
    return io.cpu_tree({n: p for n, p in model.named_parameters()})


def save(output, model, optimizer, *, step, history, rank_rng, receipt_hash):
    import torch
    if not 0 <= step <= 408 or len(history) != step or len(rank_rng) != 8:
        raise ValueError('incomplete router checkpoint')
    values = model_state(model)
    state = {'step': step, 'world': 8, 'model': values, 'optimizer': io.cpu_tree(optimizer.state_dict()),
             'rank_rng': io.cpu_tree(rank_rng), 'history': history, 'next_global_slot': step * 32,
             'receipt_hash': receipt_hash, 'parameter_hash': io.parameter_hash(values), 'may_seed_formal': False}
    directory = Path(output) / 'steps'; directory.mkdir(exist_ok=True)
    name = f'steps/router_{step:06d}_{time.time_ns()}.pt'; path = c.path(output, name)
    with path.open('xb') as f:
        torch.save(state, f); f.flush(); os.fsync(f.fileno())
    fd = os.open(directory, os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)
    record = {'step': step, 'file': name, 'sha256': c.sha(path), 'parameter_hash': state['parameter_hash'], 'receipt_hash': receipt_hash}
    c.write_new(directory / f'step_{step:06d}.json', record)
    return record


def load(output, receipt_hash):
    import torch
    markers = sorted((Path(output) / 'steps').glob('step_*.json'))
    if not markers:
        return None
    records = [c.read(m) for m in markers]
    if [r['step'] for r in records] != list(range(len(records))):
        raise ValueError('missing checkpoint receipts')
    record = records[-1]; path = c.path(output, record['file'])
    if path.parent != (Path(output) / 'steps').resolve() or c.sha(path) != record['sha256']:
        raise ValueError('checkpoint corrupt; do not fall back silently')
    state = torch.load(path, map_location='cpu', weights_only=True)
    step = state['step']
    if (not 0 <= step <= 408 or record['step'] != step or len(state['history']) != step
            or state['world'] != 8 or len(state['rank_rng']) != 8 or state['next_global_slot'] != step * 32
            or record['receipt_hash'] != receipt_hash or state['receipt_hash'] != receipt_hash
            or io.parameter_hash(state['model']) != record['parameter_hash']
            or state['parameter_hash'] != record['parameter_hash'] or state['may_seed_formal']):
        raise ValueError('checkpoint identity/progress changed')
    if any(v.dtype != torch.float32 or not torch.isfinite(v).all() for v in state['model'].values()):
        raise ValueError('router parameters must remain finite FP32')
    entries = state['optimizer']['state']
    if step and len(entries) != len(state['model']):
        raise ValueError('missing Adam state')
    for value in entries.values():
        if float(value['step']) != step or any(not torch.isfinite(value[k]).all() for k in ('exp_avg', 'exp_avg_sq')):
            raise ValueError('invalid Adam moments')
    return state


def restore(model, optimizer, state, rank, device):
    if list(dict(model.named_parameters())) != list(state['model']):
        raise ValueError('router parameter schema changed')
    model.load_state_dict(state['model'], strict=True)
    optimizer.load_state_dict(state['optimizer'])
    io.restore_rng(state['rank_rng'][rank], device)
