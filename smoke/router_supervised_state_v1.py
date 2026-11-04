"""Every completed CPU router step is immutable and exactly resumable."""
from pathlib import Path
import os
import time
from smoke import rain_native50_continue_state_v1 as io

c = io.c


def model_state(model):
    return io.cpu_tree(dict(model.named_parameters()))


def save(output, model, optimizer, *, step, history, receipt_hash, arm):
    import torch
    if not 0 <= step <= 400 or len(history) != step or arm not in ('image_only', 'image_plus_vlm'):
        raise ValueError('invalid CPU step/arm')
    values = model_state(model)
    value = {'step': step, 'arm': arm, 'next_global_slot': step*32,
             'model': values, 'optimizer': io.cpu_tree(optimizer.state_dict()),
             'rng': io.cpu_tree(io.rng_state('cpu')), 'history': history,
             'parameter_hash': io.parameter_hash(values), 'receipt_hash': receipt_hash,
             'may_seed_formal': False, 'device': 'cpu'}
    output = Path(output); directory = output/'steps'; directory.mkdir(parents=True, exist_ok=True)
    name = f'steps/router_{step:06d}_{time.time_ns()}.pt'; file = c.path(output, name)
    with file.open('xb') as f:
        torch.save(value, f); f.flush(); os.fsync(f.fileno())
    fd = os.open(directory, os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)
    marker = {'step': step, 'arm': arm, 'file': name, 'sha256': c.sha(file),
              'parameter_hash': value['parameter_hash'], 'receipt_hash': receipt_hash}
    c.write_new(directory/f'step_{step:06d}.json', marker)
    return marker


def load(output, receipt_hash, arm):
    import torch
    output = Path(output); files = sorted((output/'steps').glob('step_*.json'))
    if not files:
        return None
    markers = [c.read(p) for p in files]
    if [m['step'] for m in markers] != list(range(len(markers))):
        raise ValueError('committed step gap; no silent rollback')
    marker = markers[-1]; file = c.path(output, marker['file'])
    if file.parent != (output/'steps').resolve() or c.sha(file) != marker['sha256']:
        raise ValueError('checkpoint hash/path changed')
    value = torch.load(file, map_location='cpu', weights_only=True)
    step = value['step']
    if (not 0 <= step <= 400 or marker['step'] != step or value['arm'] != arm or marker['arm'] != arm
            or value['receipt_hash'] != receipt_hash or marker['receipt_hash'] != receipt_hash
            or value['next_global_slot'] != step*32 or len(value['history']) != step
            or value['may_seed_formal'] is not False or value['device'] != 'cpu'
            or value['parameter_hash'] != io.parameter_hash(value['model'])
            or value['parameter_hash'] != marker['parameter_hash']):
        raise ValueError('CPU checkpoint identity/progress changed')
    if any(t.dtype != torch.float32 or not torch.isfinite(t).all() for t in value['model'].values()):
        raise ValueError('nonfinite/non-FP32 model')
    entries = value['optimizer']['state']
    if step and len(entries) != len(value['model']):
        raise ValueError('Adam state missing')
    for entry in entries.values():
        if float(entry['step']) != step or any(not torch.isfinite(entry[k]).all() for k in ('exp_avg', 'exp_avg_sq')):
            raise ValueError('invalid Adam state')
    return value


def restore(model, optimizer, value):
    if list(dict(model.named_parameters())) != list(value['model']):
        raise ValueError('parameter registration changed')
    model.load_state_dict(value['model'], strict=True)
    optimizer.load_state_dict(value['optimizer'])
    io.restore_rng(value['rng'], 'cpu')
