"""Global8 topology migration and immutable step1792..9472 continuation state."""
from pathlib import Path
import torch
import random
from functools import lru_cache
from smoke import blur_delta_state_v1 as original
from smoke.rain_native50_continue_state_v1 import cpu_tree
from smoke.existing_flux_replay_contract_v1 import read, sha

START, END, WORLD = 1792, 9472, 4
EPOCH_STEPS, VIEW_COUNT, EPOCHS = 1920, 15360, 4
SNAPSHOTS = (START, START+1920, START+3840, START+5760, END)


@lru_cache(maxsize=4)
def epoch_order(epoch):
    if not 0 <= epoch < EPOCHS: raise ValueError('outside fixed four epochs')
    order = list(range(VIEW_COUNT))
    random.Random(62017+epoch).shuffle(order)
    return tuple(order)


def indices(step, rank, world=WORLD):
    if not START <= step < END or not 1 <= world <= 8 or 8%world or not 0 <= rank < world:
        raise ValueError('outside fixed global8 continuation')
    epoch, batch = divmod(step-START, EPOCH_STEPS)
    return list(epoch_order(epoch)[batch*8:(batch+1)*8][rank::world])


def validate(value, marker, receipt_sha, world=WORLD):
    step = value['step']
    for key in ('step', 'world', 'next_global_slot', 'receipt_sha256', 'parameter_hash'):
        if value[key] != marker[key]: raise ValueError('checkpoint receipt mismatch')
    if (not START <= step <= END or value['world'] != world or len(value['rank_rng']) != world
            or value['receipt_sha256'] != receipt_sha or value['next_global_slot'] != step*8
            or len(value['history']) != step or list(value['model']) != value['parameter_order']
            or original.parameter_hash(value['model']) != value['parameter_hash']):
        raise ValueError('checkpoint scope changed')
    if any(t.dtype != torch.float32 or not torch.isfinite(t).all() for t in value['model'].values()):
        raise ValueError('invalid FP32 parameters')
    groups = value['optimizer']['param_groups']; states = value['optimizer']['state']
    if (len(groups) != 1 or groups[0]['lr'] != 1e-5 or groups[0]['weight_decay'] != 0
            or len(states) != len(value['model'])): raise ValueError('Adam schema changed')
    for state in states.values():
        if float(state['step']) != step or any(state[k].dtype != torch.float32 or not torch.isfinite(state[k]).all()
                                               for k in ('exp_avg', 'exp_avg_sq')):
            raise ValueError('invalid Adam state')


def load_step(output, step, receipt_sha, world=WORLD):
    output = Path(output); marker = read(output/'steps'/f'step_{step:06d}.json'); file = output/marker['file']
    if file.resolve().parent != (output/'steps').resolve() or sha(file) != marker['sha256']:
        raise ValueError('checkpoint file corrupt; no fallback')
    value = torch.load(file, map_location='cpu', weights_only=True)
    validate(value, marker, receipt_sha, world)
    if value['step'] != step: raise ValueError('wrong step file')
    return value, marker


def latest(output, receipt_sha, world=WORLD):
    files = sorted((Path(output)/'steps').glob('step_*.json'))
    if not files: return None, None
    steps = [read(p)['step'] for p in files]
    if steps != list(range(START, START+len(steps))): raise ValueError('checkpoint gap; no rollback')
    return load_step(output, steps[-1], receipt_sha, world)


def restore_parameters_adam(model, optimizer, value):
    parameters = {n: p for n, p in model.named_parameters() if p.requires_grad}
    if list(parameters) != value['parameter_order']: raise ValueError('parameter order differs')
    with torch.no_grad():
        for name, p in parameters.items():
            v = value['model'][name]
            if p.shape != v.shape or p.dtype != v.dtype: raise ValueError('parameter schema differs')
            p.copy_(v.to(p.device))
    optimizer.load_state_dict(value['optimizer'])
    actual = cpu_tree(optimizer.state_dict())
    if actual['param_groups'] != value['optimizer']['param_groups']: raise ValueError('Adam options differ')
    for key, state in actual['state'].items():
        for field, tensor in state.items():
            if not torch.equal(tensor, value['optimizer']['state'][key][field]): raise ValueError('Adam restore changed values')


def restore(model, optimizer, value, rank, device):
    restore_parameters_adam(model, optimizer, value)
    if not 0 <= rank < value['world']: raise ValueError('rank outside saved world')
    original.restore_rng(value['rank_rng'][rank], device)
