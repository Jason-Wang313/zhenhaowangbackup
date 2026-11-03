"""Continuation branch global600..800; preserve parent Adam/RNG and every new step."""
from pathlib import Path
from smoke import existing_flux_replay_contract_v1 as c
from smoke import rain_native50_continue_state_v1 as base


def validate_state(state, record, receipt_hash, world=8):
    import torch
    step = state['step']
    if (step not in range(600, 801) or record['step'] != step or state['world'] != world
            or state['receipt_hash'] != receipt_hash or record['receipt_hash'] != receipt_hash
            or state['next_global_slot'] != step * 32 or len(state['history']) != step
            or len(state['rank_rng']) != world or state['may_seed_formal'] is not True
            or base.parameter_hash(state['model']) != record['parameter_hash']
            or state['parameter_hash'] != record['parameter_hash']):
        raise ValueError('invalid checkpoint identity/progress')
    if any(v.dtype != torch.float32 or not torch.isfinite(v).all() for v in state['model'].values()):
        raise ValueError('invalid FP32 LoRA')
    if state['parameter_order'] != list(state['model']):
        raise ValueError('parameter order changed')
    entries = state['optimizer']['state']
    groups = state['optimizer']['param_groups']
    if len(groups) != 1 or groups[0]['lr'] != 1e-5 or groups[0]['weight_decay'] != 0:
        raise ValueError('optimizer recipe changed')
    if len(entries) != len(state['model']):
        raise ValueError('missing Adam moments')
    for entry in entries.values():
        if float(entry['step']) != step or any(entry[k].dtype != torch.float32 or not torch.isfinite(entry[k]).all()
                                              for k in ('exp_avg', 'exp_avg_sq')):
            raise ValueError('invalid Adam state')


def load_latest(output, receipt_hash, world=8):
    import torch
    directory = Path(output) / 'steps'
    markers = sorted(directory.glob('step_*.json'))
    if not markers:
        return None, None
    records = [c.read(m) for m in markers]
    if [r['step'] for r in records] != list(range(600, 600 + len(records))):
        raise ValueError('checkpoint receipt gap')
    if [m.name for m in markers] != [f"step_{r['step']:06d}.json" for r in records]:
        raise ValueError('checkpoint receipt filename mismatch')
    record = records[-1]
    file = c.path(output, record['file'])
    if file.parent != directory.resolve() or c.sha(file) != record['sha256']:
        raise ValueError('checkpoint hash/path mismatch')
    state = torch.load(file, map_location='cpu', weights_only=True)
    validate_state(state, record, receipt_hash, world)
    return state, record


def save_step(output, model, optimizer, **kwargs):
    if kwargs['step'] not in range(600, 801):
        raise ValueError('outside authorized branch')
    return base.save_step(output, model, optimizer, **kwargs)
