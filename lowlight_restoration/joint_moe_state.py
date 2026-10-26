"""Complete, immutable joint-MoE state; formal-only exact sampling resume."""
from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from safetensors.torch import save_file

from lowlight_restoration.blur_single_endpoint import sha256, rng_state, restore_rng, check_adam
from lowlight_restoration.rain_blur_image_moe import trainable_state, load_expert_state


def save_state(model, optimizer, output, *, step, mode, arm, config_sha, sources, rank_rng):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    base = output / f'step_{step:06d}'
    paths = {k: base.with_suffix(s) for k, s in (
        ('weights', '.safetensors'), ('optimizer', '.optimizer.pt'), ('receipt', '.resume.json'))}
    if any(p.exists() or p.with_name(p.name + '.partial').exists() for p in paths.values()):
        raise RuntimeError('preserve existing/partial checkpoint')
    check_adam(optimizer, step)
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    if len(optimizer.state) != len(names) or len(rank_rng) != 8:
        raise RuntimeError('incomplete optimizer or rank RNG state')
    state = trainable_state(model)
    if any(v.dtype != torch.float32 or not bool(torch.isfinite(v).all()) for v in state.values()):
        raise RuntimeError('invalid checkpoint weights')
    binding = {'step': step, 'mode': mode, 'arm': arm, 'config_sha256': config_sha,
               'source_sha256': sources, 'next_sample_position': step * 32, 'world_size': 8}
    partial = paths['weights'].with_name(paths['weights'].name + '.partial')
    save_file(state, str(partial), metadata={'step': str(step), 'arm': arm, 'mode': mode, 'config_sha256': config_sha})
    os.replace(partial, paths['weights'])
    del state
    partial = paths['optimizer'].with_name(paths['optimizer'].name + '.partial')
    with partial.open('xb') as handle:
        torch.save({**binding, 'optimizer': optimizer.state_dict(), 'rank_rng': rank_rng,
                    'parameter_names': names}, handle)
    os.replace(partial, paths['optimizer'])
    receipt = {**binding}
    for key in ('weights', 'optimizer'):
        receipt[key], receipt[key + '_sha256'] = paths[key].name, sha256(paths[key])
    partial = paths['receipt'].with_name(paths['receipt'].name + '.partial')
    with partial.open('x') as handle:
        json.dump(receipt, handle, sort_keys=True, indent=2)
        handle.write('\n')
    os.replace(partial, paths['receipt'])
    return receipt


def load_state(model, optimizer, receipt_path, *, arm, config_sha, sources, rank, total_steps):
    receipt_path = Path(receipt_path)
    receipt = json.loads(receipt_path.read_text())
    if not (receipt['mode'] == 'formal' and receipt['arm'] == arm
            and receipt['config_sha256'] == config_sha and receipt['source_sha256'] == sources
            and receipt['world_size'] == 8 and 0 <= rank < 8
            and 0 < receipt['step'] < total_steps
            and receipt['next_sample_position'] == receipt['step'] * 32):
        raise RuntimeError('only matching incomplete formal state may resume')
    for key in ('weights', 'optimizer'):
        name = receipt[key]
        if Path(name).name != name or sha256(receipt_path.parent / name) != receipt[key + '_sha256']:
            raise RuntimeError('checkpoint identity mismatch')
    saved = torch.load(receipt_path.parent / receipt['optimizer'], map_location='cpu', weights_only=True)
    for key in ('step', 'mode', 'arm', 'config_sha256', 'source_sha256', 'next_sample_position', 'world_size'):
        if saved[key] != receipt[key]:
            raise RuntimeError('receipt/optimizer binding mismatch')
    if saved['parameter_names'] != [n for n, p in model.named_parameters() if p.requires_grad] or len(saved['rank_rng']) != 8:
        raise RuntimeError('optimizer order/RNG mismatch')
    load_expert_state(model, receipt_path.parent / receipt['weights'])
    optimizer.load_state_dict(saved['optimizer'])
    check_adam(optimizer, receipt['step'])
    if len(optimizer.state) != len(saved['parameter_names']):
        raise RuntimeError('missing Adam states')
    restore_rng(saved['rank_rng'][rank])
    return receipt
