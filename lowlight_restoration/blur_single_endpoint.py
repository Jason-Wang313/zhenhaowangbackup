"""Matched Blur exposure stream and receipt-bound, resumable expert state."""
from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset

from lowlight_restoration.rain_full_data import RainSourceSchedule, crop_pair
from lowlight_restoration.rain_blur_image_moe import trainable_state, load_expert_state
from lowlight_restoration.flux2_endpoint_precision import assert_fp32_optimizer


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


class BlurOnlyDataset(Dataset):
    def __init__(self, root, rows, *, steps, rank, world=8, seed=2026102602, start_step=0):
        if world != 8 or not 0 <= rank < world or not 0 <= start_step < steps:
            raise ValueError('invalid single-node Blur schedule')
        self.root, self.rows, self.steps = root, rows, steps
        self.rank, self.world, self.seed, self.start_step = rank, world, seed, start_step
        self.schedule = RainSourceSchedule(rows, seed)
        self.verified = {}

    def __len__(self):
        return (self.steps - self.start_step) * 2

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        position = (self.start_step * 2 + index) * self.world + self.rank
        row = self.rows[self.schedule.index(position)]
        lq, gt = crop_pair(self.root, row, crop_size=512, position=position,
                          seed=self.seed, verified=self.verified)
        return {'lq': lq, 'gt': gt, 'record_id': row['record_id'],
                'source_group': row['source_group'], 'position': position}


def rng_state():
    return {'python': random.getstate(), 'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state['python'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] is not None:
        torch.cuda.set_rng_state(state['cuda'].cpu())


def check_adam(optimizer, step):
    assert_fp32_optimizer(optimizer)
    for state in optimizer.state.values():
        if int(state['step']) != step:
            raise RuntimeError('Adam step mismatch')
        if any(not bool(torch.isfinite(state[k]).all()) for k in ('exp_avg', 'exp_avg_sq')):
            raise RuntimeError('nonfinite Adam state')


def save_state(model, optimizer, output, *, step, mode, config_sha, sources, rank_rng,
               initial_parity_pass):
    """Write a completion receipt last. Partial checkpoints are never resumable."""
    from safetensors.torch import save_file
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    base = output / f'step_{step:06d}'
    weights = base.with_suffix('.safetensors')
    optim = base.with_suffix('.optimizer.pt')
    receipt = base.with_suffix('.resume.json')
    if any(p.exists() or p.with_name(p.name + '.partial').exists() for p in (weights, optim, receipt)):
        raise RuntimeError('preserve existing or partial checkpoint')
    state = trainable_state(model)
    if any(v.dtype != torch.float32 or not bool(torch.isfinite(v).all()) for v in state.values()):
        raise RuntimeError('invalid trainable checkpoint tensors')
    check_adam(optimizer, step)
    binding = {'step': step, 'mode': mode, 'config_sha256': config_sha, 'source_sha256': sources,
               'next_sample_position': step * 16, 'world_size': len(rank_rng),
               'initial_parity_pass': initial_parity_pass}
    temporary = weights.with_name(weights.name + '.partial')
    save_file(state, str(temporary), metadata={'mode': mode, 'step': str(step), 'config_sha256': config_sha})
    os.replace(temporary, weights)
    del state
    temporary = optim.with_name(optim.name + '.partial')
    with temporary.open('xb') as handle:
        torch.save({**binding, 'optimizer': optimizer.state_dict(), 'rank_rng': rank_rng,
                    'parameter_names': [n for n, p in model.named_parameters() if p.requires_grad]}, handle)
    os.replace(temporary, optim)
    payload = {**binding, 'weights': weights.name, 'weights_sha256': sha256(weights),
               'optimizer': optim.name, 'optimizer_sha256': sha256(optim)}
    temporary = receipt.with_name(receipt.name + '.partial')
    with temporary.open('x') as handle:
        json.dump(payload, handle, sort_keys=True, indent=2)
        handle.write('\n')
    os.replace(temporary, receipt)
    return payload


def load_state(model, optimizer, receipt, *, config_sha, sources, rank, world, total_steps):
    receipt = Path(receipt)
    meta = json.loads(receipt.read_text())
    if not (meta['mode'] == 'formal' and meta['config_sha256'] == config_sha
            and meta['source_sha256'] == sources and meta['world_size'] == world
            and meta['initial_parity_pass'] and 0 < meta['step'] < total_steps
            and meta['next_sample_position'] == meta['step'] * 16):
        raise RuntimeError('resume requires matching incomplete formal, never qualification')
    for key in ('weights', 'optimizer'):
        if Path(meta[key]).name != meta[key] or sha256(receipt.parent / meta[key]) != meta[key + '_sha256']:
            raise RuntimeError('resume artifact identity mismatch')
    state = torch.load(receipt.parent / meta['optimizer'], map_location='cpu', weights_only=True)
    for key in ('step', 'mode', 'config_sha256', 'source_sha256', 'next_sample_position', 'world_size', 'initial_parity_pass'):
        if state[key] != meta[key]:
            raise RuntimeError('optimizer/receipt binding mismatch')
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    if names != state['parameter_names'] or len(state['rank_rng']) != world:
        raise RuntimeError('optimizer parameter order or rank RNG mismatch')
    load_expert_state(model, receipt.parent / meta['weights'])
    optimizer.load_state_dict(state['optimizer'])
    check_adam(optimizer, meta['step'])
    if len(optimizer.state) != len(names):
        raise RuntimeError('incomplete Adam state')
    restore_rng(state['rank_rng'][rank])
    return meta
