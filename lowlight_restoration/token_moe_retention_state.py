"""Immutable new-phase batch40 Adam/RNG state. Old batch32 receipts rejected."""
import json
import os
from pathlib import Path
import uuid
import random

import numpy as np
import torch
from safetensors.torch import save_file

from lowlight_restoration.blur_single_endpoint import check_adam
from lowlight_restoration.rain_blur_image_moe import trainable_state, load_expert_state
from smoke.token_residual_moe_v1 import sha, digest, write_once


def rng_state():
    state = np.random.get_state()
    return {'python': random.getstate(), 'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
            'numpy': [state[0], state[1].tolist(), *state[2:]]}


def restore_rng(state):
    random.setstate(state['python'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] is not None:
        torch.cuda.set_rng_state(state['cuda'].cpu())
    numpy_state = state['numpy']
    np.random.set_state((numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32), *numpy_state[2:]))


def save_state(model, optimizer, output, *, step, binding, rank_rng):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    check_adam(optimizer, step)
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    if step < 1 or len(rank_rng) != 8 or len(optimizer.state) != len(names):
        raise RuntimeError('incomplete phase optimizer/RNG')
    paths = {k: output / f'step_{step:06d}{ext}' for k, ext in
             [('weights', '.safetensors'), ('optimizer', '.optimizer.pt'), ('receipt', '.resume.json')]}
    if any(p.exists() for p in paths.values()):
        raise RuntimeError('refusing to overwrite prior checkpoint')
    contract = {'phase': 'five_single_retention_v1', 'mode': binding['mode'], 'step': step,
                'binding_sha256': digest(binding), 'world_size': 8, 'global_batch': 40,
                'next_sample_position': step * 40, 'parent_step': 2000}
    state = trainable_state(model)
    if any(v.dtype != torch.float32 or not bool(torch.isfinite(v).all()) for v in state.values()):
        raise RuntimeError('invalid trainable weights')
    for key in ('weights', 'optimizer'):
        temporary = paths[key].with_name(paths[key].name + '.' + uuid.uuid4().hex + '.partial')
        if key == 'weights':
            save_file(state, str(temporary), metadata={'binding_sha256': digest(binding), 'step': str(step)})
        else:
            with temporary.open('xb') as handle:
                torch.save({**contract, 'optimizer': optimizer.state_dict(), 'rank_rng': rank_rng,
                            'parameter_names': names}, handle)
                handle.flush(); os.fsync(handle.fileno())
        # Never silently replace an earlier/parallel artifact.
        os.link(temporary, paths[key]); temporary.unlink()
    receipt = {**contract}
    for key in ('weights', 'optimizer'):
        receipt[key], receipt[key + '_sha256'] = paths[key].name, sha(paths[key])
    write_once(paths['receipt'], receipt)
    return receipt


def resume(model, optimizer, path, binding, rank, total_steps):
    path = Path(path)
    receipt = json.loads(path.read_text())
    step = receipt.get('step', -1)
    expected = {'phase': 'five_single_retention_v1', 'mode': binding['mode'], 'step': step,
                'binding_sha256': digest(binding), 'world_size': 8, 'global_batch': 40,
                'next_sample_position': step * 40, 'parent_step': 2000}
    if not 0 < step <= total_steps or not 0 <= rank < 8 or any(receipt.get(k) != v for k, v in expected.items()):
        raise RuntimeError('only exact same-phase batch40 state may resume; qualification cannot seed formal')
    launch = json.loads((path.parent / 'launch.json').read_text())
    if launch['binding'] != binding:
        raise RuntimeError('resume source/config/data/cache mismatch')
    for key in ('weights', 'optimizer'):
        name = receipt[key]
        if Path(name).name != name or sha(path.parent / name) != receipt[key + '_sha256']:
            raise RuntimeError('resume tensor identity changed')
    saved = torch.load(path.parent / receipt['optimizer'], map_location='cpu', weights_only=True)
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    if (any(saved[k] != v for k, v in expected.items()) or saved['parameter_names'] != names
            or len(saved['rank_rng']) != 8):
        raise RuntimeError('optimizer/RNG binding mismatch')
    settings = lambda groups: [{k: v for k, v in g.items() if k != 'params'} for g in groups]
    if settings(optimizer.param_groups) != settings(saved['optimizer']['param_groups']):
        raise RuntimeError('optimizer configuration changed')
    load_expert_state(model, path.parent / receipt['weights'])
    optimizer.load_state_dict(saved['optimizer'])
    check_adam(optimizer, step)
    if len(optimizer.state) != len(names):
        raise RuntimeError('missing Adam states')
    restore_rng(saved['rank_rng'][rank])
    return step
