"""Immutable resumable state for the qualified PPT recipe's first full-data stage."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import random

import numpy as np
import torch
import torch.distributed as dist

from smoke.dual_retry_runtime import ROOT, sha, sources

CONFIG = ROOT / 'configs/foundir_novel_ppt_full_data_stage1_v1.json'
MODE = 'full_data_stage1'
EXTRA_SOURCES = (
    'smoke/novel_ppt_full_data_state_v1.py',
    'smoke/train_foundir_novel_ppt_full_data_stage1_v1_ddp.py',
    'smoke/run_foundir_novel_ppt_full_data_stage1_v1_8gpu.sh',
    'smoke/start_foundir_novel_ppt_full_data_stage1_v1_tmux.sh',
)


def binding():
    return sources(EXTRA_SOURCES)


def learning_rate(spec, step):
    if not 1 <= step <= spec['primary_steps']:
        raise ValueError('step outside parent LR horizon')
    if step <= spec['warmup_steps']:
        return spec['learning_rate'] * step / spec['warmup_steps']
    fraction = (step-spec['warmup_steps'])/(spec['primary_steps']-spec['warmup_steps'])
    return spec['minimum_learning_rate'] + .5*(spec['learning_rate']-spec['minimum_learning_rate'])*(1+math.cos(math.pi*fraction))


def capture_rng(device):
    return {'cpu': torch.get_rng_state(), 'python': random.getstate(),
            'numpy': np.random.get_state(),
            'cuda': torch.cuda.get_rng_state(device).cpu() if device.type == 'cuda' else None}


def restore_rng(state, device):
    torch.set_rng_state(state['cpu'])
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    if device.type == 'cuda':
        if state['cuda'] is None:
            raise RuntimeError('missing CUDA RNG')
        torch.cuda.set_rng_state(state['cuda'], device)


def validate_state(state, config_sha, source_sha):
    if (state.get('mode') != MODE or state.get('world_size') != 8
            or state.get('config_sha256') != config_sha or state.get('source_sha256') != source_sha):
        raise RuntimeError('wrong mode/config/source; qualification and trainfit cannot seed full data')
    step = state.get('step')
    if not isinstance(step, int) or not 1 <= step <= 1706 or state.get('next_sample_step') != step+1:
        raise RuntimeError('invalid optimizer/sample step')
    if len(state.get('rank_rng', [])) != 8:
        raise RuntimeError('missing per-rank RNG states')
    for value in state['model'].values():
        if not bool(torch.isfinite(value).all()):
            raise RuntimeError('nonfinite model state')
    adam = state['optimizer']
    param_ids = [p for group in adam['param_groups'] for p in group['params']]
    if len(set(param_ids)) != len(param_ids) or set(param_ids) != set(adam['state']):
        raise RuntimeError('incomplete Adam state')
    for value in adam['state'].values():
        if int(value['step']) != step:
            raise RuntimeError('Adam step does not match checkpoint')
        for name in ('exp_avg', 'exp_avg_sq'):
            if name not in value or not bool(torch.isfinite(value[name]).all()):
                raise RuntimeError('invalid Adam moment')


def read_resume(path, config_sha, source_sha):
    path = Path(path).resolve()
    cfg = json.loads(CONFIG.read_text())
    if not path.is_relative_to((ROOT/cfg['output_root']).resolve()):
        raise RuntimeError('only full-data stage1 receipt paths are accepted')
    receipt = json.loads(path.read_text())
    if (receipt.get('mode') != MODE or receipt.get('config_sha256') != config_sha
            or receipt.get('source_sha256') != source_sha):
        raise RuntimeError('resume receipt config/source/mode changed')
    name = receipt['checkpoint']
    if Path(name).name != name or name != f"step_{receipt['step']:06d}.pt":
        raise RuntimeError('invalid checkpoint name')
    checkpoint = path.parent/name
    if sha(checkpoint) != receipt['checkpoint_sha256']:
        raise RuntimeError('checkpoint hash mismatch')
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    validate_state(state, config_sha, source_sha)
    if state['step'] != receipt['step']:
        raise RuntimeError('receipt/checkpoint step mismatch')
    return state, receipt


def apply_state(model, optimizer, state, rank, device):
    if list(dict(model.named_parameters())) != state['parameter_names']:
        raise RuntimeError('optimizer parameter order changed')
    if len(optimizer.param_groups) != len(state['optimizer']['param_groups']):
        raise RuntimeError('optimizer groups changed')
    for expected, saved in zip(optimizer.param_groups, state['optimizer']['param_groups'], strict=True):
        for key in expected.keys() - {'lr', 'params'}:
            if expected[key] != saved.get(key):
                raise RuntimeError('optimizer setting changed: '+key)
    model.load_state_dict(state['model'], strict=True)
    optimizer.load_state_dict(state['optimizer'])
    restore_rng(state['rank_rng'][rank], device)


def save(output, model, optimizer, step, config_sha, source_sha, device, rank):
    rngs = [None]*8
    dist.all_gather_object(rngs, capture_rng(device))
    response = [None, None]
    if rank == 0:
        try:
            if sha(CONFIG) != config_sha or binding() != source_sha:
                raise RuntimeError('source/config changed during live run')
            state = {'mode': MODE, 'world_size': 8, 'step': step, 'next_sample_step': step+1,
                     'config_sha256': config_sha, 'source_sha256': source_sha,
                     'parameter_names': list(dict(model.named_parameters())),
                     'model': {k: v.detach().cpu() for k, v in model.state_dict().items()},
                     'optimizer': optimizer.state_dict(), 'rank_rng': rngs}
            validate_state(state, config_sha, source_sha)
            path = output/f'step_{step:06d}.pt'
            temporary = path.with_suffix('.partial')
            if path.exists():
                raise RuntimeError('preserve existing checkpoint')
            with temporary.open('xb') as stream:
                torch.save(state, stream)
                stream.flush(); os.fsync(stream.fileno())
            os.replace(temporary, path)
            receipt = {'mode': MODE, 'step': step, 'next_sample_step': step+1,
                       'config_sha256': config_sha, 'source_sha256': source_sha,
                       'checkpoint': path.name, 'checkpoint_sha256': sha(path)}
            receipt_path = output/f'step_{step:06d}.json'
            if receipt_path.exists():
                raise RuntimeError('preserve existing receipt')
            temporary_json = receipt_path.with_suffix('.json.partial')
            with temporary_json.open('x') as stream:
                json.dump(receipt, stream, indent=2, sort_keys=True, allow_nan=False)
                stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
            os.replace(temporary_json, receipt_path)
            response[1] = {**receipt, 'receipt': str(receipt_path.relative_to(ROOT))}
        except Exception as exc:
            response[0] = repr(exc)
    dist.broadcast_object_list(response, src=0)
    if response[0]:
        raise RuntimeError(response[0])
    return response[1]
