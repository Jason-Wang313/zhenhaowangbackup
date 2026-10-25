"""Strict Adam restoration and absolute-position Rain data continuation."""
from __future__ import annotations

import torch

from lowlight_restoration.rain_full_data import RainFullDataset


class RainResumeDataset(RainFullDataset):
    def __init__(self, *args, start_step, **kwargs):
        super().__init__(*args, **kwargs)
        if start_step < 0:
            raise ValueError('negative resume step')
        self.start_step = start_step

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        return super().__getitem__(index + self.start_step * 4)


def validate_adam_payload(payload, *, config_sha256, step, shapes, lr=1e-5, fused=True):
    if payload['step'] != step or payload['config_sha256'] != config_sha256:
        raise RuntimeError('optimizer step/config mismatch')
    state = payload['optimizer']
    if len(state['param_groups']) != 1:
        raise RuntimeError('unexpected optimizer groups')
    group = state['param_groups'][0]
    expected = dict(lr=lr, betas=(.9, .999), eps=1e-8, weight_decay=0,
                    amsgrad=False, maximize=False, foreach=None, capturable=False,
                    differentiable=False, fused=fused, decoupled_weight_decay=True)
    if {key: value for key, value in group.items() if key != 'params'} != expected:
        raise RuntimeError('optimizer options changed')
    indices = group['params']
    if indices != list(range(len(shapes))) or set(state['state']) != set(indices):
        raise RuntimeError('missing or reordered optimizer parameters')
    for index, shape in zip(indices, shapes):
        entry = state['state'][index]
        if set(entry) != {'step', 'exp_avg', 'exp_avg_sq'}:
            raise RuntimeError('missing or unexpected Adam state')
        if entry['step'].numel() != 1 or float(entry['step']) != step:
            raise RuntimeError('Adam step counter mismatch')
        for key in ('exp_avg', 'exp_avg_sq'):
            tensor = entry[key]
            if tuple(tensor.shape) != tuple(shape) or tensor.dtype != torch.float32:
                raise RuntimeError('Adam moment shape/dtype mismatch')
            if not bool(torch.isfinite(tensor).all()):
                raise RuntimeError('nonfinite Adam moment')
        if bool((entry['exp_avg_sq'] < 0).any()):
            raise RuntimeError('negative Adam second moment')


def restore_adam(optimizer, payload, *, config_sha256, step, fused=True):
    parameters = optimizer.param_groups[0]['params']
    validate_adam_payload(payload, config_sha256=config_sha256, step=step,
                          shapes=[p.shape for p in parameters], fused=fused)
    optimizer.load_state_dict(payload['optimizer'])
    for index, parameter in enumerate(parameters):
        loaded = optimizer.state[parameter]
        saved = payload['optimizer']['state'][index]
        for name in ('step', 'exp_avg', 'exp_avg_sq'):
            if not torch.equal(loaded[name].cpu(), saved[name].cpu()):
                raise RuntimeError('optimizer restore changed saved values')
    if optimizer.param_groups[0]['lr'] != 1e-5:
        raise RuntimeError('restored learning rate changed')
