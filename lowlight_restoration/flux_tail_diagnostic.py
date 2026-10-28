"""Frozen FLUX input gradients; no learned energy or optimizer in this module.

The continuous RGB objective is a diagnostic surrogate. Deployment evaluation
must retain BOTH original uint8 roundings. Preview and rollout are distinct.
"""
from __future__ import annotations

import torch
from torch.utils.checkpoint import checkpoint

from lowlight_restoration.flux2_trajectory_endpoint import _cfg_velocity


def preview(state, velocity, timestep):
    # This repository runs from t=1 (noise) to t=0 (clean).
    return state - timestep * velocity


def continuous_blend(observation, raw):
    return (observation.float() + .3 * (raw.float().clamp(-1, 1) - observation.float())).clamp(-1, 1)


def rms_direction(gradient):
    if not bool(torch.isfinite(gradient).all()):
        raise RuntimeError('nonfinite input gradient')
    rms = gradient.float().square().mean().sqrt()
    if float(rms) <= 1e-12:
        raise RuntimeError('zero/negligible input gradient; diagnostic cannot proceed')
    return gradient.float() / rms, float(rms)


def finite_difference(objective, state, gradient, epsilons):
    """FP32 directional check, not a BF16 finite-difference equality claim."""
    direction, _ = rms_direction(gradient)
    analytic = float((gradient.float() * direction).sum())
    rows = []
    with torch.no_grad():
        for epsilon in epsilons:
            plus = float(objective(state + epsilon * direction))
            minus = float(objective(state - epsilon * direction))
            numerical = (plus - minus) / (2 * epsilon)
            error = abs(numerical - analytic) / max(abs(analytic), abs(numerical), 1e-8)
            rows.append(dict(epsilon=epsilon, analytic=analytic, numerical=numerical,
                             relative_error=error, positive_direction=numerical > 0))
    return rows


def checkpoint_blocks(model, counters):
    """Checkpoint frozen eval-mode blocks too: input gradients still need memory.

    Original bound methods remain the implementation. Count physical block
    calls, including nested checkpoint recomputation, separately from forwards.
    """
    for block in [*model.double_blocks, *model.single_blocks]:
        if getattr(block, '_tail_checkpointed', False):
            raise RuntimeError('block already wrapped')
        original = block.forward_kv_extract

        def wrapped(*args, _original=original, **kwargs):
            def call(*values, **options):
                counters['flux_block_forward'] = counters.get('flux_block_forward', 0) + 1
                return _original(*values, **options)
            if torch.is_grad_enabled():
                return checkpoint(call, *args, use_reentrant=False, **kwargs)
            return call(*args, **kwargs)

        block.forward_kv_extract = wrapped
        block._tail_checkpointed = True


class Tail:
    """Single-image state, original two-branch CFG, real shifted times.

    BF16 velocity evaluation keeps the original timestep quantization; FP32
    latent accumulation is an EXPLICIT separate control, not the old baseline.
    Non-reentrant checkpointing bounds memory without dropping the Jacobian.
    """
    def __init__(self, model, ae, ids, reference, reference_ids, context, context_ids, schedule):
        self.model, self.ae = model, ae
        self.ids, self.reference, self.reference_ids = ids, reference, reference_ids
        self.context, self.context_ids, self.schedule = context, context_ids, schedule
        self.bf16 = True
        self.calls = dict(flux_forward=0, flux_block_forward=0, vae_decode=0, objective_backward=0)

    def velocity(self, state, timestep):
        def forward(value):
            self.calls['flux_forward'] += 1  # includes checkpoint recomputations
            dtype = torch.bfloat16 if self.bf16 else torch.float32
            value = value.to(dtype)
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=self.bf16 and value.is_cuda):
                return _cfg_velocity(
                    self.model, torch.cat((value, value)), torch.cat((self.ids, self.ids)),
                    self.context.to(dtype), self.context_ids,
                    torch.cat((self.reference, self.reference)).to(dtype),
                    torch.cat((self.reference_ids, self.reference_ids)), timestep, 1.).chunk(2)[0]
        if torch.is_grad_enabled() and state.requires_grad:
            return checkpoint(forward, state, use_reentrant=False)
        return forward(state)

    def rollout(self, state, start, *, capture=False):
        states = {}
        for index in range(start, len(self.schedule)-1):
            if capture:
                states[index] = state.detach().clone()
            current, following = self.schedule[index:index+2]
            state = state + (following-current) * self.velocity(state, current)
        return (state, states) if capture else state

    def decode(self, state):
        from flux2.sampling import scatter_ids
        def forward(value):
            self.calls['vae_decode'] += 1
            latent = torch.cat(scatter_ids(value, self.ids)).squeeze(2)
            return self.ae.decode(latent).float().clamp(-1, 1)
        if torch.is_grad_enabled() and state.requires_grad:
            return checkpoint(forward, state, use_reentrant=False)
        return forward(state)

    def image(self, state, start, mode):
        if mode == 'preview_full':
            clean = preview(state, self.velocity(state, self.schedule[start]), self.schedule[start])
        elif mode == 'preview_skip':
            with torch.no_grad():
                velocity = self.velocity(state, self.schedule[start])
            clean = preview(state, velocity, self.schedule[start])
        elif mode == 'rollout_full':
            clean = self.rollout(state, start)
        else:
            raise ValueError(mode)
        return self.decode(clean)
