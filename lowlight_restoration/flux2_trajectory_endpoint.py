"""Exact split-gradient FLUX.2 CFG trajectory utilities.

The prefix is always recomputed on the current model, but its state is detached.
Only the declared suffix keeps a graph. This preserves the deployed solver while
bounding activation memory; it is not a shortened solver surrogate.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Callable

import torch
from torch.utils.checkpoint import checkpoint


Tensor = torch.Tensor


def validate_trajectory_split(
    timesteps: Sequence[float], *, detached_prefix_steps: int, differentiable_suffix_steps: int
) -> None:
    transitions = len(timesteps) - 1
    if transitions <= 0:
        raise ValueError("trajectory must contain at least one transition")
    if detached_prefix_steps < 0 or differentiable_suffix_steps <= 0:
        raise ValueError("invalid detached/differentiable split")
    if detached_prefix_steps + differentiable_suffix_steps != transitions:
        raise ValueError(
            f"trajectory split covers {detached_prefix_steps + differentiable_suffix_steps} "
            f"transitions, expected {transitions}"
        )


def _cfg_velocity(
    model: Callable[..., Tensor],
    state: Tensor,
    state_ids: Tensor,
    context: Tensor,
    context_ids: Tensor,
    reference_tokens: Tensor | None,
    reference_ids: Tensor | None,
    timestep: float,
    guidance: float,
) -> Tensor:
    timestep_tensor = torch.full(
        (state.shape[0],), timestep, dtype=state.dtype, device=state.device
    )
    model_state = state
    model_ids = state_ids
    if reference_tokens is not None:
        if reference_ids is None:
            raise ValueError("reference tokens and ids must be provided together")
        model_state = torch.cat((state, reference_tokens), dim=1)
        model_ids = torch.cat((state_ids, reference_ids), dim=1)
    prediction = model(
        x=model_state,
        x_ids=model_ids,
        timesteps=timestep_tensor,
        ctx=context,
        ctx_ids=context_ids,
        guidance=None,
    )
    if reference_tokens is not None:
        prediction = prediction[:, : state.shape[1]]
    unconditional, conditional = prediction.chunk(2)
    guided = unconditional + guidance * (conditional - unconditional)
    return torch.cat((guided, guided), dim=0)


def denoise_cfg_split_gradient(
    model: Callable[..., Tensor],
    image_tokens: Tensor,
    image_ids: Tensor,
    context: Tensor,
    context_ids: Tensor,
    timesteps: Sequence[float],
    guidance: float,
    *,
    reference_tokens: Tensor | None = None,
    reference_ids: Tensor | None = None,
    detached_prefix_steps: int,
    differentiable_suffix_steps: int,
    checkpoint_suffix: bool,
) -> Tensor:
    """Run the exact CFG Euler trajectory with a detached prefix and live suffix."""

    validate_trajectory_split(
        timesteps,
        detached_prefix_steps=detached_prefix_steps,
        differentiable_suffix_steps=differentiable_suffix_steps,
    )
    state = torch.cat((image_tokens, image_tokens), dim=0)
    state_ids = torch.cat((image_ids, image_ids), dim=0)
    if reference_tokens is not None:
        if reference_ids is None:
            raise ValueError("reference tokens and ids must be provided together")
        reference_tokens = torch.cat((reference_tokens, reference_tokens), dim=0)
        reference_ids = torch.cat((reference_ids, reference_ids), dim=0)
    elif reference_ids is not None:
        raise ValueError("reference tokens and ids must be provided together")

    transitions = list(zip(timesteps[:-1], timesteps[1:]))
    with torch.no_grad():
        for current, previous in transitions[:detached_prefix_steps]:
            velocity = _cfg_velocity(
                model, state, state_ids, context, context_ids,
                reference_tokens, reference_ids, current, guidance,
            )
            state = state + (previous - current) * velocity
    state = state.detach()

    for current, previous in transitions[detached_prefix_steps:]:
        def velocity_fn(current_state: Tensor, *, current_t: float = current) -> Tensor:
            return _cfg_velocity(
                model, current_state, state_ids, context, context_ids,
                reference_tokens, reference_ids, current_t, guidance,
            )

        if checkpoint_suffix and torch.is_grad_enabled():
            velocity = checkpoint(velocity_fn, state, use_reentrant=False)
        else:
            velocity = velocity_fn(state)
        state = state + (previous - current) * velocity
    return state.chunk(2)[0]
