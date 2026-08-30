"""Reference-conditioned rectified-flow batch construction for FLUX.2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor


@dataclass(frozen=True)
class FlowBatch:
    model_tokens: Tensor
    model_ids: Tensor
    target_velocity: Tensor
    target_token_count: int
    timesteps: Tensor


def build_reference_flow_batch(
    clean_latents: Tensor,
    degraded_latents: Tensor,
    noise: Tensor,
    timesteps: Tensor,
    pack_image: Callable[[Tensor], tuple[Tensor, Tensor]],
    reference_time_coordinate: int = 10,
) -> FlowBatch:
    """Construct x_t and reference tokens using the official FLUX.2 token layout.

    FLUX.2 sampling integrates from t=1 noise to t=0 data. The straight path is
    x_t=(1-t)*clean+t*noise, so the velocity target is noise-clean. The degraded
    image is not the flow target; it is a reference-image condition with the
    same time-coordinate offset used by the official inference code.
    """

    if clean_latents.shape != degraded_latents.shape or clean_latents.shape != noise.shape:
        raise ValueError(
            "clean, degraded, and noise latent shapes must match: "
            f"{clean_latents.shape}, {degraded_latents.shape}, {noise.shape}"
        )
    if timesteps.ndim != 1 or timesteps.shape[0] != clean_latents.shape[0]:
        raise ValueError("timesteps must have shape [batch]")
    if not torch.isfinite(timesteps).all() or not ((timesteps >= 0) & (timesteps <= 1)).all():
        raise ValueError("timesteps must be finite and in [0, 1]")
    if not 0 <= reference_time_coordinate:
        raise ValueError("reference_time_coordinate must be non-negative")

    broadcast_t = timesteps.view(-1, *([1] * (clean_latents.ndim - 1))).to(clean_latents)
    noisy_latents = (1.0 - broadcast_t) * clean_latents + broadcast_t * noise
    velocity_latents = noise - clean_latents

    noisy_tokens, noisy_ids = pack_image(noisy_latents)
    reference_tokens, reference_ids = pack_image(degraded_latents)
    velocity_tokens, _ = pack_image(velocity_latents)
    reference_ids = reference_ids.clone()
    reference_ids[..., 0] = reference_time_coordinate

    target_token_count = noisy_tokens.shape[1]
    return FlowBatch(
        model_tokens=torch.cat([noisy_tokens, reference_tokens], dim=1),
        model_ids=torch.cat([noisy_ids, reference_ids], dim=1),
        target_velocity=velocity_tokens,
        target_token_count=target_token_count,
        timesteps=timesteps,
    )


def build_degraded_anchored_flow_batch(
    clean_latents: Tensor,
    degraded_latents: Tensor,
    noise: Tensor,
    timesteps: Tensor,
    pack_image: Callable[[Tensor], tuple[Tensor, Tensor]],
    *,
    anchor_noise_scale: float,
    reference_time_coordinate: int = 10,
) -> FlowBatch:
    """Build a fidelity-first flow from degraded latent to clean latent.

    The source endpoint is ``degraded + scale * noise`` rather than pure
    random noise. Inference integrates that same endpoint from ``t=1`` to
    ``t=0``. The native FLUX reference-image tokens remain present, so LoRA
    learns a bounded restoration residual instead of regenerating the scene.
    """

    if not 0.0 <= float(anchor_noise_scale) <= 0.25:
        raise ValueError("anchor_noise_scale must be in [0, 0.25]")
    if clean_latents.shape != degraded_latents.shape or clean_latents.shape != noise.shape:
        raise ValueError(
            "clean, degraded, and noise latent shapes must match: "
            f"{clean_latents.shape}, {degraded_latents.shape}, {noise.shape}"
        )
    if timesteps.ndim != 1 or timesteps.shape[0] != clean_latents.shape[0]:
        raise ValueError("timesteps must have shape [batch]")
    if not torch.isfinite(timesteps).all() or not ((timesteps >= 0) & (timesteps <= 1)).all():
        raise ValueError("timesteps must be finite and in [0, 1]")
    if reference_time_coordinate < 0:
        raise ValueError("reference_time_coordinate must be non-negative")

    source_latents = degraded_latents + float(anchor_noise_scale) * noise
    broadcast_t = timesteps.view(-1, *([1] * (clean_latents.ndim - 1))).to(clean_latents)
    path_latents = (1.0 - broadcast_t) * clean_latents + broadcast_t * source_latents
    velocity_latents = source_latents - clean_latents
    path_tokens, path_ids = pack_image(path_latents)
    reference_tokens, reference_ids = pack_image(degraded_latents)
    velocity_tokens, _ = pack_image(velocity_latents)
    reference_ids = reference_ids.clone()
    reference_ids[..., 0] = reference_time_coordinate
    return FlowBatch(
        model_tokens=torch.cat([path_tokens, reference_tokens], dim=1),
        model_ids=torch.cat([path_ids, reference_ids], dim=1),
        target_velocity=velocity_tokens,
        target_token_count=path_tokens.shape[1],
        timesteps=timesteps,
    )
