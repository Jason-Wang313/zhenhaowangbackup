"""Isolated numerical helpers for the authorized PPT retry, not a trainer.

Do not alter historical V2 checkpoints/results. A reference area keeps the
same update convention at a chosen crop while removing the explicit 1/(H*W)
step-size change. It does not ensure a good descent direction or quality.
"""
from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


def ranking_score(energy_map: Tensor) -> Tensor:
    if energy_map.ndim != 4 or energy_map.shape[1] != 1:
        raise ValueError('expected one energy density per pixel: [B, 1, H, W]')
    return energy_map.flatten(1).mean(1)


def refinement_objective(energy_map: Tensor, *, reference_pixels: int) -> Tensor:
    if type(reference_pixels) is not int or reference_pixels <= 0:
        raise ValueError('reference_pixels must be a fixed positive integer')
    # Positive image-size factor leaves within-image ranking unchanged. The
    # objective is summed over batch by the caller, never averaged over batch.
    return ranking_score(energy_map) * (energy_map.shape[-2] * energy_map.shape[-1] / reference_pixels)


def detached_local_mse_targets(candidates: tuple[Tensor, ...], target: Tensor, *, kernel: int) -> Tensor:
    """Supervision is a measurement, not a second path into candidate updates."""
    if not candidates or type(kernel) is not int or kernel <= 0:
        raise ValueError('nonempty candidates and positive kernel required')
    if target.ndim != 4 or target.shape[1] != 3 or any(candidate.shape != target.shape for candidate in candidates):
        raise ValueError('all RGB tensors must have identical BCHW shapes')
    with torch.no_grad():
        local = (torch.stack(candidates, dim=1) - target[:, None]).square().mean(2)
        batch, count, height, width = local.shape
        if min(height, width) < kernel:
            raise ValueError('quality kernel exceeds image size')
        pooled = F.avg_pool2d(local.reshape(batch * count, 1, height, width), kernel, kernel)
        return pooled.reshape(batch, count, *pooled.shape[-2:])
