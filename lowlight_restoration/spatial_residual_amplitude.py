"""Deployable spatial amplitude predictor for a frozen restoration residual."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        groups = 8 if channels % 8 == 0 else 1
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(groups, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(groups, channels),
        )

    def forward(self, value: Tensor) -> Tensor:
        return value + self.body(value)


class SpatialResidualAmplitudePredictor(nn.Module):
    """Predict one [0, 2] amplitude for every non-overlapping 8x8 patch.

    Inputs are all inference-available: observation, frozen proposal, frozen
    fidelity output, and the fidelity residual.  Zero initialization of the
    last layer makes the initial amplitude exactly one, reproducing fidelity.
    """

    def __init__(self, channels: int = 48, blocks: int = 4) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(12, channels, 5, stride=2, padding=2),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, stride=2, padding=1),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, stride=2, padding=1),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            *[ResidualBlock(channels) for _ in range(blocks)],
        )
        self.output = nn.Conv2d(channels, 1, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, observation: Tensor, proposal: Tensor, fidelity: Tensor) -> Tensor:
        if observation.shape != proposal.shape or proposal.shape != fidelity.shape:
            raise ValueError("observation, proposal and fidelity must have matching shapes")
        if observation.ndim != 4 or observation.shape[1] != 3:
            raise ValueError("spatial amplitude inputs must be BCHW RGB tensors")
        if observation.shape[-2] % 8 or observation.shape[-1] % 8:
            raise ValueError("spatial amplitude input dimensions must divide by 8")
        features = torch.cat((observation, proposal, fidelity, fidelity - proposal), dim=1)
        return 2 * torch.sigmoid(self.output(self.encoder(features)))


def apply_patch_amplitude(proposal: Tensor, fidelity: Tensor, amplitude: Tensor) -> Tensor:
    if amplitude.shape[-2:] != (proposal.shape[-2] // 8, proposal.shape[-1] // 8):
        raise ValueError("amplitude grid must contain one value per 8x8 patch")
    pixels = amplitude.repeat_interleave(8, -2).repeat_interleave(8, -1)
    return (proposal + pixels * (fidelity - proposal)).clamp(0, 1)


def analytic_patch_amplitude(proposal: Tensor, fidelity: Tensor, target: Tensor) -> tuple[Tensor, Tensor]:
    """Training-only least-squares amplitude and residual-energy confidence."""
    residual = fidelity - proposal
    desired = target - proposal
    numerator = torch.nn.functional.avg_pool2d((residual * desired).sum(1, keepdim=True), 8, 8)
    energy = torch.nn.functional.avg_pool2d(residual.square().sum(1, keepdim=True), 8, 8)
    amplitude = (numerator / energy.clamp_min(1e-12)).clamp(0, 2)
    confidence = energy / energy.mean(dim=(-2, -1), keepdim=True).clamp_min(1e-12)
    return amplitude, confidence.clamp(0, 4)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
