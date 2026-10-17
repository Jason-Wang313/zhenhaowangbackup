"""Candidate-conditioned local restoration risk with fail-closed selection."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
        )

    def forward(self, value: Tensor) -> Tensor:
        return value + self.body(value)


class CompleteCandidateEnergy(nn.Module):
    """Score each actual candidate with one shared, condition-aware network.

    Outputs are relative log-MSE risk, relative spatial-LPIPS risk, and one
    log-variance map. Zero initialization is deliberately fail-closed: before
    training no alternative has a certified advantage over fidelity.
    """

    def __init__(
        self,
        *,
        condition_dim: int = 5,
        widths: tuple[int, int, int, int, int] = (48, 64, 80, 96, 128),
        blocks: int = 3,
        initial_log_variance: float = 0.0,
    ) -> None:
        super().__init__()
        if len(widths) != 5 or any(width % 8 for width in widths):
            raise ValueError("five widths divisible by eight are required")
        first, second, third, fourth, fifth = widths
        layers: list[nn.Module] = []
        inputs = (15, first, second, third, fourth)
        outputs = (first, second, third, fourth, fifth)
        for index, (input_channels, output_channels) in enumerate(zip(inputs, outputs, strict=True)):
            layers.extend((
                nn.Conv2d(input_channels, output_channels, 5 if index == 0 else 3, stride=2, padding=2 if index == 0 else 1),
                nn.GroupNorm(8, output_channels),
                nn.SiLU(),
            ))
        layers.extend(ResidualBlock(fifth) for _ in range(blocks))
        self.encoder = nn.Sequential(*layers)
        self.condition = nn.Sequential(
            nn.Linear(condition_dim, fifth), nn.SiLU(), nn.Linear(fifth, 2 * fifth)
        )
        self.output = nn.Conv2d(fifth, 3, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        with torch.no_grad():
            self.output.bias[2] = float(initial_log_variance)

    def forward(
        self,
        observation: Tensor,
        proposal: Tensor,
        candidates: Tensor,
        severity: Tensor,
    ) -> Tensor:
        if observation.shape != proposal.shape or observation.ndim != 4 or observation.shape[1] != 3:
            raise ValueError("observation and proposal must be matching BCHW RGB tensors")
        if candidates.ndim != 5 or candidates.shape[0] != observation.shape[0] or candidates.shape[2:] != observation.shape[1:]:
            raise ValueError("candidates must be BKCHW and match observation")
        if severity.ndim != 2 or severity.shape[0] != observation.shape[0]:
            raise ValueError("severity must be a matching BC tensor")
        if observation.shape[-2] % 32 or observation.shape[-1] % 32:
            raise ValueError("candidate energy dimensions must divide by 32")
        batch, count = candidates.shape[:2]
        observation_k = observation[:, None].expand(-1, count, -1, -1, -1)
        proposal_k = proposal[:, None].expand_as(observation_k)
        values = torch.cat((
            observation_k,
            proposal_k,
            candidates,
            candidates - proposal_k,
            candidates - observation_k,
        ), dim=2).flatten(0, 1)
        features = self.encoder(values)
        affine = self.condition(severity).repeat_interleave(count, dim=0)
        scale, bias = affine.chunk(2, dim=1)
        features = features * (1 + scale[:, :, None, None]) + bias[:, :, None, None]
        output = self.output(features).reshape(batch, count, 3, *features.shape[-2:])
        return output


def relative_risk_targets(
    candidates: Tensor,
    target: Tensor,
    spatial_lpips: Tensor,
    *,
    safe_index: int,
    patch_size: int = 32,
    epsilon: float = 1e-8,
    log_clip: float = 4.0,
) -> tuple[Tensor, Tensor]:
    """Create GT-only training targets relative to the fidelity candidate."""
    if candidates.ndim != 5 or spatial_lpips.shape[:2] != candidates.shape[:2]:
        raise ValueError("candidate/LPIPS target shapes disagree")
    pixel_error = (candidates - target[:, None]).square().mean(2)
    mse = F.avg_pool2d(pixel_error.flatten(0, 1)[:, None], patch_size, patch_size)
    mse = mse.reshape(*candidates.shape[:2], *mse.shape[-2:])
    perceptual = F.avg_pool2d(
        spatial_lpips.flatten(0, 1)[:, None], patch_size, patch_size
    ).reshape(*candidates.shape[:2], *mse.shape[-2:])
    safe_mse = mse[:, safe_index : safe_index + 1]
    safe_perceptual = perceptual[:, safe_index : safe_index + 1]
    relative_mse = torch.log((mse + epsilon) / (safe_mse + epsilon)).clamp(-log_clip, log_clip)
    relative_perceptual = torch.log(
        (perceptual + epsilon) / (safe_perceptual + epsilon)
    ).clamp(-log_clip, log_clip)
    return relative_mse, relative_perceptual


def tie_aware_gap_ranking(predicted: Tensor, target: Tensor, *, tie_epsilon: float) -> Tensor:
    """Pairwise logistic ranking that ignores effectively tied candidates."""
    target_gap = target[:, :, None] - target[:, None, :]
    prediction_gap = predicted[:, :, None] - predicted[:, None, :]
    mask = target_gap.abs() > tie_epsilon
    if not bool(mask.any()):
        return predicted.new_zeros(())
    direction = target_gap.sign()
    weight = target_gap.abs().clamp(max=1.0)
    return (F.softplus(-direction * prediction_gap) * weight)[mask].sum() / weight[mask].sum().clamp_min(1e-8)


def select_candidates_fail_closed(
    risks: Tensor,
    *,
    safe_index: int,
    uncertainty_scale: float,
    minimum_mse_advantage: float,
    maximum_relative_perceptual_risk: float,
) -> tuple[Tensor, Tensor]:
    """Select only certified alternatives; otherwise return fidelity."""
    if risks.ndim != 5 or risks.shape[2] != 3:
        raise ValueError("risks must be BK3HW")
    mse, perceptual, log_variance = risks.unbind(2)
    uncertainty = torch.exp(.5 * log_variance.clamp(-8, 8)) * uncertainty_scale
    conservative_mse = mse + uncertainty
    conservative_perceptual = perceptual + uncertainty
    best_cost, best = conservative_mse.min(1)
    safe_cost = conservative_mse[:, safe_index]
    accept = (
        (best != safe_index)
        & (best_cost + minimum_mse_advantage < safe_cost)
        & (conservative_perceptual.gather(1, best[:, None]).squeeze(1) <= maximum_relative_perceptual_risk)
    )
    selected = torch.where(accept, best, torch.full_like(best, safe_index))
    return selected, accept


def compose_selected_candidates(candidates: Tensor, selected: Tensor, *, patch_size: int = 32) -> Tensor:
    if candidates.ndim != 5 or selected.shape != (candidates.shape[0], candidates.shape[-2] // patch_size, candidates.shape[-1] // patch_size):
        raise ValueError("candidate and selection grid shapes disagree")
    weights = F.one_hot(selected, num_classes=candidates.shape[1]).permute(0, 3, 1, 2).to(candidates.dtype)
    weights = weights.repeat_interleave(patch_size, -2).repeat_interleave(patch_size, -1)
    return (candidates * weights[:, :, None]).sum(1)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
