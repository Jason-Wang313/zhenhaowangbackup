"""Explicit pairwise safe-versus-candidate restoration gate."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels), nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
        )

    def forward(self, value: Tensor) -> Tensor:
        return value + self.body(value)


class PairwiseSafeCandidateGate(nn.Module):
    """Compare each deployable alternative directly against fidelity."""

    def __init__(
        self,
        *,
        alternative_count: int,
        condition_dim: int = 5,
        widths: tuple[int, int, int, int, int] = (48, 64, 80, 96, 128),
        blocks: int = 3,
    ) -> None:
        super().__init__()
        if alternative_count < 1 or len(widths) != 5 or any(width % 8 for width in widths):
            raise ValueError("invalid pairwise gate geometry")
        self.alternative_count = int(alternative_count)
        first, second, third, fourth, fifth = widths
        channels = (18, first, second, third, fourth)
        outputs = (first, second, third, fourth, fifth)
        layers: list[nn.Module] = []
        for index, (input_channels, output_channels) in enumerate(zip(channels, outputs, strict=True)):
            layers.extend((
                nn.Conv2d(input_channels, output_channels, 5 if index == 0 else 3, stride=2, padding=2 if index == 0 else 1),
                nn.GroupNorm(8, output_channels), nn.SiLU(),
            ))
        layers.extend(ResidualBlock(fifth) for _ in range(blocks))
        self.encoder = nn.Sequential(*layers)
        self.condition = nn.Sequential(
            nn.Linear(condition_dim, fifth), nn.SiLU(), nn.Linear(fifth, 2 * fifth)
        )
        self.identity = nn.Embedding(alternative_count, 2 * fifth)
        self.output = nn.Conv2d(fifth, 3, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        observation: Tensor,
        proposal: Tensor,
        fidelity: Tensor,
        alternatives: Tensor,
        severity: Tensor,
    ) -> Tensor:
        if observation.shape != proposal.shape or proposal.shape != fidelity.shape:
            raise ValueError("observation/proposal/fidelity shapes disagree")
        if alternatives.ndim != 5 or alternatives.shape[0] != observation.shape[0] or alternatives.shape[2:] != observation.shape[1:]:
            raise ValueError("alternatives must be BACHW and match observation")
        if alternatives.shape[1] != self.alternative_count:
            raise ValueError("alternative count changed")
        if observation.shape[-2] % 32 or observation.shape[-1] % 32:
            raise ValueError("pairwise gate dimensions must divide by 32")
        batch, count = alternatives.shape[:2]
        observation_a = observation[:, None].expand(-1, count, -1, -1, -1)
        proposal_a = proposal[:, None].expand_as(observation_a)
        fidelity_a = fidelity[:, None].expand_as(observation_a)
        values = torch.cat((
            observation_a, proposal_a, fidelity_a, alternatives,
            alternatives - fidelity_a, alternatives - observation_a,
        ), dim=2).flatten(0, 1)
        features = self.encoder(values)
        condition = self.condition(severity)[:, None]
        identities = self.identity(torch.arange(count, device=features.device))[None]
        affine = (condition + identities).reshape(batch * count, -1)
        scale, bias = affine.chunk(2, dim=1)
        features = features * (1 + scale[:, :, None, None]) + bias[:, :, None, None]
        return self.output(features).reshape(batch, count, 3, *features.shape[-2:])


def pairwise_targets(
    fidelity: Tensor,
    alternatives: Tensor,
    target: Tensor,
    fidelity_perceptual: Tensor,
    alternative_perceptual: Tensor,
    *,
    patch_size: int = 32,
    epsilon: float = 1e-8,
) -> tuple[Tensor, Tensor]:
    candidate_mse = F.avg_pool2d(
        (alternatives - target[:, None]).square().mean(2).flatten(0, 1)[:, None], patch_size, patch_size
    ).reshape(*alternatives.shape[:2], alternatives.shape[-2] // patch_size, alternatives.shape[-1] // patch_size)
    safe_mse = F.avg_pool2d((fidelity - target).square().mean(1, keepdim=True), patch_size, patch_size)
    delta_mse = torch.log((candidate_mse + epsilon) / (safe_mse + epsilon))
    candidate_perceptual = F.avg_pool2d(
        alternative_perceptual.flatten(0, 1)[:, None], patch_size, patch_size
    ).reshape_as(delta_mse)
    safe_perceptual = F.avg_pool2d(fidelity_perceptual[:, None], patch_size, patch_size)
    delta_perceptual = torch.log((candidate_perceptual + epsilon) / (safe_perceptual + epsilon))
    return delta_mse.clamp(-4, 4), delta_perceptual.clamp(-4, 4)


def balanced_binary_loss(logits: Tensor, labels: Tensor, mask: Tensor) -> Tensor:
    selected_logits, selected_labels = logits[mask], labels[mask].to(logits.dtype)
    if not selected_logits.numel():
        return logits.new_zeros(())
    positive = selected_labels.sum()
    negative = selected_labels.numel() - positive
    positive_weight = (negative / positive.clamp_min(1)).clamp(.25, 4)
    return F.binary_cross_entropy_with_logits(selected_logits, selected_labels, pos_weight=positive_weight)


def select_pairwise_candidates(
    outputs: Tensor,
    *,
    gain_probability: float,
    maximum_harm_probability: float,
) -> tuple[Tensor, Tensor]:
    if outputs.ndim != 5 or outputs.shape[2] != 3:
        raise ValueError("pairwise outputs must be BA3HW")
    gain_logit, harm_logit, predicted_delta = outputs.unbind(2)
    eligible = (
        (torch.sigmoid(gain_logit) >= gain_probability)
        & (torch.sigmoid(harm_logit) <= maximum_harm_probability)
        & (predicted_delta < 0)
    )
    cost = predicted_delta.masked_fill(~eligible, float("inf"))
    best_cost, best = cost.min(1)
    accepted = torch.isfinite(best_cost)
    return best, accepted


def compose_pairwise_selection(
    fidelity: Tensor,
    alternatives: Tensor,
    selected: Tensor,
    accepted: Tensor,
    *,
    patch_size: int = 32,
) -> Tensor:
    weights = F.one_hot(selected, num_classes=alternatives.shape[1]).permute(0, 3, 1, 2).to(alternatives.dtype)
    weights = weights.repeat_interleave(patch_size, -2).repeat_interleave(patch_size, -1)
    alternative = (alternatives * weights[:, :, None]).sum(1)
    mask = accepted[:, None].repeat_interleave(patch_size, -2).repeat_interleave(patch_size, -1)
    return torch.where(mask, alternative, fidelity)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
