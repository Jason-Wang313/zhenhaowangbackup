"""Max-complete spatial proposal-relative energy used by the PPT V2 test.

V2 keeps the frozen restoration proposal and changes only the learned control:
it exposes a spatial energy map, unrolls all three candidate-gradient steps,
and can retain the unchanged proposal when its learned energy is lowest.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .proposal_relative_energy import MatchedDirectRefiner
from .vlm_energy_restoration import ConditionEncoder, ConditionedStack


class SpatialConditionalEnergy(nn.Module):
    """Full-resolution local energy field with multi-scale context."""

    def __init__(self, *, width: int = 16, condition_dim: int = 96) -> None:
        super().__init__()
        self.required_divisor = 4
        self.intro = nn.Conv2d(9, width, 3, padding=1, padding_mode="reflect")
        self.full = ConditionedStack(width, 2, condition_dim)
        self.down1 = nn.Conv2d(width, width * 2, 3, stride=2, padding=1)
        self.half_stack = ConditionedStack(width * 2, 2, condition_dim)
        self.down2 = nn.Conv2d(width * 2, width * 4, 3, stride=2, padding=1)
        self.quarter = ConditionedStack(width * 4, 3, condition_dim)
        self.up1 = nn.Conv2d(width * 4, width * 2, 3, padding=1)
        self.decode_half = ConditionedStack(width * 2, 2, condition_dim)
        self.up2 = nn.Conv2d(width * 2, width, 3, padding=1)
        self.decode_full = ConditionedStack(width, 2, condition_dim)
        self.head = nn.Conv2d(width, 1, 3, padding=1, padding_mode="reflect", bias=False)
        nn.init.normal_(self.head.weight, std=0.01)

    def forward(self, observation: Tensor, candidate: Tensor, condition: Tensor) -> Tensor:
        if observation.shape != candidate.shape:
            raise ValueError("observation and candidate must have identical shapes")
        if observation.shape[-2] % self.required_divisor or observation.shape[-1] % self.required_divisor:
            raise ValueError(f"spatial dimensions must divide by {self.required_divisor}")
        value = self.intro(torch.cat((observation, candidate, candidate - observation), dim=1))
        full = self.full(value, condition)
        half = self.half_stack(self.down1(full), condition)
        quarter = self.quarter(self.down2(half), condition)
        value = F.interpolate(quarter, size=half.shape[-2:], mode="bilinear", align_corners=False)
        value = self.decode_half(self.up1(value) + half, condition)
        value = F.interpolate(value, size=full.shape[-2:], mode="bilinear", align_corners=False)
        return self.head(self.decode_full(self.up2(value) + full, condition))


@dataclass(frozen=True)
class SpatialRefinementTrajectory:
    proposal: Tensor
    candidates: tuple[Tensor, ...]
    energy_maps: tuple[Tensor, ...]
    energies: tuple[Tensor, ...]
    gradients: tuple[Tensor, ...]


class SpatialProposalRelativeEnergy(nn.Module):
    def __init__(
        self,
        *,
        condition_dim: int = 96,
        width: int = 16,
        maximum_steps: int = 3,
        initial_step_size: float = 8.0,
    ) -> None:
        super().__init__()
        self.maximum_steps = int(maximum_steps)
        self.condition_encoder = ConditionEncoder(condition_dim)
        self.energy = SpatialConditionalEnergy(width=width, condition_dim=condition_dim)
        raw = torch.log(torch.expm1(torch.tensor(float(initial_step_size))))
        self.raw_step_sizes = nn.Parameter(raw.repeat(self.maximum_steps))
        self.step_modulation = nn.Sequential(
            nn.Linear(condition_dim, condition_dim),
            nn.SiLU(),
            nn.Linear(condition_dim, self.maximum_steps),
        )
        nn.init.zeros_(self.step_modulation[-1].weight)
        nn.init.zeros_(self.step_modulation[-1].bias)

    def condition(self, descriptor: Tensor) -> Tensor:
        return self.condition_encoder(descriptor)

    def score_map_from_condition(
        self, observation: Tensor, candidate: Tensor, condition: Tensor
    ) -> Tensor:
        return self.energy(observation, candidate, condition)

    def score_map(self, observation: Tensor, candidate: Tensor, descriptor: Tensor) -> Tensor:
        return self.score_map_from_condition(observation, candidate, self.condition(descriptor))

    def score(self, observation: Tensor, candidate: Tensor, descriptor: Tensor) -> Tensor:
        return self.score_map(observation, candidate, descriptor).flatten(1).mean(1)

    def step_sizes(self, condition: Tensor) -> Tensor:
        base = F.softplus(self.raw_step_sizes).clamp(1e-3, 64.0)
        return base[None] * torch.exp(torch.tanh(self.step_modulation(condition)))

    def refine(
        self,
        observation: Tensor,
        proposal: Tensor,
        descriptor: Tensor,
        *,
        steps: int = 3,
        create_graph: bool,
        detach_between_steps: bool = False,
    ) -> SpatialRefinementTrajectory:
        if not 0 <= steps <= self.maximum_steps:
            raise ValueError("invalid proposal-relative energy step count")
        condition = self.condition(descriptor)
        schedule = self.step_sizes(condition)
        current = proposal
        candidates: list[Tensor] = []
        maps: list[Tensor] = []
        energies: list[Tensor] = []
        gradients: list[Tensor] = []
        for index in range(steps):
            if index and detach_between_steps:
                current = current.detach()
            current = current.requires_grad_(True)
            energy_map = self.score_map_from_condition(observation, current, condition)
            scalar = energy_map.flatten(1).mean(1)
            gradient = torch.autograd.grad(scalar.sum(), current, create_graph=create_graph)[0]
            current = (current - schedule[:, index, None, None, None] * gradient).clamp(-1.0, 1.0)
            candidates.append(current)
            maps.append(energy_map)
            energies.append(scalar)
            gradients.append(gradient)
        return SpatialRefinementTrajectory(
            proposal=proposal,
            candidates=tuple(candidates),
            energy_maps=tuple(maps),
            energies=tuple(energies),
            gradients=tuple(gradients),
        )

    def select_lowest_energy(
        self,
        observation: Tensor,
        proposal: Tensor,
        candidates: tuple[Tensor, ...],
        descriptor: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        values = (proposal,) + tuple(candidates)
        batch, count = proposal.shape[0], len(values)
        stacked = torch.stack(values, dim=1)
        flat = stacked.flatten(0, 1)
        repeated_observation = observation[:, None].expand(-1, count, -1, -1, -1).flatten(0, 1)
        repeated_descriptor = descriptor[:, None].expand(-1, count, -1).flatten(0, 1)
        energies = self.score(repeated_observation, flat, repeated_descriptor).reshape(batch, count)
        selected_index = energies.argmin(dim=1)
        gather = selected_index[:, None, None, None, None].expand(-1, 1, *proposal.shape[1:])
        selected = stacked.gather(1, gather).squeeze(1)
        return selected, selected_index, energies


class EnergyAndDirectControlsV2(nn.Module):
    """One DDP-visible graph for three-step energy and the direct control."""

    def __init__(self) -> None:
        super().__init__()
        self.energy_model = SpatialProposalRelativeEnergy()
        self.direct_model = MatchedDirectRefiner()

    def forward(
        self,
        observation: Tensor,
        proposal: Tensor,
        target: Tensor,
        descriptor: Tensor,
        hard_negatives: Tensor,
    ) -> dict[str, Tensor | tuple[Tensor, ...]]:
        if hard_negatives.ndim != 5 or hard_negatives.shape[0] != observation.shape[0]:
            raise ValueError("hard_negatives must have shape [batch, candidates, 3, H, W]")
        trajectory = self.energy_model.refine(
            observation, proposal, descriptor, steps=3, create_graph=True
        )
        direct = self.direct_model(observation, proposal, descriptor)
        contrastive = (target, proposal) + tuple(hard_negatives.unbind(dim=1)) + trajectory.candidates
        batch, count = observation.shape[0], len(contrastive)
        stacked = torch.stack(contrastive, dim=1)
        flat = stacked.flatten(0, 1)
        repeated_observation = observation[:, None].expand(-1, count, -1, -1, -1).flatten(0, 1)
        repeated_descriptor = descriptor[:, None].expand(-1, count, -1).flatten(0, 1)
        maps = self.energy_model.score_map(
            repeated_observation, flat, repeated_descriptor
        ).reshape(batch, count, 1, observation.shape[-2], observation.shape[-1])
        return {
            "direct": direct,
            "trajectory_candidates": trajectory.candidates,
            "trajectory_gradients": trajectory.gradients,
            "contrastive_candidates": contrastive,
            "contrastive_energy_maps": maps,
            "contrastive_energies": maps.flatten(2).mean(2),
        }


def parameter_counts(model: EnergyAndDirectControlsV2) -> dict[str, int]:
    return {
        "energy": sum(p.numel() for p in model.energy_model.parameters()),
        "direct": sum(p.numel() for p in model.direct_model.parameters()),
    }
