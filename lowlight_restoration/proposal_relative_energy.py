"""Proposal-relative restoration energy for frozen restoration outputs.

The proposal model is deliberately external and frozen.  This module only
learns (1) a scalar candidate energy and (2) a matched direct residual control.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .vlm_energy_restoration import (
    ConditionEncoder,
    ConditionedStack,
    MultiScaleConditionalEnergy,
)


@dataclass(frozen=True)
class ProposalRefinement:
    proposal: Tensor
    candidates: tuple[Tensor, ...]
    energies: tuple[Tensor, ...]
    gradients: tuple[Tensor, ...]

    @property
    def output(self) -> Tensor:
        return self.candidates[-1] if self.candidates else self.proposal


class ProposalRelativeEnergy(nn.Module):
    """Learn an energy over a candidate relative to an observation/proposal."""

    def __init__(
        self,
        *,
        condition_dim: int = 96,
        width: int = 24,
        blocks_per_scale: Sequence[int] = (1, 1, 2),
        maximum_steps: int = 3,
        initial_step_size: float = 24.0,
    ) -> None:
        super().__init__()
        self.maximum_steps = int(maximum_steps)
        self.condition_encoder = ConditionEncoder(condition_dim)
        self.energy = MultiScaleConditionalEnergy(
            width=width,
            condition_dim=condition_dim,
            blocks_per_scale=blocks_per_scale,
        )
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

    def score(self, observation: Tensor, candidate: Tensor, descriptor: Tensor) -> Tensor:
        return self.energy(observation, candidate, self.condition(descriptor))

    def step_sizes(self, condition: Tensor) -> Tensor:
        base = F.softplus(self.raw_step_sizes).clamp(1e-3, 1e4)
        return base[None] * torch.exp(torch.tanh(self.step_modulation(condition)))

    def refine(
        self,
        observation: Tensor,
        proposal: Tensor,
        descriptor: Tensor,
        *,
        steps: int,
        create_graph: bool,
        detach_between_steps: bool = False,
    ) -> ProposalRefinement:
        if not 0 <= steps <= self.maximum_steps:
            raise ValueError("invalid proposal-relative energy step count")
        condition = self.condition(descriptor)
        schedule = self.step_sizes(condition)
        current = proposal
        candidates: list[Tensor] = []
        energies: list[Tensor] = []
        gradients: list[Tensor] = []
        for index in range(steps):
            if index and detach_between_steps:
                current = current.detach()
            current = current.requires_grad_(True)
            energy = self.energy(observation, current, condition)
            gradient = torch.autograd.grad(
                energy.sum(), current, create_graph=create_graph
            )[0]
            current = (
                current - schedule[:, index, None, None, None] * gradient
            ).clamp(-1.0, 1.0)
            candidates.append(current)
            energies.append(energy)
            gradients.append(gradient)
        return ProposalRefinement(
            proposal=proposal,
            candidates=tuple(candidates),
            energies=tuple(energies),
            gradients=tuple(gradients),
        )


class MatchedDirectRefiner(nn.Module):
    """Feed-forward control with a bounded correction around the same proposal."""

    def __init__(
        self,
        *,
        condition_dim: int = 96,
        width: int = 40,
        blocks: int = 12,
        residual_bound: float = 0.25,
    ) -> None:
        super().__init__()
        self.residual_bound = float(residual_bound)
        self.condition_encoder = ConditionEncoder(condition_dim)
        self.intro = nn.Conv2d(9, width, 3, padding=1, padding_mode="reflect")
        self.body = ConditionedStack(width, blocks, condition_dim)
        self.ending = nn.Conv2d(width, 3, 3, padding=1, padding_mode="reflect")
        nn.init.zeros_(self.ending.weight)
        nn.init.zeros_(self.ending.bias)

    def forward(
        self, observation: Tensor, proposal: Tensor, descriptor: Tensor
    ) -> Tensor:
        condition = self.condition_encoder(descriptor)
        value = torch.cat((observation, proposal, proposal - observation), dim=1)
        correction = torch.tanh(self.ending(self.body(self.intro(value), condition)))
        return (proposal + self.residual_bound * correction).clamp(-1.0, 1.0)


class EnergyAndDirectControls(nn.Module):
    """Single DDP-visible module for the causal energy/direct comparison."""

    def __init__(self) -> None:
        super().__init__()
        self.energy_model = ProposalRelativeEnergy()
        self.direct_model = MatchedDirectRefiner()

    def forward(
        self,
        observation: Tensor,
        proposal: Tensor,
        target: Tensor,
        descriptor: Tensor,
    ) -> dict[str, Tensor]:
        direct = self.direct_model(observation, proposal, descriptor)
        trajectory = self.energy_model.refine(
            observation, proposal, descriptor, steps=1, create_graph=True
        )
        refined = trajectory.output
        gradient = trajectory.gradients[0]
        desired = (target - proposal).flatten(1)
        update = (-gradient).flatten(1)
        direction = (1.0 - F.cosine_similarity(update, desired, dim=1, eps=1e-8)).mean()

        under = 0.5 * (observation + proposal)
        over = (proposal + 0.5 * (proposal - observation)).clamp(-1.0, 1.0)
        blurred = F.avg_pool2d(proposal, 3, stride=1, padding=1)
        candidates = (target, proposal, under, over, blurred)
        condition = self.energy_model.condition(descriptor)
        energies = torch.stack(
            [self.energy_model.energy(observation, item, condition) for item in candidates],
            dim=1,
        )
        qualities = torch.stack(
            [(item - target).square().flatten(1).mean(1) for item in candidates], dim=1
        ).detach()
        better = qualities[:, :, None] + 1e-8 < qualities[:, None, :]
        gap = (qualities[:, None, :] - qualities[:, :, None]).clamp_min(0)
        normalized = gap / qualities.mean(1, keepdim=True)[:, :, None].clamp_min(1e-6)
        violations = F.relu(0.1 * normalized + energies[:, :, None] - energies[:, None, :])
        ranking = violations[better].mean() if bool(better.any()) else energies.new_zeros(())
        return {
            "direct": direct,
            "refined": refined,
            "direction": direction,
            "ranking": ranking,
            "proposal_energy": energies[:, 1].mean(),
            "target_energy": energies[:, 0].mean(),
        }


def parameter_counts(model: EnergyAndDirectControls) -> dict[str, int]:
    return {
        "energy": sum(p.numel() for p in model.energy_model.parameters()),
        "direct": sum(p.numel() for p in model.direct_model.parameters()),
    }
