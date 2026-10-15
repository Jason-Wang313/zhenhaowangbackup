"""VLM-conditioned energy restoration with variable test-time compute.

This module is the performance-first successor to the local one-step EBT
prototype.  It intentionally contains no atomic/pairwise compositional mask.
The model has three coherent parts:

1. a conditioned multi-scale proposal network;
2. a candidate-dependent, multi-scale scalar energy;
3. learned prediction-space gradient descent that may run for a variable
   number of steps at inference.

The degradation condition is a five-dimensional descriptor ordered as
``blur, noise, rain, haze, low_light``.  In deployment it comes from the
frozen VLM router; labels are only suitable for controlled training studies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


DEGRADATIONS = ("blur", "noise", "rain", "haze", "low_light")


class LayerNorm2d(nn.Module):
    def __init__(self, channels: int, epsilon: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.epsilon = float(epsilon)

    def forward(self, value: Tensor) -> Tensor:
        mean = value.mean(dim=1, keepdim=True)
        variance = (value - mean).square().mean(dim=1, keepdim=True)
        return (value - mean) * torch.rsqrt(variance + self.epsilon) * self.weight + self.bias


class ConditionedNAFBlock(nn.Module):
    """NAF restoration block with VLM-controlled feature modulation."""

    def __init__(self, channels: int, condition_dim: int, expansion: int = 2) -> None:
        super().__init__()
        hidden = channels * expansion
        if hidden % 2:
            raise ValueError("expanded width must be even")
        self.norm1 = LayerNorm2d(channels)
        self.in_project = nn.Conv2d(channels, hidden, 1)
        self.depthwise = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden)
        self.channel_scale = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(hidden // 2, hidden // 2, 1)
        )
        self.out_project = nn.Conv2d(hidden // 2, channels, 1)
        self.norm2 = LayerNorm2d(channels)
        self.ffn_in = nn.Conv2d(channels, hidden, 1)
        self.ffn_out = nn.Conv2d(hidden // 2, channels, 1)
        self.modulation = nn.Linear(condition_dim, channels * 4)
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)

    @staticmethod
    def gate(value: Tensor) -> Tensor:
        first, second = value.chunk(2, dim=1)
        return first * second

    @staticmethod
    def modulate(value: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
        return value * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]

    def forward(self, value: Tensor, condition: Tensor) -> Tensor:
        shift1, scale1, shift2, scale2 = self.modulation(condition).chunk(4, dim=1)
        residual = self.modulate(self.norm1(value), shift1, scale1)
        residual = self.gate(self.depthwise(self.in_project(residual)))
        residual = residual * self.channel_scale(residual)
        value = value + self.out_project(residual) * self.beta
        residual = self.modulate(self.norm2(value), shift2, scale2)
        residual = self.gate(self.ffn_in(residual))
        return value + self.ffn_out(residual) * self.gamma


class ConditionedStack(nn.Module):
    def __init__(self, channels: int, count: int, condition_dim: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            ConditionedNAFBlock(channels, condition_dim) for _ in range(count)
        )

    def forward(self, value: Tensor, condition: Tensor) -> Tensor:
        for block in self.blocks:
            value = block(value, condition)
        return value


class ConditionEncoder(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(len(DEGRADATIONS) + 1, width),
            nn.SiLU(),
            nn.Linear(width, width),
        )

    def forward(self, descriptor: Tensor) -> Tensor:
        if descriptor.ndim != 2 or descriptor.shape[1] != len(DEGRADATIONS):
            raise ValueError(f"descriptor must have shape [batch, {len(DEGRADATIONS)}]")
        if not torch.isfinite(descriptor).all():
            raise ValueError("descriptor contains non-finite values")
        if bool(((descriptor < 0) | (descriptor > 1)).any()):
            raise ValueError("descriptor values must lie in [0, 1]")
        order = descriptor.sum(dim=1, keepdim=True) / len(DEGRADATIONS)
        return self.network(torch.cat((descriptor, order), dim=1))


class MultiScaleProposal(nn.Module):
    """Conditioned four-scale proposal with an explicit full-frame tone path."""

    def __init__(
        self,
        *,
        width: int,
        condition_dim: int,
        encoder_blocks: Sequence[int],
        middle_blocks: int,
        residual_bound: float,
    ) -> None:
        super().__init__()
        counts = tuple(int(value) for value in encoder_blocks)
        if len(counts) < 2:
            raise ValueError("proposal needs at least three spatial scales")
        self.required_divisor = 2 ** len(counts)
        self.residual_bound = float(residual_bound)
        self.intro = nn.Conv2d(3, width, 3, padding=1, padding_mode="reflect")
        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        channels = width
        for count in counts:
            self.encoders.append(ConditionedStack(channels, count, condition_dim))
            self.downs.append(nn.Conv2d(channels, channels * 2, 2, stride=2))
            channels *= 2
        self.middle = ConditionedStack(channels, int(middle_blocks), condition_dim)
        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for count in reversed(counts):
            self.ups.append(
                nn.Sequential(nn.Conv2d(channels, channels * 2, 1), nn.PixelShuffle(2))
            )
            channels //= 2
            self.decoders.append(ConditionedStack(channels, count, condition_dim))
        self.tone_features = nn.Sequential(
            nn.Conv2d(3, width, 5, stride=2, padding=2),
            nn.SiLU(),
            nn.Conv2d(width, width * 2, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.tone = nn.Linear(width * 2 + condition_dim, 6)
        self.ending = nn.Conv2d(width, 3, 3, padding=1, padding_mode="reflect")
        nn.init.zeros_(self.tone.weight)
        nn.init.zeros_(self.tone.bias)
        nn.init.zeros_(self.ending.weight)
        nn.init.zeros_(self.ending.bias)

    def forward(self, observation: Tensor, condition: Tensor) -> Tensor:
        if observation.ndim != 4 or observation.shape[1] != 3:
            raise ValueError("observation must be BCHW RGB")
        if observation.shape[-2] % self.required_divisor or observation.shape[-1] % self.required_divisor:
            raise ValueError(f"spatial dimensions must divide by {self.required_divisor}")
        global_view = F.interpolate(
            observation, size=(128, 128), mode="bilinear", align_corners=False
        )
        tone_input = torch.cat((self.tone_features(global_view).flatten(1), condition), dim=1)
        log_gain, offset = self.tone(tone_input).chunk(2, dim=1)
        gain = torch.exp(torch.tanh(log_gain) * 1.5)[:, :, None, None]
        offset = (torch.tanh(offset) * 0.2)[:, :, None, None]
        base = (observation * gain + offset).clamp(-1.0, 1.0)

        value = self.intro(base)
        skips = []
        for encoder, down in zip(self.encoders, self.downs, strict=True):
            value = encoder(value, condition)
            skips.append(value)
            value = down(value)
        value = self.middle(value, condition)
        for up, decoder, skip in zip(
            self.ups, self.decoders, reversed(skips), strict=True
        ):
            value = decoder(up(value) + skip, condition)
        residual = torch.tanh(self.ending(value)) * self.residual_bound
        return (base + residual).clamp(-1.0, 1.0)


class MultiScaleConditionalEnergy(nn.Module):
    """Candidate-dependent scalar energy with local and global evidence."""

    def __init__(
        self,
        *,
        width: int,
        condition_dim: int,
        blocks_per_scale: Sequence[int],
    ) -> None:
        super().__init__()
        counts = tuple(int(value) for value in blocks_per_scale)
        self.required_divisor = 2 ** len(counts)
        # observation, candidate, and their signed residual
        self.intro = nn.Conv2d(9, width, 5, stride=2, padding=2)
        self.stages = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.heads = nn.ModuleList()
        channels = width
        for index, count in enumerate(counts):
            self.stages.append(ConditionedStack(channels, count, condition_dim))
            # A candidate-independent scalar bias cancels from every energy
            # comparison and has zero derivative with respect to the image.
            # Omitting it avoids structurally unidentifiable DDP parameters.
            head = nn.Conv2d(channels, 1, 1, bias=False)
            nn.init.normal_(head.weight, std=0.01)
            self.heads.append(head)
            if index + 1 < len(counts):
                self.downs.append(nn.Conv2d(channels, channels * 2, 3, stride=2, padding=1))
                channels *= 2
        self.global_head = nn.Sequential(
            nn.Linear(channels + condition_dim, channels),
            nn.SiLU(),
            nn.Linear(channels, 1, bias=False),
        )
        nn.init.normal_(self.global_head[-1].weight, std=0.01)
        self.scale_logits = nn.Parameter(torch.zeros(len(counts) + 1))

    def forward(self, observation: Tensor, candidate: Tensor, condition: Tensor) -> Tensor:
        if observation.shape != candidate.shape:
            raise ValueError("observation and candidate must have identical shapes")
        if observation.shape[-2] % self.required_divisor or observation.shape[-1] % self.required_divisor:
            raise ValueError(f"spatial dimensions must divide by {self.required_divisor}")
        value = self.intro(torch.cat((observation, candidate, candidate - observation), dim=1))
        energies = []
        for index, (stage, head) in enumerate(zip(self.stages, self.heads, strict=True)):
            value = stage(value, condition)
            # Mean patch energy preserves the EBT interpretation and keeps
            # scores comparable across image sizes.
            energies.append(head(value).flatten(1).mean(dim=1))
            if index < len(self.downs):
                value = self.downs[index](value)
        pooled = F.adaptive_avg_pool2d(value, 1).flatten(1)
        energies.append(self.global_head(torch.cat((pooled, condition), dim=1)).squeeze(1))
        weights = self.scale_logits.softmax(dim=0)
        return torch.stack(energies, dim=1).mul(weights).sum(dim=1)


@dataclass(frozen=True)
class RestorationTrajectory:
    proposal: Tensor
    states: tuple[Tensor, ...]
    candidates: tuple[Tensor, ...]
    energies: tuple[Tensor, ...]
    gradients: tuple[Tensor, ...]
    step_sizes: tuple[Tensor, ...]

    @property
    def output(self) -> Tensor:
        return self.candidates[-1] if self.candidates else self.proposal


@dataclass(frozen=True)
class VerifiedRestoration:
    output: Tensor
    selected_index: Tensor
    selected_energy: Tensor
    candidate_energies: Tensor
    executed_steps: int


class VLMConditionedEnergyRestorer(nn.Module):
    """Unified proposal and intrinsic energy optimizer for restoration."""

    def __init__(
        self,
        *,
        condition_dim: int = 192,
        proposal_width: int = 48,
        proposal_encoder_blocks: Sequence[int] = (2, 2, 2),
        proposal_middle_blocks: int = 6,
        proposal_residual_bound: float = 0.5,
        energy_width: int = 48,
        energy_blocks_per_scale: Sequence[int] = (2, 2, 4, 4),
        maximum_descent_steps: int = 8,
        initial_step_size: float = 64.0,
    ) -> None:
        super().__init__()
        if maximum_descent_steps < 1 or initial_step_size <= 0:
            raise ValueError("invalid descent schedule")
        self.maximum_descent_steps = int(maximum_descent_steps)
        self.condition_encoder = ConditionEncoder(condition_dim)
        self.proposal = MultiScaleProposal(
            width=proposal_width,
            condition_dim=condition_dim,
            encoder_blocks=proposal_encoder_blocks,
            middle_blocks=proposal_middle_blocks,
            residual_bound=proposal_residual_bound,
        )
        self.energy = MultiScaleConditionalEnergy(
            width=energy_width,
            condition_dim=condition_dim,
            blocks_per_scale=energy_blocks_per_scale,
        )
        raw = torch.log(torch.expm1(torch.tensor(float(initial_step_size))))
        self.raw_step_sizes = nn.Parameter(raw.repeat(self.maximum_descent_steps))
        self.step_controller = nn.Sequential(
            nn.Linear(condition_dim, condition_dim),
            nn.SiLU(),
            nn.Linear(condition_dim, self.maximum_descent_steps),
        )
        nn.init.zeros_(self.step_controller[-1].weight)
        nn.init.zeros_(self.step_controller[-1].bias)

    def learned_step_sizes(self, condition: Tensor) -> Tensor:
        base = F.softplus(self.raw_step_sizes).clamp(1e-3, 1e5)
        modulation = torch.exp(torch.tanh(self.step_controller(condition)))
        return modulation * base[None]

    def forward(
        self,
        observation: Tensor,
        descriptor: Tensor,
        *,
        steps: int,
        create_graph: bool,
        step_scale: Tensor | float = 1.0,
        detach_between_steps: bool = False,
        initial_candidate: Tensor | None = None,
        initial_perturbation: Tensor | None = None,
    ) -> RestorationTrajectory:
        if not 0 <= steps <= self.maximum_descent_steps:
            raise ValueError("steps exceed the configured trajectory")
        condition = self.condition_encoder(descriptor)
        proposal = self.proposal(observation, condition)
        if initial_candidate is not None and initial_candidate.shape != observation.shape:
            raise ValueError("initial_candidate must match the observation shape")
        if initial_perturbation is not None and initial_perturbation.shape != observation.shape:
            raise ValueError("initial_perturbation must match the observation shape")
        if initial_candidate is not None and initial_perturbation is not None:
            raise ValueError("provide either initial_candidate or initial_perturbation, not both")
        candidate = proposal if initial_candidate is None else initial_candidate
        if initial_perturbation is not None:
            candidate = (candidate + initial_perturbation).clamp(-1.0, 1.0)
        candidates: list[Tensor] = []
        states: list[Tensor] = []
        energies: list[Tensor] = []
        gradients: list[Tensor] = []
        used_steps: list[Tensor] = []
        all_step_sizes = self.learned_step_sizes(condition)
        if not torch.is_tensor(step_scale):
            step_scale = observation.new_full((observation.shape[0],), float(step_scale))
        step_scale = step_scale.to(device=observation.device, dtype=observation.dtype).reshape(-1)
        if step_scale.shape != (observation.shape[0],):
            raise ValueError("step_scale must be scalar or have shape [batch]")
        for index in range(steps):
            if detach_between_steps and index:
                candidate = candidate.detach()
            candidate = candidate.requires_grad_(True)
            states.append(candidate)
            energy = self.energy(observation, candidate, condition)
            gradient = torch.autograd.grad(
                energy.sum(), candidate, create_graph=create_graph
            )[0]
            step_size = all_step_sizes[:, index] * step_scale
            candidate = (candidate - step_size[:, None, None, None] * gradient).clamp(-1.0, 1.0)
            candidates.append(candidate)
            energies.append(energy)
            gradients.append(gradient)
            used_steps.append(step_size)
        return RestorationTrajectory(
            proposal=proposal,
            states=tuple(states),
            candidates=tuple(candidates),
            energies=tuple(energies),
            gradients=tuple(gradients),
            step_sizes=tuple(used_steps),
        )

    def verify_candidates(
        self,
        observation: Tensor,
        descriptor: Tensor,
        candidates: Sequence[Tensor],
    ) -> VerifiedRestoration:
        """Select the minimum-energy candidate using the restoration energy.

        No auxiliary reward model is introduced: the exact energy used to
        produce image gradients also performs intrinsic self-verification.
        """

        if not candidates:
            raise ValueError("at least one candidate is required")
        if any(candidate.shape != observation.shape for candidate in candidates):
            raise ValueError("all candidates must match the observation shape")
        condition = self.condition_encoder(descriptor)
        scores = torch.stack(
            [self.energy(observation, candidate, condition) for candidate in candidates],
            dim=1,
        )
        index = scores.argmin(dim=1)
        stacked = torch.stack(tuple(candidates), dim=1)
        batch = torch.arange(observation.shape[0], device=observation.device)
        return VerifiedRestoration(
            output=stacked[batch, index],
            selected_index=index,
            selected_energy=scores[batch, index],
            candidate_energies=scores,
            executed_steps=max(0, len(candidates) - 1),
        )

    def adaptive_inference(
        self,
        observation: Tensor,
        descriptor: Tensor,
        *,
        maximum_steps: int | None = None,
        minimum_steps: int = 1,
        relative_energy_tolerance: float = 1e-3,
        patience: int = 1,
        step_scale: float = 1.0,
    ) -> VerifiedRestoration:
        """Run intrinsic TTS until energy improvement saturates.

        This path is inference-only.  For the normal batch-size-one
        full-resolution use case, early stopping reduces actual energy-model
        evaluations.  The returned output is the best-energy state across the
        entire executed trajectory, preventing a late step from being used
        merely because it was computed last.
        """

        if torch.is_grad_enabled():
            raise RuntimeError("adaptive_inference must run under torch.no_grad() externally")
        limit = self.maximum_descent_steps if maximum_steps is None else int(maximum_steps)
        if not 1 <= minimum_steps <= limit <= self.maximum_descent_steps:
            raise ValueError("invalid adaptive inference step bounds")
        if relative_energy_tolerance < 0 or patience < 1:
            raise ValueError("invalid adaptive stopping criterion")

        # Candidate gradients require a local grad-enabled region even during
        # inference; model parameters remain frozen and no graph is retained.
        condition = self.condition_encoder(descriptor)
        proposal = self.proposal(observation, condition)
        candidates = [proposal.detach()]
        with torch.enable_grad():
            initial = self.energy(observation, proposal.detach().requires_grad_(True), condition)
        energies = [initial.detach()]
        candidate = proposal.detach()
        stagnant = 0
        all_step_sizes = self.learned_step_sizes(condition).detach()
        for index in range(limit):
            with torch.enable_grad():
                current = candidate.detach().requires_grad_(True)
                before = self.energy(observation, current, condition)
                gradient = torch.autograd.grad(before.sum(), current, create_graph=False)[0]
            step = all_step_sizes[:, index] * float(step_scale)
            candidate = (current - step[:, None, None, None] * gradient).clamp(-1.0, 1.0).detach()
            with torch.enable_grad():
                after = self.energy(observation, candidate.requires_grad_(True), condition)
            after = after.detach()
            candidates.append(candidate.detach())
            energies.append(after)
            relative = (before.detach() - after) / before.detach().abs().clamp_min(1e-6)
            stagnant = stagnant + 1 if bool((relative <= relative_energy_tolerance).all()) else 0
            if index + 1 >= minimum_steps and stagnant >= patience:
                break

        scores = torch.stack(energies, dim=1)
        selected = scores.argmin(dim=1)
        stacked = torch.stack(candidates, dim=1)
        batch = torch.arange(observation.shape[0], device=observation.device)
        return VerifiedRestoration(
            output=stacked[batch, selected],
            selected_index=selected,
            selected_energy=scores[batch, selected],
            candidate_energies=scores,
            executed_steps=len(candidates) - 1,
        )


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
