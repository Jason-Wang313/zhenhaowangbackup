"""Local candidate-action energy for a frozen restoration residual."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


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


class LocalActionEnergy(nn.Module):
    """Predict relative cost for fixed actions on non-overlapping 8x8 regions.

    The final layer is initialized so action ``safe_index`` has strictly lower
    cost than every alternative. Thus a fresh model reproduces the frozen
    fidelity result exactly instead of applying an untrained correction.
    """

    def __init__(
        self,
        *,
        candidates: tuple[float, ...],
        safe_index: int,
        widths: tuple[int, int, int] = (64, 96, 128),
        blocks: int = 4,
        initial_unsafe_margin: float = 0.25,
    ) -> None:
        super().__init__()
        if not candidates or not 0 <= safe_index < len(candidates):
            raise ValueError("invalid candidate/safe-index contract")
        if candidates[safe_index] != 1.0:
            raise ValueError("safe action must be alpha=1")
        if len(widths) != 3 or any(width % 8 for width in widths):
            raise ValueError("three widths divisible by eight are required")
        self.candidates = candidates
        self.safe_index = int(safe_index)
        first, second, third = widths
        self.encoder = nn.Sequential(
            nn.Conv2d(12, first, 5, stride=2, padding=2),
            nn.GroupNorm(8, first),
            nn.SiLU(),
            nn.Conv2d(first, second, 3, stride=2, padding=1),
            nn.GroupNorm(8, second),
            nn.SiLU(),
            nn.Conv2d(second, third, 3, stride=2, padding=1),
            nn.GroupNorm(8, third),
            nn.SiLU(),
            *[ResidualBlock(third) for _ in range(blocks)],
        )
        self.output = nn.Conv2d(third, len(candidates), 1)
        nn.init.zeros_(self.output.weight)
        nn.init.constant_(self.output.bias, float(initial_unsafe_margin))
        with torch.no_grad():
            self.output.bias[self.safe_index] = 0

    def forward(self, observation: Tensor, proposal: Tensor, fidelity: Tensor) -> Tensor:
        if observation.shape != proposal.shape or proposal.shape != fidelity.shape:
            raise ValueError("observation, proposal and fidelity must have matching shapes")
        if observation.ndim != 4 or observation.shape[1] != 3:
            raise ValueError("local energy inputs must be BCHW RGB tensors")
        if observation.shape[-2] % 8 or observation.shape[-1] % 8:
            raise ValueError("local energy input dimensions must divide by eight")
        values = torch.cat((observation, proposal, fidelity, fidelity - proposal), dim=1)
        return self.output(self.encoder(values))


def candidate_alphas(values: tuple[float, ...], reference: Tensor) -> Tensor:
    return reference.new_tensor(values).view(1, len(values), 1, 1)


def local_candidate_cost_targets(
    proposal: Tensor,
    fidelity: Tensor,
    target: Tensor,
    candidates: tuple[float, ...],
    safe_index: int,
    *,
    patch_size: int = 8,
    epsilon: float = 1e-8,
    log_clip: float = 4.0,
) -> tuple[Tensor, Tensor]:
    """Training-only relative log-MSE targets and best candidate indices."""
    residual = fidelity - proposal
    alpha = candidate_alphas(candidates, proposal).view(1, len(candidates), 1, 1, 1)
    restored = proposal[:, None] + alpha * residual[:, None]
    pixel_error = (restored - target[:, None]).square().mean(2)
    local_mse = F.avg_pool2d(pixel_error, patch_size, patch_size)
    safe = local_mse[:, safe_index : safe_index + 1]
    relative = torch.log((local_mse + epsilon) / (safe + epsilon)).clamp(-log_clip, log_clip)
    return relative, local_mse.argmin(1)


def soft_action_output(
    proposal: Tensor,
    fidelity: Tensor,
    costs: Tensor,
    candidates: tuple[float, ...],
    *,
    temperature: float,
) -> tuple[Tensor, Tensor]:
    probabilities = torch.softmax(-costs / temperature, dim=1)
    alpha = (probabilities * candidate_alphas(candidates, costs)).sum(1, keepdim=True)
    pixels = alpha.repeat_interleave(8, -2).repeat_interleave(8, -1)
    return (proposal + pixels * (fidelity - proposal)).clamp(0, 1), alpha


def select_actions_fail_closed(
    costs: Tensor,
    *,
    safe_index: int,
    decision_margin: float,
) -> tuple[Tensor, Tensor]:
    """Choose an alternative only when it clearly beats the safe action."""
    best_cost, best_index = costs.min(1)
    safe_cost = costs[:, safe_index]
    accept = (best_index != safe_index) & (best_cost + decision_margin < safe_cost)
    selected = torch.where(accept, best_index, torch.full_like(best_index, safe_index))
    return selected, accept


def apply_selected_actions(
    proposal: Tensor,
    fidelity: Tensor,
    selected: Tensor,
    candidates: tuple[float, ...],
) -> Tensor:
    alpha_values = proposal.new_tensor(candidates)
    alpha = alpha_values[selected].unsqueeze(1)
    pixels = alpha.repeat_interleave(8, -2).repeat_interleave(8, -1)
    return (proposal + pixels * (fidelity - proposal)).clamp(0, 1)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
