"""Small identity-initialized adapter for fusing frozen expert trajectories."""

from __future__ import annotations

import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor + self.body(tensor)


class StageTrajectoryCompatibilityAdapter(nn.Module):
    """Fuse LQ and three frozen expert states while preserving identity at init.

    Inputs are four RGB states in [0, 1] plus three spatially constant presence
    masks (Noise, Low-light, Blur).  The fourth RGB state is always the frozen
    route's final output and is the exact initialization output.
    """

    def __init__(self, channels: int = 48, blocks: int = 6, maximum_residual: float = 0.25) -> None:
        super().__init__()
        if channels <= 0 or blocks <= 0 or not 0.0 < maximum_residual <= 1.0:
            raise ValueError("invalid compatibility-adapter specification")
        self.maximum_residual = float(maximum_residual)
        self.stem = nn.Conv2d(15, channels, 3, padding=1)
        self.blocks = nn.Sequential(*(ResidualBlock(channels) for _ in range(blocks)))
        self.head = nn.Conv2d(channels, 3, 3, padding=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(
        self,
        lq: torch.Tensor,
        after_noise: torch.Tensor,
        after_low_light: torch.Tensor,
        after_blur: torch.Tensor,
        presence: torch.Tensor,
    ) -> torch.Tensor:
        shapes = {tuple(value.shape) for value in (lq, after_noise, after_low_light, after_blur)}
        if len(shapes) != 1 or lq.ndim != 4 or lq.shape[1] != 3:
            raise ValueError("trajectory RGB tensors must share BCHW shape")
        if presence.shape != (lq.shape[0], 3):
            raise ValueError("presence must have shape [batch, 3]")
        masks = presence[:, :, None, None].expand(-1, -1, lq.shape[2], lq.shape[3])
        features = torch.cat((lq, after_noise, after_low_light, after_blur, masks), dim=1)
        residual = torch.tanh(self.head(self.blocks(self.stem(features))))
        return (after_blur + self.maximum_residual * residual).clamp(0.0, 1.0)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
