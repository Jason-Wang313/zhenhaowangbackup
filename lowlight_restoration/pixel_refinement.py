"""Small bounded pixel-space correction attached after FLUX VAE decoding.

The module cannot generate a replacement image: it receives both the decoded
FLUX restoration and the aligned degraded input, predicts a strictly bounded
residual, and is zero-initialized to exact identity.  Its purpose is to recover
pixel frequencies that the frozen VAE cannot represent reliably.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as functional


class RefinementBlock(nn.Module):
    """Efficient NAF-style local block with a zero residual start."""

    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        if channels <= 0 or dilation <= 0:
            raise ValueError("channels and dilation must be positive")
        self.norm = nn.GroupNorm(1, channels)
        self.expand = nn.Conv2d(channels, channels * 2, 1)
        self.spatial = nn.Conv2d(
            channels * 2,
            channels * 2,
            3,
            padding=dilation,
            dilation=dilation,
            groups=channels * 2,
            padding_mode="reflect",
        )
        self.channel_scale = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid(),
        )
        self.project = nn.Conv2d(channels, channels, 1)
        self.residual_scale = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, value: Tensor) -> Tensor:
        first, second = self.spatial(self.expand(self.norm(value))).chunk(2, dim=1)
        residual = first * second
        residual = self.project(residual * self.channel_scale(residual))
        return value + residual * self.residual_scale


class BoundedPixelRefiner(nn.Module):
    """Predict a bounded correction from decoded FLUX output and aligned LQ."""

    def __init__(
        self,
        *,
        width: int = 32,
        blocks: int = 6,
        maximum_residual_m11: float = 0.25,
    ) -> None:
        super().__init__()
        if width <= 0 or blocks <= 0:
            raise ValueError("width and blocks must be positive")
        if not 0.0 < maximum_residual_m11 <= 0.5:
            raise ValueError("maximum_residual_m11 must be in (0, 0.5]")
        self.maximum_residual_m11 = float(maximum_residual_m11)
        # decoded, degraded input, and their aligned difference
        self.intro = nn.Conv2d(9, width, 3, padding=1, padding_mode="reflect")
        dilations = (1, 2, 3)
        self.blocks = nn.Sequential(
            *(RefinementBlock(width, dilations[index % len(dilations)]) for index in range(blocks))
        )
        self.ending = nn.Conv2d(width, 3, 3, padding=1, padding_mode="reflect")
        nn.init.zeros_(self.ending.weight)
        nn.init.zeros_(self.ending.bias)

    def forward(self, decoded: Tensor, degraded: Tensor) -> Tensor:
        if decoded.shape != degraded.shape or decoded.ndim != 4 or decoded.shape[1] != 3:
            raise ValueError("decoded and degraded tensors must be aligned BCHW RGB")
        features = torch.cat((decoded, degraded, degraded - decoded), dim=1)
        residual = torch.tanh(self.ending(self.blocks(self.intro(features))))
        return torch.clamp(decoded + residual * self.maximum_residual_m11, -1.0, 1.0)


class ObservationAnchoredRainHead(nn.Module):
    """Predict a bounded Rain correction while preserving an exact LQ skip.

    ``guidance`` is a frozen or trainable DiT restoration. Passing the input
    itself as guidance defines a parameter-matched image-only control.
    """

    def __init__(
        self, *, width: int = 32, blocks: int = 6, maximum_residual_m11: float = 0.5
    ) -> None:
        super().__init__()
        if width <= 0 or blocks <= 0:
            raise ValueError("width and blocks must be positive")
        if not 0.0 < maximum_residual_m11 <= 1.0:
            raise ValueError("maximum_residual_m11 must be in (0, 1]")
        self.maximum_residual_m11 = float(maximum_residual_m11)
        self.intro = nn.Conv2d(9, width, 3, padding=1, padding_mode="reflect")
        dilations = (1, 2, 3)
        self.blocks = nn.Sequential(
            *(RefinementBlock(width, dilations[index % len(dilations)]) for index in range(blocks))
        )
        self.ending = nn.Conv2d(width, 3, 3, padding=1, padding_mode="reflect")
        nn.init.zeros_(self.ending.weight)
        nn.init.zeros_(self.ending.bias)

    def forward(self, observation: Tensor, guidance: Tensor) -> Tensor:
        if observation.shape != guidance.shape or observation.ndim != 4 or observation.shape[1] != 3:
            raise ValueError("observation and guidance must be aligned BCHW RGB")
        features = torch.cat((observation, guidance, guidance - observation), dim=1)
        residual = torch.tanh(self.ending(self.blocks(self.intro(features))))
        return torch.clamp(observation + residual * self.maximum_residual_m11, -1.0, 1.0)


def spatial_highpass(value: Tensor) -> Tensor:
    """Return a fixed per-channel Laplacian response without trainable state."""
    if value.ndim != 4 or value.shape[1] != 3:
        raise ValueError("high-pass input must be BCHW RGB")
    kernel = value.new_tensor(
        [[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]]
    ).view(1, 1, 3, 3).repeat(3, 1, 1, 1)
    padded = functional.pad(value, (1, 1, 1, 1), mode="reflect")
    return functional.conv2d(padded, kernel, groups=3)


def pixel_refinement_losses(
    refined: Tensor,
    decoded: Tensor,
    target: Tensor,
    *,
    charbonnier_epsilon: float,
) -> dict[str, Tensor]:
    """Loss terms that reward fidelity while discouraging unnecessary edits."""
    if refined.shape != decoded.shape or refined.shape != target.shape:
        raise ValueError("refinement loss tensors must be aligned")
    difference = refined.float() - target.float()
    charbonnier = torch.sqrt(difference.square() + charbonnier_epsilon**2).mean()
    edge = functional.l1_loss(spatial_highpass(refined.float()), spatial_highpass(target.float()))
    residual = functional.l1_loss(refined.float(), decoded.float())
    return {"charbonnier": charbonnier, "edge": edge, "residual": residual}


def pixel_refiner_parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())
