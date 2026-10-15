"""Compact multi-scale pixel expert for condition-routed deblurring."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn

from .compact_residual import NAFBlock


def blocks(channels: int, count: int) -> nn.Sequential:
    if count < 1:
        raise ValueError("each deblur stage needs at least one block")
    return nn.Sequential(*(NAFBlock(channels) for _ in range(count)))


class DeblurPixelExpert(nn.Module):
    """Three-scale residual U-Net initialized as an exact identity mapping."""

    def __init__(
        self,
        *,
        width: int = 48,
        encoder_blocks: Sequence[int] = (2, 2),
        middle_blocks: int = 12,
        decoder_blocks: Sequence[int] = (2, 2),
        residual_bound: float = 0.5,
    ) -> None:
        super().__init__()
        encoder_counts = tuple(int(value) for value in encoder_blocks)
        decoder_counts = tuple(int(value) for value in decoder_blocks)
        if len(encoder_counts) != 2 or encoder_counts != decoder_counts:
            raise ValueError("deblur expert requires a symmetric two-level U-Net")
        if width < 8 or middle_blocks < 1 or not 0.0 < residual_bound <= 1.0:
            raise ValueError("invalid deblur expert capacity or residual bound")
        self.residual_bound = float(residual_bound)
        self.intro = nn.Conv2d(3, width, 3, padding=1, padding_mode="reflect")
        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        channels = width
        for count in encoder_counts:
            self.encoders.append(blocks(channels, count))
            self.downs.append(nn.Conv2d(channels, channels * 2, 2, stride=2))
            channels *= 2
        self.middle = blocks(channels, middle_blocks)
        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for count in reversed(decoder_counts):
            self.ups.append(
                nn.Sequential(nn.Conv2d(channels, channels * 2, 1), nn.PixelShuffle(2))
            )
            channels //= 2
            self.decoders.append(blocks(channels, count))
        self.ending = nn.Conv2d(width, 3, 3, padding=1, padding_mode="reflect")
        nn.init.zeros_(self.ending.weight)
        nn.init.zeros_(self.ending.bias)

    def forward(self, image: Tensor) -> Tensor:
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError("deblur expert input must be BCHW RGB")
        if image.shape[-2] % 4 or image.shape[-1] % 4:
            raise ValueError("deblur expert spatial dimensions must divide by four")
        value = self.intro(image)
        skips = []
        for encoder, down in zip(self.encoders, self.downs, strict=True):
            value = encoder(value)
            skips.append(value)
            value = down(value)
        value = self.middle(value)
        for up, decoder, skip in zip(self.ups, self.decoders, reversed(skips), strict=True):
            value = decoder(up(value) + skip)
        residual = torch.tanh(self.ending(value)) * self.residual_bound
        return torch.clamp(image + residual, 0.0, 1.0)


def deblur_expert_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
