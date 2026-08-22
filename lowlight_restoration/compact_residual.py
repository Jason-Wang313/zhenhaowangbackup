"""Compact deterministic low-light residual control used by Gate 3.

The global tone path is evaluated once from a low-resolution view of the
complete image.  Its parameters are then reused for every native-resolution
tile, preventing tile-dependent exposure and white-balance changes.  The local
path predicts only a bounded RGB residual and is initialized to exact identity.
"""

from __future__ import annotations

import math
from typing import Iterable

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    def __init__(self, channels: int, epsilon: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.epsilon = epsilon

    def forward(self, value: Tensor) -> Tensor:
        mean = value.mean(dim=1, keepdim=True)
        variance = (value - mean).square().mean(dim=1, keepdim=True)
        return (value - mean) * torch.rsqrt(variance + self.epsilon) * self.weight + self.bias


class NAFBlock(nn.Module):
    """Small activation-free restoration block with zero residual starts."""

    def __init__(self, channels: int, expansion: int = 2) -> None:
        super().__init__()
        hidden = channels * expansion
        if hidden % 2:
            raise ValueError("expanded channels must be even")
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
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    @staticmethod
    def gate(value: Tensor) -> Tensor:
        first, second = value.chunk(2, dim=1)
        return first * second

    def forward(self, value: Tensor) -> Tensor:
        residual = self.depthwise(self.in_project(self.norm1(value)))
        residual = self.gate(residual)
        residual = residual * self.channel_scale(residual)
        value = value + self.out_project(residual) * self.beta
        residual = self.gate(self.ffn_in(self.norm2(value)))
        return value + self.ffn_out(residual) * self.gamma


def _stack(channels: int, count: int) -> nn.Sequential:
    return nn.Sequential(*(NAFBlock(channels) for _ in range(count)))


class GlobalToneHead(nn.Module):
    """Predict bounded per-channel exposure, gamma, and offset parameters."""

    def __init__(self, width: int = 24, condition_dim: int = 0) -> None:
        super().__init__()
        self.condition_dim = condition_dim
        self.features = nn.Sequential(
            nn.Conv2d(3, width, 5, stride=2, padding=2),
            nn.SiLU(),
            nn.Conv2d(width, width * 2, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(width * 2, width * 4, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.output = nn.Linear(width * 4, 9)
        self.condition = nn.Linear(condition_dim, width * 4, bias=False) if condition_dim else None
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        if self.condition is not None:
            nn.init.zeros_(self.condition.weight)

    def forward(self, global_view: Tensor, condition: Tensor | None = None) -> Tensor:
        features = self.features(global_view).flatten(1)
        if self.condition is not None:
            if condition is None or condition.shape[1] != self.condition_dim:
                raise ValueError("the frozen condition vector is missing or has the wrong width")
            features = features + self.condition(condition)
        elif condition is not None:
            raise ValueError("an unconditional tone head cannot receive a condition")
        return self.output(features)

    @staticmethod
    def apply(image: Tensor, parameters: Tensor) -> Tensor:
        if parameters.ndim != 2 or parameters.shape[1] != 9:
            raise ValueError("tone parameters must have shape [batch, 9]")
        log_gain, log_gamma, offset = parameters.chunk(3, dim=1)
        gain = torch.exp(torch.tanh(log_gain) * math.log(8.0)).view(-1, 3, 1, 1)
        gamma = torch.exp(torch.tanh(log_gamma) * math.log(4.0)).view(-1, 3, 1, 1)
        offset = (torch.tanh(offset) * 0.15).view(-1, 3, 1, 1)
        powered = image * torch.exp(
            (gamma - 1.0) * torch.log(image.clamp_min(1.0 / 65536.0))
        )
        return torch.clamp(powered * gain + offset, 0.0, 1.0)


class CompactResidualControl(nn.Module):
    """Three-scale residual U-Net with a shared global photometric path."""

    def __init__(
        self,
        width: int = 32,
        encoder_blocks: Iterable[int] = (2, 2),
        middle_blocks: int = 4,
        decoder_blocks: Iterable[int] = (2, 2),
        global_width: int = 24,
        condition_dim: int = 0,
        local_residual_bound: float = 0.25,
    ) -> None:
        super().__init__()
        encoder_counts = tuple(encoder_blocks)
        decoder_counts = tuple(decoder_blocks)
        if encoder_counts != decoder_counts or len(encoder_counts) != 2:
            raise ValueError("Gate 3 freezes a symmetric two-level compact control")
        self.local_residual_bound = float(local_residual_bound)
        self.tone = GlobalToneHead(global_width, condition_dim=condition_dim)
        self.intro = nn.Conv2d(6, width, 3, padding=1)
        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        channels = width
        for count in encoder_counts:
            self.encoders.append(_stack(channels, count))
            self.downs.append(nn.Conv2d(channels, channels * 2, 2, stride=2))
            channels *= 2
        self.middle = _stack(channels, middle_blocks)
        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for count in reversed(decoder_counts):
            self.ups.append(nn.Sequential(nn.Conv2d(channels, channels * 2, 1), nn.PixelShuffle(2)))
            channels //= 2
            self.decoders.append(_stack(channels, count))
        self.ending = nn.Conv2d(width, 3, 3, padding=1)
        nn.init.zeros_(self.ending.weight)
        nn.init.zeros_(self.ending.bias)

    def tone_parameters(self, global_view: Tensor, condition: Tensor | None = None) -> Tensor:
        return self.tone(global_view, condition)

    def forward(
        self,
        image: Tensor,
        global_view: Tensor | None = None,
        condition: Tensor | None = None,
        tone_parameters: Tensor | None = None,
    ) -> Tensor:
        if global_view is None:
            global_view = F.interpolate(image, size=(128, 128), mode="bilinear", align_corners=False)
        if tone_parameters is None:
            tone_parameters = self.tone_parameters(global_view, condition)
        base = self.tone.apply(image, tone_parameters)
        value = self.intro(torch.cat((image, base), dim=1))
        skips = []
        for encoder, down in zip(self.encoders, self.downs):
            value = encoder(value)
            skips.append(value)
            value = down(value)
        value = self.middle(value)
        for up, decoder, skip in zip(self.ups, self.decoders, reversed(skips)):
            value = up(value) + skip
            value = decoder(value)
        residual = torch.tanh(self.ending(value)) * self.local_residual_bound
        return torch.clamp(base + residual, 0.0, 1.0)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
