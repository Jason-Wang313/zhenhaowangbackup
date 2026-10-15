"""Clean-room OneRestore-style modules for the bounded R0 diagnostic."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import vgg16


CANONICAL_DEGRADATIONS = ("blur", "noise", "rain", "haze", "low_light")
CATEGORY_NAMES = ("clean",) + CANONICAL_DEGRADATIONS + tuple(
    f"{left}+{right}"
    for index, left in enumerate(CANONICAL_DEGRADATIONS)
    for right in CANONICAL_DEGRADATIONS[index + 1 :]
)
CATEGORY_TO_INDEX = {name: index for index, name in enumerate(CATEGORY_NAMES)}


def category_name(degradations: Iterable[str]) -> str:
    active = set(map(str, degradations))
    unknown = active - set(CANONICAL_DEGRADATIONS)
    if unknown:
        raise ValueError(f"unknown degradations: {sorted(unknown)}")
    ordered = tuple(name for name in CANONICAL_DEGRADATIONS if name in active)
    if len(ordered) > 2:
        # Held-out triples have no trained categorical token. The fixed R0
        # baseline uses the normalized mean of its singleton tokens.
        return "+".join(ordered)
    return "clean" if not ordered else "+".join(ordered)


def make_descriptor_table(classes: int, dimension: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    table = torch.randn(classes, dimension, generator=generator)
    return F.normalize(table, dim=1)


class ChannelLayerNorm(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        variance = x.var(dim=1, keepdim=True, unbiased=False)
        return (x - mean) * torch.rsqrt(variance + 1e-5) * self.weight[None, :, None, None] + self.bias[None, :, None, None]


class DescriptorCrossAttention(nn.Module):
    def __init__(self, channels: int, heads: int, descriptor_dim: int) -> None:
        super().__init__()
        side = math.isqrt(descriptor_dim)
        if side * side != descriptor_dim or channels % heads:
            raise ValueError("descriptor dimension must be square and channels divisible by heads")
        self.channels = channels
        self.heads = heads
        self.side = side
        self.temperature = nn.Parameter(torch.ones(heads, 1, 1))
        self.query = nn.Linear(descriptor_dim, descriptor_dim, bias=False)
        self.key_value = nn.Conv2d(channels, channels * 2, 1, bias=False)
        self.key_value_depthwise = nn.Conv2d(
            channels * 2, channels * 2, 3, padding=1, groups=channels * 2, bias=False
        )
        self.project = nn.Conv2d(channels, channels, 1, bias=False)

    def forward(self, x: torch.Tensor, descriptor: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        head_channels = channels // self.heads
        query = self.query(descriptor).view(batch, 1, 1, -1).expand(-1, self.heads, head_channels, -1)
        key, value = self.key_value_depthwise(self.key_value(x)).chunk(2, dim=1)
        key = F.interpolate(key, size=(self.side, self.side), mode="bilinear", align_corners=False)
        key = key.view(batch, self.heads, head_channels, -1)
        value = value.view(batch, self.heads, head_channels, height * width)
        query = F.normalize(query, dim=-1)
        key = F.normalize(key, dim=-1)
        attention = torch.matmul(query, key.transpose(-2, -1)) * self.temperature
        output = torch.matmul(attention.softmax(dim=-1), value)
        return self.project(output.reshape(batch, channels, height, width))


class ChannelSelfAttention(nn.Module):
    def __init__(self, channels: int, heads: int) -> None:
        super().__init__()
        if channels % heads:
            raise ValueError("channels must be divisible by heads")
        self.heads = heads
        self.temperature = nn.Parameter(torch.ones(heads, 1, 1))
        self.qkv = nn.Conv2d(channels, channels * 3, 1, bias=False)
        self.qkv_depthwise = nn.Conv2d(
            channels * 3, channels * 3, 3, padding=1, groups=channels * 3, bias=False
        )
        self.project = nn.Conv2d(channels, channels, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        head_channels = channels // self.heads
        query, key, value = self.qkv_depthwise(self.qkv(x)).chunk(3, dim=1)
        query = F.normalize(query.view(batch, self.heads, head_channels, -1), dim=-1)
        key = F.normalize(key.view(batch, self.heads, head_channels, -1), dim=-1)
        value = value.view(batch, self.heads, head_channels, -1)
        attention = torch.matmul(query, key.transpose(-2, -1)) * self.temperature
        output = torch.matmul(attention.softmax(dim=-1), value)
        return self.project(output.reshape(batch, channels, height, width))


class GatedFeedForward(nn.Module):
    def __init__(self, channels: int, expansion: float = 2.66) -> None:
        super().__init__()
        hidden = int(channels * expansion)
        self.project_in = nn.Conv2d(channels, hidden * 2, 1, bias=False)
        self.depthwise = nn.Conv2d(hidden * 2, hidden * 2, 3, padding=1, groups=hidden * 2, bias=False)
        self.project_out = nn.Conv2d(hidden, channels, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        left, right = self.depthwise(self.project_in(x)).chunk(2, dim=1)
        return self.project_out(F.gelu(left) * right)


class DescriptorTransformerBlock(nn.Module):
    def __init__(self, channels: int, descriptor_dim: int, heads: int = 8) -> None:
        super().__init__()
        self.norm_cross = ChannelLayerNorm(channels)
        self.cross = DescriptorCrossAttention(channels, heads, descriptor_dim)
        self.norm_self = ChannelLayerNorm(channels)
        self.self_attention = ChannelSelfAttention(channels, heads)
        self.norm_ffn = ChannelLayerNorm(channels)
        self.ffn = GatedFeedForward(channels)

    def forward(self, x: torch.Tensor, descriptor: torch.Tensor, descriptor_scale: float) -> torch.Tensor:
        x = x + self.cross(self.norm_cross(x), descriptor) * descriptor_scale
        x = x + self.self_attention(self.norm_self(x))
        return x + self.ffn(self.norm_ffn(x))


class OneRestoreR0(nn.Module):
    def __init__(
        self,
        channel: int = 32,
        descriptor_dim: int = 324,
        descriptor_classes: int = 16,
        descriptor_seed: int = 2026092602,
    ) -> None:
        super().__init__()
        self.descriptor_dim = descriptor_dim
        self.register_buffer(
            "descriptor_table",
            make_descriptor_table(descriptor_classes, descriptor_dim, descriptor_seed),
            persistent=True,
        )
        self.input_projection = nn.Conv2d(3, channel, 1, bias=False)
        dims = (channel, channel * 2, channel * 4, channel * 8)
        self.encoder_blocks = nn.ModuleList(DescriptorTransformerBlock(dim, descriptor_dim) for dim in dims)
        self.down_projections = nn.ModuleList(nn.Conv2d(dims[i], dims[i + 1], 1, bias=False) for i in range(3))
        self.middle = nn.ModuleList(DescriptorTransformerBlock(dims[-1], descriptor_dim) for _ in range(2))
        self.decoder_blocks = nn.ModuleList(
            DescriptorTransformerBlock(dim, descriptor_dim) for dim in reversed(dims)
        )
        reversed_dims = tuple(reversed(dims))
        self.up_projections = nn.ModuleList(
            nn.Conv2d(reversed_dims[i], reversed_dims[i + 1], 1, bias=False) for i in range(3)
        )
        self.output_projection = nn.Conv2d(channel, 3, 1, bias=False)

    def descriptors(self, category_indices: torch.Tensor, degradations: list[list[str]] | None = None) -> torch.Tensor:
        if degradations is None:
            return self.descriptor_table[category_indices]
        rows = []
        for index, names in zip(category_indices.tolist(), degradations, strict=True):
            if len(names) <= 2:
                rows.append(self.descriptor_table[index])
            else:
                singleton_indices = [CATEGORY_TO_INDEX[name] for name in names]
                rows.append(F.normalize(self.descriptor_table[singleton_indices].mean(dim=0), dim=0))
        return torch.stack(rows)

    def forward(
        self,
        image: torch.Tensor,
        category_indices: torch.Tensor,
        *,
        descriptor_enabled: bool,
        degradations: list[list[str]] | None = None,
    ) -> torch.Tensor:
        descriptor = self.descriptors(category_indices, degradations)
        scale = 1.0 if descriptor_enabled else 0.0
        features = self.input_projection(image * 2.0 - 1.0)
        skips = []
        for index, block in enumerate(self.encoder_blocks):
            features = block(features, descriptor, scale)
            skips.append(features)
            if index < len(self.down_projections):
                features = self.down_projections[index](F.max_pool2d(features, 3, stride=2, padding=1))
        for block in self.middle:
            features = block(features, descriptor, scale)
        for index, block in enumerate(self.decoder_blocks):
            skip = skips[-1 - index]
            if features.shape[-2:] != skip.shape[-2:]:
                features = F.interpolate(features, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            features = block(features + skip, descriptor, scale)
            if index < len(self.up_projections):
                features = self.up_projections[index](features)
        return (image + self.output_projection(features)).clamp(0.0, 1.0)


class VGG16FeatureLoss(nn.Module):
    def __init__(self, weight_path: Path) -> None:
        super().__init__()
        model = vgg16(weights=None)
        state = torch.load(weight_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        self.features = model.features[:16].eval()
        for parameter in self.features.parameters():
            parameter.requires_grad_(False)
        self.capture = {3, 8, 15}

    def encode(self, image: torch.Tensor) -> list[torch.Tensor]:
        encoded = []
        for index, module in enumerate(self.features):
            image = module(image)
            if index in self.capture:
                encoded.append(image)
        return encoded

    def forward(
        self,
        degraded: torch.Tensor,
        clean: torch.Tensor,
        negative: torch.Tensor,
        restored: torch.Tensor,
    ) -> torch.Tensor:
        clean_features = self.encode(clean)
        with torch.no_grad():
            degraded_features = self.encode(degraded)
            negative_features = self.encode(negative)
        restored_features = self.encode(restored)
        terms = []
        for restored_layer, clean_layer, degraded_layer, negative_layer in zip(
            restored_features, clean_features, degraded_features, negative_features, strict=True
        ):
            positive = F.l1_loss(restored_layer, clean_layer)
            denominator = 0.5 * (
                F.l1_loss(restored_layer, degraded_layer) + F.l1_loss(restored_layer, negative_layer)
            )
            terms.append(positive / (denominator + 1e-7))
        return torch.stack(terms).mean()


def _gaussian_window(size: int, channels: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    coordinates = torch.arange(size, device=device, dtype=dtype) - size // 2
    vector = torch.exp(-(coordinates.square()) / (2 * 1.5**2))
    vector = vector / vector.sum()
    window = torch.outer(vector, vector)
    return window.expand(channels, 1, size, size).contiguous()


def ssim_components(left: torch.Tensor, right: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    channels = left.shape[1]
    size = min(11, left.shape[-2], left.shape[-1])
    window = _gaussian_window(size, channels, left.device, left.dtype)
    mean_left = F.conv2d(left, window, groups=channels)
    mean_right = F.conv2d(right, window, groups=channels)
    variance_left = F.conv2d(left * left, window, groups=channels) - mean_left.square()
    variance_right = F.conv2d(right * right, window, groups=channels) - mean_right.square()
    covariance = F.conv2d(left * right, window, groups=channels) - mean_left * mean_right
    c1, c2 = 0.01**2, 0.03**2
    contrast = (2 * covariance + c2) / (variance_left + variance_right + c2)
    score = ((2 * mean_left * mean_right + c1) * (2 * covariance + c2)) / (
        (mean_left.square() + mean_right.square() + c1) * (variance_left + variance_right + c2)
    )
    return score.mean(), contrast.mean()


def ms_ssim(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    weights = left.new_tensor((0.0448, 0.2856, 0.3001, 0.2363, 0.1333))
    scores, contrasts = [], []
    for level in range(len(weights)):
        score, contrast = ssim_components(left, right)
        scores.append(((score + 1.0) / 2.0).clamp_min(1e-6))
        contrasts.append(((contrast + 1.0) / 2.0).clamp_min(1e-6))
        if level + 1 < len(weights):
            left = F.avg_pool2d(left, 2)
            right = F.avg_pool2d(right, 2)
    result = scores[-1].pow(weights[-1])
    for index in range(len(weights) - 1):
        result = result * contrasts[index].pow(weights[index])
    return result


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
