"""Pinned task-native Restormer architecture and strict checkpoint loader.

Architecture adapted from ``swz30/Restormer`` at immutable revision
``68dc6ac472db26f16361150cb7a96a1bc87da93f``. The upstream implementation
is MIT licensed, copyright (c) 2022 Syed Waqas Zamir and contributors.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn


UPSTREAM_REVISION = "68dc6ac472db26f16361150cb7a96a1bc87da93f"
CHECKPOINT_MIRROR_REVISION = "aa2667c79d766b67da2c1489c7ce5f0423bae798"


@dataclass(frozen=True)
class TaskNativeRestormerSpec:
    filename: str
    sha256: str
    size_bytes: int
    tensors: int
    parameters: int
    layer_norm_type: str


TASK_NATIVE_RESTORMER_SPECS = {
    "deraining": TaskNativeRestormerSpec(
        filename="deraining.pth",
        sha256="225ed9acf80b3527d7080e5182918f642a23a78ef8593ad700b7e924bd660681",
        size_bytes=104_700_429,
        tensors=494,
        parameters=26_126_644,
        layer_norm_type="WithBias",
    ),
    "real_denoising": TaskNativeRestormerSpec(
        filename="real_denoising.pth",
        sha256="4cae18bed8a291b9a5deeeab48756bbe6f61fcf7f31a4cca0969b24608601387",
        size_bytes=104_611_957,
        tensors=406,
        parameters=26_111_668,
        layer_norm_type="BiasFree",
    ),
    "gaussian_color_denoising_blind": TaskNativeRestormerSpec(
        filename="gaussian_color_denoising_blind.pth",
        sha256="4e2314dae77bb907184da6e2ac625b921491d6b33de26a52512eff1c6bb37fbd",
        size_bytes=104_611_957,
        tensors=406,
        parameters=26_111_668,
        layer_norm_type="BiasFree",
    ),
}


def _to_tokens(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.flatten(2).transpose(1, 2)


def _to_image(tensor: torch.Tensor, height: int, width: int) -> torch.Tensor:
    return tensor.transpose(1, 2).reshape(tensor.shape[0], tensor.shape[2], height, width)


class BiasFreeLayerNorm(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dimension))

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        variance = tensor.var(-1, keepdim=True, unbiased=False)
        return tensor * torch.rsqrt(variance + 1e-5) * self.weight


class WithBiasLayerNorm(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dimension))
        self.bias = nn.Parameter(torch.zeros(dimension))

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        mean = tensor.mean(-1, keepdim=True)
        variance = tensor.var(-1, keepdim=True, unbiased=False)
        return (tensor - mean) * torch.rsqrt(variance + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dimension: int, layer_norm_type: str) -> None:
        super().__init__()
        if layer_norm_type == "BiasFree":
            self.body = BiasFreeLayerNorm(dimension)
        elif layer_norm_type == "WithBias":
            self.body = WithBiasLayerNorm(dimension)
        else:
            raise ValueError(f"unsupported Restormer LayerNorm type: {layer_norm_type}")

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        height, width = tensor.shape[-2:]
        return _to_image(self.body(_to_tokens(tensor)), height, width)


class FeedForward(nn.Module):
    def __init__(self, dimension: int, expansion: float, bias: bool) -> None:
        super().__init__()
        hidden = int(dimension * expansion)
        self.project_in = nn.Conv2d(dimension, hidden * 2, 1, bias=bias)
        self.dwconv = nn.Conv2d(
            hidden * 2,
            hidden * 2,
            3,
            stride=1,
            padding=1,
            groups=hidden * 2,
            bias=bias,
        )
        self.project_out = nn.Conv2d(hidden, dimension, 1, bias=bias)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        first, second = self.dwconv(self.project_in(tensor)).chunk(2, dim=1)
        return self.project_out(F.gelu(first) * second)


class Attention(nn.Module):
    def __init__(self, dimension: int, heads: int, bias: bool) -> None:
        super().__init__()
        if dimension % heads:
            raise ValueError("Restormer attention dimension must divide evenly across heads")
        self.num_heads = heads
        self.temperature = nn.Parameter(torch.ones(heads, 1, 1))
        self.qkv = nn.Conv2d(dimension, dimension * 3, 1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dimension * 3,
            dimension * 3,
            3,
            stride=1,
            padding=1,
            groups=dimension * 3,
            bias=bias,
        )
        self.project_out = nn.Conv2d(dimension, dimension, 1, bias=bias)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = tensor.shape
        q, k, v = self.qkv_dwconv(self.qkv(tensor)).chunk(3, dim=1)
        channels_per_head = channels // self.num_heads

        def heads(value: torch.Tensor) -> torch.Tensor:
            return value.reshape(
                batch, self.num_heads, channels_per_head, height * width
            )

        q, k, v = heads(q), heads(k), heads(v)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attention = ((q @ k.transpose(-2, -1)) * self.temperature).softmax(dim=-1)
        output = (attention @ v).reshape(batch, channels, height, width)
        return self.project_out(output)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dimension: int,
        heads: int,
        expansion: float,
        bias: bool,
        layer_norm_type: str,
    ) -> None:
        super().__init__()
        self.norm1 = LayerNorm(dimension, layer_norm_type)
        self.attn = Attention(dimension, heads, bias)
        self.norm2 = LayerNorm(dimension, layer_norm_type)
        self.ffn = FeedForward(dimension, expansion, bias)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = tensor + self.attn(self.norm1(tensor))
        return tensor + self.ffn(self.norm2(tensor))


class OverlapPatchEmbed(nn.Module):
    def __init__(self, input_channels: int, dimension: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(input_channels, dimension, 3, stride=1, padding=1, bias=False)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.proj(tensor)


class Downsample(nn.Module):
    def __init__(self, features: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(features, features // 2, 3, stride=1, padding=1, bias=False),
            nn.PixelUnshuffle(2),
        )

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.body(tensor)


class Upsample(nn.Module):
    def __init__(self, features: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(features, features * 2, 3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2),
        )

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.body(tensor)


def _blocks(
    count: int,
    dimension: int,
    heads: int,
    expansion: float,
    bias: bool,
    layer_norm_type: str,
) -> nn.Sequential:
    return nn.Sequential(
        *(
            TransformerBlock(dimension, heads, expansion, bias, layer_norm_type)
            for _ in range(count)
        )
    )


class Restormer(nn.Module):
    def __init__(
        self,
        input_channels: int = 3,
        output_channels: int = 3,
        dimension: int = 48,
        num_blocks: Sequence[int] = (4, 6, 6, 8),
        refinement_blocks: int = 4,
        heads: Sequence[int] = (1, 2, 4, 8),
        expansion: float = 2.66,
        bias: bool = False,
        layer_norm_type: str = "WithBias",
        dual_pixel_task: bool = False,
    ) -> None:
        super().__init__()
        if len(num_blocks) != 4 or len(heads) != 4:
            raise ValueError("Restormer requires four encoder levels")
        self.patch_embed = OverlapPatchEmbed(input_channels, dimension)
        self.encoder_level1 = _blocks(
            num_blocks[0], dimension, heads[0], expansion, bias, layer_norm_type
        )
        self.down1_2 = Downsample(dimension)
        self.encoder_level2 = _blocks(
            num_blocks[1], dimension * 2, heads[1], expansion, bias, layer_norm_type
        )
        self.down2_3 = Downsample(dimension * 2)
        self.encoder_level3 = _blocks(
            num_blocks[2], dimension * 4, heads[2], expansion, bias, layer_norm_type
        )
        self.down3_4 = Downsample(dimension * 4)
        self.latent = _blocks(
            num_blocks[3], dimension * 8, heads[3], expansion, bias, layer_norm_type
        )
        self.up4_3 = Upsample(dimension * 8)
        self.reduce_chan_level3 = nn.Conv2d(dimension * 8, dimension * 4, 1, bias=bias)
        self.decoder_level3 = _blocks(
            num_blocks[2], dimension * 4, heads[2], expansion, bias, layer_norm_type
        )
        self.up3_2 = Upsample(dimension * 4)
        self.reduce_chan_level2 = nn.Conv2d(dimension * 4, dimension * 2, 1, bias=bias)
        self.decoder_level2 = _blocks(
            num_blocks[1], dimension * 2, heads[1], expansion, bias, layer_norm_type
        )
        self.up2_1 = Upsample(dimension * 2)
        self.decoder_level1 = _blocks(
            num_blocks[0], dimension * 2, heads[0], expansion, bias, layer_norm_type
        )
        self.refinement = _blocks(
            refinement_blocks, dimension * 2, heads[0], expansion, bias, layer_norm_type
        )
        self.dual_pixel_task = dual_pixel_task
        if dual_pixel_task:
            self.skip_conv = nn.Conv2d(dimension, dimension * 2, 1, bias=bias)
        self.output = nn.Conv2d(dimension * 2, output_channels, 3, stride=1, padding=1, bias=bias)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError("task-native Restormer expects BCHW RGB input")
        if image.shape[-2] % 8 or image.shape[-1] % 8:
            raise ValueError("task-native Restormer spatial dimensions must be divisible by 8")
        encoder1 = self.encoder_level1(self.patch_embed(image))
        encoder2 = self.encoder_level2(self.down1_2(encoder1))
        encoder3 = self.encoder_level3(self.down2_3(encoder2))
        latent = self.latent(self.down3_4(encoder3))
        decoder3 = self.up4_3(latent)
        decoder3 = self.decoder_level3(
            self.reduce_chan_level3(torch.cat((decoder3, encoder3), dim=1))
        )
        decoder2 = self.up3_2(decoder3)
        decoder2 = self.decoder_level2(
            self.reduce_chan_level2(torch.cat((decoder2, encoder2), dim=1))
        )
        decoder1 = self.up2_1(decoder2)
        decoder1 = self.refinement(
            self.decoder_level1(torch.cat((decoder1, encoder1), dim=1))
        )
        if self.dual_pixel_task:
            return self.output(decoder1 + self.skip_conv(encoder1))
        return self.output(decoder1) + image


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_task_native_restormer(
    name: str,
    directory: Path,
) -> tuple[Restormer, dict[str, object]]:
    """Load one pinned checkpoint only after its full binary/tensor contract passes."""

    if name not in TASK_NATIVE_RESTORMER_SPECS:
        raise ValueError(f"unknown task-native Restormer checkpoint: {name}")
    spec = TASK_NATIVE_RESTORMER_SPECS[name]
    path = Path(directory) / spec.filename
    if not path.is_file() or path.stat().st_size != spec.size_bytes:
        raise RuntimeError(f"task-native Restormer size contract changed: {path}")
    actual_sha = _file_sha256(path)
    if actual_sha != spec.sha256:
        raise RuntimeError(f"task-native Restormer SHA-256 changed: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if type(payload) is not dict or list(payload) != ["params"]:
        raise RuntimeError(f"task-native Restormer root contract changed: {path}")
    state = payload["params"]
    if not isinstance(state, Mapping) or len(state) != spec.tensors:
        raise RuntimeError(f"task-native Restormer tensor-count contract changed: {path}")
    if any(
        type(key) is not str
        or not torch.is_tensor(value)
        or value.dtype != torch.float32
        or not bool(torch.isfinite(value).all())
        for key, value in state.items()
    ):
        raise RuntimeError(f"task-native Restormer tensor contract changed: {path}")
    if sum(value.numel() for value in state.values()) != spec.parameters:
        raise RuntimeError(f"task-native Restormer parameter contract changed: {path}")
    model = Restormer(layer_norm_type=spec.layer_norm_type)
    expected = model.state_dict()
    if set(state) != set(expected) or any(
        state[key].shape != expected[key].shape or state[key].dtype != expected[key].dtype
        for key in expected
    ):
        missing = sorted(set(expected) - set(state))
        unexpected = sorted(set(state) - set(expected))
        raise RuntimeError(
            "task-native Restormer architecture contract changed: "
            f"missing={missing} unexpected={unexpected}"
        )
    model.load_state_dict(state, strict=True)
    return model, {
        "name": name,
        "checkpoint": path.as_posix(),
        "checkpoint_sha256": actual_sha,
        "checkpoint_mirror_revision": CHECKPOINT_MIRROR_REVISION,
        "architecture_revision": UPSTREAM_REVISION,
        "layer_norm_type": spec.layer_norm_type,
        "tensors": len(state),
        "parameters": sum(value.numel() for value in state.values()),
    }
