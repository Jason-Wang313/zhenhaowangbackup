"""Blur-specific Restormer wrappers that preserve the frozen base at init."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


MOTION_RESTORMER_MLX_METADATA = {
    "source": "swz30/Restormer motion_deblurring.pth",
    "params": "26126644",
    "license": "MIT",
    "format": "pt",
    "layout": "MLX NHWC; conv (O,kH,kW,I)",
}


class BaseLQRestormerRefiner(nn.Module):
    """Let Restormer recover a residual using both its base and raw LQ evidence.

    The fusion convolution is initialized as an exact selector for ``base``.
    When Restormer's output head is zero, the complete wrapper is therefore an
    exact identity on the frozen base while gradients can subsequently learn
    how aligned LQ and base-minus-LQ information should affect the residual.
    """

    def __init__(self, restormer: nn.Module) -> None:
        super().__init__()
        self.restormer = restormer
        reference = next(restormer.parameters())
        self.fusion = nn.Conv2d(
            9,
            3,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            device=reference.device,
            dtype=reference.dtype,
        )
        nn.init.zeros_(self.fusion.weight)
        with torch.no_grad():
            for channel in range(3):
                self.fusion.weight[channel, channel, 1, 1] = 1.0

    def forward(self, base: torch.Tensor, lq: torch.Tensor) -> torch.Tensor:
        if (
            base.ndim != 4
            or lq.ndim != 4
            or base.shape != lq.shape
            or base.shape[1] != 3
        ):
            raise ValueError("Base/LQ Restormer inputs must be aligned BCHW RGB tensors")
        fused = self.fusion(torch.cat([base, lq, base - lq], dim=1))
        restored_fused = self.restormer(fused)
        if restored_fused.shape != base.shape:
            raise RuntimeError("Restormer changed the aligned base geometry")
        return base + (restored_fused - fused)


class BaseLQRestormerExpert(nn.Module):
    """Run a task-native Restormer on aligned evidence, then blend externally.

    The fusion convolution starts as an exact selector for raw ``lq``.  This
    preserves the input distribution expected by a motion-deblurring
    checkpoint while still allowing a small, faster-learning layer to mix in
    the frozen FoundIR base during domain adaptation.
    """

    def __init__(self, restormer: nn.Module) -> None:
        super().__init__()
        self.restormer = restormer
        reference = next(restormer.parameters())
        self.fusion = nn.Conv2d(
            9,
            3,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            device=reference.device,
            dtype=reference.dtype,
        )
        nn.init.zeros_(self.fusion.weight)
        with torch.no_grad():
            for channel in range(3):
                self.fusion.weight[channel, 3 + channel, 1, 1] = 1.0

    def forward(self, base: torch.Tensor, lq: torch.Tensor) -> torch.Tensor:
        _validate_aligned_inputs(base, lq)
        fused = self.fusion(torch.cat([base, lq, base - lq], dim=1))
        restored = self.restormer(fused)
        if restored.shape != base.shape:
            raise RuntimeError("Restormer changed the aligned base geometry")
        return restored


def _validate_aligned_inputs(base: torch.Tensor, lq: torch.Tensor) -> None:
    if (
        base.ndim != 4
        or lq.ndim != 4
        or base.shape != lq.shape
        or base.shape[1] != 3
    ):
        raise ValueError("Base/LQ Restormer inputs must be aligned BCHW RGB tensors")


def load_mlx_motion_restormer_weights(
    model: nn.Module, checkpoint: Path
) -> dict[str, int | str]:
    """Strictly convert the pinned MLX NHWC checkpoint back to PyTorch NCHW."""

    from safetensors import safe_open
    from safetensors.torch import load_file

    checkpoint = Path(checkpoint)
    with safe_open(checkpoint, framework="pt", device="cpu") as handle:
        metadata = handle.metadata()
    if metadata != MOTION_RESTORMER_MLX_METADATA:
        raise RuntimeError(f"motion Restormer metadata changed: {metadata}")

    source = load_file(checkpoint, device="cpu")
    converted, convolution_count = _convert_mlx_restormer_state_dict(model, source)
    model.load_state_dict(converted, strict=True)
    parameters = sum(tensor.numel() for tensor in converted.values())
    if parameters != 26_126_644 or len(converted) != 494 or convolution_count != 274:
        raise RuntimeError("motion Restormer tensor/parameter contract changed")
    return {
        "checkpoint": checkpoint.as_posix(),
        "tensors": len(converted),
        "parameters": parameters,
        "converted_convolutions": convolution_count,
    }


def _convert_mlx_restormer_state_dict(
    model: nn.Module, source: dict[str, torch.Tensor]
) -> tuple[dict[str, torch.Tensor], int]:
    expected = model.state_dict()
    if set(source) != set(expected):
        missing = sorted(set(expected) - set(source))
        unexpected = sorted(set(source) - set(expected))
        raise RuntimeError(
            "motion Restormer key contract changed: "
            f"missing={missing} unexpected={unexpected}"
        )

    convolution_weights = {
        f"{name}.weight" if name else "weight"
        for name, module in model.named_modules()
        if isinstance(module, nn.Conv2d)
    }
    converted = {}
    for key, tensor in source.items():
        value = tensor.permute(0, 3, 1, 2).contiguous() if key in convolution_weights else tensor
        if value.shape != expected[key].shape or value.dtype != expected[key].dtype:
            raise RuntimeError(
                f"motion Restormer tensor contract changed for {key}: "
                f"source={tuple(tensor.shape)}/{tensor.dtype} "
                f"converted={tuple(value.shape)}/{value.dtype} "
                f"expected={tuple(expected[key].shape)}/{expected[key].dtype}"
            )
        converted[key] = value
    return converted, len(convolution_weights)


def base_lq_restormer_parameter_count(
    model: BaseLQRestormerRefiner | BaseLQRestormerExpert,
) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
