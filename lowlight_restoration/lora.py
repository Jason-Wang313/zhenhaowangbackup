"""Minimal auditable LoRA implementation for FLUX.2 Linear layers."""

from __future__ import annotations

import fnmatch
import json
import math
import os
import tempfile
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from torch import Tensor, nn


class LoRALinear(nn.Module):
    """Frozen Linear plus a zero-initialized low-rank residual."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float = 0.0) -> None:
        super().__init__()
        if rank <= 0 or alpha <= 0:
            raise ValueError("rank and alpha must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.base = base
        self.rank = rank
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.dropout = nn.Dropout(dropout)
        self.lora_A = nn.Linear(
            base.in_features,
            rank,
            bias=False,
            device=base.weight.device,
            dtype=base.weight.dtype,
        )
        self.lora_B = nn.Linear(
            rank,
            base.out_features,
            bias=False,
            device=base.weight.device,
            dtype=base.weight.dtype,
        )
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        self.base.requires_grad_(False)

    def forward(self, value: Tensor) -> Tensor:
        base_output = self.base(value)
        residual = self.lora_B(self.lora_A(self.dropout(value)))
        return base_output + residual.to(base_output) * self.scaling


def _matches(name: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatchcase(name, pattern) for pattern in patterns)


def inject_lora(
    model: nn.Module,
    patterns: list[str],
    rank: int,
    alpha: float,
    dropout: float = 0.0,
) -> list[str]:
    """Freeze a model and replace exactly matching Linear modules with LoRA."""

    model.requires_grad_(False)
    candidates = [
        name
        for name, module in model.named_modules()
        if name and isinstance(module, nn.Linear) and _matches(name, patterns)
    ]
    if not candidates:
        raise ValueError(f"LoRA patterns matched no Linear modules: {patterns}")
    for name in candidates:
        parent_name, child_name = name.rsplit(".", 1) if "." in name else ("", name)
        parent = model.get_submodule(parent_name) if parent_name else model
        base = getattr(parent, child_name)
        if not isinstance(base, nn.Linear):
            raise TypeError(f"LoRA target changed during injection: {name}")
        setattr(parent, child_name, LoRALinear(base, rank=rank, alpha=alpha, dropout=dropout))
    return candidates


def lora_modules(model: nn.Module) -> dict[str, LoRALinear]:
    return {name: module for name, module in model.named_modules() if isinstance(module, LoRALinear)}


def lora_state_dict(model: nn.Module) -> dict[str, Tensor]:
    state: dict[str, Tensor] = {}
    for name, module in lora_modules(model).items():
        state[f"{name}.lora_A.weight"] = module.lora_A.weight.detach().cpu().contiguous()
        state[f"{name}.lora_B.weight"] = module.lora_B.weight.detach().cpu().contiguous()
    if not state:
        raise ValueError("model has no LoRA modules")
    return state


def save_lora_checkpoint(model: nn.Module, path: Path, metadata: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {str(key): str(value) for key, value in metadata.items()}
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        save_file(lora_state_dict(model), str(temporary), metadata=payload)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)

    metadata_path = path.with_suffix(path.suffix + ".json")
    encoded = json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    temporary_metadata = metadata_path.with_name(f".{metadata_path.name}.tmp")
    temporary_metadata.write_text(encoded)
    os.replace(temporary_metadata, metadata_path)


def load_lora_checkpoint(model: nn.Module, path: Path) -> None:
    expected = lora_state_dict(model)
    loaded = load_file(str(path), device="cpu")
    if set(loaded) != set(expected):
        missing = sorted(set(expected) - set(loaded))
        extra = sorted(set(loaded) - set(expected))
        raise ValueError(f"LoRA checkpoint key mismatch missing={missing} extra={extra}")
    modules = lora_modules(model)
    for name, module in modules.items():
        for part in ("lora_A", "lora_B"):
            key = f"{name}.{part}.weight"
            target = getattr(module, part).weight
            if loaded[key].shape != target.shape:
                raise ValueError(f"LoRA checkpoint shape mismatch for {key}")
            target.data.copy_(loaded[key].to(device=target.device, dtype=target.dtype))


def restoration_state_dict(model: nn.Module, pixel_refiner: nn.Module) -> dict[str, Tensor]:
    """Return one deployable state containing FLUX LoRA and pixel refinement."""
    state = lora_state_dict(model)
    for key, value in pixel_refiner.state_dict().items():
        state[f"pixel_refiner.{key}"] = value.detach().cpu().contiguous()
    return state


def save_restoration_checkpoint(
    model: nn.Module,
    pixel_refiner: nn.Module,
    path: Path,
    metadata: dict[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {str(key): str(value) for key, value in metadata.items()}
    payload["state_schema"] = "flux_lora_plus_bounded_pixel_refiner_v1"
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        save_file(restoration_state_dict(model, pixel_refiner), str(temporary), metadata=payload)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)

    metadata_path = path.with_suffix(path.suffix + ".json")
    encoded = json.dumps({**metadata, "state_schema": payload["state_schema"]}, indent=2, sort_keys=True) + "\n"
    temporary_metadata = metadata_path.with_name(f".{metadata_path.name}.tmp")
    temporary_metadata.write_text(encoded)
    os.replace(temporary_metadata, metadata_path)


def load_restoration_checkpoint(model: nn.Module, pixel_refiner: nn.Module, path: Path) -> None:
    expected = restoration_state_dict(model, pixel_refiner)
    loaded = load_file(str(path), device="cpu")
    if set(loaded) != set(expected):
        missing = sorted(set(expected) - set(loaded))
        extra = sorted(set(loaded) - set(expected))
        raise ValueError(f"restoration checkpoint key mismatch missing={missing} extra={extra}")

    modules = lora_modules(model)
    for name, module in modules.items():
        for part in ("lora_A", "lora_B"):
            key = f"{name}.{part}.weight"
            target = getattr(module, part).weight
            if loaded[key].shape != target.shape:
                raise ValueError(f"restoration checkpoint shape mismatch for {key}")
            target.data.copy_(loaded[key].to(device=target.device, dtype=target.dtype))

    refiner_state = {}
    for key, target in pixel_refiner.state_dict().items():
        source = loaded[f"pixel_refiner.{key}"]
        if source.shape != target.shape:
            raise ValueError(f"restoration checkpoint shape mismatch for pixel_refiner.{key}")
        refiner_state[key] = source.to(device=target.device, dtype=target.dtype)
    pixel_refiner.load_state_dict(refiner_state, strict=True)
