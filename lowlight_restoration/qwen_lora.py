"""Auditable Qwen visual-LoRA scope and immutable adapter checkpoints."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Mapping

import torch
from safetensors.torch import load_file, save_file
from torch import nn

from .lora import inject_lora


VISUAL_LORA_PATTERNS = (
    "model.visual.blocks.*.attn.qkv",
    "model.visual.blocks.*.attn.proj",
    "model.visual.blocks.*.mlp.gate_proj",
    "model.visual.blocks.*.mlp.up_proj",
    "model.visual.blocks.*.mlp.down_proj",
)


def configure_qwen_visual_lora(
    model: nn.Module,
    *,
    rank: int,
    alpha: float,
    dropout: float,
    merger_trainable: bool,
) -> dict[str, object]:
    """Freeze Qwen, inject visual-block LoRA, and optionally unfreeze merger."""

    matched = inject_lora(
        model,
        list(VISUAL_LORA_PATTERNS),
        rank=rank,
        alpha=alpha,
        dropout=dropout,
    )
    visual = getattr(getattr(model, "model", None), "visual", None)
    if visual is None or not hasattr(visual, "merger"):
        raise ValueError("Qwen visual merger was not found")
    if merger_trainable:
        visual.merger.requires_grad_(True)
    trainable = {
        name: parameter.numel()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    if not trainable:
        raise RuntimeError("Qwen LoRA configuration has no trainable parameters")
    bad = [
        name
        for name in trainable
        if ".lora_A.weight" not in name
        and ".lora_B.weight" not in name
        and not (merger_trainable and name.startswith("model.visual.merger."))
    ]
    if bad:
        raise RuntimeError(f"trainable scope escaped LoRA/merger: {bad[:5]}")
    merger_names = [
        name for name in trainable if name.startswith("model.visual.merger.")
    ]
    if bool(merger_names) != merger_trainable:
        raise RuntimeError("visual merger trainability differs from the ablation")
    return {
        "matched_lora_modules": matched,
        "matched_lora_module_count": len(matched),
        "trainable_tensor_count": len(trainable),
        "trainable_parameters": sum(trainable.values()),
        "merger_trainable": merger_trainable,
        "merger_trainable_parameters": sum(trainable[name] for name in merger_names),
        "trainable_names": sorted(trainable),
    }


def trainable_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    state = {
        name: parameter.detach().cpu().contiguous()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    if not state:
        raise ValueError("model has no trainable adapter state")
    return state


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_qwen_adapter(
    model: nn.Module, path: Path, metadata: Mapping[str, object]
) -> dict[str, object]:
    """Publish a new adapter without replacing any existing artifact."""

    metadata_path = path.with_suffix(path.suffix + ".json")
    if path.exists() or metadata_path.exists():
        raise FileExistsError(f"refusing to overwrite Qwen adapter: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        save_file(
            trainable_state_dict(model),
            str(temporary),
            metadata={str(key): str(value) for key, value in metadata.items()},
        )
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    summary = {**dict(metadata), "adapter_sha256": sha256(path)}
    encoded = (json.dumps(summary, indent=2, sort_keys=True) + "\n").encode()
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{metadata_path.name}.", delete=False
    ) as handle:
        temporary_metadata = Path(handle.name)
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary_metadata, metadata_path)
    finally:
        temporary_metadata.unlink(missing_ok=True)
    return summary


def load_qwen_adapter(model: nn.Module, path: Path) -> None:
    expected = trainable_state_dict(model)
    loaded = load_file(str(path), device="cpu")
    if set(loaded) != set(expected):
        raise ValueError(
            f"Qwen adapter keys differ missing={sorted(set(expected)-set(loaded))[:5]} "
            f"extra={sorted(set(loaded)-set(expected))[:5]}"
        )
    named = dict(model.named_parameters())
    for name, value in loaded.items():
        target = named[name]
        if value.shape != target.shape:
            raise ValueError(f"Qwen adapter shape mismatch: {name}")
        target.data.copy_(value.to(device=target.device, dtype=target.dtype))
