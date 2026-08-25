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

from .lora import LoRALinear, inject_lora


VISUAL_LORA_PATTERNS = (
    "model.visual.blocks.*.attn.qkv",
    "model.visual.blocks.*.attn.proj",
    "model.visual.blocks.*.mlp.gate_proj",
    "model.visual.blocks.*.mlp.up_proj",
    "model.visual.blocks.*.mlp.down_proj",
)

LANGUAGE_ATTENTION_LORA_PATTERNS = (
    "model.language_model.layers.*.self_attn.q_proj",
    "model.language_model.layers.*.self_attn.k_proj",
    "model.language_model.layers.*.self_attn.v_proj",
    "model.language_model.layers.*.self_attn.o_proj",
)


def configure_qwen_visual_lora(
    model: nn.Module,
    *,
    rank: int,
    alpha: float,
    dropout: float,
    merger_trainable: bool,
    language_rank: int | None = None,
    language_alpha: float | None = None,
) -> dict[str, object]:
    """Freeze Qwen and inject auditable visual and optional decoder LoRA."""

    visual_matched = inject_lora(
        model,
        list(VISUAL_LORA_PATTERNS),
        rank=rank,
        alpha=alpha,
        dropout=dropout,
    )
    language_matched: list[str] = []
    if language_rank is not None:
        if language_alpha is None:
            raise ValueError("language LoRA alpha is required with language rank")
        language_matched = inject_lora(
            model,
            list(LANGUAGE_ATTENTION_LORA_PATTERNS),
            rank=language_rank,
            alpha=language_alpha,
            dropout=dropout,
        )
        # The second injection freezes the complete module tree before adding
        # decoder adapters. Restore trainability for both auditable LoRA sets.
        for module in model.modules():
            if isinstance(module, LoRALinear):
                module.lora_A.requires_grad_(True)
                module.lora_B.requires_grad_(True)
    elif language_alpha is not None:
        raise ValueError("language LoRA rank is required with language alpha")
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
    language_names = [
        name
        for name in trainable
        if name.startswith("model.language_model.") and ".lora_" in name
    ]
    if bool(merger_names) != merger_trainable:
        raise RuntimeError("visual merger trainability differs from the ablation")
    return {
        "matched_lora_modules": visual_matched + language_matched,
        "matched_lora_module_count": len(visual_matched) + len(language_matched),
        "matched_visual_lora_module_count": len(visual_matched),
        "matched_language_lora_module_count": len(language_matched),
        "trainable_tensor_count": len(trainable),
        "trainable_parameters": sum(trainable.values()),
        "merger_trainable": merger_trainable,
        "merger_trainable_parameters": sum(trainable[name] for name in merger_names),
        "language_lora_trainable_parameters": sum(
            trainable[name] for name in language_names
        ),
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


def load_qwen_adapter(
    model: nn.Module,
    path: Path,
    *,
    allow_missing_prefixes: tuple[str, ...] = (),
) -> None:
    expected = trainable_state_dict(model)
    loaded = load_file(str(path), device="cpu")
    missing = set(expected) - set(loaded)
    extra = set(loaded) - set(expected)
    unexpected_missing = {
        name
        for name in missing
        if not any(name.startswith(prefix) for prefix in allow_missing_prefixes)
    }
    if extra or unexpected_missing:
        raise ValueError(
            f"Qwen adapter keys differ missing={sorted(unexpected_missing)[:5]} "
            f"extra={sorted(extra)[:5]}"
        )
    named = dict(model.named_parameters())
    for name, value in loaded.items():
        target = named[name]
        if value.shape != target.shape:
            raise ValueError(f"Qwen adapter shape mismatch: {name}")
        target.data.copy_(value.to(device=target.device, dtype=target.dtype))
