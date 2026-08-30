"""Disk-backed bounded-LRU cache for frozen FLUX Qwen3 text contexts."""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from pathlib import Path

import torch
from safetensors.torch import load_file


def prompt_sha256(prompt: str) -> str:
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("prompt must be non-empty text")
    return hashlib.sha256(prompt.encode()).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class PromptContextCache:
    """Load verified prompt tensors on demand without retaining the full cache."""

    def __init__(self, index_path: Path, *, maximum_resident: int = 16) -> None:
        if maximum_resident <= 0:
            raise ValueError("maximum_resident must be positive")
        payload = json.loads(index_path.read_text())
        if payload.get("status") != "frozen_flux_qwen3_prompt_context_cache":
            raise RuntimeError("prompt-cache index is not frozen")
        entries = payload.get("entries")
        if not isinstance(entries, dict) or len(entries) != int(payload.get("prompts", -1)):
            raise RuntimeError("prompt-cache index cardinality changed")
        self.index_path = index_path
        self.root = index_path.parent
        self.entries = entries
        self.maximum_resident = maximum_resident
        self.resident: OrderedDict[str, tuple[torch.Tensor, torch.Tensor]] = OrderedDict()
        self.verified: set[str] = set()

    def _load(self, key: str) -> tuple[torch.Tensor, torch.Tensor]:
        entry = self.entries.get(key)
        if not isinstance(entry, dict):
            raise KeyError(f"prompt is absent from frozen cache: {key}")
        path = self.root / str(entry["path"])
        if key not in self.verified:
            if file_sha256(path) != entry["sha256"]:
                raise RuntimeError(f"prompt context changed: {path}")
            self.verified.add(key)
        tensors = load_file(str(path), device="cpu")
        if set(tensors) != {"context", "context_ids"}:
            raise RuntimeError(f"prompt context tensor schema changed: {path}")
        context = tensors["context"].contiguous()
        context_ids = tensors["context_ids"].contiguous()
        if context.ndim != 2 or context_ids.ndim != 2 or context.shape[0] != context_ids.shape[0]:
            raise RuntimeError(f"prompt context shapes changed: {path}")
        return context, context_ids

    def get(self, prompt: str, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        key = prompt_sha256(prompt)
        tensors = self.resident.pop(key, None)
        if tensors is None:
            tensors = self._load(key)
        self.resident[key] = tensors
        while len(self.resident) > self.maximum_resident:
            self.resident.popitem(last=False)
        context, context_ids = tensors
        return (
            context.unsqueeze(0).to(device, non_blocking=True),
            context_ids.unsqueeze(0).to(device, non_blocking=True),
        )
