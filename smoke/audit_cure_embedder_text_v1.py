#!/usr/bin/env python3
"""Verify that frozen text prompts are identical across the two CURE tracks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "checkpoints/cure_two_track_v1"
OUTPUT = BASE / "embedder_text_audit.json"
KEYS = ("embedder.weight", "mlp.0.weight", "mlp.0.bias")


def tensor_sha256(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def main() -> None:
    tensors = {}
    for track in ("published_style", "strict_holdout"):
        path = BASE / track / "embedder/embedder_200.tar"
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        state = checkpoint["state_dict"]
        tensors[track] = {key: state[key.removeprefix("module.")] for key in KEYS}
    identical = all(
        torch.equal(tensors["published_style"][key], tensors["strict_holdout"][key])
        for key in KEYS
    )
    result = {
        "schema_version": 1,
        "frozen_text_tensors_identical": identical,
        "published_style_sha256": {key: tensor_sha256(tensors["published_style"][key]) for key in KEYS},
        "strict_holdout_sha256": {key: tensor_sha256(tensors["strict_holdout"][key]) for key in KEYS},
        "official_foundir_test_accessed": False,
    }
    content = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if OUTPUT.exists():
        if OUTPUT.read_text() != content:
            raise RuntimeError("embedder text audit drift")
    else:
        OUTPUT.write_text(content)
    print(content, end="")
    if not identical:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
