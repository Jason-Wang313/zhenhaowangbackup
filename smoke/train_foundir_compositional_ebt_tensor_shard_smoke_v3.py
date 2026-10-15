#!/usr/bin/env python3
"""Matched 26M EBT smoke reading pre-cropped contiguous tensor shards."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import smoke.train_foundir_compositional_ebt_scaled_learning_smoke_v1 as base  # noqa: E402
import smoke.train_foundir_compositional_ebt_scaled_learning_cached_v2 as cached  # noqa: E402


CONFIG_PATH = ROOT / "configs/foundir_compositional_ebt_tensor_shard_smoke_v3.json"
base.CONFIG_PATH = CONFIG_PATH


class TensorShardDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(self, indices: list[int], shards: list[dict[str, Any]]) -> None:
        self.indices = indices
        self.shards = shards
        self.offsets = np.cumsum([0] + [int(row["samples"]) for row in shards])
        self.cache: dict[int, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        global_index = self.indices[index]
        shard_index = int(np.searchsorted(self.offsets, global_index, side="right") - 1)
        local_index = global_index - int(self.offsets[shard_index])
        if shard_index not in self.cache:
            self.cache[shard_index] = np.load(
                ROOT / str(self.shards[shard_index]["path"]), mmap_mode="r", allow_pickle=False
            )
        sample = self.cache[shard_index][local_index]
        degraded = torch.from_numpy(np.array(sample["degraded"], copy=True)).float().div_(127.5).sub_(1)
        target = torch.from_numpy(np.array(sample["target"], copy=True)).float().div_(127.5).sub_(1)
        condition = torch.from_numpy(np.array(sample["condition"], copy=True)).float()
        return degraded, target, condition


def load_records(config: dict[str, Any]) -> tuple[list[int], list[dict[str, Any]]]:
    data = config["data"]
    paths = (
        (ROOT / data["manifest"], data["manifest_sha256"]),
        (ROOT / data["metadata"], data["metadata_sha256"]),
        (ROOT / data["target_manifest"], data["target_manifest_sha256"]),
        (ROOT / data["target_metadata"], data["target_metadata_sha256"]),
        (ROOT / data["shard_manifest"], data["shard_manifest_sha256"]),
    )
    if any(base.sha256(path) != expected for path, expected in paths):
        raise RuntimeError("v3 input identity changed")
    shard_metadata = json.loads((ROOT / data["shard_metadata"]).read_text())
    if shard_metadata["output_manifest_sha256"] != data["shard_manifest_sha256"]:
        raise RuntimeError("v3 shard metadata changed")
    shards = base.load_jsonl(paths[-1][0])
    if sum(int(row["samples"]) for row in shards) != int(data["train_bank_samples"]):
        raise RuntimeError("v3 train bank count changed")
    records = base.load_jsonl(paths[0][0])
    train_records = sorted(
        [row for row in records if row["split"] == "train"],
        key=lambda row: str(row["record_id"]),
    )
    generator = np.random.default_rng(int(config["model"]["initialization_seed"]) + 3000)
    offsets = np.cumsum([0] + [int(row["samples"]) for row in shards])
    by_size: dict[int, list[int]] = {1: [], 2: []}
    for shard_index, shard in enumerate(shards):
        first = int(shard["first_record_index"])
        last = int(shard["last_record_index_exclusive"])
        composition_counts = {
            size: sum(int(row["composition_size"]) == size for row in train_records[first:last])
            for size in (1, 2)
        }
        majority = max((1, 2), key=lambda size: (composition_counts[size], -size))
        if int(shard["samples"]) == int(shards[0]["samples"]):
            by_size[majority].append(shard_index)
    for values in by_size.values():
        generator.shuffle(values)
    full_shards_needed = math.ceil(int(data["train_presentations"]) / int(shards[0]["samples"]))
    if full_shards_needed % 2 or any(len(values) < full_shards_needed // 2 for values in by_size.values()):
        raise RuntimeError("cannot construct balanced sequential shard schedule")
    shard_order = [
        shard_index
        for pair in zip(by_size[1][: full_shards_needed // 2], by_size[2][: full_shards_needed // 2], strict=True)
        for shard_index in pair
    ]
    indices = []
    for shard_index in shard_order:
        indices.extend(range(int(offsets[shard_index]), int(offsets[shard_index + 1])))
        if len(indices) >= int(data["train_presentations"]):
            break
    indices = indices[: int(data["train_presentations"])]
    if len(set(indices)) != len(indices):
        raise RuntimeError("v3 train schedule repeats samples")

    targets = {str(row["record_id"]): row for row in base.load_jsonl(paths[2][0])}
    validation = [row for row in records if row["split"] == "validation"]
    seed = int(config["model"]["initialization_seed"])
    validation.sort(key=lambda row: base.stable_key(seed + 1, str(row["record_id"])))
    validation = validation[: int(data["validation_subset_records"])]
    for row in validation:
        row["cached_target_path"] = targets[str(row["record_id"])]["target_path"]
    config["_loaded_shards"] = shards
    return indices, validation


def loader(records: list[Any], config: dict[str, Any], *, train: bool) -> DataLoader:
    settings = config["training"]
    workers = int(settings["num_workers_per_arm"])
    dataset: Dataset
    if train:
        dataset = TensorShardDataset(records, config["_loaded_shards"])
    else:
        dataset = cached.CachedLowerOrderDataset(
            records, crop=int(config["data"]["crop_size"]),
            seed=int(config["model"]["initialization_seed"]) + 2000, augment=False,
        )
    return DataLoader(
        dataset,
        batch_size=int(settings["batch_size_per_arm"]) if train else 32,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        drop_last=train,
        persistent_workers=workers > 0,
        prefetch_factor=int(settings["prefetch_factor"]) if workers > 0 else None,
    )


base.load_records = load_records
base.loader = loader


if __name__ == "__main__":
    base.main()
