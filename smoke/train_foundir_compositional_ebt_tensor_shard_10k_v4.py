#!/usr/bin/env python3
"""Matched 10k Plain/Compositional EBT training over contiguous tensor shards."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import smoke.train_foundir_compositional_ebt_scaled_learning_smoke_v1 as base
import smoke.train_foundir_compositional_ebt_tensor_shard_smoke_v3 as v3


CONFIG_PATH = ROOT / "configs/foundir_compositional_ebt_tensor_shard_10k_v4.json"
base.CONFIG_PATH = CONFIG_PATH
v3.CONFIG_PATH = CONFIG_PATH


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
        raise RuntimeError("v4 input identity changed")
    shard_metadata = json.loads((ROOT / data["shard_metadata"]).read_text())
    if shard_metadata["output_manifest_sha256"] != data["shard_manifest_sha256"]:
        raise RuntimeError("v4 shard metadata changed")
    shards = base.load_jsonl(paths[-1][0])
    if sum(int(row["samples"]) for row in shards) != int(data["train_bank_samples"]):
        raise RuntimeError("v4 train bank count changed")

    records = base.load_jsonl(paths[0][0])
    train_records = sorted(
        [row for row in records if row["split"] == "train"],
        key=lambda row: str(row["record_id"]),
    )
    offsets = np.cumsum([0] + [int(row["samples"]) for row in shards])
    by_size: dict[int, list[int]] = {1: [], 2: []}
    full_shard_samples = max(int(row["samples"]) for row in shards)
    for shard_index, shard in enumerate(shards):
        if int(shard["samples"]) != full_shard_samples:
            continue
        first = int(shard["first_record_index"])
        last = int(shard["last_record_index_exclusive"])
        sizes = {int(row["composition_size"]) for row in train_records[first:last]}
        if len(sizes) != 1:
            continue
        by_size[sizes.pop()].append(shard_index)
    if not all(by_size.values()):
        raise RuntimeError("missing homogeneous singleton/double shards")

    needed = int(data["train_presentations"])
    if needed % 2:
        raise RuntimeError("v4 balanced schedule requires an even presentation count")
    generator = np.random.default_rng(int(config["model"]["initialization_seed"]) + 4000)
    per_class_needed = needed // 2
    samples_per_shard = {int(shards[index]["samples"]) for values in by_size.values() for index in values}
    if len(samples_per_shard) != 1 or per_class_needed % next(iter(samples_per_shard)):
        raise RuntimeError("v4 exact balance is incompatible with shard size")
    shards_per_class = per_class_needed // next(iter(samples_per_shard))
    class_orders: dict[int, list[int]] = {1: [], 2: []}
    for size in (1, 2):
        while len(class_orders[size]) < shards_per_class:
            epoch_order = list(by_size[size])
            generator.shuffle(epoch_order)
            class_orders[size].extend(epoch_order)
        class_orders[size] = class_orders[size][:shards_per_class]
    shard_order = [
        index
        for pair in zip(class_orders[1], class_orders[2], strict=True)
        for index in pair
    ]
    indices = [
        sample
        for shard_index in shard_order
        for sample in range(int(offsets[shard_index]), int(offsets[shard_index + 1]))
    ]
    if len(indices) != needed:
        raise RuntimeError("v4 schedule length mismatch")

    targets = {str(row["record_id"]): row for row in base.load_jsonl(paths[2][0])}
    validation = [row for row in records if row["split"] == "validation"]
    seed = int(config["model"]["initialization_seed"])
    validation.sort(key=lambda row: base.stable_key(seed + 1, str(row["record_id"])))
    validation = validation[: int(data["validation_subset_records"])]
    for row in validation:
        row["cached_target_path"] = targets[str(row["record_id"])]["target_path"]
    config["_loaded_shards"] = shards
    return indices, validation


base.load_records = load_records
base.loader = v3.loader


if __name__ == "__main__":
    base.main()
