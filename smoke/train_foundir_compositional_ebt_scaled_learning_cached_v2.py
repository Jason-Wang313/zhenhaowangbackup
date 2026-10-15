#!/usr/bin/env python3
"""v1-matched real-data smoke using immutable cached target PNGs only."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import smoke.train_foundir_compositional_ebt_scaled_learning_smoke_v1 as base  # noqa: E402
from lowlight_restoration.compositional_ebt import DEGRADATIONS  # noqa: E402


CONFIG_PATH = ROOT / "configs/foundir_compositional_ebt_scaled_learning_cached_v2.json"
base.CONFIG_PATH = CONFIG_PATH


class CachedLowerOrderDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(self, records: list[dict[str, Any]], *, crop: int, seed: int, augment: bool) -> None:
        self.records = records
        self.crop = crop
        self.seed = seed
        self.augment = augment

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        record = self.records[index]
        with Image.open(ROOT / str(record["output_path"])) as handle:
            degraded = np.asarray(handle.convert("RGB"), dtype=np.uint8).copy()
        with Image.open(ROOT / str(record["cached_target_path"])) as handle:
            clean = np.asarray(handle.convert("RGB"), dtype=np.uint8).copy()
        if degraded.shape != clean.shape or min(degraded.shape[:2]) < self.crop:
            raise RuntimeError(f"bad cached pair shape: {record['record_id']}")
        height, width = degraded.shape[:2]
        rng = random.Random(self.seed + index)
        if self.augment:
            y = rng.randrange(height - self.crop + 1)
            x = rng.randrange(width - self.crop + 1)
        else:
            y = (height - self.crop) // 2
            x = (width - self.crop) // 2
        degraded = degraded[y : y + self.crop, x : x + self.crop]
        clean = clean[y : y + self.crop, x : x + self.crop]
        if self.augment:
            if rng.randrange(2):
                degraded, clean = np.flip(degraded, 1), np.flip(clean, 1)
            if rng.randrange(2):
                degraded, clean = np.flip(degraded, 0), np.flip(clean, 0)
            if rng.randrange(2):
                degraded, clean = np.swapaxes(degraded, 0, 1), np.swapaxes(clean, 0, 1)
        degraded, clean = np.ascontiguousarray(degraded), np.ascontiguousarray(clean)
        d = torch.from_numpy(degraded).permute(2, 0, 1).float().div_(127.5).sub_(1)
        c = torch.from_numpy(clean).permute(2, 0, 1).float().div_(127.5).sub_(1)
        present = set(map(str, record["degradations"]))
        q = torch.tensor([float(name in present) for name in DEGRADATIONS])
        return d, c, q


def load_records(config: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    data = config["data"]
    paths = (
        (ROOT / data["manifest"], data["manifest_sha256"]),
        (ROOT / data["metadata"], data["metadata_sha256"]),
        (ROOT / data["target_manifest"], data["target_manifest_sha256"]),
        (ROOT / data["target_metadata"], data["target_metadata_sha256"]),
    )
    if any(base.sha256(path) != expected for path, expected in paths):
        raise RuntimeError("cached lower-order data identity changed")
    rows = base.load_jsonl(paths[0][0])
    targets = {str(row["record_id"]): row for row in base.load_jsonl(paths[2][0])}
    if len(targets) != len(rows) or {str(row["record_id"]) for row in rows} != set(targets):
        raise RuntimeError("cached target coverage changed")
    for row in rows:
        target = targets[str(row["record_id"])]
        if target["split"] != row["split"] or int(target["composition_size"]) != int(row["composition_size"]):
            raise RuntimeError(f"cached target identity mismatch: {row['record_id']}")
        row["cached_target_path"] = target["target_path"]
    allowed = set(map(int, data["allowed_composition_sizes"]))
    if any(int(row["composition_size"]) not in allowed for row in rows):
        raise RuntimeError("non-lower-order record entered cached smoke")
    train = [row for row in rows if row["split"] == "train"]
    validation = [row for row in rows if row["split"] == "validation"]
    if len(train) != int(data["train_records"]) or len(validation) != int(data["validation_records"]):
        raise RuntimeError("cached lower-order counts changed")
    seed = int(config["model"]["initialization_seed"])
    train.sort(key=lambda row: base.stable_key(seed, str(row["record_id"])))
    validation.sort(key=lambda row: base.stable_key(seed + 1, str(row["record_id"])))
    needed = int(config["training"]["optimizer_steps_per_arm"]) * int(config["training"]["batch_size_per_arm"])
    if needed > len(train):
        raise RuntimeError("cached smoke would repeat train records")
    return train[:needed], validation[: int(data["validation_subset_records"])]


def loader(records: list[dict[str, Any]], config: dict[str, Any], *, train: bool) -> DataLoader:
    settings = config["training"]
    workers = int(settings["num_workers_per_arm"])
    return DataLoader(
        CachedLowerOrderDataset(
            records,
            crop=int(config["data"]["crop_size"]),
            seed=int(config["model"]["initialization_seed"]) + (1000 if train else 2000),
            augment=train,
        ),
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
