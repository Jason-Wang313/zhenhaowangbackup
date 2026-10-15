#!/usr/bin/env python3
"""CPU-only contract and checkpoint preflight for real-data Generalist continuation."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/foundir_vlm_energy_real_generalist_v1.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    cfg = json.loads(CONFIG.read_text())
    if cfg.get("schema_version") != 1:
        raise RuntimeError("bad config schema")
    for key, digest_key in (
        ("base_config", "base_config_sha256"),
    ):
        path = ROOT / cfg[key]
        if sha256(path) != cfg[digest_key]:
            raise RuntimeError(f"identity mismatch: {key}")
    data = cfg["data"]
    manifest = ROOT / data["manifest"]
    metadata = ROOT / data["metadata"]
    if sha256(manifest) != data["manifest_sha256"]:
        raise RuntimeError("manifest identity mismatch")
    if sha256(metadata) != data["metadata_sha256"]:
        raise RuntimeError("metadata identity mismatch")
    meta = json.loads(metadata.read_text())
    expected = {
        "train": int(data["expected_train_records"]),
        "validation": int(data["expected_validation_records"]),
        "internal_test": int(data["expected_internal_test_records"]),
    }
    if meta.get("split_counts") != expected:
        raise RuntimeError("frozen split counts changed")

    counts: Counter[str] = Counter()
    category_counts: Counter[tuple[str, str]] = Counter()
    checked_paths = 0
    seen_split_category: set[tuple[str, str]] = set()
    with manifest.open(newline="") as handle:
        for index, row in enumerate(csv.DictReader(handle)):
            split = row["split"]
            counts[split] += 1
            category_counts[(split, row["category"])] += 1
            key = (split, row["category"])
            check_paths = (
                split in {"train", "validation"}
                and (key not in seen_split_category or index % 10000 == 0)
            )
            if check_paths:
                if not (ROOT / row["lq_path"]).is_file() or not (ROOT / row["gt_path"]).is_file():
                    raise RuntimeError(f"missing pair: {row['record_id']}")
                checked_paths += 2
                seen_split_category.add(key)
    if dict(counts) != expected:
        raise RuntimeError(f"observed split counts changed: {dict(counts)}")

    checkpoint_text = os.environ.get("VLM_ENERGY_REAL_INIT_CHECKPOINT", "")
    if not checkpoint_text:
        raise RuntimeError("set VLM_ENERGY_REAL_INIT_CHECKPOINT")
    checkpoint = Path(checkpoint_text)
    if not checkpoint.is_absolute():
        checkpoint = ROOT / checkpoint
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("model_state_dict")
    if not isinstance(state, dict) or not state:
        raise RuntimeError("initial checkpoint has no model_state_dict")
    if payload.get("phase") != "joint":
        raise RuntimeError("initial checkpoint is not a joint-phase Generalist")
    forbidden = ("official", "sealed", "internal_test", "parent_test")
    if any(token in checkpoint.as_posix().lower() for token in forbidden):
        raise RuntimeError("protected checkpoint path is forbidden")

    report = {
        "status": "foundir_vlm_energy_real_generalist_preflight_pass",
        "config": CONFIG.relative_to(ROOT).as_posix(),
        "config_sha256": sha256(CONFIG),
        "manifest_sha256": sha256(manifest),
        "split_counts": dict(counts),
        "train_category_counts": {
            category: category_counts[("train", category)]
            for category in sorted(meta["category_counts"])
        },
        "initial_checkpoint": checkpoint.relative_to(ROOT).as_posix(),
        "initial_checkpoint_sha256": sha256(checkpoint),
        "initial_epoch": int(payload.get("epoch", -1)),
        "initial_global_step": int(payload.get("global_step", -1)),
        "model_tensors": len(state),
        "path_existence_checks": checked_paths,
        "path_check_policy": "first_per_split_category_plus_every_10000th_manifest_record; full manifest SHA-256 verified",
        "optimizer_state_intentionally_reset_for_domain_continuation": True,
        "internal_test_accessed": False,
        "parent_test_accessed": False,
        "official_foundir_test_accessed": False,
        "sealed_internal_accessed": False,
    }
    output = ROOT / "reports/foundir_vlm_energy_real_generalist_v1_preflight.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
