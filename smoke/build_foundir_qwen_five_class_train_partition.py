#!/usr/bin/env python3
"""Freeze a source-balanced, training-only singleton Qwen partition."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lowlight_restoration.qwen_five_class_gate import (
    CLASSES,
    SPLITS,
    build_balanced_partition,
)

SOURCE = ROOT / "manifests/foundir_five_type_pairs_v1.csv"
OUTPUT = ROOT / "manifests/foundir_qwen_five_class_train_only_v1.csv"
SUMMARY = OUTPUT.with_suffix(".json")
SEED = 20260918
QUOTAS = {"fit": 450, "calibration": 90, "holdout": 90}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    # Non-training rows are discarded immediately and never serialized.
    with SOURCE.open(newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("split") == "train"]
    selected = build_balanced_partition(rows, seed=SEED, records_per_class=QUOTAS)
    fields = list(selected[0])
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT.exists() or SUMMARY.exists():
        raise FileExistsError("refusing to replace frozen Qwen partition artifacts")
    with OUTPUT.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(selected)
    counts = Counter((row["head_split"], row["label"]) for row in selected)
    payload = {
        "schema_version": 1,
        "status": "frozen_before_feature_extraction",
        "purpose": "training_only_singleton_five_class_qwen_gate",
        "source_manifest": SOURCE.relative_to(ROOT).as_posix(),
        "source_manifest_sha256": sha256(SOURCE),
        "manifest": OUTPUT.relative_to(ROOT).as_posix(),
        "manifest_sha256": sha256(OUTPUT),
        "seed": SEED,
        "classes": list(CLASSES),
        "split_assignment": "global source_group hash buckets 70/15/15",
        "selection": "one deterministic image per class/source_group",
        "records_per_class": QUOTAS,
        "records": len(selected),
        "counts": {
            split: {label: counts[(split, label)] for label in CLASSES} for split in SPLITS
        },
        "source_disjoint_across_head_splits": True,
        "development_or_internal_test_used": False,
    }
    SUMMARY.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
