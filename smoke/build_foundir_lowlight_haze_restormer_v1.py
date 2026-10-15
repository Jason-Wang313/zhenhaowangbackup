#!/usr/bin/env python3
"""Freeze source-disjoint data for the direct Low-light+Haze Restormer expert."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "manifests/foundir_five_type_pairs_v1.csv"
COMPARISON = ROOT / "manifests/foundir_composition_whole_image_v1.csv"
OUTPUT = ROOT / "manifests/foundir_lowlight_haze_restormer_v1.csv"
METADATA = OUTPUT.with_suffix(".json")
CATEGORY = "09Lowlight_Haze"
SEED = 2026092601
VALIDATION_GROUPS = 128


def digest(*parts: object) -> str:
    return hashlib.sha256("\0".join(map(str, parts)).encode()).hexdigest()


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    with COMPARISON.open(newline="") as handle:
        comparison_rows = list(csv.DictReader(handle))
    excluded_groups = {row["source_group"] for row in comparison_rows}
    category_comparison = [row for row in comparison_rows if row["category"] == CATEGORY]
    if len(comparison_rows) != 660 or len(excluded_groups) != 221:
        raise RuntimeError("frozen 660-image comparison identity changed")
    if len(category_comparison) != 60:
        raise RuntimeError("Low-light+Haze comparison cardinality changed")

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    with SOURCE.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        for row in reader:
            if row["category"] == CATEGORY and row["split"] == "train":
                grouped[row["source_group"]].append(dict(row))

    eligible_groups = sorted(
        set(grouped) - excluded_groups,
        key=lambda group: digest(SEED, "validation-group", group),
    )
    validation_groups = set(eligible_groups[:VALIDATION_GROUPS])
    if len(validation_groups) != VALIDATION_GROUPS:
        raise RuntimeError("insufficient source-disjoint Low-light+Haze validation groups")

    validation: list[dict[str, str]] = []
    for group in sorted(validation_groups):
        row = min(grouped[group], key=lambda item: digest(SEED, "validation-record", item["record_id"])).copy()
        row["original_split"] = row["split"]
        row["split"] = "validation"
        validation.append(row)

    train: list[dict[str, str]] = []
    for group, rows in grouped.items():
        if group in excluded_groups or group in validation_groups:
            continue
        for original in rows:
            row = original.copy()
            row["original_split"] = row["split"]
            row["split"] = "train"
            train.append(row)

    selected = sorted(train, key=lambda row: row["record_id"]) + sorted(
        validation, key=lambda row: row["record_id"]
    )
    record_ids = [row["record_id"] for row in selected]
    train_groups = {row["source_group"] for row in train}
    if len(record_ids) != len(set(record_ids)):
        raise RuntimeError("duplicate records in frozen Low-light+Haze manifest")
    if train_groups & validation_groups or (train_groups | validation_groups) & excluded_groups:
        raise RuntimeError("Low-light+Haze source-group isolation failed")
    if not train or len(validation) != VALIDATION_GROUPS:
        raise RuntimeError("Low-light+Haze frozen split is empty or incomplete")

    if OUTPUT.exists() or METADATA.exists():
        raise FileExistsError("refusing to overwrite frozen Low-light+Haze manifest")
    with OUTPUT.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields + ["original_split"])
        writer.writeheader()
        writer.writerows(selected)

    payload = {
        "schema_version": 1,
        "status": "frozen_training_only_lowlight_haze_restormer_v1",
        "seed": SEED,
        "category": CATEGORY,
        "records": len(selected),
        "split_records": {"train": len(train), "validation": len(validation)},
        "split_source_groups": {
            "train": len(train_groups),
            "validation": len(validation_groups),
        },
        "excluded_comparison_records": len(comparison_rows),
        "excluded_comparison_source_groups": len(excluded_groups),
        "excluded_category_comparison_records": len(category_comparison),
        "source_disjoint_train_validation_comparison": True,
        "manifest": OUTPUT.relative_to(ROOT).as_posix(),
        "manifest_sha256": sha256(OUTPUT),
        "source_manifest_sha256": sha256(SOURCE),
        "comparison_manifest_sha256": sha256(COMPARISON),
        "all_training_records_originated_from_foundir_training_split": True,
        "development_pixels_accessed": False,
        "official_foundir_test_accessed": False,
    }
    METADATA.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
