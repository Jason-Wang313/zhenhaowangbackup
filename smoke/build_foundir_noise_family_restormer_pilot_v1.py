#!/usr/bin/env python3
"""Freeze a paired, source-disjoint Noise-family Restormer pilot manifest."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "manifests/foundir_five_type_pairs_v1.csv"
PURE_VALIDATION = ROOT / "manifests/foundir_noise_family_cross_compatibility_v1.csv"
MIXED_VALIDATION = ROOT / "manifests/foundir_mixed_restoration_confirmation_v1.csv"
OUTPUT = ROOT / "manifests/foundir_noise_family_restormer_pilot_v1.csv"
METADATA = OUTPUT.with_suffix(".json")
CATEGORIES = ("05Noise", "16Lowlight_Noise")
SEED = 2026091911
TRAIN_GROUPS = 768
TRAIN_RECORDS_PER_CATEGORY = 1024
VALIDATION_RECORDS_PER_CATEGORY = 128


def digest(*parts: object) -> str:
    return hashlib.sha256("\0".join(map(str, parts)).encode()).hexdigest()


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def read_ids(path: Path, category: str) -> list[str]:
    with path.open(newline="") as handle:
        return [row["record_id"] for row in csv.DictReader(handle) if row["category"] == category]


def main() -> None:
    validation_ids = {
        "05Noise": read_ids(PURE_VALIDATION, "05Noise"),
        "16Lowlight_Noise": read_ids(MIXED_VALIDATION, "16Lowlight_Noise"),
    }
    if any(len(ids) != VALIDATION_RECORDS_PER_CATEGORY for ids in validation_ids.values()):
        raise RuntimeError("Noise-family frozen validation cardinality changed")

    pools: dict[str, dict[str, list[dict[str, str]]]] = {
        category: defaultdict(list) for category in CATEGORIES
    }
    by_id: dict[str, dict[str, str]] = {}
    with SOURCE.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        for row in reader:
            if row["split"] == "train" and row["category"] in CATEGORIES:
                item = dict(row)
                pools[row["category"]][row["source_group"]].append(item)
                by_id[row["record_id"]] = item

    validation = []
    validation_groups_by_category: dict[str, set[str]] = {}
    for category in CATEGORIES:
        rows = [by_id[record_id].copy() for record_id in validation_ids[category]]
        groups = {row["source_group"] for row in rows}
        if len(groups) != VALIDATION_RECORDS_PER_CATEGORY:
            raise RuntimeError("Noise-family validation source-group coverage changed")
        validation_groups_by_category[category] = groups
        for row in rows:
            row["original_split"] = row["split"]
            row["split"] = "validation"
            validation.append(row)
    if validation_groups_by_category[CATEGORIES[0]] != validation_groups_by_category[CATEGORIES[1]]:
        raise RuntimeError("Noise-family validation categories are no longer source paired")
    validation_groups = validation_groups_by_category[CATEGORIES[0]]

    common_groups = set(pools[CATEGORIES[0]]) & set(pools[CATEGORIES[1]])
    available = sorted(common_groups - validation_groups, key=lambda group: digest(SEED, "group", group))
    train_groups = available[:TRAIN_GROUPS]
    if len(train_groups) != TRAIN_GROUPS:
        raise RuntimeError("insufficient Noise-family source groups")

    train = []
    for category in CATEGORIES:
        mandatory = [
            min(
                pools[category][group],
                key=lambda row: digest(SEED, category, "mandatory", row["record_id"]),
            )
            for group in train_groups
        ]
        mandatory_ids = {row["record_id"] for row in mandatory}
        candidates = [row for group in train_groups for row in pools[category][group]]
        candidates.sort(key=lambda row: digest(SEED, category, "extra", row["record_id"]))
        chosen = mandatory + [
            row for row in candidates if row["record_id"] not in mandatory_ids
        ][: TRAIN_RECORDS_PER_CATEGORY - len(mandatory)]
        if len(chosen) != TRAIN_RECORDS_PER_CATEGORY:
            raise RuntimeError(f"insufficient Noise-family records for {category}")
        for original in chosen:
            row = original.copy()
            row["original_split"] = row["split"]
            row["split"] = "train"
            train.append(row)

    selected = train + validation
    record_ids = [row["record_id"] for row in selected]
    if len(record_ids) != len(set(record_ids)) or set(train_groups) & validation_groups:
        raise RuntimeError("Noise-family pilot is not record/source disjoint")
    counts = {
        split: {
            category: sum(row["split"] == split and row["category"] == category for row in selected)
            for category in CATEGORIES
        }
        for split in ("train", "validation")
    }
    expected = {
        "train": {category: TRAIN_RECORDS_PER_CATEGORY for category in CATEGORIES},
        "validation": {category: VALIDATION_RECORDS_PER_CATEGORY for category in CATEGORIES},
    }
    if counts != expected:
        raise RuntimeError("Noise-family pilot category balance changed")

    if OUTPUT.exists() or METADATA.exists():
        raise FileExistsError("refusing to overwrite frozen Noise-family pilot manifest")
    with OUTPUT.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields + ["original_split"])
        writer.writeheader()
        writer.writerows(selected)
    payload = {
        "schema_version": 1,
        "status": "frozen_training_only_noise_family_restormer_pilot_v1",
        "seed": SEED,
        "categories": list(CATEGORIES),
        "records": len(selected),
        "split_category_records": counts,
        "train_source_groups": len(set(train_groups)),
        "validation_source_groups": len(validation_groups),
        "source_paired_across_categories": True,
        "source_disjoint_train_validation": True,
        "manifest": OUTPUT.relative_to(ROOT).as_posix(),
        "manifest_sha256": sha256(OUTPUT),
        "source_manifest_sha256": sha256(SOURCE),
        "pure_validation_manifest_sha256": sha256(PURE_VALIDATION),
        "mixed_validation_manifest_sha256": sha256(MIXED_VALIDATION),
        "all_records_originated_from_foundir_training_split": True,
        "development_pixels_accessed": False,
        "official_foundir_test_accessed": False,
    }
    METADATA.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
