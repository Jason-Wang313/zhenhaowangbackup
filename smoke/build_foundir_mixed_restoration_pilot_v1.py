#!/usr/bin/env python3
"""Freeze a training-only, source-disjoint 96-record mixed restoration pilot."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "manifests/foundir_five_type_pairs_v1.csv"
OUTPUT = ROOT / "manifests/foundir_mixed_restoration_pilot_v1.csv"
METADATA = ROOT / "manifests/foundir_mixed_restoration_pilot_v1.json"
SEED = 2026091807
PER_CATEGORY = 16

CATEGORY_ORDERS = {
    "02Blur_Noise": ["noise", "blur"],
    "09Lowlight_Haze": ["haze", "low_light"],
    "13Rain_Haze": ["rain", "haze"],
    "15Lowlight_Blur": ["low_light", "blur"],
    "16Lowlight_Noise": ["noise", "low_light"],
    "18Lowlight_Blur_Noise": ["noise", "low_light", "blur"],
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_key(*parts: object) -> str:
    return hashlib.sha256("\0".join(map(str, parts)).encode()).hexdigest()


def classifier_exclusions() -> tuple[set[str], set[str], list[str]]:
    paths = [
        path
        for path in sorted((ROOT / "manifests").glob("foundir_qwen_*class*.csv"))
        if "development" not in path.name
    ]
    groups: set[str] = set()
    records: set[str] = set()
    used = []
    for path in paths:
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            if "source_group" not in (reader.fieldnames or []):
                continue
            for row in reader:
                groups.add(row["source_group"])
                if row.get("record_id"):
                    records.add(row["record_id"])
            used.append(path.relative_to(ROOT).as_posix())
    if not used:
        raise RuntimeError("no Qwen classification manifests found for exclusion")
    return groups, records, used


def main() -> None:
    excluded, excluded_records, exclusion_paths = classifier_exclusions()
    candidates = {category: [] for category in CATEGORY_ORDERS}
    with SOURCE.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        for row in reader:
            category = row["category"]
            if (
                row["split"] == "train"
                and category in candidates
                and row["record_id"] not in excluded_records
            ):
                candidates[category].append(dict(row))

    selected = []
    used_groups: set[str] = set()
    for category in CATEGORY_ORDERS:
        ordered = sorted(
            candidates[category],
            key=lambda row: (
                row["source_group"] in excluded,
                stable_key(SEED, category, row["record_id"]),
            ),
        )
        chosen = []
        for row in ordered:
            if row["source_group"] in used_groups:
                continue
            chosen.append(row)
            used_groups.add(row["source_group"])
            if len(chosen) == PER_CATEGORY:
                break
        if len(chosen) != PER_CATEGORY:
            raise RuntimeError(f"insufficient disjoint training sources for {category}")
        for row in chosen:
            row["expert_order_json"] = json.dumps(CATEGORY_ORDERS[category], separators=(",", ":"))
        selected.extend(chosen)

    if len(selected) != PER_CATEGORY * len(CATEGORY_ORDERS):
        raise RuntimeError("mixed pilot cardinality changed")
    if len(used_groups) != len(selected):
        raise RuntimeError("mixed pilot source groups are not globally disjoint")
    classifier_group_overlaps = used_groups & excluded

    output_fields = fieldnames + ["expert_order_json"]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fields)
        writer.writeheader()
        writer.writerows(selected)

    counts = Counter(row["category"] for row in selected)
    metadata = {
        "schema_version": 1,
        "status": "frozen_training_only_mixed_restoration_pilot_v1",
        "seed": SEED,
        "records": len(selected),
        "source_groups": len(used_groups),
        "records_per_category": dict(sorted(counts.items())),
        "expert_orders": CATEGORY_ORDERS,
        "source_manifest": SOURCE.relative_to(ROOT).as_posix(),
        "source_manifest_sha256": sha256(SOURCE),
        "excluded_qwen_classification_manifests": exclusion_paths,
        "excluded_source_groups": len(excluded),
        "classification_record_overlap": 0,
        "classification_source_group_overlap": len(classifier_group_overlaps),
        "classification_overlap_policy": "zero record overlap; minimize source-group overlap after enforcing 16 records per category and global pilot source disjointness",
        "manifest": OUTPUT.relative_to(ROOT).as_posix(),
        "manifest_sha256": sha256(OUTPUT),
        "development_pixels_accessed": False,
        "official_foundir_test_accessed": False,
    }
    METADATA.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metadata, sort_keys=True))


if __name__ == "__main__":
    main()
