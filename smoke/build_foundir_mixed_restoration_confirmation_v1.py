#!/usr/bin/env python3
"""Freeze 640 fresh training-only mixed records for five-category restoration confirmation."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "manifests/foundir_five_type_pairs_v1.csv"
PILOT = ROOT / "manifests/foundir_mixed_restoration_pilot_v1.csv"
OUTPUT = ROOT / "manifests/foundir_mixed_restoration_confirmation_v1.csv"
METADATA = ROOT / "manifests/foundir_mixed_restoration_confirmation_v1.json"
SEED = 2026091901
PER_CATEGORY = 128
ORDERS = {
    "09Lowlight_Haze": ["haze", "low_light"],
    "13Rain_Haze": ["rain", "haze"],
    "15Lowlight_Blur": ["low_light", "blur"],
    "16Lowlight_Noise": ["noise", "low_light"],
    "18Lowlight_Blur_Noise": ["noise", "low_light", "blur"],
}
SINGLETONS = {"01Blur", "05Noise", "08Haze", "10Rain", "14Lowlight"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_key(*parts: object) -> str:
    return hashlib.sha256("\0".join(map(str, parts)).encode()).hexdigest()


def qwen_exclusions() -> tuple[set[str], set[str], list[str]]:
    paths = [path for path in sorted((ROOT / "manifests").glob("foundir_qwen_*class*.csv")) if "development" not in path.name]
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
    return groups, records, used


def main() -> None:
    with PILOT.open(newline="") as handle:
        pilot_rows = list(csv.DictReader(handle))
    pilot_records = {row["record_id"] for row in pilot_rows}
    pilot_groups = {row["source_group"] for row in pilot_rows}
    qwen_groups, qwen_records, qwen_paths = qwen_exclusions()
    candidates = {category: [] for category in ORDERS}
    singleton_groups: set[str] = set()
    with SOURCE.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        for row in reader:
            if row["split"] != "train":
                continue
            if row["category"] in SINGLETONS:
                singleton_groups.add(row["source_group"])
            elif row["category"] in candidates:
                candidates[row["category"]].append(dict(row))
    independently_held_out = {
        category: len({row["source_group"] for row in rows if row["source_group"] not in singleton_groups})
        for category, rows in candidates.items()
    }
    if any(independently_held_out.values()):
        raise RuntimeError("expected all mixed training sources to overlap singleton training sources")
    selected = []
    used_groups: set[str] = set()
    for category in ORDERS:
        ordered = sorted(
            candidates[category],
            key=lambda row: (
                row["source_group"] in qwen_groups,
                stable_key(SEED, category, row["record_id"]),
            ),
        )
        chosen = []
        for row in ordered:
            if (
                row["record_id"] in pilot_records
                or row["record_id"] in qwen_records
                or row["source_group"] in pilot_groups
                or row["source_group"] in used_groups
            ):
                continue
            chosen.append(row)
            used_groups.add(row["source_group"])
            if len(chosen) == PER_CATEGORY:
                break
        if len(chosen) != PER_CATEGORY:
            raise RuntimeError(f"insufficient fresh mixed sources for {category}: {len(chosen)}")
        for row in chosen:
            row["expert_order_json"] = json.dumps(ORDERS[category], separators=(",", ":"))
        selected.extend(chosen)
    if len(selected) != PER_CATEGORY * len(ORDERS) or len(used_groups) != len(selected):
        raise RuntimeError("mixed confirmation cardinality or global source isolation changed")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames + ["expert_order_json"])
        writer.writeheader()
        writer.writerows(selected)
    visual_ids = []
    for category in ORDERS:
        visual_ids.extend(
            [row["record_id"] for row in selected if row["category"] == category][:2]
        )
    metadata = {
        "schema_version": 1,
        "status": "frozen_training_only_mixed_restoration_confirmation_v1",
        "seed": SEED,
        "records": len(selected),
        "source_groups": len(used_groups),
        "records_per_category": dict(sorted(Counter(row["category"] for row in selected).items())),
        "expert_orders": ORDERS,
        "visual_record_ids": visual_ids,
        "source_manifest": SOURCE.relative_to(ROOT).as_posix(),
        "source_manifest_sha256": sha256(SOURCE),
        "pilot_manifest": PILOT.relative_to(ROOT).as_posix(),
        "pilot_manifest_sha256": sha256(PILOT),
        "pilot_record_overlap": len({row["record_id"] for row in selected} & pilot_records),
        "pilot_source_group_overlap": len(used_groups & pilot_groups),
        "qwen_manifests_excluded": qwen_paths,
        "qwen_record_overlap": len({row["record_id"] for row in selected} & qwen_records),
        "qwen_source_group_overlap": len(used_groups & qwen_groups),
        "singleton_training_source_group_overlap": len(used_groups & singleton_groups),
        "singleton_training_source_disjoint": False,
        "limitation": "No mixed record in the FoundIR training split is source-group-disjoint from singleton expert training; this is a fresh mixed-pair stress validation, not a model-held-out content validation.",
        "manifest": OUTPUT.relative_to(ROOT).as_posix(),
        "manifest_sha256": sha256(OUTPUT),
        "development_pixels_accessed": False,
        "official_foundir_test_accessed": False,
    }
    METADATA.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metadata, sort_keys=True))


if __name__ == "__main__":
    main()
