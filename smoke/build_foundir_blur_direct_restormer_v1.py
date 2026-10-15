#!/usr/bin/env python3
"""Freeze source-disjoint, training-only splits for direct Blur restoration."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "manifests/foundir_five_type_pairs_v1.csv"
OUTPUT = ROOT / "manifests/foundir_blur_direct_restormer_v1.csv"
METADATA = ROOT / "manifests/foundir_blur_direct_restormer_v1.json"
SEED = 2026092005
HOLDOUT_GROUPS = {"validation": 128, "internal_test": 128}


def digest(*parts: object) -> str:
    return hashlib.sha256("\0".join(map(str, parts)).encode()).hexdigest()


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    with SOURCE.open(newline="") as handle:
        reader = csv.DictReader(handle)
        source_fields = list(reader.fieldnames or [])
        for row in reader:
            if row["split"] == "train" and row["category"] == "01Blur":
                grouped[row["source_group"]].append(dict(row))
    groups = sorted(grouped, key=lambda group: digest(SEED, "group", group))
    if len(groups) != 7339 or sum(map(len, grouped.values())) != 105057:
        raise RuntimeError("Blur training source pool changed")

    validation_groups = groups[: HOLDOUT_GROUPS["validation"]]
    internal_groups = groups[
        HOLDOUT_GROUPS["validation"] : sum(HOLDOUT_GROUPS.values())
    ]
    train_groups = groups[sum(HOLDOUT_GROUPS.values()) :]
    split_groups = {
        "train": train_groups,
        "validation": validation_groups,
        "internal_test": internal_groups,
    }
    selected: list[dict[str, str]] = []
    split_records: dict[str, int] = {}
    for split, assigned in split_groups.items():
        if split == "train":
            chosen = [row for group in assigned for row in grouped[group]]
        else:
            chosen = [
                min(
                    grouped[group],
                    key=lambda row: digest(SEED, split, row["record_id"]),
                )
                for group in assigned
            ]
        chosen.sort(key=lambda row: digest(SEED, split, row["record_id"]))
        for row in chosen:
            row["original_split"] = row["split"]
            row["split"] = split
            selected.append(row)
        split_records[split] = len(chosen)

    if any(
        set(split_groups[first]) & set(split_groups[second])
        for first in split_groups
        for second in split_groups
        if first < second
    ):
        raise RuntimeError("Blur direct Restormer source groups cross splits")
    fields = source_fields + ["original_split"]
    with OUTPUT.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(selected)
    payload = {
        "schema_version": 1,
        "status": "frozen_training_only_blur_direct_restormer_v1",
        "seed": SEED,
        "records": len(selected),
        "split_records": split_records,
        "split_source_groups": {
            split: len(assigned) for split, assigned in split_groups.items()
        },
        "manifest": OUTPUT.relative_to(ROOT).as_posix(),
        "manifest_sha256": sha256(OUTPUT),
        "source_manifest_sha256": sha256(SOURCE),
        "all_records_originated_from_foundir_training_split": True,
        "development_pixels_accessed": False,
        "official_foundir_test_accessed": False,
    }
    METADATA.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
