#!/usr/bin/env python3
"""Freeze source-disjoint training-only splits for a Blur+Noise Restormer pilot."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "manifests/foundir_five_type_pairs_v1.csv"
OUTPUT = ROOT / "manifests/foundir_blur_noise_restormer_pilot_v1.csv"
METADATA = ROOT / "manifests/foundir_blur_noise_restormer_pilot_v1.json"
SEED = 2026091813
TARGETS = {"train": (768, 2048), "validation": (83, 128), "internal_test": (83, 128)}


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
            if row["split"] == "train" and row["category"] == "02Blur_Noise":
                grouped[row["source_group"]].append(dict(row))
    groups = sorted(grouped, key=lambda group: digest(SEED, "group", group))
    if len(groups) != 934:
        raise RuntimeError("Blur+Noise training source-group count changed")

    selected = []
    cursor = 0
    split_groups = {}
    for split, (group_count, record_count) in TARGETS.items():
        assigned = groups[cursor : cursor + group_count]
        cursor += group_count
        split_groups[split] = assigned
        candidates = [row for group in assigned for row in grouped[group]]
        candidates.sort(key=lambda row: digest(SEED, split, row["record_id"]))
        if len(candidates) < record_count:
            raise RuntimeError(f"insufficient Blur+Noise records for {split}")
        mandatory = [
            min(grouped[group], key=lambda row: digest(SEED, split, "mandatory", row["record_id"]))
            for group in assigned
        ]
        mandatory_ids = {row["record_id"] for row in mandatory}
        chosen = mandatory + [
            row for row in candidates if row["record_id"] not in mandatory_ids
        ][: record_count - len(mandatory)]
        for row in chosen:
            row["original_split"] = row["split"]
            row["split"] = split
            selected.append(row)
    if cursor != len(groups) or len(selected) != 2304:
        raise RuntimeError("Blur+Noise pilot split coverage changed")
    if any(set(split_groups[a]) & set(split_groups[b]) for a in TARGETS for b in TARGETS if a < b):
        raise RuntimeError("Blur+Noise pilot source groups cross splits")

    fields = source_fields + ["original_split"]
    with OUTPUT.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(selected)
    payload = {
        "schema_version": 1,
        "status": "frozen_training_only_blur_noise_restormer_pilot_v1",
        "seed": SEED,
        "records": len(selected),
        "split_records": {split: count for split, (_, count) in TARGETS.items()},
        "split_source_groups": {split: count for split, (count, _) in TARGETS.items()},
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
