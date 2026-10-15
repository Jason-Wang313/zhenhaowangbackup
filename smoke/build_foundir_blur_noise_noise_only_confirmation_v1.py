#!/usr/bin/env python3
"""Freeze 16 fresh training-only Blur+Noise records for alpha=.5 confirmation."""

from __future__ import annotations

import csv
import glob
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "manifests/foundir_five_type_pairs_v1.csv"
CALIBRATION = ROOT / "manifests/foundir_mixed_restoration_pilot_v1.csv"
OUTPUT = ROOT / "manifests/foundir_blur_noise_noise_only_confirmation_v1.csv"
METADATA = ROOT / "manifests/foundir_blur_noise_noise_only_confirmation_v1.json"
SEED = 2026091810


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def key(row: dict[str, str]) -> str:
    return hashlib.sha256(f"{SEED}\0{row['record_id']}".encode()).hexdigest()


def main() -> None:
    with CALIBRATION.open(newline="") as handle:
        calibration_rows = [
            row for row in csv.DictReader(handle) if row["category"] == "02Blur_Noise"
        ]
    calibration_groups = {row["source_group"] for row in calibration_rows}
    calibration_records = {row["record_id"] for row in calibration_rows}

    qwen_records: set[str] = set()
    qwen_groups: set[str] = set()
    qwen_manifests = []
    for name in sorted(glob.glob(str(ROOT / "manifests/foundir_qwen_*class*.csv"))):
        path = Path(name)
        if "development" in path.name:
            continue
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                qwen_records.add(row["record_id"])
                qwen_groups.add(row["source_group"])
        qwen_manifests.append(path.relative_to(ROOT).as_posix())

    with SOURCE.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        candidates = [
            dict(row)
            for row in reader
            if row["split"] == "train"
            and row["category"] == "02Blur_Noise"
            and row["record_id"] not in calibration_records
            and row["record_id"] not in qwen_records
            and row["source_group"] not in calibration_groups
        ]
    candidates.sort(key=lambda row: (row["source_group"] in qwen_groups, key(row)))
    selected = []
    used = set()
    for row in candidates:
        if row["source_group"] in used:
            continue
        selected.append(row)
        used.add(row["source_group"])
        if len(selected) == 16:
            break
    if len(selected) != 16 or len(used) != 16:
        raise RuntimeError("insufficient fresh Blur+Noise confirmation sources")

    with OUTPUT.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selected)
    payload = {
        "schema_version": 1,
        "status": "frozen_fresh_training_only_blur_noise_confirmation_v1",
        "seed": SEED,
        "records": 16,
        "source_groups": 16,
        "calibration_record_overlap": len({row["record_id"] for row in selected} & calibration_records),
        "calibration_source_group_overlap": len(used & calibration_groups),
        "qwen_record_overlap": len({row["record_id"] for row in selected} & qwen_records),
        "qwen_source_group_overlap": len(used & qwen_groups),
        "qwen_overlap_note": "record overlap is forbidden; source-group overlap is unavoidable because prior Qwen work covered almost all Blur+Noise source groups",
        "qwen_manifests": qwen_manifests,
        "manifest": OUTPUT.relative_to(ROOT).as_posix(),
        "manifest_sha256": sha256(OUTPUT),
        "source_manifest_sha256": sha256(SOURCE),
        "development_pixels_accessed": False,
        "official_foundir_test_accessed": False,
    }
    METADATA.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
