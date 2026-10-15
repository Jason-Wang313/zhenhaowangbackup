#!/usr/bin/env python3
"""Build a training-only Blur manifest matched to official 1920x1080 geometry.

Hard-region selection happens inside the trainer from four candidate crops.
This builder never reads official test paths or pixels.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "manifests/foundir_blur_direct_restormer_v1.csv"
OUTPUT = ROOT / "manifests/foundir_blur_heavy_training_v1.csv"
SUMMARY = ROOT / "manifests/foundir_blur_heavy_training_v1.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    with SOURCE.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    selected = [
        row for row in rows
        if row["split"] in ("train", "validation")
        and row["width"] == "1920"
        and row["height"] == "1080"
    ]
    if any("foundir_official" in row["lq_path"] or "foundir_official" in row["gt_path"] for row in selected):
        raise RuntimeError("official test path escaped into training manifest")
    if OUTPUT.exists() or SUMMARY.exists():
        raise RuntimeError("heavy Blur manifest output already exists")
    with OUTPUT.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(selected)
    counts = {
        split: {
            "records": sum(row["split"] == split for row in selected),
            "source_groups": len({row["source_group"] for row in selected if row["split"] == split}),
        }
        for split in ("train", "validation")
    }
    payload = {
        "schema_version": 1,
        "status": "training_only_geometry_matched_heavy_crop_manifest_complete",
        "source_manifest": SOURCE.relative_to(ROOT).as_posix(),
        "source_manifest_sha256": sha256(SOURCE),
        "manifest": OUTPUT.relative_to(ROOT).as_posix(),
        "manifest_sha256": sha256(OUTPUT),
        "geometry": {"width": 1920, "height": 1080},
        "hard_region_policy": "trainer_selects_max_mse_of_four_deterministic_candidate_crops",
        "counts": counts,
        "official_test_accessed": False,
    }
    SUMMARY.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
