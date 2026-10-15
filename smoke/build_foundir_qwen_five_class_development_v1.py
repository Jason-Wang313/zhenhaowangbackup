#!/usr/bin/env python3
"""Create a validation-only view for the full five-class Qwen router gate."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lowlight_restoration.qwen_five_class_gate import CLASS_BY_CATEGORY, CLASSES

SOURCE = ROOT / "manifests/foundir_five_type_pairs_v1.csv"
OUTPUT = ROOT / "manifests/foundir_qwen_five_class_development_v1.csv"
SUMMARY = OUTPUT.with_suffix(".json")
EXPECTED = {"blur": 2595, "noise": 1859, "haze": 1580, "rain": 835, "low_light": 1400}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    selected = []
    with SOURCE.open(newline="") as handle:
        for row in csv.DictReader(handle):
            category = row.get("category")
            if (
                row.get("split") != "validation"
                or row.get("composition_size") != "1"
                or category not in CLASS_BY_CATEGORY
            ):
                continue
            selected.append(
                {
                    "record_id": row["record_id"],
                    "head_split": "development",
                    "label": CLASS_BY_CATEGORY[category],
                    "source_group": row["source_group"],
                    "lq_path": row["lq_path"],
                    "lq_sha256": row["lq_sha256"],
                    "width": row["width"],
                    "height": row["height"],
                }
            )
    counts = Counter(row["label"] for row in selected)
    if dict(counts) != EXPECTED or len(selected) != sum(EXPECTED.values()):
        raise RuntimeError(f"development singleton inventory changed: {counts}")
    if set(counts) != set(CLASSES):
        raise RuntimeError("development class identities changed")
    selected.sort(key=lambda row: row["record_id"])
    if OUTPUT.exists() or SUMMARY.exists():
        raise FileExistsError("refusing to replace frozen development view")
    with OUTPUT.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selected[0]))
        writer.writeheader()
        writer.writerows(selected)
    payload = {
        "schema_version": 1,
        "status": "frozen_before_full_development_feature_extraction",
        "purpose": "five_class_qwen_router_full_source_disjoint_development",
        "source_manifest": SOURCE.relative_to(ROOT).as_posix(),
        "source_manifest_sha256": sha256(SOURCE),
        "manifest": OUTPUT.relative_to(ROOT).as_posix(),
        "manifest_sha256": sha256(OUTPUT),
        "records": len(selected),
        "counts": {name: counts[name] for name in CLASSES},
        "split": "validation",
        "composition_size": 1,
        "internal_test_used": False,
    }
    SUMMARY.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
