#!/usr/bin/env python3
"""Freeze the complete mixed development set for the consensus router."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lowlight_restoration.qwen_eleven_class_gate import LABEL_BY_CATEGORY, MIXED_CLASSES
SOURCE = ROOT / "manifests/foundir_five_type_pairs_v1.csv"
V1 = ROOT / "manifests/foundir_qwen_eleven_class_train_only_v1.csv"
V2 = ROOT / "manifests/foundir_qwen_eleven_class_fresh_holdout_v2.csv"
OUTPUT = ROOT / "manifests/foundir_qwen_mixed_consensus_development_v3.csv"
SUMMARY = OUTPUT.with_suffix(".json")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    used_sources: set[str] = set()
    for path in (V1, V2):
        with path.open(newline="") as handle:
            used_sources.update(row["source_group"] for row in csv.DictReader(handle))

    selected: list[dict[str, str]] = []
    with SOURCE.open(newline="") as handle:
        for row in csv.DictReader(handle):
            category = row.get("category")
            source = row.get("source_group", "")
            if row.get("split") != "validation" or category not in LABEL_BY_CATEGORY:
                continue
            label = LABEL_BY_CATEGORY[category]
            if label not in MIXED_CLASSES:
                continue
            if source in used_sources:
                raise RuntimeError(f"development source overlaps Qwen training: {source}")
            selected.append(
                {
                    "record_id": row["record_id"],
                    "label": label,
                    "source_group": source,
                    "lq_path": row["lq_path"],
                    "lq_sha256": row["lq_sha256"],
                    "width": row["width"],
                    "height": row["height"],
                }
            )
    selected.sort(key=lambda row: (row["label"], row["record_id"]))
    expected = {
        "blur+noise": 579,
        "haze+low_light": 1580,
        "rain+haze": 1580,
        "blur+low_light": 2131,
        "noise+low_light": 1743,
        "blur+noise+low_light": 579,
    }
    if Counter(row["label"] for row in selected) != Counter(expected):
        raise RuntimeError("holdout is not balanced")
    if len({row["record_id"] for row in selected}) != len(selected):
        raise RuntimeError("holdout repeats a record")
    if {row["source_group"] for row in selected} & used_sources:
        raise RuntimeError("V3 source overlaps V1/V2")
    if OUTPUT.exists() or SUMMARY.exists():
        raise FileExistsError("refusing to replace frozen V3 holdout")
    with OUTPUT.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selected[0]))
        writer.writeheader()
        writer.writerows(selected)
    payload = {
        "schema_version": 1,
        "status": "complete_development_frozen_before_feature_extraction",
        "source_manifest": SOURCE.relative_to(ROOT).as_posix(),
        "source_manifest_sha256": sha256(SOURCE),
        "excluded_partitions": {
            path.relative_to(ROOT).as_posix(): sha256(path) for path in (V1, V2)
        },
        "manifest": OUTPUT.relative_to(ROOT).as_posix(),
        "manifest_sha256": sha256(OUTPUT),
        "classes": list(MIXED_CLASSES),
        "records": len(selected),
        "records_per_class": expected,
        "source_groups": len({row["source_group"] for row in selected}),
        "source_disjoint_from_v1_and_v2": True,
        "split": "development",
        "internal_or_official_test_used": False,
    }
    SUMMARY.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
