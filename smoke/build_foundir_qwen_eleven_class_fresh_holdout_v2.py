#!/usr/bin/env python3
"""Freeze one fresh source-disjoint holdout for the bounded eleven-class retry."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lowlight_restoration.qwen_eleven_class_gate import CLASSES, LABEL_BY_CATEGORY
from lowlight_restoration.qwen_five_class_gate import stable_digest

SOURCE = ROOT / "manifests/foundir_five_type_pairs_v1.csv"
V1 = ROOT / "manifests/foundir_qwen_eleven_class_train_only_v1.csv"
OUTPUT = ROOT / "manifests/foundir_qwen_eleven_class_fresh_holdout_v2.csv"
SUMMARY = OUTPUT.with_suffix(".json")
SEED = 20260919
FRESH_PER_CLASS = 40


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    with V1.open(newline="") as handle:
        previous = list(csv.DictReader(handle))
    selected = [dict(row) for row in previous if row["head_split"] in {"fit", "calibration"}]
    used_sources = {row["source_group"] for row in previous}
    candidates: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    with SOURCE.open(newline="") as handle:
        for row in csv.DictReader(handle):
            category = row.get("category")
            if row.get("split") != "train" or category not in LABEL_BY_CATEGORY:
                continue
            label = LABEL_BY_CATEGORY[category]
            source = row["source_group"]
            if source in used_sources:
                continue
            prior = candidates[label].get(source)
            if prior is None or stable_digest(SEED, "record", row["record_id"]) < stable_digest(SEED, "record", prior["record_id"]):
                candidates[label][source] = row
    fresh = []
    for label in CLASSES:
        pool = sorted(candidates[label].values(), key=lambda row: stable_digest(SEED, "fresh", label, row["source_group"]))
        if len(pool) < FRESH_PER_CLASS:
            raise RuntimeError(f"insufficient fresh sources for {label}: {len(pool)}")
        for row in pool[:FRESH_PER_CLASS]:
            fresh.append({
                "record_id": row["record_id"],
                "head_split": "holdout",
                "label": label,
                "source_group": row["source_group"],
                "lq_path": row["lq_path"],
                "lq_sha256": row["lq_sha256"],
                "width": row["width"],
                "height": row["height"],
            })
    selected.extend(fresh)
    selected.sort(key=lambda row: (row["head_split"], row["label"], row["record_id"]))
    counts = Counter((row["head_split"], row["label"]) for row in selected)
    expected = {"fit": 450, "calibration": 90, "holdout": FRESH_PER_CLASS}
    if any(counts[(split, label)] != quota for split, quota in expected.items() for label in CLASSES):
        raise RuntimeError("fresh holdout partition is not balanced")
    locations: dict[str, set[str]] = defaultdict(set)
    for row in selected:
        locations[row["source_group"]].add(row["head_split"])
    if any(len(splits) != 1 for splits in locations.values()):
        raise RuntimeError("source leakage in fresh holdout partition")
    if len({row["record_id"] for row in selected}) != len(selected):
        raise RuntimeError("fresh holdout partition repeats a record")
    if OUTPUT.exists() or SUMMARY.exists():
        raise FileExistsError("refusing to replace frozen fresh holdout")
    with OUTPUT.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selected[0]))
        writer.writeheader()
        writer.writerows(selected)
    payload = {
        "schema_version": 1,
        "status": "frozen_before_fresh_holdout_feature_extraction",
        "source_manifest": SOURCE.relative_to(ROOT).as_posix(),
        "source_manifest_sha256": sha256(SOURCE),
        "v1_partition": V1.relative_to(ROOT).as_posix(),
        "v1_partition_sha256": sha256(V1),
        "manifest": OUTPUT.relative_to(ROOT).as_posix(),
        "manifest_sha256": sha256(OUTPUT),
        "seed": SEED,
        "classes": list(CLASSES),
        "records": len(selected),
        "records_per_class": expected,
        "fresh_holdout_records": len(fresh),
        "fresh_holdout_sources_excluded_from_complete_v1_partition": True,
        "source_disjoint_across_head_splits": True,
        "development_or_internal_test_used": False,
    }
    SUMMARY.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
