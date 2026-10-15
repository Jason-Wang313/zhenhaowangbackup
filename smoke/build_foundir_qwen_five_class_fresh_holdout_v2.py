#!/usr/bin/env python3
"""Freeze a fully unseen training-source holdout for the conservative v2 gate."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lowlight_restoration.qwen_five_class_gate import CLASS_BY_CATEGORY, CLASSES, stable_digest

SOURCE = ROOT / "manifests/foundir_five_type_pairs_v1.csv"
V1 = ROOT / "manifests/foundir_qwen_five_class_train_only_v1.csv"
OUTPUT = ROOT / "manifests/foundir_qwen_five_class_fresh_holdout_v2.csv"
SUMMARY = OUTPUT.with_suffix(".json")
SEED = 2026091802
PER_CLASS = 60


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    with V1.open(newline="") as handle:
        v1_rows = list(csv.DictReader(handle))
    used_sources = {row["source_group"] for row in v1_rows}
    candidates: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    with SOURCE.open(newline="") as handle:
        for row in csv.DictReader(handle):
            category = row.get("category")
            if (
                row.get("split") != "train"
                or row.get("composition_size") != "1"
                or category not in CLASS_BY_CATEGORY
                or row["source_group"] in used_sources
            ):
                continue
            label = CLASS_BY_CATEGORY[category]
            source = row["source_group"]
            previous = candidates[label].get(source)
            if previous is None or stable_digest(SEED, row["record_id"]) < stable_digest(
                SEED, previous["record_id"]
            ):
                candidates[label][source] = row
    selected = []
    for label in CLASSES:
        pool = sorted(
            candidates[label].values(),
            key=lambda row: stable_digest(SEED, "fresh_holdout", label, row["source_group"]),
        )
        if len(pool) < PER_CLASS:
            raise RuntimeError(f"insufficient globally unseen sources for {label}: {len(pool)}")
        for row in pool[:PER_CLASS]:
            selected.append(
                {
                    "record_id": row["record_id"],
                    "head_split": "fresh_holdout",
                    "label": label,
                    "source_group": row["source_group"],
                    "lq_path": row["lq_path"],
                    "lq_sha256": row["lq_sha256"],
                    "width": row["width"],
                    "height": row["height"],
                }
            )
    if {row["source_group"] for row in selected} & used_sources:
        raise RuntimeError("v2 holdout leaks a v1 source")
    counts = Counter(row["label"] for row in selected)
    if any(counts[label] != PER_CLASS for label in CLASSES):
        raise RuntimeError("v2 fresh holdout is not class balanced")
    selected.sort(key=lambda row: (row["label"], row["record_id"]))
    if OUTPUT.exists() or SUMMARY.exists():
        raise FileExistsError("refusing to replace frozen v2 holdout")
    with OUTPUT.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selected[0]))
        writer.writeheader()
        writer.writerows(selected)
    payload = {
        "schema_version": 1,
        "status": "frozen_before_v2_feature_extraction",
        "purpose": "fresh_training_source_holdout_for_conservative_qwen_gate_v2",
        "source_manifest": SOURCE.relative_to(ROOT).as_posix(),
        "source_manifest_sha256": sha256(SOURCE),
        "excluded_v1_manifest": V1.relative_to(ROOT).as_posix(),
        "excluded_v1_manifest_sha256": sha256(V1),
        "manifest": OUTPUT.relative_to(ROOT).as_posix(),
        "manifest_sha256": sha256(OUTPUT),
        "seed": SEED,
        "records": len(selected),
        "records_per_class": PER_CLASS,
        "counts": dict(sorted(counts.items())),
        "v1_source_overlap": 0,
        "development_or_internal_test_used": False,
    }
    SUMMARY.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
