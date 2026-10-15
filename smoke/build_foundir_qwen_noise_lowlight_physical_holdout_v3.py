#!/usr/bin/env python3
"""Freeze every remaining source-disjoint Noise/Noise+Low-light training source."""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lowlight_restoration.qwen_eleven_class_gate import LABEL_BY_CATEGORY
from lowlight_restoration.qwen_five_class_gate import stable_digest
from smoke.run_foundir_qwen_five_class_gate_ddp import sha256

SOURCE = ROOT / "manifests/foundir_five_type_pairs_v1.csv"
EXCLUDED = (
    ROOT / "manifests/foundir_qwen_eleven_class_train_only_v1.csv",
    ROOT / "manifests/foundir_qwen_eleven_class_fresh_holdout_v2.csv",
    ROOT / "manifests/foundir_qwen_lowlight_noise_nonlinear_holdout_v1.csv",
    ROOT / "manifests/foundir_qwen_noise_side_holdout_v2.csv",
)
OUTPUT = ROOT / "manifests/foundir_qwen_noise_lowlight_physical_holdout_v3.csv"
SUMMARY = OUTPUT.with_suffix(".json")
TARGETS = ("noise", "noise+low_light")
SEED = 20260923


def main() -> None:
    used_sources: set[str] = set()
    for path in EXCLUDED:
        with path.open(newline="") as handle:
            used_sources.update(row["source_group"] for row in csv.DictReader(handle))

    pools: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    with SOURCE.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("split") != "train" or row.get("category") not in LABEL_BY_CATEGORY:
                continue
            label = LABEL_BY_CATEGORY[row["category"]]
            if label not in TARGETS or row["source_group"] in used_sources:
                continue
            previous = pools[label].get(row["source_group"])
            if previous is None or stable_digest(SEED, row["record_id"]) < stable_digest(
                SEED, previous["record_id"]
            ):
                pools[label][row["source_group"]] = row

    counts = {label: len(pools[label]) for label in TARGETS}
    if not counts["noise"] or counts["noise"] != counts["noise+low_light"]:
        raise RuntimeError(f"remaining source imbalance: {counts}")

    selected: list[dict[str, str]] = []
    for label in TARGETS:
        rows = sorted(
            pools[label].values(),
            key=lambda row: stable_digest(SEED, label, row["source_group"]),
        )
        for row in rows:
            selected.append(
                {
                    "record_id": row["record_id"],
                    "label": label,
                    "source_group": row["source_group"],
                    "lq_path": row["lq_path"],
                    "lq_sha256": row["lq_sha256"],
                    "width": row["width"],
                    "height": row["height"],
                }
            )
    selected.sort(key=lambda row: (row["label"], row["record_id"]))

    if OUTPUT.exists() or SUMMARY.exists():
        raise FileExistsError("refusing to overwrite frozen physical holdout")
    with OUTPUT.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selected[0]))
        writer.writeheader()
        writer.writerows(selected)

    payload = {
        "schema_version": 1,
        "status": "frozen_before_physical_feature_or_qwen_access",
        "manifest": OUTPUT.relative_to(ROOT).as_posix(),
        "manifest_sha256": sha256(OUTPUT),
        "source_manifest": SOURCE.relative_to(ROOT).as_posix(),
        "source_manifest_sha256": sha256(SOURCE),
        "excluded_partitions": {
            path.relative_to(ROOT).as_posix(): sha256(path) for path in EXCLUDED
        },
        "records": len(selected),
        "records_per_class": counts,
        "source_groups_per_class": counts,
        "paired_source_groups": len({row["source_group"] for row in selected}),
        "pairing_note": "Each untouched source contributes its Noise and Noise+Low-light pair.",
        "training_only": True,
        "development_internal_or_official_test_used": False,
        "effective_sample_size_warning": "Only six new source groups per class remain.",
    }
    SUMMARY.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
