#!/usr/bin/env python3
"""Freeze one pure-Noise record for each cached Noise+Low-light confirmation source."""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lowlight_restoration.qwen_five_class_gate import stable_digest
from smoke.run_foundir_qwen_five_class_gate_ddp import sha256

SOURCE = ROOT / "manifests/foundir_five_type_pairs_v1.csv"
MIXED = ROOT / "manifests/foundir_mixed_restoration_confirmation_v1.csv"
OUTPUT = ROOT / "manifests/foundir_noise_family_cross_compatibility_v1.csv"
SUMMARY = OUTPUT.with_suffix(".json")
SEED = 20260925


def main() -> None:
    with MIXED.open(newline="") as handle:
        mixed = [row for row in csv.DictReader(handle) if row["category"] == "16Lowlight_Noise"]
    source_groups = {row["source_group"] for row in mixed}
    if len(mixed) != 128 or len(source_groups) != 128:
        raise RuntimeError("mixed confirmation identity changed")
    pools: dict[str, list[dict[str, str]]] = defaultdict(list)
    with SOURCE.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("split") == "train" and row.get("category") == "05Noise" and row["source_group"] in source_groups:
                pools[row["source_group"]].append(row)
    if set(pools) != source_groups:
        raise RuntimeError("not every mixed source has a pure-Noise counterpart")
    selected = [
        min(pools[group], key=lambda row: stable_digest(SEED, group, row["record_id"]))
        for group in sorted(source_groups)
    ]
    rows = [
        {
            "record_id": row["record_id"], "split": "train", "source_group": row["source_group"],
            "category": "05Noise", "lq_path": row["lq_path"], "gt_path": row["gt_path"],
            "lq_sha256": row["lq_sha256"], "gt_sha256": row["gt_sha256"],
            "width": row["width"], "height": row["height"],
            "expert_order_json": json.dumps(["noise", "low_light"], separators=(",", ":")),
        }
        for row in selected
    ]
    rows.sort(key=lambda row: row["record_id"])
    if OUTPUT.exists() or SUMMARY.exists():
        raise FileExistsError("refusing to overwrite frozen cross-compatibility manifest")
    with OUTPUT.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "schema_version": 1, "status": "frozen_before_cross_compatibility_inference",
        "manifest": OUTPUT.relative_to(ROOT).as_posix(), "manifest_sha256": sha256(OUTPUT),
        "source_manifest_sha256": sha256(SOURCE), "mixed_confirmation_manifest_sha256": sha256(MIXED),
        "records": 128, "source_groups": 128, "category": "05Noise", "split": "train",
        "matched_to_noise_lowlight_confirmation_sources": True,
        "development_internal_or_official_test_used": False,
    }
    SUMMARY.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
