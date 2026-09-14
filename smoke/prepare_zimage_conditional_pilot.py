#!/usr/bin/env python3
"""Freeze training-archive-only, source-disjoint conditional teacher pilot."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lowlight_restoration.foundir_zimage import sha256, write_once
from lowlight_restoration.zimage_conditional_teacher import load_settings, fixed_seed
from smoke.build_foundir_blur_only_sanity_manifest import frozen_row, encoded_csv


def select_groups(rows, fixed, count, role, seed):
    groups = {}
    for row in rows:
        groups.setdefault(row["source_group"], []).append(row)
    selected = [dict(row, role=role) for row in fixed]
    used = {row["source_group"] for row in selected}
    for group in sorted(set(groups) - used, key=lambda g: (fixed_seed(seed, role, g), g)):
        row = min(groups[group], key=lambda r: (fixed_seed(seed, r["record_id"]), r["record_id"]))
        selected.append(frozen_row(row, role=role, crop_size=512, crop_seed=seed))
        if len(selected) == count:
            break
    if len(selected) != count or len({r["source_group"] for r in selected}) != count:
        raise ValueError("insufficient unique source groups")
    return selected


def freeze(config_path):
    s, base = load_settings(config_path)
    data = base["data"]
    if sha256(ROOT / data["source_manifest"]) != data["source_manifest_sha256"] or sha256(ROOT / data["frozen_manifest"]) != data["frozen_manifest_sha256"]:
        raise ValueError("parent manifest identity mismatch")
    with (ROOT / data["frozen_manifest"]).open(newline="") as f:
        fixed = list(csv.DictReader(f))
    # Filter metadata before any image access. Sealed/internal rows never enter
    # candidate lists; only the already-approved training-archive split is used.
    by_split = {"train": [], "validation": []}
    with (ROOT / data["source_manifest"]).open(newline="") as f:
        for row in csv.DictReader(f):
            if row["category"] == "01Blur" and row["split"] in by_split:
                if row["composition_size"] != "1":
                    raise ValueError("non-isolated Blur")
                by_split[row["split"]].append(row)
    a = {r["source_group"] for r in by_split["train"]}
    b = {r["source_group"] for r in by_split["validation"]}
    if a & b or len(a) != 7339 or len(b) != 165:
        raise ValueError("parent source split changed")
    rows = []
    for split, role, field in (("train", "pilot_train", "train_source_groups"),
                               ("validation", "pilot_development", "development_source_groups")):
        rows.extend(select_groups(by_split[split], [r for r in fixed if r["split"] == split],
                                  s["stages"]["pilot"][field], role, s["seed"]))
    payload = encoded_csv(rows)
    write_once(ROOT / s["pilot_manifest"], payload)
    metadata = dict(config_sha256=sha256(config_path), source_manifest_sha256=data["source_manifest_sha256"],
                    manifest_sha256=sha256(ROOT / s["pilot_manifest"]), train_records=2048,
                    development_records=128, source_group_overlap=0,
                    original_fixed_16_train_and_16_development_preserved=True,
                    sealed_internal_accessed=False, official_foundir_test_accessed=False)
    write_once(ROOT / s["pilot_manifest_metadata"], (json.dumps(metadata, indent=2, sort_keys=True) + "\n").encode())
    return metadata


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=ROOT / "configs/foundir_blur_only_zimage_conditional_teacher_v1.json")
    print("CONDITIONAL_PILOT_MANIFEST_PASS", json.dumps(freeze(p.parse_args().config), sort_keys=True))
