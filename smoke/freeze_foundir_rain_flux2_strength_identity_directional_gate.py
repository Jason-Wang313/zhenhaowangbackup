#!/usr/bin/env python3
"""Freeze the bounded Rain strength/identity directional-gate manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from freeze_foundir_haze_flux2_source_diverse_pilot import (
    FIELDS,
    ROOT,
    render_csv,
    sha256,
    stable_int,
    write_once,
)


def crop_psnr(row: dict[str, str], crop_size: int, x: int, y: int) -> float:
    arrays = []
    for field in ("lq_path", "gt_path"):
        with Image.open(ROOT / row[field]) as image:
            rgb = image.convert("RGB")
            if rgb.size != (int(row["width"]), int(row["height"])):
                raise RuntimeError(f"pair dimensions changed: {row['record_id']}")
            arrays.append(np.asarray(rgb.crop((x, y, x + crop_size, y + crop_size)), dtype=np.float32))
    mse = float(np.mean(((arrays[0] - arrays[1]) / 255.0) ** 2))
    return 100.0 if mse <= 1e-12 else -10.0 * math.log10(mse)


def psnr_bin(value: float) -> str:
    if value < 28.0:
        return "heavy"
    if value < 34.0:
        return "moderate"
    return "near_clean"


def freeze_rows(config: dict) -> tuple[list[dict[str, str]], dict[str, object]]:
    data = config["data"]
    parent_path = ROOT / data["parent_manifest"]
    source_path = ROOT / data["source_manifest"]
    if sha256(parent_path) != data["parent_manifest_sha256"]:
        raise RuntimeError("parent Rain manifest identity changed")
    if sha256(source_path) != data["source_manifest_sha256"]:
        raise RuntimeError("source manifest identity changed")

    with parent_path.open(newline="") as handle:
        parent_rows = list(csv.DictReader(handle))
    parent_train = [row for row in parent_rows if row["role"] == "pilot_train"]
    if len(parent_train) != int(data["tiny_overfit_records"]):
        raise RuntimeError("parent Rain training cardinality changed")
    training = [{**row, "role": data["train_role"]} for row in parent_train]

    excluded_groups = {row["source_group"] for row in parent_rows}
    excluded_path = ROOT / data["excluded_manifest"]
    with excluded_path.open(newline="") as handle:
        excluded_groups.update(
            row["source_group"] for row in csv.DictReader(handle)
            if row["category"] == data["category"]
        )

    with source_path.open(newline="") as handle:
        source_rows = list(csv.DictReader(handle))
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    crop = int(data["crop_size"])
    for row in source_rows:
        if (
            row["category"] == data["category"]
            and row["composition_size"] == "1"
            and row["split"] == data["frozen_validation_split"]
            and row["source_group"] not in excluded_groups
            and int(row["width"]) >= crop
            and int(row["height"]) >= crop
        ):
            groups[row["source_group"]].append(row)

    count = int(data["development_records_per_strength_bin"])
    selected: dict[str, list[dict[str, str]]] = {name: [] for name in ("heavy", "moderate", "near_clean")}
    audited = 0
    ordered_groups = sorted(groups, key=lambda group: stable_int(data["selection_seed"], "directional_development", group))
    for group in ordered_groups:
        row = min(groups[group], key=lambda value: stable_int(data["selection_seed"], "directional_development", value["record_id"]))
        width, height = int(row["width"]), int(row["height"])
        x = stable_int(data["crop_seed"], "directional_development", row["record_id"], "x") % (width - crop + 1)
        y = stable_int(data["crop_seed"], "directional_development", row["record_id"], "y") % (height - crop + 1)
        value = crop_psnr(row, crop, x, y)
        bucket = psnr_bin(value)
        audited += 1
        if len(selected[bucket]) < count:
            selected[bucket].append({
                field: str({**row, "role": data["validation_role"], "crop_x": x, "crop_y": y}[field])
                for field in FIELDS
            })
        if all(len(rows) == count for rows in selected.values()):
            break
    if any(len(rows) != count for rows in selected.values()):
        raise RuntimeError(f"insufficient balanced Rain development records: { {key: len(value) for key, value in selected.items()} }")
    development = [row for name in ("heavy", "moderate", "near_clean") for row in selected[name]]
    if {row["source_group"] for row in training} & {row["source_group"] for row in development}:
        raise RuntimeError("Rain directional train/development leakage")
    return [*training, *development], {
        "available_development_source_groups": len(groups),
        "candidate_groups_audited": audited,
        "development_strength_bin_counts": {key: len(value) for key, value in selected.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())
    if config.get("status") not in {
        "user_authorized_rain_strength_identity_directional_gate",
        "user_authorized_rain_strength_identity_ratio_gate",
    }:
        raise RuntimeError("wrong Rain directional-gate authorization status")
    rows, selection = freeze_rows(config)
    data = config["data"]
    content = render_csv(rows)
    manifest_path = ROOT / data["frozen_manifest"]
    write_once(manifest_path, content)
    train = [row for row in rows if row["role"] == data["train_role"]]
    development = [row for row in rows if row["role"] == data["validation_role"]]
    metadata = {
        "schema_version": 1,
        "status": "frozen_rain_strength_identity_directional_gate_manifest",
        "config": config_path.relative_to(ROOT).as_posix(),
        "config_sha256": sha256(config_path),
        "source_manifest": data["source_manifest"],
        "source_manifest_sha256": data["source_manifest_sha256"],
        "parent_manifest": data["parent_manifest"],
        "parent_manifest_sha256": data["parent_manifest_sha256"],
        "frozen_manifest": data["frozen_manifest"],
        "frozen_manifest_sha256": hashlib.sha256(content).hexdigest(),
        "records": len(rows),
        "train_records": len(train),
        "train_source_groups": len({row["source_group"] for row in train}),
        "development_records": len(development),
        "development_source_groups": len({row["source_group"] for row in development}),
        "train_development_source_group_overlap": 0,
        "reused_exact_parent_training_records_and_crops": True,
        "development_split": "train",
        **selection,
        "qwen_degradation_conditioning": False,
        "mixed_degradations": False,
        "sealed_internal_accessed": False,
        "official_foundir_test_accessed": False,
        "protected_foundir_test_accessed": False
    }
    write_once(ROOT / data["frozen_manifest_metadata"], (json.dumps(metadata, indent=2, sort_keys=True) + "\n").encode())
    print("FOUNDIR_RAIN_STRENGTH_IDENTITY_MANIFEST_PASS " + json.dumps(metadata, sort_keys=True))


if __name__ == "__main__":
    main()
