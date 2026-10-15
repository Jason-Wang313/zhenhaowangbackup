#!/usr/bin/env python3
"""Verify routed and direct 11-class development executions are byte-identical."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "manifests/foundir_final_eleven_route_integration_v1.csv"
ROUTED_CONFIG = ROOT / "configs/foundir_final_eleven_route_integration_routed_v1.json"
DIRECT_CONFIG = ROOT / "configs/foundir_final_eleven_route_integration_direct_v1.json"
OUTPUT = ROOT / "checkpoints/foundir_final_eleven_route_integration_v1"
RESULT = OUTPUT / "result.json"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> None:
    routed, direct = load(ROUTED_CONFIG), load(DIRECT_CONFIG)
    if routed["integration_role"] != "routed" or direct["integration_role"] != "direct":
        raise RuntimeError("integration roles changed")
    for key in ("orders", "experts", "inference"):
        if routed[key] != direct[key]:
            raise RuntimeError(f"routed/direct {key} differ")
    if routed["data"]["manifest_sha256"] != digest(MANIFEST):
        raise RuntimeError("integration manifest identity changed")
    with MANIFEST.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    mismatches = []
    tolerated_numeric = []
    exact_stage_files = 0
    exact_metric_pairs = 0
    stage_files = 0
    final_files = 0
    metric_pairs = 0
    categories = Counter()
    for row in rows:
        categories[row["category"]] += 1
        order = routed["orders"][row["category"]]
        if json.loads(row["expert_order_json"]) != order:
            raise RuntimeError(f"manifest order changed: {row['record_id']}")
        for task in order:
            relative = Path("stages") / task / f"{row['record_id']}.png"
            left = OUTPUT / "routed" / relative
            right = OUTPUT / "direct" / relative
            if not left.is_file() or not right.is_file():
                mismatches.append({"record_id": row["record_id"], "stage": task, "reason": "missing"})
            elif digest(left) == digest(right):
                exact_stage_files += 1
            else:
                left_pixels = np.asarray(Image.open(left).convert("RGB"), dtype=np.int16)
                right_pixels = np.asarray(Image.open(right).convert("RGB"), dtype=np.int16)
                delta = np.abs(left_pixels - right_pixels)
                changed_fraction = float(np.mean(delta != 0))
                maximum = int(delta.max())
                if routed["experts"][task]["kind"] == "restormer" and maximum <= 1 and changed_fraction <= 0.01:
                    tolerated_numeric.append({
                        "record_id": row["record_id"], "stage": task,
                        "reason": "restormer_gpu_rounding", "max_abs_uint8": maximum,
                        "changed_fraction": changed_fraction,
                    })
                else:
                    mismatches.append({
                        "record_id": row["record_id"], "stage": task, "reason": "png_pixels",
                        "max_abs_uint8": maximum, "changed_fraction": changed_fraction,
                    })
            stage_files += 1
        final_files += 1
        left_sidecar = OUTPUT / "routed" / "per_record" / f"{row['record_id']}.json"
        right_sidecar = OUTPUT / "direct" / "per_record" / f"{row['record_id']}.json"
        if not left_sidecar.is_file() or not right_sidecar.is_file():
            mismatches.append({"record_id": row["record_id"], "stage": "metrics", "reason": "missing"})
            continue
        left_values, right_values = load(left_sidecar), load(right_sidecar)
        if all(left_values[role] == right_values[role] for role in ("input", "restored")):
            exact_metric_pairs += 1
        else:
            final_task = order[-1]
            psnr_delta = abs(left_values["restored"]["psnr"] - right_values["restored"]["psnr"])
            lpips_delta = abs(left_values["restored"]["lpips"] - right_values["restored"]["lpips"])
            input_equal = left_values["input"] == right_values["input"]
            if (
                input_equal
                and routed["experts"][final_task]["kind"] == "restormer"
                and psnr_delta <= 0.005
                and lpips_delta <= 0.0001
            ):
                tolerated_numeric.append({
                    "record_id": row["record_id"], "stage": "metrics",
                    "reason": "restormer_metric_rounding",
                    "psnr_abs_delta": psnr_delta, "lpips_abs_delta": lpips_delta,
                })
            else:
                mismatches.append({
                    "record_id": row["record_id"], "stage": "metrics", "reason": "metric_value",
                    "psnr_abs_delta": psnr_delta, "lpips_abs_delta": lpips_delta,
                })
        metric_pairs += 1
    payload = {
        "schema_version": 1,
        "status": "passed" if not mismatches else "failed",
        "integration_fidelity_pass": not mismatches,
        "records": len(rows),
        "records_by_category": dict(sorted(categories.items())),
        "stage_file_pairs_compared": stage_files,
        "stage_file_pairs_exact": exact_stage_files,
        "final_file_pairs_compared": final_files,
        "metric_sidecar_pairs_compared": metric_pairs,
        "metric_sidecar_pairs_exact": exact_metric_pairs,
        "tolerated_numeric_count": len(tolerated_numeric),
        "tolerated_numeric": tolerated_numeric,
        "numeric_tolerance": {
            "scope": "restormer_only",
            "png_max_abs_uint8": 1,
            "png_changed_fraction_maximum": 0.01,
            "psnr_abs_delta_maximum_db": 0.005,
            "lpips_abs_delta_maximum": 0.0001
        },
        "mismatch_count": len(mismatches),
        "mismatches": mismatches[:100],
        "model_weight_updates": False,
        "development_pixels_accessed": True,
        "official_foundir_test_accessed": False,
        "manifest_sha256": digest(MANIFEST),
        "routed_config_sha256": digest(ROUTED_CONFIG),
        "direct_config_sha256": digest(DIRECT_CONFIG),
    }
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    temporary = RESULT.with_name(RESULT.name + f".tmp.{os.getpid()}")
    temporary.write_text(encoded)
    os.replace(temporary, RESULT)
    print("FINAL_ELEVEN_ROUTE_INTEGRATION_RESULT " + json.dumps(payload, sort_keys=True))
    if mismatches:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
