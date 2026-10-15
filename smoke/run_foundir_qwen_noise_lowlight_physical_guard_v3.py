#!/usr/bin/env python3
"""Freeze one simple physical Noise/Noise+Low-light rule on prior calibration data."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lowlight_restoration.noise_lowlight_physical_guard import physical_features, physical_route
from smoke.run_foundir_qwen_five_class_gate_ddp import sha256, workspace_path


def metrics(routes: list[str | None], truth: list[str]) -> dict:
    per_class = {}
    for label in ("noise", "noise+low_light"):
        indices = [index for index, value in enumerate(truth) if value == label]
        correct = sum(routes[index] == label for index in indices)
        wrong = sum(routes[index] not in (None, label) for index in indices)
        refusal = len(indices) - correct - wrong
        per_class[label] = {
            "records": len(indices),
            "correct": correct,
            "wrong": wrong,
            "refusal": refusal,
            "wrong_rate": wrong / len(indices),
            "coverage": (correct + wrong) / len(indices),
        }
    records = len(truth)
    correct = sum(route == label for route, label in zip(routes, truth, strict=True))
    wrong = sum(route not in (None, label) for route, label in zip(routes, truth, strict=True))
    return {
        "overall": {
            "records": records,
            "correct": correct,
            "wrong": wrong,
            "refusal": records - correct - wrong,
            "wrong_rate": wrong / records,
            "coverage": (correct + wrong) / records,
        },
        "per_class": per_class,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())
    config_sha = sha256(config_path)
    manifest = workspace_path(config["data"]["calibration_manifest"])
    if sha256(manifest) != config["data"]["calibration_manifest_sha256"]:
        raise RuntimeError("calibration manifest changed")
    if config["data"].get("internal_or_official_test_allowed") is not False:
        raise RuntimeError("protected test access forbidden")
    with manifest.open(newline="") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row["head_split"] == "calibration"
            and row["label"] in {"noise", "noise+low_light"}
        ]
    if len(rows) != 180 or any(sum(row["label"] == label for row in rows) != 90 for label in ("noise", "noise+low_light")):
        raise RuntimeError("calibration cardinality changed")
    if args.check_only:
        print(f"FOUNDIR_PHYSICAL_GUARD_PREFLIGHT_PASS records={len(rows)} config_sha256={config_sha}")
        return

    rule_path = workspace_path(config["outputs"]["rule"])
    predictions_path = workspace_path(config["outputs"]["calibration_predictions"])
    if rule_path.exists() or predictions_path.exists():
        raise FileExistsError("refusing to overwrite physical guard artifacts")

    measured = [physical_features(workspace_path(row["lq_path"])) for row in rows]
    truth = [row["label"] for row in rows]
    exposures = sorted({item["exposure"] for item in measured})
    candidates = [exposures[0] - 1e-9]
    candidates.extend((left + right) / 2.0 for left, right in zip(exposures[:-1], exposures[1:], strict=True))
    candidates.append(exposures[-1] + 1e-9)
    # A fixed, sensor-scale floor: about one 8-bit luma level after robust MAD scaling.
    dark_noise_floor = float(config["rule"]["dark_noise_floor"])
    selected = None
    gates = config["gates"]
    for low in candidates:
        for high in candidates:
            if low > high:
                continue
            routes = [physical_route(item, low, high, dark_noise_floor) for item in measured]
            result = metrics(routes, truth)
            valid = all(
                value["wrong_rate"] <= gates["maximum_per_class_wrong_rate"]
                for value in result["per_class"].values()
            )
            key = (int(valid), -result["overall"]["wrong_rate"], result["overall"]["coverage"], high - low)
            if selected is None or key > selected["key"]:
                selected = {"low_exposure": low, "high_exposure": high, "metrics": result, "key": key}
    assert selected is not None
    selected["dark_noise_floor"] = dark_noise_floor
    selected.pop("key")

    rule_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "status": "physical_rule_frozen_before_v3_holdout_access",
        "config_sha256": config_sha,
        "calibration_manifest_sha256": sha256(manifest),
        "rule": {
            "low_exposure": selected["low_exposure"],
            "high_exposure": selected["high_exposure"],
            "dark_noise_floor": dark_noise_floor,
            "maximum_side": 512,
        },
        "calibration": selected["metrics"],
        "gpu_hours": 0,
        "development_internal_or_official_test_used": False,
    }
    rule_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    with predictions_path.open("x", newline="") as handle:
        fields = ("record_id", "truth", "route", "exposure", "dark_noise", "p10", "p50", "p90", "dark_fraction")
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row, label, item in zip(rows, truth, measured, strict=True):
            writer.writerow({"record_id": row["record_id"], "truth": label, "route": physical_route(item, selected["low_exposure"], selected["high_exposure"], dark_noise_floor) or "abstain", **item})
    print("FOUNDIR_PHYSICAL_GUARD_FROZEN " + json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
