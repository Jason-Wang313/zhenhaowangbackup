#!/usr/bin/env python3
"""Replay the frozen physical guard over the cached 16,461-record router development set."""
from __future__ import annotations

import csv
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lowlight_restoration.noise_lowlight_physical_guard import physical_features, physical_route
from lowlight_restoration.qwen_eleven_class_gate import consensus_metrics
from smoke.run_foundir_qwen_five_class_gate_ddp import sha256, workspace_path

CONFIG = ROOT / "configs/foundir_qwen_noise_lowlight_physical_holdout_v3.json"
DUAL_RESULT = ROOT / "checkpoints/foundir_qwen_lowlight_noise_dual_nonlinear_v2/result.json"
SINGLETON_MANIFEST = ROOT / "manifests/foundir_qwen_five_class_development_v1.csv"
MIXED_MANIFEST = ROOT / "manifests/foundir_qwen_mixed_consensus_development_v3.csv"
OUTPUT = ROOT / "checkpoints/foundir_qwen_noise_lowlight_physical_guard_v3/full_development_result.json"
PREDICTIONS = ROOT / "checkpoints/foundir_qwen_noise_lowlight_physical_guard_v3/full_development_predictions.csv"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    if OUTPUT.exists() or PREDICTIONS.exists():
        raise FileExistsError("refusing to overwrite frozen full-development replay")
    config = json.loads(CONFIG.read_text())
    dual_result = json.loads(DUAL_RESULT.read_text())
    dual_predictions_path = workspace_path(dual_result["predictions"])
    if sha256(dual_predictions_path) != dual_result["predictions_sha256"]:
        raise RuntimeError("dual-router predictions changed")
    rule_path = workspace_path(config["physical_guard"]["rule"])
    if sha256(rule_path) != config["physical_guard"]["rule_sha256"]:
        raise RuntimeError("physical rule changed")
    rule = json.loads(rule_path.read_text())["rule"]

    manifest_rows = read_rows(SINGLETON_MANIFEST) + read_rows(MIXED_MANIFEST)
    by_id = {row["record_id"]: row for row in manifest_rows}
    predictions = read_rows(dual_predictions_path)
    if len(predictions) != 16461 or set(by_id) != {row["record_id"] for row in predictions}:
        raise RuntimeError("development identity changed")
    targeted = [row for row in predictions if row["route"] == "noise"]
    started = time.monotonic()

    def measure(prediction: dict[str, str]):
        source = by_id[prediction["record_id"]]
        return physical_features(workspace_path(source["lq_path"]), int(rule["maximum_side"]))

    with ThreadPoolExecutor(max_workers=16) as executor:
        measured = list(executor.map(measure, targeted))
    physical_by_id = {row["record_id"]: item for row, item in zip(targeted, measured, strict=True)}

    truth: list[str] = []
    routes: list[str | None] = []
    changed = 0
    PREDICTIONS.parent.mkdir(parents=True, exist_ok=True)
    with PREDICTIONS.open("x", newline="") as handle:
        fields = ("record_id", "truth", "previous_route", "final_route", "physical_applied", "exposure", "dark_noise")
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for prediction in predictions:
            previous = None if prediction["route"] == "abstain" else prediction["route"]
            final = previous
            item = physical_by_id.get(prediction["record_id"])
            if item is not None:
                final = physical_route(
                    item,
                    float(rule["low_exposure"]),
                    float(rule["high_exposure"]),
                    float(rule["dark_noise_floor"]),
                )
                changed += final != previous
            truth.append(prediction["truth"])
            routes.append(final)
            writer.writerow({
                "record_id": prediction["record_id"], "truth": prediction["truth"],
                "previous_route": previous or "abstain", "final_route": final or "abstain",
                "physical_applied": item is not None, "exposure": "" if item is None else item["exposure"],
                "dark_noise": "" if item is None else item["dark_noise"],
            })
    metrics = consensus_metrics(routes, truth)
    gates = config["gates"]
    overall = metrics["overall"]
    present = {name: value for name, value in metrics["per_class"].items() if value["records"]}
    passed = (
        overall["wrong_rate"] <= gates["maximum_wrong_rate"]
        and overall["coverage"] >= gates["minimum_coverage"]
        and all(value["wrong_rate"] <= gates["maximum_per_class_wrong_rate"] and value["coverage"] >= gates["minimum_per_class_coverage"] for value in present.values())
    )
    payload = {
        "schema_version": 1,
        "status": "frozen_physical_guard_full_development_replay_complete",
        "gate_pass": passed,
        "metrics": {"overall": overall, "per_class": present},
        "physical_guard_inputs": len(targeted),
        "routes_changed_or_refused": changed,
        "rule_sha256": sha256(rule_path),
        "upstream_predictions_sha256": sha256(dual_predictions_path),
        "predictions": PREDICTIONS.relative_to(ROOT).as_posix(),
        "predictions_sha256": sha256(PREDICTIONS),
        "elapsed_seconds": time.monotonic() - started,
        "gpu_hours": 0,
        "development_reused": True,
        "internal_or_official_test_used": False,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print("FOUNDIR_PHYSICAL_FULL_DEVELOPMENT_RESULT " + json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
