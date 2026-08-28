#!/usr/bin/env python3
"""Audit whether sequential synthesis suppresses labelled degradations."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
from pathlib import Path
from typing import Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lowlight_restoration.multidegradation_data import (
    read_jsonl,
    sha256,
    write_new_or_verify,
)
from lowlight_restoration.multidegradation_visibility import (
    audit_record_components,
    summarize_component_rows,
)


def _workspace_path(value: object) -> Path:
    path = (ROOT / str(value)).resolve()
    path.relative_to(ROOT.resolve())
    return path


def _audit_one(payload: tuple[dict[str, object], tuple[str, ...]]) -> list[dict[str, object]]:
    record, predicted = payload
    rows, _ = audit_record_components(record, predicted_degradations=predicted)
    return rows


def evaluate_confirmation(
    summary: Mapping[str, object],
    contract: Mapping[str, object],
) -> dict[str, object]:
    misses = summary["misses_by_position"]
    positions = summary["by_position"]
    detected = summary["detected_vs_missed"]
    assert isinstance(misses, Mapping) and isinstance(positions, Mapping) and isinstance(detected, Mapping)
    total_misses = sum(int(misses[str(position)]) for position in range(3))
    first_miss_fraction = int(misses["0"]) / total_misses if total_misses else 0.0
    first_recall = float(positions["0"]["recall"])  # type: ignore[index]
    last_recall = float(positions["2"]["recall"])  # type: ignore[index]
    recall_ratio = first_recall / last_recall if last_recall else 0.0
    metric_space = str(contract.get("metric_space", "raw_rgb"))
    survival_key = (
        "relative_rmse_survival_ratio"
        if metric_space == "luma_normalized_rgb"
        else "rmse_survival_ratio"
    )
    auc_key = (
        "stratified_relative_final_rgb_rmse_detection"
        if metric_space == "luma_normalized_rgb"
        else "stratified_final_rgb_rmse_detection"
    )
    first_survival = float(positions["0"][survival_key]["median"])  # type: ignore[index]
    last_survival = float(positions["2"][survival_key]["median"])  # type: ignore[index]
    survival_ratio = first_survival / last_survival if last_survival else 0.0
    stratified_auc = float(detected[auc_key]["auc"])  # type: ignore[index]
    checks = {
        "first_position_miss_concentration": first_miss_fraction
        >= float(contract["minimum_first_position_miss_fraction"]),
        "first_position_recall_suppression": recall_ratio
        <= float(contract["maximum_first_to_last_recall_ratio"]),
        "counterfactual_survival_suppression": survival_ratio
        <= float(contract["maximum_first_to_last_median_survival_ratio"]),
        "counterfactual_signal_predicts_detection": stratified_auc
        >= float(contract["minimum_stratified_final_rmse_detection_auc"]),
    }
    counterfactual_pass = bool(
        checks["counterfactual_survival_suppression"]
        or checks["counterfactual_signal_predicts_detection"]
    )
    passed_count = sum(checks.values())
    confirmed = passed_count >= int(contract["minimum_required_conditions"])
    if bool(contract["require_at_least_one_counterfactual_condition"]):
        confirmed = confirmed and counterfactual_pass
    return {
        "status": "confirmed" if confirmed else "not_confirmed",
        "metric_space": metric_space,
        "hypothesis_confirmed": confirmed,
        "checks": checks,
        "passed_conditions": passed_count,
        "counterfactual_condition_pass": counterfactual_pass,
        "observed": {
            "first_position_miss_fraction": first_miss_fraction,
            "first_to_last_recall_ratio": recall_ratio,
            "first_to_last_median_survival_ratio": survival_ratio,
            "stratified_final_rmse_detection_auc": stratified_auc,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/qwen_multidegradation_visibility_audit_v1.json",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0 or args.workers <= 0:
        raise SystemExit("limit and workers must be positive")
    config_path = args.config.resolve()
    config_path.relative_to(ROOT.resolve())
    config = json.loads(config_path.read_text())
    manifest_path = _workspace_path(config["manifest"])
    predictions_path = _workspace_path(config["predictions"])
    if sha256(manifest_path) != config["manifest_sha256"]:
        raise RuntimeError("visibility manifest changed")
    if sha256(predictions_path) != config["predictions_sha256"]:
        raise RuntimeError("visibility predictions changed")
    cohort = config["cohort"]
    records = [
        record
        for record in read_jsonl(manifest_path)
        if record["split"] == cohort["split"]
        and int(record["composition_size"]) == int(cohort["composition_size"])
    ]
    if len(records) != int(cohort["expected_records"]):
        raise RuntimeError("visibility cohort cardinality changed")
    predictions_payload = json.loads(predictions_path.read_text())
    prediction_map: dict[str, tuple[str, ...]] = {}
    for row in predictions_payload["records"]:
        predicted = json.loads(row["predicted"])
        prediction_map[str(row["record_id"])] = tuple(predicted["degradations"])
    if any(str(record["record_id"]) not in prediction_map for record in records):
        raise RuntimeError("visibility cohort lacks predictions")
    if args.limit is not None:
        records = records[: args.limit]
    inputs = [
        (record, prediction_map[str(record["record_id"])])
        for record in records
    ]
    component_rows: list[dict[str, object]] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        for index, rows in enumerate(executor.map(_audit_one, inputs), start=1):
            component_rows.extend(rows)
            if index % 10 == 0 or index == len(inputs):
                print(f"QWEN_VISIBILITY_AUDIT_PROGRESS {index}/{len(inputs)}", flush=True)
    summary = summarize_component_rows(component_rows)
    decision = evaluate_confirmation(summary, config["confirmation"])
    expected_components = len(records) * 3
    if len(component_rows) != expected_components:
        raise RuntimeError("visibility component audit is incomplete")
    output = {
        "schema_version": 1,
        "status": "pilot_pass" if args.limit is not None else "pass",
        "config": config_path.relative_to(ROOT).as_posix(),
        "config_sha256": sha256(config_path),
        "manifest_sha256": config["manifest_sha256"],
        "predictions_sha256": config["predictions_sha256"],
        "records": len(records),
        "components": len(component_rows),
        "summary": summary,
        "decision": decision,
        "component_rows": component_rows,
    }
    output_path = (
        args.output.resolve()
        if args.output is not None
        else _workspace_path(config["output"])
    )
    output_path.relative_to(ROOT.resolve())
    write_new_or_verify(
        output_path,
        (json.dumps(output, indent=2, sort_keys=True) + "\n").encode(),
    )
    print(
        "QWEN_VISIBILITY_AUDIT_PASS "
        + json.dumps(
            {
                "records": len(records),
                "components": len(component_rows),
                "output": output_path.relative_to(ROOT).as_posix(),
                "decision": decision,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
