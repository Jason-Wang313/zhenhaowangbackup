#!/usr/bin/env python3
"""Compare frozen Qwen candidates on FoundIR training types only."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lowlight_restoration.foundir_five_type import (  # noqa: E402
    DEGRADATION_ORDER,
    FIVE_TYPE_CATEGORIES,
)
from smoke.export_foundir_five_type_qwen_conditions import sha256, write_once  # noqa: E402


def workspace_path(value: str | Path) -> Path:
    path = (ROOT / value).resolve()
    path.relative_to(ROOT.resolve())
    return path


def calculate(predictions: Sequence[Mapping[str, object]]) -> dict[str, object]:
    counts = {
        name: {"tp": 0, "fp": 0, "fn": 0}
        for name in DEGRADATION_ORDER
    }
    valid = exact = 0
    composition = {1: [0, 0], 2: [0, 0], 3: [0, 0]}
    for row in predictions:
        ground_truth = set(FIVE_TYPE_CATEGORIES[str(row["category"])])
        levels = row.get("levels")
        is_valid = bool(row.get("valid")) and isinstance(levels, Mapping)
        predicted = (
            {name for name in DEGRADATION_ORDER if int(levels.get(name, 1)) > 1}
            if is_valid
            else set()
        )
        valid += is_valid
        match = predicted == ground_truth
        exact += match
        size = int(row["composition_size"])
        composition[size][0] += int(match)
        composition[size][1] += 1
        for name in DEGRADATION_ORDER:
            active_prediction = name in predicted
            active_truth = name in ground_truth
            if active_prediction and active_truth:
                counts[name]["tp"] += 1
            elif active_prediction:
                counts[name]["fp"] += 1
            elif active_truth:
                counts[name]["fn"] += 1
    total = len(predictions)
    if not total:
        raise ValueError("Qwen training-domain gate has no predictions")
    per_type: dict[str, dict[str, float | int]] = {}
    tp = fp = fn = 0
    for name, value in counts.items():
        tp += value["tp"]
        fp += value["fp"]
        fn += value["fn"]
        precision = value["tp"] / (value["tp"] + value["fp"]) if value["tp"] + value["fp"] else 0.0
        recall = value["tp"] / (value["tp"] + value["fn"]) if value["tp"] + value["fn"] else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_type[name] = {**value, "precision": precision, "recall": recall, "f1": f1}
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    micro_f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "records": total,
        "valid_output_rate": valid / total,
        "type_micro_precision": precision,
        "type_micro_recall": recall,
        "type_micro_f1": micro_f1,
        "type_macro_f1": sum(float(value["f1"]) for value in per_type.values()) / len(per_type),
        "degradation_set_exact": exact / total,
        "minimum_type_recall": min(float(value["recall"]) for value in per_type.values()),
        "composition_set_exact": {
            str(size): values[0] / values[1] if values[1] else 0.0
            for size, values in composition.items()
        },
        "per_type": per_type,
    }


def joint_predictions(path: Path) -> list[dict[str, object]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    predictions = []
    for row in rows:
        try:
            levels = json.loads(row["qwen_levels_json"])
            valid = set(levels) == set(DEGRADATION_ORDER)
        except Exception:
            levels, valid = None, False
        predictions.append(
            {
                "record_id": row["record_id"],
                "category": row["category"],
                "composition_size": int(row["composition_size"]),
                "levels": levels,
                "valid": valid,
            }
        )
    return predictions


def checks(metrics: Mapping[str, object], thresholds: Mapping[str, object]) -> dict[str, bool]:
    result = {
        "valid_output_rate": float(metrics["valid_output_rate"]) >= float(thresholds["minimum_valid_output_rate"]),
        "type_micro_f1": float(metrics["type_micro_f1"]) >= float(thresholds["minimum_type_micro_f1"]),
        "type_macro_f1": float(metrics["type_macro_f1"]) >= float(thresholds["minimum_type_macro_f1"]),
        "degradation_set_exact": float(metrics["degradation_set_exact"]) >= float(thresholds["minimum_degradation_set_exact"]),
        "each_type_recall": float(metrics["minimum_type_recall"]) >= float(thresholds["minimum_each_type_recall"]),
    }
    return result


def evaluate_candidates(
    candidates: Mapping[str, Sequence[Mapping[str, object]]],
    *,
    records: int,
    thresholds: Mapping[str, object],
    selection_rule: Sequence[str],
) -> tuple[dict[str, object], str | None]:
    """Evaluate candidates only when they cover the same ordered pilot.

    Keeping this check outside ``main`` makes the scientific comparison easy
    to unit-test without touching the immutable GPU artifacts.
    """

    expected_records: list[tuple[str, str, int]] | None = None
    results: dict[str, object] = {}
    for name, predictions in candidates.items():
        ids = [str(row["record_id"]) for row in predictions]
        if len(ids) != records or len(set(ids)) != len(ids):
            raise RuntimeError(f"{name} pilot cardinality changed")
        identities = [
            (
                str(row["record_id"]),
                str(row["category"]),
                int(row["composition_size"]),
            )
            for row in predictions
        ]
        if expected_records is None:
            expected_records = identities
        elif identities != expected_records:
            raise RuntimeError("Qwen candidates were not evaluated on identical records")
        metrics = calculate(predictions)
        candidate_checks = checks(metrics, thresholds)
        results[name] = {
            "metrics": metrics,
            "checks": candidate_checks,
            "automatic_gate_pass": all(candidate_checks.values()),
        }
    if not results:
        raise RuntimeError("Qwen training-domain gate has no candidates")
    passing = [
        name
        for name, result in results.items()
        if bool(result["automatic_gate_pass"])
    ]
    selected = (
        max(
            passing,
            key=lambda name: tuple(
                float(results[name]["metrics"][metric])
                for metric in selection_rule
            ),
        )
        if passing
        else None
    )
    return results, selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prelaunch-config",
        type=Path,
        default=ROOT / "configs/foundir_five_type_prelaunch_gate_v1.json",
    )
    parser.add_argument(
        "--require-selected",
        choices=("joint", "per_type"),
        help="fail unless the frozen comparison selected this restoration condition source",
    )
    args = parser.parse_args()
    config_path = workspace_path(args.prelaunch_config)
    config = json.loads(config_path.read_text())
    if (
        config.get("status") != "frozen_training_only_prelaunch_gate"
        or config.get("protected_foundir_test_accessed") is not False
    ):
        raise RuntimeError("Qwen training-domain gate is not frozen or permits protected data")
    restoration_config = workspace_path(config["restoration_config"])
    if sha256(restoration_config) != config["restoration_config_sha256"]:
        raise RuntimeError("frozen restoration config changed")
    candidates = config["candidates"]
    for candidate in candidates.values():
        for path_key, hash_key in (
            ("qwen_config", "qwen_config_sha256"),
            ("adapter", "adapter_sha256"),
        ):
            path = workspace_path(candidate[path_key])
            if sha256(path) != candidate[hash_key]:
                raise RuntimeError(f"frozen Qwen candidate changed: {path}")
    joint_path = workspace_path(candidates["joint"]["predictions"])
    joint_summary = workspace_path(candidates["joint"]["summary"])
    per_type_path = workspace_path(candidates["per_type"]["predictions"])
    for path, expected in (
        (joint_path, candidates["joint"]["predictions_sha256"]),
        (joint_summary, candidates["joint"]["summary_sha256"]),
    ):
        if sha256(path) != expected:
            raise RuntimeError(f"frozen joint Qwen artifact changed: {path}")
    joint_payload = json.loads(joint_summary.read_text())
    if (
        joint_payload.get("status") != "frozen_joint_qwen_foundir_training_conditions"
        or joint_payload.get("scope") != "balanced_overfit_pilot"
        or joint_payload.get("records") != int(config["records"])
        or joint_payload.get("manifest_sha256") != config["manifest_sha256"]
        or joint_payload.get("conditions_csv_sha256")
        != candidates["joint"]["predictions_sha256"]
        or joint_payload.get("qwen_config_sha256")
        != candidates["joint"]["qwen_config_sha256"]
        or joint_payload.get("qwen_adapter_sha256")
        != candidates["joint"]["adapter_sha256"]
        or joint_payload.get("protected_foundir_test_accessed") is not False
    ):
        raise RuntimeError("frozen joint Qwen pilot identity changed")
    per_type_payload = json.loads(per_type_path.read_text())
    if (
        per_type_payload.get("status")
        != "frozen_per_type_qwen_foundir_training_predictions"
        or per_type_payload.get("prelaunch_config_sha256") != sha256(config_path)
        or per_type_payload.get("manifest_sha256") != config["manifest_sha256"]
        or per_type_payload.get("qwen_config_sha256")
        != candidates["per_type"]["qwen_config_sha256"]
        or per_type_payload.get("qwen_adapter_sha256")
        != candidates["per_type"]["adapter_sha256"]
        or per_type_payload.get("records") != int(config["records"])
        or per_type_payload.get("decoding_version") != "left_padding_v2"
        or per_type_payload.get("padding_side") != "left"
        or per_type_payload.get("protected_foundir_test_accessed") is not False
    ):
        raise RuntimeError("per-type Qwen pilot identity changed")
    all_predictions = {
        "joint": joint_predictions(joint_path),
        "per_type": per_type_payload["predictions"],
    }
    order = list(config["selection_rule"])
    results, selected = evaluate_candidates(
        all_predictions,
        records=int(config["records"]),
        thresholds=config["thresholds"],
        selection_rule=order,
    )
    payload = {
        "schema_version": 1,
        "status": "pass" if selected else "failed",
        "prelaunch_config": config_path.relative_to(ROOT).as_posix(),
        "prelaunch_config_sha256": sha256(config_path),
        "records": int(config["records"]),
        "manifest_sha256": config["manifest_sha256"],
        "scope": "authorized_foundir_training_only_type_labels",
        "thresholds": config["thresholds"],
        "selection_rule": order,
        "candidates": results,
        "selected_candidate": selected,
        "severity_evaluated": False,
        "severity_exclusion_reason": "FoundIR training folders provide type labels but no project L1-L5 severity truth",
        "protected_foundir_test_accessed": False,
    }
    output = workspace_path(config["output"])
    content = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    write_once(output, content)
    print("FOUNDIR_FIVE_TYPE_QWEN_TRAINING_GATE " + json.dumps(payload, sort_keys=True), flush=True)
    if selected is None:
        raise SystemExit(2)
    if args.require_selected is not None and selected != args.require_selected:
        print(
            "FOUNDIR_FIVE_TYPE_QWEN_SELECTION_MISMATCH "
            f"required={args.require_selected} selected={selected}",
            flush=True,
        )
        raise SystemExit(2)


if __name__ == "__main__":
    main()
