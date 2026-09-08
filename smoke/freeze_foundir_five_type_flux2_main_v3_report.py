#!/usr/bin/env python3
"""Freeze the no-ablation five-type FLUX development result."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lowlight_restoration.flux_prompt_cache import file_sha256  # noqa: E402
from lowlight_restoration.foundir_flux2_protocol import (  # noqa: E402
    SOW_STAGE2_THRESHOLDS,
    resolve_workspace_path,
)


def load(path: Path, status: str, config_sha: str) -> dict[str, object]:
    payload = json.loads(path.read_text())
    if (
        payload.get("status") != status
        or payload.get("config_sha256") != config_sha
        or payload.get("protected_foundir_test_accessed") is not False
    ):
        raise RuntimeError(f"invalid v3 final evidence: {path}")
    return payload


def finite_number(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(float(value))


def validate_task_metrics(payload: Mapping[str, object], *, label: str) -> Mapping[str, object]:
    summary = payload.get("summary")
    if not isinstance(summary, dict):
        raise RuntimeError(f"{label} has no summary contract")
    task_metrics = summary.get("task_metrics")
    if not isinstance(task_metrics, dict) or set(task_metrics) != set(SOW_STAGE2_THRESHOLDS):
        raise RuntimeError(f"{label} does not cover all five restoration tasks")
    for task, metrics in task_metrics.items():
        if not isinstance(metrics, dict) or not all(
            finite_number(metrics.get(name)) for name in ("psnr", "lpips")
        ):
            raise RuntimeError(f"{label} has invalid metrics for {task}")
    return task_metrics


def validate_released_baseline(
    payload: Mapping[str, object],
    *,
    label: str,
    split: str,
    expected_records: int,
    conditions_sha: str,
) -> Mapping[str, object]:
    if (
        payload.get("split") != split
        or payload.get("records") != expected_records
        or payload.get("conditions_sha256") != conditions_sha
    ):
        raise RuntimeError(f"{label} identity/cardinality changed")
    candidate_summaries = payload.get("candidate_summaries")
    if (
        not isinstance(payload.get("summary"), dict)
        or not isinstance(candidate_summaries, dict)
        or candidate_summaries.get("generalist") != payload.get("summary")
    ):
        raise RuntimeError(f"{label} generalist summary contract changed")
    if payload["summary"].get("records") != expected_records:
        raise RuntimeError(f"{label} generalist coverage changed")
    return validate_task_metrics(payload, label=label)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--development-baseline", type=Path, required=True)
    parser.add_argument("--sealed", type=Path, required=True)
    parser.add_argument("--sealed-baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for key in vars(args):
        setattr(args, key, resolve_workspace_path(ROOT, getattr(args, key)))
    config = json.loads(args.config.read_text())
    config_sha = file_sha256(args.config)
    if (
        config["scope"].get("exclude_jpeg") is not True
        or config["evaluation"].get("condition_arms") != ["qwen"]
        or config["evaluation"].get("qwen_condition_ablation")
        != "disabled_by_user_2026_09_06"
        or config.get("pixel_refinement", {}).get("enabled") is not True
    ):
        raise RuntimeError("v3 final report is not bound to the no-JPEG/no-ablation route")
    selection = json.loads(args.selection.read_text())
    if (
        selection.get("status") != "frozen_development_selected_checkpoint"
        or selection.get("config_sha256") != config_sha
        or selection.get("sealed_internal_test_accessed") is not False
        or selection.get("protected_foundir_test_accessed") is not False
    ):
        raise RuntimeError("v3 checkpoint selection identity changed")
    checkpoint_path = resolve_workspace_path(ROOT, selection["checkpoint"])
    if file_sha256(checkpoint_path) != selection.get("checkpoint_sha256"):
        raise RuntimeError("v3 selected checkpoint changed")
    qwen_path = resolve_workspace_path(ROOT, selection["selected_development_result"])
    if file_sha256(qwen_path) != next(
        (
            candidate["result_sha256"]
            for candidate in selection.get("candidates", [])
            if candidate.get("result") == selection["selected_development_result"]
        ),
        None,
    ):
        raise RuntimeError("v3 selected development result changed")
    development = load(qwen_path, "development_evaluation_complete", config_sha)
    native = load(args.native, "native_resolution_development_complete", config_sha)
    development_baseline = load(
        args.development_baseline,
        "released_foundir_baseline_comparison_complete",
        config_sha,
    )
    sealed = load(args.sealed, "sealed_internal_test_complete", config_sha)
    sealed_baseline = load(
        args.sealed_baseline,
        "released_foundir_baseline_comparison_complete",
        config_sha,
    )
    if development.get("split") != "validation" or development.get("arm") != "qwen":
        raise RuntimeError("v3 development result is not the deployed Qwen condition")
    if sealed.get("split") != "internal_test" or sealed.get("arm") != "qwen":
        raise RuntimeError("v3 sealed result is not the deployed Qwen condition")
    conditions_path = resolve_workspace_path(ROOT, config["full_conditions"]["csv"])
    conditions_sha = file_sha256(conditions_path)
    expected_records = (
        int(config["evaluation"]["folds"])
        * int(config["evaluation"]["records_per_category_per_fold"])
        * len(config["data"]["expected_categories"])
    )
    for label, result in (("development", development), ("sealed", sealed)):
        if (
            result.get("checkpoint_sha256") != selection["checkpoint_sha256"]
            or result.get("conditions_sha256") != conditions_sha
            or result.get("summary", {}).get("records") != expected_records
        ):
            raise RuntimeError(f"v3 {label} selected-system identity/cardinality changed")
        validate_task_metrics(result, label=f"v3 {label}")
    native_expected_records = (
        int(config["evaluation"]["full_resolution_records_per_category"])
        * len(config["data"]["expected_categories"])
    )
    if (
        native.get("checkpoint_sha256") != selection["checkpoint_sha256"]
        or native.get("records") != native_expected_records
        or not finite_number(native.get("mean_seam_error_ratio"))
    ):
        raise RuntimeError("v3 native-resolution identity/cardinality changed")
    validate_task_metrics(native, label="v3 native-resolution")
    validate_released_baseline(
        development_baseline,
        label="released development baseline",
        split="validation",
        expected_records=expected_records,
        conditions_sha=conditions_sha,
    )
    released_metrics = validate_released_baseline(
        sealed_baseline,
        label="released sealed baseline",
        split="internal_test",
        expected_records=expected_records,
        conditions_sha=conditions_sha,
    )
    consumption_path = resolve_workspace_path(
        ROOT, Path(config["runtime"]["output_dir"]) / "sealed_internal_consumption.json"
    )
    if (
        not consumption_path.is_file()
        or sealed_baseline.get("sealed_internal_consumption_sha256") != file_sha256(consumption_path)
    ):
        raise RuntimeError("released sealed baseline is not bound to sealed-system consumption")

    sealed_metrics = sealed["summary"]["task_metrics"]
    threshold_checks = {
        task: {
            "psnr": float(sealed_metrics[task]["psnr"]) >= threshold["psnr"],
            "lpips": float(sealed_metrics[task]["lpips"]) <= threshold["lpips"],
        }
        for task, threshold in SOW_STAGE2_THRESHOLDS.items()
    }
    threshold_pass = all(all(values.values()) for values in threshold_checks.values())
    released_deltas = {
        task: {
            "psnr": float(sealed_metrics[task]["psnr"]) - float(released_metrics[task]["psnr"]),
            "lpips_improvement": float(released_metrics[task]["lpips"]) - float(sealed_metrics[task]["lpips"]),
        }
        for task in SOW_STAGE2_THRESHOLDS
    }
    payload = {
        "schema_version": 1,
        "status": "internal_protocol_pass" if threshold_pass else "internal_protocol_scientific_failure",
        "automatic_gate_pass": threshold_pass,
        "config_sha256": config_sha,
        "selected_checkpoint": selection["checkpoint"],
        "selected_checkpoint_sha256": selection["checkpoint_sha256"],
        "condition": "frozen_physical_v8_qwen_predictions_only",
        "qwen_condition_ablation_performed": False,
        "pixel_refinement": config["pixel_refinement"],
        "jpeg_included": False,
        "sealed_stage2_caliber_pass": threshold_pass,
        "sealed_threshold_checks": threshold_checks,
        "sealed_task_metrics": sealed_metrics,
        "released_baseline_deltas": released_deltas,
        "native_mean_seam_error_ratio": native["mean_seam_error_ratio"],
        "evidence": {
            "selection": {"path": args.selection.relative_to(ROOT).as_posix(), "sha256": file_sha256(args.selection)},
            "development": {"path": qwen_path.relative_to(ROOT).as_posix(), "sha256": file_sha256(qwen_path)},
            "native": {"path": args.native.relative_to(ROOT).as_posix(), "sha256": file_sha256(args.native)},
            "development_baseline": {"path": args.development_baseline.relative_to(ROOT).as_posix(), "sha256": file_sha256(args.development_baseline)},
            "sealed": {"path": args.sealed.relative_to(ROOT).as_posix(), "sha256": file_sha256(args.sealed)},
            "sealed_baseline": {"path": args.sealed_baseline.relative_to(ROOT).as_posix(), "sha256": file_sha256(args.sealed_baseline)},
        },
        "protected_foundir_test_accessed": False,
        "official_foundir_test_ready": threshold_pass,
    }
    content = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() and args.output.read_bytes() != content:
        raise RuntimeError("refusing to overwrite differing v3 final report")
    if not args.output.exists():
        with args.output.open("xb") as handle:
            handle.write(content)
    print("FOUNDIR_FIVE_TYPE_MAIN_V3_FINAL " + json.dumps(payload, sort_keys=True))
    raise SystemExit(0 if threshold_pass else 2)


if __name__ == "__main__":
    main()
