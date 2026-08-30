#!/usr/bin/env python3
"""Freeze the complete internal five-type FLUX evidence without official-test access."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lowlight_restoration.flux_prompt_cache import file_sha256  # noqa: E402
from lowlight_restoration.foundir_flux2_protocol import (  # noqa: E402
    SOW_STAGE2_THRESHOLDS,
    resolve_workspace_path,
    worst_normalized_margin,
)


def load(path: Path, status: str, config_sha: str) -> dict[str, object]:
    payload = json.loads(path.read_text())
    if (
        payload.get("status") != status
        or payload.get("config_sha256") != config_sha
        or payload.get("protected_foundir_test_accessed") is not False
    ):
        raise RuntimeError(f"invalid final evidence: {path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--generic", type=Path, required=True)
    parser.add_argument("--shuffled", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--development-baseline", type=Path, required=True)
    parser.add_argument("--sealed", type=Path, required=True)
    parser.add_argument("--sealed-baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for key in vars(args):
        setattr(args, key, resolve_workspace_path(ROOT, getattr(args, key)))
    config_sha = file_sha256(args.config)
    selection = json.loads(args.selection.read_text())
    if selection.get("status") != "frozen_development_selected_checkpoint" or selection.get("config_sha256") != config_sha:
        raise RuntimeError("final report selection identity changed")
    qwen_path = resolve_workspace_path(ROOT, selection["selected_development_result"])
    qwen = load(qwen_path, "development_evaluation_complete", config_sha)
    generic = load(args.generic, "development_evaluation_complete", config_sha)
    shuffled = load(args.shuffled, "development_evaluation_complete", config_sha)
    oracle = load(args.oracle, "development_evaluation_complete", config_sha)
    native = load(args.native, "native_resolution_development_complete", config_sha)
    development_baseline = load(args.development_baseline, "released_foundir_baseline_comparison_complete", config_sha)
    sealed = load(args.sealed, "sealed_internal_test_complete", config_sha)
    sealed_baseline = load(args.sealed_baseline, "released_foundir_baseline_comparison_complete", config_sha)
    expected_arms = {"qwen": qwen, "generic": generic, "shuffled_wrong": shuffled, "type_oracle": oracle}
    if any(payload.get("split") != "validation" or payload.get("arm") != arm for arm, payload in expected_arms.items()):
        raise RuntimeError("causal development arm identity changed")
    if sealed.get("split") != "internal_test" or sealed.get("arm") != "qwen":
        raise RuntimeError("sealed selected-system identity changed")
    if development_baseline.get("split") != "validation" or sealed_baseline.get("split") != "internal_test":
        raise RuntimeError("released baseline split identity changed")
    arms = {name: payload["summary"]["task_metrics"] for name, payload in {
        "qwen": qwen,
        "generic": generic,
        "shuffled_wrong": shuffled,
        "type_oracle": oracle,
    }.items()}
    arm_margins = {name: worst_normalized_margin(metrics) for name, metrics in arms.items()}
    causal_deltas = {}
    causal_pass = True
    for baseline_name in ("generic", "shuffled_wrong"):
        task_deltas = {
            task: {
                "psnr": float(arms["qwen"][task]["psnr"]) - float(arms[baseline_name][task]["psnr"]),
                "lpips_improvement": float(arms[baseline_name][task]["lpips"]) - float(arms["qwen"][task]["lpips"]),
            }
            for task in SOW_STAGE2_THRESHOLDS
        }
        causal_deltas[baseline_name] = task_deltas
        causal_pass = causal_pass and all(
            delta["psnr"] >= -0.02 and delta["lpips_improvement"] >= -0.001
            for delta in task_deltas.values()
        )
        causal_pass = causal_pass and sum(delta["psnr"] for delta in task_deltas.values()) > 0
        causal_pass = causal_pass and sum(delta["lpips_improvement"] for delta in task_deltas.values()) > 0
    sealed_metrics = sealed["summary"]["task_metrics"]
    threshold_pass = all(
        float(sealed_metrics[task]["psnr"]) >= threshold["psnr"]
        and float(sealed_metrics[task]["lpips"]) <= threshold["lpips"]
        for task, threshold in SOW_STAGE2_THRESHOLDS.items()
    )
    automatic_pass = causal_pass and threshold_pass
    payload = {
        "schema_version": 1,
        "status": "internal_protocol_pass" if automatic_pass else "internal_protocol_scientific_failure",
        "automatic_gate_pass": automatic_pass,
        "config_sha256": config_sha,
        "selected_checkpoint": selection["checkpoint"],
        "selected_checkpoint_sha256": selection["checkpoint_sha256"],
        "development_condition_margins": arm_margins,
        "qwen_causal_task_deltas": causal_deltas,
        "qwen_causal_pass": causal_pass,
        "sealed_stage2_caliber_pass": threshold_pass,
        "sealed_task_metrics": sealed_metrics,
        "native_mean_seam_error_ratio": native["mean_seam_error_ratio"],
        "evidence": {
            "selection": {"path": args.selection.relative_to(ROOT).as_posix(), "sha256": file_sha256(args.selection)},
            "qwen": {"path": qwen_path.relative_to(ROOT).as_posix(), "sha256": file_sha256(qwen_path)},
            "generic": {"path": args.generic.relative_to(ROOT).as_posix(), "sha256": file_sha256(args.generic)},
            "shuffled": {"path": args.shuffled.relative_to(ROOT).as_posix(), "sha256": file_sha256(args.shuffled)},
            "oracle": {"path": args.oracle.relative_to(ROOT).as_posix(), "sha256": file_sha256(args.oracle)},
            "native": {"path": args.native.relative_to(ROOT).as_posix(), "sha256": file_sha256(args.native)},
            "development_baseline": {"path": args.development_baseline.relative_to(ROOT).as_posix(), "sha256": file_sha256(args.development_baseline)},
            "sealed": {"path": args.sealed.relative_to(ROOT).as_posix(), "sha256": file_sha256(args.sealed)},
            "sealed_baseline": {"path": args.sealed_baseline.relative_to(ROOT).as_posix(), "sha256": file_sha256(args.sealed_baseline)},
        },
        "protected_foundir_test_accessed": False,
        "official_foundir_test_ready": automatic_pass,
    }
    content = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() and args.output.read_bytes() != content:
        raise RuntimeError("refusing to overwrite differing final internal report")
    if not args.output.exists():
        with args.output.open("xb") as handle:
            handle.write(content)
    print("FOUNDIR_FIVE_TYPE_FINAL_INTERNAL " + json.dumps(payload, sort_keys=True), flush=True)
    raise SystemExit(0 if automatic_pass else 2)


if __name__ == "__main__":
    main()
