#!/usr/bin/env python3
"""Compare frozen validation results for the two advisor-requested methods."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lowlight_restoration.multidegradation_data import sha256, write_new_or_verify


RESULTS = {
    "joint": ROOT
    / "checkpoints/qwen_multidegradation_100k_rank64_joint_v1/merger_trainable/validation_result.json",
    "per_type": ROOT
    / "checkpoints/qwen_multidegradation_100k_rank64_per_type_v1/merger_trainable/validation_result.json",
}
OUTPUT = ROOT / "reports/qwen_multidegradation_100k_rank64_comparison_v1.json"


def score(metrics: dict[str, object]) -> tuple[float, ...]:
    triple = metrics["by_composition_size"]["3"]
    return (
        float(triple["combined_label_exact_match_rate"]),
        float(metrics["combined_label_exact_match_rate"]),
        float(triple["degradation_set_exact_match_rate"]),
        float(metrics["type_micro_f1"]),
        -float(metrics["severity_typed_mse_with_missing_penalty"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--joint-result", type=Path, default=RESULTS["joint"])
    parser.add_argument("--per-type-result", type=Path, default=RESULTS["per_type"])
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    result_paths = {
        "joint": args.joint_result.resolve(),
        "per_type": args.per_type_result.resolve(),
    }
    output = args.output.resolve()
    for path in (*result_paths.values(), output):
        path.relative_to(ROOT.resolve())
    rows: dict[str, object] = {}
    loaded: dict[str, dict[str, object]] = {}
    for method, path in result_paths.items():
        result = json.loads(path.read_text())
        metrics = result["metrics"]
        if result.get("split") != "validation" or metrics.get("samples") != 10000:
            raise RuntimeError(f"{method} result is not the frozen 10k validation")
        triple = metrics["by_composition_size"]["3"]
        loaded[method] = result
        rows[method] = {
            "result": path.relative_to(ROOT).as_posix(),
            "result_sha256": sha256(path),
            "automatic_gate_pass": result["automatic_gate_pass"],
            "overall_type_micro_f1": metrics["type_micro_f1"],
            "overall_set_exact": metrics["degradation_set_exact_match_rate"],
            "overall_combined_label_exact": metrics[
                "combined_label_exact_match_rate"
            ],
            "triple_set_exact": triple["degradation_set_exact_match_rate"],
            "triple_combined_label_exact": triple[
                "combined_label_exact_match_rate"
            ],
            "severity_typed_accuracy": metrics["severity_typed_accuracy"],
            "severity_typed_mae": metrics[
                "severity_typed_mae_with_missing_penalty"
            ],
            "severity_typed_mse": metrics[
                "severity_typed_mse_with_missing_penalty"
            ],
        }
    winner = max(loaded, key=lambda method: score(loaded[method]["metrics"]))
    payload = {
        "schema_version": 1,
        "status": "validation_comparison_complete",
        "primary_selection_metric": "triple_combined_label_exact_match_rate",
        "tie_breakers": [
            "overall_combined_label_exact_match_rate",
            "triple_degradation_set_exact_match_rate",
            "type_micro_f1",
            "negative_severity_typed_mse_with_missing_penalty",
        ],
        "methods": rows,
        "selected_method": winner,
        "protected_test_accessed": False,
        "foundir_used": False,
    }
    write_new_or_verify(
        output,
        (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode(),
    )
    print("QWEN_100K_RANK64_COMPARISON_PASS " + json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
