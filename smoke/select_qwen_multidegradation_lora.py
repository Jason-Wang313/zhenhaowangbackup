#!/usr/bin/env python3
"""Select one merger ablation using validation only, before fixed test use."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lowlight_restoration.multidegradation_data import sha256, write_new_or_verify
from smoke.qwen_multidegradation_lora_preflight import load_config, workspace_path


def score(result: dict[str, object]) -> tuple[float, ...]:
    metrics = result["metrics"]
    assert isinstance(metrics, dict)
    composition = metrics.get("by_composition_size", {})
    assert isinstance(composition, dict)
    triple = composition.get("3", {})
    assert isinstance(triple, dict)
    if result.get("validation_checkpoint_stage") is not None:
        return (
            float(triple["degradation_set_exact_match_rate"]),
            float(metrics["combined_label_exact_match_rate"]),
            float(metrics["severity_typed_accuracy"]),
            float(metrics["degradation_set_exact_match_rate"]),
            float(metrics["type_macro_f1"]),
            -float(metrics["severity_typed_mae_with_missing_penalty"]),
        )
    return (
        float(metrics["combined_label_exact_match_rate"]),
        float(metrics["degradation_set_exact_match_rate"]),
        float(metrics["type_macro_f1"]),
        -float(metrics["severity_typed_mae_with_missing_penalty"]),
        int(result["variant"] == "merger_frozen"),
    )


def validation_candidates(
    config: dict[str, object], root: Path
) -> list[tuple[str, str, Path]]:
    data = config["data"]
    model = config["model"]
    assert isinstance(data, dict) and isinstance(model, dict)
    stages = data.get("validation_candidate_stages")
    if not isinstance(stages, list):
        return [
            (str(variant), str(variant), root / str(variant) / "validation_result.json")
            for variant in model["variants"]
        ]
    final_stage = str(data["curriculum"][-1]["name"])
    return [
        (
            str(stage),
            "merger_trainable",
            root
            / "merger_trainable"
            / (
                "validation_result.json"
                if str(stage) == final_stage
                else f"validation_result_{stage}.json"
            ),
        )
        for stage in stages
    ]


def select(config_path: Path) -> dict[str, object]:
    config, config_sha = load_config(config_path)
    runtime = config["runtime"]
    model = config["model"]
    assert isinstance(runtime, dict) and isinstance(model, dict)
    root = workspace_path(runtime["output_root"])
    results: dict[str, tuple[dict[str, object], Path]] = {}
    for candidate, variant, path in validation_candidates(config, root):
        if not path.is_file():
            raise FileNotFoundError(f"missing validation result: {path}")
        result = json.loads(path.read_text())
        if result.get("config_sha256") != config_sha or result.get("variant") != variant:
            raise RuntimeError(f"validation result contract mismatch: {candidate}")
        if result.get("validation_checkpoint_stage") not in (None, candidate):
            raise RuntimeError(f"validation checkpoint mismatch: {candidate}")
        results[candidate] = (result, path)
    eligible = [
        result
        for result, _ in results.values()
        if result.get("automatic_gate_pass") is True
    ]
    selected = max(eligible, key=score) if eligible else None
    continuation_selection = isinstance(
        config["data"].get("validation_candidate_stages"), list
    )
    serialized_results = {}
    for name, (result, path) in results.items():
        item = {
            "automatic_gate_pass": result["automatic_gate_pass"],
            "validation_result_sha256": sha256(path),
            "score": list(score(result)),
        }
        if continuation_selection:
            item["validation_result"] = path.relative_to(ROOT).as_posix()
        serialized_results[name] = item
    payload = {
        "schema_version": 1,
        "status": "selected" if selected else "no_variant_passed",
        "config_sha256": config_sha,
        "selection_data": "synthetic_validation_only",
        "fixed_test_accessed": False,
        "selected_variant": selected["variant"] if selected else None,
        "selected_adapter": selected["adapter"] if selected else None,
        "selected_adapter_sha256": selected["adapter_sha256"] if selected else None,
    }
    payload[
        "candidate_results"
        if continuation_selection
        else "variant_results"
    ] = serialized_results
    if continuation_selection:
        payload["selected_stage"] = (
            selected.get("validation_checkpoint_stage") if selected else None
        )
    output = root / "checkpoint_selection.json"
    write_new_or_verify(output, (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode())
    print("QWEN_MULTIDEGRADATION_SELECTION " + json.dumps(payload, sort_keys=True), flush=True)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/qwen_multidegradation_lora_v1.json",
    )
    args = parser.parse_args()
    return 0 if select(args.config)["selected_variant"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
