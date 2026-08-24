#!/usr/bin/env python3
"""Select one merger ablation using validation only, before fixed test use."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lowlight_restoration.multidegradation_data import sha256, write_new_or_verify
from smoke.qwen_multidegradation_lora_preflight import load_config, workspace_path


def score(result: dict[str, object]) -> tuple[float, float, float, float, int]:
    metrics = result["metrics"]
    assert isinstance(metrics, dict)
    return (
        float(metrics["combined_label_exact_match_rate"]),
        float(metrics["degradation_set_exact_match_rate"]),
        float(metrics["type_macro_f1"]),
        -float(metrics["severity_typed_mae_with_missing_penalty"]),
        int(result["variant"] == "merger_frozen"),
    )


def select(config_path: Path) -> dict[str, object]:
    config, config_sha = load_config(config_path)
    runtime = config["runtime"]
    model = config["model"]
    assert isinstance(runtime, dict) and isinstance(model, dict)
    root = workspace_path(runtime["output_root"])
    results: dict[str, dict[str, object]] = {}
    for variant in model["variants"]:
        path = root / variant / "validation_result.json"
        if not path.is_file():
            raise FileNotFoundError(f"missing validation result: {path}")
        result = json.loads(path.read_text())
        if result.get("config_sha256") != config_sha or result.get("variant") != variant:
            raise RuntimeError(f"validation result contract mismatch: {variant}")
        results[variant] = result
    eligible = [result for result in results.values() if result.get("automatic_gate_pass") is True]
    selected = max(eligible, key=score) if eligible else None
    payload = {
        "schema_version": 1,
        "status": "selected" if selected else "no_variant_passed",
        "config_sha256": config_sha,
        "selection_data": "synthetic_validation_only",
        "fixed_test_accessed": False,
        "variant_results": {
            name: {
                "automatic_gate_pass": result["automatic_gate_pass"],
                "validation_result_sha256": sha256(root / name / "validation_result.json"),
                "score": list(score(result)),
            }
            for name, result in results.items()
        },
        "selected_variant": selected["variant"] if selected else None,
        "selected_adapter": selected["adapter"] if selected else None,
        "selected_adapter_sha256": selected["adapter_sha256"] if selected else None,
    }
    output = root / "checkpoint_selection.json"
    write_new_or_verify(output, (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode())
    print("QWEN_MULTIDEGRADATION_SELECTION " + json.dumps(payload, sort_keys=True), flush=True)
    return payload


def main() -> int:
    config = ROOT / "configs/qwen_multidegradation_lora_v1.json"
    return 0 if select(config)["selected_variant"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
