#!/usr/bin/env python3
"""Authorize the one-time v2 synthetic test from frozen validation evidence."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lowlight_restoration.multidegradation_data import sha256
from smoke.compare_qwen_multidegradation_100k_rank64 import score
from smoke.qwen_multidegradation_lora_preflight import load_config, workspace_path

COMPARISON = ROOT / "reports/qwen_multidegradation_100k_rank64_comparison_v2.json"
CONFIG = ROOT / "configs/qwen_multidegradation_100k_rank64_per_type_v2.json"


def verify(root: Path = ROOT) -> dict[str, object]:
    comparison_path = root / COMPARISON.relative_to(ROOT)
    config_path = root / CONFIG.relative_to(ROOT)
    if not comparison_path.is_file():
        raise RuntimeError("the frozen v2 validation comparison is missing")
    comparison = json.loads(comparison_path.read_text())
    if (
        comparison.get("status") != "validation_comparison_complete"
        or comparison.get("selected_method") != "per_type"
        or comparison.get("protected_test_accessed") is not False
        or comparison.get("foundir_used") is not False
    ):
        raise RuntimeError("the v2 comparison does not authorize per-type testing")

    loaded: dict[str, dict[str, object]] = {}
    for method in ("joint", "per_type"):
        row = comparison.get("methods", {}).get(method, {})
        result_value = row.get("result")
        if not isinstance(result_value, str):
            raise RuntimeError(f"missing {method} validation result path")
        result_path = (root / result_value).resolve()
        result_path.relative_to(root.resolve())
        if not result_path.is_file() or sha256(result_path) != row.get("result_sha256"):
            raise RuntimeError(f"{method} validation result changed after comparison")
        result = json.loads(result_path.read_text())
        if result.get("split") != "validation" or result.get("metrics", {}).get("samples") != 10000:
            raise RuntimeError(f"{method} result is not the frozen 10k validation")
        loaded[method] = result
    if max(loaded, key=lambda method: score(loaded[method]["metrics"])) != "per_type":
        raise RuntimeError("recomputed v2 winner is not per_type")
    if loaded["per_type"].get("automatic_gate_pass") is not True:
        raise RuntimeError("the selected per-type validation did not pass")

    config, config_sha = load_config(config_path)
    output_root = workspace_path(config["runtime"]["output_root"])
    selection_path = output_root / "checkpoint_selection.json"
    if not selection_path.is_file():
        raise RuntimeError("the per-type checkpoint selection is missing")
    selection = json.loads(selection_path.read_text())
    if (
        selection.get("status") != "selected"
        or selection.get("config_sha256") != config_sha
        or selection.get("selection_data") != "synthetic_validation_only"
        or selection.get("fixed_test_accessed") is not False
        or selection.get("selected_variant") != "merger_trainable"
    ):
        raise RuntimeError("the per-type checkpoint selection is not frozen correctly")
    adapter_value = selection.get("selected_adapter")
    if not isinstance(adapter_value, str):
        raise RuntimeError("the selected adapter path is missing")
    adapter = (root / adapter_value).resolve()
    adapter.relative_to(root.resolve())
    if not adapter.is_file() or sha256(adapter) != selection.get("selected_adapter_sha256"):
        raise RuntimeError("the selected adapter changed after validation")

    output = output_root / "merger_trainable"
    for path in (output / "test_predictions.json", output / "fixed_test_result.json"):
        if path.exists():
            raise RuntimeError(f"fixed synthetic test was already accessed: {path}")
    return {
        "status": "pass",
        "selected_method": "per_type",
        "config_sha256": config_sha,
        "comparison_sha256": sha256(comparison_path),
        "selection_sha256": sha256(selection_path),
        "selected_adapter": adapter_value,
        "selected_adapter_sha256": sha256(adapter),
        "fixed_test_accessed": False,
        "foundir_used": False,
    }


def main() -> None:
    print(
        "QWEN_100K_V2_FIXED_TEST_AUTHORIZATION_PASS "
        + json.dumps(verify(), sort_keys=True),
        flush=True,
    )


if __name__ == "__main__":
    main()
