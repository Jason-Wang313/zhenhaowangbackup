"""Fail-closed authorization for the fair Qwen-conditioned FLUX.2 run."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Mapping

from .authorization import sha256
from .config import ROOT, TrainConfig, load_train_config

CONDITIONED_STATUS = "approved_after_expanded_qwen_ridge_pass"
CONDITIONED_CONFIG_SHA256 = (
    "ae50f3b289f6f3733bf7904f9659c93147896b7280771e0a2a66d3455fd3d41b"
)
BASELINE_CONFIG_SHA256 = (
    "c71235efe88e8f3bc98ca863350af372a78aa35f1a2cc065a1867c5bb6bfde18"
)
QWEN_CONFIG_SHA256 = (
    "b04bf893e073cf50423a384456398b12ce92625a979453c8f8c61e210f8bd33c"
)
QWEN_METRICS_SHA256 = (
    "18285e16fad062d8653e9468159f6679531a7f97ea1d5b794bc12fe4994cc29a"
)
QWEN_CHECKPOINT_SHA256 = (
    "27f58645d992c97f8b3e44af82a2a9ae11d93d0fa4bc5090b3d1bde39271b368"
)
QWEN_MODEL_REVISION = "cc594898137f460bfe9f0759e9844b3ce807cfb5"
SHA256_RE = re.compile(r"[0-9a-f]{64}")


def workspace_path(value: object, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"{label} must be a non-empty workspace path")
    path = (ROOT / value).resolve()
    try:
        path.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise RuntimeError(f"{label} escapes workspace: {value}") from exc
    return path


def _require_exact(
    mapping: Mapping[str, object], key: str, expected: object, *, label: str
) -> None:
    if mapping.get(key) != expected:
        raise RuntimeError(f"{label}.{key} must equal {expected!r}")


def _load_bound_json(
    path: Path, expected_sha256: str, *, label: str, require_artifact: bool
) -> dict[str, object] | None:
    if SHA256_RE.fullmatch(expected_sha256) is None:
        raise RuntimeError(f"{label} expected hash is invalid")
    if not path.is_file():
        if require_artifact:
            raise RuntimeError(f"missing {label}: {path}")
        return None
    if sha256(path) != expected_sha256:
        raise RuntimeError(f"{label} SHA-256 mismatch: {path}")
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot parse {label}: {path}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"{label} must be a JSON object")
    return payload


def _verify_bound_file(
    path: Path, expected_sha256: str, *, label: str, require_artifact: bool
) -> None:
    if SHA256_RE.fullmatch(expected_sha256) is None:
        raise RuntimeError(f"{label} expected hash is invalid")
    if not path.is_file():
        if require_artifact:
            raise RuntimeError(f"missing {label}: {path}")
        return
    if sha256(path) != expected_sha256:
        raise RuntimeError(f"{label} SHA-256 mismatch: {path}")


def _verify_gate_artifact(
    gate: Mapping[str, object],
    path_key: str,
    hash_key: str,
    *,
    require_artifact: bool,
) -> None:
    path = workspace_path(gate.get(path_key), label=f"baseline_gate.{path_key}")
    expected = gate.get(hash_key)
    if not isinstance(expected, str):
        raise RuntimeError(f"baseline_gate.{hash_key} must be a SHA-256 string")
    _verify_bound_file(
        path,
        expected,
        label=f"baseline gate artifact {path_key}",
        require_artifact=require_artifact,
    )


def _validate_fair_sections(config: TrainConfig, baseline: TrainConfig) -> None:
    for section in ("model", "data", "lora", "training", "inference"):
        if config.section(section) != baseline.section(section):
            raise RuntimeError(
                f"conditioned {section} differs from the frozen baseline; comparison is unfair"
            )
    runtime = config.section("runtime")
    baseline_runtime = baseline.section("runtime")
    for key in ("device", "offline", "minimum_vram_gib"):
        if runtime.get(key) != baseline_runtime.get(key):
            raise RuntimeError(f"conditioned runtime.{key} differs from baseline")
    _require_exact(
        runtime,
        "output_dir",
        "checkpoints/lowlight_qwen_conditioned_full",
        label="runtime",
    )


def _validate_condition_section(condition: Mapping[str, object]) -> None:
    expected = {
        "mode": "frozen_qwen_ridge_predicted_level",
        "degradation_type": "low_light",
        "adapter_type": "append_one_learned_context_token",
        "context_dim": 7680,
        "levels": 5,
        "init_std": 0.02,
        "expected_trainable_parameters": 38_400,
        "expected_total_trainable_parameters": 8_148_480,
        "baseline_config_sha256": BASELINE_CONFIG_SHA256,
        "qwen_expanded_config_sha256": QWEN_CONFIG_SHA256,
        "qwen_metrics_sha256": QWEN_METRICS_SHA256,
        "qwen_checkpoint_sha256": QWEN_CHECKPOINT_SHA256,
        "qwen_model_revision": QWEN_MODEL_REVISION,
        "feature_dim": 7168,
        "feature_cache_schema_version": 2,
        "feature_extraction_batch_size": 20,
        "expected_prediction_records": 10_000,
        "expected_prediction_splits": {
            "train": 8_000,
            "validation": 1_000,
            "test": 1_000,
        },
    }
    for key, value in expected.items():
        _require_exact(condition, key, value, label="condition")


def validate_conditioned_authorization(
    config: TrainConfig, *, require_artifacts: bool = True
) -> str:
    """Validate the immutable fair-comparison contract without creating outputs."""

    if config.raw.get("status") != CONDITIONED_STATUS:
        raise RuntimeError("conditioned run status is not authorized")
    if config.sha256 != CONDITIONED_CONFIG_SHA256:
        raise RuntimeError("the approved conditioned config hash has changed")
    _require_exact(
        config.raw,
        "run_name",
        "flux2_lowlight_qwen_conditioned_full",
        label="config",
    )
    condition = config.section("condition")
    _validate_condition_section(condition)

    baseline_path = workspace_path(
        condition.get("baseline_config"), label="condition.baseline_config"
    )
    _verify_bound_file(
        baseline_path,
        BASELINE_CONFIG_SHA256,
        label="baseline config",
        require_artifact=True,
    )
    baseline = load_train_config(baseline_path)
    _validate_fair_sections(config, baseline)

    baseline_gate_path = workspace_path(
        condition.get("baseline_gate"), label="condition.baseline_gate"
    )
    baseline_gate_hash = condition.get("baseline_gate_sha256")
    if not isinstance(baseline_gate_hash, str):
        raise RuntimeError("condition.baseline_gate_sha256 is invalid")
    baseline_gate = _load_bound_json(
        baseline_gate_path,
        baseline_gate_hash,
        label="baseline gate",
        require_artifact=True,
    )
    assert baseline_gate is not None
    gate_expected = {
        "schema_version": 1,
        "gate": "flux2_lowlight_baseline_full_internal",
        "status": "passed",
        "approved_next_scope": "prepare_qwen_vl_lowlight_baseline_comparison",
        "config_sha256": BASELINE_CONFIG_SHA256,
        "validation_automatic_sanity_pass": True,
        "test_automatic_sanity_pass": True,
        "visual_identity_gate_pass": True,
    }
    for key, value in gate_expected.items():
        _require_exact(baseline_gate, key, value, label="baseline_gate")
    for path_key, hash_key in (
        ("checkpoint", "checkpoint_sha256"),
        ("training_summary", "training_summary_sha256"),
        ("validation_metrics", "validation_metrics_sha256"),
        ("test_metrics", "test_metrics_sha256"),
    ):
        _verify_gate_artifact(
            baseline_gate,
            path_key,
            hash_key,
            require_artifact=require_artifacts,
        )

    for path_key, hash_key, label in (
        ("baseline_prompt_context", "baseline_prompt_context_sha256", "baseline prompt context"),
        ("baseline_prompt_metadata", "baseline_prompt_metadata_sha256", "baseline prompt metadata"),
        ("qwen_base_config", "qwen_base_config_sha256", "Qwen base config"),
        ("qwen_expanded_config", "qwen_expanded_config_sha256", "Qwen expanded config"),
        ("feature_extractor", "feature_extractor_sha256", "Qwen feature extractor"),
    ):
        expected = condition.get(hash_key)
        if not isinstance(expected, str):
            raise RuntimeError(f"condition.{hash_key} is invalid")
        _verify_bound_file(
            workspace_path(condition.get(path_key), label=f"condition.{path_key}"),
            expected,
            label=label,
            require_artifact=True,
        )

    qwen_metrics_path = workspace_path(
        condition.get("qwen_metrics"), label="condition.qwen_metrics"
    )
    qwen_metrics = _load_bound_json(
        qwen_metrics_path,
        QWEN_METRICS_SHA256,
        label="Qwen expanded metrics",
        require_artifact=require_artifacts,
    )
    qwen_checkpoint = workspace_path(
        condition.get("qwen_checkpoint"), label="condition.qwen_checkpoint"
    )
    _verify_bound_file(
        qwen_checkpoint,
        QWEN_CHECKPOINT_SHA256,
        label="Qwen expanded checkpoint",
        require_artifact=require_artifacts,
    )
    if qwen_metrics is not None:
        expected_metrics = {
            "schema_version": 5,
            "status": "passed",
            "scope": "expanded_frozen_qwen_ridge_lowlight_unexposed_validation_only",
            "config_sha256": QWEN_CONFIG_SHA256,
            "checkpoint_sha256": QWEN_CHECKPOINT_SHA256,
            "model_revision": QWEN_MODEL_REVISION,
            "selected_alpha": 3000.0,
            "test_split_used": False,
            "official_foundir_used": False,
        }
        for key, value in expected_metrics.items():
            _require_exact(qwen_metrics, key, value, label="qwen_metrics")
        gate = qwen_metrics.get("gate")
        if not isinstance(gate, dict) or gate.get("automatic_gate_pass") is not True:
            raise RuntimeError("Qwen expanded validation gate did not pass")
        recalls = gate.get("realized_per_level_recall")
        if (
            not isinstance(recalls, dict)
            or set(recalls) != {"1", "2", "3", "4", "5"}
            or min(float(value) for value in recalls.values()) < 0.4
        ):
            raise RuntimeError("Qwen expanded validation recall evidence is invalid")

    comparison = config.section("comparison")
    expected_comparison = {
        "primary_scope": "test_l2_l5",
        "require_psnr_improvement": True,
        "require_ssim_improvement": True,
        "require_lpips_improvement": True,
        "maximum_l1_ssim_regression": 0.01,
    }
    for key, value in expected_comparison.items():
        _require_exact(comparison, key, value, label="comparison")
    for path_key, hash_key in (
        ("baseline_validation_metrics", "baseline_validation_metrics_sha256"),
        ("baseline_test_metrics", "baseline_test_metrics_sha256"),
    ):
        expected = comparison.get(hash_key)
        if not isinstance(expected, str):
            raise RuntimeError(f"comparison.{hash_key} is invalid")
        _verify_bound_file(
            workspace_path(comparison.get(path_key), label=f"comparison.{path_key}"),
            expected,
            label=f"comparison artifact {path_key}",
            require_artifact=require_artifacts,
        )

    implementation = config.section("implementation")
    expected_implementation_paths = {
        "lowlight_restoration/condition_adapter.py",
        "lowlight_restoration/conditioned_checkpoint.py",
        "lowlight_restoration/qwen_predictions.py",
        "lowlight_restoration/flow.py",
        "lowlight_restoration/lora.py",
        "lowlight_restoration/data.py",
        "lowlight_restoration/config.py",
        "train_flux2_lowlight.py",
        "evaluate_flux2_lowlight_split.py",
        "smoke/evaluate_lowlight_overfit.py",
        "smoke/export_qwen_lowlight_condition_predictions.py",
        "smoke/train_flux2_lowlight_qwen_conditioned.py",
        "smoke/evaluate_flux2_lowlight_qwen_conditioned.py",
        "smoke/compare_lowlight_baseline_qwen.py",
        "smoke/lowlight_qwen_conditioned_preflight.py",
        "smoke/run_lowlight_qwen_conditioned_full.sh",
    }
    if set(implementation) != expected_implementation_paths:
        raise RuntimeError("conditioned implementation file set changed")
    for relative, expected in implementation.items():
        if not isinstance(expected, str):
            raise RuntimeError(f"implementation hash is invalid: {relative}")
        _verify_bound_file(
            workspace_path(relative, label=f"implementation.{relative}"),
            expected,
            label=f"conditioned implementation {relative}",
            require_artifact=True,
        )
    return "full_qwen_conditioned"
