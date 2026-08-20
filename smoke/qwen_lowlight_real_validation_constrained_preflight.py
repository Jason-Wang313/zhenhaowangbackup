#!/usr/bin/env python3
"""Preflight the post-failure canonical-scoring Qwen diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lowlight_restoration.qwen_constrained import CANONICAL_LEVELS  # noqa: E402
from lowlight_restoration.qwen_real_validation import (  # noqa: E402
    PRIMARY_SUBSET,
    STRESS_SUBSET,
    load_real_validation_config,
    validate_real_validation_contract,
)
from smoke.qwen_lowlight_real_validation_preflight import (  # noqa: E402
    run_preflight as run_parent_preflight,
)
from smoke.qwen_lowlight_supervised_preflight import (  # noqa: E402
    sha256,
    workspace_path,
)
from smoke.train_qwen_lowlight_supervised import (  # noqa: E402
    atomic_write_new_json,
    read_json,
)


APPROVED_STATUS = "approved_for_post_failure_constrained_scoring_diagnostic_only"


def load_diagnostic_config(path: Path) -> tuple[dict[str, object], str]:
    raw = path.read_bytes()
    payload = json.loads(raw)
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("bad constrained-diagnostic config schema")
    if payload.get("status") != APPROVED_STATUS:
        raise ValueError("constrained diagnostic is not approved")
    return payload, hashlib.sha256(raw).hexdigest()


def validate_diagnostic_contract(config: Mapping[str, object]) -> None:
    required = (
        "parent_failure",
        "model",
        "data",
        "scoring",
        "diagnostic_evaluation",
        "runtime",
        "telemetry",
    )
    sections = {name: config.get(name) for name in required}
    if any(not isinstance(value, dict) for value in sections.values()):
        raise ValueError("constrained diagnostic is missing a required section")
    parent = sections["parent_failure"]
    model = sections["model"]
    data = sections["data"]
    scoring = sections["scoring"]
    evaluation = sections["diagnostic_evaluation"]
    runtime = sections["runtime"]
    telemetry = sections["telemetry"]
    assert all(
        isinstance(section, dict)
        for section in (parent, model, data, scoring, evaluation, runtime, telemetry)
    )

    if (
        config.get("scope")
        != "post_failure_frozen_step600_canonical_scoring_diagnostic"
        or config.get("interpretation")
        != "not_an_independent_validation_gate_and_not_a_retroactive_pass"
        or config.get("official_foundir_used") is not False
        or config.get("original_failed_predictions_consulted_during_method_selection")
        is not True
        or config.get("labels_changed_after_failure") is not False
        or config.get("thresholds_changed_after_failure") is not False
        or config.get("model_weights_changed_after_failure") is not False
    ):
        raise ValueError("post-failure interpretation or frozen scope changed")
    if (
        parent.get("expected_status") != "failed"
        or parent.get("expected_automatic_gate_pass") is not False
        or parent.get("expected_records") != 233
        or parent.get("expected_batch_files") != 47
    ):
        raise ValueError("parent failure contract changed")
    if (
        model.get("selected_checkpoint_step") != 600
        or model.get("selected_visual_sha256")
        != "b0e65a8597077973a2a103dc3369e0cb1ea703de95e2e4841d359c3d34eff77f"
        or model.get("all_parameters_frozen") is not True
        or model.get("dtype") != "bfloat16"
        or model.get("attention_implementation") != "sdpa"
    ):
        raise ValueError("constrained diagnostic model contract changed")
    if (
        data.get("reuse_parent_manifest_and_proxy_labels_exactly") is not True
        or data.get("manifest_sha256")
        != "52f6fc9c5cdc0347559b37a30f7f5a607ac9a3d5576c6b9e8dc22424c478106b"
        or data.get("metadata_sha256")
        != "3b9be6683208580f51ddb93779436270b28e97ab4b27a2df720bfc6238cdd1fb"
        or data.get("annotation_kind")
        != "reference_based_proxy_not_human_ground_truth"
        or data.get("model_predictions_consulted_during_original_record_selection")
        is not False
        or data.get("records") != 233
        or data.get("subsets") != [PRIMARY_SUBSET, STRESS_SUBSET]
    ):
        raise ValueError("constrained diagnostic data contract changed")
    if (
        scoring.get("method")
        != "mean_teacher_forced_target_token_negative_log_likelihood"
        or scoring.get("candidate_project_levels") != list(CANONICAL_LEVELS)
        or scoring.get("candidate_targets")
        != "the_five_existing_canonical_JSON_strings"
        or scoring.get("causal_shift") is not True
        or scoring.get("score_only_unmasked_assistant_target_tokens") is not True
        or scoring.get("length_normalization")
        != "mean_over_scored_target_tokens"
        or scoring.get("selection") != "minimum_mean_negative_log_likelihood"
        or scoring.get("tie_break") != "lower_project_level"
        or scoring.get("batch_size") != 5
        or scoring.get("free_form_generation_used") is not False
        or scoring.get("prompt_changed") is not False
        or scoring.get("guaranteed_canonical_structured_output") is not True
    ):
        raise ValueError("canonical scoring contract changed")
    if (
        evaluation.get("reuse_original_predeclared_thresholds_exactly") is not True
        or evaluation.get("both_parent_subsets_must_clear_original_thresholds")
        is not True
        or evaluation.get("passing_does_not_erase_or_replace_parent_failure")
        is not True
        or evaluation.get(
            "passing_requires_fresh_independent_real_holdout_before_diffusion_claims"
        )
        is not True
    ):
        raise ValueError("diagnostic interpretation contract changed")
    if (
        runtime.get("seed") != 2026082001
        or runtime.get("offline") is not True
        or runtime.get("device") != "cuda"
        or runtime.get("required_visible_device_count") != 1
        or runtime.get("minimum_gpu_memory_gib") != 70
        or runtime.get("deterministic_algorithms") is not True
        or runtime.get("sdpa_backend") != "math"
        or runtime.get("required_cublas_workspace_config") != ":4096:8"
        or runtime.get("output_dir")
        != "checkpoints/qwen_lowlight_real_validation_constrained_step600"
        or runtime.get("required_python") != "3.12.13"
        or runtime.get("required_pytorch") != "2.7.0+cu128"
        or runtime.get("required_transformers") != "4.56.1"
    ):
        raise ValueError("constrained diagnostic runtime contract changed")
    if (
        telemetry.get("csv")
        != "logs/qwen_lowlight_real_validation_constrained_step600_gpu.csv"
        or telemetry.get("sample_interval_seconds") != 2
        or telemetry.get("monitor_all_selected_gpus") is not True
    ):
        raise ValueError("constrained diagnostic telemetry contract changed")


def verify_parent_failure(
    config: Mapping[str, object], parent_config: Mapping[str, object]
) -> dict[str, object]:
    parent = config["parent_failure"]
    model = config["model"]
    data = config["data"]
    assert isinstance(parent, dict) and isinstance(model, dict) and isinstance(data, dict)
    result_path = workspace_path(parent["result"])
    if not result_path.is_file() or sha256(result_path) != parent.get("result_sha256"):
        raise RuntimeError("original failed real-validation result changed")
    result = read_json(result_path)
    checks = result.get("gate_checks")
    if (
        result.get("status") != parent.get("expected_status")
        or result.get("automatic_gate_pass")
        is not parent.get("expected_automatic_gate_pass")
        or result.get("config_sha256") != parent.get("config_sha256")
        or result.get("git_revision") != parent.get("git_revision")
        or result.get("selected_visual_sha256")
        != model.get("selected_visual_sha256")
        or result.get("manifest_sha256") != data.get("manifest_sha256")
        or result.get("metadata_sha256") != data.get("metadata_sha256")
        or not isinstance(checks, dict)
        or not checks
        or all(checks.values())
    ):
        raise RuntimeError("original failed real-validation result is inconsistent")
    artifacts = result.get("artifact_sha256")
    if not isinstance(artifacts, dict) or len(artifacts) != 8:
        raise RuntimeError("original failed result artifact inventory changed")
    for relative, expected in artifacts.items():
        path = workspace_path(relative)
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"original failed artifact changed: {path}")

    parent_runtime = parent_config["runtime"]
    diagnostic_runtime = config["runtime"]
    assert isinstance(parent_runtime, dict) and isinstance(diagnostic_runtime, dict)
    for key in (
        "offline",
        "device",
        "required_visible_device_count",
        "minimum_gpu_memory_gib",
        "deterministic_algorithms",
        "sdpa_backend",
        "required_cublas_workspace_config",
        "required_python",
        "required_pytorch",
        "required_transformers",
    ):
        if diagnostic_runtime.get(key) != parent_runtime.get(key):
            raise RuntimeError(f"diagnostic runtime differs from parent: {key}")

    batch_root = result_path.parent / "batches"
    batch_files = sorted(batch_root.glob("*/*.json"))
    output_count = 0
    for path in batch_files:
        payload = read_json(path)
        outputs = payload.get("outputs")
        pair_ids = payload.get("pair_ids")
        if (
            payload.get("config_sha256") != parent.get("config_sha256")
            or payload.get("selected_visual_sha256")
            != model.get("selected_visual_sha256")
            or not isinstance(outputs, list)
            or not isinstance(pair_ids, list)
            or len(outputs) != len(pair_ids)
        ):
            raise RuntimeError(f"original failed batch is inconsistent: {path}")
        output_count += len(outputs)
    if (
        len(batch_files) != parent.get("expected_batch_files")
        or output_count != parent.get("expected_records")
    ):
        raise RuntimeError("original failed batch cardinality changed")
    return {
        "result": str(parent["result"]),
        "result_sha256": str(parent["result_sha256"]),
        "status": "failed",
        "automatic_gate_pass": False,
        "failed_checks": sorted(key for key, value in checks.items() if not value),
        "verified_evaluation_artifacts": len(artifacts),
        "verified_batch_files": len(batch_files),
        "verified_predictions": output_count,
    }


def run_preflight(
    config_path: Path,
    *,
    contract_only: bool,
    skip_large_model_hashes: bool,
    require_gpu: bool,
    write_artifact: bool,
) -> dict[str, object]:
    config, config_sha = load_diagnostic_config(config_path)
    validate_diagnostic_contract(config)
    parent = config["parent_failure"]
    assert isinstance(parent, dict)
    parent_config_path = workspace_path(parent["config"])
    parent_config, parent_sha, _ = load_real_validation_config(parent_config_path)
    validate_real_validation_contract(parent_config)
    if parent_sha != parent.get("config_sha256"):
        raise RuntimeError("original real-validation config changed")
    parent_failure = verify_parent_failure(config, parent_config)
    base = run_parent_preflight(
        parent_config_path,
        contract_only=contract_only,
        skip_large_model_hashes=skip_large_model_hashes,
        require_gpu=require_gpu,
        write_artifact=False,
    )
    payload: dict[str, object] = {
        "schema_version": 1,
        "status": "passed",
        "scope": "post_failure_frozen_step600_canonical_scoring_preflight",
        "config_sha256": config_sha,
        "contract_only": contract_only,
        "interpretation": config["interpretation"],
        "parent_failure": parent_failure,
        "model": base["model"],
        "data": base["data"],
        "processor": base["processor"],
        "runtime": base["runtime"],
        "scoring": config["scoring"],
    }
    if write_artifact:
        runtime = config["runtime"]
        assert isinstance(runtime, dict)
        output_dir = workspace_path(runtime["output_dir"])
        name = (
            "preflight_contract_only.json"
            if contract_only
            else "preflight_full.json"
            if not skip_large_model_hashes and require_gpu
            else "preflight_cpu.json"
            if skip_large_model_hashes
            else "preflight_cpu_full_model_hashes.json"
        )
        atomic_write_new_json(output_dir / name, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--contract-only", action="store_true")
    parser.add_argument("--skip-large-model-hashes", action="store_true")
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--no-write-artifact", action="store_true")
    args = parser.parse_args()
    payload = run_preflight(
        args.config,
        contract_only=args.contract_only,
        skip_large_model_hashes=args.skip_large_model_hashes,
        require_gpu=args.require_gpu,
        write_artifact=not args.no_write_artifact,
    )
    print(
        "QWEN_REAL_CONSTRAINED_PREFLIGHT_PASS",
        json.dumps(payload, sort_keys=True),
        flush=True,
    )


if __name__ == "__main__":
    main()
