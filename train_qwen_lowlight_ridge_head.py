#!/usr/bin/env python3
"""Fit a scalar ridge head on cached frozen-Qwen features, then gate validation."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from safetensors import safe_open
from safetensors.torch import save_file

from lowlight_restoration.degradation_assessment import (
    assessment_gate,
    canonical_condition_json,
    evaluate_severity,
    parse_protocol,
    record_bundle_sha256,
    require_task_ready,
    strict_condition_payload,
    validate_source_isolation,
)
from lowlight_restoration.data import load_pair_manifest
from lowlight_restoration.qwen_ordinal import select_complete_source_sequences
from lowlight_restoration.qwen_ridge import predict, standardize_fit
from predict_qwen_lowlight import atomic_write_new, sha256, validate_model_files, workspace_path
from predict_qwen_lowlight_fewshot import manifest_rows, select_unique_source_records
from train_qwen_lowlight_ordinal_head import (
    extract_features,
    feature_metadata,
    flatten,
    load_cached_feature,
    load_config as load_feature_config,
    scalar_baseline_metrics,
)

ROOT = Path(__file__).resolve().parent


def load_config(path: Path):
    raw = path.read_bytes()
    config = json.loads(raw)
    if not isinstance(config, dict) or config.get("schema_version") != 4:
        raise ValueError("bad Qwen ridge-head config")
    for section in (
        "feature_config",
        "failed_linear_selection",
        "selection_protocol",
        "final_validation_protocol",
        "ridge",
        "cache",
        "runtime",
    ):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"missing ridge config section: {section}")
    feature_reference = config["feature_config"]
    failure_reference = config["failed_linear_selection"]
    assert isinstance(feature_reference, dict) and isinstance(failure_reference, dict)
    feature_path = workspace_path(feature_reference["path"])
    failure_path = workspace_path(failure_reference["path"])
    if sha256(feature_path) != feature_reference["sha256"]:
        raise RuntimeError("frozen feature config hash mismatch")
    if sha256(failure_path) != failure_reference["sha256"]:
        raise RuntimeError("failed linear selection artifact hash mismatch")
    feature_config, feature_hash, base_config = load_feature_config(feature_path)
    if feature_hash != feature_reference["sha256"]:
        raise RuntimeError("feature config parser hash mismatch")
    failed = json.loads(failure_path.read_text())
    if (
        failed.get("status") != "failed"
        or failed.get("validation_split_touched") is not False
        or failed.get("test_split_used") is not False
        or failed.get("official_foundir_used") is not False
    ):
        raise RuntimeError("linear-head failure evidence is not safe to extend")
    selection_protocol = parse_protocol(config["selection_protocol"])
    final_protocol = parse_protocol(config["final_validation_protocol"])
    require_task_ready(selection_protocol.degradation)
    require_task_ready(final_protocol.degradation)
    if (
        selection_protocol.calibration_mode != "frozen_visual_head"
        or final_protocol.calibration_mode != "frozen_visual_head"
        or "validation" not in selection_protocol.protected_splits
    ):
        raise ValueError("bad ridge-head split/calibration protocol")
    ridge = config["ridge"]
    cache = config["cache"]
    runtime = config["runtime"]
    assert isinstance(ridge, dict) and isinstance(cache, dict) and isinstance(runtime, dict)
    choices = ridge.get("selection_choice_set")
    if not isinstance(choices, dict) or ridge.get("alpha") not in choices.get("alphas", []):
        raise ValueError("ridge alpha is outside the recorded selection choice set")
    if (
        ridge.get("solver") != "lsqr"
        or float(ridge["alpha"]) <= 0
        or float(ridge["tolerance"]) <= 0
        or int(ridge["maximum_iterations"]) <= 0
        or ridge.get("fit_intercept") is not True
        or int(ridge["expected_fitted_parameters"]) != int(cache["feature_dim"]) + 1
    ):
        raise ValueError("bad frozen ridge configuration")
    if (
        cache.get("feature_config_sha256") != feature_hash
        or int(cache["feature_dim"]) != int(feature_config["features"]["feature_dim"])  # type: ignore[index]
        or runtime.get("offline") is not True
        or runtime.get("device") != "cuda"
    ):
        raise ValueError("ridge cache/runtime does not match frozen feature extraction")
    return config, sha256(path), feature_config, base_config, failed


def git_revision() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, text=True, capture_output=True
    ).stdout.strip()


def load_feature_matrix(
    records: Sequence[object],
    *,
    feature_dir: Path,
    feature_config_sha: str,
    extractor_sha: str,
    model_revision: str,
    feature_dim: int,
    schema_version: int,
) -> np.ndarray:
    features: list[np.ndarray] = []
    for record in records:
        metadata = feature_metadata(
            record,  # type: ignore[arg-type]
            config_sha=feature_config_sha,
            model_revision=model_revision,
            feature_dim=feature_dim,
            schema_version=schema_version,
            extractor_sha256=extractor_sha,
        )
        tensor = load_cached_feature(
            feature_dir / f"{record.pair_id}.safetensors", metadata, feature_dim  # type: ignore[attr-defined]
        )
        if tensor is None:
            raise RuntimeError(f"required cached Qwen feature is missing: {record.pair_id}")  # type: ignore[attr-defined]
        features.append(tensor.numpy())
    return np.stack(features)


def metric_rows(records: Sequence[object], predictions: np.ndarray):
    rows = [
        {
            "true_level": int(record.level_index),  # type: ignore[attr-defined]
            "predicted_level": int(prediction),
            "valid_structured_output": True,
        }
        for record, prediction in zip(records, predictions.tolist(), strict=True)
    ]
    return rows, evaluate_severity(rows)


def save_ridge_new_or_equal(path: Path, tensors: Mapping[str, np.ndarray], metadata: Mapping[str, str]):
    import torch

    concrete = {
        name: torch.from_numpy(np.asarray(value, dtype=np.float64)).contiguous()
        for name, value in tensors.items()
    }
    if path.exists():
        with safe_open(path, framework="pt", device="cpu") as handle:
            if handle.metadata() != dict(metadata) or set(handle.keys()) != set(concrete):
                raise RuntimeError("existing ridge checkpoint metadata differs")
            for name, value in concrete.items():
                if not torch.equal(handle.get_tensor(name), value):
                    raise RuntimeError(f"existing ridge checkpoint differs: {name}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(concrete, path, metadata=dict(metadata))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config, config_sha, feature_config, base_config, failed_linear = load_config(args.config)
    ridge_config = config["ridge"]
    cache_config = config["cache"]
    runtime_config = config["runtime"]
    data_config = feature_config["data"]
    feature_settings = feature_config["features"]
    assert all(
        isinstance(value, dict)
        for value in (ridge_config, cache_config, runtime_config, data_config, feature_settings)
    )
    selection_protocol = parse_protocol(config["selection_protocol"])
    final_protocol = parse_protocol(config["final_validation_protocol"])
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")

    manifest = workspace_path(data_config["manifest"])
    if sha256(manifest) != data_config["manifest_sha256"]:
        raise RuntimeError("ridge manifest hash mismatch")
    records = load_pair_manifest(manifest)
    sequences = select_complete_source_sequences(
        records,
        split="train",
        source_count=int(data_config["selected_train_sources"]),
        seed=int(data_config["source_selection_seed"]),
    )
    train_records = flatten(sequences[: int(data_config["head_train_sources"])])
    selection_records = flatten(sequences[int(data_config["head_train_sources"]) :])
    validate_source_isolation(train_records, selection_records)
    model_config = base_config["model"]
    assert isinstance(model_config, dict)
    model_revision = str(model_config["revision"])
    feature_dir = workspace_path(cache_config["feature_dir"])
    feature_kwargs = {
        "feature_dir": feature_dir,
        "feature_config_sha": str(cache_config["feature_config_sha256"]),
        "extractor_sha": str(cache_config["extractor_sha256"]),
        "model_revision": model_revision,
        "feature_dim": int(cache_config["feature_dim"]),
        "schema_version": int(feature_settings["cache_schema_version"]),
    }
    train_features = load_feature_matrix(train_records, **feature_kwargs)
    selection_features = load_feature_matrix(selection_records, **feature_kwargs)
    if train_features.shape[0] + selection_features.shape[0] != int(
        cache_config["required_cached_train_selection_features"]
    ):
        raise RuntimeError("cached train/selection feature count mismatch")

    from sklearn import __version__ as sklearn_version
    from sklearn.linear_model import Ridge

    train_x, feature_mean, feature_scale = standardize_fit(train_features)
    train_y = np.array([record.level_index for record in train_records], dtype=np.float64)
    ridge = Ridge(
        alpha=float(ridge_config["alpha"]),
        solver=str(ridge_config["solver"]),
        tol=float(ridge_config["tolerance"]),
        max_iter=int(ridge_config["maximum_iterations"]),
        fit_intercept=bool(ridge_config["fit_intercept"]),
    ).fit(train_x, train_y)
    coefficient = np.asarray(ridge.coef_, dtype=np.float64)
    intercept = float(ridge.intercept_)
    _, selection_predictions = predict(
        selection_features,
        mean=feature_mean,
        scale=feature_scale,
        coefficient=coefficient,
        intercept=intercept,
    )
    _, selection_metrics = metric_rows(selection_records, selection_predictions)
    rows = manifest_rows(manifest)
    luminance = {row["pair_id"]: float(row["final_luminance"]) for row in rows}
    selection_centroids, selection_baseline = scalar_baseline_metrics(
        train_records, selection_records, luminance
    )
    selection_gate = assessment_gate(selection_metrics, selection_baseline, selection_protocol)
    if (
        abs(float(selection_metrics["severity_accuracy"]) - 0.74) > 1e-12
        or abs(float(selection_metrics["severity_mean_absolute_error"]) - 0.2633333333333333)
        > 1e-12
    ):
        raise RuntimeError("frozen ridge selection result changed")
    output_dir = workspace_path(runtime_config["output_dir"])
    checkpoint = output_dir / "ridge_head.safetensors"
    checkpoint_metadata = {
        "schema_version": "1",
        "config_sha256": config_sha,
        "feature_config_sha256": str(cache_config["feature_config_sha256"]),
        "extractor_sha256": str(cache_config["extractor_sha256"]),
        "model_revision": model_revision,
        "train_bundle_sha256": record_bundle_sha256(train_records),
        "selection_bundle_sha256": record_bundle_sha256(selection_records),
        "sklearn_version": sklearn_version,
    }
    save_ridge_new_or_equal(
        checkpoint,
        {
            "coefficient": coefficient,
            "intercept": np.array([intercept]),
            "feature_mean": feature_mean,
            "feature_scale": feature_scale,
        },
        checkpoint_metadata,
    )
    selection_summary = {
        "schema_version": 1,
        "status": "passed" if selection_gate["automatic_gate_pass"] else "failed",
        "config_sha256": config_sha,
        "git_revision": git_revision(),
        "model_revision": model_revision,
        "feature_config_sha256": cache_config["feature_config_sha256"],
        "extractor_sha256": cache_config["extractor_sha256"],
        "train_bundle_sha256": record_bundle_sha256(train_records),
        "selection_bundle_sha256": record_bundle_sha256(selection_records),
        "failed_linear_selection_sha256": config["failed_linear_selection"]["sha256"],  # type: ignore[index]
        "failed_linear_metrics": failed_linear["metrics"],
        "ridge": dict(ridge_config),
        "sklearn_version": sklearn_version,
        "metrics": selection_metrics,
        "gate": selection_gate,
        "baseline": {
            "centroids": {str(key): value for key, value in selection_centroids.items()},
            "metrics": selection_baseline,
        },
        "validation_split_touched": False,
        "test_split_used": False,
        "official_foundir_used": False,
        "checkpoint": str(checkpoint.relative_to(ROOT)),
        "checkpoint_sha256": sha256(checkpoint),
    }
    atomic_write_new(output_dir / "selection_metrics.json", selection_summary)
    print("QWEN_RIDGE_SELECTION_GATE " + json.dumps(selection_gate, sort_keys=True), flush=True)
    if not selection_gate["automatic_gate_pass"]:
        raise SystemExit(2)

    # Only after selection passes do we select/open fixed validation and load Qwen.
    validation_records = select_unique_source_records(
        records,
        split="validation",
        sample_count=int(data_config["final_validation_samples"]),
        seed=int(data_config["final_validation_selection_seed"]),
    )
    validate_source_isolation([*train_records, *selection_records], validation_records)
    all_train = [record for record in records if record.split == "train"]
    final_centroids, final_baseline = scalar_baseline_metrics(
        all_train, validation_records, luminance
    )
    if (
        abs(float(final_baseline["severity_accuracy"]) - 0.64) > 1e-12
        or abs(float(final_baseline["severity_mean_absolute_error"]) - 0.40) > 1e-12
    ):
        raise RuntimeError("frozen validation baseline changed")

    required_cublas = str(runtime_config["required_cublas_workspace_config"])
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != required_cublas:
        raise SystemExit(f"CUBLAS_WORKSPACE_CONFIG_MUST_EQUAL_{required_cublas}")
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    if not torch.cuda.is_available():
        raise SystemExit("CUDA_NOT_AVAILABLE_AFTER_SELECTION_PASS")
    torch.use_deterministic_algorithms(True)
    model_hashes = validate_model_files(base_config)
    processor_config = base_config["processor"]
    assert isinstance(processor_config, dict)
    local_model = workspace_path(model_config["local_path"])
    processor = AutoProcessor.from_pretrained(
        local_model,
        local_files_only=True,
        use_fast=bool(processor_config["use_fast"]),
        min_pixels=int(processor_config["min_pixels"]),
        max_pixels=int(processor_config["max_pixels"]),
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        local_model,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation=str(model_config["attention_implementation"]),
    ).cuda()
    model.eval().requires_grad_(False)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("Qwen must remain frozen for ridge validation")
    validation_tensors = extract_features(
        validation_records,
        model=model,
        processor=processor,
        row_luminance=luminance,
        feature_dir=feature_dir,
        config_sha=str(cache_config["feature_config_sha256"]),
        model_revision=model_revision,
        feature_dim=int(cache_config["feature_dim"]),
        schema_version=int(feature_settings["cache_schema_version"]),
        batch_size=int(feature_settings["extraction_batch_size"]),
        extractor_sha256=str(cache_config["extractor_sha256"]),
    )
    validation_features = validation_tensors.numpy()
    scores, predictions = predict(
        validation_features,
        mean=feature_mean,
        scale=feature_scale,
        coefficient=coefficient,
        intercept=intercept,
    )
    metric_data, metrics = metric_rows(validation_records, predictions)
    prediction_records = []
    for record, score, prediction, metric in zip(
        validation_records, scores.tolist(), predictions.tolist(), metric_data, strict=True
    ):
        raw_output = canonical_condition_json("low_light", int(prediction))
        if strict_condition_payload(raw_output, "low_light") != prediction:
            raise RuntimeError("ridge structured serialization mismatch")
        result = {
            "schema_version": 4,
            "pair_id": record.pair_id,
            "source_id": record.source_id,
            "split": record.split,
            "degraded_sha256": record.degraded_sha256,
            **metric,
            "raw_score": score,
            "raw_output": raw_output,
            "config_sha256": config_sha,
            "checkpoint_sha256": sha256(checkpoint),
        }
        atomic_write_new(output_dir / "records" / f"{record.pair_id}.json", result)
        prediction_records.append(result)
    gate = assessment_gate(metrics, final_baseline, final_protocol)
    summary = {
        "schema_version": 4,
        "status": "passed" if gate["automatic_gate_pass"] else "failed",
        "scope": "frozen_qwen_ridge_lowlight_validation_only",
        "config": config,
        "config_sha256": config_sha,
        "git_revision": git_revision(),
        "model_revision": model_revision,
        "model_file_sha256": model_hashes,
        "fitted_parameters": int(coefficient.size + 1),
        "checkpoint": str(checkpoint.relative_to(ROOT)),
        "checkpoint_sha256": sha256(checkpoint),
        "train_bundle_sha256": record_bundle_sha256(train_records),
        "selection_bundle_sha256": record_bundle_sha256(selection_records),
        "validation_bundle_sha256": record_bundle_sha256(validation_records),
        "selection": selection_summary,
        "baseline": {
            "centroids": {str(key): value for key, value in final_centroids.items()},
            "metrics": final_baseline,
        },
        "metrics": metrics,
        "gate": gate,
        "records": prediction_records,
        "test_split_used": False,
        "official_foundir_used": False,
        "environment": {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "sklearn": sklearn_version,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
            "max_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
        },
    }
    atomic_write_new(output_dir / "metrics.json", summary)
    marker = "QWEN_LOWLIGHT_RIDGE_VALIDATION_PASS" if gate["automatic_gate_pass"] else "QWEN_LOWLIGHT_RIDGE_VALIDATION_FAILED"
    print(marker + " " + json.dumps(gate, sort_keys=True), flush=True)
    if not gate["automatic_gate_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
