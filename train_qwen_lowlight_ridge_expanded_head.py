#!/usr/bin/env python3
"""Select an expanded frozen-Qwen ridge head and gate unexposed validation."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from lowlight_restoration.data import PairRecord, load_pair_manifest
from lowlight_restoration.degradation_assessment import (
    assessment_gate,
    canonical_condition_json,
    parse_protocol,
    record_bundle_sha256,
    require_task_ready,
    strict_condition_payload,
    validate_source_isolation,
)
from lowlight_restoration.qwen_ordinal import select_complete_source_sequences
from lowlight_restoration.qwen_ridge import predict, standardize_fit
from lowlight_restoration.qwen_ridge_expanded import (
    source_dataset_counts,
    source_dataset_map,
    stratified_source_partition,
    stratified_unexposed_selection,
)
from predict_qwen_lowlight import (
    atomic_write_new,
    load_config as load_base_config,
    sha256,
    validate_model_files,
    workspace_path,
)
from predict_qwen_lowlight_fewshot import manifest_rows
from train_qwen_lowlight_ordinal_head import (
    extract_features,
    feature_metadata,
    flatten,
    load_cached_feature,
    scalar_baseline_metrics,
)
from train_qwen_lowlight_ridge_head import metric_rows, save_ridge_new_or_equal

ROOT = Path(__file__).resolve().parent


def load_config(path: Path):
    raw = path.read_bytes()
    config = json.loads(raw)
    if not isinstance(config, dict) or config.get("schema_version") != 5:
        raise ValueError("bad expanded Qwen ridge config")
    for section in (
        "base_model_config",
        "failed_ridge_validation",
        "selection_protocol",
        "final_validation_protocol",
        "data",
        "features",
        "ridge",
        "runtime",
    ):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"missing expanded ridge config section: {section}")

    base_reference = config["base_model_config"]
    failure_reference = config["failed_ridge_validation"]
    assert isinstance(base_reference, dict) and isinstance(failure_reference, dict)
    base_path = workspace_path(base_reference.get("path"))
    failure_path = workspace_path(failure_reference.get("path"))
    if sha256(base_path) != base_reference.get("sha256"):
        raise RuntimeError("expanded ridge base-model config hash mismatch")
    if sha256(failure_path) != failure_reference.get("sha256"):
        raise RuntimeError("expanded ridge failure-artifact hash mismatch")
    base_config, base_sha = load_base_config(base_path)
    if base_sha != base_reference["sha256"]:
        raise RuntimeError("expanded ridge base-model parser hash mismatch")
    failed = json.loads(failure_path.read_text())
    if (
        not isinstance(failed, dict)
        or failed.get("status") != "failed"
        or failed.get("scope") != "frozen_qwen_ridge_lowlight_validation_only"
        or failed.get("test_split_used") is not False
        or failed.get("official_foundir_used") is not False
        or failed.get("gate", {}).get("automatic_gate_pass") is not False
    ):
        raise RuntimeError(
            "referenced ridge artifact is not a protected validation failure"
        )
    exposed_records = failed.get("records")
    if (
        not isinstance(exposed_records, list)
        or len(exposed_records) != int(failure_reference["expected_exposed_sources"])
        or any(
            not isinstance(record, dict) or record.get("split") != "validation"
            for record in exposed_records
        )
    ):
        raise RuntimeError("failed ridge artifact has unexpected exposed records")
    exposed_sources = [str(record.get("source_id", "")) for record in exposed_records]
    if any(not source for source in exposed_sources) or len(
        set(exposed_sources)
    ) != len(exposed_sources):
        raise RuntimeError("failed ridge artifact has invalid exposed source IDs")

    selection_protocol = parse_protocol(config["selection_protocol"])
    final_protocol = parse_protocol(config["final_validation_protocol"])
    require_task_ready(selection_protocol.degradation)
    require_task_ready(final_protocol.degradation)
    if (
        selection_protocol.degradation != "low_light"
        or final_protocol.degradation != "low_light"
        or "validation" not in selection_protocol.protected_splits
        or set(final_protocol.protected_splits) != {"test", "foundir_test"}
    ):
        raise ValueError("expanded ridge split protocol is not Lowlight leakage-safe")

    data = config["data"]
    features = config["features"]
    ridge = config["ridge"]
    runtime = config["runtime"]
    assert all(isinstance(value, dict) for value in (data, features, ridge, runtime))
    if (
        int(data["records_per_source"]) != 5
        or int(data["fit_sources"]) + int(data["selection_sources"])
        != int(data["train_sources"])
        or int(data["fit_records"]) + int(data["selection_records"])
        != int(data["train_records"])
        or int(data["train_records"]) != 5 * int(data["train_sources"])
        or int(data["validation_records"]) != 5 * int(data["validation_sources"])
    ):
        raise ValueError("expanded ridge source/record counts are inconsistent")
    if (
        features.get("pooling") != "token_mean_and_std"
        or int(features["feature_dim"]) != 2 * int(features["embedding_dim"])
        or int(features["extraction_batch_size"]) <= 0
        or int(features["cache_schema_version"]) != 2
    ):
        raise ValueError("bad expanded ridge feature configuration")
    extractor = workspace_path(features.get("extractor_path"))
    if sha256(extractor) != features.get("extractor_sha256"):
        raise RuntimeError("expanded ridge feature extractor hash mismatch")
    alphas = ridge.get("candidate_alphas")
    if (
        ridge.get("type") != "scalar_ridge_regression"
        or ridge.get("solver") != "lsqr"
        or ridge.get("standardization") != "train_fit_mean_and_std"
        or ridge.get("decoding") != "nearest_integer_clipped_1_5"
        or not isinstance(alphas, list)
        or len(alphas) < 2
        or any(
            isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or alpha <= 0
            for alpha in alphas
        )
        or len(set(float(alpha) for alpha in alphas)) != len(alphas)
        or int(ridge["expected_fitted_parameters"]) != int(features["feature_dim"]) + 1
        or float(ridge["tolerance"]) <= 0
        or int(ridge["maximum_iterations"]) <= 0
        or ridge.get("fit_intercept") is not True
    ):
        raise ValueError("bad expanded ridge candidate configuration")
    if (
        runtime.get("offline") is not True
        or runtime.get("device") != "cuda"
        or runtime.get("required_cublas_workspace_config") not in {":4096:8", ":16:8"}
    ):
        raise ValueError("bad expanded ridge runtime configuration")
    return config, sha256(path), base_config, failed


def git_revision() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()


def resolve_records(config: Mapping[str, object], failed: Mapping[str, object]):
    data = config["data"]
    assert isinstance(data, dict)
    manifest = workspace_path(data["manifest"])
    if sha256(manifest) != data["manifest_sha256"]:
        raise RuntimeError("expanded ridge manifest hash mismatch")
    records = load_pair_manifest(manifest)
    rows = manifest_rows(manifest)
    datasets = source_dataset_map(rows)

    train_sequences = select_complete_source_sequences(
        records,
        split="train",
        source_count=int(data["train_sources"]),
        seed=int(data["train_partition_seed"]),
    )
    if (
        source_dataset_counts(train_sequences, datasets)
        != data["train_source_dataset_counts"]
    ):
        raise RuntimeError("expanded ridge train dataset counts changed")
    fit_sequences, selection_sequences = stratified_source_partition(
        train_sequences,
        source_datasets=datasets,
        selection_quotas=data["selection_source_quotas"],
        seed=int(data["train_partition_seed"]),
    )
    fit_records = flatten(fit_sequences)
    selection_records = flatten(selection_sequences)

    exposed_sources = {str(record["source_id"]) for record in failed["records"]}  # type: ignore[index]
    validation_sequences = select_complete_source_sequences(
        records,
        split="validation",
        source_count=int(data["validation_sources_total"]),
        seed=int(data["validation_selection_seed"]),
    )
    selected_validation_sequences = stratified_unexposed_selection(
        validation_sequences,
        source_datasets=datasets,
        excluded_source_ids=exposed_sources,
        selection_quotas=data["validation_source_quotas"],
        seed=int(data["validation_selection_seed"]),
    )
    validation_records = flatten(selected_validation_sequences)

    expected = (
        (
            "fit",
            fit_sequences,
            fit_records,
            "fit_sources",
            "fit_records",
            "fit_source_dataset_counts",
            "fit_bundle_sha256",
        ),
        (
            "selection",
            selection_sequences,
            selection_records,
            "selection_sources",
            "selection_records",
            "selection_source_dataset_counts",
            "selection_bundle_sha256",
        ),
        (
            "validation",
            selected_validation_sequences,
            validation_records,
            "validation_sources",
            "validation_records",
            "validation_source_dataset_counts",
            "validation_bundle_sha256",
        ),
    )
    for (
        name,
        sequences,
        flat,
        source_key,
        record_key,
        dataset_key,
        bundle_key,
    ) in expected:
        if len(sequences) != int(data[source_key]) or len(flat) != int(
            data[record_key]
        ):
            raise RuntimeError(f"expanded ridge {name} counts changed")
        if source_dataset_counts(sequences, datasets) != data[dataset_key]:
            raise RuntimeError(f"expanded ridge {name} dataset counts changed")
        if record_bundle_sha256(flat) != data[bundle_key]:
            raise RuntimeError(f"expanded ridge {name} bundle changed")

    expected_validation_ids = data["expected_validation_source_ids"]
    realized_validation_ids = sorted(
        sequence[0].source_id for sequence in selected_validation_sequences
    )
    if (
        not isinstance(expected_validation_ids, list)
        or realized_validation_ids != expected_validation_ids
    ):
        raise RuntimeError("expanded ridge frozen validation source IDs changed")
    if exposed_sources & set(realized_validation_ids):
        raise RuntimeError("expanded ridge validation reuses an exposed source")
    validate_source_isolation(fit_records, selection_records)
    validate_source_isolation([*fit_records, *selection_records], validation_records)
    return (
        manifest,
        rows,
        fit_records,
        selection_records,
        validation_records,
        exposed_sources,
    )


def feature_kwargs(
    config: Mapping[str, object], config_sha: str, base_config: Mapping[str, object]
) -> dict[str, object]:
    features = config["features"]
    model = base_config["model"]
    assert isinstance(features, dict) and isinstance(model, dict)
    return {
        "feature_dir": workspace_path(features["feature_dir"]),
        "config_sha": config_sha,
        "model_revision": str(model["revision"]),
        "feature_dim": int(features["feature_dim"]),
        "schema_version": int(features["cache_schema_version"]),
        "extractor_sha256": str(features["extractor_sha256"]),
    }


def cached_feature_count(
    records: Sequence[PairRecord], *, kwargs: Mapping[str, object]
) -> int:
    count = 0
    feature_dir = kwargs["feature_dir"]
    assert isinstance(feature_dir, Path)
    for record in records:
        metadata = feature_metadata(
            record,
            **{key: value for key, value in kwargs.items() if key != "feature_dir"},
        )
        if (
            load_cached_feature(
                feature_dir / f"{record.pair_id}.safetensors",
                metadata,
                int(kwargs["feature_dim"]),
            )
            is not None
        ):
            count += 1
    return count


def select_ridge_candidate(
    train_features: np.ndarray,
    train_records: Sequence[PairRecord],
    selection_features: np.ndarray,
    selection_records: Sequence[PairRecord],
    *,
    ridge_config: Mapping[str, object],
    selection_protocol,
    selection_baseline: Mapping[str, object],
):
    from sklearn.linear_model import Ridge

    train_x, mean, scale = standardize_fit(train_features)
    selection_x = (selection_features.astype(np.float64) - mean) / scale
    train_y = np.array(
        [record.level_index for record in train_records], dtype=np.float64
    )
    candidates: list[dict[str, object]] = []
    for raw_alpha in ridge_config["candidate_alphas"]:  # type: ignore[index]
        alpha = float(raw_alpha)
        model = Ridge(
            alpha=alpha,
            solver=str(ridge_config["solver"]),
            tol=float(ridge_config["tolerance"]),
            max_iter=int(ridge_config["maximum_iterations"]),
            fit_intercept=bool(ridge_config["fit_intercept"]),
        ).fit(train_x, train_y)
        scores = selection_x @ np.asarray(model.coef_, dtype=np.float64) + float(
            model.intercept_
        )
        predictions = np.rint(scores).clip(1, 5).astype(np.int64)
        _, metrics = metric_rows(selection_records, predictions)
        gate = assessment_gate(metrics, selection_baseline, selection_protocol)
        recalls = [float(metrics["per_level"][str(level)]["recall"]) for level in range(1, 6)]  # type: ignore[index]
        key = (
            int(gate["automatic_gate_pass"]),
            float(metrics["severity_accuracy"]),
            -float(metrics["severity_mean_absolute_error"]),
            min(recalls),
            -alpha,
        )
        candidates.append(
            {
                "alpha": alpha,
                "metrics": metrics,
                "gate": gate,
                "selection_key": list(key),
            }
        )
        print(
            "QWEN_EXPANDED_RIDGE_CANDIDATE "
            + json.dumps(
                {
                    "alpha": alpha,
                    "accuracy": metrics["severity_accuracy"],
                    "mae": metrics["severity_mean_absolute_error"],
                    "recall": {
                        str(level): metrics["per_level"][str(level)]["recall"]
                        for level in range(1, 6)
                    },
                    "automatic_gate_pass": gate["automatic_gate_pass"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    selected = max(candidates, key=lambda candidate: tuple(candidate["selection_key"]))
    return candidates, selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config, config_sha, base_config, failed = load_config(args.config)
    _, rows, fit_records, selection_records, validation_records, exposed_sources = (
        resolve_records(config, failed)
    )
    data = config["data"]
    features = config["features"]
    ridge_config = config["ridge"]
    runtime = config["runtime"]
    assert all(
        isinstance(value, dict) for value in (data, features, ridge_config, runtime)
    )
    selection_protocol = parse_protocol(config["selection_protocol"])
    final_protocol = parse_protocol(config["final_validation_protocol"])
    luminance = {row["pair_id"]: float(row["final_luminance"]) for row in rows}
    kwargs = feature_kwargs(config, config_sha, base_config)
    validation_cached_before_selection = cached_feature_count(
        validation_records, kwargs=kwargs
    )

    required_cublas = str(runtime["required_cublas_workspace_config"])
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != required_cublas:
        raise SystemExit(f"CUBLAS_WORKSPACE_CONFIG_MUST_EQUAL_{required_cublas}")
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    if not torch.cuda.is_available():
        raise SystemExit("CUDA_NOT_AVAILABLE_FOR_EXPANDED_RIDGE")
    torch.use_deterministic_algorithms(True)
    model_hashes = validate_model_files(base_config)
    model_config = base_config["model"]
    processor_config = base_config["processor"]
    assert isinstance(model_config, dict) and isinstance(processor_config, dict)
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
        raise RuntimeError("Qwen must remain frozen for expanded ridge extraction")
    print(
        "QWEN_EXPANDED_RIDGE_CUDA_READY "
        + json.dumps(
            {
                "gpu": torch.cuda.get_device_name(0),
                "fit_records": len(fit_records),
                "selection_records": len(selection_records),
                "validation_records_after_selection": len(validation_records),
                "extraction_batch_size": int(features["extraction_batch_size"]),
                "validation_cached_before_selection": validation_cached_before_selection,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    train_records = [*fit_records, *selection_records]
    train_tensors = extract_features(
        train_records,
        model=model,
        processor=processor,
        row_luminance=luminance,
        batch_size=int(features["extraction_batch_size"]),
        **kwargs,
    )
    fit_count = len(fit_records)
    fit_features = train_tensors[:fit_count].numpy()
    selection_features = train_tensors[fit_count:].numpy()
    selection_centroids, selection_baseline = scalar_baseline_metrics(
        fit_records, selection_records, luminance
    )
    candidates, selected = select_ridge_candidate(
        fit_features,
        fit_records,
        selection_features,
        selection_records,
        ridge_config=ridge_config,
        selection_protocol=selection_protocol,
        selection_baseline=selection_baseline,
    )
    output_dir = workspace_path(runtime["output_dir"])
    selection_summary = {
        "schema_version": 1,
        "status": "passed" if selected["gate"]["automatic_gate_pass"] else "failed",  # type: ignore[index]
        "config_sha256": config_sha,
        "git_revision": git_revision(),
        "model_revision": model_config["revision"],
        "fit_bundle_sha256": record_bundle_sha256(fit_records),
        "selection_bundle_sha256": record_bundle_sha256(selection_records),
        "failed_validation_sha256": config["failed_ridge_validation"]["sha256"],  # type: ignore[index]
        "ridge": ridge_config,
        "candidates": candidates,
        "selected": selected,
        "baseline": {
            "centroids": {
                str(key): value for key, value in selection_centroids.items()
            },
            "metrics": selection_baseline,
        },
        "validation_used_for_selection": False,
        "validation_features_present_before_selection": validation_cached_before_selection,
        "exposed_validation_sources_excluded": len(exposed_sources),
        "test_split_used": False,
        "official_foundir_used": False,
    }
    atomic_write_new(output_dir / "selection_metrics.json", selection_summary)
    print(
        "QWEN_EXPANDED_RIDGE_SELECTION " + json.dumps(selected, sort_keys=True),
        flush=True,
    )
    if not selected["gate"]["automatic_gate_pass"]:  # type: ignore[index]
        print("QWEN_EXPANDED_RIDGE_SELECTION_FAILED", flush=True)
        raise SystemExit(2)

    from sklearn import __version__ as sklearn_version
    from sklearn.linear_model import Ridge

    all_features = train_tensors.numpy()
    all_x, feature_mean, feature_scale = standardize_fit(all_features)
    all_y = np.array([record.level_index for record in train_records], dtype=np.float64)
    selected_alpha = float(selected["alpha"])
    final_model = Ridge(
        alpha=selected_alpha,
        solver=str(ridge_config["solver"]),
        tol=float(ridge_config["tolerance"]),
        max_iter=int(ridge_config["maximum_iterations"]),
        fit_intercept=bool(ridge_config["fit_intercept"]),
    ).fit(all_x, all_y)
    coefficient = np.asarray(final_model.coef_, dtype=np.float64)
    intercept = float(final_model.intercept_)
    checkpoint = output_dir / "ridge_head.safetensors"
    save_ridge_new_or_equal(
        checkpoint,
        {
            "coefficient": coefficient,
            "intercept": np.array([intercept]),
            "feature_mean": feature_mean,
            "feature_scale": feature_scale,
        },
        {
            "schema_version": "1",
            "config_sha256": config_sha,
            "model_revision": str(model_config["revision"]),
            "train_bundle_sha256": record_bundle_sha256(train_records),
            "selected_alpha": str(selected_alpha),
            "sklearn_version": sklearn_version,
        },
    )

    validation_tensors = extract_features(
        validation_records,
        model=model,
        processor=processor,
        row_luminance=luminance,
        batch_size=int(features["extraction_batch_size"]),
        **kwargs,
    )
    scores, predictions = predict(
        validation_tensors.numpy(),
        mean=feature_mean,
        scale=feature_scale,
        coefficient=coefficient,
        intercept=intercept,
    )
    metric_data, metrics = metric_rows(validation_records, predictions)
    prediction_records = []
    checkpoint_sha = sha256(checkpoint)
    for record, score, prediction, metric in zip(
        validation_records,
        scores.tolist(),
        predictions.tolist(),
        metric_data,
        strict=True,
    ):
        raw_output = canonical_condition_json("low_light", int(prediction))
        if strict_condition_payload(raw_output, "low_light") != prediction:
            raise RuntimeError("expanded ridge structured serialization mismatch")
        result = {
            "schema_version": 5,
            "pair_id": record.pair_id,
            "source_id": record.source_id,
            "split": record.split,
            "degraded_sha256": record.degraded_sha256,
            **metric,
            "raw_score": score,
            "raw_output": raw_output,
            "config_sha256": config_sha,
            "checkpoint_sha256": checkpoint_sha,
        }
        atomic_write_new(output_dir / "records" / f"{record.pair_id}.json", result)
        prediction_records.append(result)

    validation_centroids, validation_baseline = scalar_baseline_metrics(
        train_records, validation_records, luminance
    )
    gate = assessment_gate(metrics, validation_baseline, final_protocol)
    summary = {
        "schema_version": 5,
        "status": "passed" if gate["automatic_gate_pass"] else "failed",
        "scope": "expanded_frozen_qwen_ridge_lowlight_unexposed_validation_only",
        "config": config,
        "config_sha256": config_sha,
        "git_revision": git_revision(),
        "model_revision": model_config["revision"],
        "model_file_sha256": model_hashes,
        "fitted_parameters": int(coefficient.size + 1),
        "selected_alpha": selected_alpha,
        "checkpoint": str(checkpoint.relative_to(ROOT)),
        "checkpoint_sha256": checkpoint_sha,
        "fit_bundle_sha256": record_bundle_sha256(fit_records),
        "selection_bundle_sha256": record_bundle_sha256(selection_records),
        "validation_bundle_sha256": record_bundle_sha256(validation_records),
        "selection": selection_summary,
        "baseline": {
            "centroids": {
                str(key): value for key, value in validation_centroids.items()
            },
            "metrics": validation_baseline,
        },
        "metrics": metrics,
        "gate": gate,
        "records": prediction_records,
        "excluded_exposed_validation_sources": sorted(exposed_sources),
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
    marker = (
        "QWEN_LOWLIGHT_EXPANDED_RIDGE_VALIDATION_PASS"
        if gate["automatic_gate_pass"]
        else "QWEN_LOWLIGHT_EXPANDED_RIDGE_VALIDATION_FAILED"
    )
    print(marker + " " + json.dumps(gate, sort_keys=True), flush=True)
    if not gate["automatic_gate_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
