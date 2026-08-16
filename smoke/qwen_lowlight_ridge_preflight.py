#!/usr/bin/env python3
"""CPU preflight and cached train-holdout gate for the Qwen ridge head."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lowlight_restoration.data import load_pair_manifest
from lowlight_restoration.degradation_assessment import (
    assessment_gate,
    parse_protocol,
    record_bundle_sha256,
)
from lowlight_restoration.qwen_ordinal import select_complete_source_sequences
from lowlight_restoration.qwen_ridge import predict, standardize_fit
from predict_qwen_lowlight import sha256, validate_model_files, workspace_path
from predict_qwen_lowlight_fewshot import manifest_rows
from train_qwen_lowlight_ordinal_head import flatten, scalar_baseline_metrics
from train_qwen_lowlight_ridge_head import load_config, load_feature_matrix, metric_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config, config_sha, feature_config, base_config, _ = load_config(args.config)
    ridge_config = config["ridge"]
    cache = config["cache"]
    data = feature_config["data"]
    features = feature_config["features"]
    assert all(isinstance(value, dict) for value in (ridge_config, cache, data, features))
    selection_protocol = parse_protocol(config["selection_protocol"])
    manifest = workspace_path(data["manifest"])
    if sha256(manifest) != data["manifest_sha256"]:
        raise SystemExit("BAD_RIDGE_MANIFEST_HASH")
    records = load_pair_manifest(manifest)
    sequences = select_complete_source_sequences(
        records,
        split="train",
        source_count=int(data["selected_train_sources"]),
        seed=int(data["source_selection_seed"]),
    )
    train_records = flatten(sequences[: int(data["head_train_sources"])])
    selection_records = flatten(sequences[int(data["head_train_sources"]) :])
    model_config = base_config["model"]
    assert isinstance(model_config, dict)
    feature_kwargs = {
        "feature_dir": workspace_path(cache["feature_dir"]),
        "feature_config_sha": str(cache["feature_config_sha256"]),
        "extractor_sha": str(cache["extractor_sha256"]),
        "model_revision": str(model_config["revision"]),
        "feature_dim": int(cache["feature_dim"]),
        "schema_version": int(features["cache_schema_version"]),
    }
    train_features = load_feature_matrix(train_records, **feature_kwargs)
    selection_features = load_feature_matrix(selection_records, **feature_kwargs)
    train_x, mean, scale = standardize_fit(train_features)
    from sklearn import __version__ as sklearn_version
    from sklearn.linear_model import Ridge

    model = Ridge(
        alpha=float(ridge_config["alpha"]),
        solver=str(ridge_config["solver"]),
        tol=float(ridge_config["tolerance"]),
        max_iter=int(ridge_config["maximum_iterations"]),
        fit_intercept=bool(ridge_config["fit_intercept"]),
    ).fit(train_x, np.array([record.level_index for record in train_records]))
    _, predictions = predict(
        selection_features,
        mean=mean,
        scale=scale,
        coefficient=np.asarray(model.coef_, dtype=np.float64),
        intercept=float(model.intercept_),
    )
    _, metrics = metric_rows(selection_records, predictions)
    rows = manifest_rows(manifest)
    luminance = {row["pair_id"]: float(row["final_luminance"]) for row in rows}
    _, baseline = scalar_baseline_metrics(train_records, selection_records, luminance)
    gate = assessment_gate(metrics, baseline, selection_protocol)
    if not gate["automatic_gate_pass"]:
        raise SystemExit(f"FROZEN_RIDGE_SELECTION_GATE_FAILED {json.dumps(gate, sort_keys=True)}")
    if metrics["severity_accuracy"] != 0.74 or abs(float(metrics["severity_mean_absolute_error"]) - 0.2633333333333333) > 1e-12:
        raise SystemExit("FROZEN_RIDGE_SELECTION_RESULT_CHANGED")
    hashes = validate_model_files(base_config)
    print(
        "QWEN_LOWLIGHT_RIDGE_PREFLIGHT_PASS "
        + json.dumps(
            {
                "config_sha256": config_sha,
                "model_revision": model_config["revision"],
                "model_files": len(hashes),
                "cached_features": len(train_records) + len(selection_records),
                "train_bundle_sha256": record_bundle_sha256(train_records),
                "selection_bundle_sha256": record_bundle_sha256(selection_records),
                "fitted_parameters": int(np.asarray(model.coef_).size + 1),
                "selection_accuracy": metrics["severity_accuracy"],
                "selection_mae": metrics["severity_mean_absolute_error"],
                "selection_recall": {
                    str(level): metrics["per_level"][str(level)]["recall"] for level in range(1, 6)
                },
                "baseline_accuracy": baseline["severity_accuracy"],
                "baseline_mae": baseline["severity_mean_absolute_error"],
                "sklearn": sklearn_version,
                "validation_images_opened": 0,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
