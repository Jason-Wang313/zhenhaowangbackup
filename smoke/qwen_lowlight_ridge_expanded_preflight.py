#!/usr/bin/env python3
"""CPU preflight for expanded train-only Qwen ridge selection."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from predict_qwen_lowlight import validate_model_files
from train_qwen_lowlight_ordinal_head import scalar_baseline_metrics
from train_qwen_lowlight_ridge_expanded_head import (
    cached_feature_count,
    feature_kwargs,
    load_config,
    resolve_records,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config, config_sha, base_config, failed = load_config(args.config)
    _, rows, fit_records, selection_records, validation_records, exposed_sources = (
        resolve_records(config, failed)
    )
    features = config["features"]
    ridge = config["ridge"]
    assert isinstance(features, dict) and isinstance(ridge, dict)
    kwargs = feature_kwargs(config, config_sha, base_config)
    train_records = [*fit_records, *selection_records]
    cached_train = cached_feature_count(train_records, kwargs=kwargs)
    feature_dir = kwargs["feature_dir"]
    assert isinstance(feature_dir, Path)
    cached_validation_paths = sum(
        (feature_dir / f"{record.pair_id}.safetensors").exists()
        for record in validation_records
    )
    luminance = {row["pair_id"]: float(row["final_luminance"]) for row in rows}
    _, selection_baseline = scalar_baseline_metrics(
        fit_records, selection_records, luminance
    )
    model_hashes = validate_model_files(base_config)
    model = base_config["model"]
    assert isinstance(model, dict)
    print(
        "QWEN_LOWLIGHT_EXPANDED_RIDGE_PREFLIGHT_PASS "
        + json.dumps(
            {
                "config_sha256": config_sha,
                "model_revision": model["revision"],
                "model_files": len(model_hashes),
                "fit_sources": len({record.source_id for record in fit_records}),
                "fit_records": len(fit_records),
                "selection_sources": len(
                    {record.source_id for record in selection_records}
                ),
                "selection_records": len(selection_records),
                "selection_baseline_accuracy": selection_baseline["severity_accuracy"],
                "selection_baseline_mae": selection_baseline[
                    "severity_mean_absolute_error"
                ],
                "candidate_alphas": ridge["candidate_alphas"],
                "feature_dim": features["feature_dim"],
                "extraction_batch_size": features["extraction_batch_size"],
                "cached_train_features": cached_train,
                "train_features_to_extract": len(train_records) - cached_train,
                "frozen_unexposed_validation_sources": len(
                    {record.source_id for record in validation_records}
                ),
                "frozen_unexposed_validation_records": len(validation_records),
                "exposed_validation_sources_excluded": len(exposed_sources),
                "validation_feature_paths_present": cached_validation_paths,
                "validation_images_opened": 0,
                "test_split_used": False,
                "official_foundir_used": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
