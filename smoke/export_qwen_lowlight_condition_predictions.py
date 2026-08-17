#!/usr/bin/env python3
"""Export frozen ridge Lowlight predictions for the fair FLUX comparison."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lowlight_restoration.conditioned_authorization import (
    CONDITIONED_CONFIG_SHA256,
    QWEN_CHECKPOINT_SHA256,
    QWEN_CONFIG_SHA256,
    QWEN_METRICS_SHA256,
    QWEN_MODEL_REVISION,
    validate_conditioned_authorization,
    workspace_path,
)
from lowlight_restoration.config import load_train_config
from lowlight_restoration.data import load_pair_manifest
from lowlight_restoration.degradation_assessment import canonical_condition_json
from lowlight_restoration.qwen_predictions import (
    load_frozen_ridge,
    load_prediction_manifest,
    prediction_bundle_sha256,
)
from lowlight_restoration.qwen_ridge import predict
from predict_qwen_lowlight import atomic_write_new, validate_model_files
from predict_qwen_lowlight_fewshot import manifest_rows
from train_qwen_lowlight_ordinal_head import extract_features
from train_qwen_lowlight_ridge_expanded_head import load_config as load_expanded_config


def git_revision() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    config = load_train_config(args.config)
    try:
        scope = validate_conditioned_authorization(config)
    except RuntimeError as exc:
        raise SystemExit(f"RUN_NOT_AUTHORIZED {exc}") from exc
    if scope != "full_qwen_conditioned":
        raise SystemExit("QWEN_CONDITIONED_SCOPE_REQUIRED")
    condition = config.section("condition")
    records = load_pair_manifest(config.workspace_path("data", "manifest"))
    if len(records) != int(condition["expected_prediction_records"]):
        raise SystemExit("BAD_PREDICTION_RECORD_COUNT")
    prediction_manifest = workspace_path(
        condition["prediction_manifest"], label="condition.prediction_manifest"
    )
    if prediction_manifest.exists():
        _, summary, digest = load_prediction_manifest(
            config, records, verify_sidecars=True
        )
        print(
            "QWEN_LOWLIGHT_CONDITION_PREDICTIONS_REUSED "
            + json.dumps(
                {
                    "records": summary["records"],
                    "prediction_manifest_sha256": digest,
                    "prediction_bundle_sha256": summary["prediction_bundle_sha256"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return

    required_cublas = ":4096:8"
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != required_cublas:
        raise SystemExit(f"CUBLAS_WORKSPACE_CONFIG_MUST_EQUAL_{required_cublas}")
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")

    expanded_path = workspace_path(
        condition["qwen_expanded_config"], label="condition.qwen_expanded_config"
    )
    expanded, expanded_sha, base_config, _ = load_expanded_config(expanded_path)
    if expanded_sha != QWEN_CONFIG_SHA256:
        raise RuntimeError("expanded Qwen config changed before prediction export")
    model_config = base_config["model"]
    processor_config = base_config["processor"]
    features = expanded["features"]
    assert isinstance(model_config, dict)
    assert isinstance(processor_config, dict)
    assert isinstance(features, dict)
    if (
        model_config.get("revision") != QWEN_MODEL_REVISION
        or features.get("feature_dir") != condition["feature_dir"]
        or features.get("feature_dim") != condition["feature_dim"]
        or features.get("cache_schema_version")
        != condition["feature_cache_schema_version"]
        or features.get("extraction_batch_size")
        != condition["feature_extraction_batch_size"]
        or features.get("extractor_sha256") != condition["feature_extractor_sha256"]
    ):
        raise RuntimeError("prediction feature settings differ from validated assessor")

    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    if not torch.cuda.is_available():
        raise SystemExit("CUDA_NOT_AVAILABLE_FOR_QWEN_PREDICTION_EXPORT")
    torch.manual_seed(2026081611)
    torch.cuda.manual_seed_all(2026081611)
    torch.use_deterministic_algorithms(True)
    model_hashes = validate_model_files(base_config)
    local_model = workspace_path(
        model_config["local_path"], label="qwen.model.local_path"
    )
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
        raise RuntimeError("Qwen must remain frozen during prediction export")
    print(
        "QWEN_CONDITION_EXPORT_CUDA_READY "
        + json.dumps(
            {
                "gpu": torch.cuda.get_device_name(0),
                "records": len(records),
                "batch_size": condition["feature_extraction_batch_size"],
            },
            sort_keys=True,
        ),
        flush=True,
    )

    rows = manifest_rows(config.workspace_path("data", "manifest"))
    row_luminance = {row["pair_id"]: float(row["final_luminance"]) for row in rows}
    feature_tensors = extract_features(
        records,
        model=model,
        processor=processor,
        row_luminance=row_luminance,
        feature_dir=workspace_path(condition["feature_dir"], label="condition.feature_dir"),
        config_sha=QWEN_CONFIG_SHA256,
        model_revision=QWEN_MODEL_REVISION,
        feature_dim=int(condition["feature_dim"]),
        schema_version=int(condition["feature_cache_schema_version"]),
        batch_size=int(condition["feature_extraction_batch_size"]),
        extractor_sha256=str(condition["feature_extractor_sha256"]),
    )
    ridge = load_frozen_ridge(
        workspace_path(condition["qwen_checkpoint"], label="condition.qwen_checkpoint")
    )
    scores, predictions = predict(
        feature_tensors.numpy(),
        mean=np.asarray(ridge["feature_mean"]),
        scale=np.asarray(ridge["feature_scale"]),
        coefficient=np.asarray(ridge["coefficient"]),
        intercept=float(ridge["intercept"]),
    )

    output_dir = workspace_path(
        condition["prediction_output_dir"], label="condition.prediction_output_dir"
    )
    prediction_records: list[dict[str, object]] = []
    split_counts: Counter[str] = Counter()
    predicted_counts: Counter[int] = Counter()
    predicted_by_split: dict[str, Counter[int]] = {
        split: Counter() for split in ("train", "validation", "test")
    }
    for index, (pair, score, prediction) in enumerate(
        zip(records, scores.tolist(), predictions.tolist(), strict=True)
    ):
        level = int(prediction)
        result = {
            "schema_version": 1,
            "index": index,
            "pair_id": pair.pair_id,
            "source_id": pair.source_id,
            "split": pair.split,
            "degraded_sha256": pair.degraded_sha256,
            "predicted_level": level,
            "raw_score": float(score),
            "raw_output": canonical_condition_json("low_light", level),
            "conditioned_config_sha256": CONDITIONED_CONFIG_SHA256,
            "qwen_config_sha256": QWEN_CONFIG_SHA256,
            "qwen_metrics_sha256": QWEN_METRICS_SHA256,
            "qwen_checkpoint_sha256": QWEN_CHECKPOINT_SHA256,
            "qwen_model_revision": QWEN_MODEL_REVISION,
        }
        atomic_write_new(output_dir / "records" / f"{pair.pair_id}.json", result)
        prediction_records.append(result)
        split_counts[pair.split] += 1
        predicted_counts[level] += 1
        predicted_by_split[pair.split][level] += 1
        if (index + 1) % 500 == 0 or index + 1 == len(records):
            print(
                f"QWEN_CONDITION_PREDICTION_PROGRESS {index + 1}/{len(records)}",
                flush=True,
            )

    if dict(split_counts) != condition["expected_prediction_splits"]:
        raise RuntimeError("realized prediction split counts changed")
    if set(predicted_counts) != {1, 2, 3, 4, 5}:
        raise RuntimeError("frozen assessor did not output every level")
    summary = {
        "schema_version": 1,
        "status": "qwen_lowlight_condition_predictions_complete",
        "conditioned_config_sha256": CONDITIONED_CONFIG_SHA256,
        "qwen_config_sha256": QWEN_CONFIG_SHA256,
        "qwen_metrics_sha256": QWEN_METRICS_SHA256,
        "qwen_checkpoint_sha256": QWEN_CHECKPOINT_SHA256,
        "qwen_model_revision": QWEN_MODEL_REVISION,
        "model_file_sha256": model_hashes,
        "git_revision": git_revision(),
        "records": len(prediction_records),
        "split_counts": dict(split_counts),
        "predicted_level_counts": {
            str(level): predicted_counts[level] for level in range(1, 6)
        },
        "predicted_level_counts_by_split": {
            split: {
                str(level): predicted_by_split[split][level] for level in range(1, 6)
            }
            for split in ("train", "validation", "test")
        },
        "prediction_bundle_sha256": prediction_bundle_sha256(prediction_records),
        "target_labels_exported": False,
        "test_metrics_computed": False,
        "official_foundir_used": False,
        "predictions": prediction_records,
        "environment": {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        },
    }
    atomic_write_new(prediction_manifest, summary)
    _, validated, manifest_sha = load_prediction_manifest(
        config, records, verify_sidecars=True
    )
    print(
        "QWEN_LOWLIGHT_CONDITION_PREDICTIONS_COMPLETE "
        + json.dumps(
            {
                "records": validated["records"],
                "prediction_manifest_sha256": manifest_sha,
                "prediction_bundle_sha256": validated["prediction_bundle_sha256"],
                "predicted_level_counts": validated["predicted_level_counts"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
