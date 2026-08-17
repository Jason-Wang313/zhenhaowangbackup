"""Frozen ridge-checkpoint and prediction-manifest helpers for conditioning."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from safetensors import safe_open

from .authorization import sha256
from .conditioned_authorization import (
    CONDITIONED_CONFIG_SHA256,
    QWEN_CHECKPOINT_SHA256,
    QWEN_CONFIG_SHA256,
    QWEN_METRICS_SHA256,
    QWEN_MODEL_REVISION,
    workspace_path,
)
from .config import TrainConfig
from .data import PairRecord
from .degradation_assessment import canonical_condition_json, strict_condition_payload


def load_frozen_ridge(path: Path) -> dict[str, np.ndarray | float]:
    """Load and validate the exact expanded ridge checkpoint."""

    if not path.is_file() or sha256(path) != QWEN_CHECKPOINT_SHA256:
        raise RuntimeError("frozen Qwen ridge checkpoint is missing or changed")
    with safe_open(path, framework="pt", device="cpu") as handle:
        expected_keys = {"coefficient", "feature_mean", "feature_scale", "intercept"}
        if set(handle.keys()) != expected_keys:
            raise RuntimeError("frozen Qwen ridge checkpoint keys changed")
        metadata = handle.metadata()
        expected_metadata = {
            "schema_version": "1",
            "config_sha256": QWEN_CONFIG_SHA256,
            "model_revision": QWEN_MODEL_REVISION,
            "train_bundle_sha256": "947c1472d23b9a7a423cfd3826e81a88453dbc002ee4f24317c238c41177e493",
            "selected_alpha": "3000.0",
            "sklearn_version": "1.5.2",
        }
        if metadata != expected_metadata:
            raise RuntimeError("frozen Qwen ridge checkpoint metadata changed")
        coefficient = handle.get_tensor("coefficient").numpy().astype(np.float64)
        feature_mean = handle.get_tensor("feature_mean").numpy().astype(np.float64)
        feature_scale = handle.get_tensor("feature_scale").numpy().astype(np.float64)
        intercept_tensor = handle.get_tensor("intercept").numpy().astype(np.float64)
    if (
        coefficient.shape != (7168,)
        or feature_mean.shape != (7168,)
        or feature_scale.shape != (7168,)
        or intercept_tensor.shape != (1,)
        or not np.isfinite(coefficient).all()
        or not np.isfinite(feature_mean).all()
        or not np.isfinite(feature_scale).all()
        or not np.isfinite(intercept_tensor).all()
        or np.any(feature_scale <= 0)
    ):
        raise RuntimeError("frozen Qwen ridge checkpoint tensors are invalid")
    return {
        "coefficient": coefficient,
        "feature_mean": feature_mean,
        "feature_scale": feature_scale,
        "intercept": float(intercept_tensor[0]),
    }


def prediction_bundle_sha256(records: Sequence[Mapping[str, object]]) -> str:
    payload = [
        {
            "pair_id": record.get("pair_id"),
            "source_id": record.get("source_id"),
            "split": record.get("split"),
            "degraded_sha256": record.get("degraded_sha256"),
            "predicted_level": record.get("predicted_level"),
            "raw_score": record.get("raw_score"),
            "raw_output": record.get("raw_output"),
            "qwen_checkpoint_sha256": record.get("qwen_checkpoint_sha256"),
        }
        for record in records
    ]
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def validate_prediction_record(
    payload: Mapping[str, object], pair: PairRecord, *, index: int
) -> None:
    expected = {
        "schema_version": 1,
        "index": index,
        "pair_id": pair.pair_id,
        "source_id": pair.source_id,
        "split": pair.split,
        "degraded_sha256": pair.degraded_sha256,
        "conditioned_config_sha256": CONDITIONED_CONFIG_SHA256,
        "qwen_config_sha256": QWEN_CONFIG_SHA256,
        "qwen_metrics_sha256": QWEN_METRICS_SHA256,
        "qwen_checkpoint_sha256": QWEN_CHECKPOINT_SHA256,
        "qwen_model_revision": QWEN_MODEL_REVISION,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise RuntimeError(f"prediction record differs at {key}: {pair.pair_id}")
    if "true_level" in payload or "project_level" in payload:
        raise RuntimeError(f"conditioning record leaks the target label: {pair.pair_id}")
    level = payload.get("predicted_level")
    if isinstance(level, bool) or not isinstance(level, int) or level not in range(1, 6):
        raise RuntimeError(f"prediction level is invalid: {pair.pair_id}")
    score = payload.get("raw_score")
    if (
        isinstance(score, bool)
        or not isinstance(score, (int, float))
        or not math.isfinite(float(score))
    ):
        raise RuntimeError(f"prediction score is invalid: {pair.pair_id}")
    raw_output = payload.get("raw_output")
    if raw_output != canonical_condition_json("low_light", level):
        raise RuntimeError(f"prediction JSON is not canonical: {pair.pair_id}")
    if strict_condition_payload(str(raw_output), "low_light") != level:
        raise RuntimeError(f"prediction JSON does not parse: {pair.pair_id}")


def load_prediction_manifest(
    config: TrainConfig,
    records: Sequence[PairRecord],
    *,
    verify_sidecars: bool = False,
) -> tuple[dict[str, int], dict[str, object], str]:
    """Load the exact all-split prediction bundle and return pair-to-level mapping."""

    condition = config.section("condition")
    path = workspace_path(
        condition.get("prediction_manifest"), label="condition.prediction_manifest"
    )
    if not path.is_file():
        raise RuntimeError(f"missing frozen Qwen prediction manifest: {path}")
    manifest_sha = sha256(path)
    try:
        summary = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("cannot parse frozen Qwen prediction manifest") from exc
    if not isinstance(summary, dict):
        raise RuntimeError("frozen Qwen prediction manifest must be an object")
    expected = {
        "schema_version": 1,
        "status": "qwen_lowlight_condition_predictions_complete",
        "conditioned_config_sha256": CONDITIONED_CONFIG_SHA256,
        "qwen_config_sha256": QWEN_CONFIG_SHA256,
        "qwen_metrics_sha256": QWEN_METRICS_SHA256,
        "qwen_checkpoint_sha256": QWEN_CHECKPOINT_SHA256,
        "qwen_model_revision": QWEN_MODEL_REVISION,
        "records": int(condition["expected_prediction_records"]),
        "split_counts": condition["expected_prediction_splits"],
        "target_labels_exported": False,
        "test_metrics_computed": False,
        "official_foundir_used": False,
    }
    for key, value in expected.items():
        if summary.get(key) != value:
            raise RuntimeError(f"prediction manifest differs at {key}")
    payloads = summary.get("predictions")
    if not isinstance(payloads, list) or len(payloads) != len(records):
        raise RuntimeError("prediction manifest record count changed")
    mapping: dict[str, int] = {}
    split_counts: Counter[str] = Counter()
    predicted_counts: Counter[int] = Counter()
    output_dir = workspace_path(
        condition.get("prediction_output_dir"), label="condition.prediction_output_dir"
    )
    for index, (payload, pair) in enumerate(zip(payloads, records, strict=True)):
        if not isinstance(payload, dict):
            raise RuntimeError(f"prediction record {index} is not an object")
        validate_prediction_record(payload, pair, index=index)
        if pair.pair_id in mapping:
            raise RuntimeError(f"duplicate prediction pair: {pair.pair_id}")
        mapping[pair.pair_id] = int(payload["predicted_level"])
        split_counts[pair.split] += 1
        predicted_counts[int(payload["predicted_level"])] += 1
        if verify_sidecars:
            sidecar = output_dir / "records" / f"{pair.pair_id}.json"
            if not sidecar.is_file() or json.loads(sidecar.read_text()) != payload:
                raise RuntimeError(f"prediction sidecar missing or differs: {pair.pair_id}")
    expected_splits = condition["expected_prediction_splits"]
    if not isinstance(expected_splits, dict):
        raise RuntimeError("prediction split contract is invalid")
    realized_splits = {str(split): split_counts[str(split)] for split in expected_splits}
    if realized_splits != expected_splits or set(split_counts) - set(expected_splits):
        raise RuntimeError("prediction split counts changed")
    if set(predicted_counts) != {1, 2, 3, 4, 5}:
        raise RuntimeError("prediction bundle does not represent all five output levels")
    bundle_sha = prediction_bundle_sha256(payloads)
    if summary.get("prediction_bundle_sha256") != bundle_sha:
        raise RuntimeError("prediction bundle SHA-256 mismatch")
    return mapping, summary, manifest_sha
