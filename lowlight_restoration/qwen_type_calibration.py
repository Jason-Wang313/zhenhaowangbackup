"""Training-only partitioning and calibration for Qwen L1--L5 decisions."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F

from .multidegradation_data import stable_digest
from .qwen_multidegradation import DEGRADATIONS


CALIBRATION_SCHEMA = "qwen_per_type_candidate_affine_v1"
PARTITION_SCHEMA = "qwen_visibility_v2_training_source_partition_v1"


def _record_digest(record: Mapping[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(record, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()


def build_training_source_partition(
    records: Sequence[Mapping[str, object]],
    *,
    manifest_sha256: str,
    seed: int,
    calibration_fraction_numerator: int = 1,
    calibration_fraction_denominator: int = 10,
    training_image_batch_multiple: int = 40,
    minimum_calibration_type_level_records: int = 50,
) -> dict[str, object]:
    """Reserve whole training sources for calibration before model training.

    The validation and test splits are never consulted for selection.  A tiny
    tail of gradient-eligible records may be omitted per epoch so that every
    optimizer batch contains all five dialogue questions equally often.
    """

    if len(manifest_sha256) != 64:
        raise ValueError("manifest SHA-256 is malformed")
    if not 0 < calibration_fraction_numerator < calibration_fraction_denominator:
        raise ValueError("calibration fraction must be strictly between zero and one")
    if training_image_batch_multiple <= 0:
        raise ValueError("training image batch multiple must be positive")
    train = [record for record in records if record.get("split") == "train"]
    if not train:
        raise ValueError("training split is empty")
    sources_by_dataset: dict[str, set[str]] = defaultdict(set)
    for record in train:
        sources_by_dataset[str(record["source_dataset"])].add(
            str(record["source_id"])
        )
    calibration_sources: set[str] = set()
    dataset_source_counts: dict[str, dict[str, int]] = {}
    for dataset in sorted(sources_by_dataset):
        source_ids = sorted(
            sources_by_dataset[dataset],
            key=lambda source_id: stable_digest(
                seed, "qwen_v2_calibration", dataset, source_id
            ),
        )
        count = (
            len(source_ids) * calibration_fraction_numerator
            + calibration_fraction_denominator // 2
        ) // calibration_fraction_denominator
        count = max(1, min(count, len(source_ids) - 1))
        calibration_sources.update(source_ids[:count])
        dataset_source_counts[dataset] = {
            "all": len(source_ids),
            "calibration": count,
            "gradient": len(source_ids) - count,
        }
    calibration = [
        record for record in train if str(record["source_id"]) in calibration_sources
    ]
    gradient = [
        record for record in train if str(record["source_id"]) not in calibration_sources
    ]
    calibration_ids = {str(record["record_id"]) for record in calibration}
    gradient_ids = {str(record["record_id"]) for record in gradient}
    if calibration_ids & gradient_ids:
        raise RuntimeError("calibration and gradient records overlap")
    calibration_source_check = {str(record["source_id"]) for record in calibration}
    gradient_sources = {str(record["source_id"]) for record in gradient}
    if calibration_source_check & gradient_sources:
        raise RuntimeError("calibration and gradient sources overlap")
    usable_gradient_records = (
        len(gradient) // training_image_batch_multiple * training_image_batch_multiple
    )
    if usable_gradient_records <= 0:
        raise RuntimeError("source partition leaves no complete gradient batch")
    type_level_counts: Counter[tuple[str, int]] = Counter()
    composition_counts: Counter[int] = Counter()
    for record in calibration:
        composition_counts[int(record["composition_size"])] += 1
        levels = record["levels"]
        if not isinstance(levels, Mapping):
            raise ValueError("calibration record levels are malformed")
        for degradation in DEGRADATIONS:
            type_level_counts[(degradation, int(levels.get(degradation, 1)))] += 1
    if min(type_level_counts.values(), default=0) < minimum_calibration_type_level_records:
        raise RuntimeError("training-only calibration lacks type/level support")
    return {
        "schema_version": 1,
        "status": "frozen_before_qwen_visibility_v2_training",
        "partition_schema": PARTITION_SCHEMA,
        "manifest_sha256": manifest_sha256,
        "seed": seed,
        "selection": {
            "data": "training_split_only",
            "unit": "complete_source_id",
            "fraction_numerator": calibration_fraction_numerator,
            "fraction_denominator": calibration_fraction_denominator,
            "stable_order": "sha256(seed:qwen_v2_calibration:dataset:source_id)",
        },
        "dataset_source_counts": dataset_source_counts,
        "calibration_source_ids": sorted(calibration_sources),
        "gradient_source_ids": sorted(gradient_sources),
        "calibration_record_ids": sorted(calibration_ids),
        "gradient_eligible_record_ids": sorted(gradient_ids),
        "calibration_records": len(calibration),
        "gradient_eligible_records": len(gradient),
        "gradient_records_per_epoch": usable_gradient_records,
        "omitted_gradient_tail_per_epoch": len(gradient) - usable_gradient_records,
        "training_image_batch_multiple": training_image_batch_multiple,
        "calibration_composition_counts": {
            str(size): composition_counts[size] for size in range(4)
        },
        "calibration_type_level_counts": {
            f"{degradation}:L{level}": type_level_counts[(degradation, level)]
            for degradation in DEGRADATIONS
            for level in range(1, 6)
        },
        "minimum_calibration_type_level_records": minimum_calibration_type_level_records,
        "source_disjoint": True,
        "validation_or_test_used_for_selection": False,
        "record_content_sha256": {
            str(record["record_id"]): _record_digest(record)
            for record in calibration
        },
    }


def validate_training_source_partition(
    payload: Mapping[str, object],
    records: Sequence[Mapping[str, object]],
    *,
    manifest_sha256: str,
) -> dict[str, object]:
    """Rebuild and compare a frozen partition against its source manifest."""

    if (
        payload.get("status") != "frozen_before_qwen_visibility_v2_training"
        or payload.get("partition_schema") != PARTITION_SCHEMA
        or payload.get("manifest_sha256") != manifest_sha256
        or payload.get("validation_or_test_used_for_selection") is not False
    ):
        raise RuntimeError("training-only calibration partition identity changed")
    selection = payload.get("selection")
    if not isinstance(selection, Mapping):
        raise RuntimeError("calibration partition selection is malformed")
    rebuilt = build_training_source_partition(
        records,
        manifest_sha256=manifest_sha256,
        seed=int(payload["seed"]),
        calibration_fraction_numerator=int(selection["fraction_numerator"]),
        calibration_fraction_denominator=int(selection["fraction_denominator"]),
        training_image_batch_multiple=int(payload["training_image_batch_multiple"]),
        minimum_calibration_type_level_records=int(
            payload["minimum_calibration_type_level_records"]
        ),
    )
    if "manifest" in payload:
        rebuilt["manifest"] = payload["manifest"]
    if rebuilt != dict(payload):
        raise RuntimeError("training-only calibration partition is not deterministic")
    return rebuilt


def partition_records(
    records: Sequence[Mapping[str, object]], payload: Mapping[str, object]
) -> tuple[list[Mapping[str, object]], list[Mapping[str, object]]]:
    calibration_ids = set(map(str, payload["calibration_record_ids"]))
    gradient_ids = set(map(str, payload["gradient_eligible_record_ids"]))
    calibration = [
        record for record in records if str(record.get("record_id")) in calibration_ids
    ]
    gradient = [
        record for record in records if str(record.get("record_id")) in gradient_ids
    ]
    if len(calibration) != len(calibration_ids) or len(gradient) != len(gradient_ids):
        raise RuntimeError("calibration partition record identities are incomplete")
    return calibration, gradient


def _calibration_tensors(
    rows: Sequence[Mapping[str, object]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not rows:
        raise ValueError("candidate calibration rows are empty")
    type_indices = []
    scores = []
    labels = []
    for row in rows:
        degradation = str(row["degradation"])
        if degradation not in DEGRADATIONS:
            raise ValueError(f"unknown calibration degradation: {degradation}")
        values = row["candidate_scores"]
        if not isinstance(values, Sequence) or len(values) != 5:
            raise ValueError("candidate calibration scores must contain L1--L5")
        numeric = [float(value) for value in values]
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError("candidate calibration scores must be finite")
        level = int(row["true_level"])
        if level not in range(1, 6):
            raise ValueError("candidate calibration label must be L1--L5")
        type_indices.append(DEGRADATIONS.index(degradation))
        scores.append(numeric)
        labels.append(level - 1)
    return (
        torch.tensor(type_indices, dtype=torch.long),
        torch.tensor(scores, dtype=torch.float64),
        torch.tensor(labels, dtype=torch.long),
    )


def fit_type_calibration(
    rows: Sequence[Mapping[str, object]],
    *,
    regularization: float = 1e-3,
    maximum_iterations: int = 200,
) -> dict[str, object]:
    """Fit bounded per-type temperature and class biases on training data only."""

    if not math.isfinite(regularization) or regularization <= 0:
        raise ValueError("calibration regularization must be positive")
    if maximum_iterations <= 0:
        raise ValueError("calibration iterations must be positive")
    type_indices, scores, labels = _calibration_tensors(rows)
    counts = Counter(
        (str(row["degradation"]), int(row["true_level"])) for row in rows
    )
    if any(counts[(name, level)] == 0 for name in DEGRADATIONS for level in range(1, 6)):
        raise ValueError("calibration rows do not cover every type and level")
    raw_temperature = torch.full(
        (len(DEGRADATIONS),), math.log(0.2 / 0.8), dtype=torch.float64,
        requires_grad=True,
    )
    raw_bias = torch.zeros(
        (len(DEGRADATIONS), 5), dtype=torch.float64, requires_grad=True
    )
    optimizer = torch.optim.LBFGS(
        [raw_temperature, raw_bias],
        lr=0.5,
        max_iter=maximum_iterations,
        tolerance_grad=1e-10,
        tolerance_change=1e-12,
        line_search_fn="strong_wolfe",
    )

    def parameters() -> tuple[torch.Tensor, torch.Tensor]:
        temperature = 0.25 + 3.75 * torch.sigmoid(raw_temperature)
        bias = raw_bias - raw_bias.mean(dim=1, keepdim=True)
        return temperature, bias

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        temperature, bias = parameters()
        logits = scores / temperature[type_indices, None] + bias[type_indices]
        loss = F.cross_entropy(logits, labels)
        loss = loss + regularization * (
            bias.square().mean() + torch.log(temperature).square().mean()
        )
        loss.backward()
        return loss

    raw_nll = float(F.cross_entropy(scores, labels).item())
    optimizer.step(closure)
    with torch.no_grad():
        temperature, bias = parameters()
        calibrated = scores / temperature[type_indices, None] + bias[type_indices]
        calibrated_nll = float(F.cross_entropy(calibrated, labels).item())
        raw_accuracy = float(scores.argmax(dim=1).eq(labels).double().mean().item())
        calibrated_accuracy = float(
            calibrated.argmax(dim=1).eq(labels).double().mean().item()
        )
    if not math.isfinite(calibrated_nll) or calibrated_nll > raw_nll + 1e-9:
        raise RuntimeError("training-only calibration failed to improve NLL")
    return {
        "schema_version": 1,
        "status": "fit_on_training_sources_only",
        "calibration_schema": CALIBRATION_SCHEMA,
        "degradation_order": list(DEGRADATIONS),
        "level_order": [1, 2, 3, 4, 5],
        "temperature": {
            name: float(temperature[index].item())
            for index, name in enumerate(DEGRADATIONS)
        },
        "class_bias": {
            name: [float(value) for value in bias[index].tolist()]
            for index, name in enumerate(DEGRADATIONS)
        },
        "regularization": regularization,
        "maximum_iterations": maximum_iterations,
        "records": len(rows),
        "type_level_counts": {
            f"{name}:L{level}": counts[(name, level)]
            for name in DEGRADATIONS
            for level in range(1, 6)
        },
        "raw_nll": raw_nll,
        "calibrated_nll": calibrated_nll,
        "raw_accuracy": raw_accuracy,
        "calibrated_accuracy": calibrated_accuracy,
        "validation_or_test_used_for_fit": False,
    }


def calibrated_level(
    degradation: str,
    candidate_scores: Sequence[float],
    calibration: Mapping[str, object],
) -> tuple[int, list[float]]:
    if (
        calibration.get("calibration_schema") != CALIBRATION_SCHEMA
        or calibration.get("validation_or_test_used_for_fit") is not False
        or calibration.get("degradation_order") != list(DEGRADATIONS)
    ):
        raise RuntimeError("candidate calibration identity changed")
    if degradation not in DEGRADATIONS or len(candidate_scores) != 5:
        raise ValueError("candidate decision must specify one type and five scores")
    temperature = float(calibration["temperature"][degradation])
    bias = [float(value) for value in calibration["class_bias"][degradation]]
    adjusted = [
        float(score) / temperature + offset
        for score, offset in zip(candidate_scores, bias, strict=True)
    ]
    return max(range(1, 6), key=lambda level: adjusted[level - 1]), adjusted
