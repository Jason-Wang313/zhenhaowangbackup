"""Counterfactual visibility measurements for synthetic degradations."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import cv2
import numpy as np
from scipy.stats import rankdata
from skimage.metrics import structural_similarity

from .multidegradation_data import generate_record, generate_record_stages, sha256


def image_delta_metrics(full: np.ndarray, counterfactual: np.ndarray) -> dict[str, float]:
    """Measure the observable pixel effect of one causal intervention."""

    if full.shape != counterfactual.shape or full.dtype != np.uint8 or counterfactual.dtype != np.uint8:
        raise ValueError("visibility images must be shape-matched uint8 arrays")
    delta = full.astype(np.float32) - counterfactual.astype(np.float32)
    gray_full = cv2.cvtColor(full, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray_counterfactual = cv2.cvtColor(counterfactual, cv2.COLOR_BGR2GRAY).astype(np.float32)
    edge_full = cv2.Laplacian(gray_full, cv2.CV_32F)
    edge_counterfactual = cv2.Laplacian(gray_counterfactual, cv2.CV_32F)
    rgb_rmse = float(np.sqrt(np.square(delta).mean()))
    full_luma_mean = float(gray_full.mean())
    counterfactual_luma_mean = float(gray_counterfactual.mean())
    return {
        "rgb_mae_0_255": float(np.abs(delta).mean()),
        "rgb_rmse_0_255": rgb_rmse,
        "full_luma_mean_0_255": full_luma_mean,
        "counterfactual_luma_mean_0_255": counterfactual_luma_mean,
        "rgb_rmse_over_full_luma": rgb_rmse / max(full_luma_mean, 1.0),
        "luma_rmse_0_255": float(np.sqrt(np.square(gray_full - gray_counterfactual).mean())),
        "edge_rmse": float(np.sqrt(np.square(edge_full - edge_counterfactual).mean())),
        "changed_pixel_fraction_gt_2": float((np.max(np.abs(delta), axis=2) > 2.0).mean()),
        "ssim": float(
            structural_similarity(
                full,
                counterfactual,
                channel_axis=2,
                data_range=255,
            )
        ),
    }


def audit_record_components(
    record: Mapping[str, object],
    *,
    predicted_degradations: Sequence[str],
    require_frozen_output: bool = True,
) -> tuple[list[dict[str, object]], str]:
    """Render exact local/final counterfactuals for every active component."""

    order = tuple(str(name) for name in record["application_order"])
    composition_size = int(record["composition_size"])
    if not 1 <= len(order) <= 3 or len(order) != composition_size:
        raise ValueError(
            "visibility audit record must contain one to three active degradations"
        )
    prefixes = generate_record_stages(record)
    full = prefixes[-1]
    full_sha = hashlib.sha256(cv2.imencode(".png", full)[1].tobytes()).hexdigest()
    output_path = Path(__file__).resolve().parents[1] / str(record["output_path"])
    if require_frozen_output and (
        not output_path.is_file() or sha256(output_path) != full_sha
    ):
        raise RuntimeError(f"frozen full render does not reproduce: {record['record_id']}")
    predicted = set(predicted_degradations)
    rows: list[dict[str, object]] = []
    levels = record["levels"]
    assert isinstance(levels, Mapping)
    for position, name in enumerate(order):
        without = tuple(item for item in order if item != name)
        counterfactual, counterfactual_sha = generate_record(
            record,
            application_order=without,
        )
        local = image_delta_metrics(prefixes[position + 1], prefixes[position])
        final = image_delta_metrics(full, counterfactual)
        local_rmse = local["rgb_rmse_0_255"]
        local_relative_rmse = local["rgb_rmse_over_full_luma"]
        rows.append(
            {
                "record_id": str(record["record_id"]),
                "source_id": str(record["source_id"]),
                "degradation": name,
                "level": int(levels[name]),
                "position": position,
                "detected": name in predicted,
                "full_sha256": full_sha,
                "counterfactual_sha256": counterfactual_sha,
                "local": local,
                "final": final,
                "rmse_survival_ratio": final["rgb_rmse_0_255"] / local_rmse if local_rmse > 0 else 0.0,
                "relative_rmse_survival_ratio": (
                    final["rgb_rmse_over_full_luma"] / local_relative_rmse
                    if local_relative_rmse > 0
                    else 0.0
                ),
            }
        )
    return rows, full_sha


def binary_auc(scores: Sequence[float], positives: Sequence[bool]) -> float:
    """Return tie-aware ROC AUC without fitting a model."""

    if len(scores) != len(positives) or not scores:
        raise ValueError("AUC inputs must have the same nonzero length")
    positive_count = sum(positives)
    negative_count = len(positives) - positive_count
    if not positive_count or not negative_count:
        return 0.5
    ranks = rankdata(np.asarray(scores, dtype=np.float64), method="average")
    positive_rank_sum = float(sum(rank for rank, positive in zip(ranks, positives, strict=True) if positive))
    return (
        positive_rank_sum - positive_count * (positive_count + 1) / 2
    ) / (positive_count * negative_count)


def _summary(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return {"mean": 0.0, "median": 0.0, "p10": 0.0, "p90": 0.0}
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p10": float(np.quantile(array, 0.10)),
        "p90": float(np.quantile(array, 0.90)),
    }


def _stratified_auc(
    rows: Sequence[Mapping[str, object]],
    score_key: str,
) -> dict[str, float | int]:
    strata: dict[tuple[str, int], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        strata[(str(row["degradation"]), int(row["level"]))].append(row)
    weighted_auc = 0.0
    pair_weight = 0
    covered_components = 0
    for group in strata.values():
        positives = [bool(row["detected"]) for row in group]
        positive_count = sum(positives)
        negative_count = len(group) - positive_count
        if not positive_count or not negative_count:
            continue
        if score_key == "final_rgb_rmse":
            scores = [float(row["final"]["rgb_rmse_0_255"]) for row in group]  # type: ignore[index]
        elif score_key == "survival_ratio":
            scores = [float(row["rmse_survival_ratio"]) for row in group]
        elif score_key == "final_relative_rmse":
            scores = [float(row["final"]["rgb_rmse_over_full_luma"]) for row in group]  # type: ignore[index]
        elif score_key == "relative_survival_ratio":
            scores = [float(row["relative_rmse_survival_ratio"]) for row in group]
        else:
            raise ValueError(f"unsupported stratified score: {score_key}")
        weight = positive_count * negative_count
        weighted_auc += binary_auc(scores, positives) * weight
        pair_weight += weight
        covered_components += len(group)
    return {
        "auc": weighted_auc / pair_weight if pair_weight else 0.5,
        "positive_negative_pairs": pair_weight,
        "covered_components": covered_components,
    }


def summarize_component_rows(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    if not rows:
        raise ValueError("visibility summary requires component rows")
    by_position: dict[int, list[Mapping[str, object]]] = defaultdict(list)
    by_type: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    misses = Counter()
    for row in rows:
        by_position[int(row["position"])].append(row)
        by_type[str(row["degradation"])].append(row)
        if not bool(row["detected"]):
            misses[int(row["position"])] += 1

    def group_summary(group: Sequence[Mapping[str, object]]) -> dict[str, object]:
        detected = [bool(row["detected"]) for row in group]
        final_rmse = [float(row["final"]["rgb_rmse_0_255"]) for row in group]  # type: ignore[index]
        survival = [float(row["rmse_survival_ratio"]) for row in group]
        relative_final = [float(row["final"]["rgb_rmse_over_full_luma"]) for row in group]  # type: ignore[index]
        relative_survival = [float(row["relative_rmse_survival_ratio"]) for row in group]
        return {
            "components": len(group),
            "detected": sum(detected),
            "recall": sum(detected) / len(group),
            "final_rgb_rmse": _summary(final_rmse),
            "rmse_survival_ratio": _summary(survival),
            "relative_final_rgb_rmse": _summary(relative_final),
            "relative_rmse_survival_ratio": _summary(relative_survival),
        }

    detected_flags = [bool(row["detected"]) for row in rows]
    final_rmse = [float(row["final"]["rgb_rmse_0_255"]) for row in rows]  # type: ignore[index]
    survival = [float(row["rmse_survival_ratio"]) for row in rows]
    relative_final = [float(row["final"]["rgb_rmse_over_full_luma"]) for row in rows]  # type: ignore[index]
    relative_survival = [float(row["relative_rmse_survival_ratio"]) for row in rows]
    hit_rmse = [value for value, hit in zip(final_rmse, detected_flags, strict=True) if hit]
    miss_rmse = [value for value, hit in zip(final_rmse, detected_flags, strict=True) if not hit]
    hit_survival = [value for value, hit in zip(survival, detected_flags, strict=True) if hit]
    miss_survival = [value for value, hit in zip(survival, detected_flags, strict=True) if not hit]
    return {
        "components": len(rows),
        "misses_by_position": {str(position): misses[position] for position in range(3)},
        "by_position": {str(key): group_summary(value) for key, value in sorted(by_position.items())},
        "by_type": {key: group_summary(value) for key, value in sorted(by_type.items())},
        "detected_vs_missed": {
            "detected_final_rgb_rmse": _summary(hit_rmse),
            "missed_final_rgb_rmse": _summary(miss_rmse),
            "detected_survival_ratio": _summary(hit_survival),
            "missed_survival_ratio": _summary(miss_survival),
            "final_rgb_rmse_detection_auc": binary_auc(final_rmse, detected_flags),
            "survival_ratio_detection_auc": binary_auc(survival, detected_flags),
            "stratified_final_rgb_rmse_detection": _stratified_auc(
                rows, "final_rgb_rmse"
            ),
            "stratified_survival_ratio_detection": _stratified_auc(
                rows, "survival_ratio"
            ),
            "stratified_relative_final_rgb_rmse_detection": _stratified_auc(
                rows, "final_relative_rmse"
            ),
            "stratified_relative_survival_ratio_detection": _stratified_auc(
                rows, "relative_survival_ratio"
            ),
            "relative_final_rgb_rmse_detection_auc": binary_auc(
                relative_final, detected_flags
            ),
            "relative_survival_ratio_detection_auc": binary_auc(
                relative_survival, detected_flags
            ),
        },
    }


def canonical_json_sha(payload: Mapping[str, object]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
