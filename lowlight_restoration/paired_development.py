"""Reusable paired-real development selection and photometric ceiling tools."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import cv2
import numpy as np
from skimage.metrics import structural_similarity


REQUIRED_PARENT_FIELDS = {
    "pair_id",
    "source_id",
    "clean_path",
    "degraded_path",
    "project_level",
    "split",
    "clean_sha256",
    "degraded_sha256",
    "dataset",
}


@dataclass(frozen=True)
class DevelopmentSource:
    source_name: str
    rows: tuple[Mapping[str, str], ...]
    required_dataset: str
    required_split: str
    required_subset: str | None
    target_kind: str


def select_paired_development_rows(
    sources: Sequence[DevelopmentSource],
) -> list[dict[str, str]]:
    """Select degraded/reference pairs without any absolute severity claim."""

    if not sources:
        raise ValueError("paired development needs at least one source")
    output: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    seen_pairs: set[tuple[str, str]] = set()
    for source in sources:
        selected = 0
        for row in source.rows:
            missing = REQUIRED_PARENT_FIELDS - set(row)
            if missing:
                raise ValueError(
                    f"{source.source_name} row is missing fields: {sorted(missing)}"
                )
            if row["split"] != source.required_split:
                continue
            if row["dataset"] != source.required_dataset:
                continue
            if source.required_subset is not None and row.get("subset") != source.required_subset:
                continue
            if row["project_level"] == "L1":
                continue
            if row["project_level"] not in {"L2", "L3", "L4", "L5"}:
                raise ValueError(f"bad proxy level: {row['pair_id']}")
            if not row["pair_id"] or not row["source_id"]:
                raise ValueError("paired development identity is empty")
            for field in ("clean_sha256", "degraded_sha256"):
                value = row[field]
                if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                    raise ValueError(f"bad {field}: {row['pair_id']}")
            development_id = f"{source.source_name}:{row['pair_id']}"
            pair_identity = (row["clean_sha256"], row["degraded_sha256"])
            if development_id in seen_ids or pair_identity in seen_pairs:
                raise ValueError(f"duplicate paired development record: {development_id}")
            seen_ids.add(development_id)
            seen_pairs.add(pair_identity)
            output.append(
                {
                    "development_id": development_id,
                    "dataset": source.source_name,
                    "scene_id": f"{source.source_name}:{row['source_id']}",
                    "parent_pair_id": row["pair_id"],
                    "target_path": row["clean_path"],
                    "input_path": row["degraded_path"],
                    "target_sha256": row["clean_sha256"],
                    "input_sha256": row["degraded_sha256"],
                    "target_kind": source.target_kind,
                    "proxy_project_level": row["project_level"],
                    "parent_split": row["split"],
                }
            )
            selected += 1
        if selected == 0:
            raise ValueError(f"paired development source is empty: {source.source_name}")
    output.sort(
        key=lambda row: (row["dataset"], row["scene_id"], row["parent_pair_id"])
    )
    return output


def peak_signal_noise_ratio(reference: np.ndarray, prediction: np.ndarray) -> float:
    if reference.shape != prediction.shape or reference.ndim != 3 or reference.shape[2] != 3:
        raise ValueError("PSNR inputs must be aligned RGB arrays")
    if not np.isfinite(reference).all() or not np.isfinite(prediction).all():
        raise ValueError("PSNR input is non-finite")
    mse = float(np.mean((reference.astype(np.float64) - prediction.astype(np.float64)) ** 2))
    if mse == 0.0:
        return 100.0
    return 10.0 * math.log10(1.0 / mse)


def rgb_structural_similarity(reference: np.ndarray, prediction: np.ndarray) -> float:
    if reference.shape != prediction.shape or reference.ndim != 3 or reference.shape[2] != 3:
        raise ValueError("SSIM inputs must be aligned RGB arrays")
    if not np.isfinite(reference).all() or not np.isfinite(prediction).all():
        raise ValueError("SSIM input is non-finite")
    value = structural_similarity(
        reference.astype(np.float64),
        prediction.astype(np.float64),
        channel_axis=2,
        data_range=1.0,
    )
    if not math.isfinite(float(value)):
        raise ValueError("SSIM result is non-finite")
    return float(value)


def _standardized_correlation(first: np.ndarray, second: np.ndarray) -> float:
    first_values = first.astype(np.float64).reshape(-1)
    second_values = second.astype(np.float64).reshape(-1)
    first_values -= first_values.mean()
    second_values -= second_values.mean()
    denominator = float(
        np.sqrt(np.dot(first_values, first_values) * np.dot(second_values, second_values))
    )
    if denominator <= 1e-12:
        return 1.0 if np.allclose(first, second) else 0.0
    return float(np.dot(first_values, second_values) / denominator)


def gradient_alignment_diagnostic(
    source: np.ndarray,
    target: np.ndarray,
    *,
    maximum_dimension: int = 512,
) -> dict[str, float]:
    """Estimate photometrically robust translation and edge agreement.

    The diagnostic never warps an image used for restoration metrics. It is
    only evidence about whether strict pixel losses are trustworthy.
    """

    if (
        source.shape != target.shape
        or source.ndim != 3
        or source.shape[2] != 3
        or maximum_dimension <= 0
    ):
        raise ValueError("alignment diagnostic requires aligned RGB arrays")
    height, width = source.shape[:2]
    scale = min(1.0, maximum_dimension / float(max(height, width)))
    resized_width = max(8, int(round(width * scale)))
    resized_height = max(8, int(round(height * scale)))

    def edge_map(image: np.ndarray) -> np.ndarray:
        luma = (
            0.2126 * image[..., 0]
            + 0.7152 * image[..., 1]
            + 0.0722 * image[..., 2]
        ).astype(np.float32)
        if (resized_width, resized_height) != (width, height):
            luma = cv2.resize(
                luma,
                (resized_width, resized_height),
                interpolation=cv2.INTER_AREA,
            )
        lower, upper = np.percentile(luma, (1.0, 99.0))
        if upper > lower + 1e-6:
            luma = np.clip((luma - lower) / (upper - lower), 0.0, 1.0)
        x_gradient = cv2.Scharr(luma, cv2.CV_32F, 1, 0)
        y_gradient = cv2.Scharr(luma, cv2.CV_32F, 0, 1)
        return cv2.magnitude(x_gradient, y_gradient)

    source_edges = edge_map(source)
    target_edges = edge_map(target)
    window = cv2.createHanningWindow(
        (resized_width, resized_height), cv2.CV_32F
    )
    shift, response = cv2.phaseCorrelate(source_edges, target_edges, window)
    transform = np.array(
        [[1.0, 0.0, shift[0]], [0.0, 1.0, shift[1]]], dtype=np.float32
    )
    shifted = cv2.warpAffine(
        source_edges,
        transform,
        (resized_width, resized_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )
    inverse_scale = 1.0 / scale
    shift_x = float(shift[0] * inverse_scale)
    shift_y = float(shift[1] * inverse_scale)
    return {
        "estimated_shift_x_pixels": shift_x,
        "estimated_shift_y_pixels": shift_y,
        "estimated_shift_magnitude_pixels": float(math.hypot(shift_x, shift_y)),
        "phase_correlation_response": float(response),
        "gradient_correlation_unshifted": _standardized_correlation(
            source_edges, target_edges
        ),
        "gradient_correlation_shifted": _standardized_correlation(
            shifted, target_edges
        ),
    }


def _sample_aligned_pixels(
    source: np.ndarray, target: np.ndarray, maximum_pixels: int
) -> tuple[np.ndarray, np.ndarray]:
    if (
        source.shape != target.shape
        or source.ndim != 3
        or source.shape[2] != 3
        or maximum_pixels <= 0
    ):
        raise ValueError("photometric fitting requires aligned RGB arrays")
    flat_source = source.reshape(-1, 3).astype(np.float64)
    flat_target = target.reshape(-1, 3).astype(np.float64)
    count = flat_source.shape[0]
    if count <= maximum_pixels:
        return flat_source, flat_target
    indices = np.linspace(0, count - 1, maximum_pixels, dtype=np.int64)
    return flat_source[indices], flat_target[indices]


def _fit_color_matrix(
    source_pixels: np.ndarray,
    target_pixels: np.ndarray,
    *,
    ridge: float,
) -> tuple[np.ndarray, float]:
    features = np.concatenate(
        [source_pixels, np.ones((source_pixels.shape[0], 1), dtype=np.float64)],
        axis=1,
    )
    gram = features.T @ features
    regularizer = np.eye(4, dtype=np.float64) * ridge
    regularizer[-1, -1] = 0.0
    coefficients = np.linalg.solve(
        gram + regularizer,
        features.T @ target_pixels,
    )
    prediction = np.clip(features @ coefficients, 0.0, 1.0)
    mse = float(np.mean((prediction - target_pixels) ** 2))
    return coefficients, mse


def fit_oracle_gamma_color(
    source: np.ndarray,
    target: np.ndarray,
    *,
    gamma_candidates: Iterable[float],
    maximum_fit_pixels: int = 32768,
    ridge: float = 1e-6,
) -> dict[str, object]:
    """Fit a GT-referenced global gamma plus 3x3 color matrix and bias.

    This is deliberately an oracle diagnostic. Its parameters must never be
    fitted from FoundIR test GT or used as an inference procedure.
    """

    if not math.isfinite(ridge) or ridge <= 0.0:
        raise ValueError("ridge must be finite and positive")
    source_pixels, target_pixels = _sample_aligned_pixels(
        source, target, maximum_fit_pixels
    )
    candidates = tuple(float(value) for value in gamma_candidates)
    if (
        not candidates
        or len(set(candidates)) != len(candidates)
        or any(not math.isfinite(value) or value <= 0.0 for value in candidates)
    ):
        raise ValueError("gamma candidates must be unique finite positive values")
    best: tuple[float, float, np.ndarray] | None = None
    for gamma in candidates:
        transformed = np.clip(source_pixels, 0.0, 1.0) ** gamma
        coefficients, mse = _fit_color_matrix(
            transformed, target_pixels, ridge=ridge
        )
        candidate = (mse, gamma, coefficients)
        if best is None or (candidate[0], candidate[1]) < (best[0], best[1]):
            best = candidate
    assert best is not None
    mse, gamma, coefficients = best
    transformed_full = np.clip(source.astype(np.float64), 0.0, 1.0) ** gamma
    features = np.concatenate(
        [
            transformed_full,
            np.ones((*transformed_full.shape[:2], 1), dtype=np.float64),
        ],
        axis=2,
    )
    prediction = np.clip(features @ coefficients, 0.0, 1.0).astype(np.float32)
    return {
        "gamma": gamma,
        "coefficients": coefficients,
        "fit_mse": mse,
        "prediction": prediction,
    }


def _mean_fields(
    rows: Sequence[Mapping[str, object]], fields: Sequence[str]
) -> dict[str, float]:
    if not rows:
        raise ValueError("cannot summarize empty metric rows")
    result: dict[str, float] = {}
    for field in fields:
        values = np.asarray([float(row[field]) for row in rows], dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"non-finite metric field: {field}")
        result[field] = float(values.mean())
    return result


def _scene_balanced_fields(
    scene_groups: Sequence[Sequence[Mapping[str, object]]], fields: Sequence[str]
) -> dict[str, float]:
    if not scene_groups:
        raise ValueError("cannot summarize empty scene groups")
    per_scene = [_mean_fields(group, fields) for group in scene_groups]
    return {
        field: float(np.mean([scene[field] for scene in per_scene]))
        for field in fields
    }


def clustered_metric_summary(
    rows: Sequence[Mapping[str, object]],
    *,
    fields: Sequence[str],
    bootstrap_seed: int,
    bootstrap_resamples: int,
    confidence: float = 0.95,
) -> dict[str, object]:
    """Record- and scene-balanced estimates with dataset-stratified bootstrap."""

    if (
        not rows
        or not fields
        or bootstrap_resamples <= 0
        or not 0.0 < confidence < 1.0
    ):
        raise ValueError("invalid clustered metric summary request")
    grouped: dict[str, dict[str, list[Mapping[str, object]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        dataset = str(row["dataset"])
        scene_id = str(row["scene_id"])
        if not dataset or not scene_id:
            raise ValueError("metric row has empty dataset or scene identity")
        grouped[dataset][scene_id].append(row)

    def estimates(
        selected: Mapping[str, Sequence[Sequence[Mapping[str, object]]]],
    ) -> dict[str, dict[str, float]]:
        all_groups = [group for groups in selected.values() for group in groups]
        all_rows = [row for group in all_groups for row in group]
        return {
            "record_weighted": _mean_fields(all_rows, fields),
            "scene_balanced": _scene_balanced_fields(all_groups, fields),
        }

    original = {
        dataset: list(scenes.values()) for dataset, scenes in sorted(grouped.items())
    }
    point = estimates(original)
    by_dataset = {
        dataset: estimates({dataset: groups})
        for dataset, groups in original.items()
    }
    generator = np.random.default_rng(bootstrap_seed)
    distributions = {
        weighting: {field: [] for field in fields}
        for weighting in ("record_weighted", "scene_balanced")
    }
    dataset_distributions = {
        dataset: {
            weighting: {field: [] for field in fields}
            for weighting in ("record_weighted", "scene_balanced")
        }
        for dataset in original
    }
    for _ in range(bootstrap_resamples):
        sampled: dict[str, list[Sequence[Mapping[str, object]]]] = {}
        for dataset, groups in original.items():
            indices = generator.integers(0, len(groups), size=len(groups))
            sampled[dataset] = [groups[int(index)] for index in indices]
        draw = estimates(sampled)
        for weighting in distributions:
            for field in fields:
                distributions[weighting][field].append(draw[weighting][field])
        for dataset, groups in sampled.items():
            dataset_draw = estimates({dataset: groups})
            for weighting in dataset_distributions[dataset]:
                for field in fields:
                    dataset_distributions[dataset][weighting][field].append(
                        dataset_draw[weighting][field]
                    )

    tail = (1.0 - confidence) / 2.0
    intervals: dict[str, dict[str, dict[str, float]]] = {}
    for weighting, metric_values in distributions.items():
        intervals[weighting] = {}
        for field, values in metric_values.items():
            lower, upper = np.quantile(values, (tail, 1.0 - tail))
            intervals[weighting][field] = {
                "estimate": point[weighting][field],
                "lower": float(lower),
                "upper": float(upper),
            }
    dataset_intervals: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    for dataset, weighting_values in dataset_distributions.items():
        dataset_intervals[dataset] = {}
        for weighting, metric_values in weighting_values.items():
            dataset_intervals[dataset][weighting] = {}
            for field, values in metric_values.items():
                lower, upper = np.quantile(values, (tail, 1.0 - tail))
                dataset_intervals[dataset][weighting][field] = {
                    "estimate": by_dataset[dataset][weighting][field],
                    "lower": float(lower),
                    "upper": float(upper),
                }
    return {
        "records": len(rows),
        "scenes": sum(len(groups) for groups in original.values()),
        "datasets": {
            dataset: {
                "records": sum(len(group) for group in groups),
                "scenes": len(groups),
                **by_dataset[dataset],
                "confidence_intervals": dataset_intervals[dataset],
            }
            for dataset, groups in original.items()
        },
        "point_estimates": point,
        "confidence_intervals": intervals,
        "bootstrap": {
            "confidence": confidence,
            "resamples": bootstrap_resamples,
            "seed": bootstrap_seed,
            "unit": "scene_cluster_stratified_by_dataset",
        },
    }
