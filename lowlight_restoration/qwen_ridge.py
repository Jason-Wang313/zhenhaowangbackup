"""Deterministic scalar ridge-head helpers for ordered Qwen severity levels."""

from __future__ import annotations

import numpy as np


def standardize_fit(features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if features.ndim != 2 or features.shape[0] == 0 or not np.isfinite(features).all():
        raise ValueError("features must be a non-empty finite matrix")
    mean = features.mean(axis=0, dtype=np.float64)
    scale = features.std(axis=0, dtype=np.float64)
    scale = np.maximum(scale, 1e-6)
    standardized = (features.astype(np.float64) - mean) / scale
    return standardized, mean, scale


def standardize_apply(features: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    if (
        features.ndim != 2
        or mean.shape != (features.shape[1],)
        or scale.shape != mean.shape
        or not np.isfinite(features).all()
        or not np.isfinite(mean).all()
        or not np.isfinite(scale).all()
        or np.any(scale <= 0)
    ):
        raise ValueError("invalid features or standardization statistics")
    return (features.astype(np.float64) - mean) / scale


def decode_scores(scores: np.ndarray) -> np.ndarray:
    """Round scalar severity scores to the nearest valid ordered level."""

    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("ridge severity scores must be a finite vector")
    return np.rint(values).clip(1, 5).astype(np.int64)


def predict(
    features: np.ndarray,
    *,
    mean: np.ndarray,
    scale: np.ndarray,
    coefficient: np.ndarray,
    intercept: float,
) -> tuple[np.ndarray, np.ndarray]:
    if coefficient.shape != mean.shape or not np.isfinite(coefficient).all() or not np.isfinite(intercept):
        raise ValueError("invalid ridge-head parameters")
    scores = standardize_apply(features, mean, scale) @ coefficient + float(intercept)
    return scores, decode_scores(scores)
