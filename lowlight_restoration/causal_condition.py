"""Condition features and zero-start adapter for the Gate 3 causal matrix."""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np
from PIL import Image
import torch
from torch import Tensor, nn


def _small_rgb(image: np.ndarray, size: int = 256) -> np.ndarray:
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("expected an RGB uint8 image")
    height, width = image.shape[:2]
    scale = min(1.0, size / max(height, width))
    if scale < 1.0:
        image = np.asarray(
            Image.fromarray(image, mode="RGB").resize(
                (max(1, round(width * scale)), max(1, round(height * scale))),
                Image.Resampling.BILINEAR,
            ),
            dtype=np.uint8,
        )
    return image.astype(np.float32) / 255.0


def measured_image_features(image: np.ndarray) -> np.ndarray:
    """Deterministic input-only tone, clipping, noise, and spatial statistics."""

    rgb = _small_rgb(image)
    flat = rgb.reshape(-1, 3)
    luma = rgb @ np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float32)
    values: list[float] = []
    values.extend(flat.mean(0).tolist())
    values.extend(flat.std(0).tolist())
    for quantile in (0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99):
        values.extend(np.quantile(flat, quantile, axis=0).tolist())
    values.extend([float(luma.mean()), float(luma.std())])
    values.extend(np.quantile(luma, (0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99)).tolist())
    for threshold in (1.5 / 255.0, 4.5 / 255.0, 250.5 / 255.0, 253.5 / 255.0):
        if threshold < 0.5:
            values.extend((flat <= threshold).mean(0).tolist())
            values.append(float((luma <= threshold).mean()))
        else:
            values.extend((flat >= threshold).mean(0).tolist())
            values.append(float((luma >= threshold).mean()))
    horizontal = np.diff(luma, axis=1).reshape(-1)
    vertical = np.diff(luma, axis=0).reshape(-1)
    gradient = np.concatenate((np.abs(horizontal), np.abs(vertical)))
    values.extend(
        [
            float(gradient.mean()),
            float(gradient.std()),
            *np.quantile(gradient, (0.5, 0.9, 0.99)).tolist(),
        ]
    )
    padded = np.pad(luma, 1, mode="reflect")
    smooth = sum(
        padded[y : y + luma.shape[0], x : x + luma.shape[1]]
        for y in range(3)
        for x in range(3)
    ) / 9.0
    noise = np.abs(luma - smooth)
    values.extend([float(noise.mean()), float(np.quantile(noise, 0.5)), float(np.quantile(noise, 0.9))])
    for row in np.array_split(luma, 4, axis=0):
        values.extend(float(cell.mean()) for cell in np.array_split(row, 4, axis=1))
    result = np.asarray(values, dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError("measured features are non-finite")
    return result


def oracle_pair_features(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Bounded target-derived rendering summary used only by the oracle arm."""

    source_rgb = _small_rgb(source)
    target_rgb = _small_rgb(target)
    if source_rgb.shape != target_rgb.shape:
        target_rgb = np.asarray(
            Image.fromarray(target, mode="RGB").resize(
                (source_rgb.shape[1], source_rgb.shape[0]), Image.Resampling.BILINEAR
            ),
            dtype=np.float32,
        ) / 255.0
    source_flat, target_flat = source_rgb.reshape(-1, 3), target_rgb.reshape(-1, 3)
    weights = np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float32)
    source_luma, target_luma = source_rgb @ weights, target_rgb @ weights
    epsilon = 1.0 / 1024.0
    values: list[float] = []
    values.extend(np.clip(np.log((target_flat.mean(0) + epsilon) / (source_flat.mean(0) + epsilon)), -4, 4).tolist())
    source_q = np.quantile(source_luma, (0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99))
    target_q = np.quantile(target_luma, (0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99))
    values.extend(np.clip(np.log((target_q + epsilon) / (source_q + epsilon)), -4, 4).tolist())
    values.extend(target_flat.mean(0).tolist())
    values.extend(target_flat.std(0).tolist())
    for threshold in (4.5 / 255.0, 250.5 / 255.0):
        comparator = np.less_equal if threshold < 0.5 else np.greater_equal
        values.extend((comparator(target_flat, threshold).mean(0) - comparator(source_flat, threshold).mean(0)).tolist())
        values.append(float(comparator(target_luma, threshold).mean() - comparator(source_luma, threshold).mean()))
    ratio = np.clip((target_luma + epsilon) / (source_luma + epsilon), 0.0, 16.0)
    for row in np.array_split(ratio, 4, axis=0):
        values.extend(float(np.median(cell)) for cell in np.array_split(row, 4, axis=1))
    result = np.asarray(values, dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError("oracle features are non-finite")
    return result


def qwen_condition_from_scores(
    mean_nll: Mapping[int | str, float], visual_summary: Sequence[float], temperature: float = 1.0
) -> np.ndarray:
    """Canonical soft posterior plus uncertainty and frozen visual summary."""

    nll = np.asarray([float(mean_nll.get(level, mean_nll.get(str(level)))) for level in range(1, 6)], dtype=np.float64)
    logits = -(nll - nll.min()) / float(temperature)
    posterior = np.exp(logits - np.max(logits))
    posterior /= posterior.sum()
    levels = np.arange(1, 6, dtype=np.float64)
    expected = float(posterior @ levels)
    entropy = float(-(posterior * np.log(posterior + 1e-12)).sum() / math.log(5.0))
    ordered = np.sort(posterior)
    margin = float(ordered[-1] - ordered[-2])
    visual = np.asarray(visual_summary, dtype=np.float32)
    result = np.concatenate((posterior.astype(np.float32), np.asarray([expected / 5.0, entropy, margin], dtype=np.float32), visual))
    if not np.isfinite(result).all():
        raise ValueError("Qwen condition is non-finite")
    return result


def normalization(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = features.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = features.std(axis=0, dtype=np.float64).astype(np.float32)
    scale = np.maximum(scale, 1e-4)
    return mean, scale


class CausalToneAdapter(nn.Module):
    """One shared adapter for measured, oracle, Qwen, and shuffled-Qwen arms."""

    def __init__(self, measured_dim: int, qwen_dim: int, oracle_dim: int, width: int = 64) -> None:
        super().__init__()
        self.measured_dim, self.qwen_dim, self.oracle_dim = measured_dim, qwen_dim, oracle_dim
        self.measured = nn.Sequential(nn.Linear(measured_dim, width), nn.SiLU(), nn.Linear(width, width))
        self.qwen = nn.Sequential(nn.Linear(qwen_dim, width, bias=False), nn.SiLU(), nn.Linear(width, width, bias=False))
        self.oracle = nn.Sequential(nn.Linear(oracle_dim, width, bias=False), nn.SiLU(), nn.Linear(width, width, bias=False))
        self.output = nn.Sequential(nn.SiLU(), nn.Linear(width, 9))
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def forward(
        self,
        measured: Tensor,
        qwen: Tensor | None = None,
        oracle: Tensor | None = None,
    ) -> Tensor:
        if measured.ndim != 2 or measured.shape[1] != self.measured_dim:
            raise ValueError("measured condition has the wrong shape")
        value = self.measured(measured)
        if qwen is not None:
            if qwen.ndim != 2 or qwen.shape[1] != self.qwen_dim:
                raise ValueError("Qwen condition has the wrong shape")
            value = value + self.qwen(qwen)
        if oracle is not None:
            if oracle.ndim != 2 or oracle.shape[1] != self.oracle_dim:
                raise ValueError("oracle condition has the wrong shape")
            value = value + self.oracle(oracle)
        return self.output(value)


def group_derangement(rows: Sequence[Mapping[str, str]], seed: int) -> dict[str, str]:
    """Map records to a different group in the same family, deterministically."""

    rng = np.random.default_rng(seed)
    mapping: dict[str, str] = {}
    families = sorted({row["family"] for row in rows})
    for family in families:
        grouped: dict[str, list[str]] = {}
        for row in rows:
            if row["family"] == family:
                grouped.setdefault(row["group_id"], []).append(row["record_id"])
        groups = sorted(grouped)
        if len(groups) < 2:
            raise ValueError(f"cannot derange singleton family {family}")
        offset = int(rng.integers(1, len(groups)))
        for index, group in enumerate(groups):
            source_records = sorted(grouped[groups[(index + offset) % len(groups)]])
            for record_index, record_id in enumerate(sorted(grouped[group])):
                mapping[record_id] = source_records[record_index % len(source_records)]
    return mapping
