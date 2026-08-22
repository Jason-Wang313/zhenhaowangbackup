"""Deterministic full-resolution tiling for representation diagnostics."""

from __future__ import annotations

import math
from collections.abc import Callable

import cv2
import numpy as np


def tile_origins(length: int, tile_size: int, overlap: int) -> tuple[int, ...]:
    if length <= 0 or tile_size <= 0 or overlap < 0 or overlap >= tile_size:
        raise ValueError("invalid tiled-reconstruction geometry")
    if length <= tile_size:
        return (0,)
    stride = tile_size - overlap
    origins = list(range(0, length - tile_size + 1, stride))
    final = length - tile_size
    if origins[-1] != final:
        origins.append(final)
    return tuple(origins)


def _blend_window(height: int, width: int) -> np.ndarray:
    if height <= 0 or width <= 0:
        raise ValueError("blend window dimensions must be positive")
    vertical = np.hanning(height).astype(np.float32) if height > 2 else np.ones(height)
    horizontal = np.hanning(width).astype(np.float32) if width > 2 else np.ones(width)
    window = np.sqrt(np.outer(vertical, horizontal)).astype(np.float32)
    return np.maximum(window, np.float32(1e-3))


def reconstruct_tiled(
    image: np.ndarray,
    process_tile: Callable[[np.ndarray], np.ndarray],
    *,
    tile_size: int,
    overlap: int,
) -> tuple[np.ndarray, dict[str, object]]:
    """Apply a tile transform and blend, retaining exact original dimensions."""

    if image.ndim != 3 or image.shape[2] != 3 or not np.isfinite(image).all():
        raise ValueError("tiled reconstruction requires a finite RGB array")
    height, width = image.shape[:2]
    y_origins = tile_origins(height, tile_size, overlap)
    x_origins = tile_origins(width, tile_size, overlap)
    accumulation = np.zeros((height, width, 3), dtype=np.float32)
    normalization = np.zeros((height, width), dtype=np.float32)
    for y in y_origins:
        for x in x_origins:
            tile = image[y : min(y + tile_size, height), x : min(x + tile_size, width)]
            output = np.asarray(process_tile(tile), dtype=np.float32)
            if output.shape != tile.shape or not np.isfinite(output).all():
                raise RuntimeError("tile processor returned an invalid RGB array")
            window = _blend_window(tile.shape[0], tile.shape[1])
            accumulation[
                y : y + tile.shape[0], x : x + tile.shape[1]
            ] += output * window[..., None]
            normalization[y : y + tile.shape[0], x : x + tile.shape[1]] += window
    if np.any(normalization <= 0.0):
        raise RuntimeError("tile blending left uncovered pixels")
    reconstruction = accumulation / normalization[..., None]
    return reconstruction, {
        "tile_count": len(y_origins) * len(x_origins),
        "tile_y_origins": list(y_origins),
        "tile_x_origins": list(x_origins),
    }


def reconstruct_tiled_average(
    image: np.ndarray,
    process_tile: Callable[[np.ndarray], np.ndarray],
    *,
    tile_size: int,
    overlap: int,
) -> tuple[np.ndarray, dict[str, object]]:
    """Apply a tile transform and uniformly average overlapping predictions."""

    if image.ndim != 3 or image.shape[2] != 3 or not np.isfinite(image).all():
        raise ValueError("tiled reconstruction requires a finite RGB array")
    height, width = image.shape[:2]
    y_origins = tile_origins(height, tile_size, overlap)
    x_origins = tile_origins(width, tile_size, overlap)
    accumulation = np.zeros((height, width, 3), dtype=np.float32)
    normalization = np.zeros((height, width), dtype=np.float32)
    for y in y_origins:
        for x in x_origins:
            tile = image[y : min(y + tile_size, height), x : min(x + tile_size, width)]
            output = np.asarray(process_tile(tile), dtype=np.float32)
            if output.shape != tile.shape or not np.isfinite(output).all():
                raise RuntimeError("tile processor returned an invalid RGB array")
            accumulation[y : y + tile.shape[0], x : x + tile.shape[1]] += output
            normalization[y : y + tile.shape[0], x : x + tile.shape[1]] += 1.0
    if np.any(normalization <= 0.0):
        raise RuntimeError("tile blending left uncovered pixels")
    return accumulation / normalization[..., None], {
        "tile_count": len(y_origins) * len(x_origins),
        "tile_y_origins": list(y_origins),
        "tile_x_origins": list(x_origins),
        "merge": "uniform_overlap_average",
    }


def centered_crop(image: np.ndarray, maximum_size: int) -> np.ndarray:
    if (
        image.ndim != 3
        or image.shape[2] != 3
        or maximum_size <= 0
        or not np.isfinite(image).all()
    ):
        raise ValueError("centered crop requires a finite RGB array")
    height, width = image.shape[:2]
    crop_height = min(height, maximum_size)
    crop_width = min(width, maximum_size)
    y = (height - crop_height) // 2
    x = (width - crop_width) // 2
    return image[y : y + crop_height, x : x + crop_width]


def high_frequency_diagnostic(
    reference: np.ndarray,
    reconstruction: np.ndarray,
    *,
    gaussian_sigma: float = 1.2,
) -> dict[str, float]:
    if (
        reference.shape != reconstruction.shape
        or reference.ndim != 3
        or reference.shape[2] != 3
        or gaussian_sigma <= 0.0
    ):
        raise ValueError("high-frequency diagnostic requires aligned RGB arrays")

    def high_pass(image: np.ndarray) -> np.ndarray:
        luma = (
            0.2126 * image[..., 0]
            + 0.7152 * image[..., 1]
            + 0.0722 * image[..., 2]
        ).astype(np.float32)
        return luma - cv2.GaussianBlur(luma, (0, 0), gaussian_sigma)

    reference_high = high_pass(reference)
    reconstruction_high = high_pass(reconstruction)
    error = reference_high - reconstruction_high
    reference_energy = float(np.sqrt(np.mean(reference_high.astype(np.float64) ** 2)))
    reconstruction_energy = float(
        np.sqrt(np.mean(reconstruction_high.astype(np.float64) ** 2))
    )
    centered_reference = reference_high.astype(np.float64).reshape(-1)
    centered_reconstruction = reconstruction_high.astype(np.float64).reshape(-1)
    centered_reference -= centered_reference.mean()
    centered_reconstruction -= centered_reconstruction.mean()
    denominator = math.sqrt(
        float(np.dot(centered_reference, centered_reference))
        * float(np.dot(centered_reconstruction, centered_reconstruction))
    )
    correlation = (
        float(np.dot(centered_reference, centered_reconstruction) / denominator)
        if denominator > 1e-12
        else 1.0
    )
    return {
        "high_frequency_reference_rms": reference_energy,
        "high_frequency_reconstruction_rms": reconstruction_energy,
        "high_frequency_energy_ratio": (
            reconstruction_energy / reference_energy if reference_energy > 1e-12 else 1.0
        ),
        "high_frequency_error_rms": float(
            np.sqrt(np.mean(error.astype(np.float64) ** 2))
        ),
        "high_frequency_correlation": correlation,
    }


def seam_error_ratio(
    reference: np.ndarray,
    reconstruction: np.ndarray,
    tile_metadata: dict[str, object],
    *,
    band_width: int = 2,
) -> float:
    """Compare reconstruction MAE near internal tile origins with global MAE."""

    if reference.shape != reconstruction.shape or band_width <= 0:
        raise ValueError("seam diagnostic requires aligned images and a positive band")
    error = np.mean(np.abs(reference.astype(np.float64) - reconstruction), axis=2)
    mask = np.zeros(error.shape, dtype=bool)
    for y in tile_metadata["tile_y_origins"]:
        y = int(y)
        if y > 0:
            mask[max(0, y - band_width) : min(error.shape[0], y + band_width + 1)] = True
    for x in tile_metadata["tile_x_origins"]:
        x = int(x)
        if x > 0:
            mask[:, max(0, x - band_width) : min(error.shape[1], x + band_width + 1)] = True
    global_mae = float(error.mean())
    if not mask.any() or global_mae <= 1e-7:
        return 1.0
    return float(error[mask].mean() / global_mae)
