"""Small, deterministic image-space guard for Noise versus Noise+Low-light."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter


def physical_features(path: Path, maximum_side: int = 512) -> dict[str, float]:
    """Measure robust exposure and high-frequency energy inside dark pixels."""
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        rgb.thumbnail((maximum_side, maximum_side), Image.Resampling.BILINEAR)
        values = np.asarray(rgb, dtype=np.float32) / 255.0
    luma = 0.2126 * values[..., 0] + 0.7152 * values[..., 1] + 0.0722 * values[..., 2]
    p10, p25, p50, p75, p90 = np.quantile(luma, (0.10, 0.25, 0.50, 0.75, 0.90))
    exposure = 0.25 * p25 + 0.50 * p50 + 0.25 * p75

    smooth_image = Image.fromarray(np.round(luma * 255.0).astype(np.uint8)).filter(
        ImageFilter.BoxBlur(1)
    )
    smooth = np.asarray(smooth_image, dtype=np.float32) / 255.0
    residual = luma - smooth
    dark = luma <= min(float(p50), 0.35)
    if int(dark.sum()) < 64:
        dark_noise = 0.0
    else:
        centered = residual[dark] - np.median(residual[dark])
        dark_noise = float(1.4826 * np.median(np.abs(centered)))
    return {
        "exposure": float(exposure),
        "dark_noise": dark_noise,
        "p10": float(p10),
        "p50": float(p50),
        "p90": float(p90),
        "dark_fraction": float((luma < 0.15).mean()),
    }


def physical_route(
    features: dict[str, float],
    low_exposure: float,
    high_exposure: float,
    dark_noise_floor: float,
) -> str | None:
    """Return a route only outside the frozen ambiguity interval."""
    if features["exposure"] <= low_exposure:
        return "noise+low_light" if features["dark_noise"] >= dark_noise_floor else None
    if features["exposure"] >= high_exposure:
        return "noise"
    return None
