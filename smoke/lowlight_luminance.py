"""Shared luminance helpers for the approved AgenticIR Lowlight recipe."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def mean_luma(path: Path) -> float:
    """Return mean Rec. 709 luma for an RGB image in the 0-255 scale."""
    with Image.open(path).convert("RGB") as image:
        pixels = np.asarray(image, dtype=np.float32)
    weights = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    return float(np.tensordot(pixels, weights, axes=([-1], [0])).mean())


def darken_to_target(source: Path, target: Path, target_luma: float) -> float:
    """Apply a measured global exposure scale and return resulting luminance."""
    with Image.open(source).convert("RGB") as image:
        pixels = np.asarray(image, dtype=np.float32)
        weights = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
        raw = float(np.tensordot(pixels, weights, axes=([-1], [0])).mean())
        if raw <= 0.001:
            output = image.copy()
        else:
            desired = max(0.5, min(255.0, target_luma))
            values = np.rint(pixels * (desired / raw)).clip(0, 255).astype(np.uint8)
            output = Image.fromarray(values, mode="RGB")
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        output.save(temporary, format="PNG")
        temporary.replace(target)
    return mean_luma(target)
