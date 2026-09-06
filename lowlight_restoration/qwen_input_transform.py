"""Deterministic, label-neutral input transforms for Qwen replay records."""

from __future__ import annotations

import io
from collections.abc import Mapping

from PIL import Image


def apply_qwen_input_transform(
    image: Image.Image, transform: Mapping[str, object] | None
) -> Image.Image:
    """Apply a frozen transform after the target degradation pipeline.

    JPEG is deliberately decoded back to RGB in memory.  It remains a
    non-target nuisance: the five degradation labels are not changed.
    """

    rgb = image.convert("RGB")
    if transform is None:
        return rgb
    family = str(transform.get("family", "none"))
    if family == "none":
        return rgb
    if family != "jpeg":
        raise ValueError(f"unsupported Qwen input transform: {family}")
    if transform.get("order") != "after_target_degradations":
        raise ValueError("post-codec JPEG must follow target degradations")
    quality = transform.get("quality")
    subsampling = transform.get("subsampling", 2)
    if (
        isinstance(quality, bool)
        or not isinstance(quality, int)
        or not 1 <= quality <= 100
        or isinstance(subsampling, bool)
        or not isinstance(subsampling, int)
        or subsampling not in (0, 1, 2)
    ):
        raise ValueError("invalid deterministic JPEG parameters")
    encoded = io.BytesIO()
    rgb.save(
        encoded,
        format="JPEG",
        quality=quality,
        subsampling=subsampling,
        optimize=False,
        progressive=False,
    )
    encoded.seek(0)
    with Image.open(encoded) as decoded:
        return decoded.convert("RGB").copy()
