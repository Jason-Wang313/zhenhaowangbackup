#!/usr/bin/env python3
"""Estimate official uncompressed CURE HDF5 storage before building it."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
VIEW = ROOT / "data/third_party/cure_two_track_v1/paired_train/main_data/clear"


def estimate(view: Path, patch_size: int = 256, stride: int = 200) -> dict:
    files = sorted(view.glob("*.png"))
    if not files:
        raise RuntimeError(f"no paired training sources: {view}")
    patch_count = 0
    for path in files:
        with Image.open(path) as image:
            width, height = image.size
            image.verify()
        if min(width, height) < patch_size:
            raise RuntimeError(f"source smaller than patch size: {path}")
        rows = max(1, math.ceil((height - patch_size) / stride + 1))
        columns = max(1, math.ceil((width - patch_size) / stride + 1))
        patch_count += rows * columns
    image_bytes = 3 * patch_size * patch_size * 4
    # Each official HDF5 row stores all class images as float32 arrays.
    breakdown = {
        "published_baseline": patch_count * 12 * image_bytes,
        "strict_baseline": patch_count * 10 * image_bytes,
        "shared_cure_finetune": patch_count * 38 * image_bytes,
    }
    total = sum(breakdown.values())
    free = shutil.disk_usage(view).free
    return {
        "source_count": len(files),
        "patch_count": patch_count,
        "uncompressed_h5_bytes_lower_bound": total,
        "breakdown_bytes": breakdown,
        "filesystem_free_bytes": free,
        "passes_free_space_with_25pct_headroom": free >= math.ceil(total * 1.25),
        "warning": "Filesystem free space is not the project quota; verify quota separately before building.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--view", type=Path, default=VIEW)
    args = parser.parse_args()
    result = estimate(args.view)
    print(json.dumps(result, indent=2))
    if not result["passes_free_space_with_25pct_headroom"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
