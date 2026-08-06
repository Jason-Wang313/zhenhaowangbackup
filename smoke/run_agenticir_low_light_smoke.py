#!/usr/bin/env python3
"""Generate a five-level AgenticIR official dark-degradation smoke set."""

from __future__ import annotations

import argparse
import hashlib
import random
import sys
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
AGENTICIR_DEPS = ROOT / "third_party/AgenticIR/installation/tune_depictqa"
sys.path.insert(0, str(AGENTICIR_DEPS))
from add_single_degradation import darken  # noqa: E402


def luma(img: np.ndarray) -> float:
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    return float((rgb * np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)).sum(2).mean())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="001.png")
    parser.add_argument("--darken-type", default="gamma correction", choices=["constant shift", "gamma correction", "linear mapping"])
    parser.add_argument("--seed", type=int, default=2026080600)
    args = parser.parse_args()

    source = ROOT / "data/clean" / args.source
    if not source.is_file():
        raise RuntimeError(f"missing source: {source}")
    image = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"cannot read source: {source}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    method_slug = args.darken_type.replace(" ", "_")
    output = ROOT / ("smoke/agenticir_low_light" if method_slug == "gamma_correction" else f"smoke/agenticir_low_light_{method_slug}")
    output.mkdir(parents=True, exist_ok=True)

    levels = {"L1": image}
    for level in range(1, 5):
        levels[f"L{level + 1}"] = darken(image, level=level, darken_type=args.darken_type)

    for name, result in levels.items():
        path = output / f"{source.stem}_{name}.png"
        if not cv2.imwrite(str(path), result):
            raise RuntimeError(f"failed to write {path}")
        print(f"{name}: luma={luma(result):.2f} size={result.shape[1]}x{result.shape[0]} sha256={hashlib.sha256(path.read_bytes()).hexdigest()[:16]}", flush=True)
    print(f"AgenticIR low-light smoke outputs: {output}", flush=True)


if __name__ == "__main__":
    main()
