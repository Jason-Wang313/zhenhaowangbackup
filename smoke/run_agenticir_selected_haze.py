#!/usr/bin/env python3
"""Generate a five-level official AgenticIR haze smoke set."""

from pathlib import Path
import argparse
import hashlib
import sys
import random

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
AGENTICIR_DEPS = ROOT / "third_party/AgenticIR/installation/tune_depictqa"
sys.path.insert(0, str(AGENTICIR_DEPS))
from add_single_degradation import add_haze  # noqa: E402
from scipy.io import loadmat  # noqa: E402


def add_haze_strengthened(img: np.ndarray, idx: str, level: int, depth_dir: Path, beta_multiplier: float) -> np.ndarray:
    """AgenticIR haze with scaled beta, preserving the original A schedule."""
    depth = loadmat(depth_dir / idx / "predict_depth.mat")["data_obj"]
    depth = cv2.resize(depth, (0, 0), fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    depth = depth / depth.max()
    a_mean, beta_mean = [
        [0.6, 0.4], [0.7, 0.6], [0.8, 1.2], [1.0, 1.8]
    ][level - 1]
    airlight = max(np.random.normal(a_mean, 0.015), 0.4)
    beta = np.random.normal(beta_mean * beta_multiplier, 0.05 * beta_multiplier)
    transmission = np.exp(-beta * depth)[..., np.newaxis]
    return (img * transmission + airlight * 255 * (1 - transmission)).clip(0, 255).round().astype(np.uint8)


def write_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp" + path.suffix)
    if not cv2.imwrite(str(temporary), image):
        raise RuntimeError(f"failed to write {temporary}")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=int, choices=[1, 2, 3, 4])
    parser.add_argument("--strength-200", action="store_true", help="triple haze beta")
    parser.add_argument("--current-plus-100", action="store_true", help="double current beta-tripled strength")
    args = parser.parse_args()

    source = ROOT / "data/clean/002.png"
    image = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"cannot read {source}")

    seed = 2026080722
    random.seed(seed)
    np.random.seed(seed)
    if args.current_plus_100:
        output = ROOT / "smoke/agenticir_haze_selected_002_current_plus100"
    elif args.strength_200:
        output = ROOT / "smoke/agenticir_haze_selected_002_strength200"
    else:
        output = ROOT / "smoke/agenticir_haze_selected_002"
    write_png(output / "002_L1.png", image)
    print("L1: clean", flush=True)

    levels = [args.level] if args.level else range(1, 5)
    for level in levels:
        if args.current_plus_100 or args.strength_200:
            multiplier = 6.0 if args.current_plus_100 else 3.0
            result = add_haze_strengthened(image, "002", level, ROOT / "data/agenticir/haze", multiplier)
        else:
            result = add_haze(image, idx="002", level=level, depth_dir=ROOT / "data/agenticir/haze")
        write_png(output / f"002_L{level + 1}.png", result)
        suffix = " beta×6" if args.current_plus_100 else (" beta×3" if args.strength_200 else "")
        print(f"L{level + 1}: haze level={level}{suffix}", flush=True)

    for path in sorted(output.glob("*.png")):
        print(f"{path.name}: sha256={hashlib.sha256(path.read_bytes()).hexdigest()}", flush=True)
    print(f"Final images: {output}", flush=True)


if __name__ == "__main__":
    main()
