#!/usr/bin/env python3
"""Generate one five-level official AgenticIR motion-blur smoke set."""

from pathlib import Path
import argparse
import hashlib
import sys
import random
import math

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
AGENTICIR_DEPS = ROOT / "third_party/AgenticIR/installation/tune_depictqa"
sys.path.insert(0, str(AGENTICIR_DEPS))
from add_single_degradation import add_motion_blur  # noqa: E402


def add_motion_blur_custom(img: np.ndarray, radius: int, sigma: float) -> np.ndarray:
    angle = np.random.uniform(-90, 90)
    width = radius * 2 + 1
    k = (np.exp(-np.arange(width)**2 / (2 * sigma**2)) / (np.sqrt(2 * np.pi) * sigma))
    kernel = k / np.sum(k)
    point = (width * np.sin(np.deg2rad(angle)), width * np.cos(np.deg2rad(angle)))
    hypot = math.hypot(point[0], point[1])
    blurred = np.zeros_like(img, dtype=np.float32)
    for i in range(width):
        dy = -math.ceil(((i * point[0]) / hypot) - 0.5)
        dx = -math.ceil(((i * point[1]) / hypot) - 0.5)
        if abs(dy) >= img.shape[0] or abs(dx) >= img.shape[1]:
            break
        if dx < 0:
            shifted = np.roll(img, shift=img.shape[1] + dx, axis=1)
            shifted[:, dx:] = shifted[:, dx - 1:dx]
        elif dx > 0:
            shifted = np.roll(img, shift=dx, axis=1)
            shifted[:, :dx] = shifted[:, dx:dx + 1]
        else:
            shifted = img
        if dy < 0:
            shifted = np.roll(shifted, shift=img.shape[0] + dy, axis=0)
            shifted[dy:, :] = shifted[dy - 1:dy, :]
        elif dy > 0:
            shifted = np.roll(shifted, shift=dy, axis=0)
            shifted[:dy, :] = shifted[dy:dy + 1, :]
        blurred += kernel[i] * shifted
    return np.clip(blurred, 0, 255).round().astype(np.uint8)


def write_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp" + path.suffix)
    if not cv2.imwrite(str(temporary), image):
        raise RuntimeError(f"failed to write {temporary}")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=int, choices=[1, 2, 3, 4])
    parser.add_argument("--double", action="store_true")
    parser.add_argument("--sigma-200", action="store_true", help="triple the doubled-set sigma values")
    args = parser.parse_args()
    source = ROOT / "data/clean/002.png"
    image = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"cannot read {source}")
    seed = 2026080720
    random.seed(seed)
    np.random.seed(seed)
    if args.sigma_200:
        output = ROOT / "smoke/agenticir_blur_selected_002_sigma200"
    elif args.double:
        output = ROOT / "smoke/agenticir_blur_selected_002_doubled"
    else:
        output = ROOT / "smoke/agenticir_blur_selected_002"
    write_png(output / "002_L1.png", image)
    print("L1: clean", flush=True)
    levels = [args.level] if args.level else range(1, 5)
    if args.sigma_200:
        params = [(10, 12), (20, 18), (30, 30), (30, 48)]
    elif args.double:
        params = [(10, 4), (20, 6), (30, 10), (30, 16)]
    else:
        params = [(5, 2), (10, 3), (15, 5), (15, 8)]
    for level in levels:
        result = add_motion_blur_custom(image, *params[level - 1]) if (args.double or args.sigma_200) else add_motion_blur(image, level=level)
        write_png(output / f"002_L{level + 1}.png", result)
        print(f"L{level + 1}: motion-blur radius,sigma={params[level-1]}", flush=True)
    for path in sorted(output.glob("*.png")):
        print(f"{path.name}: sha256={hashlib.sha256(path.read_bytes()).hexdigest()}", flush=True)
    print(f"Final images: {output}", flush=True)


if __name__ == "__main__":
    main()
