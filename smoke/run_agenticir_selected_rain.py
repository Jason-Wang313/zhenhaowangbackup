#!/usr/bin/env python3
"""Generate a five-level official AgenticIR rain smoke set."""

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
from add_single_degradation import add_rain  # noqa: E402


def add_rain_custom(img: np.ndarray, mean_value: float, std: float) -> np.ndarray:
    """Official AgenticIR rain with a custom rain-density distribution."""
    img = img.copy()
    width = 3
    length = np.random.randint(20, 40)
    angle = np.random.randint(-30, 30)
    value = max(np.random.normal(mean_value, std), 3)
    noise = np.random.uniform(0, 256, img.shape[0:2])
    noise[noise < (256 - value * 0.01)] = 0
    kernel = np.array([[0, 0.1, 0], [0.1, 8, 0.1], [0, 0.1, 0]])
    noise = cv2.filter2D(noise, -1, kernel)
    transform = cv2.getRotationMatrix2D((length / 2, length / 2), angle - 45, 1 - length / 100.0)
    diagonal = np.diag(np.ones(length))
    kernel = cv2.warpAffine(diagonal, transform, (length, length))
    kernel = cv2.GaussianBlur(kernel, (width, width), 0)
    blurred = cv2.filter2D(noise, -1, kernel)
    cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
    rain = np.repeat(blurred.astype(np.uint8)[..., np.newaxis], 3, axis=2)
    result = img.astype("float32") + rain
    np.clip(result, 0, 255, out=result)
    return result.round().astype(np.uint8)


def write_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp" + path.suffix)
    if not cv2.imwrite(str(temporary), image):
        raise RuntimeError(f"failed to write {temporary}")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=int, choices=[1, 2, 3, 4])
    parser.add_argument("--l5-plus300", action="store_true")
    parser.add_argument("--l4-plus200-l5-plus300", action="store_true")
    parser.add_argument("--l4-150-l5-340", action="store_true")
    args = parser.parse_args()

    source = ROOT / "data/clean/002.png"
    image = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"cannot read {source}")

    seed = 2026080723
    random.seed(seed)
    np.random.seed(seed)
    if args.l4_150_l5_340:
        output = ROOT / "smoke/agenticir_rain_selected_002_l4_150_l5_340"
    elif args.l4_plus200_l5_plus300:
        output = ROOT / "smoke/agenticir_rain_selected_002_l4_plus200_l5_plus300"
    elif args.l5_plus300:
        output = ROOT / "smoke/agenticir_rain_selected_002_l5_plus300"
    else:
        output = ROOT / "smoke/agenticir_rain_selected_002"
    write_png(output / "002_L1.png", image)
    print("L1: clean", flush=True)

    levels = [args.level] if args.level else range(1, 5)
    for level in levels:
        result = add_rain(image, level=level)
        write_png(output / f"002_L{level + 1}.png", result)
        print(f"L{level + 1}: AgenticIR rain level={level}", flush=True)

    if args.l4_150_l5_340:
        result = add_rain_custom(image, mean_value=150, std=10)
        write_png(output / "002_L4.png", result)
        print("L4: custom rain value mean=150", flush=True)
        result = add_rain_custom(image, mean_value=340, std=20)
        write_png(output / "002_L5.png", result)
        print("L5: custom rain value mean=340", flush=True)
    elif args.l4_plus200_l5_plus300:
        result = add_rain_custom(image, mean_value=75 * 3.0, std=5 * 3.0)
        write_png(output / "002_L4.png", result)
        print("L4: AgenticIR rain level=3, rain value mean x3", flush=True)
        result = add_rain_custom(image, mean_value=85 * 4.0, std=5 * 4.0)
        write_png(output / "002_L5.png", result)
        print("L5: AgenticIR rain level=4, rain value mean x4", flush=True)
    elif args.l5_plus300:
        result = add_rain_custom(image, mean_value=85 * 4.0, std=5 * 4.0)
        write_png(output / "002_L5.png", result)
        print("L5: AgenticIR rain level=4, rain value mean x4", flush=True)

    for path in sorted(output.glob("*.png")):
        print(f"{path.name}: sha256={hashlib.sha256(path.read_bytes()).hexdigest()}", flush=True)
    print(f"Final images: {output}", flush=True)


if __name__ == "__main__":
    main()
