#!/usr/bin/env python3
"""Generate one five-level image with the frozen AgenticIR noise recipe."""

from pathlib import Path
import hashlib
import argparse

import cv2
import numpy as np
import torch
from basicsr.data.degradations import random_add_gaussian_noise_pt


ROOT = Path(__file__).resolve().parents[1]
RANGES = [[16.848, 49.9824], [37.575, 75.15], [75, 105], [159.6, 199.5]]


def write_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp" + path.suffix)
    if not cv2.imwrite(str(temporary), image):
        raise RuntimeError(f"failed to write {temporary}")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=int, choices=[1, 2, 3, 4])
    args = parser.parse_args()
    source = ROOT / "data/clean/002.png"
    image = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"cannot read {source}")
    seed = 2026080710
    np.random.seed(seed)
    torch.manual_seed(seed)
    output = ROOT / "smoke/agenticir_noise_selected_002"
    write_png(output / "002_L1.png", image)
    print("L1: clean", flush=True)
    levels = [args.level] if args.level else range(1, 5)
    for level in levels:
        sigma_range = RANGES[level - 1]
        tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        noisy = random_add_gaussian_noise_pt(tensor, sigma_range=sigma_range, clip=True, rounds=False)
        result = (noisy.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).round().astype(np.uint8)
        write_png(output / f"002_L{level + 1}.png", result)
        print(f"L{level + 1}: sigma range {sigma_range}", flush=True)
    for path in sorted(output.glob("*.png")):
        print(f"{path.name}: sha256={hashlib.sha256(path.read_bytes()).hexdigest()}", flush=True)
    print(f"Final images: {output}", flush=True)


if __name__ == "__main__":
    main()
