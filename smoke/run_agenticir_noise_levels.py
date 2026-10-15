#!/usr/bin/env python3
"""Generate a five-level official AgenticIR Gaussian-noise smoke set."""

from pathlib import Path
import hashlib
import random
import sys
import argparse

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
AGENTICIR_DEPS = ROOT / "third_party/AgenticIR/installation/tune_depictqa"
sys.path.insert(0, str(AGENTICIR_DEPS))
from add_single_degradation import add_noise  # noqa: E402
from basicsr.data.degradations import random_add_gaussian_noise_pt  # noqa: E402


ORIGINAL_RANGES = [[2, 6], [10, 20], [25, 35], [40, 50]]
STRONGER_RANGES = [[2.2, 6.6], [11.5, 23], [31.25, 43.75], [56, 70]]
VERY_STRONGER_RANGES = [[3.0, 8.9], [16.7, 33.4], [50, 70], [106.4, 133]]
EXTREME_RANGES = [[4.5, 13.35], [25.05, 50.1], [75, 105], [159.6, 199.5]]
CUSTOM_RANGES = [[9, 26.7], [37.575, 75.15], [75, 105], [159.6, 199.5]]
CUSTOM_L2_RANGES = [[11.7, 34.71], [37.575, 75.15], [75, 105], [159.6, 199.5]]
CUSTOM_L2_50_RANGES = [[17.55, 52.065], [37.575, 75.15], [75, 105], [159.6, 199.5]]
CUSTOM_L2_50_50_RANGES = [[26.325, 78.0975], [37.575, 75.15], [75, 105], [159.6, 199.5]]
CUSTOM_L2_50_50_DOWN20_RANGES = [[21.06, 62.478], [37.575, 75.15], [75, 105], [159.6, 199.5]]
CUSTOM_L2_DOWN20_RANGES = [[16.848, 49.9824], [37.575, 75.15], [75, 105], [159.6, 199.5]]


def write_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp" + path.suffix)
    if not cv2.imwrite(str(temporary), image):
        raise RuntimeError(f"failed to write {temporary}")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=int, choices=[1, 2, 3, 4])
    parser.add_argument("--stronger", action="store_true")
    parser.add_argument("--very-stronger", action="store_true")
    parser.add_argument("--extreme", action="store_true")
    parser.add_argument("--custom", action="store_true")
    parser.add_argument("--custom-l2", action="store_true")
    parser.add_argument("--custom-l2-50", action="store_true")
    parser.add_argument("--custom-l2-50-50", action="store_true")
    parser.add_argument("--custom-l2-down20", action="store_true")
    parser.add_argument("--custom-l2-down20-again", action="store_true")
    args = parser.parse_args()
    if sum([args.stronger, args.very_stronger, args.extreme, args.custom, args.custom_l2, args.custom_l2_50, args.custom_l2_50_50, args.custom_l2_down20, args.custom_l2_down20_again]) > 1:
        parser.error("choose only one stronger mode")
    source = ROOT / "data/clean/001.png"
    image = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"cannot read {source}")
    seed = 2026080700
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if args.custom_l2_down20_again:
        output = ROOT / "smoke/agenticir_noise_levels_001_custom_l2_down20_again"
    elif args.custom_l2_down20:
        output = ROOT / "smoke/agenticir_noise_levels_001_custom_l2_down20"
    elif args.custom_l2_50_50:
        output = ROOT / "smoke/agenticir_noise_levels_001_custom_l2_50_50"
    elif args.custom_l2_50:
        output = ROOT / "smoke/agenticir_noise_levels_001_custom_l2_50"
    elif args.custom_l2:
        output = ROOT / "smoke/agenticir_noise_levels_001_custom_l2"
    elif args.custom:
        output = ROOT / "smoke/agenticir_noise_levels_001_custom"
    elif args.extreme:
        output = ROOT / "smoke/agenticir_noise_levels_001_extreme"
    elif args.very_stronger:
        output = ROOT / "smoke/agenticir_noise_levels_001_very_stronger"
    elif args.stronger:
        output = ROOT / "smoke/agenticir_noise_levels_001_stronger"
    else:
        output = ROOT / "smoke/agenticir_noise_levels_001"
    write_png(output / "001_L1.png", image)
    print("L1: clean", flush=True)
    levels = [args.level] if args.level else range(1, 5)
    for level in levels:
        if args.stronger or args.very_stronger or args.extreme or args.custom or args.custom_l2 or args.custom_l2_50 or args.custom_l2_50_50 or args.custom_l2_down20 or args.custom_l2_down20_again:
            ranges = CUSTOM_L2_DOWN20_RANGES if args.custom_l2_down20_again else (CUSTOM_L2_50_50_DOWN20_RANGES if args.custom_l2_down20 else (CUSTOM_L2_50_50_RANGES if args.custom_l2_50_50 else (CUSTOM_L2_50_RANGES if args.custom_l2_50 else (CUSTOM_L2_RANGES if args.custom_l2 else (CUSTOM_RANGES if args.custom else (EXTREME_RANGES if args.extreme else (VERY_STRONGER_RANGES if args.very_stronger else STRONGER_RANGES)))))))
            tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            result_tensor = random_add_gaussian_noise_pt(
                tensor,
                sigma_range=ranges[level - 1],
                clip=True,
                rounds=False,
            )
            result = (result_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).round().astype(np.uint8)
        else:
            result = add_noise(image, level=level, noise_type="Gaussian")
        write_png(output / f"001_L{level + 1}.png", result)
        ranges = CUSTOM_L2_DOWN20_RANGES if args.custom_l2_down20_again else (CUSTOM_L2_50_50_DOWN20_RANGES if args.custom_l2_down20 else (CUSTOM_L2_50_50_RANGES if args.custom_l2_50_50 else (CUSTOM_L2_50_RANGES if args.custom_l2_50 else (CUSTOM_L2_RANGES if args.custom_l2 else (CUSTOM_RANGES if args.custom else (EXTREME_RANGES if args.extreme else (VERY_STRONGER_RANGES if args.very_stronger else (STRONGER_RANGES if args.stronger else ORIGINAL_RANGES))))))))
        print(f"L{level + 1}: Gaussian sigma range {ranges[level-1]}", flush=True)
    for path in sorted(output.glob("*.png")):
        print(f"{path.name}: sha256={hashlib.sha256(path.read_bytes()).hexdigest()}", flush=True)
    print(f"Noise level images: {output}", flush=True)


if __name__ == "__main__":
    main()
