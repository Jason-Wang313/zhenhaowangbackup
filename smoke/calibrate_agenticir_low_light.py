#!/usr/bin/env python3
"""Calibrate official AgenticIR linear-mapping levels for visual review."""

from pathlib import Path
import shutil
import argparse

from PIL import Image

from apply_gendeg_level_remap import darken_to_target, mean_luma


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/clean/001.png"
RAW = ROOT / "smoke/agenticir_low_light_linear_mapping"
OUT = ROOT / "smoke/agenticir_low_light_linear_mapping_calibrated"
TARGETS = {"L2": 0.75, "L3": 0.60, "L4": 0.45, "L5": 0.30}
DARKER_TARGETS = {"L2": 0.60, "L3": 0.45, "L4": 0.30, "L5": 0.18}
DARKER_23_TARGETS = {"L2": 0.50, "L3": 0.38, "L4": 0.30, "L5": 0.18}
VERY_DARK_TARGETS = {"L2": 0.35, "L3": 0.18, "L4": 0.09, "L5": 0.04}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--darker", action="store_true")
    parser.add_argument("--darker-23", action="store_true")
    parser.add_argument("--very-dark", action="store_true")
    args = parser.parse_args()
    if sum([args.darker, args.darker_23, args.very_dark]) > 1:
        parser.error("choose only one darker mode")
    if not SOURCE.is_file():
        raise RuntimeError(f"missing source: {SOURCE}")
    if any(not (RAW / f"001_L{i}.png").is_file() for i in range(1, 6)):
        raise RuntimeError(f"missing AgenticIR linear-mapping inputs in {RAW}")
    output = OUT
    targets = TARGETS
    if args.darker:
        output = ROOT / "smoke/agenticir_low_light_linear_mapping_calibrated_darker"
        targets = DARKER_TARGETS
    elif args.darker_23:
        output = ROOT / "smoke/agenticir_low_light_linear_mapping_calibrated_darker_23"
        targets = DARKER_23_TARGETS
    elif args.very_dark:
        output = ROOT / "smoke/agenticir_low_light_linear_mapping_calibrated_very_dark"
        targets = VERY_DARK_TARGETS
    output.mkdir(parents=True, exist_ok=True)
    clean_luma = mean_luma(SOURCE)
    shutil.copy2(SOURCE, output / "001_L1.png")
    print(f"L1: luma={mean_luma(output / '001_L1.png'):.2f}", flush=True)
    for level, ratio in targets.items():
        target = clean_luma * ratio
        final = darken_to_target(RAW / f"001_{level}.png", output / f"001_{level}.png", target)
        print(f"{level}: target={target:.2f} final={final:.2f} ratio={final / clean_luma:.3f}", flush=True)
    print(f"Calibrated AgenticIR outputs: {output}", flush=True)


if __name__ == "__main__":
    main()
