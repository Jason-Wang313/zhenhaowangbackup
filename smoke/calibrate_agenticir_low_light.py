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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--darker", action="store_true")
    args = parser.parse_args()
    if not SOURCE.is_file():
        raise RuntimeError(f"missing source: {SOURCE}")
    if any(not (RAW / f"001_L{i}.png").is_file() for i in range(1, 6)):
        raise RuntimeError(f"missing AgenticIR linear-mapping inputs in {RAW}")
    output = (ROOT / "smoke/agenticir_low_light_linear_mapping_calibrated_darker") if args.darker else OUT
    targets = DARKER_TARGETS if args.darker else TARGETS
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
