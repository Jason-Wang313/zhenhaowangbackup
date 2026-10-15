#!/usr/bin/env python3
"""Generate one five-level image with the frozen AgenticIR low-light recipe."""

from pathlib import Path
import hashlib
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
AGENTICIR_DEPS = ROOT / "third_party/AgenticIR/installation/tune_depictqa"
sys.path.insert(0, str(AGENTICIR_DEPS))
from add_single_degradation import darken  # noqa: E402


TARGETS = {"L2": 0.50, "L3": 0.38, "L4": 0.30, "L5": 0.14}
WEIGHTS = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def luma(image: np.ndarray) -> float:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
    return float(np.tensordot(rgb, WEIGHTS, axes=([-1], [0])).mean())


def write_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp" + path.suffix)
    if not cv2.imwrite(str(temporary), image):
        raise RuntimeError(f"failed to write {temporary}")
    temporary.replace(path)


def main() -> None:
    source = ROOT / "data/clean/002.png"
    if not source.is_file():
        raise RuntimeError(f"missing source: {source}")
    clean = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if clean is None:
        raise RuntimeError(f"cannot read source: {source}")

    np.random.seed(2026080610)
    raw_root = ROOT / "smoke/agenticir_low_light_selected_002_raw"
    final_root = ROOT / "smoke/agenticir_low_light_selected_002"
    clean_luma = luma(clean)

    write_png(final_root / "002_L1.png", clean)
    print(f"L1: luma={clean_luma:.2f}", flush=True)
    for level_index in range(1, 5):
        level = f"L{level_index + 1}"
        raw = darken(clean, level=level_index, darken_type="linear mapping")
        write_png(raw_root / f"002_{level}.png", raw)
        raw_luma = luma(raw)
        target = clean_luma * TARGETS[level]
        scale = target / raw_luma
        calibrated = np.rint(raw.astype(np.float32) * scale).clip(0, 255).astype(np.uint8)
        write_png(final_root / f"002_{level}.png", calibrated)
        print(f"{level}: raw={raw_luma:.2f} target={target:.2f} final={luma(calibrated):.2f} scale={scale:.4f}", flush=True)

    for path in sorted(final_root.glob("*.png")):
        print(f"{path.name}: sha256={hashlib.sha256(path.read_bytes()).hexdigest()}", flush=True)
    print(f"Final images: {final_root}", flush=True)


if __name__ == "__main__":
    main()
