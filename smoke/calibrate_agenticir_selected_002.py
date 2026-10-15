from pathlib import Path
import shutil

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/clean/002.png"
RAW = ROOT / "smoke/agenticir_low_light_linear_mapping"
OUT = ROOT / "smoke/agenticir_low_light_selected_002"
TARGETS = {"L2": 0.50, "L3": 0.38, "L4": 0.30, "L5": 0.14}
WEIGHTS = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def luma(array: np.ndarray) -> float:
    return float(np.tensordot(array.astype(np.float32), WEIGHTS, axes=([-1], [0])).mean())


def save(path: Path, array: np.ndarray) -> None:
    temporary = path.with_name(path.stem + ".tmp" + path.suffix)
    Image.fromarray(array.astype(np.uint8), mode="RGB").save(temporary, format="PNG")
    temporary.replace(path)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    clean = np.asarray(Image.open(SOURCE).convert("RGB"))
    clean_luma = luma(clean)
    shutil.copy2(SOURCE, OUT / "002_L1.png")
    print(f"L1: luma={clean_luma:.2f}", flush=True)
    for level, ratio in TARGETS.items():
        raw = np.asarray(Image.open(RAW / f"002_{level}.png").convert("RGB"))
        raw_luma = luma(raw)
        target = clean_luma * ratio
        scale = target / raw_luma
        final = np.rint(raw.astype(np.float32) * scale).clip(0, 255).astype(np.uint8)
        save(OUT / f"002_{level}.png", final)
        print(f"{level}: raw={raw_luma:.2f} target={target:.2f} final={luma(final):.2f} scale={scale:.4f}", flush=True)
    print(f"Final images: {OUT}", flush=True)


if __name__ == "__main__":
    main()
