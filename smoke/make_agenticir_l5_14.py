from pathlib import Path
import shutil

from PIL import Image
import numpy as np

from apply_gendeg_level_remap import mean_luma

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/clean/001.png"
CURRENT = ROOT / "smoke/agenticir_low_light_linear_mapping_calibrated_darker_23"
OUTPUT = ROOT / "smoke/agenticir_low_light_linear_mapping_calibrated_l5_14"

OUTPUT.mkdir(parents=True, exist_ok=True)
for level in ("L1", "L2", "L3", "L4"):
    shutil.copy2(CURRENT / f"001_{level}.png", OUTPUT / f"001_{level}.png")

clean_luma = mean_luma(SOURCE)
target_luma = clean_luma * 0.14
source = CURRENT / "001_L5.png"
with Image.open(source).convert("RGB") as image:
    pixels = np.asarray(image, dtype=np.float32)
    weights = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    current_luma = float(np.tensordot(pixels, weights, axes=([-1], [0])).mean())
    scaled = np.rint(pixels * (target_luma / current_luma)).clip(0, 255).astype(np.uint8)
    Image.fromarray(scaled, mode="RGB").save(OUTPUT / "001_L5.png", format="PNG")

print(f"clean={clean_luma:.2f} target_L5={target_luma:.2f} final_L5={mean_luma(OUTPUT / '001_L5.png'):.2f}")
print(OUTPUT)
