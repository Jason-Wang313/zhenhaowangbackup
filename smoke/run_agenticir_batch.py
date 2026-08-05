"""Generate the required 100-image AgenticIR batch with official functions."""

from pathlib import Path
import csv
import hashlib
import random
import time

import cv2
import numpy as np
import torch

from dataset.add_single_degradation import add_haze, add_motion_blur, add_noise, add_rain


ROOT = Path(__file__).resolve().parents[1]
clean_dir = ROOT / "data/clean"
depth_dir = ROOT / "data/agenticir/haze"
output_root = ROOT / "data/agenticir"
manifest = ROOT / "manifests/agenticir_100x4.csv"
rows = []
clean_paths = sorted(clean_dir.glob("*.png"))
if len(clean_paths) != 100:
    raise RuntimeError(f"expected 100 clean images, found {len(clean_paths)}")

for index, source_path in enumerate(clean_paths):
    image_id = source_path.stem
    image = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise RuntimeError(f"unreadable source: {source_path}")
    cases = [
        ("noise", lambda: add_noise(image, noise_type="Gaussian", arg=25), {"noise_type": "Gaussian", "sigma": 25}),
        ("blur", lambda: add_motion_blur(image, severity=index % 3), {"blur_type": "motion", "severity": index % 3}),
        ("haze", lambda: add_haze(image, image_id, depth_dir=depth_dir, A=0.85, beta=1.0), {"A": 0.85, "beta": 1.0}),
        ("rain", lambda: add_rain(image, value=70), {"value": 70}),
    ]
    for type_index, (kind, fn, params) in enumerate(cases):
        seed = 2026080400 + index * 10 + type_index
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        output = fn()
        if output.shape != image.shape or output.dtype != np.uint8 or not np.any(output):
            raise RuntimeError(f"invalid {kind} output for {image_id}")
        destination = output_root / kind / f"{image_id}.png"
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(destination), output):
            raise RuntimeError(f"write failed: {destination}")
        decoded = cv2.imread(str(destination), cv2.IMREAD_COLOR)
        if decoded is None or decoded.shape != image.shape or not np.any(decoded):
            raise RuntimeError(f"validation failed: {destination}")
        rows.append({
            "image_id": image_id,
            "source": str(source_path.relative_to(ROOT)),
            "kind": kind,
            "output": str(destination.relative_to(ROOT)),
            "seed": seed,
            "params": repr(params),
            "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
            "bytes": destination.stat().st_size,
            "height": decoded.shape[0],
            "width": decoded.shape[1],
        })
    if (index + 1) % 10 == 0:
        print(f"completed {index + 1}/100 sources", flush=True)

with manifest.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print(f"wrote {len(rows)} rows to {manifest.relative_to(ROOT)}")
