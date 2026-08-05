"""Run official AgenticIR one-image degradation smoke tests.

The functions are imported unchanged from third_party/AgenticIR.  This script
only fixes seeds, supplies the official matching example depth map for haze,
and records output hashes for repeatability.
"""

from pathlib import Path
import hashlib
import random

import cv2
import numpy as np
import torch

from dataset.add_single_degradation import (
    add_haze,
    add_motion_blur,
    add_noise,
    add_rain,
)


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "third_party/AgenticIR/dataset/HQ/example.png"
DEPTH = ROOT / "third_party/AgenticIR/dataset/depth"
OUT = ROOT / "smoke/agenticir"
OUT.mkdir(parents=True, exist_ok=True)


def run_once(tag: str) -> dict:
    image = cv2.imread(str(INPUT), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"cannot read {INPUT}")
    cases = {
        "noise": lambda: add_noise(image, noise_type="Gaussian", arg=25),
        "blur": lambda: add_motion_blur(image, severity=1),
        "haze": lambda: add_haze(image, idx="example", depth_dir=DEPTH, A=0.85, beta=1.0),
        "rain": lambda: add_rain(image, value=70),
    }
    results = {}
    for name, fn in cases.items():
        seed = 20260804 + list(cases).index(name)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        output = fn()
        if output.shape != image.shape or output.dtype != np.uint8:
            raise RuntimeError(f"{name}: bad output {output.shape} {output.dtype}")
        path = OUT / f"{name}_{tag}.png"
        if not cv2.imwrite(str(path), output) or path.stat().st_size == 0:
            raise RuntimeError(f"{name}: failed to write {path}")
        decoded = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if decoded is None or decoded.shape != image.shape or not np.any(decoded):
            raise RuntimeError(f"{name}: output is unreadable/empty")
        results[name] = {
            "path": str(path.relative_to(ROOT)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size,
            "shape": list(decoded.shape),
        }
    return results


first = run_once("run1")
second = run_once("run2")
for name in first:
    if first[name]["sha256"] != second[name]["sha256"]:
        raise RuntimeError(f"{name}: nondeterministic output")
print({"input": str(INPUT.relative_to(ROOT)), "outputs": first, "deterministic": True})
