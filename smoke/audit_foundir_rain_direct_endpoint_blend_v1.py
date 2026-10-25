#!/usr/bin/env python3
"""Read-only PSNR audit of saved Rain FLUX triptychs; never a model gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "checkpoints/foundir_rain_flux2_direct_endpoint_distillation_formal_v1"
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)


def psnr(target: np.ndarray, output: np.ndarray) -> float:
    mse = float(np.mean((target - output) ** 2, dtype=np.float64))
    return -10.0 * np.log10(max(mse, 1e-12))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=int, default=2000, choices=(250, 500, 1000, 2000, 3000))
    args = parser.parse_args()
    summary = json.loads((RUN / "result.json").read_text())
    if summary["official_foundir_test_accessed"] or summary["sealed_internal_accessed"]:
        raise RuntimeError("protected-data flag is set")
    snapshot = RUN / "validation_triptychs" / f"step_{args.step:06d}"
    files = sorted(snapshot.glob("*.png"))
    if len(files) != 100:
        raise RuntimeError(f"expected 100 frozen development triptychs, got {len(files)}")
    records: list[dict[str, object]] = []
    for path in files:
        rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)[28:] / 255.0
        height, width, _ = rgb.shape
        if height != 512 or width != 1536:
            raise RuntimeError(f"unexpected triptych shape: {path} {rgb.shape}")
        observation, output, target = np.split(rgb, 3, axis=1)
        values = [psnr(target, (1.0 - alpha) * observation + alpha * output) for alpha in ALPHAS]
        records.append({"record_id": path.stem.split("_", 1)[1], "psnr_by_alpha": values})
    means = np.mean([record["psnr_by_alpha"] for record in records], axis=0)
    oracle = np.mean([max(record["psnr_by_alpha"]) for record in records])
    print(json.dumps({
        "status": "exploratory_saved_export_audit_not_model_performance",
        "step": args.step,
        "records": len(records),
        "metric": "mean_per_image_psnr_on_quantized_png_triptychs",
        "alpha_values": ALPHAS,
        "mean_psnr_by_alpha": means.tolist(),
        "best_fixed_alpha": ALPHAS[int(np.argmax(means))],
        "best_fixed_psnr": float(np.max(means)),
        "gt_only_per_image_oracle_psnr": float(oracle),
        "official_or_sealed_test_accessed": False,
    }, indent=2))


if __name__ == "__main__":
    main()
