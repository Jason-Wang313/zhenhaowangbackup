#!/usr/bin/env python3
"""Select one fixed Noise-only residual scale from the frozen 16-record cache."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lowlight_restoration.scalar_residual import quantized_scalar_residual  # noqa: E402
from smoke.evaluate_foundir_mixed_restoration_pilot_ddp import crop_pair  # noqa: E402
from smoke.train_foundir_blur_only_flux2_overfit_ddp import lpips_value, psnr  # noqa: E402

CONFIG = ROOT / "configs/foundir_mixed_restoration_pilot_v1.json"
MANIFEST = ROOT / "manifests/foundir_mixed_restoration_pilot_v1.csv"
CACHE = ROOT / "checkpoints/foundir_mixed_restoration_pilot_v1/stages/noise"
OUTPUT = ROOT / "checkpoints/foundir_mixed_restoration_pilot_v1/blur_noise_noise_only_alpha_selection_v1.json"
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    config = json.loads(CONFIG.read_text())
    with MANIFEST.open(newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["category"] == "02Blur_Noise"]
    if len(rows) != 16 or len({row["source_group"] for row in rows}) != 16:
        raise RuntimeError("Blur+Noise calibration cache selection changed")

    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

    metric = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).eval()
    totals = {str(alpha): [] for alpha in ALPHAS}
    for row in rows:
        lq, gt, _ = crop_pair(row, config)
        cache_path = CACHE / f"{row['record_id']}.png"
        if not cache_path.is_file():
            raise RuntimeError(f"missing Noise cache: {cache_path}")
        with Image.open(cache_path) as handle:
            array = np.asarray(handle.convert("RGB"), dtype=np.uint8).copy()
        noise = torch.from_numpy(array).permute(2, 0, 1).float().div_(127.5).sub_(1)
        for alpha in ALPHAS:
            _, candidate = quantized_scalar_residual(
                lq.unsqueeze(0), noise.unsqueeze(0), alpha
            )
            reference = gt.unsqueeze(0)
            totals[str(alpha)].append(
                {
                    "record_id": row["record_id"],
                    "psnr": psnr(reference, candidate),
                    "lpips": lpips_value(metric, reference, candidate),
                }
            )
            metric.reset()

    summaries = {}
    for alpha in ALPHAS:
        values = totals[str(alpha)]
        summaries[str(alpha)] = {
            "psnr": float(np.mean([value["psnr"] for value in values])),
            "lpips": float(np.mean([value["lpips"] for value in values])),
        }
    baseline = summaries["0.0"]
    passing = [
        alpha
        for alpha in ALPHAS
        if alpha > 0
        and summaries[str(alpha)]["psnr"] > baseline["psnr"]
        and summaries[str(alpha)]["lpips"] < baseline["lpips"]
    ]
    selected = max(passing, key=lambda alpha: summaries[str(alpha)]["psnr"]) if passing else None
    payload = {
        "schema_version": 1,
        "status": "fixed_alpha_selected" if selected is not None else "no_fixed_alpha_passed",
        "selection_split": "training_only_mixed_pilot_16_record_cache",
        "records": len(rows),
        "source_groups": len({row["source_group"] for row in rows}),
        "config_sha256": sha256(CONFIG),
        "manifest_sha256": sha256(MANIFEST),
        "candidate_alphas": ALPHAS,
        "selection_rule": "both_mean_metrics_improve_over_alpha0_then_maximize_psnr",
        "summaries": summaries,
        "selected_alpha": selected,
        "model_inference_reused": True,
        "gpu_hours": 0,
        "development_pixels_accessed": False,
        "official_foundir_test_accessed": False,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
