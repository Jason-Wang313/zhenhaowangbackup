#!/usr/bin/env python3
"""Fit one calibration-only RGB affine map and gate it on Haze validation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "checkpoints/foundir_four_task_flux2_capacity_sanity_v1/haze"
DEFAULT_OUTPUT = ROOT / "checkpoints/foundir_haze_global_rgb_affine_gate_v1"
PSNR_FLOOR = 22.1
LPIPS_CAP = 0.19


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_triptych(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    image = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    if image.shape != (540, 1536, 3):
        raise ValueError(f"unexpected triptych shape for {path}: {image.shape}")
    panels = tuple(image[28:540, offset : offset + 512].copy() for offset in (0, 512, 1024))
    return panels  # type: ignore[return-value]


def fit_rgb_affine(pairs: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares GT ~= scale * FLUX + bias, bounded to a mild global edit."""
    sum_x = np.zeros(3, dtype=np.float64)
    sum_y = np.zeros(3, dtype=np.float64)
    sum_xx = np.zeros(3, dtype=np.float64)
    sum_xy = np.zeros(3, dtype=np.float64)
    count = 0
    for output, gt in pairs:
        x = output.reshape(-1, 3).astype(np.float64) / 255.0
        y = gt.reshape(-1, 3).astype(np.float64) / 255.0
        sum_x += x.sum(axis=0)
        sum_y += y.sum(axis=0)
        sum_xx += np.square(x).sum(axis=0)
        sum_xy += (x * y).sum(axis=0)
        count += x.shape[0]
    mean_x = sum_x / count
    mean_y = sum_y / count
    covariance = sum_xy / count - mean_x * mean_y
    variance = sum_xx / count - np.square(mean_x)
    scale = covariance / np.maximum(variance, 1e-12)
    bias = mean_y - scale * mean_x
    # Keep the correction photometric and conservative; do not permit a fit to
    # turn into an unrestricted color remapping on sixteen images.
    scale = np.clip(scale, 0.8, 1.2)
    bias = np.clip(bias, -0.1, 0.1)
    return scale, bias


def apply_rgb_affine(image: np.ndarray, scale: np.ndarray, bias: np.ndarray) -> np.ndarray:
    value = image.astype(np.float64) / 255.0
    value = np.clip(value * scale.reshape(1, 1, 3) + bias.reshape(1, 1, 3), 0.0, 1.0)
    return np.rint(value * 255.0).astype(np.uint8)


def psnr(candidate: np.ndarray, gt: np.ndarray) -> float:
    mse = float(np.mean(np.square(candidate.astype(np.float64) / 255.0 - gt.astype(np.float64) / 255.0)))
    return 100.0 if mse <= 1e-12 else -10.0 * math.log10(mse)


def tensor(image: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(image.copy()).permute(2, 0, 1).float().div_(255.0).unsqueeze(0)


def save_triptych(path: Path, record_id: str, lq: np.ndarray, candidate: np.ndarray, gt: np.ndarray) -> None:
    canvas = Image.new("RGB", (1536, 540), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (label, image) in enumerate((("Input", lq), ("Affine Candidate", candidate), ("Ground Truth", gt))):
        x = index * 512
        canvas.paste(Image.fromarray(image, mode="RGB"), (x, 28))
        draw.text((x + 6, 7), f"{record_id} | {label}", fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def evaluate(paths: list[Path], scale: np.ndarray, bias: np.ndarray, metric, visual_dir: Path | None) -> dict:
    rows = []
    for path in paths:
        lq, output, gt = load_triptych(path)
        candidate = apply_rgb_affine(output, scale, bias)
        with torch.inference_mode():
            candidate_lpips = float(metric(tensor(candidate), tensor(gt)).item())
            base_lpips = float(metric(tensor(output), tensor(gt)).item())
        record_id = path.stem.split("_", 1)[1]
        row = {
            "record_id": record_id,
            "base_psnr": psnr(output, gt),
            "base_lpips": base_lpips,
            "candidate_psnr": psnr(candidate, gt),
            "candidate_lpips": candidate_lpips,
        }
        row["psnr_delta_vs_base"] = row["candidate_psnr"] - row["base_psnr"]
        row["lpips_improvement_vs_base"] = row["base_lpips"] - row["candidate_lpips"]
        rows.append(row)
        if visual_dir is not None:
            save_triptych(visual_dir / path.name, record_id, lq, candidate, gt)
    aggregate = {
        "records": len(rows),
        "base_psnr": float(np.mean([row["base_psnr"] for row in rows])),
        "base_lpips": float(np.mean([row["base_lpips"] for row in rows])),
        "candidate_psnr": float(np.mean([row["candidate_psnr"] for row in rows])),
        "candidate_lpips": float(np.mean([row["candidate_lpips"] for row in rows])),
        "rows": rows,
    }
    aggregate["psnr_delta_vs_base"] = aggregate["candidate_psnr"] - aggregate["base_psnr"]
    aggregate["lpips_improvement_vs_base"] = aggregate["base_lpips"] - aggregate["candidate_lpips"]
    return aggregate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    calibration_paths = sorted((args.input / "training_triptychs/step_000400").glob("*.png"))
    validation_paths = sorted((args.input / "validation_triptychs/step_000400").glob("*.png"))
    if len(calibration_paths) != 16 or len(validation_paths) != 16:
        raise RuntimeError("expected exactly 16 calibration and 16 validation triptychs")
    calibration_pairs = [(load_triptych(path)[1], load_triptych(path)[2]) for path in calibration_paths]
    scale, bias = fit_rgb_affine(calibration_pairs)
    metric = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).eval().requires_grad_(False)
    calibration = evaluate(calibration_paths, scale, bias, metric, None)
    validation = evaluate(validation_paths, scale, bias, metric, args.output / "validation_triptychs")
    checks = {
        "psnr_sow": validation["candidate_psnr"] >= PSNR_FLOOR,
        "lpips_sow": validation["candidate_lpips"] <= LPIPS_CAP,
        "psnr_better_than_base": validation["psnr_delta_vs_base"] > 0,
        "lpips_not_worse_than_base": validation["lpips_improvement_vs_base"] >= 0,
    }
    result = {
        "schema_version": 1,
        "method": "single_global_per_channel_rgb_affine",
        "selection": "ordinary_least_squares_on_16_calibration_triptychs_only",
        "bounds": {"scale": [0.8, 1.2], "bias_normalized_rgb": [-0.1, 0.1]},
        "scale_rgb": scale.tolist(),
        "bias_rgb": bias.tolist(),
        "calibration": calibration,
        "validation": validation,
        "sow": {"psnr_minimum": PSNR_FLOOR, "lpips_maximum": LPIPS_CAP},
        "checks": checks,
        "automatic_pass": all(checks.values()),
        "protected_foundir_test_accessed": False,
        "sealed_internal_accessed": False,
        "gpu_wall_time": 0,
        "gpu_hours": 0,
        "input_triptych_sha256": {
            "calibration": {path.name: sha256(path) for path in calibration_paths},
            "validation": {path.name: sha256(path) for path in validation_paths},
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    result_path = args.output / "result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("scale_rgb", "bias_rgb", "validation", "checks", "automatic_pass")}, indent=2))
    print(f"result={result_path} sha256={sha256(result_path)}")
    return 0 if result["automatic_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
