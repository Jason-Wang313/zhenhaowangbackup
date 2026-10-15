#!/usr/bin/env python3
"""CPU-only output-blend ceiling audit for the two frozen Rain identity gates."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "manifests/foundir_rain_flux2_strength_identity_ratio_gate_v1.csv"
EQUAL_RESULT = ROOT / "checkpoints/foundir_rain_flux2_strength_identity_directional_gate_v1/result.json"
RATIO_RESULT = ROOT / "checkpoints/foundir_rain_flux2_strength_identity_ratio_gate_v1/result.json"
EQUAL_DIR = ROOT / "checkpoints/foundir_rain_flux2_strength_identity_directional_gate_v1/final_all_16/validation_triptychs/step_002250"
RATIO_DIR = ROOT / "checkpoints/foundir_rain_flux2_strength_identity_ratio_gate_v1/final_all_16/validation_triptychs/step_002250"
OUTPUT = ROOT / "reports/foundir_rain_identity_checkpoint_interpolation_audit_2026-09-18.json"

EXPECTED = {
    MANIFEST: "6e39096a93418a54bd573dee2affdee9421f2167060e7e011a3af070d45acfd9",
    EQUAL_RESULT: "81199c746840e9363c3f7846068e3871b070a513337effe43e76e61413b6b6b9",
    RATIO_RESULT: "4514ae7d1129bc58c940600a7d519959009845fdbd2f3b5d1d694a049a32f534",
}
BETAS = (0.0, 0.25, 0.5, 0.75, 1.0)
SOW_PSNR = 34.2
SOW_LPIPS = 0.08


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def psnr(gt: np.ndarray, output: np.ndarray) -> float:
    mse = float(np.mean(((gt.astype(np.float32) - output.astype(np.float32)) / 255.0) ** 2))
    return 100.0 if mse <= 1e-12 else -10.0 * math.log10(mse)


def strength(value: float) -> str:
    return "heavy" if value < 28.0 else "moderate" if value < 34.0 else "near_clean"


def triptych_output(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        if rgb.size != (1536, 540):
            raise RuntimeError(f"unexpected triptych shape: {path} {rgb.size}")
        return np.asarray(rgb.crop((512, 28, 1024, 540)), dtype=np.uint8).copy()


def crop(row: dict[str, str], field: str) -> np.ndarray:
    x, y, size = int(row["crop_x"]), int(row["crop_y"]), 512
    with Image.open(ROOT / row[field]) as image:
        return np.asarray(image.convert("RGB").crop((x, y, x + size, y + size)), dtype=np.uint8).copy()


def lpips_batch(metric, gt: list[np.ndarray], outputs: list[np.ndarray]) -> list[float]:
    values = []
    with torch.inference_mode():
        for begin in range(0, len(gt), 4):
            reference = torch.from_numpy(np.stack(gt[begin:begin + 4])).permute(0, 3, 1, 2).float().div_(255)
            prediction = torch.from_numpy(np.stack(outputs[begin:begin + 4])).permute(0, 3, 1, 2).float().div_(255)
            batch = metric(prediction, reference)
            # torchmetrics normally reduces the batch, so evaluate individually
            # when it does not expose per-sample values.
            if batch.numel() == 1:
                for pred, ref in zip(prediction, reference):
                    values.append(float(metric(pred.unsqueeze(0), ref.unsqueeze(0)).item()))
            else:
                values.extend(float(value) for value in batch.flatten())
    return values


def main() -> None:
    for path, expected in EXPECTED.items():
        if sha256(path) != expected:
            raise RuntimeError(f"frozen audit input changed: {path}")
    with MANIFEST.open(newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["role"] == "directional_development"]
    if len(rows) != 48:
        raise RuntimeError("directional development cardinality changed")

    gt, lq, equal, ratio = [], [], [], []
    for index, row in enumerate(rows):
        name = f"{index:02d}_{row['record_id']}.png"
        gt.append(crop(row, "gt_path"))
        lq.append(crop(row, "lq_path"))
        equal.append(triptych_output(EQUAL_DIR / name))
        ratio.append(triptych_output(RATIO_DIR / name))
    input_psnr = [psnr(ref, inp) for ref, inp in zip(gt, lq)]
    bins = [strength(value) for value in input_psnr]
    if {name: bins.count(name) for name in set(bins)} != {"heavy": 16, "moderate": 16, "near_clean": 16}:
        raise RuntimeError("development strength balance changed")

    parent = json.loads(EQUAL_RESULT.read_text())["parent_development_metrics"]
    parent_strength = json.loads(EQUAL_RESULT.read_text())["parent_strength_metrics"]
    metric = LearnedPerceptualImagePatchSimilarity(net_type="alex", reduction="none", normalize=True).eval()
    fixed = []
    fixed_outputs: dict[float, list[np.ndarray]] = {}
    for beta in BETAS:
        outputs = [
            np.rint((1.0 - beta) * left.astype(np.float32) + beta * right.astype(np.float32)).clip(0, 255).astype(np.uint8)
            for left, right in zip(equal, ratio)
        ]
        fixed_outputs[beta] = outputs
        psnrs = [psnr(ref, out) for ref, out in zip(gt, outputs)]
        lpips = lpips_batch(metric, gt, outputs)
        subgroup = {
            name: {
                "records": bins.count(name),
                "mean_output_psnr": float(np.mean([value for value, label in zip(psnrs, bins) if label == name])),
                "mean_output_lpips": float(np.mean([value for value, label in zip(lpips, bins) if label == name])),
            }
            for name in ("heavy", "moderate", "near_clean")
        }
        mean_psnr, mean_lpips = float(np.mean(psnrs)), float(np.mean(lpips))
        checks = {
            "heavy_psnr_retention_vs_parent": subgroup["heavy"]["mean_output_psnr"] >= parent_strength["heavy"]["mean_output_psnr"] - 0.1,
            "moderate_psnr_retention_vs_parent": subgroup["moderate"]["mean_output_psnr"] >= parent_strength["moderate"]["mean_output_psnr"] - 0.1,
            "near_clean_psnr_gain_vs_parent": subgroup["near_clean"]["mean_output_psnr"] >= parent_strength["near_clean"]["mean_output_psnr"] + 0.5,
            "overall_psnr_gain_vs_parent": mean_psnr >= parent["mean_output_psnr"] + 0.2,
            "overall_lpips_retention_vs_parent": mean_lpips <= parent["mean_output_lpips"] + 0.005,
        }
        fixed.append({
            "beta_ratio_checkpoint": beta,
            "mean_output_psnr": mean_psnr,
            "mean_output_lpips": mean_lpips,
            "sow_psnr_margin_db": mean_psnr - SOW_PSNR,
            "sow_psnr_margin_percent": (mean_psnr - SOW_PSNR) / SOW_PSNR * 100.0,
            "sow_lpips_safety": SOW_LPIPS - mean_lpips,
            "sow_lpips_safety_percent": (SOW_LPIPS - mean_lpips) / SOW_LPIPS * 100.0,
            "subgroups": subgroup,
            "checks": checks,
            "all_directional_checks_pass": all(checks.values()),
        })

    grid = np.linspace(0.0, 1.0, 101)
    oracle_psnr = []
    oracle_beta = []
    for ref, left, right in zip(gt, equal, ratio):
        values = [psnr(ref, np.rint((1.0 - beta) * left + beta * right).clip(0, 255).astype(np.uint8)) for beta in grid]
        best = int(np.argmax(values))
        oracle_psnr.append(values[best])
        oracle_beta.append(float(grid[best]))
    discrete_psnr = [max(psnr(ref, left), psnr(ref, right)) for ref, left, right in zip(gt, equal, ratio)]
    equal_lpips = lpips_batch(metric, gt, equal)
    ratio_lpips = lpips_batch(metric, gt, ratio)
    payload = {
        "schema_version": 1,
        "status": "complete_read_only_audit",
        "method": "output_space_proxy_for_fixed_checkpoint_interpolation",
        "warning": "Output blending is a screening proxy; it does not claim exact equivalence to LoRA weight interpolation.",
        "fixed_betas": fixed,
        "per_image_discrete_two_checkpoint_oracle": {
            "mean_output_psnr": float(np.mean(discrete_psnr)),
            "mean_output_lpips_independent_oracle": float(np.mean(np.minimum(equal_lpips, ratio_lpips))),
        },
        "per_image_continuous_output_blend_psnr_oracle": {
            "grid_step": 0.01,
            "mean_output_psnr": float(np.mean(oracle_psnr)),
            "mean_beta": float(np.mean(oracle_beta)),
            "sow_psnr_pass": float(np.mean(oracle_psnr)) >= SOW_PSNR,
        },
        "inputs": {str(path.relative_to(ROOT)): expected for path, expected in EXPECTED.items()},
        "records": 48,
        "protected_foundir_test_accessed": False,
        "sealed_internal_accessed": False,
        "gpu_used": False,
    }
    content = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    if OUTPUT.exists() and OUTPUT.read_bytes() != content:
        raise RuntimeError("refusing to overwrite differing Rain interpolation audit")
    if not OUTPUT.exists():
        OUTPUT.write_bytes(content)
    print("FOUNDIR_RAIN_INTERPOLATION_AUDIT_COMPLETE " + json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
