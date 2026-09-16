"""Deterministic scalar residual correction and frozen selection helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import torch


def alpha_key(alpha: float) -> str:
    return f"{float(alpha):.6f}"


def quantized_scalar_residual(
    lq: torch.Tensor, restored: torch.Tensor, alpha: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return uint8 and re-normalized [-1, 1] tensors for LQ + alpha*(restored-LQ)."""

    if lq.shape != restored.shape or lq.ndim != 4:
        raise ValueError("lq/restored must be matching NCHW tensors")
    if not 0.0 <= float(alpha) <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    blended = lq.float().add((restored.float() - lq.float()) * float(alpha)).clamp(-1, 1)
    uint8 = blended.add(1).mul(127.5).round().clamp(0, 255).to(torch.uint8)
    normalized = uint8.float().div(127.5).sub(1.0)
    return uint8, normalized


def aggregate_alpha_metrics(
    rows: Sequence[Mapping[str, object]], alpha_grid: Sequence[float]
) -> dict[str, dict[str, float]]:
    if not rows:
        raise ValueError("rows must be non-empty")
    result: dict[str, dict[str, float]] = {}
    for alpha in alpha_grid:
        key = alpha_key(alpha)
        psnr_values = np.asarray(
            [float(row["alpha_metrics"][key]["psnr"]) for row in rows], dtype=np.float64
        )
        lpips_values = np.asarray(
            [float(row["alpha_metrics"][key]["lpips"]) for row in rows], dtype=np.float64
        )
        result[key] = {
            "alpha": float(alpha),
            "records": len(rows),
            "mean_psnr": float(psnr_values.mean()),
            "mean_lpips": float(lpips_values.mean()),
        }
    return result


def select_alpha(
    aggregates: Mapping[str, Mapping[str, float]], *, maximum_lpips: float
) -> float:
    eligible = [
        item for item in aggregates.values() if float(item["mean_lpips"]) <= maximum_lpips
    ]
    if not eligible:
        raise RuntimeError("no calibration alpha satisfies the LPIPS constraint")
    selected = max(eligible, key=lambda item: (float(item["mean_psnr"]), float(item["alpha"])))
    return float(selected["alpha"])


def compare_to_base(
    rows: Sequence[Mapping[str, object]], *, selected_alpha: float, base_alpha: float = 1.0
) -> dict[str, float]:
    selected_key, base_key = alpha_key(selected_alpha), alpha_key(base_alpha)
    selected_psnr = np.asarray(
        [float(row["alpha_metrics"][selected_key]["psnr"]) for row in rows], dtype=np.float64
    )
    base_psnr = np.asarray(
        [float(row["alpha_metrics"][base_key]["psnr"]) for row in rows], dtype=np.float64
    )
    selected_lpips = np.asarray(
        [float(row["alpha_metrics"][selected_key]["lpips"]) for row in rows], dtype=np.float64
    )
    base_lpips = np.asarray(
        [float(row["alpha_metrics"][base_key]["lpips"]) for row in rows], dtype=np.float64
    )
    regressions = base_psnr - selected_psnr
    return {
        "selected_psnr": float(selected_psnr.mean()),
        "selected_lpips": float(selected_lpips.mean()),
        "base_psnr": float(base_psnr.mean()),
        "base_lpips": float(base_lpips.mean()),
        "psnr_gain_over_base_db": float((selected_psnr - base_psnr).mean()),
        "lpips_regression_vs_base": float((selected_lpips - base_lpips).mean()),
        "psnr_win_fraction_vs_base": float((selected_psnr > base_psnr).mean()),
        "maximum_per_record_psnr_regression_vs_base_db": float(regressions.max()),
    }


def gate_comparison(comparison: Mapping[str, float], gate: Mapping[str, object]) -> dict[str, bool]:
    return {
        "minimum_psnr_gain_over_alpha_1_db": comparison["psnr_gain_over_base_db"]
        >= float(gate["minimum_psnr_gain_over_alpha_1_db"]),
        "maximum_lpips": comparison["selected_lpips"] <= float(gate["maximum_lpips"]),
        "maximum_lpips_regression_vs_alpha_1": comparison["lpips_regression_vs_base"]
        <= float(gate["maximum_lpips_regression_vs_alpha_1"]),
        "minimum_psnr_win_fraction_vs_alpha_1": comparison["psnr_win_fraction_vs_base"]
        >= float(gate["minimum_psnr_win_fraction_vs_alpha_1"]),
        "maximum_per_record_psnr_regression_vs_alpha_1_db": comparison[
            "maximum_per_record_psnr_regression_vs_base_db"
        ]
        <= float(gate["maximum_per_record_psnr_regression_vs_alpha_1_db"]),
    }
