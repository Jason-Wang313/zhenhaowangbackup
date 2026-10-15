from __future__ import annotations

import pytest

from smoke.evaluate_foundir_noise_rain_task_native_capacity_ddp import (
    build_result,
    candidate_names,
    select_rows,
)


def test_isolated_capacity_selection_and_candidate_families_are_exact() -> None:
    rows = [
        {"record_id": "n", "split": "validation", "category": "05Noise"},
        {"record_id": "r", "split": "validation", "category": "10Rain"},
        {"record_id": "x", "split": "train", "category": "05Noise"},
        {"record_id": "h", "split": "validation", "category": "08Haze"},
    ]
    import smoke.evaluate_foundir_noise_rain_task_native_capacity_ddp as evaluator

    original = evaluator.EXPECTED_CATEGORIES
    evaluator.EXPECTED_CATEGORIES = {"05Noise": 1, "10Rain": 1}
    try:
        selected = select_rows(rows)
    finally:
        evaluator.EXPECTED_CATEGORIES = original
    assert [row["record_id"] for row in selected] == ["n", "r"]
    assert candidate_names(rows[0]) == [
        "real_denoising",
        "gaussian_color_denoising_blind",
    ]
    assert candidate_names(rows[1]) == ["deraining"]


def metric(psnr: float, lpips: float) -> dict[str, float]:
    return {"psnr": psnr, "lpips": lpips, "residual_l1": 0.01, "residual_rms": 0.02}


def test_capacity_result_separates_oracle_diagnostic_from_actual_qwen_hybrid() -> None:
    base = {"input": metric(20.0, 0.3), "restored": metric(35.0, 0.1)}
    rows = []
    for task, category, candidates in (
        (
            "noise",
            "05Noise",
            {
                "real_denoising": metric(42.0, 0.05),
                "gaussian_color_denoising_blind": metric(41.0, 0.06),
            },
        ),
        ("rain", "10Rain", {"deraining": metric(38.0, 0.04)}),
    ):
        for group, level in (("a", 1), ("b", 3)):
            qwen = {"blur": 1, "noise": 1, "rain": 1, "haze": 1, "low_light": 1}
            qwen[task] = level
            rows.append(
                {
                    "category": category,
                    "source_group": f"{task}-{group}",
                    "qwen_levels": qwen,
                    "baseline": base,
                    "candidates": candidates,
                }
            )
    config = {
        "metrics": {"bootstrap_resamples": 20, "bootstrap_confidence": 0.95, "bootstrap_seed": 7},
        "routing_diagnostics": {"thresholds": [2, 3, 4, 5]},
        "strong_gates": {
            "noise": {
                "mean_psnr_minimum": 39.9,
                "mean_lpips_maximum": 0.08,
                "group_bootstrap_psnr_lower_minimum": 39.4,
                "group_bootstrap_lpips_upper_maximum": 0.09,
            },
            "rain": {
                "mean_psnr_minimum": 35.2,
                "mean_lpips_maximum": 0.06,
                "group_bootstrap_psnr_lower_minimum": 34.7,
                "group_bootstrap_lpips_upper_maximum": 0.07,
            },
        },
    }
    result = build_result(rows, config, "config")
    assert result["deployable_selection_frozen"] is False
    assert result["candidate_capacity"]["noise:real_denoising"]["scope"].startswith(
        "category_selected_capacity"
    )
    hybrid = result["actual_qwen_hybrid_diagnostics"][
        "noise:real_denoising:qwen_l2_baseline_fallback"
    ]
    assert hybrid["triggered_records"] == 1
    assert hybrid["metrics"]["psnr"] == pytest.approx((35.0 + 42.0) / 2)
