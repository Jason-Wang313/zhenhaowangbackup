from __future__ import annotations

import pytest
import torch

from lowlight_restoration.scalar_residual import (
    aggregate_alpha_metrics,
    alpha_key,
    compare_to_base,
    gate_comparison,
    quantized_scalar_residual,
    select_alpha,
)


def _row(a05_psnr, a05_lpips, a1_psnr, a1_lpips):
    return {"alpha_metrics": {
        alpha_key(0.5): {"psnr": a05_psnr, "lpips": a05_lpips},
        alpha_key(1.0): {"psnr": a1_psnr, "lpips": a1_lpips},
    }}


def test_quantized_endpoints_are_exact():
    lq = torch.tensor([[[[-1.0, 0.0]]]])
    restored = torch.tensor([[[[1.0, 0.5]]]])
    u0, q0 = quantized_scalar_residual(lq, restored, 0.0)
    u1, q1 = quantized_scalar_residual(lq, restored, 1.0)
    assert torch.equal(u0, torch.tensor([[[[0, 128]]]], dtype=torch.uint8))
    assert torch.equal(u1, torch.tensor([[[[255, 191]]]], dtype=torch.uint8))
    assert torch.allclose(q0, u0.float() / 127.5 - 1)
    assert torch.allclose(q1, u1.float() / 127.5 - 1)


def test_rejects_extrapolation_and_shape_mismatch():
    value = torch.zeros(1, 3, 2, 2)
    with pytest.raises(ValueError):
        quantized_scalar_residual(value, value, 1.01)
    with pytest.raises(ValueError):
        quantized_scalar_residual(value, value[:, :, :, :1], 0.5)


def test_calibration_selection_and_validation_gate():
    rows = [_row(26.2, 0.16, 25.7, 0.145), _row(26.0, 0.17, 25.8, 0.15)]
    aggregate = aggregate_alpha_metrics(rows, [0.5, 1.0])
    assert select_alpha(aggregate, maximum_lpips=0.17) == 0.5
    comparison = compare_to_base(rows, selected_alpha=0.5)
    checks = gate_comparison(comparison, {
        "minimum_psnr_gain_over_alpha_1_db": 0.25,
        "maximum_lpips": 0.17,
        "maximum_lpips_regression_vs_alpha_1": 0.02,
        "minimum_psnr_win_fraction_vs_alpha_1": 0.5,
        "maximum_per_record_psnr_regression_vs_alpha_1_db": 2.0,
    })
    assert all(checks.values())


def test_selection_fails_without_lpips_eligible_alpha():
    rows = [_row(26.2, 0.18, 25.7, 0.19)]
    with pytest.raises(RuntimeError):
        select_alpha(aggregate_alpha_metrics(rows, [0.5, 1.0]), maximum_lpips=0.17)
