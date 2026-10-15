from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from smoke.train_foundir_blur_restormer_refinement_ddp import (
    QwenLevelWeightedBlurDataset,
    configure_foundir_runtime_paths,
    continuation_decision,
    evaluation_residual_scales,
    loss_terms,
    quantized,
    resume,
    save_checkpoint,
    selection_score,
    summary,
    validate_gpu_hardware,
)


class DifferentiablePerceptual(torch.nn.Module):
    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (prediction - target).square().mean()


def training_config() -> dict[str, object]:
    return {
        "model": {},
        "training": {
            "loss": {
                "charbonnier_weight": 1.0,
                "charbonnier_epsilon": 0.001,
                "rgb_mse_weight": 1.0,
                "edge_l1_weight": 0.05,
                "lpips_weight": 0.1,
                "lpips_every_steps": 1,
            }
        },
        "gate": {
            "require_exact_v2_base_metric_reproduction": True,
            "base_metric_tolerance": 0.0001,
            "blur_psnr_minimum": 25.9,
            "blur_lpips_maximum": 0.18,
            "noise_psnr_minimum": 38.9,
            "noise_lpips_maximum": 0.1,
        },
    }


def corrective_training_config() -> dict[str, object]:
    config = training_config()
    config["training"]["loss"].update(
        {
            "charbonnier_weight": 0.25,
            "rgb_mse_weight": 10.0,
            "psnr_surrogate_weight": 0.02,
            "psnr_surrogate_epsilon": 0.0001,
        }
    )
    return config


def test_lpips_term_keeps_fp32_gradient() -> None:
    raw = torch.randn(1, 3, 16, 16, requires_grad=True)
    prediction = raw.sigmoid()
    target = torch.rand_like(prediction)
    loss, terms = loss_terms(
        prediction,
        target,
        training_config(),
        DifferentiablePerceptual(),
        step=1,
    )
    loss.backward()
    assert loss.dtype == torch.float32
    assert terms["lpips_active"] == 1.0
    assert raw.grad is not None
    assert torch.isfinite(raw.grad).all()
    assert float(raw.grad.abs().sum()) > 0.0


def test_psnr_surrogate_is_finite_and_uses_per_image_log_mse() -> None:
    values = torch.tensor([0.1, 0.3], requires_grad=True)
    raw = values.reshape(2, 1, 1, 1).expand(-1, 3, 2, 2)
    target = torch.zeros_like(raw)
    loss, terms = loss_terms(
        raw,
        target,
        corrective_training_config(),
        DifferentiablePerceptual(),
        step=1,
    )
    loss.backward()
    expected = (torch.log1p(torch.tensor(100.0)) + torch.log1p(torch.tensor(900.0))) / 2
    assert terms["psnr_surrogate"] == pytest.approx(float(expected))
    assert torch.isfinite(loss)
    assert values.grad is not None
    assert torch.isfinite(values.grad).all()


def test_weighted_blur_level_schedule_is_deterministic_and_frozen() -> None:
    rows = {4: [{"record_id": "l4"}], 5: [{"record_id": "l5"}]}
    probabilities = {4: 0.25, 5: 0.75}
    config = {"data": {"seed": 2026090908}}
    first = QwenLevelWeightedBlurDataset(rows, probabilities, config)
    second = QwenLevelWeightedBlurDataset(rows, probabilities, config)
    levels = [first.level_for_presentation(index) for index in range(10_000)]
    assert levels == [second.level_for_presentation(index) for index in range(10_000)]
    assert set(levels) == {4, 5}
    assert levels.count(5) / len(levels) == pytest.approx(0.75, abs=0.015)


def test_residual_scale_contract_rejects_unsorted_or_out_of_range_values() -> None:
    assert evaluation_residual_scales({"model": {}}) == [1.0]
    with pytest.raises(RuntimeError, match="residual-scale"):
        evaluation_residual_scales({"model": {"evaluation_residual_scales": [1.0, 0.5]}})
    with pytest.raises(RuntimeError, match="residual-scale"):
        evaluation_residual_scales({"model": {"evaluation_residual_scales": [0.0, 2.1]}})


def test_quantized_uses_png_rounding_semantics() -> None:
    values = torch.tensor([0.0, 0.5 / 255, 1.5 / 255, 1.0])
    assert torch.equal(quantized(values), torch.tensor([0.0, 0.0, 2.0 / 255, 1.0]))


def test_summary_reproduces_frozen_base_and_counts_routes() -> None:
    rows = []
    for index in range(60):
        rows.append(
            {
                "category": "01Blur",
                "route": "blur_specialization+none" if index < 54 else "generalist+none",
                "base": {"psnr": 25.3164, "lpips": 0.2881},
                "refined_routed": {"psnr": 26.0, "lpips": 0.17},
            }
        )
        rows.append(
            {
                "category": "05Noise",
                "route": "generalist+none",
                "base": {"psnr": 39.4291, "lpips": 0.0659},
                "refined_routed": {"psnr": 39.4291, "lpips": 0.0659},
            }
        )
    frozen = {
        "task_metrics": {
            "blur": {"psnr": 25.3164, "lpips": 0.2881},
            "noise": {"psnr": 39.4291, "lpips": 0.0659},
        }
    }
    result = summary(rows, step=1000, config=training_config(), router_development=frozen)
    assert result["records"] == 120
    assert result["blur"]["qwen_blur_l4plus_records"] == 54
    assert result["noise"]["qwen_false_blur_l4plus_records"] == 0
    assert result["selected_residual_scale"] == 1.0
    assert result["sealed_internal_test_accessed"] is False
    assert result["protected_foundir_test_accessed"] is False


def test_summary_rejects_base_metric_drift() -> None:
    rows = [
        {
            "category": category,
            "route": "generalist+none",
            "base": {"psnr": 20.0, "lpips": 0.2},
            "refined_routed": {"psnr": 20.0, "lpips": 0.2},
        }
        for category in ("01Blur", "05Noise")
    ]
    frozen = {
        "task_metrics": {
            "blur": {"psnr": 21.0, "lpips": 0.2},
            "noise": {"psnr": 20.0, "lpips": 0.2},
        }
    }
    with pytest.raises(RuntimeError, match="does not reproduce integrated v2"):
        summary(rows, step=1000, config=training_config(), router_development=frozen)


def test_selection_score_uses_worst_normalized_margin() -> None:
    result = {
        "blur": {"refined_routed": {"psnr": 26.1, "lpips": 0.17}},
        "noise": {"refined_routed": {"psnr": 39.1, "lpips": 0.09}},
    }
    gate = {
        "blur_psnr_minimum": 25.9,
        "blur_lpips_maximum": 0.18,
        "noise_psnr_minimum": 38.9,
        "noise_lpips_maximum": 0.1,
    }
    assert selection_score(result, gate) == pytest.approx(0.2)


def test_selection_score_supports_balanced_psnr_and_lpips_units() -> None:
    result = {
        "blur": {"refined_routed": {"psnr": 26.1, "lpips": 0.17}},
        "noise": {"refined_routed": {"psnr": 39.1, "lpips": 0.09}},
    }
    gate = {
        "blur_psnr_minimum": 25.9,
        "blur_lpips_maximum": 0.18,
        "noise_psnr_minimum": 38.9,
        "noise_lpips_maximum": 0.1,
        "psnr_margin_unit_db": 0.1,
        "lpips_margin_unit": 0.01,
    }
    assert selection_score(result, gate) == pytest.approx(1.0)


def continuation_config() -> dict[str, object]:
    config = training_config()
    config["training"]["optimizer_steps"] = 3000
    config["continuation"] = {
        "minimum_blur_psnr": 25.316358061390666,
        "maximum_blur_lpips": 0.2349482521453562,
        "minimum_selection_score_gain": 0.0,
        "stop_on_gate_pass": True,
    }
    return config


def milestone_result(*, step: int, psnr: float, lpips: float) -> dict[str, object]:
    return {
        "step": step,
        "blur": {"refined_routed": {"psnr": psnr, "lpips": lpips}},
        "noise": {"refined_routed": {"psnr": 39.4291, "lpips": 0.0659}},
    }


def test_corrective_continuation_requires_psnr_lpips_and_margin_progress() -> None:
    config = continuation_config()
    accepted = continuation_decision(
        milestone_result(step=250, psnr=25.5, lpips=0.22),
        config,
        previous_selection_score=-5.49482521453562,
    )
    assert accepted["continue_training"] is True
    assert accepted["reasons"] == []

    rejected = continuation_decision(
        milestone_result(step=250, psnr=25.2, lpips=0.22),
        config,
        previous_selection_score=-5.49482521453562,
    )
    assert rejected["continue_training"] is False
    assert "blur_psnr_below_frozen_base" in rejected["reasons"]


def test_corrective_continuation_auto_stops_on_pass_or_final_milestone() -> None:
    config = continuation_config()
    passed = continuation_decision(
        milestone_result(step=250, psnr=26.0, lpips=0.17),
        config,
        previous_selection_score=-5.49482521453562,
    )
    assert passed["continue_training"] is False
    assert passed["reasons"] == ["gate_pass"]

    final = continuation_decision(
        milestone_result(step=3000, psnr=25.8, lpips=0.19),
        config,
        previous_selection_score=-1.5,
    )
    assert final["continue_training"] is False
    assert final["reasons"] == ["final_milestone"]


def test_motion_continuation_requires_balanced_progress_and_safety_floors() -> None:
    config = training_config()
    config["training"]["optimizer_steps"] = 3000
    config["gate"].update(
        {"psnr_margin_unit_db": 0.1, "lpips_margin_unit": 0.01}
    )
    config["continuation"] = {
        "policy": "fail_closed_balanced_progress_from_frozen_base",
        "minimum_blur_psnr": 25.216358061390665,
        "maximum_blur_lpips": 0.28808404979063196,
        "minimum_selection_score_gain": 0.05,
        "stop_on_gate_pass": True,
    }
    accepted = continuation_decision(
        milestone_result(step=0, psnr=25.25, lpips=0.24),
        config,
        previous_selection_score=-10.808404979063196,
    )
    assert accepted["continue_training"] is True
    assert accepted["reasons"] == []

    rejected = continuation_decision(
        milestone_result(step=0, psnr=25.1, lpips=0.23),
        config,
        previous_selection_score=-10.808404979063196,
    )
    assert rejected["continue_training"] is False
    assert "blur_psnr_below_allowed_floor" in rejected["reasons"]


def test_summary_selects_best_residual_scale_by_gate_margin() -> None:
    config = training_config()
    config["model"]["evaluation_residual_scales"] = [0.0, 0.5, 1.0]
    config["gate"].update(
        {
            "blur_psnr_minimum": 25.9,
            "blur_lpips_maximum": 0.18,
            "noise_psnr_minimum": 38.9,
            "noise_lpips_maximum": 0.1,
        }
    )
    rows = []
    for category, base in (
        ("01Blur", {"psnr": 25.3164, "lpips": 0.2881}),
        ("05Noise", {"psnr": 39.4291, "lpips": 0.0659}),
    ):
        rows.append(
            {
                "category": category,
                "route": "blur_specialization+none" if category == "01Blur" else "generalist+none",
                "base": base,
                "refined_routed_by_scale": {
                    "0.000000": base,
                    "0.500000": {"psnr": 26.0, "lpips": 0.179} if category == "01Blur" else base,
                    "1.000000": {"psnr": 25.8, "lpips": 0.17} if category == "01Blur" else base,
                },
            }
        )
    frozen = {
        "task_metrics": {
            "blur": {"psnr": 25.3164, "lpips": 0.2881},
            "noise": {"psnr": 39.4291, "lpips": 0.0659},
        }
    }
    result = summary(rows, step=250, config=config, router_development=frozen)
    assert result["selected_residual_scale"] == 0.5
    assert result["blur"]["refined_routed"]["psnr"] == pytest.approx(26.0)
    assert set(result["residual_scale_candidates"]) == {
        "0.000000",
        "0.500000",
        "1.000000",
    }


def test_checkpoint_is_immutable_and_resume_is_schedule_bound(tmp_path) -> None:
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    path = tmp_path / "step_001000.pt"
    save_checkpoint(path, model, optimizer, step=1000, config_sha="a" * 64)
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        save_checkpoint(path, model, optimizer, step=1000, config_sha="a" * 64)

    restored = torch.nn.Linear(3, 2)
    restored_optimizer = torch.optim.AdamW(restored.parameters())
    assert resume(
        tmp_path,
        restored,
        restored_optimizer,
        "a" * 64,
        [1000, 3000],
    ) == 1000
    for expected, actual in zip(model.parameters(), restored.parameters(), strict=True):
        assert torch.equal(expected, actual)

    with pytest.raises(RuntimeError, match="invalid Blur Restormer resume checkpoint"):
        resume(tmp_path, restored, restored_optimizer, "a" * 64, [3000])


def test_runtime_paths_pin_released_dependency_versions() -> None:
    configure_foundir_runtime_paths()
    import huggingface_hub
    import tokenizers
    import transformers

    assert transformers.__version__ == "4.47.1"
    assert tokenizers.__version__ == "0.21.0"
    assert huggingface_hub.__version__ == "0.27.1"


def test_gpu_hardware_requires_exact_homogeneous_node(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 8)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda index: SimpleNamespace(name="NVIDIA H100 80GB HBM3"),
    )
    assert len(validate_gpu_hardware(8)) == 8

    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda index: SimpleNamespace(
            name="NVIDIA H100 80GB HBM3" if index < 7 else "NVIDIA H200"
        ),
    )
    with pytest.raises(RuntimeError, match="homogeneous H100 or H200"):
        validate_gpu_hardware(8)
