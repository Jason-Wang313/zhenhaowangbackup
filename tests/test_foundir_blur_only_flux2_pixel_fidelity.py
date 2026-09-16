from __future__ import annotations

import copy
import csv
import hashlib
import json
from pathlib import Path

import pytest
import torch

from smoke.train_foundir_blur_only_flux2_pixel_fidelity_ddp import (
    compare_final_arms,
    decoded_rgb_mse,
    predicted_clean_from_pure_flow,
    validate_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/foundir_blur_only_flux2_pixel_fidelity_gate_v1.json"


def config() -> dict[str, object]:
    return json.loads(CONFIG.read_text())


def test_config_is_exact_paired_ablation() -> None:
    payload = config()
    validate_config(payload)
    assert payload["training"]["arms"] == [
        {"name": "flow_control", "flow_mse_weight": 1.0, "rgb_mse_weight": 0.0},
        {"name": "rgb_mse", "flow_mse_weight": 1.0, "rgb_mse_weight": 1.0},
    ]
    assert payload["training"]["optimizer_initialization"] == "fresh_adamw_per_arm"
    assert payload["gate"]["predeclared_candidate_step"] == 400
    assert payload["gate"]["full_scale_automatic"] is False


@pytest.mark.parametrize(
    "field",
    [
        "qwen_degradation_conditioning",
        "mixed_degradations",
        "pixel_refiner",
        "decoder_tuning",
        "development_gt_optimization",
        "sealed_internal_accessed",
        "official_foundir_test_accessed",
    ],
)
def test_config_rejects_protected_or_confounded_scope(field: str) -> None:
    payload = config()
    payload["scope"][field] = True
    with pytest.raises(RuntimeError, match="scope contract"):
        validate_config(payload)


def test_h100_h200_profiles_preserve_global_batch() -> None:
    payload = config()
    world = payload["distributed"]["required_world_size"]
    expected = payload["training"]["global_batch_size"]
    for profile in payload["training"]["hardware_profiles"].values():
        assert world * profile["micro_batch_size_per_rank"] * profile["gradient_accumulation_steps"] == expected


def test_manifest_is_hash_pinned_and_source_disjoint() -> None:
    payload = config()
    manifest = ROOT / payload["data"]["frozen_manifest"]
    assert hashlib.sha256(manifest.read_bytes()).hexdigest() == payload["data"]["frozen_manifest_sha256"]
    with manifest.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    train = [row for row in rows if row["role"] == "tiny_overfit_train"]
    development = [row for row in rows if row["role"] == "future_blur_validation"]
    assert len(train) == len(development) == 16
    assert {row["source_group"] for row in train}.isdisjoint(
        {row["source_group"] for row in development}
    )


def test_pure_flow_predicted_clean_identity() -> None:
    generator = torch.Generator().manual_seed(7)
    clean = torch.randn((2, 3, 4, 5), generator=generator)
    noise = torch.randn(clean.shape, generator=generator)
    velocity = noise - clean
    packed = velocity.flatten(2).transpose(1, 2).contiguous()
    recovered = predicted_clean_from_pure_flow(packed, noise, clean.shape)
    torch.testing.assert_close(recovered, clean, rtol=0, atol=1e-6)


class FrozenDecoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = torch.nn.Conv2d(2, 3, 1)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.projection(latent)


def test_rgb_loss_routes_gradient_to_prediction_not_frozen_decoder() -> None:
    decoder = FrozenDecoder().eval().requires_grad_(False)
    prediction = torch.randn((1, 16, 2), requires_grad=True)
    noise = torch.randn((1, 2, 4, 4))
    target = torch.randn((1, 3, 4, 4)).clamp(-1, 1)
    loss, decoded = decoded_rgb_mse(prediction, noise, noise.shape, decoder, target)
    assert loss.isfinite() and decoded.shape == target.shape
    loss.backward()
    assert prediction.grad is not None and float(prediction.grad.abs().sum()) > 0
    assert all(parameter.grad is None for parameter in decoder.parameters())


def metric(psnr: float, lpips: float, rows: list[dict[str, object]]) -> dict[str, object]:
    return {
        "mean_output_psnr": psnr,
        "mean_output_lpips": lpips,
        "rows": rows,
    }


def arm(parent_psnr: float, parent_lpips: float, final_psnr: float, final_lpips: float, offset: float) -> dict[str, object]:
    rows = [
        {"record_id": f"r{index}", "output_psnr": 25.0 + offset + index / 100}
        for index in range(16)
    ]
    return {
        "parent_train_metrics": metric(parent_psnr, parent_lpips, rows),
        "final_train_metrics": metric(final_psnr, final_lpips, rows),
    }


def test_gate_requires_candidate_to_beat_control_and_parent_without_lpips_damage() -> None:
    payload = config()
    results = {
        "flow_control": arm(25.0, 0.14, 25.1, 0.14, 0.0),
        "rgb_mse": arm(25.0, 0.14, 25.4, 0.139, 0.3),
    }
    comparison = compare_final_arms(payload, results)
    assert all(comparison["checks"].values())
    damaged = copy.deepcopy(results)
    damaged["rgb_mse"]["final_train_metrics"]["mean_output_lpips"] = 0.15
    comparison = compare_final_arms(payload, damaged)
    assert comparison["checks"]["train_lpips_not_worse_than_control"] is False
    assert comparison["checks"]["train_lpips_not_worse_than_parent"] is False


def test_parent_checkpoint_is_exact_step_10k() -> None:
    payload = config()
    path = ROOT / payload["model"]["parent_checkpoint"]
    assert path.is_file()
    assert hashlib.sha256(path.read_bytes()).hexdigest() == payload["model"]["parent_checkpoint_sha256"]
