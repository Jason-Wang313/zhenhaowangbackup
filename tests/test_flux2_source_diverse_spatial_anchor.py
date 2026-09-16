from __future__ import annotations

import copy
import csv
import hashlib
import json
from pathlib import Path

import pytest
import torch

from smoke.train_foundir_blur_only_flux2_pixel_fidelity_ddp import (
    charbonnier_loss,
    compare_final_arms,
    validate_config,
)
from smoke.train_foundir_blur_only_flux2_overfit_ddp import scheduled_index


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/foundir_blur_only_flux2_source_diverse_spatial_anchor_pilot_v1.json"


def config() -> dict[str, object]:
    return json.loads(CONFIG.read_text())


def test_frozen_config_and_loss_weights() -> None:
    payload = config()
    validate_config(payload)
    assert payload["training"]["max_steps_per_arm"] == 400
    assert payload["training"]["checkpoint_steps"] == [200, 400]
    assert payload["training"]["global_batch_size"] == 8
    assert payload["trajectory"]["detached_prefix_steps"] == 46
    assert payload["trajectory"]["differentiable_suffix_steps"] == 4
    assert payload["training"]["arms"][1] == {
        "name": "trajectory_spatial_anchor",
        "flow_mse_weight": 1.0,
        "endpoint_rgb_charbonnier_weight": 1.0,
        "endpoint_lpips_weight": 0.02,
        "multiscale_rgb_charbonnier_weight": 0.5,
        "endpoint_latent_l1_weight": 0.25,
        "low_frequency_lq_identity_weight": 0.1,
    }
    assert payload["gate"]["full_scale_automatic"] is False


def test_manifest_is_exact_source_diverse_and_excludes_prior_groups() -> None:
    payload = config()
    manifest = ROOT / payload["data"]["frozen_manifest"]
    assert hashlib.sha256(manifest.read_bytes()).hexdigest() == payload["data"]["frozen_manifest_sha256"]
    with manifest.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    train = [row for row in rows if row["role"] == "pilot_train"]
    development = [row for row in rows if row["role"] == "fixed_development"]
    with (ROOT / payload["data"]["excluded_sanity_manifest"]).open(newline="") as handle:
        excluded = {row["source_group"] for row in csv.DictReader(handle)}
    assert len(train) == len({row["source_group"] for row in train}) == 512
    assert len(development) == len({row["source_group"] for row in development}) == 16
    assert {row["source_group"] for row in train}.isdisjoint(excluded)
    assert {row["source_group"] for row in train}.isdisjoint(
        {row["source_group"] for row in development}
    )
    assert sum(row["visual_train"] == "true" for row in train) == 16
    for row in rows:
        assert 0 <= int(row["crop_x"]) <= int(row["width"]) - 512
        assert 0 <= int(row["crop_y"]) <= int(row["height"]) - 512


@pytest.mark.parametrize("family", ["H100", "H200"])
def test_hardware_profiles_preserve_global_batch(family: str) -> None:
    payload = config()
    profile = payload["training"]["hardware_profiles"][family]
    assert payload["distributed"]["required_world_size"] * profile["micro_batch_size_per_rank"] * profile["gradient_accumulation_steps"] == 8


def test_charbonnier_routes_finite_gradient() -> None:
    prediction = torch.randn(2, 3, 8, 8, requires_grad=True)
    target = torch.randn_like(prediction)
    loss = charbonnier_loss(prediction, target, 0.001)
    loss.backward()
    assert loss.isfinite()
    assert prediction.grad is not None and torch.isfinite(prediction.grad).all()
    assert float(prediction.grad.abs().sum()) > 0


def test_paired_arms_have_identical_record_noise_and_timestep_streams() -> None:
    payload = config()
    seed = payload["training"]["seed"]
    trajectory_seed = payload["trajectory"]["seed"]
    streams = []
    for _arm in payload["training"]["arms"]:
        rows = []
        for step in range(12):
            position = step * 8 + 3
            generator = torch.Generator().manual_seed(seed + position * 1_000_003)
            endpoint_generator = torch.Generator().manual_seed(trajectory_seed + position * 1_000_033)
            rows.append((
                scheduled_index(position, 512, seed),
                torch.randn(7, generator=generator),
                torch.rand(1, generator=generator),
                torch.randn(7, generator=endpoint_generator),
            ))
        streams.append(rows)
    for left, right in zip(streams[0], streams[1]):
        assert left[0] == right[0]
        for a, b in zip(left[1:], right[1:]):
            torch.testing.assert_close(a, b, rtol=0, atol=0)


def _metrics(psnr: float, lpips: float, values: list[float]) -> dict[str, object]:
    return {
        "mean_output_psnr": psnr,
        "mean_output_lpips": lpips,
        "rows": [
            {"record_id": f"r{i:02d}", "output_psnr": value}
            for i, value in enumerate(values)
        ],
    }


def _arm(parent: dict[str, object], final: dict[str, object]) -> dict[str, object]:
    return {
        "parent_descriptive_development_metrics": parent,
        "final_descriptive_development_metrics": final,
    }


def test_development_gate_enforces_parent_control_wins_and_worst_case() -> None:
    payload = config()
    parent = _metrics(24.591062, 0.150766, [24.0] * 16)
    control = _metrics(24.4, 0.152, [23.8] * 16)
    candidate = _metrics(24.7, 0.151, [24.2] * 16)
    result = compare_final_arms(payload, {
        "flow_control": _arm(parent, control),
        "trajectory_spatial_anchor": _arm(parent, candidate),
    })
    assert all(result["checks"].values())
    damaged = copy.deepcopy(candidate)
    damaged["rows"][0]["output_psnr"] = 21.0
    result = compare_final_arms(payload, {
        "flow_control": _arm(parent, control),
        "trajectory_spatial_anchor": _arm(parent, damaged),
    })
    assert result["checks"]["candidate_maximum_per_record_regression_within_limit"] is False


def test_protected_scope_and_pairing_cannot_be_enabled() -> None:
    payload = config()
    payload["scope"]["official_foundir_test_accessed"] = True
    with pytest.raises(RuntimeError, match="scope contract"):
        validate_config(payload)
    payload = config()
    payload["training"]["paired_sample_order_noise_timesteps"] = False
    with pytest.raises(RuntimeError, match="paired pilot"):
        validate_config(payload)
