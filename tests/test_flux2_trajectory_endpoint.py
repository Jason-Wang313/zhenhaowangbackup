from __future__ import annotations

import sys
import json
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party/flux2/src"))

from flux2.sampling import denoise_cfg  # noqa: E402
from lowlight_restoration.flux2_trajectory_endpoint import (  # noqa: E402
    denoise_cfg_split_gradient,
    validate_trajectory_split,
)
from smoke.train_foundir_blur_only_flux2_pixel_fidelity_ddp import validate_config  # noqa: E402


CONFIG = ROOT / "configs/foundir_blur_only_flux2_trajectory_endpoint_gate_v1.json"


def config():
    return json.loads(CONFIG.read_text())


def test_frozen_trajectory_config_contract():
    payload = config()
    validate_config(payload)
    assert payload["training"]["max_steps_per_arm"] == 200
    assert payload["trajectory"]["detached_prefix_steps"] == 46
    assert payload["trajectory"]["differentiable_suffix_steps"] == 4
    assert payload["training"]["arms"][1]["endpoint_lpips_weight"] == 0.02
    assert payload["scope"]["distillation"] is False
    assert payload["gate"]["full_scale_automatic"] is False


def test_trajectory_config_rejects_shortcut_or_development_selection():
    payload = config()
    payload["trajectory"]["surrogate_shortcut"] = True
    with pytest.raises(RuntimeError, match="trajectory solver contract"):
        validate_config(payload)
    payload = config()
    payload["data"]["development_used_for_selection"] = True
    with pytest.raises(RuntimeError, match="development GT"):
        validate_config(payload)


class ToyFlux(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.2))

    def forward(self, *, x, x_ids, timesteps, ctx, ctx_ids, guidance):
        del x_ids, ctx_ids, guidance
        context_term = ctx.mean(dim=tuple(range(1, ctx.ndim))).reshape(-1, 1, 1)
        return self.scale * x + timesteps[:, None, None] + context_term


def inputs():
    torch.manual_seed(7)
    image = torch.randn(1, 3, 4)
    image_ids = torch.zeros(1, 3, 4)
    reference = torch.randn(1, 2, 4)
    reference_ids = torch.ones(1, 2, 4)
    context = torch.tensor([[[0.0]], [[0.3]]])
    context_ids = torch.zeros(2, 1, 4)
    schedule = [1.0, 0.8, 0.5, 0.2, 0.0]
    return image, image_ids, reference, reference_ids, context, context_ids, schedule


def test_split_must_cover_every_transition():
    with pytest.raises(ValueError, match="expected 4"):
        validate_trajectory_split([1, 0.8, 0.5, 0.2, 0], detached_prefix_steps=2, differentiable_suffix_steps=1)


def test_split_matches_official_cfg_without_gradients():
    model = ToyFlux()
    image, ids, reference, reference_ids, context, context_ids, schedule = inputs()
    with torch.no_grad():
        expected = denoise_cfg(
            model, image, ids, context, context_ids, schedule, 1.0,
            img_cond_seq=reference, img_cond_seq_ids=reference_ids,
        )
        actual = denoise_cfg_split_gradient(
            model, image, ids, context, context_ids, schedule, 1.0,
            reference_tokens=reference, reference_ids=reference_ids,
            detached_prefix_steps=2, differentiable_suffix_steps=2,
            checkpoint_suffix=False,
        )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_prefix_is_detached_and_suffix_reaches_model():
    model = ToyFlux()
    image, ids, reference, reference_ids, context, context_ids, schedule = inputs()
    image.requires_grad_(True)
    output = denoise_cfg_split_gradient(
        model, image, ids, context, context_ids, schedule, 1.0,
        reference_tokens=reference, reference_ids=reference_ids,
        detached_prefix_steps=2, differentiable_suffix_steps=2,
        checkpoint_suffix=True,
    )
    output.square().mean().backward()
    assert image.grad is None
    assert model.scale.grad is not None and torch.isfinite(model.scale.grad)
    assert float(model.scale.grad.abs()) > 0


def test_checkpointed_suffix_matches_output_and_gradient():
    gradients = []
    outputs = []
    for use_checkpoint in (False, True):
        model = ToyFlux()
        image, ids, reference, reference_ids, context, context_ids, schedule = inputs()
        output = denoise_cfg_split_gradient(
            model, image, ids, context, context_ids, schedule, 1.0,
            reference_tokens=reference, reference_ids=reference_ids,
            detached_prefix_steps=2, differentiable_suffix_steps=2,
            checkpoint_suffix=use_checkpoint,
        )
        output.square().mean().backward()
        outputs.append(output.detach())
        gradients.append(model.scale.grad.detach())
    torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
    torch.testing.assert_close(gradients[0], gradients[1], rtol=1e-6, atol=1e-7)
