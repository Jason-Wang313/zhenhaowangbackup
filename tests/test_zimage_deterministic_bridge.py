import copy
import json
import os
from pathlib import Path

import pytest
import torch

os.environ["DIFFSYNTH_ATTENTION_IMPLEMENTATION"] = "torch"

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/foundir_blur_only_zimage_deterministic_bridge_tiny_v1.json"

from lowlight_restoration.zimage_deterministic_bridge import (
    bridge_pair, clean_from_velocity, load_settings, multidepth_velocity, pyramid_tokens,
)


def small_dit():
    from diffsynth.models.z_image_dit import ZImageDiT
    torch.manual_seed(19)
    model = ZImageDiT(dim=128, n_layers=2, n_refiner_layers=1, n_heads=1,
                      n_kv_heads=1, cap_feat_dim=32)
    with torch.no_grad():
        model.x_pad_token.zero_()
        model.cap_pad_token.zero_()
    return model.to(torch.bfloat16).requires_grad_(False)


def test_bridge_endpoints_target_and_clean_estimate():
    gt = torch.randn(1, 16, 4, 4)
    lq = torch.randn_like(gt)
    for sigma in (0, .2, .75, 1):
        state, target = bridge_pair(gt, lq, sigma)
        torch.testing.assert_close(state, (1 - sigma) * gt + sigma * lq)
        torch.testing.assert_close(target, lq - gt)
        torch.testing.assert_close(clean_from_velocity(state, target, sigma), gt)
    assert torch.equal(bridge_pair(gt, lq, 0)[0], gt)
    assert torch.equal(bridge_pair(gt, lq, 1)[0], lq)
    with pytest.raises(ValueError):
        bridge_pair(gt, lq[:, :, :-1], .5)


def test_three_scale_tokens_preserve_patch_correspondence():
    dit = small_dit()
    prompt = torch.randn(5, 32, dtype=torch.bfloat16)
    lq = torch.arange(1 * 16 * 4 * 4, dtype=torch.float32).reshape(1, 16, 4, 4).to(torch.bfloat16)
    tokens = pyramid_tokens(dit, lq, prompt)
    # Native Z-Image sequence padding expands four real 2x2 patches to 32
    # tokens; all three condition scales must follow that exact alignment.
    assert tokens.shape == (32, 192)
    expected = lq[0, :, :2, :2].permute(1, 2, 0).flatten().float()
    assert torch.equal(tokens[0, :64], expected)


def test_zero_gate_is_exact_identity_and_first_update_is_learnable():
    dit = small_dit()
    prompt = torch.randn(5, 32, dtype=torch.bfloat16)
    state = torch.randn(1, 16, 4, 4, dtype=torch.bfloat16)
    lq = torch.randn_like(state)
    projections = torch.nn.ModuleDict({key: torch.nn.Linear(192, 128, bias=False)
                                       for key in ("input", "0", "1")})
    for projection in projections.values():
        torch.nn.init.zeros_(projection.weight)
    gate = torch.nn.Parameter(torch.zeros(1, 16, 1, 1))
    output = multidepth_velocity(dit, projections, gate, state, lq, .5, prompt, grad=True)
    assert torch.equal(output, torch.zeros_like(output))
    target = torch.randn_like(output)
    (output - target).square().mean().backward()
    assert gate.grad is not None and gate.grad.abs().max() > 0
    assert all(projection.weight.grad is not None and projection.weight.grad.abs().max() == 0
               for projection in projections.values())
    with torch.no_grad():
        gate.add_(-.1 * gate.grad)
    dit.zero_grad(set_to_none=True)
    projections.zero_grad(set_to_none=True)
    changed = multidepth_velocity(dit, projections, gate, state, lq, .5, prompt, grad=True)
    changed.square().mean().backward()
    assert any(projection.weight.grad.abs().max() > 0 for projection in projections.values())


def test_config_locks_pure_zimage_layer1_scope(tmp_path):
    settings, base = load_settings(CONFIG)
    assert settings["independent_restoration_branch"] is False
    assert settings["objective"].startswith("deterministic_lq_latent_to_gt_latent")
    assert settings["data"]["development_is_descriptive"] is True
    assert base["model"]["name"] == "Tongyi-MAI/Z-Image"
    for mutation in ({"independent_restoration_branch": True},
                     {"objective": "conditional_noise_to_gt_rectified_flow"},
                     {"lora_rank": 32}):
        bad = copy.deepcopy(settings)
        bad.update(mutation)
        path = tmp_path / "bad.json"
        path.write_text(json.dumps(bad))
        with pytest.raises(ValueError):
            load_settings(path)
