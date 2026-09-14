from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / ".deps/zimage"))
sys.path.insert(0, str(ROOT / "third_party/diffsynth-studio"))
from lowlight_restoration.zimage_conditional_teacher import (
    aligned_tokens, conditioned_velocity, euler_step, load_settings,
    noise_flow_pair, shifted_grid, stage_gate,
)
from lowlight_restoration.zimage_padding_compat import build_padding_function
from smoke.prepare_zimage_conditional_pilot import select_groups

CONFIG = ROOT / "configs/foundir_blur_only_zimage_conditional_teacher_v1.json"


def test_gaussian_flow_endpoints_and_solver_sign():
    gt, noise = torch.randn(1, 16, 4, 4), torch.randn(1, 16, 4, 4)
    clean, target = noise_flow_pair(gt, noise, 0)
    state, _ = noise_flow_pair(gt, noise, 1)
    assert torch.equal(clean, gt) and torch.equal(state, noise)
    for sigma, following in zip(shifted_grid(50)[:-1], shifted_grid(50)[1:]):
        state = euler_step(state, target, sigma, following)
    torch.testing.assert_close(state, gt, atol=2e-6, rtol=2e-6)


def test_grid_matches_native_base():
    from diffsynth.diffusion.flow_match import FlowMatchScheduler
    native = FlowMatchScheduler("Z-Image")
    native.set_timesteps(50, shift=6)
    torch.testing.assert_close(shifted_grid(50)[:-1], native.sigmas, rtol=0, atol=1e-7)
    assert shifted_grid(50)[-1] == 0


def small_dit():
    from diffsynth.models.z_image_dit import ZImageDiT
    torch.manual_seed(4)
    dit = ZImageDiT(dim=128, n_layers=1, n_refiner_layers=1, n_heads=1, n_kv_heads=1, cap_feat_dim=32)
    with torch.no_grad():
        dit.x_pad_token.zero_()
        dit.cap_pad_token.zero_()
    return dit.to(torch.bfloat16).requires_grad_(False)


def test_patch_correspondence_and_zero_native_gradient_equivalence():
    from diffsynth.pipelines import z_image
    dit = small_dit()
    projection = torch.nn.Linear(64, 128, bias=False)
    torch.nn.init.zeros_(projection.weight)
    prompt = torch.randn(7, 32, dtype=torch.bfloat16)
    state = torch.randn(1, 16, 4, 4, dtype=torch.bfloat16).requires_grad_(True)
    lq = torch.randn_like(state)
    xs, _, _ = aligned_tokens(dit, state, prompt)
    expected = state[0, :, :2, :2].permute(1, 2, 0).flatten()
    assert torch.equal(xs[0][0], expected)
    _, native, _ = build_padding_function(z_image)
    a = native(dit=dit, latents=state, timestep=state.new_tensor([500]), prompt_embeds=prompt, use_gradient_checkpointing=True)
    a.float().square().mean().backward()
    gradient = state.grad.clone()
    state.grad = None
    b = conditioned_velocity(dit, projection, state, lq, .5, prompt, grad=True)
    b.float().square().mean().backward()
    assert torch.equal(a, b)
    assert torch.equal(gradient, state.grad)
    assert projection.weight.grad.abs().max() > 0
    with torch.no_grad():
        projection.weight.add_(-.01 * projection.weight.grad)
        changed = conditioned_velocity(dit, projection, state, lq, .5, prompt)
        swapped = conditioned_velocity(dit, projection, state, torch.zeros_like(lq), .5, prompt)
    assert not torch.equal(changed, swapped)
    with pytest.raises(ValueError):
        conditioned_velocity(dit, projection, state, lq[:, :, :-1], .5, prompt)


def test_tiny_generalization_is_descriptive_pilot_is_not():
    s, _ = load_settings(CONFIG)
    train = dict(psnr_gain_db=2., lpips_improvement=.05)
    assert stage_gate("tiny", train, 1, .5, s)["automatic_metric_gate_pass"]
    assert not stage_gate("tiny", train, 1, .9, s)["automatic_metric_gate_pass"]
    dev = dict(psnr_gain_db=-.1, lpips_improvement=.02)
    assert not stage_gate("pilot", dev, 1, .5, s)["automatic_metric_gate_pass"]
    dev["psnr_gain_db"] = .1
    assert stage_gate("pilot", dev, 1, .5, s)["automatic_metric_gate_pass"]


def test_protected_flags_and_batch_rejected(tmp_path):
    s = json.loads(CONFIG.read_text())
    for mutation in ({"sealed_internal_accessed": True}, {"trainable_dtype": "bfloat16"}):
        bad = copy.deepcopy(s)
        bad.update(mutation)
        p = tmp_path / "bad.json"
        p.write_text(json.dumps(bad))
        with pytest.raises(ValueError):
            load_settings(p)
    bad = copy.deepcopy(s)
    bad["stages"]["pilot"]["global_batch"] = 31
    p.write_text(json.dumps(bad))
    with pytest.raises(ValueError):
        load_settings(p)


def test_source_group_selection_keeps_fixed_rows_and_is_order_independent():
    rows = []
    for group in range(8):
        for record in range(2):
            rows.append(dict(record_id=f"{group}_{record}", source_group=str(group), split="train",
                             width="512", height="512", category="01Blur", composition_size="1",
                             lq_path="lq", gt_path="gt", lq_sha256="a", gt_sha256="b"))
    fixed = [dict(rows[0], crop_x="0", crop_y="0", role="tiny_overfit_train")]
    a = select_groups(rows, fixed, 5, "pilot_train", 4)
    b = select_groups(list(reversed(rows)), fixed, 5, "pilot_train", 4)
    assert a == b and a[0]["record_id"] == rows[0]["record_id"]
    assert len({r["source_group"] for r in a}) == 5


def test_pilot_requires_review_before_training():
    from smoke.train_zimage_conditional_teacher import validate_review
    s, _ = load_settings(CONFIG)
    with pytest.raises(ValueError, match="requires"):
        validate_review(s, "unused", None)


def _accumulation_worker(rank, rendezvous):
    import torch.distributed as dist
    from lowlight_restoration.zimage_latent_first import average_gradients
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    torch.manual_seed(19)
    model = torch.nn.Linear(3, 2)
    reference = copy.deepcopy(model)
    x = torch.randn(4, 3)
    target = torch.randn(4, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=0)
    ref_optimizer = torch.optim.AdamW(reference.parameters(), lr=.001, weight_decay=0)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        ref_optimizer.zero_grad(set_to_none=True)
        for index in (rank, rank + 2):
            ((model(x[index:index+1]) - target[index:index+1]).square().mean() / 2).backward()
        average_gradients(model.parameters(), 2)
        (reference(x) - target).square().mean().backward()
        for a, b in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(a.grad, b.grad, atol=2e-7, rtol=2e-6)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        torch.nn.utils.clip_grad_norm_(reference.parameters(), 1)
        optimizer.step()
        ref_optimizer.step()
        for a, b in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(a, b, atol=2e-7, rtol=2e-6)
    dist.destroy_process_group()


def test_accumulated_two_rank_updates_match_global_batch(tmp_path):
    torch.multiprocessing.spawn(_accumulation_worker, args=("file://" + str(tmp_path / "gloo"),), nprocs=2, join=True)


def test_rgb_crop_cache_preserves_paired_augmentation(monkeypatch):
    from types import SimpleNamespace
    import numpy as np
    from PIL import Image
    from smoke import train_zimage_conditional_teacher as runner
    array = np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3)
    pair = (Image.fromarray(array), Image.fromarray(255 - array))
    pipe = SimpleNamespace(preprocess_image=lambda image: torch.from_numpy(np.array(image).copy()),
                           vae_encoder=lambda tensor: tensor)
    model = SimpleNamespace(pipe=pipe)
    monkeypatch.setattr(runner, "load_aligned_pair", lambda row, size: pair)
    expected = [runner.encode(model, {}, 8, transform) for transform in range(8)]
    def reject_read(*args):
        raise AssertionError("cached crop must not decode full PNG again")
    monkeypatch.setattr(runner, "load_aligned_pair", reject_read)
    for transform in range(8):
        cached = runner.encode(model, {}, 8, transform, pair)
        assert torch.equal(cached["lq_latent"], expected[transform]["lq_latent"])
        assert torch.equal(cached["gt_latent"], expected[transform]["gt_latent"])
        assert torch.equal(cached["gt_latent"], 255 - cached["lq_latent"])
    assert np.array_equal(np.array(pair[0]), array)
