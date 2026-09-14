"""Persistent aligned image conditioning for the pinned Z-Image Base DiT.

The native state/timestep/sign conventions are preserved. No mutable hooks or
module context are used, so activation recomputation receives the same inputs.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

ROOT = Path(__file__).resolve().parents[1]
DIT_SHA256 = "1cb527b1575cd1d070593062da3f929ec9bc5a6f906f6e7e9e83c48da9972d2e"
PROTECTED = ("qwen_degradation_conditioning", "mixed_degradations", "decoder_finetuning",
             "sealed_internal_accessed", "official_foundir_test_accessed", "development_gt_optimization")


def load_settings(path):
    from lowlight_restoration.foundir_zimage import sha256
    s = json.loads(Path(path).read_text())
    if s.get("status") != "authorized_blur_only_zimage_conditional_teacher_v1":
        raise ValueError("unexpected conditional teacher config")
    if any(s.get(k) is not False for k in PROTECTED):
        raise ValueError("protected scope changed")
    if s["initialization"] != "fresh_base_fresh_fp32_lora_zero_lq_projection":
        raise ValueError("failed adapter inheritance is forbidden")
    if s["objective"] != "conditional_noise_to_gt_rectified_flow" or s["sigma_sampling"] != "uniform_0_1":
        raise ValueError("flow objective changed")
    if s["teacher_steps"] != 50 or s["inference_shift"] != 6 or s["cfg_scale"] != 1:
        raise ValueError("teacher inference contract changed")
    if s["trainable_dtype"] != "float32" or s["lora_rank"] != 32 or s["crop_size"] != 512:
        raise ValueError("model precision/rank/resolution changed")
    for key in ("learning_rate", "condition_learning_rate", "max_grad_norm"):
        if not math.isfinite(s[key]) or s[key] <= 0:
            raise ValueError(f"invalid {key}")
    for stage in ("tiny", "pilot"):
        c = s["stages"][stage]
        if c["global_batch"] % 8 or c["global_batch"] < 8:
            raise ValueError("eight-rank batch arithmetic changed")
        if c["snapshot_steps"] != sorted(set(c["snapshot_steps"])) or c["snapshot_steps"][0] != 0 or c["snapshot_steps"][-1] != c["steps"]:
            raise ValueError("incomplete snapshot schedule")
        if c["output_dir"] != f"checkpoints/foundir_blur_only_zimage_conditional_teacher_v1_{stage}":
            raise ValueError("unexpected output directory")
    parent = ROOT / s["base_config"]
    if sha256(parent) != s["base_config_sha256"]:
        raise ValueError("base artifact config changed")
    dit = ROOT / "third_party/diffsynth-studio/diffsynth/models/z_image_dit.py"
    if sha256(dit) != DIT_SHA256:
        raise ValueError("upstream DiT changed; re-audit conditioning")
    return s, json.loads(parent.read_text())


def noise_flow_pair(gt, noise, sigma):
    if gt.shape != noise.shape or not 0 <= float(sigma) <= 1:
        raise ValueError("invalid flow state")
    return (1 - sigma) * gt.float() + sigma * noise.float(), noise.float() - gt.float()


def shifted_grid(steps, shift=6.0):
    if steps < 1 or shift <= 0:
        raise ValueError("invalid solver")
    # The pinned Base schedule uses linspace(1, 1/steps, steps), then zero.
    s = torch.linspace(1, 1 / steps, steps, dtype=torch.float32)
    return torch.cat((shift * s / (1 + (shift - 1) * s), torch.zeros(1)))


def euler_step(state, velocity, sigma, next_sigma):
    return state.float() + (float(next_sigma) - float(sigma)) * velocity.float()


def aligned_tokens(dit, latent, prompt):
    if latent.ndim != 4 or latent.shape[0] != 1 or latent.shape[1] != 16:
        raise ValueError("one 16-channel latent per call required")
    if any(n % 2 for n in latent.shape[-2:]):
        raise ValueError("latent geometry must support 2x2 patches")
    return dit.patchify_and_embed([latent.permute(1, 0, 2, 3)], [prompt])


def conditioned_velocity(dit, projection, state, lq, sigma, prompt, grad=False):
    """Native model_fn_z_image_turbo plus explicit W_y patch(LQ).

    That upstream function also serves Base weights; its name does not imply
    Turbo weights. Callers supply the scalar noise fraction, not the embedded
    time. Native embedding is 1000*(1-sigma); native final sign is negative.
    """
    if state.shape != lq.shape:
        raise ValueError("LQ/state geometry differs")
    while isinstance(prompt, list):
        prompt = prompt[0]
    timestep = 1000 - state.new_tensor([float(sigma)], dtype=torch.float32) * 1000
    t = dit.t_embedder(timestep)
    xs, caps, meta = aligned_tokens(dit, state, prompt)
    ys, _, _ = aligned_tokens(dit, lq, prompt)
    x = dit.all_x_embedder["2-1"](xs[0])
    condition = projection(ys[0].float()).to(x.dtype)
    x = x + condition
    x = torch.where(torch.cat(meta["x_pad_mask"])[:, None], dit.x_pad_token.to(x), x)
    xf = dit.rope_embedder(torch.cat(meta["x_pos_ids"])).unsqueeze(0)
    x = x.unsqueeze(0)

    def run(layer, values, freqs, modulation=None):
        def call(a, f, c):
            return layer(x=a, attn_mask=None, freqs_cis=f, adaln_input=c)
        return checkpoint(call, values, freqs, modulation, use_reentrant=False) if grad else call(values, freqs, modulation)

    for layer in dit.noise_refiner:
        x = run(layer, x, xf, t)
    cap = dit.cap_embedder(caps[0])
    cap = torch.where(torch.cat(meta["cap_pad_mask"])[:, None], dit.cap_pad_token.to(cap), cap)
    cf = dit.rope_embedder(torch.cat(meta["cap_pos_ids"])).unsqueeze(0)
    cap = cap.unsqueeze(0)
    for layer in dit.context_refiner:
        cap = run(layer, cap, cf)
    unified = torch.cat((x, cap), dim=1)
    freqs = torch.cat((xf, cf), dim=1)
    for layer in dit.layers:
        unified = run(layer, unified, freqs, t)
    output = dit.all_final_layer["2-1"](unified, t)
    return -dit.unpatchify([output[0]], meta["x_size"])[0].permute(1, 0, 2, 3)


class ConditionalTeacher(nn.Module):
    def __init__(self, config, settings, device):
        super().__init__()
        from smoke.train_foundir_blur_only_zimage_teacher_paired_flow_ddp import ZImagePairedFlowTeacherModule
        # Constructor loads only the pinned Base/VAE/native text weights and
        # fresh LoRA. Its paired-flow forward/scheduler are never used here.
        native = ZImagePairedFlowTeacherModule(config, device)
        self.pipe = native.pipe
        for p in self.pipe.parameters():
            if p.requires_grad:
                p.data = p.data.float()
        self.lq_projection = nn.Linear(64, self.pipe.dit.dim, bias=False, device=device, dtype=torch.float32)
        nn.init.zeros_(self.lq_projection.weight)
        if any(p.requires_grad for p in self.pipe.vae_encoder.parameters()) or any(p.requires_grad for p in self.pipe.vae_decoder.parameters()):
            raise RuntimeError("VAE must be frozen")

    def forward(self, state, lq, sigma, prompt, grad=None):
        return conditioned_velocity(self.pipe.dit, self.lq_projection,
                                    state.to(self.pipe.torch_dtype), lq.to(self.pipe.torch_dtype),
                                    sigma, prompt, torch.is_grad_enabled() if grad is None else grad)

    @torch.no_grad()
    def restore(self, lq, prompt, seed, steps=50, shift=6.0):
        generator = torch.Generator(device=lq.device).manual_seed(seed)
        state = torch.randn(lq.shape, generator=generator, device=lq.device, dtype=torch.float32)
        grid = shifted_grid(steps, shift)
        for sigma, following in zip(grid[:-1], grid[1:]):
            velocity = self(state, lq, sigma, prompt, grad=False)
            state = euler_step(state, velocity, sigma, following)
            if not torch.isfinite(state).all():
                raise RuntimeError("nonfinite solver state")
        return state


def fixed_seed(seed, *parts):
    return int.from_bytes(hashlib.sha256(":".join(map(str, (seed, *parts))).encode()).digest()[:8], "big") % (2**63 - 1)


def stage_gate(stage, metrics, initial_probe, final_probe, settings):
    c = settings["stages"][stage]["gate"]
    if not all(math.isfinite(v) for v in (*metrics.values(), initial_probe, final_probe)):
        raise ValueError("nonfinite gate")
    if stage == "tiny":
        checks = dict(train_psnr=metrics["psnr_gain_db"] > c["train_psnr_gain_db"],
                      train_lpips=metrics["lpips_improvement"] > c["train_lpips_improvement"],
                      fixed_probe=final_probe < c["probe_loss_ratio_max"] * initial_probe)
    else:
        checks = dict(development_psnr=metrics["psnr_gain_db"] > c["development_psnr_gain_db"],
                      development_lpips=metrics["lpips_improvement"] > c["development_lpips_improvement"])
    return dict(checks=checks, automatic_metric_gate_pass=all(checks.values()), visual_review_required=True)
