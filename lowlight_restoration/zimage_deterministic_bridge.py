"""Pure Z-Image deterministic LQ-to-GT latent bridge with persistent conditioning."""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from lowlight_restoration.zimage_conditional_teacher import aligned_tokens, euler_step, shifted_grid
from lowlight_restoration.foundir_zimage_paired_flow import sha256

ROOT = Path(__file__).resolve().parents[1]
PROTECTED = (
    "qwen_degradation_conditioning", "mixed_degradations", "independent_restoration_branch",
    "distillation", "decoder_finetuning", "sealed_internal_accessed",
    "official_foundir_test_accessed", "development_gt_optimization",
)


def load_settings(path: Path):
    settings = json.loads(Path(path).read_text())
    if settings.get("status") != "user_authorized_blur_only_zimage_deterministic_bridge_tiny_v1":
        raise ValueError("unexpected deterministic bridge experiment")
    if any(settings.get(key) is not False for key in PROTECTED):
        raise ValueError("protected scope or independent branch enabled")
    if settings.get("objective") != "deterministic_lq_latent_to_gt_latent_rectified_flow":
        raise ValueError("unexpected bridge objective")
    if settings.get("solver_initialization") != "exact_lq_vae_latent":
        raise ValueError("deterministic solver must start at LQ latent")
    if settings.get("identity_path") != "zero_initialized_per_channel_velocity_gate":
        raise ValueError("identity path changed")
    if settings.get("condition_scales") != [1, 2, 4]:
        raise ValueError("condition pyramid changed")
    layers = settings.get("condition_layer_indices")
    if layers != sorted(set(layers)) or not layers or layers[0] < 0 or layers[-1] >= 30:
        raise ValueError("invalid condition injection depths")
    if (settings.get("crop_size"), settings.get("lora_rank"), settings.get("trainable_dtype")) != (512, 64, "float32"):
        raise ValueError("model capacity/precision changed")
    if (settings.get("teacher_steps"), settings.get("inference_shift"), settings.get("cfg_scale")) != (50, 6.0, 1.0):
        raise ValueError("teacher inference contract changed")
    if (settings.get("steps"), settings.get("global_batch"), settings.get("snapshot_steps")) != (1000, 8, [0, 250, 500, 1000]):
        raise ValueError("tiny budget changed")
    if settings.get("microbatch_per_rank") != 1 or settings.get("gradient_accumulation") != 1:
        raise ValueError("eight-rank batch arithmetic changed")
    for key in ("learning_rate", "condition_learning_rate", "velocity_gate_learning_rate",
                "max_grad_norm"):
        if not math.isfinite(settings[key]) or settings[key] <= 0:
            raise ValueError(f"invalid {key}")
    expected_loss = {"flow_mse", "clean_latent_charbonnier", "rgb_charbonnier", "lpips",
                     "charbonnier_epsilon"}
    if set(settings.get("loss", {})) != expected_loss or any(
            not math.isfinite(value) or value <= 0 for value in settings["loss"].values()):
        raise ValueError("invalid fidelity loss")
    data = settings["data"]
    if (data["train_records"], data["train_source_groups"], data["development_records"],
            data["development_source_groups"], data["development_is_descriptive"]) != (16, 16, 16, 16, True):
        raise ValueError("tiny data contract changed")
    if settings["output_dir"] != "checkpoints/foundir_blur_only_zimage_deterministic_bridge_tiny_v1":
        raise ValueError("unexpected output directory")
    base_path = ROOT / settings["base_config"]
    if sha256(base_path) != settings["base_config_sha256"]:
        raise ValueError("base config changed")
    return settings, json.loads(base_path.read_text())


def bridge_pair(gt: torch.Tensor, lq: torch.Tensor, sigma: float):
    """State at sigma and d(state)/d(sigma), with sigma 1=LQ and 0=GT."""
    if gt.shape != lq.shape or not 0 <= float(sigma) <= 1:
        raise ValueError("invalid deterministic bridge pair")
    gt, lq = gt.float(), lq.float()
    return (1 - float(sigma)) * gt + float(sigma) * lq, lq - gt


def clean_from_velocity(state: torch.Tensor, velocity: torch.Tensor, sigma: float):
    if state.shape != velocity.shape or not 0 <= float(sigma) <= 1:
        raise ValueError("invalid clean estimate")
    return state.float() - float(sigma) * velocity.float()


def pyramid_tokens(dit, lq: torch.Tensor, prompt, scales=(1, 2, 4)):
    """Aligned 2x2 tokens from a fixed, non-learned LQ latent pyramid."""
    if tuple(scales) != (1, 2, 4):
        raise ValueError("unexpected pyramid scales")
    tokens = []
    for scale in scales:
        value = lq.float()
        if scale > 1:
            pooled = F.avg_pool2d(value, kernel_size=scale, stride=scale)
            value = F.interpolate(pooled, size=lq.shape[-2:], mode="nearest")
        part, _, _ = aligned_tokens(dit, value.to(lq.dtype), prompt)
        tokens.append(part[0].float())
    if any(part.shape[:-1] != tokens[0].shape[:-1] for part in tokens):
        raise RuntimeError("condition pyramid token geometry differs")
    return torch.cat(tokens, dim=-1)


def multidepth_velocity(dit, projections, velocity_gate, state, lq, sigma, prompt, grad=False):
    """Native Base DiT with zero-init LQ injections and an exact identity gate."""
    if state.shape != lq.shape:
        raise ValueError("LQ/state geometry differs")
    while isinstance(prompt, list):
        prompt = prompt[0]
    timestep = 1000 - state.new_tensor([float(sigma)], dtype=torch.float32) * 1000
    modulation = dit.t_embedder(timestep)
    xs, caps, meta = aligned_tokens(dit, state, prompt)
    condition = pyramid_tokens(dit, lq, prompt)
    x = dit.all_x_embedder["2-1"](xs[0])
    x = x + projections["input"](condition).to(x.dtype)
    x = torch.where(torch.cat(meta["x_pad_mask"])[:, None], dit.x_pad_token.to(x), x)
    xf = dit.rope_embedder(torch.cat(meta["x_pos_ids"])).unsqueeze(0)
    x = x.unsqueeze(0)

    def run(layer, values, freqs, adaln=None):
        def call(a, f, c):
            return layer(x=a, attn_mask=None, freqs_cis=f, adaln_input=c)
        return checkpoint(call, values, freqs, adaln, use_reentrant=False) if grad else call(values, freqs, adaln)

    for layer in dit.noise_refiner:
        x = run(layer, x, xf, modulation)
    cap = dit.cap_embedder(caps[0])
    cap = torch.where(torch.cat(meta["cap_pad_mask"])[:, None], dit.cap_pad_token.to(cap), cap)
    cf = dit.rope_embedder(torch.cat(meta["cap_pos_ids"])).unsqueeze(0)
    cap = cap.unsqueeze(0)
    for layer in dit.context_refiner:
        cap = run(layer, cap, cf)
    image_tokens = x.shape[1]
    unified, freqs = torch.cat((x, cap), dim=1), torch.cat((xf, cf), dim=1)
    for index, layer in enumerate(dit.layers):
        key = str(index)
        if key in projections:
            injection = projections[key](condition).to(unified.dtype).unsqueeze(0)
            unified = torch.cat((unified[:, :image_tokens] + injection,
                                 unified[:, image_tokens:]), dim=1)
        unified = run(layer, unified, freqs, modulation)
    output = dit.all_final_layer["2-1"](unified, modulation)
    raw = -dit.unpatchify([output[0]], meta["x_size"])[0].permute(1, 0, 2, 3)
    return raw.float() * velocity_gate.float()


class DeterministicBridgeTeacher(nn.Module):
    def __init__(self, base_config, settings, device):
        super().__init__()
        from smoke.train_foundir_blur_only_zimage_teacher_paired_flow_ddp import ZImagePairedFlowTeacherModule
        model_config = copy.deepcopy(base_config)
        model_config["lora"]["rank"] = settings["lora_rank"]
        model_config["lora"]["alpha"] = settings["lora_rank"]
        native = ZImagePairedFlowTeacherModule(model_config, device)
        self.pipe = native.pipe
        for parameter in self.pipe.parameters():
            if parameter.requires_grad:
                parameter.data = parameter.data.float()
        width = 64 * len(settings["condition_scales"])
        dim = self.pipe.dit.dim
        keys = ["input", *map(str, settings["condition_layer_indices"])]
        self.lq_projections = nn.ModuleDict({key: nn.Linear(width, dim, bias=False,
                                                           device=device, dtype=torch.float32)
                                             for key in keys})
        for projection in self.lq_projections.values():
            nn.init.zeros_(projection.weight)
        self.velocity_gate = nn.Parameter(torch.zeros(1, 16, 1, 1, device=device, dtype=torch.float32))
        self.scales = tuple(settings["condition_scales"])
        if any(parameter.requires_grad for parameter in self.pipe.vae_encoder.parameters()) or any(
                parameter.requires_grad for parameter in self.pipe.vae_decoder.parameters()):
            raise RuntimeError("VAE must remain frozen")

    def forward(self, state, lq, sigma, prompt, grad=None):
        return multidepth_velocity(self.pipe.dit, self.lq_projections, self.velocity_gate,
                                   state.to(self.pipe.torch_dtype), lq.to(self.pipe.torch_dtype),
                                   sigma, prompt, torch.is_grad_enabled() if grad is None else grad)

    @torch.no_grad()
    def restore(self, lq, prompt, steps=50, shift=6.0):
        state = lq.float().clone()
        grid = shifted_grid(steps, shift)
        for sigma, following in zip(grid[:-1], grid[1:]):
            velocity = self(state, lq, float(sigma), prompt, grad=False)
            state = euler_step(state, velocity, sigma, following)
            if not torch.isfinite(state).all():
                raise RuntimeError("nonfinite deterministic solver state")
        return state
