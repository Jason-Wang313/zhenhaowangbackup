"""Energy read from pretrained FLUX tokens, with independent in-FLUX LoRA.

No CNN, image scorer backbone, Restormer, or RGB refiner is constructed here.
Generation and energy share immutable pretrained/restoration weights. A
context-local adapter gate is captured explicitly by checkpoint recomputation;
there is no state_dict/weight swap while an autograd graph is alive.
"""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.checkpoint import checkpoint

from .lora import LoRALinear

_ENERGY_MODE = ContextVar('flux_native_energy_mode', default=False)
_CHECKPOINTED_ENERGY_ATTENTION = ContextVar('flux_native_checkpointed_attention', default=True)


@contextmanager
def energy_attention_mode(checkpointed):
    token = _CHECKPOINTED_ENERGY_ATTENTION.set(bool(checkpointed))
    try:
        yield
    finally:
        _CHECKPOINTED_ENERGY_ATTENTION.reset(token)


def install_energy_attention():
    # A context-local dispatch at the native attention boundary. Frozen
    # generation calls the original function unchanged, even while an energy
    # graph lives. Never patch torch's global SDPA or mutate native source.
    import flux2.model as native
    if getattr(native.causal_attn_fn, '_native_energy_dispatch', False):
        return
    original = native.causal_attn_fn

    def dispatch(q, k, v, num_txt_tokens, num_ref_tokens, kv_cache=None):
        if not _ENERGY_MODE.get() or not _CHECKPOINTED_ENERGY_ATTENTION.get():
            return original(q, k, v, num_txt_tokens, num_ref_tokens, kv_cache)
        if num_ref_tokens != 0 or kv_cache is not None:
            raise RuntimeError('energy attention requires full keys without reference KV cache')
        from .flux_native_attention import checkpointed_attention
        # Preserve the exact native concatenation/layout path as well as SDPA.
        def native_call(q, k, v):
            return original(q, k, v, num_txt_tokens, num_ref_tokens, kv_cache)
        return checkpointed_attention(q, k, v, native_call)

    dispatch._native_energy_dispatch = True
    native.causal_attn_fn = dispatch


@contextmanager
def adapter_mode(enabled):
    token = _ENERGY_MODE.set(bool(enabled))
    try:
        yield
    finally:
        _ENERGY_MODE.reset(token)


class EnergyAdapterLinear(nn.Module):
    """Exactly original(x) when off; additive FP32 trainable LoRA when on."""
    def __init__(self, original, rank, alpha):
        super().__init__()
        if not isinstance(original, LoRALinear) or rank < 1 or alpha <= 0:
            raise ValueError('requires a frozen restoration LoRALinear')
        self.original = original.requires_grad_(False)
        base = original.base
        self.energy_a = nn.Linear(base.in_features, rank, bias=False,
                                  device=base.weight.device, dtype=torch.float32)
        self.energy_b = nn.Linear(rank, base.out_features, bias=False,
                                  device=base.weight.device, dtype=torch.float32)
        nn.init.kaiming_uniform_(self.energy_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.energy_b.weight)
        self.scale = float(alpha) / rank

    def forward(self, value):
        original = self.original(value)
        if not _ENERGY_MODE.get():
            return original
        # Explicit FP32 adapter math; do not let an outer BF16 autocast erase
        # tiny adapter/energy gradients. Cast only the additive output.
        with torch.autocast(value.device.type, enabled=False):
            residual = self.energy_b(self.energy_a(value.to(self.energy_a.weight.dtype))) * self.scale
        return original + residual.to(original.dtype)


def install_checkpointing(flow, counters):
    for block in (*flow.double_blocks, *flow.single_blocks):
        if getattr(block, '_native_energy_checkpointed', False):
            raise RuntimeError('checkpoint wrapper already installed')
        original = block.forward_kv_extract

        def wrapped(*args, _original=original, **kwargs):
            enabled = _ENERGY_MODE.get()
            checkpointed = _CHECKPOINTED_ENERGY_ATTENTION.get()

            def call(*values, **options):
                counters['block_forward'] += 1
                with adapter_mode(enabled), energy_attention_mode(checkpointed), (sdpa_kernel(SDPBackend.MATH) if enabled else nullcontext()):
                    return _original(*values, **options)

            if torch.is_grad_enabled():
                return checkpoint(call, *args, use_reentrant=False, **kwargs)
            return call(*args, **kwargs)

        block.forward_kv_extract = wrapped
        block._native_energy_checkpointed = True


class FluxNativeEnergy(nn.Module):
    def __init__(self, flow, rank=16, alpha=16, *, expected_adapters=50,
                 checkpoint_blocks=True):
        super().__init__()
        install_energy_attention()
        self.flow = flow.eval().requires_grad_(False)
        names = [n for n, m in flow.named_modules() if isinstance(m, LoRALinear)]
        if len(names) != expected_adapters:
            raise ValueError(f'expected {expected_adapters} restoration adapters, got {len(names)}')
        for name in names:
            parent, leaf = name.rsplit('.', 1)
            owner = flow.get_submodule(parent)
            setattr(owner, leaf, EnergyAdapterLinear(getattr(owner, leaf), rank, alpha))
        device = next(flow.parameters()).device
        self.condition = nn.Linear(5, flow.txt_in.in_features, device=device, dtype=torch.float32)
        self.readout = nn.Linear(flow.hidden_size, 1, bias=False, device=device, dtype=torch.float32)
        nn.init.normal_(self.condition.weight, std=.01)
        nn.init.zeros_(self.condition.bias)
        nn.init.normal_(self.readout.weight, std=.001)
        self.counters = dict(energy_forward=0, generation_forward=0, block_forward=0)
        if checkpoint_blocks:
            install_checkpointing(flow, self.counters)
        self.architecture_receipt()

    def generation(self, **kwargs):
        self.counters['generation_forward'] += 1
        with adapter_mode(False):
            return self.flow(**kwargs)

    def forward(self, state, ids, timestep, reference, reference_ids,
                proposal_reference, proposal_ids, context, context_ids, descriptor):
        """Time-aware latent energy; no target/GT argument, no pixel scorer."""
        from flux2.model import timestep_embedding
        flow = self.flow
        if flow.use_guidance_embed:
            raise ValueError('this contract is Klein base 4B without guidance embedding')
        batch, candidate_tokens, _ = state.shape
        if descriptor.shape != (batch, 5) or timestep.shape != (batch,):
            raise ValueError('per-image descriptor/time shape mismatch')
        if context.shape[0] != batch or reference.shape[0] != batch or proposal_reference.shape[0] != batch:
            raise ValueError('one matching context/reference per image required')
        dtype = flow.img_in.weight.dtype
        self.counters['energy_forward'] += 1
        with adapter_mode(True), sdpa_kernel(SDPBackend.MATH):
            with torch.autocast(state.device.type, enabled=False):
                condition = self.condition(descriptor.to(self.condition.weight.dtype))[:, None]
            ctx = torch.cat((context.to(dtype), condition.to(dtype)), dim=1)
            condition_ids = context_ids[:, -1:].clone()
            condition_ids[..., 3] = context_ids[..., 3].amax(dim=1, keepdim=True) + 1
            ctx_ids = torch.cat((context_ids, condition_ids), dim=1)
            tokens = torch.cat((state.to(dtype), reference.to(dtype), proposal_reference.to(dtype)), dim=1)
            image_ids = torch.cat((ids, reference_ids, proposal_ids), dim=1)
            vec = flow.time_in(timestep_embedding(timestep.to(dtype), 256))
            mod_img = flow.double_stream_modulation_img(vec)
            mod_txt = flow.double_stream_modulation_txt(vec)
            mod_single, _ = flow.single_stream_modulation(vec)
            img, txt = flow.img_in(tokens), flow.txt_in(ctx)
            pe_img, pe_txt = flow.pe_embedder(image_ids), flow.pe_embedder(ctx_ids)
            # Same full-attention/reference-coordinate convention as the
            # frozen native model.forward, not a detached reference KV cache.
            for block in flow.double_blocks:
                img, txt, _ = block.forward_kv_extract(
                    img, txt, pe_img, pe_txt, mod_img, mod_txt, num_ref_tokens=0)
            joined = torch.cat((txt, img), dim=1)
            pe = torch.cat((pe_txt, pe_img), dim=2)
            for block in flow.single_blocks:
                joined, _ = block.forward_kv_extract(joined, pe, mod_single, ctx.shape[1], num_ref_tokens=0)
            features = joined[:, ctx.shape[1]:ctx.shape[1]+candidate_tokens]
            with torch.autocast(state.device.type, enabled=False):
                local_scores = self.readout(features.to(self.readout.weight.dtype))
            return local_scores.mean(dim=(1, 2))

    def architecture_receipt(self):
        forbidden = (nn.Conv1d, nn.Conv2d, nn.Conv3d,
                     nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)
        if any(isinstance(m, forbidden) for m in self.modules()):
            raise RuntimeError('convolution found in FLUX energy graph')
        trainable = {}
        for name, parameter in self.named_parameters():
            if parameter.requires_grad:
                if not ('.energy_a.' in name or '.energy_b.' in name
                        or name.startswith(('condition.', 'readout.'))):
                    raise RuntimeError('unexpected trainable parameter: '+name)
                trainable[name] = parameter.numel()
            elif parameter.grad is not None:
                raise RuntimeError('gradient on frozen generator parameter: '+name)
        return dict(backbone='pretrained_FLUX.2_Klein_base_4B',
                    energy_adapters=sum(isinstance(m, EnergyAdapterLinear) for m in self.modules()),
                    trainable_parameters=sum(trainable.values()), trainable=trainable,
                    energy_attention='native_full_math_attention_isolated_autograd_vjp',
                    external_image_backbone=False, cnn_energy=False, restormer=False)

    def trainable_state(self):
        return {n: p.detach().cpu().clone() for n, p in self.named_parameters() if p.requires_grad}

    def load_trainable_state(self, state):
        expected = {n: p for n, p in self.named_parameters() if p.requires_grad}
        if set(state) != set(expected):
            raise ValueError('energy checkpoint parameter key mismatch')
        with torch.no_grad():
            for name, parameter in expected.items():
                if state[name].shape != parameter.shape:
                    raise ValueError('energy checkpoint parameter shape mismatch: '+name)
                parameter.copy_(state[name].to(parameter))


def energy_value_and_gradient(energy, state, *, training, recompute_derivative=True):
    """Checkpoint the differentiable VJP, not only the energy forward blocks.

    Block checkpointing alone does not discard tensors saved by create_graph's
    derivative operations. An outer non-reentrant checkpoint also rematerializes
    those operations when the training loss backpropagates. Nothing is detached;
    closed-over energy parameters remain differentiable.
    """
    def value_and_gradient(value):
        with torch.enable_grad():
            score = energy(value)
            gradient = torch.autograd.grad(score.sum(), value, create_graph=training)[0]
        return score, gradient

    if training and recompute_derivative:
        return checkpoint(value_and_gradient, state, use_reentrant=False)
    return value_and_gradient(state)


def unroll(energy, initial, *, updates=3, step_size=.005, radius=.02, training,
           recompute_derivative=True, observe=None):
    """Same target-free update at training/inference; no training detach."""
    if updates != 3 or not 0 < step_size <= radius:
        raise ValueError('requires the declared full three-update trajectory')
    state = initial.detach().clone().requires_grad_(True)
    origin = state.detach()
    states = [state]
    scores = []
    with torch.enable_grad():
        for update in range(updates):
            if observe is not None:
                observe(f'unroll_{update+1}_start')
            score, gradient = energy_value_and_gradient(
                energy, state, training=training, recompute_derivative=recompute_derivative)
            if not bool(torch.isfinite(score).all() and torch.isfinite(gradient).all()):
                raise RuntimeError('nonfinite native FLUX energy/latent gradient')
            axes = tuple(range(1, gradient.ndim))
            norm = gradient.square().mean(dim=axes, keepdim=True).clamp_min(1e-16).sqrt()
            delta = state - step_size * gradient / norm - origin
            length = delta.square().mean(dim=axes, keepdim=True).clamp_min(1e-16).sqrt()
            state = origin + delta * (radius / length).clamp(max=1.)
            if not training:
                state = state.detach().requires_grad_(True)
            states.append(state)
            scores.append(score)
            if observe is not None:
                observe(f'unroll_{update+1}_done')
        scores.append(energy(state))
    return states, scores


def direct_quality_loss(images, target, proposal, perceptual):
    """Actual rendered images, not fitted score/gradient-cache supervision."""
    if len(images) != 3:
        raise ValueError('all three differentiable rendered images are required')
    with torch.no_grad():
        base_mse = ((proposal-target)/2).square().mean().clamp_min(1e-8)
        base_lpips = perceptual(proposal, target).mean().clamp_min(1e-4)
    losses, diagnostics = [], []
    for image in images:
        mse = ((image-target)/2).square().mean()
        lpips = perceptual(image, target).mean()
        relative_mse, relative_lpips = mse/base_mse, lpips/base_lpips
        harm = F.relu(relative_mse-1) + F.relu(relative_lpips-1)
        losses.append(relative_mse+relative_lpips+harm)
        diagnostics.append(dict(mse=float(mse.detach()), lpips=float(lpips.detach())))
    return torch.stack(losses).mean(), diagnostics


def pareto_ranking_loss(scores, costs, *, margin=.05):
    """Only empirically dominant pairs, no fake ranking of conflicting metrics."""
    if scores.ndim != 1 or costs.shape != (len(scores), 2):
        raise ValueError('candidate scalar energies and paired MSE/LPIPS costs required')
    detached = costs.detach()
    normalizer = detached.mean(dim=0).clamp_min(1e-8)
    normalized = detached / normalizer
    losses = []
    for i in range(len(scores)):
        for j in range(i+1, len(scores)):
            gap = normalized[j]-normalized[i]
            if bool((gap >= 0).all()) and float(gap.max()) > .01:
                losses.append(F.relu(margin*gap.mean().clamp(max=5) - (scores[j]-scores[i])))
            elif bool((gap <= 0).all()) and float(gap.min()) < -.01:
                losses.append(F.relu(margin*(-gap).mean().clamp(max=5) - (scores[i]-scores[j])))
    return torch.stack(losses).mean() if losses else scores.sum()*0


def select_without_target(scores, margin=.001):
    """Predicted-energy choice, NOT a proof that actual quality cannot regress."""
    values = [float(v.detach()) for v in scores]
    if len(values) != 4 or not all(math.isfinite(v) for v in values):
        raise ValueError('finite baseline plus three scores required')
    best = min(range(4), key=lambda k: values[k])
    return best if values[best] < values[0]-margin else 0
