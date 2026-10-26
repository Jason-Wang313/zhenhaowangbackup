"""Jointly adaptable Rain/Blur MoE; VLM predictions enter only its router.

One shared trainable LoRA is NOT a LoRA expert bank. FFN experts, image
routers and the shared attention/stream LoRA learn together in both arms.
"""
from __future__ import annotations

import csv
import json
from contextlib import contextmanager
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from lowlight_restoration.lora import lora_modules
from lowlight_restoration.rain_blur_image_moe import ImageExpertFFN, image_moe_modules
from lowlight_restoration.rain_blur_moe_data import RainBlurDataset

ATOMS = ('blur', 'noise', 'rain', 'haze', 'low_light')
ARMS = ('image_only', 'vlm_assisted')


def predicted_levels(row):
    """Never read oracle/type/category labels to construct the VLM input."""
    levels = json.loads(row['qwen_levels_json'])
    if set(levels) != set(ATOMS) or any(type(levels[k]) is not int or not 1 <= levels[k] <= 5 for k in ATOMS):
        raise ValueError('invalid five-slot VLM prediction')
    return [(levels[k] - 1) / 4 for k in ATOMS]


def join_conditions(path, allowed_rows):
    expected = {r['record_id']: r for r in allowed_rows}
    if len(expected) != len(allowed_rows):
        raise ValueError('duplicate allowed record')
    result = {}
    with Path(path).open(newline='') as handle:
        for row in csv.DictReader(handle):
            key = row['record_id']
            if key not in expected:
                continue  # Metadata scan only; never open excluded/protected pixels.
            ref = expected[key]
            if row['split'] not in ('train', 'validation') or key in result:
                raise ValueError('protected or duplicated condition record')
            for name in ('split', 'source_group', 'lq_path', 'lq_sha256', 'gt_path', 'gt_sha256'):
                if row[name] != ref[name]:
                    raise ValueError(f'condition/paired record mismatch: {key} {name}')
            result[key] = predicted_levels(row)
    if result.keys() != expected.keys():
        raise ValueError('missing VLM predictions; never substitute oracle labels')
    return result


class JointVLMExpertFFN(ImageExpertFFN):
    def __init__(self, original, dim, *, hidden=256, arm='image_only'):
        if arm not in ARMS:
            raise ValueError('unknown arm')
        super().__init__(original, dim, hidden=hidden, experts=2)
        self.arm = arm
        # No LayerNorm: absolute ordinal strength and all-absent zero retain meaning.
        self.vlm_projector = nn.Sequential(nn.Linear(5, hidden), nn.SiLU(), nn.Linear(hidden, 2))
        self.vlm_projector.to(device=next(original.parameters()).device, dtype=torch.float32)
        nn.init.zeros_(self.vlm_projector[-1].weight)
        nn.init.zeros_(self.vlm_projector[-1].bias)
        self.vlm_projector.requires_grad_(arm == 'vlm_assisted')
        self.condition = None
        self.use_vlm = arm == 'vlm_assisted'
        self.last_adjustment = None
        self.activation_checkpointing = False  # Whole transformer blocks are checkpointed.

    def forward(self, tokens):
        first = self.experts[0](tokens)
        with torch.autocast(tokens.device.type, enabled=False):
            logits = self.router(tokens.float().mean(dim=1))
            if self.use_vlm:
                if self.condition is None or self.condition.shape != (tokens.shape[0], 5):
                    raise ValueError('VLM-assisted routing requires per-image predictions')
                adjustment = self.vlm_projector(self.condition.float())
            else:
                adjustment = torch.zeros_like(logits)
            logits = logits + adjustment
            weights = logits.softmax(-1)
        self.last_logits, self.last_adjustment = logits, adjustment
        if self.force_expert is not None:
            if self.training or self.force_expert not in (0, 1):
                raise RuntimeError('fixed expert is inference-only')
            weights = F.one_hot(torch.full((tokens.shape[0],), self.force_expert, device=tokens.device), 2).float()
        elif self.teacher_fraction:
            if not self.training or self.labels is None or not 0 <= self.teacher_fraction <= 1:
                raise RuntimeError('task labels may only enter training warmup')
            weights = ((1 - self.teacher_fraction) * weights
                       + self.teacher_fraction * F.one_hot(self.labels, 2))
        self.last_weights = weights
        second = self.experts[1](tokens)
        return (first.float() + weights[:, 1, None, None] * (second.float() - first.float())).to(first.dtype)


def install_joint_moe(model, *, arm, indices=(0, 1, 2, 3, 4), hidden=256):
    if not indices or len(set(indices)) != len(indices):
        raise ValueError('invalid block indices')
    model.requires_grad_(False)
    # Reopen the already loaded, shared Rain LoRA, including spatial attention.
    for module in lora_modules(model).values():
        module.lora_A.float().requires_grad_(True)
        module.lora_B.float().requires_grad_(True)
    if not lora_modules(model):
        raise ValueError('shared adaptation is required; FFN-only repeats the failed restriction')
    for index in indices:
        block = model.double_blocks[index]
        if isinstance(block.img_mlp, ImageExpertFFN):
            raise ValueError('already injected')
        block.img_mlp = JointVLMExpertFFN(block.img_mlp, block.hidden_size, hidden=hidden, arm=arm)
        block.img_mlp.experts.requires_grad_(True)
        block.img_mlp.router.requires_grad_(True)
    return image_moe_modules(model)


def enable_block_checkpointing(model):
    """Keep state-dict names unchanged. Conditioning must span backward too."""
    def wrap(block):
        if getattr(block, '_joint_checkpointed', False):
            raise ValueError('block already checkpointed')
        original = block.forward
        def forward(*args, **kwargs):
            if block.training and torch.is_grad_enabled():
                return checkpoint(original, *args, use_reentrant=False, **kwargs)
            return original(*args, **kwargs)
        block.forward = forward
        block._joint_checkpointed = True
    for block in list(model.double_blocks) + list(model.single_blocks):
        wrap(block)


@contextmanager
def joint_conditioning(model, condition, *, labels=None, fraction=0.0, disable_vlm=False):
    modules = image_moe_modules(model)
    if (labels is not None or fraction) and not model.training:
        raise ValueError('training labels cannot enter evaluation')
    if fraction and labels is None:
        raise ValueError('warmup requires training labels')
    if condition is not None and (not bool(torch.isfinite(condition).all())
                                  or bool(((condition < 0) | (condition > 1)).any())):
        raise ValueError('invalid normalized VLM predictions')
    try:
        for module in modules.values():
            module.condition = condition
            module.labels, module.teacher_fraction = labels, fraction
            module.use_vlm = module.arm == 'vlm_assisted' and not disable_vlm
        yield
    finally:
        for module in modules.values():
            module.condition = module.labels = None
            module.teacher_fraction = 0.0
            module.use_vlm = module.arm == 'vlm_assisted'
            module.last_logits = module.last_weights = module.last_adjustment = None


class JointDataset(RainBlurDataset):
    def __init__(self, *args, conditions, start_step=0, **kwargs):
        super().__init__(*args, **kwargs)
        if not 0 <= start_step < self.steps:
            raise ValueError('invalid resume position')
        self.conditions, self.start_step = conditions, start_step

    def __len__(self):
        return (self.steps - self.start_step) * 4

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        item = super().__getitem__(index + self.start_step * 4)
        item['condition'] = torch.tensor(self.conditions[item['record_id']], dtype=torch.float32)
        return item


def restoration_loss(result, clean, gt, metric, pack_image, weights):
    """Task-neutral endpoint supervision. No Rain mask, identity or teacher."""
    from smoke.train_foundir_rain_flux2_precision_fit_v1_ddp import spatial_highpass
    output = result['decoded']
    target, _ = pack_image(result['degraded'] - clean)
    terms = {
        'flow_mse': F.mse_loss(result['prediction'].float(), target.float()),
        'latent_l1': F.l1_loss(result['predicted_clean'], clean),
        'rgb_mse': F.mse_loss(output, gt),
        'charbonnier': ((output - gt).square() + 1e-6).sqrt().mean(),
        'highpass': F.l1_loss(spatial_highpass(output), spatial_highpass(gt)),
        'lpips': metric((output + 1) * .5, (gt + 1) * .5),
    }
    if terms.keys() != weights.keys():
        raise ValueError('loss contract mismatch')
    return sum(terms[k] * weights[k] for k in terms), terms
