"""Ordinary image-only FFN experts initialized from a working Rain DiT.

One shared frozen Rain model, two full FFN branches per selected double block.
No LoRA expert bank, VLM, pixel refiner or task labels at deployment.
"""
from __future__ import annotations

import copy
from contextlib import contextmanager

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


class ImageExpertFFN(nn.Module):
    def __init__(self, original, dim, hidden=256, experts=2):
        super().__init__()
        if experts not in (1, 2):
            raise ValueError('only dense control or two experts are supported')
        self.experts = nn.ModuleList([copy.deepcopy(original).float() for _ in range(experts)])
        self.router = (nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.SiLU(),
                                     nn.Linear(hidden, 2)).to(next(original.parameters()).device)
                       if experts == 2 else None)
        if self.router is not None:
            nn.init.zeros_(self.router[-1].weight)
            nn.init.zeros_(self.router[-1].bias)
        self.labels = None
        self.teacher_fraction = 0.0
        self.force_expert = None
        self.last_logits = None
        self.last_weights = None
        self.activation_checkpointing = True

    def forward(self, tokens):
        def execute(expert):
            if self.training and torch.is_grad_enabled() and self.activation_checkpointing:
                return checkpoint(expert, tokens, use_reentrant=False)
            return expert(tokens)
        first = execute(self.experts[0])
        if self.router is None:
            return first
        with torch.autocast(tokens.device.type, enabled=False):
            logits = self.router(tokens.float().mean(dim=1))
            weights = logits.softmax(dim=-1)
        self.last_logits = logits
        if self.force_expert is not None:
            if self.training or self.force_expert not in (0, 1):
                raise RuntimeError('constant-expert diagnostic is evaluation only')
            weights = F.one_hot(torch.full((tokens.shape[0],), self.force_expert,
                               device=tokens.device), 2).float()
        elif self.teacher_fraction:
            if not self.training or self.labels is None or not 0 <= self.teacher_fraction <= 1:
                raise RuntimeError('task labels are permitted only during training warmup')
            labels = self.labels.to(tokens.device)
            if labels.shape != (tokens.shape[0],) or bool(((labels < 0) | (labels > 1)).any()):
                raise RuntimeError('invalid training task labels')
            weights = (1 - self.teacher_fraction) * weights + self.teacher_fraction * F.one_hot(labels, 2)
        self.last_weights = weights
        second = execute(self.experts[1])
        # Equivalent to w0*E0+w1*E1; exactly preserves identical initial experts.
        mixed = first.float() + weights[:, 1, None, None] * (second.float() - first.float())
        return mixed.to(first.dtype)


def inject_image_moe(model, *, indices=(0, 1, 2, 3, 4), experts=2, hidden=256):
    if not indices or len(set(indices)) != len(indices):
        raise ValueError('invalid block indices')
    model.requires_grad_(False)
    modules = {}
    for index in indices:
        block = model.double_blocks[index]
        if isinstance(block.img_mlp, ImageExpertFFN):
            raise ValueError('already injected')
        module = ImageExpertFFN(block.img_mlp, block.hidden_size, hidden, experts)
        module.requires_grad_(True)
        block.img_mlp = module
        modules[f'double_blocks.{index}.img_mlp'] = module
    return modules


def image_moe_modules(model):
    return {name: module for name, module in model.named_modules() if isinstance(module, ImageExpertFFN)}


def warmup_fraction(step, hard_steps, transition_end):
    if not 0 <= hard_steps < transition_end:
        raise ValueError('invalid warmup')
    if step < hard_steps:
        return 1.0
    return max(0.0, 1 - (step - hard_steps) / (transition_end - hard_steps))


@contextmanager
def training_routes(model, labels, fraction):
    modules = image_moe_modules(model)
    if not model.training:
        raise RuntimeError('training routes cannot enter inference')
    try:
        for module in modules.values():
            module.labels, module.teacher_fraction = labels, fraction
        yield
    finally:
        for module in modules.values():
            module.labels, module.teacher_fraction = None, 0.0


def router_classification_loss(model, labels):
    terms = [F.cross_entropy(m.last_logits, labels) for m in image_moe_modules(model).values()
             if m.router is not None]
    return torch.stack(terms).mean() if terms else torch.zeros((), device=labels.device)


def trainable_state(model):
    return {name: p.detach().cpu().contiguous() for name, p in model.named_parameters() if p.requires_grad}


def load_expert_state(model, path):
    """Load a saved arm into the same Rain-initialized/injected architecture."""
    from safetensors.torch import load_file
    loaded = load_file(str(path), device='cpu')
    expected = {name: p for name, p in model.named_parameters() if p.requires_grad}
    if set(loaded) != set(expected):
        raise RuntimeError('expert checkpoint keys mismatch')
    for name, parameter in expected.items():
        value = loaded[name]
        if value.shape != parameter.shape or value.dtype != torch.float32 or not bool(torch.isfinite(value).all()):
            raise RuntimeError('invalid expert checkpoint tensor')
    with torch.no_grad():
        for name, parameter in expected.items():
            parameter.copy_(loaded[name].to(parameter.device))


def clear_routing_graphs(model):
    for module in image_moe_modules(model).values():
        module.last_logits = module.last_weights = None
