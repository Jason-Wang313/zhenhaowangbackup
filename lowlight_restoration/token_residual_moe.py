"""Right-diagram router: existing image logits + pooled frozen VLM logits.

No restoration branch is added. The five existing FLUX FFN expert banks and
shared LoRA retain their weights. No task identity or GT enters routing.
"""
from contextlib import contextmanager

import torch
from torch import nn

from lowlight_restoration.rain_blur_image_moe import ImageExpertFFN, image_moe_modules


class TokenResidualFFN(ImageExpertFFN):
    def __init__(self, parent, feature_dim=3584, hidden=256):
        # Reuse trained tensors, NOT clone/reset trained experts or router.
        nn.Module.__init__(self)
        self.experts, self.router = parent.experts, parent.router
        self.feature_dim = feature_dim
        self.vlm_projector = nn.Sequential(nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden), nn.SiLU(), nn.Linear(hidden, 2))
        self.vlm_projector.to(device=next(self.router.parameters()).device, dtype=torch.float32)
        nn.init.zeros_(self.vlm_projector[-1].weight)
        nn.init.zeros_(self.vlm_projector[-1].bias)
        self.condition = None
        self.use_vlm = True
        self.labels, self.teacher_fraction, self.force_expert = None, 0., None
        self.last_logits = self.last_weights = self.last_adjustment = None

    def forward(self, tokens):
        if self.labels is not None or self.teacher_fraction or self.force_expert is not None:
            raise RuntimeError('right-diagram routing accepts neither task labels nor forced experts')
        first = self.experts[0](tokens)
        with torch.autocast(tokens.device.type, enabled=False):
            logits = self.router(tokens.float().mean(1))
            if self.use_vlm:
                if self.condition is None or self.condition.shape != (tokens.shape[0], self.feature_dim):
                    raise ValueError('pooled VLM tokens required; five severity labels are not tokens')
                if self.condition.requires_grad or not torch.isfinite(self.condition).all():
                    raise ValueError('VLM features must be finite and frozen')
                adjustment = self.vlm_projector(self.condition.float())
            else:
                adjustment = torch.zeros_like(logits)
            self.last_adjustment = adjustment
            self.last_logits = logits + adjustment
            self.last_weights = self.last_logits.softmax(-1)
        second = self.experts[1](tokens)
        return (first.float() + self.last_weights[:, 1, None, None]
                * (second.float() - first.float())).to(first.dtype)


def attach_token_router(model, feature_dim=3584, hidden=256):
    replacements = []
    for name, module in image_moe_modules(model).items():
        if isinstance(module, TokenResidualFFN):
            raise ValueError('already attached')
        container_name, _, leaf = name.rpartition('.')
        if not leaf:
            raise ValueError('MoE must be a named child of the restoration model')
        replacements.append((model.get_submodule(container_name), leaf,
                             TokenResidualFFN(module, feature_dim, hidden)))
    if not replacements:
        raise ValueError('requires a loaded image-only MoE')
    for container, leaf, replacement in replacements:
        setattr(container, leaf, replacement)
    return image_moe_modules(model)


@contextmanager
def token_conditioning(model, features, *, enabled=True):
    modules = image_moe_modules(model)
    if not modules or any(not isinstance(m, TokenResidualFFN) for m in modules.values()):
        raise ValueError('wrong router architecture')
    try:
        for module in modules.values():
            module.condition, module.use_vlm = features, enabled
        yield
    finally:
        for module in modules.values():
            module.condition = None
            module.use_vlm = True
            module.last_logits = module.last_weights = module.last_adjustment = None


def batch_slot(step, micro, rank):
    """Four microbatches on eight ranks: exactly 32 real samples, no padding.

    DDP averages eight ranks, so each local sample contributes 1/4. Preserve
    the old global Rain/Blur sequence while using four two-GPU nodes.
    """
    if step < 0 or not 0 <= micro < 4 or not 0 <= rank < 8:
        raise ValueError('invalid eight-rank batch slot')
    return step * 32 + micro * 8 + rank, .25
