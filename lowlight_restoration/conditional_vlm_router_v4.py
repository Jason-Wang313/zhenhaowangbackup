"""Frozen image logits plus a VLM correction conditioned on those same logits."""
import torch
from torch import nn
from .frozen_expert_router import FrozenExpertRouter


class ScoreConditionedCorrection(nn.Module):
    def __init__(self, vlm_dim=3584, hidden=256):
        super().__init__()
        self.norm = nn.LayerNorm(vlm_dim)
        self.net = nn.Sequential(nn.Linear(vlm_dim + 5, hidden), nn.SiLU(), nn.Linear(hidden, 5))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, vlm, image_logits):
        # Centering removes irrelevant common offsets; fixed /5 retains margins/confidence.
        scores = image_logits.detach().float()
        scores = (scores - scores.mean(-1, keepdim=True)) / 5.
        return self.net(torch.cat((self.norm(vlm.float()), scores), dim=-1))


class ConditionalVLMRouter(FrozenExpertRouter):
    def __init__(self, image_dim=6144, vlm_dim=3584, hidden=256):
        super().__init__(image_dim, vlm_dim, hidden)
        self.vlm = ScoreConditionedCorrection(vlm_dim, hidden)
        self.image.requires_grad_(False)

    def forward(self, image, vlm, *, vlm_enabled=True):
        if image.requires_grad or vlm.requires_grad or not torch.isfinite(image).all() or not torch.isfinite(vlm).all():
            raise ValueError('finite frozen features required; no labels as router inputs')
        im = self.image(image.float())
        delta = self.vlm(vlm, im)
        logits = im + (delta if vlm_enabled else delta * 0.)
        return logits, logits.softmax(-1), im, delta
