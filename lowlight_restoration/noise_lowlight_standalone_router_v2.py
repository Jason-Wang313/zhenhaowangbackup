"""LQ-only binary discriminator; no inherited logit in the learned margin."""
import torch
from torch import nn
from lowlight_restoration.noise_lowlight_physical_router_v1 import FEATURE_NAMES


class StandaloneRouter(nn.Module):
    def __init__(self, mean, scale):
        super().__init__()
        self.register_buffer('mean', mean.float())
        self.register_buffer('scale', scale.float().clamp_min(1e-6))
        self.head = nn.Sequential(nn.Linear(len(FEATURE_NAMES), 32), nn.SiLU(), nn.Linear(32, 1))

    def forward(self, base_logits, physical):
        if base_logits.ndim != 2 or base_logits.shape[1] != 5 or physical.shape != (len(base_logits), len(FEATURE_NAMES)):
            raise ValueError('invalid binary input shapes')
        if not torch.isfinite(base_logits).all() or not torch.isfinite(physical).all():
            raise ValueError('nonfinite binary input')
        z = ((physical-self.mean)/self.scale).clamp(-8, 8)
        margin = self.head(z).squeeze(-1)
        old = base_logits.argmax(-1)
        route = torch.where((old == 1)|(old == 3), torch.where(margin >= 0, 1, 3), old)
        return margin, route, margin
