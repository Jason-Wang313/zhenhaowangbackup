"""Learn two conditional quality costs, never a restoration image.

Training supervises scores AND their image derivatives. Deployment composes
these derivatives with the actual frozen FLUX tail, not a pixel refiner.
"""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F

from .proposal_relative_energy_v2 import SpatialConditionalEnergy


class TailQualityEnergy(nn.Module):
    def __init__(self, width=32, condition_dim=96):
        super().__init__()
        self.condition = nn.Sequential(nn.Linear(6, condition_dim), nn.SiLU(),
                                       nn.Linear(condition_dim, condition_dim), nn.SiLU())
        self.field = SpatialConditionalEnergy(width=width, condition_dim=condition_dim)
        self.field.intro = nn.Conv2d(15, width, 3, padding=1, padding_mode='reflect')
        self.field.head = nn.Conv2d(width, 2, 3, padding=1, padding_mode='reflect')
        nn.init.normal_(self.field.head.weight, std=.01)
        with torch.no_grad():
            self.field.head.bias.copy_(torch.tensor([-9., -3.]))

    def score(self, observation, proposal, candidate, descriptor, timestep):
        if observation.shape != proposal.shape or observation.shape != candidate.shape:
            raise ValueError('paired RGB shapes must match')
        if candidate.ndim != 4 or candidate.shape[1] != 3 or any(n % 4 for n in candidate.shape[-2:]):
            raise ValueError('RGB dimensions must divide by four')
        batch = candidate.shape[0]
        if descriptor.shape != (batch, 5) or timestep.shape != (batch, 1):
            raise ValueError('per-image five VLM levels plus actual shifted time required')
        condition = self.condition(torch.cat((descriptor, timestep), 1))
        field = self.field
        value = field.intro(torch.cat((observation, proposal, candidate,
                                      candidate-proposal, candidate-observation), 1))
        full = field.full(value, condition)
        half = field.half_stack(field.down1(full), condition)
        quarter = field.quarter(field.down2(half), condition)
        value = field.decode_half(field.up1(F.interpolate(quarter, size=half.shape[-2:],
                                  mode='bilinear', align_corners=False))+half, condition)
        value = field.decode_full(field.up2(F.interpolate(value, size=full.shape[-2:],
                                  mode='bilinear', align_corners=False))+full, condition)
        # log(mean(local positive cost)); heads estimate log MSE and log LPIPS.
        maps = field.head(value).flatten(2)
        return torch.logsumexp(maps, dim=2)-math.log(maps.shape[2])

    def forward(self, observation, proposal, candidate, descriptor, timestep):
        candidate = candidate.detach().requires_grad_(True)
        scores = self.score(observation, proposal, candidate, descriptor, timestep)
        derivatives = tuple(torch.autograd.grad(scores[:, h].sum(), candidate,
            create_graph=True, retain_graph=True)[0] for h in range(2))
        return scores, derivatives


def quality_targets(candidate, target, perceptual):
    """Real metric derivatives at this image, NOT derivatives of uint8 rounding.

    Cached samples contain both continuous and deployed-quantized images.
    Each is treated as an RGB leaf: the same smooth metric function supplies
    labels and derivatives. Quantization is never called differentiable.
    """
    value = candidate.detach().requires_grad_(True)
    mse = ((value-target.detach())/2).square().flatten(1).mean(1).clamp_min(1e-12)
    lpips = perceptual(value, target.detach()).reshape(value.shape[0], -1).mean(1).clamp_min(1e-8)
    costs = torch.stack((mse.log(), lpips.log()), 1)
    derivatives = tuple(torch.autograd.grad(costs[:, h].sum(), value,
        retain_graph=h == 0)[0].detach() for h in range(2))
    return costs.detach(), derivatives


def learning_loss(prediction, target, *, gradient_weight=1., pair_weight=.25):
    scores, gradients = prediction
    wanted, derivatives = target
    if scores.shape != (2, 2) or wanted.shape != scores.shape:
        raise ValueError('one same-source candidate pair, two quality heads required')
    score_loss = F.smooth_l1_loss(scores, wanted)
    gradient_losses, cosines, ratios = [], [], []
    for gradient, teacher in zip(gradients, derivatives, strict=True):
        raw_scale = teacher.square().flatten(1).mean(1).sqrt()
        # A clamped zero-cost target has no defined direction; do not amplify
        # arbitrary numerical gradients by 1e10. Retain an absolute zero fit.
        scale = torch.where(raw_scale > 1e-10, raw_scale, torch.ones_like(raw_scale))
        gradient_losses.append(((gradient-teacher)/scale[:, None, None, None]).square().mean())
        cosines.append(F.cosine_similarity(gradient.flatten(1), teacher.flatten(1), dim=1, eps=1e-10).mean())
        ratios.append((gradient.square().flatten(1).mean(1).sqrt()/scale).mean())
    gradient_loss = torch.stack(gradient_losses).mean()
    gap, predicted_gap = wanted[1]-wanted[0], scores[1]-scores[0]
    valid = gap.abs() > .01
    hinge = F.relu(.5*gap.abs()-gap.sign()*predicted_gap)
    pair_loss = hinge[valid].mean() if bool(valid.any()) else scores.sum()*0
    loss = score_loss+gradient_weight*gradient_loss+pair_weight*pair_loss
    diagnostics = dict(score=score_loss, gradient=gradient_loss, pair=pair_loss,
        mse_gradient_cosine=cosines[0], lpips_gradient_cosine=cosines[1],
        mse_gradient_rms_ratio=ratios[0], lpips_gradient_rms_ratio=ratios[1])
    return loss, diagnostics


def predicted_metrics(log_costs):
    """Convert predictions to the existing policy's units; not measured metrics."""
    if log_costs.shape != (1, 2) or not bool(log_costs.isfinite().all()):
        raise RuntimeError('invalid predicted costs')
    return dict(psnr=float(-10/math.log(10)*log_costs[0, 0]),
                lpips=float(log_costs[0, 1].clamp(-30, 10).exp()))
