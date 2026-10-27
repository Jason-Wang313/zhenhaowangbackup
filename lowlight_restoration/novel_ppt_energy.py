"""Candidate-dependent energy in normalized residual coordinates.

The frozen strong fidelity image is an anchor, NOT a trainable prediction.
Every deployment update is the derivative of the SAME energy used for ranking.
Initial zero potential yields exact identity; this alone is not a quality claim.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .proposal_relative_energy_v2 import SpatialConditionalEnergy
from .vlm_energy_restoration import ConditionEncoder


class NovelPPTEnergy(nn.Module):
    def __init__(self, width=24, condition_dim=96, residual_scale=.02, step_size=.5, bound=4.):
        super().__init__()
        if min(residual_scale, step_size, bound) <= 0:
            raise ValueError('positive residual-coordinate constants required')
        self.residual_scale, self.step_size, self.bound = residual_scale, step_size, bound
        self.encoder = ConditionEncoder(condition_dim)
        self.field = SpatialConditionalEnergy(width=width, condition_dim=condition_dim)
        self.field.intro = nn.Conv2d(15, width, 3, padding=1, padding_mode='reflect')
        nn.init.zeros_(self.field.head.weight)

    def density(self, observation, proposal, anchor, u, condition):
        if not (observation.shape == proposal.shape == anchor.shape == u.shape):
            raise ValueError('all images/residuals must have the same shape')
        if u.ndim != 4 or u.shape[1] != 3 or any(n % 4 for n in u.shape[-2:]):
            raise ValueError('RGB spatial dimensions must divide by four')
        candidate = anchor + self.residual_scale * u
        field = self.field
        value = field.intro(torch.cat((observation, proposal, anchor, candidate, u), dim=1))
        full = field.full(value, condition)
        half = field.half_stack(field.down1(full), condition)
        quarter = field.quarter(field.down2(half), condition)
        value = field.decode_half(field.up1(F.interpolate(quarter, size=half.shape[-2:], mode='bilinear', align_corners=False)) + half, condition)
        value = field.decode_full(field.up2(F.interpolate(value, size=full.shape[-2:], mode='bilinear', align_corners=False)) + full, condition)
        return .5 * u.square().sum(1, keepdim=True) + field.head(value)

    def score_map(self, observation, proposal, anchor, candidate, descriptor):
        return self.density(observation, proposal, anchor,
                            (candidate - anchor) / self.residual_scale, self.encoder(descriptor))

    def trajectory(self, observation, proposal, anchor, descriptor, *, steps=3, create_graph=False):
        if steps not in (1, 2, 3):
            raise ValueError('only the predeclared 1/2/3-step trajectory is supported')
        condition = self.encoder(descriptor)
        u = torch.zeros_like(anchor).requires_grad_(True)
        candidates, gradients, coordinates = [], [], []
        for index in range(steps):
            if index and not create_graph:
                u = u.detach().requires_grad_(True)
            density = self.density(observation, proposal, anchor, u, condition)
            # Sum over pixels AND batch: no hidden 1/(B*H*W) in the update.
            gradient = torch.autograd.grad(density.sum(), u, create_graph=create_graph)[0]
            coordinates.append(u)
            gradients.append(gradient)
            u = (u - self.step_size * gradient).clamp(-self.bound, self.bound)
            candidate = (anchor + self.residual_scale * u).clamp(-1, 1)
            u = (candidate - anchor) / self.residual_scale
            candidates.append(candidate)
        return tuple(candidates), tuple(gradients), tuple(coordinates)

    def forward(self, observation, proposal, anchor, descriptor, candidates):
        trajectory, gradients, coordinates = self.trajectory(
            observation, proposal, anchor, descriptor, create_graph=True)
        values = tuple(candidates) + trajectory
        maps = torch.stack([self.score_map(observation, proposal, anchor, item, descriptor)
                            for item in values], dim=1)
        return {'trajectory': trajectory, 'gradients': gradients, 'coordinates': coordinates,
                'rank_candidates': values, 'rank_maps': maps}

    @torch.no_grad()
    def select(self, observation, proposal, anchor, descriptor, candidates):
        values = (anchor,) + tuple(candidates)
        scores = torch.stack([self.score_map(observation, proposal, anchor, item, descriptor).flatten(1).mean(1)
                              for item in values], dim=1)
        if not bool(torch.isfinite(scores).all()):
            raise RuntimeError('nonfinite candidate energy')
        # argmin tie resolves to the anchor, without access to GT.
        indices = scores.argmin(1)
        outputs = torch.stack(values, dim=1)
        gather = indices[:, None, None, None, None].expand(-1, 1, *anchor.shape[1:])
        return outputs.gather(1, gather).squeeze(1), indices, scores


def ranking_loss(energy, quality, margin=.05, tie=1e-6):
    """Frozen quality measurements; no gradient through target/margin values."""
    quality = quality.detach()
    difference = quality[:, None, :] - quality[:, :, None]
    mask = difference > tie
    scale = quality.mean(1, keepdim=True)[:, :, None].clamp_min(1e-6)
    violations = F.relu(margin * difference.clamp_min(0) / scale
                        + energy[:, :, None] - energy[:, None, :])
    return violations[mask].mean() if bool(mask.any()) else energy.sum() * 0


def objective(output, anchor, target, perceptual, *, residual_scale, weights, ranking_lpips_weight=2.):
    trajectory = output['trajectory']
    states = (anchor,) + trajectory
    mse = torch.stack([((value - target) / residual_scale).square().flatten(1).mean(1) for value in states], dim=1)
    lp = torch.stack([perceptual(value, target).reshape(value.shape[0], -1).mean(1) for value in states], dim=1)
    wanted = ((target - anchor) / residual_scale).detach()
    gradient_fit = torch.stack([(g - (u.detach() - wanted)).square().mean()
                               for g, u in zip(output['gradients'], output['coordinates'], strict=True)]).mean()
    values = output['rank_candidates']
    # Both global and local supervision are detached from candidate generation.
    with torch.no_grad():
        quality = torch.stack([((value - target) / residual_scale).square().mean(1)
                               for value in values], dim=1)
        b, n, h, w = quality.shape
        quality = F.avg_pool2d(quality.reshape(b*n, 1, h, w), 8, 8).reshape(b, n, -1)
        perceptual_quality = torch.stack([perceptual(value.detach(), target).reshape(b, -1).mean(1)
                                          for value in values], dim=1)
        global_quality = quality.mean(2) + ranking_lpips_weight * perceptual_quality
    maps = output['rank_maps']
    local = F.avg_pool2d(maps.flatten(0, 1), 8, 8).reshape(b, n, -1)
    terms = {'mse': mse[:, 1:].mean(), 'lpips': lp[:, 1:].mean(),
             'gradient_fit': gradient_fit,
             'global_rank': ranking_loss(maps.flatten(2).mean(2), global_quality),
             'local_rank': ranking_loss(local, quality),
             'no_harm_mse': F.relu(mse[:, 1:] - mse[:, :-1].detach()).mean(),
             'no_harm_lpips': F.relu(lp[:, 1:] - lp[:, :-1].detach()).mean()}
    if set(terms) != set(weights):
        raise ValueError('loss weights must exactly match declared terms')
    loss = sum(weights[name] * value for name, value in terms.items())
    with torch.no_grad():
        directions = [F.cosine_similarity(-g.flatten(1), (wanted-u).flatten(1), dim=1, eps=1e-8).mean()
                      for g, u in zip(output['gradients'], output['coordinates'], strict=True)]
        diagnostics = {'gradient_cosine': torch.stack(directions).mean(),
                       'gradient_rms': torch.stack([g.square().mean().sqrt() for g in output['gradients']]).mean(),
                       'update_abs_mean': (trajectory[-1] - anchor).abs().mean()}
    return loss, terms, diagnostics
