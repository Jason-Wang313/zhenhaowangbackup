"""Relative quality learning and bounded one-step frozen-FLUX guidance.

No restoration network is added. Training labels and deployment interfaces are
separate; bounds limit edits, not a claim of true no-harm without ground truth.
"""
from __future__ import annotations

import hashlib
import math
import random
import torch
from torch.nn import functional as F

from .flux_tail_budget import mixed_directions, choose_trial
from .flux_tail_learned_energy import predicted_metrics


def relative_loss(prediction, target, weights):
    scores, gradients = prediction
    labels, teachers = target
    if scores.shape != (2, 2) or labels.shape != scores.shape:
        raise ValueError('requires two same-source candidates and two quality heads')
    relative = F.smooth_l1_loss(scores[1]-scores[0], labels[1]-labels[0],
                              beta=weights['relative_beta'])
    absolute = F.smooth_l1_loss(scores, labels)
    fits, directions, cosines, ratios = [], [], [], []
    for g, teacher in zip(gradients, teachers, strict=True):
        scale = teacher.square().flatten(1).mean(1).sqrt()
        valid = scale > 1e-10
        divisor = torch.where(valid, scale, torch.ones_like(scale))[:, None, None, None]
        a, b = (g/divisor).flatten(1), (teacher/divisor).flatten(1)
        fits.append((a-b).square().mean())
        # Compare RMS-normalized vectors; epsilon caps the inverse norm near
        # zero. Zero target derivatives have no valid orientation supervision.
        ar = torch.linalg.vector_norm(a,dim=1)/math.sqrt(a.shape[1])
        br = torch.linalg.vector_norm(b,dim=1)/math.sqrt(b.shape[1])
        cosine = (a*b).mean(1)/(ar.clamp_min(1e-3)*br.clamp_min(1e-3))
        directions.append(((1-cosine)*valid).sum()/valid.sum().clamp_min(1))
        cosines.append((cosine.detach()*valid).sum()/valid.sum().clamp_min(1))
        ratios.append((ar.detach()*valid).sum()/valid.sum().clamp_min(1))
    components = dict(relative=relative, absolute=absolute,
                      gradient=torch.stack(fits).mean(), direction=torch.stack(directions).mean())
    loss = sum(weights[k]*v for k, v in components.items())
    gap = labels[1]-labels[0]
    mask = gap.abs() > .01
    correct = ((scores[1]-scores[0])*gap > 0) & mask
    diagnostics = {k: v.detach() for k,v in components.items()}
    diagnostics.update(mse_cosine=cosines[0], lpips_cosine=cosines[1],
                       mse_rms_ratio=ratios[0], lpips_rms_ratio=ratios[1],
                       rank_correct=correct.sum().float(), rank_valid=mask.sum().float())
    return loss, diagnostics, components


def parameter_gradient_audit(model, components, weights):
    """Read-only per-objective parameter-gradient norms, not loss-value proxies."""
    parameters = tuple(model.parameters())
    output = {}
    for name, value in components.items():
        gradients = torch.autograd.grad(weights[name]*value, parameters,
                                        retain_graph=True, allow_unused=True)
        if any(g is not None and not bool(g.isfinite().all()) for g in gradients):
            raise RuntimeError('nonfinite component parameter derivative')
        output[name] = math.sqrt(sum(float(g.detach().double().square().sum())
                                    for g in gradients if g is not None))
    return output


def generation_for_step(step, refresh_steps):
    if step < 1:
        raise ValueError('optimizer steps are one-based')
    return max([0]+[s for s in refresh_steps if s < step])


def pair_plan(slot, seed, base_sizes, mining_sizes, teacher_indices, proposal_indices):
    """Same global stream on 4/8 GPUs. Half of pairs include self-mined frames."""
    digest = hashlib.sha256(f'{seed}:relative_pair:{slot}'.encode()).hexdigest()
    rng = random.Random(int(digest[:16],16))
    index = rng.randrange(len(base_sizes))
    if min(base_sizes[index], mining_sizes[index]) < 2:
        raise ValueError('missing base or self-mined candidates')
    kind = slot % 4
    proposal = ('base', proposal_indices[index])
    if kind == 0:
        options = teacher_indices[index]
        alternative = rng.choice(options) if options else rng.randrange(base_sizes[index])
        pair = [proposal, ('base', alternative)]
    elif kind == 1:
        pair = [proposal, ('mining', rng.randrange(mining_sizes[index]))]
    elif kind == 2:
        pair = [('base', rng.randrange(base_sizes[index])),
                ('mining', rng.randrange(mining_sizes[index]))]
    else:
        pair = [('base', n) for n in rng.sample(range(base_sizes[index]), 2)]
    if rng.randrange(2): pair.reverse()
    return index, pair


def edit_stats(image, proposal):
    # Inputs are [-1,1]; statistics are in [0,1] RGB units.
    delta = (image.float()-proposal.float())/2
    return dict(pixel_rms=float(delta.square().mean().sqrt()),
                channel_mean_max=float(delta.mean(dim=(-2,-1)).abs().max()))


def trust_limits(training_edits, spec):
    if not training_edits:
        raise ValueError('no training-source teacher first-step edits')
    limits = {}
    for key in ('pixel_rms','channel_mean_max'):
        values = torch.tensor([r[key] for r in training_edits],dtype=torch.float64)
        if not bool(values.isfinite().all()) or bool((values < 0).any()):
            raise ValueError('invalid training-derived radius')
        limits[key] = max(spec['trust_quantization_floor_0_1'],
                          float(torch.quantile(values,spec['trust_quantile']))*spec['trust_multiplier'])
    return limits


def within_trust(image, proposal, limits):
    stats = edit_stats(image,proposal)
    return all(stats[k] <= limits[k]+1e-12 for k in stats), stats


def bounded_one_step(session, energy, descriptor, parent_spec, limits):
    """No GT. Preserve exact original proposal if none passes predicted rules.

    Both the raw self-scored choice and bounded choice are returned. A bound
    rejecting everything is reported as no-change, never a quality gain.
    """
    timestep = torch.tensor([[session.timestep]],device=descriptor.device)
    def objective(image):
        session.tail.calls['energy_forward'] += 1
        return energy.score(session.observation,session.proposal,image,descriptor,timestep)
    def score(image):
        with torch.no_grad(): return predicted_metrics(objective(image))
    baseline = score(session.proposal)
    original = ((session.proposal+1)*127.5).round().clamp(0,255).to(torch.uint8).cpu()
    with torch.no_grad(): control = session.render(session.initial)
    gm,gp = session.gradients(session.initial,objective)
    if not bool(gm.isfinite().all() and gp.isfinite().all()):
        raise RuntimeError('nonfinite guidance derivative')
    trials, images, admissible = [], [], []
    if min(float(g.square().mean().sqrt()) for g in (gm,gp)) > 1e-12:
        directions, gradient_info = mixed_directions(gm,gp,parent_spec['direction_weights'])
        with torch.no_grad():
            for weight,direction in directions.items():
                if direction is None: continue
                for amplitude in parent_spec['rms_latent_steps']:
                    rendered = session.render(session.initial-amplitude*direction)
                    valid,stats = within_trust(rendered[2],session.proposal,limits)
                    trials.append(dict(lpips_weight=weight,amplitude=amplitude,
                                       metrics=score(rendered[2]),within_trust=valid,edit_stats=stats))
                    images.append(rendered[1].detach().cpu())
                    if valid:admissible.append(len(trials)-1)
    else:
        gradient_info = dict(negligible=True)
    raw,_ = choose_trial(trials,baseline,baseline,'psnr_budget',parent_spec['psnr_gain_target'])
    bounded,_ = choose_trial([trials[i] for i in admissible],baseline,baseline,
                             'psnr_budget',parent_spec['psnr_gain_target'])
    bounded = None if bounded is None else admissible[bounded]
    return dict(outputs=dict(original=original,fp32_control=control[1].detach().cpu(),
                             raw_step1=original if raw is None else images[raw],
                             bounded_step1=original if bounded is None else images[bounded]),
                trials=trials,raw_choice=raw,bounded_choice=bounded,
                predicted_baseline=baseline,gradient=gradient_info,gt_used_for_selection=False)
