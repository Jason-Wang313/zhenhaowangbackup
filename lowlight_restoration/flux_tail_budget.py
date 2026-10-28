"""Bounded GT-only PSNR-budget search; not an energy, router or novelty claim."""
from __future__ import annotations

import math
import torch

from lowlight_restoration.flux_tail_diagnostic import rms_direction


def mixed_directions(mse_gradient, lpips_gradient, weights):
    """weight=0 is MSE, weight=1 is LPIPS; normalize BEFORE and AFTER mixing."""
    m, mr = rms_direction(mse_gradient)
    p, pr = rms_direction(lpips_gradient)
    result = {}
    for weight in weights:
        if not 0 <= weight <= 1 or weight in result:
            raise ValueError('weights must be unique and in [0,1]')
        # Preserve V2's balanced arithmetic; avoid an unnecessary rescaling.
        vector = m+p if weight == .5 else (1-weight)*m + weight*p
        size = float(vector.float().square().mean().sqrt())
        result[weight] = None if size <= 1e-12 else rms_direction(vector)[0]
    return result, dict(mse_gradient_rms=mr, lpips_gradient_rms=pr,
                        mse_to_lpips_gradient_rms_ratio=mr/pr,
                        cosine=float((m*p).mean().clamp(-1, 1)))


def thresholds(current, baseline, rule, gain=.3):
    if rule == 'monotone':
        floor = current['psnr']
    elif rule == 'psnr_budget':
        # Before the target, retain PSNR progress. After it, spend only excess.
        floor = min(current['psnr'], baseline['psnr']+gain)
    else:
        raise ValueError(rule)
    return dict(psnr_floor=max(baseline['psnr'], floor),
                lpips_ceiling=min(baseline['lpips'], current['lpips']))


def choose_trial(trials, current, baseline, rule, gain=.3):
    """GT quantized scores only. Both paired policies use the same ranking.

    After reaching baseline+.3 dB, budget mode may lose excess PSNR only while
    reducing LPIPS. It may never lower the final PSNR target or increase LPIPS.
    First close the PSNR-target shortfall. Among feasible candidates, minimize
    LPIPS. No early stop on target: explicitly measure headroom up to cap.
    """
    bounds = thresholds(current, baseline, rule, gain)
    eligible, decisions = [], []
    for index, trial in enumerate(trials):
        value = trial['metrics']
        if not all(math.isfinite(value[k]) for k in ('psnr', 'lpips')):
            raise RuntimeError('nonfinite trial metrics')
        safe = (value['psnr'] >= bounds['psnr_floor']
                and value['lpips'] <= bounds['lpips_ceiling'])
        improves = (value['lpips'] < current['lpips']
                    or value['psnr'] > current['psnr'])
        decisions.append(dict(psnr_ok=value['psnr'] >= bounds['psnr_floor'],
                              lpips_ok=value['lpips'] <= bounds['lpips_ceiling'],
                              strictly_improves=improves, eligible=safe and improves))
        if safe and improves:
            shortfall = max(0., baseline['psnr']+gain-value['psnr'])
            eligible.append((shortfall, value['lpips'], -value['psnr'], trial['amplitude'],
                             trial['lpips_weight'], index))
    return (min(eligible)[-1] if eligible else None), dict(**bounds, trials=decisions)
