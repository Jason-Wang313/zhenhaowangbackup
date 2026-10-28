"""GT-only trade-off diagnostics, NOT a learned/deployable quality selector."""
from __future__ import annotations

import math
import torch

from lowlight_restoration.flux_tail_diagnostic import continuous_blend, rms_direction


def round8_ste(value):
    """Exact rounded forward, identity surrogate backward (not true derivative)."""
    quantized = value.add(1).mul(127.5).round().clamp(0, 255).div(127.5).sub(1)
    return quantized.detach() + (value-value.detach())


def surrogate_image(observation, raw, kind):
    if kind == 'continuous':
        return continuous_blend(observation, raw)
    if kind != 'double_round_ste':
        raise ValueError(kind)
    # Preserve even alpha=1 arithmetic from deployed_proposal; near a rounding
    # boundary lq+(raw-lq) need not be bit-identical to raw.
    first = observation.float().add((raw.float()-observation.float())*1.).clamp(-1, 1)
    first = round8_ste(first)
    second = observation.float().add((first-observation.float())*.3).clamp(-1, 1)
    return round8_ste(second)


def objective_directions(mse_gradient, lpips_gradient, legacy_gradient=None):
    """Separate value normalization from gradient-scale balancing."""
    m, m_rms = rms_direction(mse_gradient)
    p, p_rms = rms_direction(lpips_gradient)
    cosine = float((m*p).mean().clamp(-1, 1))
    vectors = dict(mse=m, lpips=p)
    diagnostics = dict(normalized_mse_gradient_rms=m_rms, normalized_lpips_gradient_rms=p_rms,
                       mse_to_lpips_gradient_rms_ratio=m_rms/p_rms, cosine=cosine,
                       directions={})
    # A separate backward of the joint scalar avoids silently replacing V1's
    # mixed-precision joint backward with a sum of separately rounded gradients.
    legacy = mse_gradient+lpips_gradient if legacy_gradient is None else legacy_gradient
    for name, vector in (('legacy_sum', legacy), ('balanced', m+p)):
        if float(vector.float().square().mean().sqrt()) <= 1e-12:
            vectors[name] = None  # opposing gradients: do not fabricate a direction
        else:
            vectors[name] = rms_direction(vector)[0]
    for name, direction in vectors.items():
        diagnostics['directions'][name] = None if direction is None else dict(
            predicted_mse_derivative_along_negative_direction=-float((mse_gradient*direction).sum()),
            predicted_lpips_derivative_along_negative_direction=-float((lpips_gradient*direction).sum()))
    return vectors, diagnostics


def joint_target(metrics, baseline, psnr_gain=.3, lpips_drop=.09928107050377874):
    return (metrics['psnr']-baseline['psnr'] >= psnr_gain
            and metrics['lpips'] <= baseline['lpips']*(1-lpips_drop))


def choose_safe_trial(trials, current, baseline):
    """GT-only local line search; exact quantized metrics, no deployment claim.

    No-harm is deliberately strict. A failed finite grid is inconclusive about
    wider neighborhoods or paths that initially trade one metric for another.
    """
    eligible = []
    for index, trial in enumerate(trials):
        value = trial['metrics']
        if not all(math.isfinite(value[k]) for k in ('psnr', 'lpips')):
            raise RuntimeError('nonfinite trial metric')
        safe = value['psnr'] >= current['psnr'] and value['lpips'] <= current['lpips']
        improves = value['psnr'] > current['psnr'] or value['lpips'] < current['lpips']
        if safe and improves:
            score = 10**((baseline['psnr']-value['psnr'])/10) + value['lpips']/max(baseline['lpips'], 1e-12)
            eligible.append((score, trial['amplitude'], index))
    return min(eligible)[2] if eligible else None
