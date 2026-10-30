"""Current-run FLUX reference; historical drift is reported, never a launch gate.

No GT or learning here. The original runtime still checks the current native
entry against the current BF16 tail; historical assets are never overwritten.
"""
from __future__ import annotations

import hashlib
import math


def uint8_image(normalized):
    import torch
    if not bool(torch.isfinite(normalized).all()):
        raise RuntimeError('nonfinite current FLUX reference')
    return ((normalized.detach().cpu()+1)*127.5).round().clamp(0,255).to(torch.uint8)


def image_sha(image):
    return hashlib.sha256(image.contiguous().numpy().tobytes()).hexdigest()


def current_session(flow, observation, seed, historical, current_expected=None):
    import torch
    # Do NOT modify the old bound runtime. Only its obsolete cross-environment
    # expected_proposal gate is omitted; same-environment native/tail check stays.
    session=flow.session(observation,seed,expected_proposal=None)
    current=uint8_image(session.proposal)[0]
    historical=historical.detach().cpu()
    if current.shape!=historical.shape or historical.dtype!=torch.uint8:
        raise RuntimeError('historical/current geometry or dtype differs')
    if current_expected is not None and not torch.equal(current,current_expected.cpu()):
        raise RuntimeError('current run reference changed since bootstrap')
    delta=current.double()-historical.double()
    info=dict(policy='record_historical_drift_use_current_run_reference',
              historical_equal=bool(torch.equal(current,historical)),
              different_elements=int((current!=historical).sum()),
              max_abs_uint8=float(delta.abs().max()),rms_uint8=float(delta.square().mean().sqrt()),
              historical_sha256=image_sha(historical),current_sha256=image_sha(current),
              same_environment_native_tail_check_retained=True)
    return session,current,info


def reference_tensors(session,current,labeled_reference):
    """Label current proposal once; use it for BOTH conditioning and anchor pairs."""
    return dict(proposal=current,proposal_normalized=session.proposal[0].detach().cpu(),
                proposal_scores=labeled_reference['scores'][0],
                proposal_gm=labeled_reference['gm'][0],proposal_gp=labeled_reference['gp'][0])


def aggregate_metrics(rows,spec):
    """Keep current, historical and FP32-control comparisons separate."""
    names=('original','historical','fp32_control','raw_step1','bounded_step1')
    if not rows or any(set(row['metrics'])!=set(names) for row in rows):
        raise RuntimeError('missing matched control metrics')
    means={name:{metric:sum(row['metrics'][name][metric] for row in rows)/len(rows)
                 for metric in ('psnr','lpips')} for name in names}
    if any(not math.isfinite(v) for metrics in means.values() for v in metrics.values()):
        raise RuntimeError('nonfinite metric')
    for name,value in means.items():
        comparisons={}
        for control in ('original','historical','fp32_control'):
            baseline=means[control]
            gain=value['psnr']-baseline['psnr']
            reduction=1-value['lpips']/max(baseline['lpips'],1e-12)
            comparisons[control]=dict(psnr_gain=gain,lpips_relative_reduction=reduction,
                joint_target=gain>=spec['psnr_gain_target'] and reduction>=spec['lpips_relative_reduction_target'])
        value.update(comparisons['original'])
        value['compared_with']=comparisons
    return means
