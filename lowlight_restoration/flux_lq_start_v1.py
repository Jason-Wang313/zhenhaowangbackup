"""Fixed inference-only LQ-start counterfactual. No GT, training or strength search."""
from contextlib import nullcontext
import math


def select_suffix(schedule, maximum_start_time=0.5):
    if (len(schedule)<2 or schedule[0]!=1.0 or schedule[-1]!=0.0
            or not all(math.isfinite(t) for t in schedule)
            or not all(a>b for a,b in zip(schedule,schedule[1:]))
            or not 0 < maximum_start_time <= 1):
        raise ValueError('strict descending noise-to-data schedule required')
    index=next(i for i,t in enumerate(schedule) if t<=maximum_start_time)
    if index==len(schedule)-1:raise ValueError('no nonzero start on this schedule')
    return index,schedule[index:]


def initial_latent(reference, noise, time):
    import torch
    if reference.shape!=noise.shape or reference.dtype!=noise.dtype or reference.device!=noise.device:
        raise ValueError('matched reference and deterministic noise required')
    if not 0 < time <= 1:raise ValueError('nonzero interpolation time required')
    # Preserve the existing native BF16 straight-path arithmetic.
    t=torch.as_tensor(time,device=reference.device,dtype=reference.dtype)
    return (1-t)*reference+t*noise


def restore_lq_start(model, ae, lq, prompt, *, seed):
    import torch
    from flux2.sampling import batched_prc_img, denoise_cfg, get_schedule, scatter_ids
    from smoke.evaluate_foundir_blur_only_flux2_full_development_ddp import deterministic_noise, expand_cfg_condition
    with torch.inference_mode():
        reference=ae.encode(lq).to(torch.bfloat16)
        reference_tokens,reference_ids=batched_prc_img(reference)
        reference_ids=reference_ids.clone();reference_ids[...,0]=10
        schedule=get_schedule(50,reference_tokens.shape[1]);index,suffix=select_suffix(schedule)
        noise=deterministic_noise(reference,[seed])
        tokens,ids=batched_prc_img(initial_latent(reference,noise,suffix[0]))
        context,context_ids=expand_cfg_condition(prompt['cfg_context'],prompt['cfg_ids'],lq.shape[0])
        calls=0
        def counted(**kwargs):
            nonlocal calls
            calls+=1
            return model(**kwargs)
        amp=torch.autocast(device_type='cuda',dtype=torch.bfloat16) if lq.is_cuda else nullcontext()
        with amp:
            restored=denoise_cfg(counted,tokens,ids,context,context_ids,timesteps=suffix,guidance=1.,
                                 img_cond_seq=reference_tokens,img_cond_seq_ids=reference_ids)
        latent=torch.cat(scatter_ids(restored,ids)).squeeze(2)
        raw=ae.decode(latent).float().clamp(-1,1)
        if calls!=len(suffix)-1 or not torch.isfinite(raw).all():raise ValueError('invalid suffix rollout')
    return raw,{'start_index':index,'start_time':suffix[0],'model_calls':calls,'original_steps':50,
                'maximum_start_time':0.5,'seed':seed,'reference_time':10,'guidance':1.,'GT_input':False}
