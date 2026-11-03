"""Current-policy prefix, detached state, one differentiable native FLUX call.

There is no GT in state generation and no extra restoration network. The prefix
uses the SAME parameters as the last call, refreshed for every presentation.
This is a semi-gradient objective: derivatives through the prefix are omitted.
"""
class StateReady(Exception):
    pass


class CaptureBeforeCall:
    def __init__(self, model):
        self.model = model
        self.calls = 0
        self.inputs = None

    def __call__(self, **kwargs):
        if self.calls == 25:
            self.inputs = kwargs
            raise StateReady()
        result = self.model(**kwargs)
        self.calls += 1
        return result


def detached_prefix(model, sampler):
    """sampler may itself use inference_mode; clone OUTSIDE it for autograd."""
    import torch
    observer = CaptureBeforeCall(model)
    with torch.no_grad():
        try:
            sampler(observer)
        except StateReady:
            pass
        else:
            raise ValueError('sampler did not reach the fixed readout')
    if observer.calls != 25 or observer.inputs is None:
        raise ValueError('expected exactly 25 prefix model calls')
    # inference tensors cannot be saved for backward. Merely detach() is not enough.
    with torch.inference_mode(False), torch.no_grad():
        inputs = {k: v.detach().clone() if isinstance(v, torch.Tensor) else v
                  for k, v in observer.inputs.items()}
    return inputs


def generated_inputs(model, ae, lq, prompt, seed):
    from flux2.sampling import batched_prc_img
    from smoke.evaluate_foundir_blur_only_flux2_full_development_ddp import restore_batch
    return detached_prefix(model, lambda current: restore_batch(
        current, ae, lq, prompt['cfg_context'], prompt['cfg_ids'], batched_prc_img,
        seeds=[seed], steps=50, reference_time=10))


def predict_endpoint(model, inputs):
    import torch
    from flux2.sampling import scatter_ids
    x = inputs['x']
    if x.shape[0] != 2 or x.shape[1] % 2 or inputs['guidance'] is not None:
        raise ValueError('expected native duplicated CFG with reference tokens')
    n = x.shape[1] // 2
    with torch.autocast(x.device.type, dtype=torch.bfloat16, enabled=x.is_cuda):
        result = model(**inputs)
    uncond, cond = result[:, :n].chunk(2)
    # Preserve native BF16 CFG arithmetic even at guidance=1. Do not replace by cond.
    guided = uncond + 1.0 * (cond - uncond)
    tokens = x[:1, :n].float() - inputs['timesteps'][:1].float().view(-1, 1, 1) * guided.float()
    return torch.cat(scatter_ids(tokens, inputs['x_ids'][:1, :n])).squeeze(2)


def decode_endpoint(ae, latent, *, checkpointed):
    import torch
    from torch.utils.checkpoint import checkpoint
    def decode(z):
        with torch.autocast(z.device.type, dtype=torch.bfloat16, enabled=z.is_cuda):
            return ae.decode(z).float().clamp(-1, 1)
    return checkpoint(decode, latent, use_reentrant=False) if checkpointed else decode(latent)


def objective(model, ae, inputs, gt, metric, weights):
    import torch
    if set(weights) != {'gt_rgb_mse', 'gt_lpips'}:
        raise ValueError('only the two authorized GT image losses are allowed')
    latent = predict_endpoint(model, inputs)
    decoded = decode_endpoint(ae, latent, checkpointed=True)
    if decoded.shape != gt.shape or not torch.isfinite(decoded).all():
        raise ValueError('invalid decoded image')
    terms = {'gt_rgb_mse': (decoded - gt).square().mean(),
             'gt_lpips': metric((decoded + 1) * .5, (gt + 1) * .5)}
    return sum(weights[k] * v for k, v in terms.items()), terms


def training_position(step, micro, rank):
    if step not in range(400, 600) or micro not in range(4) or rank not in range(8):
        raise ValueError('outside the authorized 200 updates / global32')
    return step * 32 + micro * 8 + rank


def probe_indices(step, rank):
    if rank not in range(8) or step not in (400, 450, 500, 600):
        raise ValueError('unexpected checkpoint or rank')
    # Actual first 32 training presentations, not unrelated never-seen crops.
    values = list(range(12800 + rank, 12832, 8))
    if step == 600:
        values += list(range(64032 + rank, 64132, 8))
    return values
