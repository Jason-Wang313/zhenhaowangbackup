"""Image logits + zero-initialized VLM correction for the frozen five-LoRA bank.

Only two small MLPs train. Full native rollout is differentiated (checkpointed),
not just a last-call surrogate. Discrete anchor is detached; train-label CE
teaches that choice. Quantization uses exact forward and straight-through backward.
"""
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from .five_expert_flux import ORDER, plan, weights_for_call
from .protocol_routed_flux import RouteBoundFlux


def enable_routed_checkpointing(model):
    """Rebind gates during EACH block recomputation, after outer call has returned.

    The old native checkpoint wrapper captures only activations. It would see
    restored NaN sentinel buffers in backward. Carry gates explicitly instead.
    """
    from torch.func import functional_call
    class Invocation(nn.Module):
        def __init__(self, block):
            super().__init__()
            self.block = block
            self.original = block.forward_kv_extract

        def forward(self, *args, **kwargs):
            return self.original(*args, **kwargs)

    def wrap(block):
        if getattr(block, '_router_checkpointed', False) or getattr(block, '_rain_flow_checkpointed', False):
            raise ValueError('checkpoint wrapper already installed')
        proxy = Invocation(block)
        names = tuple(n for n, _ in block.named_buffers() if n.endswith('route_weights'))
        if not names:
            raise ValueError('block lacks explicit expert gates')
        def call(*args, **kwargs):
            if not torch.is_grad_enabled():
                return proxy.original(*args, **kwargs)
            values = tuple(block.get_buffer(n).clone() for n in names)
            def replay(gates, *inputs, **kw):
                # Outer functional_call temporarily aliases all gate buffers to
                # one tensor. Each captured clone is explicit, not a tied parameter.
                return functional_call(proxy, {'block.'+n: w for n, w in zip(names, gates)}, inputs, kw,
                                       tie_weights=False)
            return checkpoint(replay, values, *args, use_reentrant=False, **kwargs)
        block.forward_kv_extract = call
        block._router_checkpointed = True
    for block in list(model.double_blocks) + list(model.single_blocks):
        wrap(block)


class FrozenExpertRouter(nn.Module):
    def __init__(self, image_dim=6144, vlm_dim=3584, hidden=256):
        super().__init__()
        def head(dim):
            return nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.SiLU(), nn.Linear(hidden, 5))
        self.image = head(image_dim)
        self.vlm = head(vlm_dim)
        nn.init.zeros_(self.vlm[-1].weight)
        nn.init.zeros_(self.vlm[-1].bias)

    def forward(self, image, vlm, *, vlm_enabled=True):
        if image.requires_grad or vlm.requires_grad or not torch.isfinite(image).all() or not torch.isfinite(vlm).all():
            raise ValueError('router inputs must be finite frozen image features, not task labels')
        image_logits = self.image(image.float())
        delta = self.vlm(vlm.float())
        logits = image_logits + (delta if vlm_enabled else delta * 0.)
        return logits, logits.softmax(-1), image_logits, delta


def pooled_image_features(model, reference):
    from flux2.sampling import batched_prc_img
    with torch.no_grad():
        tokens, _ = batched_prc_img(reference.to(torch.bfloat16))
        features = model.img_in(tokens).float()
        return torch.cat((features.mean(1), features.std(1, unbiased=False)), -1).detach()


class RoundUint8STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return ((x.float() + 1) * 127.5).round().clamp(0, 255) / 127.5 - 1

    @staticmethod
    def backward(ctx, grad):
        return grad


def quantized_output(lq, raw, route):
    if route['anchor'] == 'rain':
        return RoundUint8STE.apply(raw.float())
    blended = (lq.float() + (raw.float() - lq.float())).clamp(-1, 1)
    first = RoundUint8STE.apply(blended)
    if not route['second_quantization']:
        return first
    return RoundUint8STE.apply((lq.float() + (first - lq.float()) * route['alpha']).clamp(-1, 1))


def run_steps(state, weights, schedule, anchor, predict, *, checkpointed):
    """Generic native CFG loop. Explicit immutable index in each recompute closure.

    state is duplicated [unconditional, conditional]. predict returns native
    guided (single-batch) velocity, retaining BF16 CFG arithmetic.
    """
    if len(schedule) != 51:
        raise ValueError('retain the original 50-step time grid')
    calls = 26 if anchor == 'rain' else 50
    for index in range(calls):
        # Defaults freeze every iteration's values. Late-bound loop variables
        # would silently recompute all backward calls at the last time/index.
        def step(x, w, ci=index, t=schedule[index], prev=schedule[index+1]):
            gate = weights_for_call(w, ci, anchor)
            velocity = predict(x, gate, ci, t)
            if anchor == 'rain' and ci == 25:
                # Use BF16-rounded model time, exactly as endpoint qualification.
                time_value = torch.full((1, 1, 1), t, device=x.device, dtype=x.dtype).float()
                return x[:1].float() - time_value * velocity.float()
            return x + (prev - t) * torch.cat((velocity, velocity), 0)
        state = checkpoint(step, state, weights, use_reentrant=False) if checkpointed else step(state, weights)
    return state if anchor == 'rain' else state[:1]


def differentiable_restore(model, gates, protocols, weights, ae, lq, reference, prompts, *, seed, checkpointed=True):
    from flux2.sampling import batched_prc_img, get_schedule, scatter_ids
    from smoke.evaluate_foundir_blur_only_flux2_full_development_ddp import deterministic_noise, expand_cfg_condition
    route = plan(protocols, weights)
    prompt = prompts[route['anchor']]
    # Frozen cached encoding, no GT and no gradients through feature extraction.
    ref_tokens, ref_ids = batched_prc_img(reference.to(torch.bfloat16))
    ref_ids = ref_ids.clone(); ref_ids[..., 0] = 10
    tokens, ids = batched_prc_img(deterministic_noise(reference, [seed]))
    state = torch.cat((tokens, tokens), 0)
    image_ids = torch.cat((ids, ids), 0)
    ref_tokens, ref_ids = torch.cat((ref_tokens, ref_tokens), 0), torch.cat((ref_ids, ref_ids), 0)
    ctx, ctx_ids = expand_cfg_condition(prompt['cfg_context'], prompt['cfg_ids'], 1)
    schedule = get_schedule(50, tokens.shape[1])
    def predict(x, gate, ci, t):
        bound = RouteBoundFlux(model, gates, gate)
        if bound.expert_order != ORDER:
            raise ValueError('wrong frozen expert bank')
        kwargs = {'x': torch.cat((x, ref_tokens), 1), 'x_ids': torch.cat((image_ids, ref_ids), 1),
                  'timesteps': torch.full((2,), t, device=x.device, dtype=x.dtype),
                  'ctx': ctx, 'ctx_ids': ctx_ids, 'guidance': None}
        with torch.autocast(x.device.type, dtype=torch.bfloat16, enabled=x.is_cuda):
            result = bound(**kwargs)[:, :tokens.shape[1]]
        uncond, cond = result.chunk(2)
        return uncond + 1.0 * (cond - uncond)
    endpoint = run_steps(state, weights, schedule, route['anchor'], predict, checkpointed=checkpointed)
    latent = torch.cat(scatter_ids(endpoint, ids)).squeeze(2)
    def decode(z):
        if route['anchor'] == 'rain':
            with torch.autocast(z.device.type, dtype=torch.bfloat16, enabled=z.is_cuda):
                return ae.decode(z).float().clamp(-1, 1)
        return ae.decode(z).float().clamp(-1, 1)
    raw = checkpoint(decode, latent, use_reentrant=False) if checkpointed else decode(latent)
    return raw, quantized_output(lq, raw, route), route


def objective(image, gt, metric, logits, label, *, lpips_weight):
    # Labels supervise the decision only here. They never enter model.forward,
    # prompt/horizon selection, generated state, or inference candidate selection.
    terms = {'rgb_mse': (image.float() - gt.float()).square().mean(),
             'lpips': metric((image.float()+1)*.5, (gt.float()+1)*.5),
             'route_ce': nn.functional.cross_entropy(logits, label)}
    loss = 10. * terms['rgb_mse'] + lpips_weight * terms['lpips'] + .1 * terms['route_ce']
    return loss, terms
