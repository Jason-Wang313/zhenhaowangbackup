"""Inference-only five-LoRA adapter; preserve each anchor's native readout.

Rain step800 is qualified only through call index25 of the original 50-step
schedule. It must not be advertised as a full-trajectory velocity expert.
No learned router, RGB refiner, output ensemble or GT-dependent policy here.
"""
import torch

from .protocol_routed_flux import ProtocolRouter, RouteBoundFlux, validate_weights

ORDER = ('blur', 'noise', 'haze', 'lowlight', 'rain')


def plan(protocols: ProtocolRouter, weights):
    if tuple(p.name for p in protocols.protocols) != ORDER:
        raise ValueError('fixed five-expert order required')
    if any(p.alpha != a or p.second_quantization != (p.name not in ('haze', 'rain'))
           for p, a in zip(protocols.protocols, (.75, .30, 1., .925, 1.))):
        raise ValueError('inherited output protocols must not change')
    protocol = protocols.anchor(weights)
    return {'anchor': protocol.name, 'prompt': protocol.prompt, 'alpha': protocol.alpha,
            'second_quantization': protocol.second_quantization,
            'schedule_steps': 50, 'model_calls': 26 if protocol.name == 'rain' else 50,
            'readout': 'endpoint_at_call25' if protocol.name == 'rain' else 'final_flow',
            'rain_after_call25': 'zero_and_renormalize_nonrain',
            'weights': weights.detach().cpu().tolist()}


def weights_for_call(weights, call_index, anchor):
    validate_weights(weights, 5)
    if weights.shape[0] != 1 or anchor not in ORDER:
        raise ValueError('batch-one fixed anchor required')
    if ORDER[int(weights[0].detach().argmax())] != anchor:
        raise ValueError('anchor must be selected from the original route')
    calls = 26 if anchor == 'rain' else 50
    if type(call_index) is not int or not 0 <= call_index < calls:
        raise ValueError('outside anchor readout horizon')
    if call_index <= 25:
        return weights
    # Not a claim of smooth/optimal routing; a fixed, testable compatibility rule.
    masked = torch.cat((weights[:, :4], torch.zeros_like(weights[:, 4:])), -1)
    total = masked.sum(-1, keepdim=True)
    if (total <= 0).any():
        raise ValueError('no valid expert for the late trajectory')
    return masked / total


class InferenceTrajectory:
    """Fresh per rollout. Deliberately rejects training/checkpoint recomputation.

    Future router training must pass call index and gates explicitly; this
    inference counter is NOT safe to reuse inside autograd checkpointing.
    """
    def __init__(self, model, gates, weights, anchor):
        self.model, self.gates = model, gates
        self.weights, self.anchor = weights.clone(), anchor
        self.calls = 0

    def __call__(self, *args, **kwargs):
        if torch.is_grad_enabled():
            raise ValueError('inference-only adapter; no implicit counters in training')
        weights = weights_for_call(self.weights, self.calls, self.anchor)
        bound = RouteBoundFlux(self.model, self.gates, weights)
        if bound.expert_order != ORDER:
            raise ValueError('bank order mismatch')
        result = bound(*args, **kwargs)
        self.calls += 1
        return result


def restore_five_expert(model, gates, protocols, weights, ae, lq, prompts, *, seed):
    from flux2.sampling import batched_prc_img
    from smoke.evaluate_foundir_blur_only_flux2_full_development_ddp import restore_batch
    from .rain_onpolicy_midpoint import generated_inputs, predict_endpoint, decode_endpoint
    route = plan(protocols, weights)
    trajectory = InferenceTrajectory(model, gates, weights, route['anchor'])
    prompt = prompts[route['anchor']]
    with torch.no_grad():
        if route['anchor'] == 'rain':
            inputs = generated_inputs(trajectory, ae, lq, prompt, seed)
            raw = decode_endpoint(ae, predict_endpoint(trajectory, inputs), checkpointed=False)
        else:
            raw = restore_batch(trajectory, ae, lq, prompt['cfg_context'], prompt['cfg_ids'],
                                batched_prc_img, seeds=[seed], steps=50, reference_time=10)
        # Rain did NOT use LQ + (raw-LQ) even at alpha1: that roundtrip can
        # change quantization ties. Preserve its actual original arithmetic.
        image = ((raw.float() + 1) * 127.5).round().clamp(0, 255).to(torch.uint8) \
            if route['anchor'] == 'rain' else protocols.output(lq, raw, weights)
    if trajectory.calls != route['model_calls']:
        raise ValueError('actual model calls differ from declared readout')
    return raw, image, route
