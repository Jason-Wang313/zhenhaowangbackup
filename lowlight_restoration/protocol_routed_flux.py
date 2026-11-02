"""Keep inherited expert protocols while mixing LoRA deltas inside one FLUX rollout.

Routing probabilities are explicit inputs. Their argmax selects the original
prompt/output protocol; the full vector mixes internal deltas. No image ensemble,
alpha averaging, GT labels, CNN or learned router is implemented here.
"""
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.func import functional_call

from .existing_flux_experts import FrozenLoRAExpertBank, HardSelectedReplayBank, install_replay_bank
from .scalar_residual import quantized_scalar_residual


def validate_weights(weights: Tensor, count: int):
    if (weights.ndim != 2 or weights.shape[1] != count or weights.shape[0] < 1
            or not weights.is_floating_point() or not torch.isfinite(weights).all()
            or (weights < 0).any() or not torch.allclose(weights.float().sum(-1),
                torch.ones(weights.shape[0],device=weights.device),atol=1e-6,rtol=1e-6)):
        raise ValueError("route needs finite nonnegative [B,E] probabilities summing to one")


def probabilities(image_logits: Tensor, vlm_delta: Tensor | None = None) -> Tensor:
    """Score plumbing only, not a trained image/VLM router. Zero delta is identity."""
    if image_logits.ndim != 2 or not torch.isfinite(image_logits).all():
        raise ValueError("invalid image scores")
    if vlm_delta is not None:
        if vlm_delta.shape != image_logits.shape or not torch.isfinite(vlm_delta).all():
            raise ValueError("invalid VLM score correction")
        image_logits = image_logits + vlm_delta
    return image_logits.float().softmax(-1)


@dataclass(frozen=True)
class ExpertProtocol:
    name: str
    prompt: str
    alpha: float
    second_quantization: bool


class ProtocolRouter:
    def __init__(self, protocols: list[ExpertProtocol]):
        if not protocols or len({p.name for p in protocols}) != len(protocols):
            raise ValueError("nonempty unique protocols required")
        if any(not p.prompt or not 0 <= p.alpha <= 1 or (not p.second_quantization and p.alpha != 1)
               for p in protocols):
            raise ValueError("invalid inherited output protocol")
        self.protocols = tuple(protocols)

    def anchor(self, weights: Tensor) -> ExpertProtocol:
        validate_weights(weights,len(self.protocols))
        if weights.shape[0] != 1:
            raise ValueError("full-image protocol adapter currently supports batch one")
        # Ties follow fixed expert order; no GT score or threshold is consulted.
        return self.protocols[int(weights[0].detach().argmax())]

    def output(self, lq: Tensor, decoded: Tensor, weights: Tensor) -> Tensor:
        protocol = self.anchor(weights)
        if lq.shape[0] != 1:
            raise ValueError("batch-one output adapter")
        first, normalized = quantized_scalar_residual(lq,decoded,1.)
        if not protocol.second_quantization:
            return first
        return quantized_scalar_residual(lq,normalized,protocol.alpha)[0]


class ExplicitGateLinear(nn.Module):
    def __init__(self, bank: FrozenLoRAExpertBank):
        super().__init__()
        self.bank = bank
        # Sentinel makes an accidental direct forward fail, rather than silently selecting expert0.
        self.register_buffer("route_weights",torch.full((1,len(bank.names)),float("nan"),device=bank.base.weight.device))

    def forward(self, value: Tensor) -> Tensor:
        weights = self.route_weights
        validate_weights(weights,len(self.bank.names))
        if value.shape[0] == 2*weights.shape[0]:
            weights = weights.repeat(2,1)  # CFG is [unconditional batch, conditional batch].
        elif value.shape[0] != weights.shape[0]:
            raise ValueError("unexpected image/CFG batch layout")
        return self.bank(value,weights)


def install_protocol_bank(model, *, modules, expert_states):
    """Old validated loader, new explicit per-invocation gates; parent source untouched."""
    install_replay_bank(model,modules=modules,expert_states=expert_states,expert_index=0)
    names = []
    for name in modules:
        previous = model.get_submodule(name)
        if not isinstance(previous,HardSelectedReplayBank):
            raise ValueError("unexpected original bank wrapper")
        parent,_,child = name.rpartition(".")
        target = model.get_submodule(parent) if parent else model
        setattr(target,child,ExplicitGateLinear(previous.bank))
        names.append(name+".route_weights")
    return tuple(names)


class RouteBoundFlux(nn.Module):
    """Bind one input route across all 50 sampling steps, without sticky global state.

functional_call substitutes buffers only for this invocation and restores the
sentinels afterwards. Activation-checkpoint callers must pass weights explicitly
again (tested on CPU); this helper is not a complete router training pipeline.
"""
    def __init__(self, model, gate_names, weights):
        super().__init__()
        self.model = model
        self.gate_names = tuple(gate_names)
        if not self.gate_names:
            raise ValueError("no installed gate sites")
        orders = {model.get_submodule(n.rsplit(".",1)[0]).bank.names for n in self.gate_names}
        if len(orders) != 1:
            raise ValueError("inconsistent expert banks")
        self.expert_order = orders.pop()
        validate_weights(weights,len(self.expert_order))
        self.weights = weights.clone()  # Caller-side mutation cannot change a running plan.

    def forward(self, *args, **kwargs):
        return functional_call(self.model,{name:self.weights for name in self.gate_names},args,kwargs,strict=False)


def restore_protocol_routed(model, gate_names, protocols, weights, autoencoder, lq, prompt_tensors,
                            pack_image, restore_batch, *, seed, steps=50, reference_time=10):
    """One FLUX trajectory with coupled protocol selection; inference batch one."""
    bound = RouteBoundFlux(model,gate_names,weights)
    if bound.expert_order != tuple(p.name for p in protocols.protocols):
        raise ValueError("protocol order must match internal expert order")
    protocol = protocols.anchor(bound.weights)
    prompt = prompt_tensors[protocol.name]
    decoded = restore_batch(bound,autoencoder,lq,prompt["cfg_context"],prompt["cfg_ids"],pack_image,
                            seeds=[seed],steps=steps,reference_time=reference_time)
    image = protocols.output(lq,decoded,bound.weights)
    return decoded,image,{"anchor":protocol.name,"prompt":protocol.prompt,"alpha":protocol.alpha,
                          "second_quantization":protocol.second_quantization,
                          "weights":bound.weights.detach().cpu().tolist()}
