"""Explicit-input frozen LoRA bank. No sampler, router, or image decoder.

Mix B_i(A_i(x)), never mean(B_i)(mean(A_i)(x)). Routing is an explicit input,
not mutable module state: checkpoint recomputation must see identical gates.
This primitive does NOT prove full FLUX rollout/prompt compatibility.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

FLOW_CONTRACT = "native_reference_noise_to_clean_velocity_v1"


@dataclass(frozen=True)
class LoRAWeights:
    name: str
    a: Tensor
    b: Tensor
    alpha: float
    contract: str = FLOW_CONTRACT


class FrozenLoRAExpertBank(nn.Module):
    """One original Linear with immutable, independently loaded LoRA deltas.

    Preserve checkpoint dtype. Caller must provide the SAME autocast context as
    the legacy forward. Mixed residual accumulation is FP32; the original base
    addition and single-expert residual scaling are retained.
    """

    def __init__(self, base: nn.Linear, experts: list[LoRAWeights]):
        super().__init__()
        if not experts or len({e.name for e in experts}) != len(experts):
            raise ValueError("expert names must be nonempty and unique")
        for e in experts:
            if e.contract != FLOW_CONTRACT:
                raise ValueError("endpoint experts cannot be installed in the flow bank")
            if (not e.name or e.a.ndim != 2 or e.b.ndim != 2 or e.a.shape[0] <= 0
                    or e.a.shape[1] != base.in_features
                    or e.b.shape != (base.out_features, e.a.shape[0])):
                raise ValueError("expert shape does not match original Linear")
            if (not math.isfinite(e.alpha) or e.alpha <= 0
                    or e.a.dtype != e.b.dtype or not e.a.is_floating_point()
                    or not torch.isfinite(e.a).all() or not torch.isfinite(e.b).all()):
                raise ValueError("invalid expert dtype, alpha, or nonfinite weights")
        self.base = base.requires_grad_(False)
        self.names = tuple(e.name for e in experts)
        self.scales = tuple(e.alpha / e.a.shape[0] for e in experts)
        self.a = nn.ParameterList([
            nn.Parameter(e.a.detach().clone().to(base.weight.device), requires_grad=False)
            for e in experts])
        self.b = nn.ParameterList([
            nn.Parameter(e.b.detach().clone().to(base.weight.device), requires_grad=False)
            for e in experts])

    def _delta(self, value: Tensor, index: int, base_output: Tensor) -> Tensor:
        # Exactly the original LoRALinear order: matmuls -> cast -> scale.
        return F.linear(F.linear(value, self.a[index]), self.b[index]).to(base_output) * self.scales[index]

    def forward_expert(self, value: Tensor, index: int) -> Tensor:
        """Diagnostic hard selection, not a trainable/discrete router."""
        if not isinstance(index, int) or not 0 <= index < len(self.names):
            raise ValueError("invalid expert index")
        base_output = self.base(value)
        return base_output + self._delta(value, index, base_output)

    def forward(self, value: Tensor, weights: Tensor) -> Tensor:
        if value.ndim < 2:
            raise ValueError("input needs batch and feature axes")
        if weights.shape != (value.shape[0], len(self.names)):
            raise ValueError("weights must be [batch, experts], not hidden global state")
        if (not weights.is_floating_point() or weights.device != value.device
                or not torch.isfinite(weights).all() or (weights < 0).any()
                or not torch.allclose(weights.float().sum(-1),
                                      torch.ones(value.shape[0], device=value.device), atol=1e-6, rtol=1e-6)):
            raise ValueError("weights must be finite, nonnegative and sum to one")
        base_output = self.base(value)
        merged = torch.zeros_like(base_output, dtype=torch.float32)
        shape = (value.shape[0],) + (1,) * (value.ndim - 1)
        for i in range(len(self.names)):
            # Do not skip zero weights: soft router gradients must remain defined.
            merged = merged + self._delta(value, i, base_output).float() * weights[:, i].float().view(shape)
        return base_output + merged.to(base_output)


class HardSelectedReplayBank(nn.Module):
    """Constant expert for old sampler equivalence checks, NOT learned routing."""

    def __init__(self, bank: FrozenLoRAExpertBank, expert_index: int):
        super().__init__()
        if not 0 <= expert_index < len(bank.names):
            raise ValueError("invalid expert index")
        self.bank = bank
        self.expert_index = expert_index

    def forward(self, value: Tensor) -> Tensor:
        return self.bank.forward_expert(value, self.expert_index)


def install_replay_bank(model: nn.Module, *, modules: list[str],
                        expert_states: list[tuple[str, dict[str, Tensor], float, str]],
                        expert_index: int) -> list[str]:
    """Install all four banks, statically selecting one for original-sampler replay.

    Call on a fresh unwrapped model, NOT the existing trained FFN-MoE. Check all
    keys/shapes/contracts and construct replacements before changing the model.
    Dynamic VLM routing is deliberately not implemented by this replay helper.
    """
    if not modules or len(modules) != len(set(modules)):
        raise ValueError("module list must be nonempty and unique")
    if not 0 <= expert_index < len(expert_states):
        raise ValueError("invalid expert index")
    expected = {f"{m}.lora_{part}.weight" for m in modules for part in ("A", "B")}
    for _, state, _, contract in expert_states:
        if set(state) != expected:
            raise ValueError("checkpoint keys must match every selected module exactly")
        if contract != FLOW_CONTRACT:
            raise ValueError("endpoint experts cannot be installed in the flow bank")
    replacements = {}
    for module in modules:
        base = model.get_submodule(module)
        if type(base) is not nn.Linear:
            raise ValueError("requires original unwrapped Linear modules")
        weights = [LoRAWeights(name, state[f"{module}.lora_A.weight"],
                               state[f"{module}.lora_B.weight"], alpha, contract)
                   for name, state, alpha, contract in expert_states]
        # Use a shallow shell to avoid changing original requires_grad on a failed audit.
        # Actual base parameters are shared only after all replacements validate.
        import copy
        shell = copy.copy(base)
        shell._parameters = {k: nn.Parameter(v.detach(), requires_grad=False) if v is not None else None
                             for k, v in base._parameters.items()}
        replacements[module] = HardSelectedReplayBank(FrozenLoRAExpertBank(shell, weights), expert_index)
    model.requires_grad_(False)
    for name, replacement in replacements.items():
        parent_name, _, child = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child, replacement)
    return list(replacements)
