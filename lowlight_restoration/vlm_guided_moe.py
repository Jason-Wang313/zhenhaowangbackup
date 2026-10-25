"""Minimal image-router MoE with an optional frozen-VLM score branch.

The module deliberately contains one expert bank.  VLM tokens never enter the
experts and never form a second MoE; a small projector turns their pooled
representation into an additive adjustment to the image router's logits.
"""

from __future__ import annotations

import copy
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class MoERouting:
    """Inspectable routing values returned by :class:`VLMGuidedSoftMoE`."""

    image_logits: Tensor
    vlm_logit_adjustment: Tensor
    final_logits: Tensor
    weights: Tensor


class FeedForwardRestorationExpert(nn.Module):
    """A conventional residual-branch FFN expert for qualification/pilots."""

    def __init__(self, feature_dim: int, hidden_dim: int) -> None:
        super().__init__()
        if feature_dim <= 0 or hidden_dim <= 0:
            raise ValueError("expert dimensions must be positive")
        self.network = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, feature_dim),
        )

    def forward(self, image_tokens: Tensor) -> Tensor:
        return self.network(image_tokens)


class VLMGuidedSoftMoE(nn.Module):
    """One expert bank routed by image features plus optional VLM score bias.

    Parameters
    ----------
    feature_dim:
        Channel dimension of restoration/DiT image tokens.
    vlm_dim:
        Channel dimension of each frozen VLM token (or pooled VLM feature).
    num_experts:
        Number of experts in the single MoE bank.
    expert_hidden_dim:
        Hidden width of the default FFN experts.
    router_hidden_dim:
        Hidden width of the image router and VLM projector.
    top_k:
        Number of experts retained after routing. ``None`` uses dense soft MoE.
    vlm_scale:
        Fixed multiplier applied to the VLM logit adjustment.
    zero_initialize_vlm_output:
        If true, enabling the VLM branch initially reproduces image-only
        routing exactly while still allowing the score head to learn.
    """

    def __init__(
        self,
        *,
        feature_dim: int,
        vlm_dim: int,
        num_experts: int,
        expert_hidden_dim: int,
        router_hidden_dim: int,
        top_k: int | None = None,
        vlm_scale: float = 1.0,
        zero_initialize_vlm_output: bool = True,
        experts: nn.ModuleList | None = None,
    ) -> None:
        super().__init__()
        if feature_dim <= 0 or vlm_dim <= 0 or router_hidden_dim <= 0:
            raise ValueError("router dimensions must be positive")
        if num_experts < 2:
            raise ValueError("MoE requires at least two experts")
        if top_k is not None and not 1 <= top_k <= num_experts:
            raise ValueError("top_k must be between one and num_experts")
        if not torch.isfinite(torch.tensor(vlm_scale)):
            raise ValueError("vlm_scale must be finite")

        self.feature_dim = int(feature_dim)
        self.vlm_dim = int(vlm_dim)
        self.num_experts = int(num_experts)
        self.top_k = top_k
        self.vlm_scale = float(vlm_scale)

        self.image_router = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, router_hidden_dim),
            nn.SiLU(),
            nn.Linear(router_hidden_dim, num_experts),
        )
        self.vlm_projector = nn.Sequential(
            nn.LayerNorm(vlm_dim),
            nn.Linear(vlm_dim, router_hidden_dim),
            nn.SiLU(),
            nn.Linear(router_hidden_dim, num_experts),
        )
        if zero_initialize_vlm_output:
            nn.init.zeros_(self.vlm_projector[-1].weight)
            nn.init.zeros_(self.vlm_projector[-1].bias)

        if experts is None:
            experts = nn.ModuleList(
                [
                    FeedForwardRestorationExpert(feature_dim, expert_hidden_dim)
                    for _ in range(num_experts)
                ]
            )
        if len(experts) != num_experts:
            raise ValueError("expert count does not match num_experts")
        self.experts = experts

    @staticmethod
    def _pool_tokens(tokens: Tensor, expected_dim: int, name: str) -> Tensor:
        if tokens.ndim == 2:
            pooled = tokens
        elif tokens.ndim == 3:
            if tokens.shape[1] == 0:
                raise ValueError(f"{name} cannot contain zero tokens")
            pooled = tokens.mean(dim=1)
        else:
            raise ValueError(f"{name} must have shape [B,D] or [B,L,D]")
        if pooled.shape[-1] != expected_dim:
            raise ValueError(
                f"{name} feature dimension changed: expected {expected_dim}, got {pooled.shape[-1]}"
            )
        return pooled

    def _routing(
        self,
        image_tokens: Tensor,
        vlm_tokens: Tensor | None,
        *,
        use_vlm: bool,
    ) -> MoERouting:
        image_summary = self._pool_tokens(image_tokens, self.feature_dim, "image_tokens")
        image_logits = self.image_router(image_summary)

        if use_vlm:
            if vlm_tokens is None:
                raise ValueError("vlm_tokens are required when use_vlm=True")
            vlm_summary = self._pool_tokens(vlm_tokens, self.vlm_dim, "vlm_tokens")
            if vlm_summary.shape[0] != image_summary.shape[0]:
                raise ValueError("image and VLM batch sizes differ")
            vlm_adjustment = self.vlm_projector(vlm_summary) * self.vlm_scale
        else:
            vlm_adjustment = torch.zeros_like(image_logits)

        final_logits = image_logits + vlm_adjustment
        if self.top_k is None or self.top_k == self.num_experts:
            weights = torch.softmax(final_logits, dim=-1)
        else:
            selected_logits, selected_indices = torch.topk(final_logits, self.top_k, dim=-1)
            selected_weights = torch.softmax(selected_logits, dim=-1)
            weights = torch.zeros_like(final_logits).scatter(-1, selected_indices, selected_weights)
        return MoERouting(
            image_logits=image_logits,
            vlm_logit_adjustment=vlm_adjustment,
            final_logits=final_logits,
            weights=weights,
        )

    def forward(
        self,
        image_tokens: Tensor,
        vlm_tokens: Tensor | None = None,
        *,
        use_vlm: bool = True,
    ) -> tuple[Tensor, MoERouting]:
        if image_tokens.ndim != 3:
            raise ValueError("image_tokens must have shape [B,L,D]")
        routing = self._routing(image_tokens, vlm_tokens, use_vlm=use_vlm)
        expert_outputs = torch.stack(
            [expert(image_tokens) for expert in self.experts],
            dim=1,
        )
        mixed = torch.sum(
            expert_outputs * routing.weights[:, :, None, None],
            dim=1,
        )
        return mixed, routing


def routing_load_balance_loss(weights: Tensor) -> Tensor:
    """Penalize batch-average expert utilization away from a uniform target."""

    if weights.ndim != 2 or weights.shape[1] < 2:
        raise ValueError("weights must have shape [B,num_experts>=2]")
    mean_usage = weights.mean(dim=0)
    target = torch.full_like(mean_usage, 1.0 / weights.shape[1])
    return torch.mean((mean_usage - target).square())


class ConditionedMoEMLP(nn.Module):
    """Drop-in replacement for one existing image MLP inside a DiT block.

    Every expert starts as an exact clone of the replaced MLP.  Consequently,
    the weighted output equals the pretrained MLP at insertion time, regardless
    of router weights.  Runtime VLM conditioning is supplied by the scoped
    :func:`vlm_moe_conditioning` context manager.
    """

    def __init__(
        self,
        original_mlp: nn.Module,
        *,
        feature_dim: int,
        vlm_dim: int,
        num_experts: int,
        router_hidden_dim: int,
        top_k: int | None,
        vlm_scale: float,
    ) -> None:
        super().__init__()
        experts = nn.ModuleList([copy.deepcopy(original_mlp) for _ in range(num_experts)])
        self.moe = VLMGuidedSoftMoE(
            feature_dim=feature_dim,
            vlm_dim=vlm_dim,
            num_experts=num_experts,
            expert_hidden_dim=feature_dim,
            router_hidden_dim=router_hidden_dim,
            top_k=top_k,
            vlm_scale=vlm_scale,
            zero_initialize_vlm_output=True,
            experts=experts,
        )
        self._vlm_tokens: Tensor | None = None
        self._use_vlm = False
        self.last_routing: MoERouting | None = None

    def set_conditioning(self, vlm_tokens: Tensor | None, *, use_vlm: bool) -> None:
        if use_vlm and vlm_tokens is None:
            raise ValueError("vlm_tokens are required when use_vlm=True")
        self._vlm_tokens = vlm_tokens
        self._use_vlm = bool(use_vlm)

    def clear_conditioning(self) -> None:
        self._vlm_tokens = None
        self._use_vlm = False
        self.last_routing = None

    def forward(self, image_tokens: Tensor) -> Tensor:
        output, routing = self.moe(
            image_tokens,
            self._vlm_tokens,
            use_vlm=self._use_vlm,
        )
        self.last_routing = routing
        return output


def inject_vlm_guided_moe_into_flux_double_blocks(
    model: nn.Module,
    *,
    block_indices: list[int],
    vlm_dim: int,
    num_experts: int,
    router_hidden_dim: int,
    top_k: int | None = None,
    vlm_scale: float = 1.0,
    trainable: bool = True,
) -> dict[str, ConditionedMoEMLP]:
    """Replace selected FLUX double-block image MLPs with one soft-MoE bank."""

    if not block_indices or len(set(block_indices)) != len(block_indices):
        raise ValueError("block_indices must be non-empty and unique")
    blocks = getattr(model, "double_blocks", None)
    if not isinstance(blocks, nn.ModuleList):
        raise TypeError("model.double_blocks must be an nn.ModuleList")

    injected: dict[str, ConditionedMoEMLP] = {}
    for index in sorted(block_indices):
        if not 0 <= index < len(blocks):
            raise IndexError(f"double block index out of range: {index}")
        block = blocks[index]
        original = getattr(block, "img_mlp", None)
        if not isinstance(original, nn.Module) or isinstance(original, ConditionedMoEMLP):
            raise TypeError(f"double_blocks.{index}.img_mlp is not an uninjected module")
        feature_dim = getattr(block, "hidden_size", None)
        if not isinstance(feature_dim, int) or feature_dim <= 0:
            raise TypeError(f"double_blocks.{index}.hidden_size is invalid")
        replacement = ConditionedMoEMLP(
            original,
            feature_dim=feature_dim,
            vlm_dim=vlm_dim,
            num_experts=num_experts,
            router_hidden_dim=router_hidden_dim,
            top_k=top_k,
            vlm_scale=vlm_scale,
        )
        reference_parameter = next(original.parameters(), None)
        if reference_parameter is not None:
            replacement.to(
                device=reference_parameter.device,
                dtype=reference_parameter.dtype,
            )
        replacement.requires_grad_(trainable)
        setattr(block, "img_mlp", replacement)
        injected[f"double_blocks.{index}.img_mlp"] = replacement
    return injected


def conditioned_moe_modules(model: nn.Module) -> dict[str, ConditionedMoEMLP]:
    return {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, ConditionedMoEMLP)
    }


@contextmanager
def vlm_moe_conditioning(
    model: nn.Module,
    vlm_tokens: Tensor | None,
    *,
    use_vlm: bool,
) -> Iterator[None]:
    """Set one forward's VLM condition and reliably clear it afterwards."""

    modules = conditioned_moe_modules(model)
    if not modules:
        raise RuntimeError("model contains no ConditionedMoEMLP modules")
    for module in modules.values():
        module.set_conditioning(vlm_tokens, use_vlm=use_vlm)
    try:
        yield
    finally:
        for module in modules.values():
            module.clear_conditioning()
