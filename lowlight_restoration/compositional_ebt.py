"""Minimal matched Energy-Based Transformer modules for restoration research.

This is a project adaptation of the Energy-Based Transformer idea in
Gladstone et al. (arXiv:2507.02092).  The official image-denoising code assigns
patch energies to a candidate image and updates the candidate with the energy
gradient.  Restoration additionally requires an observed degraded image, so
this module embeds ``[observation, candidate]`` patch pairs.

No training or evaluation is performed by this module itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Literal

import torch
from torch import Tensor, nn


DEGRADATIONS = ("blur", "noise", "rain", "haze", "low_light")
PAIR_INDICES = tuple(combinations(range(len(DEGRADATIONS)), 2))
NUM_ENERGY_COMPONENTS = 1 + len(DEGRADATIONS) + len(PAIR_INDICES)


def compositional_descriptor(q: Tensor) -> Tensor:
    """Return ``[base, five atomic, ten normalized pairwise]`` activations."""

    if q.ndim != 2 or q.shape[1] != len(DEGRADATIONS):
        raise ValueError(f"q must have shape [batch, {len(DEGRADATIONS)}]")
    if not torch.isfinite(q).all():
        raise ValueError("q contains non-finite values")
    if bool(((q < 0) | (q > 1)).any()):
        raise ValueError("q must lie in [0, 1]")

    pairwise = torch.stack([q[:, i] * q[:, j] for i, j in PAIR_INDICES], dim=1)
    pair_normalizer = pairwise.sum(dim=1, keepdim=True).clamp_min(1.0)
    pairwise = pairwise / pair_normalizer
    return torch.cat((torch.ones_like(q[:, :1]), q, pairwise), dim=1)


class EnergyBlock(nn.Module):
    """DiT-style adaLN block using math attention for higher-order gradients."""

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float) -> None:
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size)
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden, hidden_size),
        )
        self.modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size)
        )

    @staticmethod
    def _modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
        return x * (1 + scale[:, None]) + shift[:, None]

    def forward(self, x: Tensor, condition: Tensor) -> Tensor:
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = self.modulation(
            condition
        ).chunk(6, dim=1)
        attention_input = self._modulate(self.norm1(x), shift_a, scale_a)
        batch, tokens, hidden = attention_input.shape
        qkv = self.qkv(attention_input).reshape(
            batch, tokens, 3, self.num_heads, self.head_dim
        )
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        # The official EBT implementation also forces math attention because
        # flash/memory-efficient kernels do not support required higher-order
        # derivatives consistently.
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            attended = torch.nn.functional.scaled_dot_product_attention(
                query, key, value, dropout_p=0.0
            )
        attended = attended.transpose(1, 2).reshape(batch, tokens, hidden)
        x = x + gate_a[:, None] * self.proj(attended)
        mlp_input = self._modulate(self.norm2(x), shift_m, scale_m)
        return x + gate_m[:, None] * self.mlp(mlp_input)


@dataclass(frozen=True)
class EnergyStep:
    candidate: Tensor
    energy: Tensor
    candidate_gradient: Tensor


class RestorationEnergyTransformer(nn.Module):
    """Matched Plain/Compositional EBT over observation-candidate pairs.

    Both modes have exactly the same tensors, parameters and forward path.
    ``plain`` learns a dense descriptor-to-energy mixing matrix, while
    ``compositional`` masks that matrix to its diagonal, enforcing one base,
    five atomic and ten pairwise terms.
    """

    def __init__(
        self,
        *,
        image_size: int = 64,
        patch_size: int = 8,
        hidden_size: int = 64,
        depth: int = 2,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        mode: Literal["plain", "compositional"] = "compositional",
        zero_init_energy_head: bool = True,
    ) -> None:
        super().__init__()
        if image_size % patch_size:
            raise ValueError("image_size must be divisible by patch_size")
        if mode not in {"plain", "compositional"}:
            raise ValueError(f"unsupported mode: {mode}")
        self.image_size = image_size
        self.patch_size = patch_size
        self.mode = mode
        self.patch_embed = nn.Conv2d(
            6, hidden_size, kernel_size=patch_size, stride=patch_size
        )
        patch_grid = image_size // patch_size
        self.position = nn.Parameter(
            torch.zeros(1, patch_grid * patch_grid, hidden_size)
        )
        self.condition = nn.Parameter(torch.zeros(1, hidden_size))
        self.blocks = nn.ModuleList(
            [EnergyBlock(hidden_size, num_heads, mlp_ratio) for _ in range(depth)]
        )
        self.final_norm = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6
        )
        self.energy_head = nn.Linear(
            hidden_size, NUM_ENERGY_COMPONENTS, bias=False
        )
        # Same parameter exists and is evaluated in both arms. The scientific
        # intervention is only the fixed dense/diagonal mask.
        self.descriptor_mixer = nn.Parameter(
            torch.eye(NUM_ENERGY_COMPONENTS)
        )
        mask = (
            torch.ones(NUM_ENERGY_COMPONENTS, NUM_ENERGY_COMPONENTS)
            if mode == "plain"
            else torch.eye(NUM_ENERGY_COMPONENTS)
        )
        self.register_buffer("descriptor_mask", mask, persistent=True)
        self._initialize(zero_init_energy_head=zero_init_energy_head)

    def _initialize(self, *, zero_init_energy_head: bool) -> None:
        nn.init.xavier_uniform_(self.patch_embed.weight.flatten(1))
        nn.init.zeros_(self.patch_embed.bias)
        nn.init.normal_(self.position, std=0.02)
        nn.init.normal_(self.condition, std=0.02)
        for block in self.blocks:
            for module in block.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
        if zero_init_energy_head:
            nn.init.zeros_(self.energy_head.weight)
        else:
            nn.init.normal_(self.energy_head.weight, std=0.02)

    def component_energies(self, observation: Tensor, candidate: Tensor) -> Tensor:
        if observation.shape != candidate.shape:
            raise ValueError("observation and candidate shapes must match")
        if observation.ndim != 4 or observation.shape[1] != 3:
            raise ValueError("images must have shape [batch, 3, height, width]")
        if observation.shape[-2:] != (self.image_size, self.image_size):
            raise ValueError(
                f"expected {self.image_size}x{self.image_size} phase-0 tiles"
            )
        tokens = self.patch_embed(torch.cat((observation, candidate), dim=1))
        tokens = tokens.flatten(2).transpose(1, 2) + self.position
        condition = self.condition.expand(tokens.shape[0], -1)
        for block in self.blocks:
            tokens = block(tokens, condition)
        patch_energies = self.energy_head(self.final_norm(tokens))
        return patch_energies.mean(dim=1)

    def forward(self, observation: Tensor, candidate: Tensor, q: Tensor) -> Tensor:
        components = self.component_energies(observation, candidate)
        descriptor = compositional_descriptor(q).to(
            device=components.device, dtype=components.dtype
        )
        mixing = self.descriptor_mixer * self.descriptor_mask
        weights = descriptor @ mixing.transpose(0, 1)
        normalizer = weights.abs().sum(dim=1).clamp_min(1.0)
        return (components * weights).sum(dim=1) / normalizer

    def descent_step(
        self,
        observation: Tensor,
        candidate: Tensor,
        q: Tensor,
        *,
        step_size: float,
        create_graph: bool,
        clamp: tuple[float, float] | None = (-1.0, 1.0),
    ) -> EnergyStep:
        candidate = candidate.requires_grad_(True)
        energy = self(observation, candidate, q)
        gradient = torch.autograd.grad(
            energy.sum(), candidate, create_graph=create_graph
        )[0]
        updated = candidate - float(step_size) * gradient
        if clamp is not None:
            updated = updated.clamp(*clamp)
        return EnergyStep(updated, energy, gradient)


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
