"""Minimal learned context-token adapter for Lowlight severity conditioning."""
from __future__ import annotations

import torch
from torch import Tensor, nn


class LowLightLevelContext(nn.Module):
    """Append one learned FLUX text-context token for a Lowlight level."""

    def __init__(self, context_dim: int = 7680, levels: int = 5, init_std: float = 0.02) -> None:
        super().__init__()
        if context_dim <= 0 or levels <= 0 or init_std <= 0:
            raise ValueError("context_dim, levels, and init_std must be positive")
        self.context_dim = context_dim
        self.levels = levels
        self.embedding = nn.Embedding(levels, context_dim)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=init_std)

    def forward(
        self,
        context: Tensor,
        context_ids: Tensor,
        level: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        if context.ndim != 3 or context.shape[-1] != self.context_dim:
            raise ValueError(
                f"context must have shape [batch, tokens, {self.context_dim}]: {context.shape}"
            )
        if context_ids.ndim != 3 or context_ids.shape[:2] != context.shape[:2] or context_ids.shape[-1] != 4:
            raise ValueError("context_ids must have shape [batch, tokens, 4]")
        if level is None:
            return context, context_ids
        if level.ndim != 1 or level.shape[0] != context.shape[0]:
            raise ValueError("level must have shape [batch]")
        if level.dtype == torch.bool or torch.is_floating_point(level):
            raise ValueError("level must use an integer tensor dtype")
        if not ((level >= 1) & (level <= self.levels)).all():
            raise ValueError(f"levels must be in [1, {self.levels}]")

        token = self.embedding(level.to(device=self.embedding.weight.device) - 1)
        token = token.to(device=context.device, dtype=context.dtype).unsqueeze(1)
        token_ids = torch.zeros(
            context_ids.shape[0],
            1,
            4,
            device=context_ids.device,
            dtype=context_ids.dtype,
        )
        token_ids[..., 3] = context_ids[..., 3].amax(dim=1, keepdim=True) + 1
        return torch.cat([context, token], dim=1), torch.cat([context_ids, token_ids], dim=1)
