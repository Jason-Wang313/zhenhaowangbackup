"""Small ordinal head and deterministic source selection for frozen VLM features."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Sequence

import torch
from torch import nn

from lowlight_restoration.data import PairRecord

LEVELS = tuple(range(1, 6))


class OrdinalSeverityHead(nn.Module):
    """CORAL-style linear head predicting the four ordered level boundaries."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        self.input_dim = input_dim
        self.boundaries = nn.Linear(input_dim, 4)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != self.input_dim:
            raise ValueError(
                f"expected [batch,{self.input_dim}] features, got {tuple(features.shape)}"
            )
        return self.boundaries(features)


def ordinal_targets(levels: torch.Tensor) -> torch.Tensor:
    """Encode L1-L5 as P(level > boundary) targets for boundaries 1-4."""

    if levels.ndim != 1 or levels.dtype not in (torch.int32, torch.int64):
        raise ValueError("levels must be a one-dimensional integer tensor")
    if levels.numel() == 0 or bool(((levels < 1) | (levels > 5)).any()):
        raise ValueError("levels must contain only integers 1 through 5")
    boundaries = torch.arange(1, 5, device=levels.device)
    return (levels[:, None] > boundaries[None, :]).float()


def decode_ordinal_logits(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 2 or logits.shape[1] != 4 or not bool(torch.isfinite(logits).all()):
        raise ValueError("ordinal logits must be finite with shape [batch,4]")
    return 1 + (logits > 0).sum(dim=1)


def monotonic_penalty(logits: torch.Tensor) -> torch.Tensor:
    """Penalize later-boundary confidence exceeding earlier-boundary confidence."""

    if logits.ndim != 2 or logits.shape[1] != 4:
        raise ValueError("ordinal logits must have shape [batch,4]")
    return torch.relu(logits[:, 1:] - logits[:, :-1]).mean()


def select_complete_source_sequences(
    records: Sequence[PairRecord], *, split: str, source_count: int, seed: int
) -> list[list[PairRecord]]:
    """Select deterministic sources having exactly one record at every level."""

    if source_count <= 0:
        raise ValueError("source_count must be positive")
    grouped: dict[str, list[PairRecord]] = defaultdict(list)
    for record in records:
        if record.split == split:
            grouped[record.source_id].append(record)
    complete: list[tuple[str, list[PairRecord]]] = []
    for source_id, source_records in grouped.items():
        ordered = sorted(source_records, key=lambda record: record.level_index)
        if [record.level_index for record in ordered] == list(LEVELS):
            complete.append((source_id, ordered))
    complete.sort(
        key=lambda item: hashlib.sha256(f"{seed}:{item[0]}".encode()).hexdigest()
    )
    if len(complete) < source_count:
        raise ValueError(
            f"requested {source_count} complete sources but only {len(complete)} are available"
        )
    return [records_for_source for _, records_for_source in complete[:source_count]]
