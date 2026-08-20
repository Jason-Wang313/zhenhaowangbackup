"""Canonical-target scoring helpers for post-failure Qwen diagnostics."""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F


CANONICAL_LEVELS = (1, 2, 3, 4, 5)


def mean_shifted_target_token_nll(
    logits: torch.Tensor, labels: torch.Tensor
) -> tuple[list[float], list[int]]:
    """Return mean causal NLL over each row's unmasked assistant tokens.

    ``labels`` follows the supervised-Qwen contract: prompt and padding tokens
    are ``-100`` and assistant target tokens are unmasked. Causal shifting is
    applied exactly as in the model's language-model loss.
    """

    if logits.ndim != 3 or labels.ndim != 2:
        raise ValueError("candidate logits/labels have bad rank")
    if logits.shape[:2] != labels.shape or logits.shape[1] < 2:
        raise ValueError("candidate logits/labels have incompatible shapes")
    if logits.shape[2] <= 1:
        raise ValueError("candidate logits have no vocabulary dimension")

    shifted_logits = logits[:, :-1, :].float().contiguous()
    shifted_labels = labels[:, 1:].contiguous()
    mask = shifted_labels.ne(-100)
    counts = mask.sum(dim=1)
    if torch.any(counts <= 0):
        raise ValueError("candidate row has no scored assistant token")
    token_nll = F.cross_entropy(
        shifted_logits.reshape(-1, shifted_logits.shape[-1]),
        shifted_labels.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).reshape_as(shifted_labels)
    totals = (token_nll * mask).sum(dim=1)
    means = totals / counts
    if not torch.isfinite(means).all():
        raise ValueError("candidate score is non-finite")
    return [float(value) for value in means.detach().cpu()], [
        int(value) for value in counts.detach().cpu()
    ]


def select_minimum_mean_nll(
    scores: Mapping[int | str, float],
    *,
    levels: Sequence[int] = CANONICAL_LEVELS,
) -> tuple[int, float, float | None]:
    """Select the lowest-NLL canonical level with a lower-level tie-break.

    Returns ``(selected_level, selected_mean_nll, runner_up_margin)``. The
    margin is runner-up NLL minus selected NLL and is ``None`` only when a
    single level is supplied.
    """

    normalized_levels = tuple(levels)
    if (
        not normalized_levels
        or len(set(normalized_levels)) != len(normalized_levels)
        or any(
            isinstance(level, bool) or not isinstance(level, int)
            for level in normalized_levels
        )
    ):
        raise ValueError("candidate levels must be unique integers")
    normalized: list[tuple[float, int]] = []
    for level in normalized_levels:
        if level in scores:
            raw = scores[level]
        elif str(level) in scores:
            raw = scores[str(level)]
        else:
            raise ValueError(f"candidate score is missing L{level}")
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise ValueError(f"candidate score is not numeric for L{level}")
        value = float(raw)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"candidate score is invalid for L{level}")
        normalized.append((value, level))
    normalized.sort(key=lambda item: (item[0], item[1]))
    selected_nll, selected_level = normalized[0]
    margin = None if len(normalized) == 1 else normalized[1][0] - selected_nll
    return selected_level, selected_nll, margin
