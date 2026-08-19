"""Pure scheduling helpers for distributed supervised Qwen-VL training."""

from __future__ import annotations

import math
from collections import OrderedDict
from typing import Sequence

from .data import PairRecord


def complete_source_groups(
    records: Sequence[PairRecord],
) -> list[list[PairRecord]]:
    """Return ordered L1--L5 groups and fail if the selection is incomplete."""

    grouped: OrderedDict[str, list[PairRecord]] = OrderedDict()
    for record in records:
        grouped.setdefault(record.source_id, []).append(record)
    if not grouped:
        raise ValueError("distributed selection is empty")
    result: list[list[PairRecord]] = []
    for source_id, source_records in grouped.items():
        by_level = {record.level_index: record for record in source_records}
        if len(by_level) != len(source_records):
            raise ValueError(f"duplicate level in source group: {source_id}")
        if set(by_level) != set(range(1, 6)):
            raise ValueError(f"incomplete L1--L5 source group: {source_id}")
        result.append([by_level[level] for level in range(1, 6)])
    return result


def partition_source_groups(
    records: Sequence[PairRecord], *, world_size: int, rank: int
) -> list[list[PairRecord]]:
    """Partition whole source groups evenly without splitting severity levels."""

    if world_size <= 0:
        raise ValueError("world_size must be positive")
    if rank < 0 or rank >= world_size:
        raise ValueError("rank is outside world_size")
    groups = complete_source_groups(records)
    if len(groups) % world_size:
        raise ValueError(
            f"{len(groups)} source groups are not divisible by world size {world_size}"
        )
    local = groups[rank::world_size]
    expected = len(groups) // world_size
    if len(local) != expected:
        raise RuntimeError("source-group partition is imbalanced")
    return local


def validate_full_batch_contract(
    *,
    sample_count: int,
    source_count: int,
    world_size: int,
    micro_batch_size: int,
    gradient_accumulation_steps: int,
    global_batch_size: int,
) -> None:
    """Validate one globally balanced L1--L5 pass per optimizer update."""

    if sample_count != source_count * 5:
        raise ValueError("sample count must equal five records per source")
    if source_count % world_size:
        raise ValueError("source count must be divisible by world size")
    if micro_batch_size != 5:
        raise ValueError("each microbatch must contain one complete L1--L5 source")
    expected_accumulation = source_count // world_size
    if gradient_accumulation_steps != expected_accumulation:
        raise ValueError(
            "gradient accumulation must cover every local source exactly once"
        )
    realized = micro_batch_size * gradient_accumulation_steps * world_size
    if global_batch_size != sample_count or realized != global_batch_size:
        raise ValueError("global batch must equal all selected records")


def cosine_learning_rate(
    *,
    step: int,
    total_steps: int,
    warmup_steps: int,
    base_learning_rate: float,
    minimum_learning_rate: float,
) -> float:
    """Return a closed-form warmup plus cosine-decay learning rate."""

    if total_steps <= 0 or step <= 0 or step > total_steps:
        raise ValueError("step must be inside the configured training run")
    if warmup_steps < 0 or warmup_steps >= total_steps:
        raise ValueError("warmup_steps must be in [0, total_steps)")
    if not 0.0 <= minimum_learning_rate <= base_learning_rate:
        raise ValueError("learning-rate bounds are invalid")
    if warmup_steps and step <= warmup_steps:
        return base_learning_rate * step / warmup_steps
    decay_steps = total_steps - warmup_steps
    progress = (step - warmup_steps) / decay_steps
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return minimum_learning_rate + (
        base_learning_rate - minimum_learning_rate
    ) * cosine


def merge_indexed_outputs(
    gathered: Sequence[Sequence[tuple[int, str]]], *, expected_count: int
) -> list[str]:
    """Merge rank-local generated outputs into authoritative selection order."""

    flattened = [item for rank_items in gathered for item in rank_items]
    indices = [index for index, _ in flattened]
    if len(flattened) != expected_count or sorted(indices) != list(
        range(expected_count)
    ):
        raise ValueError("distributed generation indices are incomplete or duplicated")
    return [output for _, output in sorted(flattened)]
