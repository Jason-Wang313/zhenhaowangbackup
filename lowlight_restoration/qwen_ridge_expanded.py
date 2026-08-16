"""Leakage-safe source partition helpers for the expanded Qwen ridge head."""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from typing import Mapping, Sequence

from lowlight_restoration.data import PairRecord


def source_dataset_map(manifest_rows: Sequence[Mapping[str, str]]) -> dict[str, str]:
    """Return one verified dataset name for every source in a manifest."""

    result: dict[str, str] = {}
    for row in manifest_rows:
        source_id = row.get("source_id", "")
        dataset = row.get("source_dataset", "")
        if not source_id or not dataset:
            raise ValueError("manifest rows require source_id and source_dataset")
        previous = result.setdefault(source_id, dataset)
        if previous != dataset:
            raise ValueError(f"source dataset changed within manifest: {source_id}")
    return result


def _validate_sequences(
    sequences: Sequence[Sequence[PairRecord]], source_datasets: Mapping[str, str]
) -> None:
    seen: set[str] = set()
    for sequence in sequences:
        if len(sequence) != 5 or [record.level_index for record in sequence] != list(
            range(1, 6)
        ):
            raise ValueError("every source sequence must contain ordered L1-L5 records")
        source_id = sequence[0].source_id
        if any(record.source_id != source_id for record in sequence):
            raise ValueError("a source sequence contains multiple source IDs")
        if source_id in seen:
            raise ValueError(f"duplicate source sequence: {source_id}")
        if source_id not in source_datasets:
            raise ValueError(f"source dataset is missing: {source_id}")
        seen.add(source_id)


def stratified_source_partition(
    sequences: Sequence[Sequence[PairRecord]],
    *,
    source_datasets: Mapping[str, str],
    selection_quotas: Mapping[str, int],
    seed: int,
) -> tuple[list[list[PairRecord]], list[list[PairRecord]]]:
    """Split complete sources with exact per-dataset selection quotas."""

    if seed < 0 or not selection_quotas:
        raise ValueError("partition seed and selection quotas must be valid")
    _validate_sequences(sequences, source_datasets)
    grouped: dict[str, list[list[PairRecord]]] = defaultdict(list)
    for sequence in sequences:
        grouped[source_datasets[sequence[0].source_id]].append(list(sequence))
    if set(grouped) != set(selection_quotas):
        raise ValueError("selection quotas do not match the realized source datasets")

    fit: list[list[PairRecord]] = []
    selection: list[list[PairRecord]] = []
    for dataset in sorted(grouped):
        quota = selection_quotas[dataset]
        if (
            isinstance(quota, bool)
            or not isinstance(quota, int)
            or not 0 < quota < len(grouped[dataset])
        ):
            raise ValueError(f"bad selection quota for {dataset}: {quota!r}")
        ordered = sorted(
            grouped[dataset],
            key=lambda sequence: hashlib.sha256(
                f"{seed}:{dataset}:{sequence[0].source_id}".encode()
            ).hexdigest(),
        )
        selection.extend(ordered[:quota])
        fit.extend(ordered[quota:])

    ordering = lambda sequence: hashlib.sha256(
        f"{seed}:output:{sequence[0].source_id}".encode()
    ).hexdigest()
    return sorted(fit, key=ordering), sorted(selection, key=ordering)


def stratified_unexposed_selection(
    sequences: Sequence[Sequence[PairRecord]],
    *,
    source_datasets: Mapping[str, str],
    excluded_source_ids: set[str],
    selection_quotas: Mapping[str, int],
    seed: int,
) -> list[list[PairRecord]]:
    """Select a deterministic validation subset after excluding exposed sources."""

    if seed < 0 or not excluded_source_ids or not selection_quotas:
        raise ValueError("validation seed, exclusions, and quotas are required")
    _validate_sequences(sequences, source_datasets)
    available: dict[str, list[list[PairRecord]]] = defaultdict(list)
    realized_excluded = 0
    for sequence in sequences:
        source_id = sequence[0].source_id
        if source_id in excluded_source_ids:
            realized_excluded += 1
            continue
        available[source_datasets[source_id]].append(list(sequence))
    if realized_excluded != len(excluded_source_ids):
        raise ValueError("some exposed validation sources were not found in validation")
    if set(available) != set(selection_quotas):
        raise ValueError("validation quotas do not match the available source datasets")

    selected: list[list[PairRecord]] = []
    for dataset in sorted(available):
        quota = selection_quotas[dataset]
        if (
            isinstance(quota, bool)
            or not isinstance(quota, int)
            or not 0 < quota <= len(available[dataset])
        ):
            raise ValueError(f"bad validation quota for {dataset}: {quota!r}")
        ordered = sorted(
            available[dataset],
            key=lambda sequence: hashlib.sha256(
                f"{seed}:{dataset}:{sequence[0].source_id}".encode()
            ).hexdigest(),
        )
        selected.extend(ordered[:quota])
    return sorted(
        selected,
        key=lambda sequence: hashlib.sha256(
            f"{seed}:output:{sequence[0].source_id}".encode()
        ).hexdigest(),
    )


def source_dataset_counts(
    sequences: Sequence[Sequence[PairRecord]], source_datasets: Mapping[str, str]
) -> dict[str, int]:
    """Count source sequences by their verified dataset."""

    _validate_sequences(sequences, source_datasets)
    return dict(
        sorted(Counter(source_datasets[x[0].source_id] for x in sequences).items())
    )
