"""Frozen sampling and evaluation protocol for five-type FoundIR restoration."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass
from typing import Mapping, Sequence
from pathlib import Path

from .foundir_five_type import DEGRADATION_ORDER, FIVE_TYPE_CATEGORIES


SOW_STAGE2_THRESHOLDS: dict[str, dict[str, float]] = {
    "blur": {"psnr": 25.9, "lpips": 0.18},
    "noise": {"psnr": 38.9, "lpips": 0.10},
    "rain": {"psnr": 34.2, "lpips": 0.08},
    "haze": {"psnr": 22.1, "lpips": 0.19},
    "low_light": {"psnr": 18.9, "lpips": 0.28},
}


def resolve_workspace_path(root: Path, value: str | Path) -> Path:
    """Resolve a CLI/config path while refusing to escape the workspace."""

    path = Path(value)
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    path.relative_to(root.resolve())
    return path


def evaluation_artifact_stem(record_id: str) -> str:
    """Return a filesystem-safe, collision-resistant record artifact name."""

    safe = "".join(character if character.isalnum() or character in "-_" else "_" for character in record_id)
    safe = safe[:96] or "record"
    return f"{hashlib.sha256(record_id.encode()).hexdigest()[:16]}_{safe}"


def stable_digest(seed: int, *parts: object) -> str:
    return hashlib.sha256(
        (str(seed) + ":" + ":".join(map(str, parts))).encode()
    ).hexdigest()


@dataclass(frozen=True)
class CurriculumPools:
    by_composition_category: dict[int, dict[str, tuple[int, ...]]]

    def validate_stage(self, maximum_composition_size: int) -> None:
        if maximum_composition_size not in (1, 2, 3):
            raise ValueError("curriculum stage must end at single, double, or triple")
        for composition_size in range(1, maximum_composition_size + 1):
            categories = self.by_composition_category.get(composition_size, {})
            if not categories or any(not values for values in categories.values()):
                raise RuntimeError(f"empty curriculum pool for composition {composition_size}")


def build_curriculum_pools(rows: Sequence[Mapping[str, str]]) -> CurriculumPools:
    pools: dict[int, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for index, row in enumerate(rows):
        if row["split"] != "train":
            continue
        composition = int(row["composition_size"])
        category = str(row["category"])
        expected = FIVE_TYPE_CATEGORIES.get(category)
        if expected is None or len(expected) != composition:
            raise RuntimeError(f"condition row has invalid category/composition: {category}")
        pools[composition][category].append(index)
    frozen = {
        composition: {
            category: tuple(indices)
            for category, indices in sorted(categories.items())
        }
        for composition, categories in sorted(pools.items())
    }
    result = CurriculumPools(frozen)
    result.validate_stage(3)
    return result


def curriculum_record_index(
    pools: CurriculumPools,
    *,
    maximum_composition_size: int,
    presentation_index: int,
    seed: int,
) -> int:
    """Return a deterministic composition/category-balanced training index."""

    if presentation_index < 0:
        raise ValueError("presentation index must be non-negative")
    pools.validate_stage(maximum_composition_size)
    composition = 1 + (presentation_index % maximum_composition_size)
    categories = pools.by_composition_category[composition]
    names = tuple(sorted(categories))
    category_cycle = presentation_index // maximum_composition_size
    category = names[category_cycle % len(names)]
    candidates = categories[category]
    offset = int(
        stable_digest(seed, maximum_composition_size, presentation_index, category)[:16],
        16,
    ) % len(candidates)
    return candidates[offset]


def fold_for_group(source_group: str, *, seed: int, folds: int) -> int:
    if not source_group or folds < 2:
        raise ValueError("fold assignment needs a source group and at least two folds")
    return int(stable_digest(seed, "fold", source_group)[:16], 16) % folds


def balanced_evaluation_subset(
    rows: Sequence[Mapping[str, str]],
    *,
    split: str,
    records_per_category_per_fold: int,
    folds: int,
    seed: int,
) -> list[dict[str, str]]:
    """Freeze a category-balanced, group-disjoint multi-fold subset."""

    if split not in {"validation", "internal_test"}:
        raise ValueError("evaluation subset must use validation or internal_test")
    if records_per_category_per_fold <= 0:
        raise ValueError("evaluation quota must be positive")
    buckets: dict[tuple[int, str], list[dict[str, str]]] = defaultdict(list)
    group_split: dict[str, str] = {}
    for original in rows:
        if original["split"] != split:
            continue
        row = dict(original)
        group = row["source_group"]
        previous = group_split.setdefault(group, split)
        if previous != split:
            raise RuntimeError("source group crosses evaluation splits")
        fold = fold_for_group(group, seed=seed, folds=folds)
        buckets[(fold, row["category"])].append(row)
    selected: list[dict[str, str]] = []
    for fold in range(folds):
        for category in sorted(FIVE_TYPE_CATEGORIES):
            candidates = sorted(
                buckets[(fold, category)],
                key=lambda row: stable_digest(seed, split, fold, category, row["record_id"]),
            )
            seen_groups: set[str] = set()
            chosen: list[dict[str, str]] = []
            for row in candidates:
                if row["source_group"] in seen_groups:
                    continue
                seen_groups.add(row["source_group"])
                chosen.append({**row, "evaluation_fold": str(fold)})
                if len(chosen) == records_per_category_per_fold:
                    break
            if len(chosen) != records_per_category_per_fold:
                raise RuntimeError(
                    f"insufficient {split} groups for fold={fold} category={category}: {len(chosen)}"
                )
            selected.extend(chosen)
    groups_by_fold: dict[str, int] = {}
    for row in selected:
        previous = groups_by_fold.setdefault(row["source_group"], int(row["evaluation_fold"]))
        if previous != int(row["evaluation_fold"]):
            raise RuntimeError("source group crosses evaluation folds")
    return selected


def aggregate_task_metrics(rows: Sequence[Mapping[str, object]]) -> dict[str, dict[str, float]]:
    """Aggregate restoration metrics by every degradation present in a category."""

    values: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        category = str(row["category"])
        if category not in FIVE_TYPE_CATEGORIES:
            raise ValueError(f"unknown FoundIR category in metrics: {category}")
        for degradation in FIVE_TYPE_CATEGORIES[category]:
            values[degradation].append(row)
    if set(values) != set(DEGRADATION_ORDER):
        raise RuntimeError("metric rows do not cover all five restoration tasks")
    return {
        degradation: {
            "records": float(len(task_rows)),
            "psnr": sum(float(row["psnr"]) for row in task_rows) / len(task_rows),
            "lpips": sum(float(row["lpips"]) for row in task_rows) / len(task_rows),
        }
        for degradation, task_rows in sorted(values.items())
    }


def worst_normalized_margin(task_metrics: Mapping[str, Mapping[str, float]]) -> float:
    """Score a checkpoint by its worst five-task PSNR/LPIPS acceptance margin."""

    if set(task_metrics) != set(SOW_STAGE2_THRESHOLDS):
        raise ValueError("checkpoint metrics must cover exactly five tasks")
    margins: list[float] = []
    for task, thresholds in SOW_STAGE2_THRESHOLDS.items():
        metrics = task_metrics[task]
        margins.append((float(metrics["psnr"]) - thresholds["psnr"]) / 1.0)
        margins.append((thresholds["lpips"] - float(metrics["lpips"])) / 0.01)
    return min(margins)


def select_checkpoint(candidates: Sequence[Mapping[str, object]]) -> Mapping[str, object]:
    if not candidates:
        raise ValueError("checkpoint selection requires candidates")
    scored = []
    for candidate in candidates:
        score = worst_normalized_margin(candidate["task_metrics"])
        scored.append((score, -int(candidate["step"]), candidate))
    return max(scored, key=lambda item: (item[0], item[1]))[2]
