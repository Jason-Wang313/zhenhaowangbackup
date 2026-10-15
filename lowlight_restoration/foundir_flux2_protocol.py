"""Frozen sampling and evaluation protocol for five-type FoundIR restoration."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction
from typing import Mapping, Sequence
from pathlib import Path

from .foundir_five_type import FIVE_TYPE_CATEGORIES


SOW_STAGE2_THRESHOLDS: dict[str, dict[str, float]] = {
    "blur": {"psnr": 25.9, "lpips": 0.18},
    "noise": {"psnr": 38.9, "lpips": 0.10},
    "rain": {"psnr": 34.2, "lpips": 0.08},
    "haze": {"psnr": 22.1, "lpips": 0.19},
    "low_light": {"psnr": 18.9, "lpips": 0.28},
}

# The Stage-2 SOW reports each restoration task on its corresponding isolated
# FoundIR category.  Coupled categories remain useful diagnostics, but they
# must not influence SOW metrics or checkpoint selection.
SOW_ISOLATED_CATEGORIES: dict[str, str] = {
    "blur": "01Blur",
    "noise": "05Noise",
    "haze": "08Haze",
    "rain": "10Rain",
    "low_light": "14Lowlight",
}


def resolve_workspace_path(root: Path, value: str | Path) -> Path:
    """Resolve a CLI/config path while refusing to escape the workspace."""

    path = Path(value)
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    path.relative_to(root.resolve())
    return path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_pixel_refinement_vae_evidence(
    root: Path,
    config: Mapping[str, object],
    manifest_path: Path,
) -> dict[str, object]:
    """Validate the frozen v2 VAE result that admitted pixel refinement.

    The VAE, target-selection contract, and manifest are unchanged in v3, so
    the measured v2 ceiling is reused by exact identity instead of spending
    eight GPUs to reproduce the same 44 round trips.
    """

    refinement = config.get("pixel_refinement")
    if not isinstance(refinement, Mapping) or refinement.get("enabled") is not True:
        raise RuntimeError("pixel refinement is not enabled")
    evidence = refinement.get("source_vae_ceiling")
    if not isinstance(evidence, Mapping):
        raise RuntimeError("pixel refinement has no frozen source VAE evidence")
    source_config_path = resolve_workspace_path(root, str(evidence["config"]))
    result_path = resolve_workspace_path(root, str(evidence["result"]))
    if file_sha256(source_config_path) != evidence.get("config_sha256"):
        raise RuntimeError("source VAE config identity changed")
    if file_sha256(result_path) != evidence.get("result_sha256"):
        raise RuntimeError("source VAE result identity changed")
    source_config = json.loads(source_config_path.read_text())
    result = json.loads(result_path.read_text())
    if (
        result.get("status") != "pixel_refinement_required"
        or result.get("config_sha256") != evidence.get("config_sha256")
        or result.get("manifest_sha256") != file_sha256(manifest_path)
        or result.get("protected_foundir_test_accessed") is not False
        or not result.get("failures")
    ):
        raise RuntimeError("source VAE result does not admit pixel refinement")
    if (
        source_config.get("vae_ceiling") != config.get("vae_ceiling")
        or source_config.get("scope", {}).get("degradations")
        != config.get("scope", {}).get("degradations")
        or source_config.get("scope", {}).get("exclude_jpeg") is not True
        or source_config.get("model", {}).get("name") != config.get("model", {}).get("name")
        or source_config.get("model", {}).get("autoencoder_weights")
        != config.get("model", {}).get("autoencoder_weights")
        or source_config.get("model", {}).get("autoencoder_weights_sha256")
        != config.get("model", {}).get("autoencoder_weights_sha256")
        or source_config.get("data", {}).get("split_seed")
        != config.get("data", {}).get("split_seed")
    ):
        raise RuntimeError("v3 changed a VAE-ceiling input relative to its frozen evidence")
    expected_records = len(FIVE_TYPE_CATEGORIES) * int(config["vae_ceiling"]["records_per_category"])
    if (
        int(result.get("records", -1)) != expected_records
        or set(result.get("categories", {})) != set(FIVE_TYPE_CATEGORIES)
    ):
        raise RuntimeError("source VAE evidence is incomplete")
    return result


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


def balanced_group_fold_assignment(
    rows: Sequence[Mapping[str, str]],
    *,
    split: str,
    folds: int,
    seed: int,
) -> dict[str, int]:
    """Assign complete source groups while balancing category diversity first."""

    if split not in {"validation", "internal_test"}:
        raise ValueError("evaluation fold assignment needs an evaluation split")
    if folds < 2:
        raise ValueError("evaluation fold assignment needs at least two folds")
    group_split: dict[str, str] = {}
    group_category_records: dict[str, dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    category_records: dict[str, int] = defaultdict(int)
    for row in rows:
        group = row["source_group"]
        if not group:
            raise ValueError("evaluation record has an empty source group")
        row_split = row["split"]
        previous = group_split.setdefault(group, row_split)
        if previous != row_split:
            raise RuntimeError("source group crosses evaluation splits")
        if row_split != split:
            continue
        category = row["category"]
        if category not in FIVE_TYPE_CATEGORIES:
            raise RuntimeError(f"unknown FoundIR evaluation category: {category}")
        group_category_records[group][category] += 1
        category_records[category] += 1
    category_groups: dict[str, int] = defaultdict(int)
    for counts in group_category_records.values():
        for category in counts:
            category_groups[category] += 1

    def group_priority(item: tuple[str, dict[str, int]]) -> tuple[object, ...]:
        group, counts = item
        scarcity = max(Fraction(1, category_groups[category]) for category in counts)
        return (
            -scarcity,
            -max(counts.values()),
            -sum(counts.values()),
            stable_digest(seed, split, "group-order", group),
        )

    fold_category_groups = [defaultdict(int) for _ in range(folds)]
    fold_category_records = [defaultdict(int) for _ in range(folds)]
    fold_group_counts = [0 for _ in range(folds)]
    assignment: dict[str, int] = {}
    for group, counts in sorted(group_category_records.items(), key=group_priority):
        def fold_score(fold: int) -> tuple[object, ...]:
            group_ratios = [
                Fraction(
                    fold_category_groups[fold][category] + 1,
                    category_groups[category],
                )
                for category in counts
            ]
            record_ratios = [
                Fraction(
                    fold_category_records[fold][category] + count,
                    category_records[category],
                )
                for category, count in counts.items()
            ]
            return (
                max(group_ratios),
                sum(group_ratios),
                max(record_ratios),
                sum(record_ratios),
                fold_group_counts[fold],
                stable_digest(seed, split, "fold-tie", group, fold),
            )

        fold = min(range(folds), key=fold_score)
        assignment[group] = fold
        fold_group_counts[fold] += 1
        for category, count in counts.items():
            fold_category_groups[fold][category] += 1
            fold_category_records[fold][category] += count
    return assignment


def balanced_evaluation_subset(
    rows: Sequence[Mapping[str, str]],
    *,
    split: str,
    records_per_category_per_fold: int,
    folds: int,
    seed: int,
) -> list[dict[str, str]]:
    """Freeze a category-balanced subset with groups confined to one fold.

    Selection maximizes source-group diversity before reusing any group.  A
    sparse bucket is then filled by deterministic round-robin passes over its
    groups, so the configured record quota remains exact without pretending
    that correlated records are independent groups.
    """

    if split not in {"validation", "internal_test"}:
        raise ValueError("evaluation subset must use validation or internal_test")
    if records_per_category_per_fold <= 0:
        raise ValueError("evaluation quota must be positive")
    group_folds = balanced_group_fold_assignment(
        rows,
        split=split,
        folds=folds,
        seed=seed,
    )
    buckets: dict[tuple[int, str], list[dict[str, str]]] = defaultdict(list)
    for original in rows:
        if original["split"] != split:
            continue
        row = dict(original)
        fold = group_folds[row["source_group"]]
        buckets[(fold, row["category"])].append(row)
    selected: list[dict[str, str]] = []
    for fold in range(folds):
        for category in sorted(FIVE_TYPE_CATEGORIES):
            grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
            for row in buckets[(fold, category)]:
                grouped[row["source_group"]].append(row)
            group_order = sorted(
                grouped,
                key=lambda group: stable_digest(
                    seed, split, fold, category, "group", group
                ),
            )
            for group, group_rows in grouped.items():
                group_rows.sort(
                    key=lambda row: stable_digest(
                        seed,
                        split,
                        fold,
                        category,
                        "record",
                        group,
                        row["record_id"],
                    )
                )
            available_records = sum(len(group_rows) for group_rows in grouped.values())
            if available_records < records_per_category_per_fold:
                raise RuntimeError(
                    f"insufficient {split} records for fold={fold} category={category}: "
                    f"{available_records}"
                )
            chosen: list[dict[str, str]] = []
            round_index = 0
            while len(chosen) < records_per_category_per_fold:
                added_this_round = False
                for group in group_order:
                    group_rows = grouped[group]
                    if round_index >= len(group_rows):
                        continue
                    chosen.append(
                        {**group_rows[round_index], "evaluation_fold": str(fold)}
                    )
                    added_this_round = True
                    if len(chosen) == records_per_category_per_fold:
                        break
                if not added_this_round:
                    break
                round_index += 1
            if len(chosen) != records_per_category_per_fold:
                raise RuntimeError(
                    f"failed to fill {split} quota for fold={fold} category={category}: "
                    f"{len(chosen)}"
                )
            selected.extend(chosen)
    record_ids = [row["record_id"] for row in selected]
    if len(record_ids) != len(set(record_ids)):
        raise RuntimeError("evaluation subset duplicates record IDs")
    groups_by_fold: dict[str, int] = {}
    for row in selected:
        previous = groups_by_fold.setdefault(row["source_group"], int(row["evaluation_fold"]))
        if previous != int(row["evaluation_fold"]):
            raise RuntimeError("source group crosses evaluation folds")
    return selected


def aggregate_task_metrics(rows: Sequence[Mapping[str, object]]) -> dict[str, dict[str, float]]:
    """Aggregate SOW task metrics from the five isolated FoundIR categories."""

    isolated_task_by_category = {
        category: task for task, category in SOW_ISOLATED_CATEGORIES.items()
    }
    values: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        category = str(row["category"])
        if category not in FIVE_TYPE_CATEGORIES:
            raise ValueError(f"unknown FoundIR category in metrics: {category}")
        task = isolated_task_by_category.get(category)
        if task is not None:
            values[task].append(row)
    if set(values) != set(SOW_ISOLATED_CATEGORIES):
        missing = sorted(set(SOW_ISOLATED_CATEGORIES) - set(values))
        raise RuntimeError(
            "metric rows do not cover all five isolated SOW restoration tasks; "
            f"missing={missing}"
        )
    return {
        task: {
            "records": float(len(task_rows)),
            "psnr": sum(float(row["psnr"]) for row in task_rows) / len(task_rows),
            "lpips": sum(float(row["lpips"]) for row in task_rows) / len(task_rows),
        }
        for task, task_rows in sorted(values.items())
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
