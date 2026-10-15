"""Leakage-safe helpers for the FoundIR singleton Qwen routing gate."""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from typing import Mapping, Sequence


CLASS_BY_CATEGORY = {
    "01Blur": "blur",
    "05Noise": "noise",
    "08Haze": "haze",
    "10Rain": "rain",
    "14Lowlight": "low_light",
}
CLASSES = tuple(CLASS_BY_CATEGORY.values())
SPLITS = ("fit", "calibration", "holdout")


def stable_digest(*parts: object) -> str:
    return hashlib.sha256(":".join(map(str, parts)).encode()).hexdigest()


def assigned_split(source_group: str, *, seed: int) -> str:
    """Assign a source globally so it can never leak between head splits."""

    bucket = int(stable_digest(seed, "source_split", source_group), 16) % 100
    if bucket < 70:
        return "fit"
    if bucket < 85:
        return "calibration"
    return "holdout"


def build_balanced_partition(
    rows: Sequence[Mapping[str, str]],
    *,
    seed: int,
    records_per_class: Mapping[str, int],
) -> list[dict[str, str]]:
    """Select one image per (class, source) with equal class quotas."""

    if set(records_per_class) != set(SPLITS):
        raise ValueError("records_per_class must specify fit/calibration/holdout")
    if any(int(records_per_class[name]) <= 0 for name in SPLITS):
        raise ValueError("partition quotas must be positive")
    candidates: dict[tuple[str, str], dict[str, dict[str, str]]] = defaultdict(dict)
    for raw in rows:
        if raw.get("split") != "train":
            continue
        category = raw.get("category")
        if category not in CLASS_BY_CATEGORY or raw.get("composition_size") != "1":
            continue
        source = str(raw["source_group"])
        label = CLASS_BY_CATEGORY[str(category)]
        split = assigned_split(source, seed=seed)
        row = dict(raw)
        previous = candidates[(label, split)].get(source)
        if previous is None or stable_digest(seed, "record", row["record_id"]) < stable_digest(
            seed, "record", previous["record_id"]
        ):
            candidates[(label, split)][source] = row

    selected: list[dict[str, str]] = []
    for label in CLASSES:
        for split in SPLITS:
            quota = int(records_per_class[split])
            pool = sorted(
                candidates[(label, split)].values(),
                key=lambda row: stable_digest(seed, "select", label, row["source_group"]),
            )
            if len(pool) < quota:
                raise RuntimeError(
                    f"insufficient {label}/{split} sources: {len(pool)} < {quota}"
                )
            for row in pool[:quota]:
                selected.append(
                    {
                        "record_id": row["record_id"],
                        "head_split": split,
                        "label": label,
                        "source_group": row["source_group"],
                        "lq_path": row["lq_path"],
                        "lq_sha256": row["lq_sha256"],
                        "width": row["width"],
                        "height": row["height"],
                    }
                )
    validate_partition(selected, records_per_class=records_per_class)
    return sorted(selected, key=lambda row: (row["head_split"], row["label"], row["record_id"]))


def validate_partition(
    rows: Sequence[Mapping[str, str]], *, records_per_class: Mapping[str, int]
) -> None:
    identities = [str(row["record_id"]) for row in rows]
    if len(identities) != len(set(identities)):
        raise RuntimeError("a record was selected more than once")
    counts = Counter((str(row["head_split"]), str(row["label"])) for row in rows)
    for split in SPLITS:
        for label in CLASSES:
            if counts[(split, label)] != int(records_per_class[split]):
                raise RuntimeError(f"unbalanced partition at {split}/{label}")
    source_splits: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        source_splits[str(row["source_group"])].add(str(row["head_split"]))
    leaked = [source for source, splits in source_splits.items() if len(splits) != 1]
    if leaked:
        raise RuntimeError(f"source leakage across head splits: {leaked[:3]}")


def routing_metrics(
    probabilities: Sequence[Sequence[float]],
    labels: Sequence[int],
    *,
    threshold: float,
) -> dict[str, object]:
    """Measure accepted-correct, wrong routing, and safe refusal rates."""

    if len(probabilities) != len(labels) or not labels:
        raise ValueError("probabilities and labels must be nonempty and aligned")
    totals = Counter()
    per_class = {name: Counter() for name in CLASSES}
    for values, truth in zip(probabilities, labels, strict=True):
        if len(values) != len(CLASSES) or truth not in range(len(CLASSES)):
            raise ValueError("bad five-class prediction row")
        prediction = max(range(len(CLASSES)), key=lambda index: float(values[index]))
        accepted = float(values[prediction]) >= threshold
        outcome = "refusal" if not accepted else "correct" if prediction == truth else "wrong"
        totals[outcome] += 1
        per_class[CLASSES[truth]][outcome] += 1
    count = len(labels)

    def rates(counter: Counter[str], denominator: int) -> dict[str, float | int]:
        accepted = counter["correct"] + counter["wrong"]
        if denominator == 0:
            return {
                "records": 0,
                "correct": 0,
                "wrong": 0,
                "refusal": 0,
                "correct_rate": 0.0,
                "wrong_rate": 0.0,
                "refusal_rate": 0.0,
                "coverage": 0.0,
                "accepted_accuracy": 1.0,
            }
        return {
            "records": denominator,
            "correct": counter["correct"],
            "wrong": counter["wrong"],
            "refusal": counter["refusal"],
            "correct_rate": counter["correct"] / denominator,
            "wrong_rate": counter["wrong"] / denominator,
            "refusal_rate": counter["refusal"] / denominator,
            "coverage": accepted / denominator,
            "accepted_accuracy": counter["correct"] / accepted if accepted else 1.0,
        }

    return {
        "threshold": threshold,
        "overall": rates(totals, count),
        "per_class": {
            name: rates(per_class[name], sum(per_class[name].values())) for name in CLASSES
        },
    }


def image_embedding_sequence(output: object) -> tuple[object, ...]:
    """Normalize the Transformers v4 tuple and v5 ModelOutput APIs."""

    import torch

    pooled = getattr(output, "pooler_output", output)
    if isinstance(pooled, torch.Tensor):
        return tuple(pooled)
    if isinstance(pooled, (tuple, list)) and all(
        isinstance(value, torch.Tensor) for value in pooled
    ):
        return tuple(pooled)
    raise RuntimeError("Qwen visual feature return type changed")
