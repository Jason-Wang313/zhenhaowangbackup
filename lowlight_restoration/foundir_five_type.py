"""Contracts for the advisor-approved five-type FoundIR restoration subset."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEGRADATION_ORDER = ("blur", "noise", "rain", "haze", "low_light")

# Only categories whose official folder name is exactly expressible by the
# frozen five-slot Qwen schema are admitted.  Raindrop and NightRain remain
# excluded rather than being silently relabelled as Rain / Rain+Lowlight.
FIVE_TYPE_CATEGORIES: dict[str, tuple[str, ...]] = {
    "01Blur": ("blur",),
    "02Blur_Noise": ("blur", "noise"),
    "05Noise": ("noise",),
    "08Haze": ("haze",),
    "09Lowlight_Haze": ("haze", "low_light"),
    "10Rain": ("rain",),
    "13Rain_Haze": ("rain", "haze"),
    "14Lowlight": ("low_light",),
    "15Lowlight_Blur": ("blur", "low_light"),
    "16Lowlight_Noise": ("noise", "low_light"),
    "18Lowlight_Blur_Noise": ("blur", "noise", "low_light"),
}

EXCLUDED_JPEG_CATEGORIES = frozenset(
    {
        "03Blur_JPEG",
        "04Blur_Noise_JPEG",
        "06JPEG",
        "07Noise_JPEG",
        "17Lowlight_JPEG",
        "19Lowlight_Blur_JPEG",
        "20Lowlight_Noise_JPEG",
    }
)
EXCLUDED_NONCANONICAL_CATEGORIES = frozenset({"11Raindrop", "12NightRain"})


@dataclass(frozen=True)
class FoundIRMetaPair:
    category: str
    degradations: tuple[str, ...]
    gt_relative: str
    lq_relative: str

    @property
    def composition_size(self) -> int:
        return len(self.degradations)


def parse_meta_line(line: str) -> tuple[str, str]:
    parts = line.strip().split(", ")
    if len(parts) != 2 or not all(parts):
        raise ValueError(f"invalid FoundIR metadata row: {line!r}")
    gt_relative, lq_relative = parts
    gt_parts = Path(gt_relative).parts
    lq_parts = Path(lq_relative).parts
    if (
        len(gt_parts) != 3
        or len(lq_parts) != 3
        or gt_parts[0] != lq_parts[0]
        or gt_parts[1] != "GT"
        or lq_parts[1] != "LQ"
    ):
        raise ValueError(f"unexpected FoundIR pair layout: {line!r}")
    if Path(gt_relative).is_absolute() or Path(lq_relative).is_absolute():
        raise ValueError("FoundIR metadata paths must be relative")
    return gt_relative, lq_relative


def load_five_type_metadata(path: Path) -> tuple[list[FoundIRMetaPair], dict[str, int]]:
    records: list[FoundIRMetaPair] = []
    counts: dict[str, int] = {}
    seen: set[tuple[str, str]] = set()
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        gt_relative, lq_relative = parse_meta_line(line)
        category = Path(gt_relative).parts[0]
        counts[category] = counts.get(category, 0) + 1
        if category not in FIVE_TYPE_CATEGORIES:
            continue
        identity = (gt_relative, lq_relative)
        if identity in seen:
            raise ValueError(f"duplicate FoundIR pair at line {line_number}: {identity}")
        seen.add(identity)
        records.append(
            FoundIRMetaPair(
                category=category,
                degradations=FIVE_TYPE_CATEGORIES[category],
                gt_relative=gt_relative,
                lq_relative=lq_relative,
            )
        )
    return records, counts


def canonical_restoration_text(levels: dict[str, int]) -> str:
    if set(levels) != set(DEGRADATION_ORDER):
        raise ValueError("canonical condition requires exactly five degradation slots")
    descriptions: list[str] = []
    labels = {
        "blur": "Blur",
        "noise": "Noise",
        "rain": "Rain",
        "haze": "Haze",
        "low_light": "Low-light",
    }
    for degradation in DEGRADATION_ORDER:
        level = int(levels[degradation])
        if level not in range(1, 6):
            raise ValueError(f"invalid {degradation} level: {level}")
        value = "absent" if level == 1 else f"L{level}"
        descriptions.append(f"{labels[degradation]}: {value}")
    return (
        "Restore this image faithfully. "
        + "; ".join(descriptions)
        + ". Preserve content, geometry, identity, texture, and natural color."
    )


def stable_group_split(
    group_identity: str,
    *,
    seed: int,
    train_fraction: float = 0.96,
    validation_fraction: float = 0.02,
) -> str:
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1:
        raise ValueError("split fractions must be in (0, 1)")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train and validation fractions must leave internal-test data")
    digest = hashlib.sha256(f"{seed}:{group_identity}".encode()).digest()
    unit = int.from_bytes(digest[:8], "big") / 2**64
    if unit < train_fraction:
        return "train"
    if unit < train_fraction + validation_fraction:
        return "validation"
    return "internal_test"


def category_counts(records: Iterable[FoundIRMetaPair]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        counts[record.category] = counts.get(record.category, 0) + 1
    return dict(sorted(counts.items()))


class DisjointSet:
    """Small deterministic union-find used for exact/perceptual GT grouping."""

    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, value: int) -> int:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def hamming64(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def perceptual_components(hashes: list[int], *, maximum_distance: int = 4) -> list[int]:
    """Group 64-bit hashes without an O(N^2) all-pairs scan.

    Splitting a 64-bit hash into ``maximum_distance + 1`` chunks is exact for
    this search: two hashes within the distance threshold must share at least
    one complete chunk (pigeonhole principle).  Candidate pairs are verified
    with the full Hamming distance before union.
    """

    if not 0 <= maximum_distance <= 8:
        raise ValueError("maximum_distance must be between 0 and 8")
    if any(value < 0 or value >= 2**64 for value in hashes):
        raise ValueError("perceptual hashes must be unsigned 64-bit integers")
    if not hashes:
        return []
    chunks = maximum_distance + 1
    base, remainder = divmod(64, chunks)
    widths = [base + (1 if index < remainder else 0) for index in range(chunks)]
    offsets: list[int] = []
    cursor = 0
    for width in widths:
        offsets.append(cursor)
        cursor += width
    buckets: list[dict[int, list[int]]] = [{} for _ in range(chunks)]
    groups = DisjointSet(len(hashes))
    for index, value in enumerate(hashes):
        candidates: set[int] = set()
        keys: list[int] = []
        for chunk, (offset, width) in enumerate(zip(offsets, widths, strict=True)):
            key = (value >> offset) & ((1 << width) - 1)
            keys.append(key)
            candidates.update(buckets[chunk].get(key, ()))
        for other in candidates:
            if hamming64(value, hashes[other]) <= maximum_distance:
                groups.union(index, other)
        for chunk, key in enumerate(keys):
            buckets[chunk].setdefault(key, []).append(index)
    return [groups.find(index) for index in range(len(hashes))]
