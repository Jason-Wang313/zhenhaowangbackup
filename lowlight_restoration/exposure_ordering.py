"""Objective same-scene exposure-ordering data and metrics.

The project L1--L5 schema is useful for controlled synthetic data, but an
unrelated real photograph has no objective absolute project level.  Exposure
stacks provide a narrower fact that can be evaluated without inventing such a
label: within one fixed scene, a darker exposure must not be predicted as less
severe than a brighter exposure.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence


REQUIRED_CANDIDATE_FIELDS = {
    "path",
    "subset",
    "filename",
    "sha256",
    "bytes",
    "width",
    "height",
    "mode",
    "mean_luma",
    "component_id",
}
REQUIRED_FROZEN_FIELDS = {
    "stack_id",
    "image_id",
    "path",
    "sha256",
    "subset",
    "filename",
    "bytes",
    "width",
    "height",
    "mode",
    "mean_luma",
    "order_dark_to_bright",
    "darkness_rank",
    "ev_darker_than_brightest",
}
LEVELS = (1, 2, 3, 4, 5)


@dataclass(frozen=True)
class ExposureImage:
    stack_id: str
    path: str
    subset: str
    filename: str
    sha256: str
    byte_size: int
    width: int
    height: int
    mode: str
    mean_luma: float
    order_dark_to_bright: int
    darkness_rank: int
    ev_darker_than_brightest: float


@dataclass(frozen=True)
class ExposurePair:
    stack_id: str
    darker_sha256: str
    brighter_sha256: str
    ev_gap: float


def _canonical_exact_unique_rows(
    rows: Sequence[Mapping[str, str]], *, required_subset: str
) -> list[Mapping[str, str]]:
    """Deduplicate exact copies while preferring the requested source subset."""

    by_sha256: dict[str, list[Mapping[str, str]]] = {}
    for row in rows:
        by_sha256.setdefault(row["sha256"], []).append(row)
    selected: list[Mapping[str, str]] = []
    for copies in by_sha256.values():
        ordered = sorted(
            copies,
            key=lambda row: (
                row["subset"] != required_subset,
                row["path"],
            ),
        )
        selected.append(ordered[0])
    return selected


def build_exposure_stacks(
    rows: Iterable[Mapping[str, str]],
    *,
    required_subset: str = "MEF",
    excluded_components: Iterable[str] = (),
) -> dict[str, tuple[ExposureImage, ...]]:
    """Select exact-unique author-provided exposure stacks and order by luma.

    Only images from ``required_subset`` are retained.  This intentionally
    rejects cross-dataset perceptual neighbors as exposure evidence.  The
    input component grouping is assumed to have been frozen before inference.
    """

    excluded = set(excluded_components)
    grouped: dict[str, list[Mapping[str, str]]] = {}
    seen_paths: set[str] = set()
    for raw in rows:
        missing = REQUIRED_CANDIDATE_FIELDS - set(raw)
        if missing:
            raise ValueError(f"candidate row is missing fields: {sorted(missing)}")
        row = dict(raw)
        path = row["path"]
        if not path or path in seen_paths:
            raise ValueError(f"empty or duplicate candidate path: {path!r}")
        seen_paths.add(path)
        component = row["component_id"]
        if not component:
            raise ValueError(f"candidate has no component: {path}")
        grouped.setdefault(component, []).append(row)

    stacks: dict[str, tuple[ExposureImage, ...]] = {}
    used_sha256: set[str] = set()
    for component, component_rows in sorted(grouped.items()):
        if component in excluded:
            continue
        unique = _canonical_exact_unique_rows(
            component_rows, required_subset=required_subset
        )
        selected = [row for row in unique if row["subset"] == required_subset]
        if len(selected) < 2:
            continue

        parsed: list[tuple[float, Mapping[str, str]]] = []
        for row in selected:
            try:
                mean_luma = float(row["mean_luma"])
                byte_size = int(row["bytes"])
                width = int(row["width"])
                height = int(row["height"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"bad numeric candidate metadata: {row['path']}") from exc
            if (
                not math.isfinite(mean_luma)
                or mean_luma <= 0.0
                or mean_luma > 255.0
                or byte_size <= 0
                or width <= 0
                or height <= 0
            ):
                raise ValueError(f"invalid candidate metadata: {row['path']}")
            sha256 = row["sha256"]
            if len(sha256) != 64 or any(value not in "0123456789abcdef" for value in sha256):
                raise ValueError(f"invalid candidate SHA-256: {row['path']}")
            if sha256 in used_sha256:
                raise ValueError(f"candidate appears in multiple stacks: {row['path']}")
            parsed.append((mean_luma, row))

        parsed.sort(key=lambda item: (item[0], item[1]["sha256"]))
        lumas = [item[0] for item in parsed]
        if any(right <= left for left, right in zip(lumas, lumas[1:])):
            raise ValueError(f"stack has ambiguous mean-luma order: {component}")
        brightest = lumas[-1]
        size = len(parsed)
        images: list[ExposureImage] = []
        for index, (mean_luma, row) in enumerate(parsed):
            image = ExposureImage(
                stack_id=component,
                path=row["path"],
                subset=row["subset"],
                filename=row["filename"],
                sha256=row["sha256"],
                byte_size=int(row["bytes"]),
                width=int(row["width"]),
                height=int(row["height"]),
                mode=row["mode"],
                mean_luma=mean_luma,
                order_dark_to_bright=index,
                darkness_rank=size - index,
                ev_darker_than_brightest=math.log2(brightest / mean_luma),
            )
            images.append(image)
            used_sha256.add(image.sha256)
        stacks[component] = tuple(images)

    if not stacks:
        raise ValueError("no eligible exposure stacks")
    return stacks


def exposure_pairs(
    stack: Sequence[ExposureImage], *, minimum_ev_gap: float = 0.0
) -> tuple[ExposurePair, ...]:
    if not math.isfinite(minimum_ev_gap) or minimum_ev_gap < 0.0:
        raise ValueError("minimum EV gap must be finite and non-negative")
    if len(stack) < 2:
        raise ValueError("an exposure stack needs at least two images")
    pairs: list[ExposurePair] = []
    for darker_index, darker in enumerate(stack[:-1]):
        for brighter in stack[darker_index + 1 :]:
            if darker.stack_id != brighter.stack_id:
                raise ValueError("exposure stack mixes component IDs")
            ev_gap = math.log2(brighter.mean_luma / darker.mean_luma)
            if ev_gap + 1e-12 >= minimum_ev_gap:
                pairs.append(
                    ExposurePair(
                        stack_id=darker.stack_id,
                        darker_sha256=darker.sha256,
                        brighter_sha256=brighter.sha256,
                        ev_gap=ev_gap,
                    )
                )
    return tuple(pairs)


def load_frozen_exposure_stacks(
    rows: Iterable[Mapping[str, str]],
) -> tuple[dict[str, tuple[ExposureImage, ...]], dict[str, str]]:
    """Validate the published ordinal manifest and reconstruct its stacks."""

    grouped: dict[str, list[tuple[int, ExposureImage]]] = {}
    image_sha_by_id: dict[str, str] = {}
    seen_paths: set[str] = set()
    seen_sha256: set[str] = set()
    for raw in rows:
        missing = REQUIRED_FROZEN_FIELDS - set(raw)
        if missing:
            raise ValueError(f"frozen exposure row is missing fields: {sorted(missing)}")
        try:
            byte_size = int(raw["bytes"])
            width = int(raw["width"])
            height = int(raw["height"])
            mean_luma = float(raw["mean_luma"])
            order = int(raw["order_dark_to_bright"])
            darkness_rank = int(raw["darkness_rank"])
            ev = float(raw["ev_darker_than_brightest"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"bad frozen exposure row: {raw.get('image_id')}") from exc
        stack_id = raw["stack_id"]
        image_id = raw["image_id"]
        path = raw["path"]
        sha256 = raw["sha256"]
        if not stack_id or not image_id or not path:
            raise ValueError("frozen exposure identity is empty")
        if image_id in image_sha_by_id or path in seen_paths or sha256 in seen_sha256:
            raise ValueError(f"duplicate frozen exposure identity: {image_id}")
        if (
            byte_size <= 0
            or width <= 0
            or height <= 0
            or not math.isfinite(mean_luma)
            or mean_luma <= 0.0
            or mean_luma > 255.0
            or order < 0
            or darkness_rank <= 0
            or not math.isfinite(ev)
            or ev < 0.0
        ):
            raise ValueError(f"invalid frozen exposure metadata: {image_id}")
        image = ExposureImage(
            stack_id=stack_id,
            path=path,
            subset=raw["subset"],
            filename=raw["filename"],
            sha256=sha256,
            byte_size=byte_size,
            width=width,
            height=height,
            mode=raw["mode"],
            mean_luma=mean_luma,
            order_dark_to_bright=order,
            darkness_rank=darkness_rank,
            ev_darker_than_brightest=ev,
        )
        grouped.setdefault(stack_id, []).append((order, image))
        image_sha_by_id[image_id] = sha256
        seen_paths.add(path)
        seen_sha256.add(sha256)

    stacks: dict[str, tuple[ExposureImage, ...]] = {}
    for stack_id, indexed in sorted(grouped.items()):
        indexed.sort(key=lambda item: item[0])
        if len(indexed) < 2 or [item[0] for item in indexed] != list(range(len(indexed))):
            raise ValueError(f"frozen exposure stack order is incomplete: {stack_id}")
        images = tuple(item[1] for item in indexed)
        if [image.darkness_rank for image in images] != list(
            range(len(images), 0, -1)
        ):
            raise ValueError(f"frozen darkness ranks differ: {stack_id}")
        if any(
            right.mean_luma <= left.mean_luma
            for left, right in zip(images, images[1:])
        ):
            raise ValueError(f"frozen mean-luma order differs: {stack_id}")
        brightest = images[-1].mean_luma
        for image in images:
            expected_ev = math.log2(brightest / image.mean_luma)
            if not math.isclose(
                image.ev_darker_than_brightest,
                expected_ev,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError(f"frozen exposure EV differs: {stack_id}")
        stacks[stack_id] = images
    if not stacks:
        raise ValueError("frozen exposure manifest is empty")
    return stacks, image_sha_by_id


def expected_level_from_nll(
    scores: Mapping[int | str, float], *, temperature: float = 1.0
) -> tuple[float, dict[str, float], float]:
    """Convert canonical-target NLLs to a soft severity and entropy."""

    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive")
    nlls: list[float] = []
    for level in LEVELS:
        raw = scores[level] if level in scores else scores.get(str(level))
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise ValueError(f"missing or invalid NLL for L{level}")
        value = float(raw)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"invalid NLL for L{level}")
        nlls.append(value)
    offset = min(nlls)
    weights = [math.exp(-(value - offset) / temperature) for value in nlls]
    total = sum(weights)
    probabilities = [value / total for value in weights]
    expected = sum(level * probability for level, probability in zip(LEVELS, probabilities))
    entropy = -sum(
        probability * math.log(probability)
        for probability in probabilities
        if probability > 0.0
    )
    return expected, {
        str(level): probability
        for level, probability in zip(LEVELS, probabilities)
    }, entropy


def _rank(values: Sequence[float]) -> list[float]:
    ordered = sorted(enumerate(values), key=lambda item: (item[1], item[0]))
    ranks = [0.0] * len(values)
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end][1] == ordered[start][1]:
            end += 1
        average = (start + 1 + end) / 2.0
        for index in range(start, end):
            ranks[ordered[index][0]] = average
        start = end
    return ranks


def _pearson(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        raise ValueError("correlation needs equally sized nontrivial inputs")
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    numerator = sum(
        (left_value - left_mean) * (right_value - right_mean)
        for left_value, right_value in zip(left, right)
    )
    left_energy = sum((value - left_mean) ** 2 for value in left)
    right_energy = sum((value - right_mean) ** 2 for value in right)
    denominator = math.sqrt(left_energy * right_energy)
    if denominator == 0.0:
        return 0.0
    return numerator / denominator


def spearman_correlation(left: Sequence[float], right: Sequence[float]) -> float:
    return _pearson(_rank(left), _rank(right))


def evaluate_exposure_ordering(
    stacks: Mapping[str, Sequence[ExposureImage]],
    candidate_nll_by_sha256: Mapping[str, Mapping[int | str, float]],
    *,
    minimum_ev_gaps: Sequence[float] = (0.5, 1.0, 2.0),
    temperature: float = 1.0,
) -> dict[str, object]:
    """Evaluate soft and hard monotonic severity over fixed exposure stacks."""

    expected: dict[str, float] = {}
    probabilities: dict[str, dict[str, float]] = {}
    entropies: dict[str, float] = {}
    hard_levels: dict[str, int] = {}
    all_images = [image for stack in stacks.values() for image in stack]
    expected_sha = {image.sha256 for image in all_images}
    if set(candidate_nll_by_sha256) != expected_sha:
        missing = sorted(expected_sha - set(candidate_nll_by_sha256))
        extra = sorted(set(candidate_nll_by_sha256) - expected_sha)
        raise ValueError(f"prediction identity mismatch: missing={missing[:3]} extra={extra[:3]}")
    for image in all_images:
        scores = candidate_nll_by_sha256[image.sha256]
        soft, probability, entropy = expected_level_from_nll(
            scores, temperature=temperature
        )
        expected[image.sha256] = soft
        probabilities[image.sha256] = probability
        entropies[image.sha256] = entropy
        hard_levels[image.sha256] = min(
            LEVELS,
            key=lambda level: (
                float(scores[level] if level in scores else scores[str(level)]),
                level,
            ),
        )

    per_stack: dict[str, dict[str, object]] = {}
    correlations: list[float] = []
    for stack_id, stack in sorted(stacks.items()):
        darkness = [image.ev_darker_than_brightest for image in stack]
        soft = [expected[image.sha256] for image in stack]
        correlation = spearman_correlation(darkness, soft)
        correlations.append(correlation)
        adjacent = exposure_pairs(stack)
        adjacent = tuple(
            pair
            for pair in adjacent
            if any(
                first.sha256 == pair.darker_sha256
                and second.sha256 == pair.brighter_sha256
                for first, second in zip(stack, stack[1:])
            )
        )
        adjacent_inversions = sum(
            expected[pair.darker_sha256] < expected[pair.brighter_sha256]
            for pair in adjacent
        )
        per_stack[stack_id] = {
            "images": len(stack),
            "spearman_expected_severity_vs_darkness_ev": correlation,
            "expected_level_span": max(soft) - min(soft),
            "adjacent_pairs": len(adjacent),
            "adjacent_inversions": adjacent_inversions,
        }

    pairwise: dict[str, dict[str, object]] = {}
    for minimum_gap in minimum_ev_gaps:
        pairs = [
            pair
            for stack in stacks.values()
            for pair in exposure_pairs(stack, minimum_ev_gap=float(minimum_gap))
        ]
        if not pairs:
            raise ValueError(f"no exposure pairs at minimum EV gap {minimum_gap}")
        correct = ties = inverted = hard_non_inversions = 0
        for pair in pairs:
            difference = expected[pair.darker_sha256] - expected[pair.brighter_sha256]
            if difference > 1e-12:
                correct += 1
            elif difference < -1e-12:
                inverted += 1
            else:
                ties += 1
            hard_non_inversions += int(
                hard_levels[pair.darker_sha256] >= hard_levels[pair.brighter_sha256]
            )
        pairwise[f"minimum_ev_{minimum_gap:g}"] = {
            "pairs": len(pairs),
            "strict_correct": correct,
            "ties": ties,
            "inversions": inverted,
            "continuous_concordance_with_half_credit_for_ties": (
                correct + 0.5 * ties
            )
            / len(pairs),
            "hard_level_non_inversion_rate": hard_non_inversions / len(pairs),
        }

    return {
        "stacks": len(stacks),
        "images": len(all_images),
        "temperature": temperature,
        "median_stack_spearman": statistics.median(correlations),
        "mean_stack_spearman": statistics.fmean(correlations),
        "positive_stack_spearman_fraction": sum(value > 0.0 for value in correlations)
        / len(correlations),
        "pooled_expected_level_span": max(expected.values()) - min(expected.values()),
        "mean_prediction_entropy": statistics.fmean(entropies.values()),
        "pairwise": pairwise,
        "per_stack": per_stack,
        "expected_level_by_sha256": expected,
        "probabilities_by_sha256": probabilities,
        "hard_level_by_sha256": hard_levels,
    }


def completed_exposure_result_status(
    summary: Mapping[str, object],
    *,
    config_sha256: str,
    manifest_sha256: str,
    checkpoint_sha256: str,
) -> int:
    """Return 0 for a verified pass and 2 for a verified scientific failure."""

    checks = summary.get("gate_checks")
    if (
        summary.get("config_sha256") != config_sha256
        or summary.get("manifest_sha256") != manifest_sha256
        or summary.get("selected_visual_sha256") != checkpoint_sha256
        or summary.get("official_foundir_used") is not False
        or summary.get("absolute_real_l1_l5_accuracy_claimed") is not False
        or summary.get("downstream_restoration_gain_claimed") is not False
        or summary.get("relative_exposure_evidence_only") is not True
        or not isinstance(checks, Mapping)
        or not checks
        or any(not isinstance(value, bool) for value in checks.values())
    ):
        raise ValueError("exposure-ordering result provenance is inconsistent")
    automatic_pass = all(checks.values())
    if (
        summary.get("automatic_gate_pass") is not automatic_pass
        or summary.get("status") != ("passed" if automatic_pass else "failed")
    ):
        raise ValueError("exposure-ordering result status is inconsistent")
    return 0 if automatic_pass else 2
