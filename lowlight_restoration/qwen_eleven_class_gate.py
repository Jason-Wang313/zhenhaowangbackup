"""Canonical FoundIR singleton and observed mixed-degradation classes."""

from __future__ import annotations

from collections import Counter
from typing import Sequence

from .qwen_five_class_gate import routing_metrics


LABEL_BY_CATEGORY = {
    "01Blur": "blur",
    "05Noise": "noise",
    "08Haze": "haze",
    "10Rain": "rain",
    "14Lowlight": "low_light",
    "02Blur_Noise": "blur+noise",
    "09Lowlight_Haze": "haze+low_light",
    "13Rain_Haze": "rain+haze",
    "15Lowlight_Blur": "blur+low_light",
    "16Lowlight_Noise": "noise+low_light",
    "18Lowlight_Blur_Noise": "blur+noise+low_light",
}
CLASSES = tuple(LABEL_BY_CATEGORY.values())
SINGLETON_CLASSES = CLASSES[:5]
MIXED_CLASSES = CLASSES[5:]


def consensus_route(
    five_probabilities: Sequence[float],
    eleven_probabilities: Sequence[float],
    *,
    five_threshold: float = 0.925,
    eleven_threshold: float = 0.95,
) -> str | None:
    """Route mixed proposals directly and require two-head agreement for singletons."""

    if len(five_probabilities) != len(SINGLETON_CLASSES):
        raise ValueError("bad five-class probability row")
    if len(eleven_probabilities) != len(CLASSES):
        raise ValueError("bad eleven-class probability row")
    eleven_index = max(range(len(CLASSES)), key=lambda index: float(eleven_probabilities[index]))
    proposal = CLASSES[eleven_index]
    if proposal in MIXED_CLASSES:
        return proposal if float(eleven_probabilities[eleven_index]) >= eleven_threshold else None
    five_index = max(
        range(len(SINGLETON_CLASSES)), key=lambda index: float(five_probabilities[index])
    )
    if float(five_probabilities[five_index]) < five_threshold:
        return None
    return proposal if SINGLETON_CLASSES[five_index] == proposal else None


def consensus_metrics(routes: Sequence[str | None], truths: Sequence[str]) -> dict[str, object]:
    """Score a precomputed fail-closed consensus route without model-dependent logic."""

    if len(routes) != len(truths) or not truths:
        raise ValueError("routes and truths must be nonempty and aligned")
    totals: Counter[str] = Counter()
    per_class = {name: Counter() for name in CLASSES}
    for route, truth in zip(routes, truths, strict=True):
        if truth not in CLASSES or (route is not None and route not in CLASSES):
            raise ValueError("unknown route label")
        outcome = "refusal" if route is None else "correct" if route == truth else "wrong"
        totals[outcome] += 1
        per_class[truth][outcome] += 1

    def rates(counter: Counter[str], denominator: int) -> dict[str, float | int]:
        accepted = counter["correct"] + counter["wrong"]
        return {
            "records": denominator,
            "correct": counter["correct"],
            "wrong": counter["wrong"],
            "refusal": counter["refusal"],
            "correct_rate": counter["correct"] / denominator if denominator else 0.0,
            "wrong_rate": counter["wrong"] / denominator if denominator else 0.0,
            "refusal_rate": counter["refusal"] / denominator if denominator else 0.0,
            "coverage": accepted / denominator if denominator else 0.0,
            "accepted_accuracy": counter["correct"] / accepted if accepted else 1.0,
        }

    return {
        "overall": rates(totals, len(truths)),
        "per_class": {
            name: rates(per_class[name], sum(per_class[name].values())) for name in CLASSES
        },
    }


def eleven_class_metrics(
    probabilities: Sequence[Sequence[float]],
    labels: Sequence[int],
    *,
    threshold: float,
) -> dict[str, object]:
    """Reuse the fail-closed metric definition with eleven canonical classes."""

    # The five-class helper is intentionally generic apart from its class names.
    # Reimplementing the small naming-dependent portion here keeps frozen v1 code unchanged.
    from collections import Counter

    if len(probabilities) != len(labels) or not labels:
        raise ValueError("probabilities and labels must be nonempty and aligned")
    totals = Counter()
    per_class = {name: Counter() for name in CLASSES}
    for values, truth in zip(probabilities, labels, strict=True):
        if len(values) != len(CLASSES) or truth not in range(len(CLASSES)):
            raise ValueError("bad eleven-class prediction row")
        prediction = max(range(len(CLASSES)), key=lambda index: float(values[index]))
        accepted = float(values[prediction]) >= threshold
        outcome = "refusal" if not accepted else "correct" if prediction == truth else "wrong"
        totals[outcome] += 1
        per_class[CLASSES[truth]][outcome] += 1

    def rates(counter: Counter[str], denominator: int):
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
        "overall": rates(totals, len(labels)),
        "per_class": {
            name: rates(per_class[name], sum(per_class[name].values())) for name in CLASSES
        },
    }
