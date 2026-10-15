"""Compositional routing helpers for the eleven supported FoundIR classes."""

from __future__ import annotations

from typing import Sequence

from .qwen_eleven_class_gate import CLASSES, consensus_metrics

COMPONENTS = ("blur", "noise", "rain", "haze", "low_light")
COMPONENTS_BY_CLASS = {
    "blur": frozenset(("blur",)),
    "noise": frozenset(("noise",)),
    "haze": frozenset(("haze",)),
    "rain": frozenset(("rain",)),
    "low_light": frozenset(("low_light",)),
    "blur+noise": frozenset(("blur", "noise")),
    "haze+low_light": frozenset(("haze", "low_light")),
    "rain+haze": frozenset(("rain", "haze")),
    "blur+low_light": frozenset(("blur", "low_light")),
    "noise+low_light": frozenset(("noise", "low_light")),
    "blur+noise+low_light": frozenset(("blur", "noise", "low_light")),
}
CLASS_BY_COMPONENTS = {value: key for key, value in COMPONENTS_BY_CLASS.items()}


def component_targets(labels: Sequence[str]) -> list[list[int]]:
    """Convert class labels into five independent present/absent targets."""

    targets = []
    for label in labels:
        if label not in COMPONENTS_BY_CLASS:
            raise ValueError(f"unsupported class: {label}")
        active = COMPONENTS_BY_CLASS[label]
        targets.append([int(component in active) for component in COMPONENTS])
    return targets


def compositional_route(probabilities: Sequence[float], *, margin: float) -> str | None:
    """Accept only confident component decisions that form one supported route."""

    if len(probabilities) != len(COMPONENTS):
        raise ValueError("bad component probability row")
    if not 0.5 < margin < 1.0:
        raise ValueError("margin must be between 0.5 and 1")
    active = []
    for component, probability in zip(COMPONENTS, probabilities, strict=True):
        value = float(probability)
        if value >= margin:
            active.append(component)
        elif value > 1.0 - margin:
            return None
    return CLASS_BY_COMPONENTS.get(frozenset(active))


def compositional_metrics(
    probabilities: Sequence[Sequence[float]], labels: Sequence[str], *, margin: float
) -> dict[str, object]:
    routes = [compositional_route(row, margin=margin) for row in probabilities]
    metrics = consensus_metrics(routes, labels)
    metrics["margin"] = margin
    return metrics
