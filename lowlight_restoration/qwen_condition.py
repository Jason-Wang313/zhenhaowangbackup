"""Strict Qwen-VL Lowlight conditioning schema and evaluation helpers."""
from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass

VALID_LEVELS = frozenset(range(1, 6))
FALLBACK_LEVEL = 3

SYSTEM_PROMPT = (
    "You are a low-light degradation assessor. Judge only low-light exposure severity. "
    "Do not describe image content, aesthetics, objects, color style, or restoration actions. "
    "Return exactly one compact JSON object with no markdown or extra text."
)

USER_PROMPT = """Classify the input image into exactly one low-light level.

Level rubric:
1 = no low-light degradation; normally exposed or clean reference.
2 = mild low-light; visibly dim but most detail is clear.
3 = moderate low-light; substantial dimming and some shadow detail loss.
4 = strong low-light; very dark with major shadow detail loss.
5 = extreme low-light; near-black regions dominate and content visibility is severely limited.

Output constraints:
- The top-level keys must be exactly "degradations" and "levels".
- "degradations" must be exactly the array ["low_light"].
- "levels" must contain exactly the key "low_light".
- Its value must be your selected integer 1, 2, 3, 4, or 5, not a string or placeholder.
- Do not add any other field or text."""


@dataclass(frozen=True)
class LowLightCondition:
    level: int

    def __post_init__(self) -> None:
        if isinstance(self.level, bool) or self.level not in VALID_LEVELS:
            raise ValueError(f"low_light level must be an integer from 1 to 5: {self.level!r}")

    def payload(self) -> dict[str, object]:
        return {"degradations": ["low_light"], "levels": {"low_light": self.level}}

    def canonical_json(self) -> str:
        return json.dumps(self.payload(), separators=(",", ":"), sort_keys=True)


def parse_qwen_condition(raw_output: str) -> LowLightCondition:
    """Accept only the exact frozen JSON shape; do not repair model output."""

    if not isinstance(raw_output, str) or not raw_output.strip():
        raise ValueError("empty Qwen output")
    try:
        payload = json.loads(raw_output.strip())
    except json.JSONDecodeError as exc:
        raise ValueError("Qwen output is not one JSON value") from exc
    if not isinstance(payload, dict) or set(payload) != {"degradations", "levels"}:
        raise ValueError("Qwen output has unexpected top-level fields")
    if payload["degradations"] != ["low_light"]:
        raise ValueError("Qwen output must contain exactly the low_light degradation")
    levels = payload["levels"]
    if not isinstance(levels, dict) or set(levels) != {"low_light"}:
        raise ValueError("Qwen output has unexpected level fields")
    level = levels["low_light"]
    if isinstance(level, bool) or not isinstance(level, int) or level not in VALID_LEVELS:
        raise ValueError("Qwen low_light level must be an integer from 1 to 5")
    return LowLightCondition(level)


def evaluate_predictions(records: list[dict[str, object]]) -> dict[str, object]:
    """Compute the frozen single-degradation understanding metrics."""

    if not records:
        raise ValueError("prediction records are empty")
    confusion: Counter[tuple[int, int]] = Counter()
    true_counts: Counter[int] = Counter()
    valid = 0
    exact = 0
    absolute_error = 0.0
    for record in records:
        true_level = record.get("true_level")
        if isinstance(true_level, bool) or not isinstance(true_level, int) or true_level not in VALID_LEVELS:
            raise ValueError(f"invalid true_level: {true_level!r}")
        true_counts[true_level] += 1
        if record.get("valid_structured_output") is True:
            predicted_level = record.get("predicted_level")
            if (
                isinstance(predicted_level, bool)
                or not isinstance(predicted_level, int)
                or predicted_level not in VALID_LEVELS
            ):
                raise ValueError(f"invalid predicted_level: {predicted_level!r}")
            valid += 1
        else:
            predicted_level = FALLBACK_LEVEL
        confusion[(true_level, predicted_level)] += 1
        exact += int(predicted_level == true_level)
        absolute_error += abs(predicted_level - true_level)

    per_level: dict[str, dict[str, float | int]] = {}
    f1_values: list[float] = []
    for level in sorted(VALID_LEVELS):
        tp = confusion[(level, level)]
        fp = sum(confusion[(other, level)] for other in VALID_LEVELS if other != level)
        fn = sum(confusion[(level, other)] for other in VALID_LEVELS if other != level)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_values.append(f1)
        per_level[str(level)] = {
            "support": true_counts[level],
            "correct": tp,
            "accuracy": tp / true_counts[level] if true_counts[level] else 0.0,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    count = len(records)
    result: dict[str, object] = {
        "samples": count,
        "valid_structured_output_rate": valid / count,
        "degradation_type_accuracy": valid / count,
        "degradation_type_macro_f1": valid / count,
        "severity_accuracy": exact / count,
        "combination_exact_match_accuracy": exact / count,
        "severity_mean_absolute_error": absolute_error / count,
        "invalid_output_fallback_level": FALLBACK_LEVEL,
        "per_level": per_level,
        "confusion": {
            str(true): {str(predicted): confusion[(true, predicted)] for predicted in sorted(VALID_LEVELS)}
            for true in sorted(VALID_LEVELS)
        },
    }
    for value in (
        result["valid_structured_output_rate"],
        result["severity_accuracy"],
        result["severity_mean_absolute_error"],
    ):
        if not math.isfinite(float(value)):
            raise ValueError("non-finite prediction metric")
    return result
