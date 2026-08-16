"""Leakage-safe, fail-closed degradation assessment protocol helpers.

The protocol is intentionally generic across the five All-in-One tasks.  A
task is not usable merely because a VLM can emit syntactically valid JSON: it
must have task-specific calibration, a source-disjoint validation result, and
a passing quality gate.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

SUPPORTED_DEGRADATIONS = ("low_light", "noise", "blur", "haze", "rain")
VALID_LEVELS = frozenset(range(1, 6))
CALIBRATED_MODES = frozenset({"visual_few_shot", "frozen_visual_head", "qwen_lora"})
FORBIDDEN_UNCALIBRATED_MODES = frozenset({"zero_shot", "rubric_only", "none"})

# This registry is an execution interlock, not a claim that the later tasks
# have been prepared.  A new task must explicitly replace its blocked state
# only after its own protocol/configuration is reviewed and validated.
TASK_READINESS = {
    "low_light": "validation_v2_configured",
    "noise": "blocked_no_task_calibration",
    "blur": "blocked_no_task_calibration",
    "haze": "blocked_no_task_calibration",
    "rain": "blocked_no_task_calibration",
}


@dataclass(frozen=True)
class AssessmentProtocol:
    degradation: str
    calibration_mode: str
    calibration_split: str
    validation_split: str
    protected_splits: tuple[str, ...]
    baseline_name: str
    require_all_predicted_levels: bool
    minimum_per_level_recall: float
    minimum_accuracy: float
    require_baseline_match: bool


def _finite_rate(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return result


def parse_protocol(payload: Mapping[str, object]) -> AssessmentProtocol:
    """Validate a task protocol and reject the original rubric-only mistake."""

    if not isinstance(payload, Mapping):
        raise ValueError("assessment protocol must be an object")
    degradation = payload.get("degradation")
    if degradation not in SUPPORTED_DEGRADATIONS:
        raise ValueError(f"unsupported degradation: {degradation!r}")
    mode = payload.get("calibration_mode")
    if mode in FORBIDDEN_UNCALIBRATED_MODES or mode not in CALIBRATED_MODES:
        raise ValueError(
            f"{degradation} severity assessment requires task-specific calibration; "
            f"mode {mode!r} is forbidden"
        )
    calibration_split = payload.get("calibration_split")
    validation_split = payload.get("validation_split")
    if not isinstance(calibration_split, str) or not calibration_split:
        raise ValueError("calibration_split must be a non-empty string")
    if not isinstance(validation_split, str) or not validation_split:
        raise ValueError("validation_split must be a non-empty string")
    if calibration_split == validation_split:
        raise ValueError("calibration and validation splits must differ")
    protected = payload.get("protected_splits")
    if (
        not isinstance(protected, list)
        or not protected
        or any(not isinstance(value, str) or not value for value in protected)
        or len(set(protected)) != len(protected)
    ):
        raise ValueError("protected_splits must be a non-empty unique string list")
    if calibration_split in protected or validation_split in protected:
        raise ValueError("calibration/validation cannot use a protected split")
    baseline = payload.get("baseline_name")
    if not isinstance(baseline, str) or not baseline or baseline == "none":
        raise ValueError("a task-specific comparison baseline is required")
    gates = payload.get("gates")
    if not isinstance(gates, Mapping):
        raise ValueError("assessment gates are required")
    if gates.get("structured_output_rate") != 1.0:
        raise ValueError("structured_output_rate gate must be exactly 1.0")
    if gates.get("require_all_predicted_levels") is not True:
        raise ValueError("all five predicted levels must be required")
    if gates.get("require_baseline_match") is not True:
        raise ValueError("the task baseline comparison must be required")
    minimum_recall = _finite_rate(gates.get("minimum_per_level_recall"), "minimum recall")
    if minimum_recall <= 0.0:
        raise ValueError("minimum per-level recall must be positive")
    minimum_accuracy = _finite_rate(gates.get("minimum_accuracy"), "minimum accuracy")
    return AssessmentProtocol(
        degradation=str(degradation),
        calibration_mode=str(mode),
        calibration_split=calibration_split,
        validation_split=validation_split,
        protected_splits=tuple(protected),
        baseline_name=baseline,
        require_all_predicted_levels=True,
        minimum_per_level_recall=minimum_recall,
        minimum_accuracy=minimum_accuracy,
        require_baseline_match=True,
    )


def require_task_ready(degradation: str) -> None:
    """Block accidental execution of an unconfigured degradation task."""

    if degradation not in SUPPORTED_DEGRADATIONS:
        raise ValueError(f"unsupported degradation: {degradation!r}")
    state = TASK_READINESS[degradation]
    if state != "validation_v2_configured":
        raise RuntimeError(f"{degradation} assessment is blocked: {state}")


def validate_source_isolation(
    calibration_records: Sequence[object], validation_records: Sequence[object]
) -> None:
    """Reject source or exact-image leakage between calibration and validation."""

    def values(records: Sequence[object], attribute: str) -> set[object]:
        return {getattr(record, attribute) for record in records}

    calibration_sources = values(calibration_records, "source_id")
    validation_sources = values(validation_records, "source_id")
    overlap = calibration_sources & validation_sources
    if overlap:
        raise ValueError(f"calibration/validation source leakage: {sorted(overlap)!r}")
    calibration_hashes = values(calibration_records, "degraded_sha256")
    validation_hashes = values(validation_records, "degraded_sha256")
    hash_overlap = calibration_hashes & validation_hashes
    if hash_overlap:
        raise ValueError("calibration/validation degraded-image hash leakage")


def strict_condition_payload(raw_output: str, degradation: str) -> int:
    """Parse only the canonical two-field schema for one degradation."""

    if degradation not in SUPPORTED_DEGRADATIONS:
        raise ValueError(f"unsupported degradation: {degradation!r}")
    if not isinstance(raw_output, str) or not raw_output.strip():
        raise ValueError("empty assessor output")
    try:
        payload = json.loads(raw_output.strip())
    except json.JSONDecodeError as exc:
        raise ValueError("assessor output is not one JSON value") from exc
    if not isinstance(payload, dict) or set(payload) != {"degradations", "levels"}:
        raise ValueError("assessor output has unexpected top-level fields")
    if payload["degradations"] != [degradation]:
        raise ValueError("assessor output has the wrong degradation list")
    levels = payload["levels"]
    if not isinstance(levels, dict) or set(levels) != {degradation}:
        raise ValueError("assessor output has unexpected level fields")
    level = levels[degradation]
    if isinstance(level, bool) or not isinstance(level, int) or level not in VALID_LEVELS:
        raise ValueError("assessor level must be an integer from 1 to 5")
    return level


def canonical_condition_json(degradation: str, level: int) -> str:
    if degradation not in SUPPORTED_DEGRADATIONS:
        raise ValueError(f"unsupported degradation: {degradation!r}")
    if isinstance(level, bool) or level not in VALID_LEVELS:
        raise ValueError("level must be an integer from 1 to 5")
    return json.dumps(
        {"degradations": [degradation], "levels": {degradation: level}},
        separators=(",", ":"),
    )


def evaluate_severity(records: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Evaluate strict records without silently repairing invalid predictions."""

    if not records:
        raise ValueError("prediction records are empty")
    confusion: Counter[tuple[int, int]] = Counter()
    true_counts: Counter[int] = Counter()
    predicted_counts: Counter[int] = Counter()
    valid = 0
    exact = 0
    absolute_error = 0
    for record in records:
        true_level = record.get("true_level")
        predicted_level = record.get("predicted_level")
        if isinstance(true_level, bool) or true_level not in VALID_LEVELS:
            raise ValueError(f"invalid true level: {true_level!r}")
        true_counts[true_level] += 1
        if record.get("valid_structured_output") is not True:
            continue
        if isinstance(predicted_level, bool) or predicted_level not in VALID_LEVELS:
            raise ValueError(f"invalid predicted level: {predicted_level!r}")
        valid += 1
        predicted_counts[predicted_level] += 1
        confusion[(true_level, predicted_level)] += 1
        exact += int(true_level == predicted_level)
        absolute_error += abs(true_level - predicted_level)

    count = len(records)
    per_level: dict[str, dict[str, float | int]] = {}
    for level in sorted(VALID_LEVELS):
        support = true_counts[level]
        correct = confusion[(level, level)]
        per_level[str(level)] = {
            "support": support,
            "correct": correct,
            "recall": correct / support if support else 0.0,
            "predicted_count": predicted_counts[level],
        }
    # Invalid output is a full assessment failure. Accuracy uses all records,
    # while MAE takes the worst possible five-level error so the artifact stays
    # finite, strict JSON instead of serializing a non-standard Infinity value.
    metrics: dict[str, object] = {
        "samples": count,
        "valid_structured_output_rate": valid / count,
        "invalid_outputs": count - valid,
        "degradation_type_accuracy": valid / count,
        "degradation_type_macro_f1": valid / count,
        "severity_accuracy": exact / count,
        "combination_exact_match_accuracy": exact / count,
        "severity_mean_absolute_error": absolute_error / count if valid == count else 4.0,
        "predicted_levels": sorted(predicted_counts),
        "predicted_level_counts": {
            str(level): predicted_counts[level] for level in sorted(VALID_LEVELS)
        },
        "per_level": per_level,
        "confusion": {
            str(true): {
                str(predicted): confusion[(true, predicted)] for predicted in sorted(VALID_LEVELS)
            }
            for true in sorted(VALID_LEVELS)
        },
    }
    return metrics


def fit_scalar_centroids(
    labeled_values: Iterable[tuple[int, float]],
) -> dict[int, float]:
    """Fit the deliberately simple scalar baseline using calibration data only."""

    grouped: dict[int, list[float]] = {level: [] for level in VALID_LEVELS}
    for level, value in labeled_values:
        if isinstance(level, bool) or level not in VALID_LEVELS:
            raise ValueError(f"invalid baseline level: {level!r}")
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError("baseline values must be finite")
        grouped[level].append(numeric)
    if any(not grouped[level] for level in VALID_LEVELS):
        raise ValueError("baseline calibration must contain every level")
    return {level: sum(grouped[level]) / len(grouped[level]) for level in sorted(VALID_LEVELS)}


def predict_scalar_centroid(value: float, centroids: Mapping[int, float]) -> int:
    numeric = float(value)
    if not math.isfinite(numeric) or set(centroids) != VALID_LEVELS:
        raise ValueError("invalid scalar or centroid map")
    return min(VALID_LEVELS, key=lambda level: (abs(numeric - centroids[level]), level))


def assessment_gate(
    metrics: Mapping[str, object],
    baseline_metrics: Mapping[str, object],
    protocol: AssessmentProtocol,
) -> dict[str, object]:
    """Apply absolute, class-collapse, per-level, and baseline-relative gates."""

    per_level = metrics.get("per_level")
    if not isinstance(per_level, Mapping) or set(per_level) != {str(x) for x in VALID_LEVELS}:
        raise ValueError("metrics do not contain all five levels")
    recalls = {
        level: float(per_level[str(level)]["recall"])  # type: ignore[index]
        for level in sorted(VALID_LEVELS)
    }
    support_complete = all(int(per_level[str(level)]["support"]) > 0 for level in VALID_LEVELS)  # type: ignore[index]
    structured_pass = float(metrics["valid_structured_output_rate"]) == 1.0
    coverage_pass = set(metrics.get("predicted_levels", [])) == VALID_LEVELS
    recall_pass = all(value >= protocol.minimum_per_level_recall for value in recalls.values())
    accuracy = float(metrics["severity_accuracy"])
    mae = float(metrics["severity_mean_absolute_error"])
    baseline_accuracy = float(baseline_metrics["severity_accuracy"])
    baseline_mae = float(baseline_metrics["severity_mean_absolute_error"])
    absolute_accuracy_pass = accuracy >= protocol.minimum_accuracy
    baseline_accuracy_pass = accuracy >= baseline_accuracy
    baseline_mae_pass = mae <= baseline_mae
    passed = all(
        (
            support_complete,
            structured_pass,
            coverage_pass,
            recall_pass,
            absolute_accuracy_pass,
            baseline_accuracy_pass,
            baseline_mae_pass,
        )
    )
    return {
        "automatic_gate_pass": passed,
        "support_complete": support_complete,
        "structured_output_pass": structured_pass,
        "all_predicted_levels_pass": coverage_pass,
        "minimum_per_level_recall_pass": recall_pass,
        "absolute_accuracy_pass": absolute_accuracy_pass,
        "baseline_accuracy_pass": baseline_accuracy_pass,
        "baseline_mae_pass": baseline_mae_pass,
        "minimum_per_level_recall": protocol.minimum_per_level_recall,
        "minimum_accuracy": protocol.minimum_accuracy,
        "realized_per_level_recall": {str(key): value for key, value in recalls.items()},
        "severity_accuracy": accuracy,
        "severity_mean_absolute_error": mae,
        "baseline_severity_accuracy": baseline_accuracy,
        "baseline_severity_mean_absolute_error": baseline_mae,
    }


def record_bundle_sha256(records: Sequence[object]) -> str:
    """Bind a calibration/validation selection to IDs, sources, labels, and hashes."""

    rows = [
        {
            "pair_id": getattr(record, "pair_id"),
            "source_id": getattr(record, "source_id"),
            "level": getattr(record, "level_index"),
            "split": getattr(record, "split"),
            "degraded_sha256": getattr(record, "degraded_sha256"),
        }
        for record in records
    ]
    encoded = json.dumps(rows, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()
