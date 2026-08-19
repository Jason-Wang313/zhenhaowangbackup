"""Contracts and metrics for frozen-Qwen real-domain Lowlight validation."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence


APPROVED_STATUS = "approved_for_frozen_qwen_real_lowlight_validation_only"
PROJECT_LUMA_RATIO_ANCHORS = {2: 0.50, 3: 0.38, 4: 0.30, 5: 0.14}
SELECTED_CHECKPOINT_STEP = 600
SELECTED_VISUAL_SHA256 = (
    "b0e65a8597077973a2a103dc3369e0cb1ea703de95e2e4841d359c3d34eff77f"
)
PRIMARY_SUBSET = "sice_balanced_proxy"
STRESS_SUBSET = "lolv2_extreme_stress_proxy"


def load_real_validation_config(
    path: Path,
) -> tuple[dict[str, object], str, bytes]:
    """Load one immutable real-validation contract and its exact digest."""

    raw = path.read_bytes()
    config = json.loads(raw)
    if not isinstance(config, dict) or config.get("schema_version") != 1:
        raise ValueError("bad Qwen real-validation config schema")
    if config.get("status") != APPROVED_STATUS:
        raise ValueError("config is not approved for frozen-Qwen real validation")
    return config, hashlib.sha256(raw).hexdigest(), raw


def validate_real_validation_contract(config: Mapping[str, object]) -> None:
    """Reject a real-domain run whose frozen scientific contract changed."""

    required_sections = (
        "model",
        "processor",
        "prompt",
        "data",
        "evaluation",
        "gates",
        "runtime",
        "telemetry",
    )
    sections = {name: config.get(name) for name in required_sections}
    if any(not isinstance(value, dict) for value in sections.values()):
        raise ValueError("real-validation config is missing a required section")
    model = sections["model"]
    processor = sections["processor"]
    data = sections["data"]
    evaluation = sections["evaluation"]
    gates = sections["gates"]
    runtime = sections["runtime"]
    telemetry = sections["telemetry"]
    assert all(
        isinstance(section, dict)
        for section in (model, processor, data, evaluation, gates, runtime, telemetry)
    )

    if (
        config.get("scope") != "frozen_step600_non_FoundIR_real_Lowlight_validation"
        or config.get("official_foundir_used") is not False
        or config.get("model_predictions_consulted_during_selection") is not False
    ):
        raise ValueError("real-domain scope or FoundIR protection changed")
    if (
        model.get("architecture") != "Qwen2_5_VLForConditionalGeneration"
        or model.get("selected_checkpoint_step") != SELECTED_CHECKPOINT_STEP
        or model.get("selected_visual_sha256") != SELECTED_VISUAL_SHA256
        or model.get("attention_implementation") != "sdpa"
        or model.get("dtype") != "bfloat16"
    ):
        raise ValueError("frozen step-600 Qwen checkpoint contract changed")
    groups = model.get("expected_parameter_groups")
    if not isinstance(groups, dict):
        raise ValueError("Qwen parameter-group contract is missing")
    visual = sum(
        int(groups.get(name, 0))
        for name in ("visual.patch_embed", "visual.blocks", "visual.merger")
    )
    frozen = sum(int(groups.get(name, 0)) for name in ("language_model", "lm_head"))
    if (
        visual != model.get("expected_trainable_parameters")
        or frozen != model.get("expected_frozen_parameters")
        or visual + frozen != model.get("expected_total_parameters")
    ):
        raise ValueError("Qwen parameter-count contract changed")
    if (
        processor.get("use_fast") is not False
        or processor.get("padding_side") != "right"
        or processor.get("min_pixels") != processor.get("max_pixels")
    ):
        raise ValueError("real-domain processor contract changed")

    if (
        data.get("annotation_kind")
        != "reference_based_proxy_not_human_ground_truth"
        or data.get("official_foundir_used") is not False
        or data.get("model_predictions_consulted_during_selection") is not False
    ):
        raise ValueError("real-domain proxy-label disclosure changed")
    subsets = data.get("subsets")
    if not isinstance(subsets, dict) or set(subsets) != {PRIMARY_SUBSET, STRESS_SUBSET}:
        raise ValueError("real-domain subset contract changed")
    for name, required_levels in (
        (PRIMARY_SUBSET, [1, 2, 3, 4, 5]),
        (STRESS_SUBSET, [1, 3, 4, 5]),
    ):
        subset = subsets[name]
        if (
            not isinstance(subset, dict)
            or subset.get("records", 0) <= 0
            or subset.get("levels_present") != required_levels
            or not isinstance(subset.get("record_bundle_sha256"), str)
        ):
            raise ValueError(f"bad real-domain subset contract: {name}")

    if (
        evaluation.get("do_sample") is not False
        or evaluation.get("batch_size") != 5
        or evaluation.get("max_new_tokens") != 64
        or evaluation.get("subset_order") != [PRIMARY_SUBSET, STRESS_SUBSET]
    ):
        raise ValueError("real-domain generation contract changed")
    if gates.get("automatic_pass_requires") != [PRIMARY_SUBSET, STRESS_SUBSET]:
        raise ValueError("real-domain automatic gate composition changed")
    subset_gates = gates.get("subsets")
    if not isinstance(subset_gates, dict) or set(subset_gates) != {
        PRIMARY_SUBSET,
        STRESS_SUBSET,
    }:
        raise ValueError("real-domain gate thresholds are incomplete")
    for name, gate in subset_gates.items():
        if not isinstance(gate, dict):
            raise ValueError(f"real-domain gate is malformed: {name}")
        required_levels = gate.get("required_levels")
        if not isinstance(required_levels, list) or not required_levels:
            raise ValueError(f"real-domain gate has no required levels: {name}")
        for threshold in (
            "minimum_valid_structured_output_rate",
            "minimum_degradation_type_accuracy",
            "minimum_degradation_type_macro_f1",
            "minimum_severity_accuracy",
            "minimum_severity_macro_f1",
            "minimum_within_one_level_accuracy",
            "minimum_per_level_accuracy",
            "minimum_quadratic_weighted_kappa",
        ):
            value = gate.get(threshold)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not 0.0 <= float(value) <= 1.0
            ):
                raise ValueError(f"bad real-domain threshold: {name}.{threshold}")
        maximum_mae = gate.get("maximum_severity_mean_absolute_error")
        if (
            isinstance(maximum_mae, bool)
            or not isinstance(maximum_mae, (int, float))
            or not 0.0 <= float(maximum_mae) <= 4.0
        ):
            raise ValueError(f"bad real-domain MAE threshold: {name}")

    if (
        runtime.get("offline") is not True
        or runtime.get("device") != "cuda"
        or runtime.get("required_visible_device_count") != 1
        or runtime.get("deterministic_algorithms") is not True
        or runtime.get("sdpa_backend") != "math"
    ):
        raise ValueError("real-domain runtime contract changed")
    if (
        telemetry.get("sample_interval_seconds") != 2
        or telemetry.get("monitor_all_selected_gpus") is not True
        or not isinstance(telemetry.get("csv"), str)
    ):
        raise ValueError("real-domain telemetry contract changed")


def proxy_level_from_luma_ratio(ratio: float) -> int:
    """Map a positive input/reference luma ratio to the nearest L2--L5 anchor."""

    if isinstance(ratio, bool) or not isinstance(ratio, (int, float)):
        raise ValueError("luma ratio must be numeric")
    value = float(ratio)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("luma ratio must be finite and positive")
    return min(
        PROJECT_LUMA_RATIO_ANCHORS,
        key=lambda level: (
            abs(math.log2(value / PROJECT_LUMA_RATIO_ANCHORS[level])),
            level,
        ),
    )


def proxy_anchor_error_ev(ratio: float, level: int) -> float:
    """Return absolute log2 distance from a project's real-domain proxy anchor."""

    if level not in PROJECT_LUMA_RATIO_ANCHORS:
        raise ValueError("proxy anchor error is defined only for L2--L5")
    proxy_level_from_luma_ratio(ratio)
    return abs(math.log2(float(ratio) / PROJECT_LUMA_RATIO_ANCHORS[level]))


def select_balanced_proxy_candidates(
    candidates: Sequence[Mapping[str, object]],
    *,
    maximum_per_level: int,
    minimum_per_level: int,
    maximum_anchor_error_ev: float,
    seed: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Select a deterministic balanced L2--L5 subset before model inference.

    Each candidate must expose a unique ``candidate_id`` and a positive
    ``luma_ratio``. A candidate is assigned to its nearest frozen project
    anchor. Selection favors the closest real exposures; a seeded digest is
    only the deterministic tie-breaker.
    """

    if maximum_per_level <= 0 or not 0 < minimum_per_level <= maximum_per_level:
        raise ValueError("bad balanced proxy quota")
    if not math.isfinite(maximum_anchor_error_ev) or maximum_anchor_error_ev <= 0:
        raise ValueError("bad proxy anchor tolerance")
    if seed < 0:
        raise ValueError("selection seed must be nonnegative")

    seen: set[str] = set()
    eligible: dict[int, list[dict[str, object]]] = {
        level: [] for level in PROJECT_LUMA_RATIO_ANCHORS
    }
    rejected = 0
    for source in candidates:
        candidate = dict(source)
        candidate_id = candidate.get("candidate_id")
        ratio = candidate.get("luma_ratio")
        if not isinstance(candidate_id, str) or not candidate_id:
            raise ValueError("proxy candidate has no stable candidate_id")
        if candidate_id in seen:
            raise ValueError(f"duplicate proxy candidate: {candidate_id}")
        seen.add(candidate_id)
        if isinstance(ratio, bool) or not isinstance(ratio, (int, float)):
            raise ValueError(f"proxy candidate has bad luma_ratio: {candidate_id}")
        level = proxy_level_from_luma_ratio(float(ratio))
        error = proxy_anchor_error_ev(float(ratio), level)
        candidate["proxy_level"] = level
        candidate["proxy_anchor"] = PROJECT_LUMA_RATIO_ANCHORS[level]
        candidate["proxy_anchor_error_ev"] = error
        candidate["selection_tiebreak"] = hashlib.sha256(
            f"{seed}:{candidate_id}".encode("utf-8")
        ).hexdigest()
        if error <= maximum_anchor_error_ev:
            eligible[level].append(candidate)
        else:
            rejected += 1

    for level, values in eligible.items():
        values.sort(
            key=lambda item: (
                float(item["proxy_anchor_error_ev"]),
                str(item["selection_tiebreak"]),
                str(item["candidate_id"]),
            )
        )
        if len(values) < minimum_per_level:
            raise ValueError(
                f"only {len(values)} eligible real candidates for L{level}; "
                f"need {minimum_per_level}"
            )

    quota = min(maximum_per_level, *(len(values) for values in eligible.values()))
    if quota < minimum_per_level:
        raise ValueError("balanced real-domain quota fell below its minimum")
    selected = [
        dict(item)
        for level in sorted(eligible)
        for item in eligible[level][:quota]
    ]
    for item in selected:
        item.pop("selection_tiebreak", None)
    evidence: dict[str, object] = {
        "candidate_count": len(candidates),
        "eligible_count_by_level": {
            str(level): len(eligible[level]) for level in sorted(eligible)
        },
        "rejected_outside_anchor_tolerance": rejected,
        "selected_per_level": quota,
        "selected_count": len(selected),
        "maximum_per_level": maximum_per_level,
        "minimum_per_level": minimum_per_level,
        "maximum_anchor_error_ev": maximum_anchor_error_ev,
        "seed": seed,
    }
    return selected, evidence


def augment_real_metrics(
    prediction_rows: Sequence[Mapping[str, object]],
    base_metrics: Mapping[str, object],
) -> dict[str, object]:
    """Add ordinal real-domain metrics to the existing structured metrics."""

    if not prediction_rows:
        raise ValueError("real-domain predictions are empty")
    metrics = dict(base_metrics)
    within_one = 0
    valid_pairs: list[tuple[int, int]] = []
    truth_support: Counter[int] = Counter()
    severity_true_positive: Counter[int] = Counter()
    severity_false_positive: Counter[int] = Counter()
    severity_false_negative: Counter[int] = Counter()
    type_true_positive: Counter[int] = Counter()
    type_false_positive: Counter[int] = Counter()
    type_false_negative: Counter[int] = Counter()
    per_level_absolute_error: Counter[int] = Counter()
    for row in prediction_rows:
        true_level = row.get("true_level")
        predicted_level = row.get("predicted_level")
        if isinstance(true_level, bool) or not isinstance(true_level, int):
            raise ValueError("real-domain prediction has no integer true_level")
        if true_level not in range(1, 6):
            raise ValueError("real-domain prediction has an out-of-range true_level")
        truth_support[true_level] += 1
        if isinstance(predicted_level, int) and not isinstance(predicted_level, bool):
            if predicted_level not in range(1, 6):
                raise ValueError("real-domain prediction has an out-of-range prediction")
            valid_pairs.append((true_level, predicted_level))
            within_one += int(abs(predicted_level - true_level) <= 1)
            per_level_absolute_error[true_level] += abs(predicted_level - true_level)
        else:
            per_level_absolute_error[true_level] += 4

        for level in range(1, 6):
            truth_matches = true_level == level
            prediction_matches = predicted_level == level
            severity_true_positive[level] += int(truth_matches and prediction_matches)
            severity_false_positive[level] += int(not truth_matches and prediction_matches)
            severity_false_negative[level] += int(truth_matches and not prediction_matches)

        true_type = int(true_level > 1)
        predicted_type = (
            None if not isinstance(predicted_level, int) else int(predicted_level > 1)
        )
        for class_value in (0, 1):
            truth_matches = true_type == class_value
            prediction_matches = predicted_type == class_value
            type_true_positive[class_value] += int(truth_matches and prediction_matches)
            type_false_positive[class_value] += int(not truth_matches and prediction_matches)
            type_false_negative[class_value] += int(truth_matches and not prediction_matches)

    def f1(
        class_value: int,
        true_positive: Counter[int],
        false_positive: Counter[int],
        false_negative: Counter[int],
    ) -> float:
        denominator = (
            2 * true_positive[class_value]
            + false_positive[class_value]
            + false_negative[class_value]
        )
        return 2 * true_positive[class_value] / denominator if denominator else 0.0

    evaluated_levels = sorted(truth_support)
    severity_f1 = {
        str(level): f1(
            level,
            severity_true_positive,
            severity_false_positive,
            severity_false_negative,
        )
        for level in evaluated_levels
    }
    degradation_type_f1 = {
        "clean": f1(0, type_true_positive, type_false_positive, type_false_negative),
        "low_light": f1(
            1, type_true_positive, type_false_positive, type_false_negative
        ),
    }
    metrics["within_one_level_accuracy"] = within_one / len(prediction_rows)
    metrics["quadratic_weighted_kappa_valid_outputs"] = _quadratic_weighted_kappa(
        valid_pairs
    )
    metrics["quadratic_weighted_kappa_valid_output_count"] = len(valid_pairs)
    metrics["evaluated_severity_levels"] = evaluated_levels
    metrics["severity_f1"] = severity_f1
    metrics["severity_macro_f1"] = sum(severity_f1.values()) / len(severity_f1)
    metrics["degradation_type_f1"] = degradation_type_f1
    metrics["degradation_type_macro_f1"] = sum(
        degradation_type_f1.values()
    ) / len(degradation_type_f1)
    metrics["per_level_mean_absolute_error"] = {
        str(level): per_level_absolute_error[level] / truth_support[level]
        for level in evaluated_levels
    }
    return metrics


def _quadratic_weighted_kappa(pairs: Sequence[tuple[int, int]]) -> float:
    if not pairs:
        return 0.0
    levels = range(1, 6)
    observed = Counter(pairs)
    truth = Counter(left for left, _ in pairs)
    predicted = Counter(right for _, right in pairs)
    count = len(pairs)
    observed_cost = sum(
        ((left - right) / 4.0) ** 2 * observed[(left, right)]
        for left in levels
        for right in levels
    )
    expected_cost = sum(
        ((left - right) / 4.0) ** 2
        * truth[left]
        * predicted[right]
        / count
        for left in levels
        for right in levels
    )
    if expected_cost == 0.0:
        return 1.0 if observed_cost == 0.0 else 0.0
    return 1.0 - observed_cost / expected_cost


def real_metric_gate_checks(
    metrics: Mapping[str, object],
    gate: Mapping[str, object],
    *,
    prefix: str,
) -> dict[str, bool]:
    """Evaluate one predeclared real-domain subset contract."""

    per_level = metrics.get("per_level")
    if not isinstance(per_level, dict):
        raise ValueError("real-domain metrics have no per-level evidence")
    required_levels = gate.get("required_levels")
    if not isinstance(required_levels, list) or not required_levels:
        raise ValueError("real-domain gate has no required levels")
    level_accuracies: list[float] = []
    for level in required_levels:
        entry = per_level.get(str(level))
        if not isinstance(entry, dict) or not isinstance(entry.get("accuracy"), (int, float)):
            raise ValueError(f"real-domain metrics omit required L{level}")
        level_accuracies.append(float(entry["accuracy"]))
    checks = {
        f"{prefix}_valid_structure_passed": float(
            metrics["valid_structured_output_rate"]
        )
        >= float(gate["minimum_valid_structured_output_rate"]),
        f"{prefix}_degradation_type_accuracy_passed": float(
            metrics["degradation_type_accuracy"]
        )
        >= float(gate["minimum_degradation_type_accuracy"]),
        f"{prefix}_degradation_type_macro_f1_passed": float(
            metrics["degradation_type_macro_f1"]
        )
        >= float(gate["minimum_degradation_type_macro_f1"]),
        f"{prefix}_severity_accuracy_passed": float(metrics["severity_accuracy"])
        >= float(gate["minimum_severity_accuracy"]),
        f"{prefix}_severity_macro_f1_passed": float(metrics["severity_macro_f1"])
        >= float(gate["minimum_severity_macro_f1"]),
        f"{prefix}_severity_mae_passed": float(metrics["severity_mean_absolute_error"])
        <= float(gate["maximum_severity_mean_absolute_error"]),
        f"{prefix}_within_one_level_passed": float(
            metrics["within_one_level_accuracy"]
        )
        >= float(gate["minimum_within_one_level_accuracy"]),
        f"{prefix}_required_level_accuracy_passed": min(level_accuracies)
        >= float(gate["minimum_per_level_accuracy"]),
        f"{prefix}_quadratic_weighted_kappa_passed": float(
            metrics["quadratic_weighted_kappa_valid_outputs"]
        )
        >= float(gate["minimum_quadratic_weighted_kappa"]),
    }
    return checks
