"""Frozen schema, prompt, batching, and metrics for multi-degradation Qwen.

This module is intentionally separate from :mod:`qwen_supervised`, which is
the preserved Lowlight-only experiment.  The contract here follows the
advisor-approved All-in-One route: clean is the sole L1 case and every active
degradation has an independent L2--L5 level.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable, Mapping, Sequence

import torch
from PIL import Image

if TYPE_CHECKING:
    from transformers import Qwen2_5_VLProcessor


DEGRADATIONS = ("blur", "noise", "rain", "haze", "low_light")
DEGRADATION_SET = frozenset(DEGRADATIONS)
ACTIVE_LEVELS = frozenset({2, 3, 4, 5})
LEVEL_WORDS = {2: "mild", 3: "moderate", 4: "strong", 5: "extreme"}
VARIABLE_ACTIVE_SCHEMA = "clean_l1_multilabel_independent_active_l2_l5_v1"
FIXED_FIVE_SLOT_SCHEMA = "fixed_five_slot_l1_absent_l2_l5_v1"
PER_TYPE_DIALOGUE_SCHEMA = "per_type_dialogue_l1_absent_l2_l5_v1"

SYSTEM_PROMPT = (
    "You are a degradation-assessment model for image restoration. Analyze "
    "only visible image degradation and return exactly one compact JSON object "
    "with no markdown, explanation, or additional keys."
)

DEGRADATION_QUESTION = (
    "Identify every present degradation among blur, noise, rain, haze, and "
    "low_light. Assign each present degradation its own project strength: L2 "
    "mild, L3 moderate, L4 strong, or L5 extreme. L1 is reserved exclusively "
    "for a clean image: for clean return an empty degradations array and empty "
    "levels object. List degradations in this exact canonical order: blur, "
    "noise, rain, haze, low_light. Return exactly the keys degradations and "
    "levels."
)

FIXED_FIVE_SLOT_QUESTION = (
    "For each degradation blur, noise, rain, haze, and low_light, assign its "
    "project level independently. Use L1 when that degradation is absent, L2 "
    "mild, L3 moderate, L4 strong, or L5 extreme. Return exactly one compact "
    "JSON object with these five keys in this exact order: blur, noise, rain, "
    "haze, low_light. Use integer values 1 through 5 and no additional keys."
)

PER_TYPE_SYSTEM_PROMPT = (
    "You are a degradation-assessment dialogue model for image restoration. "
    "Judge only the requested visible degradation and answer with exactly the "
    "required sentence, with no explanation or additional text."
)

PER_TYPE_QUESTION_TEMPLATE = (
    "Considering the {degradation} degradation, how would you rate its level "
    "in this image? Use L1 when {degradation} is absent, L2 mild, L3 moderate, "
    "L4 strong, or L5 extreme. Answer exactly: The {degradation} level is "
    "L<level>."
)

PER_TYPE_RESPONSE_TEMPLATE = "The {degradation} level is L{level}."


@dataclass(frozen=True)
class DegradationLabel:
    """Canonical immutable structured label."""

    degradations: tuple[str, ...]
    levels: tuple[tuple[str, int], ...]

    @property
    def level_map(self) -> dict[str, int]:
        return dict(self.levels)

    @property
    def composition_size(self) -> int:
        return len(self.degradations)


def _ordered(names: Iterable[str]) -> tuple[str, ...]:
    supplied = tuple(names)
    if len(supplied) != len(set(supplied)):
        raise ValueError("degradations must be unique")
    unknown = set(supplied) - DEGRADATION_SET
    if unknown:
        raise ValueError(f"unknown degradations: {sorted(unknown)}")
    return tuple(name for name in DEGRADATIONS if name in supplied)


def make_label(
    degradations: Iterable[str], levels: Mapping[str, int]
) -> DegradationLabel:
    """Validate and canonicalize one clean/single/double/triple label."""

    ordered = _ordered(degradations)
    if len(ordered) > 3:
        raise ValueError("the approved corpus supports at most three degradations")
    if set(levels) != set(ordered):
        raise ValueError("levels keys must exactly equal active degradations")
    normalized: list[tuple[str, int]] = []
    for name in ordered:
        level = levels[name]
        if isinstance(level, bool) or not isinstance(level, int):
            raise ValueError(f"{name} level must be an integer")
        if level not in ACTIVE_LEVELS:
            raise ValueError(f"active {name} level must be L2 through L5")
        normalized.append((name, level))
    return DegradationLabel(ordered, tuple(normalized))


def make_prediction_label(
    degradations: Iterable[str], levels: Mapping[str, int]
) -> DegradationLabel:
    """Validate a model prediction without imposing the corpus cardinality.

    Ground-truth examples are deliberately limited to clean/single/double/triple
    compositions by :func:`make_label`.  A model can nevertheless predict four
    or five active types; those extra types must survive parsing so evaluation
    counts them as false positives instead of treating valid per-type answers
    as a malformed record.
    """

    ordered = _ordered(degradations)
    if set(levels) != set(ordered):
        raise ValueError("levels keys must exactly equal active degradations")
    normalized: list[tuple[str, int]] = []
    for name in ordered:
        level = levels[name]
        if isinstance(level, bool) or not isinstance(level, int):
            raise ValueError(f"{name} level must be an integer")
        if level not in ACTIVE_LEVELS:
            raise ValueError(f"active {name} level must be L2 through L5")
        normalized.append((name, level))
    return DegradationLabel(ordered, tuple(normalized))


def canonical_target(
    label_or_degradations: DegradationLabel | Iterable[str],
    levels: Mapping[str, int] | None = None,
) -> str:
    """Serialize exactly one authoritative compact JSON target."""

    if isinstance(label_or_degradations, DegradationLabel):
        if levels is not None:
            raise ValueError("levels must be omitted when passing a label")
        label = label_or_degradations
        # Revalidate labels constructed outside ``make_label``.
        label = make_label(label.degradations, label.level_map)
    else:
        if levels is None:
            raise ValueError("levels are required")
        label = make_label(label_or_degradations, levels)
    payload = {
        "degradations": list(label.degradations),
        "levels": label.level_map,
    }
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=True)


def canonical_prediction_target(label: DegradationLabel) -> str:
    """Serialize a parsed prediction, including possible four/five-type errors."""

    label = make_prediction_label(label.degradations, label.level_map)
    return json.dumps(
        {
            "degradations": list(label.degradations),
            "levels": label.level_map,
        },
        separators=(",", ":"),
        ensure_ascii=True,
    )


def canonical_fixed_five_slot_target(label: DegradationLabel) -> str:
    """Serialize one fixed-length independent L1--L5 decision per type."""

    label = make_prediction_label(label.degradations, label.level_map)
    payload = {
        name: label.level_map.get(name, 1)
        for name in DEGRADATIONS
    }
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=True)


def parse_target(raw_output: str) -> DegradationLabel:
    """Strictly parse the frozen schema; reject aliases and extra prose."""

    if not isinstance(raw_output, str) or not raw_output.strip():
        raise ValueError("empty structured output")
    if raw_output != raw_output.strip():
        raise ValueError("leading or trailing whitespace is not canonical")
    try:
        payload = json.loads(raw_output)
    except json.JSONDecodeError as exc:
        raise ValueError("output is not exactly one JSON value") from exc
    if not isinstance(payload, dict) or list(payload) != ["degradations", "levels"]:
        raise ValueError("top-level keys or key order differ from the contract")
    names = payload["degradations"]
    levels = payload["levels"]
    if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
        raise ValueError("degradations must be a string array")
    if not isinstance(levels, dict):
        raise ValueError("levels must be an object")
    label = make_label(names, levels)
    if tuple(names) != label.degradations:
        raise ValueError("degradations are not in canonical order")
    if list(levels) != list(label.degradations):
        raise ValueError("levels keys are not in canonical order")
    if canonical_target(label) != raw_output:
        raise ValueError("JSON serialization is not canonical")
    return label


def parse_fixed_five_slot_target(raw_output: str) -> DegradationLabel:
    """Strictly parse the fixed five-slot L1-absent schema."""

    if not isinstance(raw_output, str) or not raw_output.strip():
        raise ValueError("empty structured output")
    if raw_output != raw_output.strip():
        raise ValueError("leading or trailing whitespace is not canonical")
    try:
        payload = json.loads(raw_output)
    except json.JSONDecodeError as exc:
        raise ValueError("output is not exactly one JSON value") from exc
    if not isinstance(payload, dict) or list(payload) != list(DEGRADATIONS):
        raise ValueError("fixed-slot keys or key order differ from the contract")
    for name, level in payload.items():
        if isinstance(level, bool) or not isinstance(level, int) or level not in range(1, 6):
            raise ValueError(f"fixed-slot {name} level must be L1 through L5")
    active = [name for name in DEGRADATIONS if payload[name] > 1]
    label = make_prediction_label(active, {name: payload[name] for name in active})
    if canonical_fixed_five_slot_target(label) != raw_output:
        raise ValueError("fixed-slot JSON serialization is not canonical")
    return label


def parse_output(raw_output: str, target_schema: str) -> DegradationLabel:
    if target_schema == VARIABLE_ACTIVE_SCHEMA:
        return parse_target(raw_output)
    if target_schema == FIXED_FIVE_SLOT_SCHEMA:
        return parse_fixed_five_slot_target(raw_output)
    raise ValueError(f"unknown Qwen target schema: {target_schema}")


def training_target(label: DegradationLabel, target_schema: str) -> str:
    if target_schema == VARIABLE_ACTIVE_SCHEMA:
        return canonical_target(label)
    if target_schema == FIXED_FIVE_SLOT_SCHEMA:
        return canonical_fixed_five_slot_target(label)
    raise ValueError(f"unknown Qwen target schema: {target_schema}")


def canonical_restoration_text(label: DegradationLabel) -> str:
    """Deterministically translate Qwen JSON to the later FLUX text condition."""

    label = make_prediction_label(label.degradations, label.level_map)
    if not label.degradations:
        return "Restore conservatively; the image is clean and has no active degradation."
    descriptions = [
        f"{name} at {LEVEL_WORDS[label.level_map[name]]} strength (L{label.level_map[name]})"
        for name in label.degradations
    ]
    return "Restore the image by correcting " + ", ".join(descriptions) + "."


def _validate_query_degradation(degradation: str) -> str:
    if degradation not in DEGRADATION_SET:
        raise ValueError(f"unknown query degradation: {degradation}")
    return degradation


def per_type_question(degradation: str) -> str:
    """Return the advisor-directed question for exactly one degradation."""

    degradation = _validate_query_degradation(degradation)
    return PER_TYPE_QUESTION_TEMPLATE.format(degradation=degradation)


def per_type_response(degradation: str, level: int) -> str:
    """Return one exact dialogue target without introducing classifier heads."""

    degradation = _validate_query_degradation(degradation)
    if isinstance(level, bool) or not isinstance(level, int) or level not in range(1, 6):
        raise ValueError("per-type response level must be L1 through L5")
    return PER_TYPE_RESPONSE_TEMPLATE.format(degradation=degradation, level=level)


def parse_per_type_response(raw_output: str, degradation: str) -> int:
    """Strictly parse one of the five permitted type-specific responses."""

    degradation = _validate_query_degradation(degradation)
    if not isinstance(raw_output, str) or raw_output != raw_output.strip():
        raise ValueError("per-type response has noncanonical whitespace")
    candidates = {
        per_type_response(degradation, level): level for level in range(1, 6)
    }
    if raw_output not in candidates:
        raise ValueError("per-type response differs from the required sentence")
    return candidates[raw_output]


def aggregate_per_type_responses(
    raw_outputs: Mapping[str, str],
) -> tuple[str | None, dict[str, dict[str, object]]]:
    """Aggregate five Qwen dialogue answers into the frozen five-slot JSON."""

    if list(raw_outputs) != list(DEGRADATIONS):
        raise ValueError("per-type responses must contain all five canonical queries")
    levels: dict[str, int] = {}
    details: dict[str, dict[str, object]] = {}
    all_valid = True
    for degradation in DEGRADATIONS:
        raw = raw_outputs[degradation]
        level: int | None = None
        error: str | None = None
        try:
            level = parse_per_type_response(raw, degradation)
            levels[degradation] = level
        except ValueError as exc:
            all_valid = False
            error = str(exc)
        details[degradation] = {
            "raw_output": raw,
            "predicted_level": level,
            "valid_response": level is not None,
            "parse_error": error,
        }
    if not all_valid:
        return None, details
    active = [name for name in DEGRADATIONS if levels[name] > 1]
    label = make_prediction_label(active, {name: levels[name] for name in active})
    return canonical_fixed_five_slot_target(label), details


def per_type_prompt_messages(degradation: str) -> list[dict[str, object]]:
    degradation = _validate_query_degradation(degradation)
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": PER_TYPE_SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": per_type_question(degradation)},
            ],
        },
    ]


def per_type_prompt_text(
    processor: "Qwen2_5_VLProcessor", degradation: str
) -> str:
    return processor.apply_chat_template(
        per_type_prompt_messages(degradation),
        tokenize=False,
        add_generation_prompt=True,
    )


def per_type_supervised_text(
    processor: "Qwen2_5_VLProcessor",
    label: DegradationLabel,
    degradation: str,
) -> str:
    degradation = _validate_query_degradation(degradation)
    label = make_label(label.degradations, label.level_map)
    messages = per_type_prompt_messages(degradation)
    messages.append(
        {
            "role": "assistant",
            "content": per_type_response(
                degradation, label.level_map.get(degradation, 1)
            ),
        }
    )
    return processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )


def prompt_messages(
    target_schema: str = VARIABLE_ACTIVE_SCHEMA,
) -> list[dict[str, object]]:
    question = (
        DEGRADATION_QUESTION
        if target_schema == VARIABLE_ACTIVE_SCHEMA
        else FIXED_FIVE_SLOT_QUESTION
        if target_schema == FIXED_FIVE_SLOT_SCHEMA
        else None
    )
    if question is None:
        raise ValueError(f"unknown Qwen target schema: {target_schema}")
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question},
            ],
        },
    ]


def prompt_text(
    processor: "Qwen2_5_VLProcessor",
    target_schema: str = VARIABLE_ACTIVE_SCHEMA,
) -> str:
    return processor.apply_chat_template(
        prompt_messages(target_schema), tokenize=False, add_generation_prompt=True
    )


def supervised_text(
    processor: "Qwen2_5_VLProcessor",
    label: DegradationLabel,
    target_schema: str = VARIABLE_ACTIVE_SCHEMA,
) -> str:
    messages = prompt_messages(target_schema)
    messages.append(
        {"role": "assistant", "content": training_target(label, target_schema)}
    )
    return processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )


def _expanded_prompt_ids(
    processor: "Qwen2_5_VLProcessor", text: str, image_grid_thw: torch.Tensor
) -> list[int]:
    if text.count(processor.image_token) != 1:
        raise ValueError("the fixed prompt must contain exactly one image token")
    merge_size = int(processor.image_processor.merge_size)
    image_tokens = int(image_grid_thw.prod().item()) // (merge_size**2)
    if image_tokens <= 0:
        raise ValueError("processor produced no image tokens")
    expanded = text.replace(processor.image_token, processor.image_token * image_tokens)
    return list(processor.tokenizer(expanded, add_special_tokens=False)["input_ids"])


def prepare_supervised_batch(
    processor: "Qwen2_5_VLProcessor",
    labels: Sequence[DegradationLabel],
    images: Sequence[Image.Image],
    target_schema: str = VARIABLE_ACTIVE_SCHEMA,
) -> tuple[dict[str, torch.Tensor], list[int]]:
    """Build a batch whose loss covers assistant answer tokens only."""

    if not labels or len(labels) != len(images):
        raise ValueError("labels and images must have the same nonzero length")
    if processor.tokenizer.padding_side != "right":
        raise ValueError("supervised batching requires right padding")
    prefix = prompt_text(processor, target_schema)
    encoded = processor(
        text=[supervised_text(processor, label, target_schema) for label in labels],
        images=list(images),
        padding=True,
        return_tensors="pt",
    )
    required = {"input_ids", "attention_mask", "pixel_values", "image_grid_thw"}
    if not required.issubset(encoded):
        raise RuntimeError(f"processor omitted inputs: {sorted(required - set(encoded))}")
    output = {
        key: value for key, value in encoded.items() if isinstance(value, torch.Tensor)
    }
    supervised = output["input_ids"].clone()
    supervised[output["attention_mask"] == 0] = -100
    counts: list[int] = []
    for row, grid in enumerate(output["image_grid_thw"]):
        prefix_ids = _expanded_prompt_ids(processor, prefix, grid)
        if output["input_ids"][row, : len(prefix_ids)].tolist() != prefix_ids:
            raise RuntimeError("assistant-loss boundary is not an exact token prefix")
        supervised[row, : len(prefix_ids)] = -100
        count = int((supervised[row] != -100).sum().item())
        if count <= 0:
            raise RuntimeError("assistant target produced no supervised tokens")
        counts.append(count)
    output["labels"] = supervised
    return output, counts


def prepare_per_type_supervised_batch(
    processor: "Qwen2_5_VLProcessor",
    labels: Sequence[DegradationLabel],
    images: Sequence[Image.Image],
    query_degradations: Sequence[str],
) -> tuple[dict[str, torch.Tensor], list[int]]:
    """Build five-query dialogue batches with assistant-only CE supervision."""

    if (
        not labels
        or len(labels) != len(images)
        or len(labels) != len(query_degradations)
    ):
        raise ValueError(
            "labels, images, and query degradations must have the same nonzero length"
        )
    if processor.tokenizer.padding_side != "right":
        raise ValueError("supervised batching requires right padding")
    queries = [_validate_query_degradation(name) for name in query_degradations]
    prefixes = [
        per_type_prompt_text(processor, degradation) for degradation in queries
    ]
    encoded = processor(
        text=[
            per_type_supervised_text(processor, label, degradation)
            for label, degradation in zip(labels, queries, strict=True)
        ],
        images=list(images),
        padding=True,
        return_tensors="pt",
    )
    required = {"input_ids", "attention_mask", "pixel_values", "image_grid_thw"}
    if not required.issubset(encoded):
        raise RuntimeError(f"processor omitted inputs: {sorted(required - set(encoded))}")
    output = {
        key: value for key, value in encoded.items() if isinstance(value, torch.Tensor)
    }
    supervised = output["input_ids"].clone()
    supervised[output["attention_mask"] == 0] = -100
    counts: list[int] = []
    for row, (grid, prefix) in enumerate(
        zip(output["image_grid_thw"], prefixes, strict=True)
    ):
        prefix_ids = _expanded_prompt_ids(processor, prefix, grid)
        if output["input_ids"][row, : len(prefix_ids)].tolist() != prefix_ids:
            raise RuntimeError("per-type assistant-loss boundary is not an exact prefix")
        supervised[row, : len(prefix_ids)] = -100
        count = int((supervised[row] != -100).sum().item())
        if count <= 0:
            raise RuntimeError("per-type assistant target produced no supervised tokens")
        counts.append(count)
    output["labels"] = supervised
    return output, counts


def _safe_div(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def evaluate_outputs(
    truths: Sequence[DegradationLabel],
    raw_outputs: Sequence[str],
    *,
    target_schema: str = VARIABLE_ACTIVE_SCHEMA,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Evaluate strict structure, multi-label types, and independent severity."""

    if not truths or len(truths) != len(raw_outputs):
        raise ValueError("truths and outputs must have the same nonzero length")
    type_counts = {
        name: Counter({"tp": 0, "fp": 0, "fn": 0, "tn": 0})
        for name in DEGRADATIONS
    }
    rows: list[dict[str, object]] = []
    valid_count = set_exact_count = label_exact_count = 0
    severity_correct = severity_error = severity_support = 0
    overlap_correct = overlap_error = overlap_support = 0
    composition = {
        size: Counter({"samples": 0, "valid": 0, "set_exact": 0, "label_exact": 0})
        for size in range(4)
    }

    for truth_input, raw in zip(truths, raw_outputs, strict=True):
        truth = make_label(truth_input.degradations, truth_input.level_map)
        predicted: DegradationLabel | None = None
        error: str | None = None
        try:
            predicted = parse_output(raw, target_schema)
        except ValueError as exc:
            error = str(exc)
        true_set = set(truth.degradations)
        predicted_set = set(predicted.degradations) if predicted else set()
        valid = predicted is not None
        set_exact = valid and true_set == predicted_set
        label_exact = valid and predicted == truth
        valid_count += int(valid)
        set_exact_count += int(set_exact)
        label_exact_count += int(label_exact)
        stratum = composition[truth.composition_size]
        stratum["samples"] += 1
        stratum["valid"] += int(valid)
        stratum["set_exact"] += int(set_exact)
        stratum["label_exact"] += int(label_exact)

        for name in DEGRADATIONS:
            present = name in true_set
            predicted_present = name in predicted_set
            key = "tp" if present and predicted_present else (
                "fp" if predicted_present else ("fn" if present else "tn")
            )
            type_counts[name][key] += 1

        predicted_levels = predicted.level_map if predicted else {}
        component_results: dict[str, object] = {}
        for name in truth.degradations:
            true_level = truth.level_map[name]
            predicted_level = predicted_levels.get(name)
            correct = predicted_level == true_level
            penalty = abs(predicted_level - true_level) if predicted_level else 3
            severity_support += 1
            severity_correct += int(correct)
            severity_error += penalty
            if predicted_level is not None:
                overlap_support += 1
                overlap_correct += int(correct)
                overlap_error += abs(predicted_level - true_level)
            component_results[name] = {
                "true_level": true_level,
                "predicted_level": predicted_level,
                "correct": correct,
            }
        rows.append(
            {
                "target": canonical_target(truth),
                "raw_output": raw,
                "predicted": canonical_prediction_target(predicted) if predicted else None,
                "valid_structured_output": valid,
                "degradation_set_exact": set_exact,
                "combined_label_exact": label_exact,
                "components": component_results,
                "parse_error": error,
            }
        )

    per_type: dict[str, object] = {}
    for name, counts in type_counts.items():
        precision = _safe_div(counts["tp"], counts["tp"] + counts["fp"])
        recall = _safe_div(counts["tp"], counts["tp"] + counts["fn"])
        f1 = _safe_div(2 * counts["tp"], 2 * counts["tp"] + counts["fp"] + counts["fn"])
        per_type[name] = {**dict(counts), "precision": precision, "recall": recall, "f1": f1}
    aggregate = Counter()
    for counts in type_counts.values():
        aggregate.update(counts)
    total = len(truths)
    metrics: dict[str, object] = {
        "samples": total,
        "valid_structured_output_rate": valid_count / total,
        "degradation_set_exact_match_rate": set_exact_count / total,
        "combined_label_exact_match_rate": label_exact_count / total,
        "type_micro_precision": _safe_div(aggregate["tp"], aggregate["tp"] + aggregate["fp"]),
        "type_micro_recall": _safe_div(aggregate["tp"], aggregate["tp"] + aggregate["fn"]),
        "type_micro_f1": _safe_div(2 * aggregate["tp"], 2 * aggregate["tp"] + aggregate["fp"] + aggregate["fn"]),
        "type_macro_f1": sum(float(value["f1"]) for value in per_type.values()) / len(DEGRADATIONS),
        "severity_typed_accuracy": _safe_div(severity_correct, severity_support),
        "severity_typed_mae_with_missing_penalty": _safe_div(severity_error, severity_support),
        "severity_overlap_only_accuracy": _safe_div(overlap_correct, overlap_support),
        "severity_overlap_only_mae": _safe_div(overlap_error, overlap_support),
        "active_severity_support": severity_support,
        "per_type": per_type,
        "by_composition_size": {
            str(size): {
                "samples": counts["samples"],
                "valid_structured_output_rate": _safe_div(counts["valid"], counts["samples"]),
                "degradation_set_exact_match_rate": _safe_div(counts["set_exact"], counts["samples"]),
                "combined_label_exact_match_rate": _safe_div(counts["label_exact"], counts["samples"]),
            }
            for size, counts in composition.items()
        },
    }
    return rows, metrics
