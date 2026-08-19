"""Supervised Qwen2.5-VL Lowlight data, prompt, and metric helpers.

The advisor-directed target differs from the historical scalar-condition
prototype: project L1 is a clean negative with no active degradation, while
L2--L5 contain one active ``low_light`` degradation and its independent level.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from typing import TYPE_CHECKING, Sequence

import torch
from PIL import Image

from .data import PairRecord

if TYPE_CHECKING:
    from transformers import Qwen2_5_VLProcessor


SYSTEM_PROMPT = (
    "You are a degradation-assessment model. Analyze only the requested image "
    "degradation and return exactly one compact JSON object with no markdown "
    "or extra text."
)

DEGRADATION_QUESTION = (
    "Assess low-light degradation in this image using project levels L1 through L5. "
    "L1 means clean or normally exposed with no active degradation; L2 means mild; "
    "L3 means moderate; L4 means strong; L5 means extreme. Return exactly the keys "
    "degradations and levels. For L1, return an empty degradations array and empty "
    "levels object. For L2 through L5, return low_light in degradations and its "
    "integer project level in levels."
)

VALID_LEVELS = frozenset(range(1, 6))


def canonical_target(level: int) -> str:
    """Return the one deterministic supervision string for a project level."""

    if isinstance(level, bool) or level not in VALID_LEVELS:
        raise ValueError(f"project level must be an integer from 1 to 5: {level!r}")
    if level == 1:
        payload: dict[str, object] = {"degradations": [], "levels": {}}
    else:
        payload = {
            "degradations": ["low_light"],
            "levels": {"low_light": level},
        }
    return json.dumps(payload, separators=(",", ":"), sort_keys=True)


def parse_target(raw_output: str) -> int:
    """Parse only the authoritative Lowlight schema and return L1--L5."""

    if not isinstance(raw_output, str) or not raw_output.strip():
        raise ValueError("empty structured output")
    try:
        payload = json.loads(raw_output.strip())
    except json.JSONDecodeError as exc:
        raise ValueError("output is not one JSON value") from exc
    if not isinstance(payload, dict) or set(payload) != {"degradations", "levels"}:
        raise ValueError("unexpected top-level fields")
    if payload == {"degradations": [], "levels": {}}:
        return 1
    if payload.get("degradations") != ["low_light"]:
        raise ValueError("active degradation list must be exactly ['low_light']")
    levels = payload.get("levels")
    if not isinstance(levels, dict) or set(levels) != {"low_light"}:
        raise ValueError("active level map must contain only low_light")
    level = levels["low_light"]
    if (
        isinstance(level, bool)
        or not isinstance(level, int)
        or level not in {2, 3, 4, 5}
    ):
        raise ValueError("active low_light level must be an integer from 2 to 5")
    return level


def prompt_messages() -> list[dict[str, object]]:
    """Return a fresh copy of the fixed image-plus-question conversation."""

    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": DEGRADATION_QUESTION},
            ],
        },
    ]


def prompt_text(processor: "Qwen2_5_VLProcessor") -> str:
    return processor.apply_chat_template(
        prompt_messages(), tokenize=False, add_generation_prompt=True
    )


def supervised_text(processor: "Qwen2_5_VLProcessor", level: int) -> str:
    messages = prompt_messages()
    messages.append({"role": "assistant", "content": canonical_target(level)})
    return processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )


def select_complete_sources(
    records: Sequence[PairRecord],
    *,
    split: str,
    source_count: int,
    seed: int,
) -> list[PairRecord]:
    """Select deterministic sources that each contain exactly L1--L5."""

    if source_count <= 0:
        raise ValueError("source_count must be positive")
    by_source: dict[str, list[PairRecord]] = defaultdict(list)
    for record in records:
        if record.split == split:
            by_source[record.source_id].append(record)
    complete: list[tuple[str, list[PairRecord]]] = []
    for source_id, source_records in by_source.items():
        by_level = {record.level_index: record for record in source_records}
        if len(by_level) != len(source_records):
            raise ValueError(f"duplicate project level for source {source_id}")
        if set(by_level) == VALID_LEVELS:
            complete.append(
                (source_id, [by_level[level] for level in sorted(VALID_LEVELS)])
            )
    if len(complete) < source_count:
        raise ValueError(
            f"only {len(complete)} complete sources in {split}; need {source_count}"
        )
    complete.sort(
        key=lambda item: hashlib.sha256(f"{seed}:{item[0]}".encode("utf-8")).hexdigest()
    )
    chosen = complete[:source_count]
    return [record for _, source_records in chosen for record in source_records]


def record_bundle_sha256(records: Sequence[PairRecord]) -> str:
    payload = [
        {
            "degraded_sha256": record.degraded_sha256,
            "level": record.level_index,
            "pair_id": record.pair_id,
            "source_id": record.source_id,
            "target": canonical_target(record.level_index),
        }
        for record in records
    ]
    encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _expanded_prompt_ids(
    processor: "Qwen2_5_VLProcessor",
    text: str,
    image_grid_thw: torch.Tensor,
) -> list[int]:
    """Mirror the public processor's one-image placeholder expansion."""

    if text.count(processor.image_token) != 1:
        raise ValueError("the fixed prompt must contain exactly one image token")
    merge_size = int(processor.image_processor.merge_size)
    image_tokens = int(image_grid_thw.prod().item()) // (merge_size**2)
    if image_tokens <= 0:
        raise ValueError("processor produced no image tokens")
    expanded = text.replace(processor.image_token, processor.image_token * image_tokens)
    encoded = processor.tokenizer(expanded, add_special_tokens=False)
    return list(encoded["input_ids"])


def prepare_supervised_batch(
    processor: "Qwen2_5_VLProcessor",
    records: Sequence[PairRecord],
    images: Sequence[Image.Image],
) -> tuple[dict[str, torch.Tensor], list[int]]:
    """Create teacher-forced inputs with loss masked to assistant tokens only."""

    if not records or len(records) != len(images):
        raise ValueError("records and images must have the same nonzero length")
    if processor.tokenizer.padding_side != "right":
        raise ValueError("supervised batching requires right padding")
    fixed_prompt = prompt_text(processor)
    full_texts = [supervised_text(processor, record.level_index) for record in records]
    encoded = processor(
        text=full_texts,
        images=list(images),
        padding=True,
        return_tensors="pt",
    )
    required = {"input_ids", "attention_mask", "pixel_values", "image_grid_thw"}
    if not required.issubset(encoded):
        raise RuntimeError(
            f"processor omitted inputs: {sorted(required - set(encoded))}"
        )
    if encoded.input_ids.shape[0] != len(records) or encoded.image_grid_thw.shape[
        0
    ] != len(records):
        raise RuntimeError("processor batch cardinality mismatch")

    labels = encoded.input_ids.clone()
    labels[encoded.attention_mask == 0] = -100
    target_token_counts: list[int] = []
    for row, grid in enumerate(encoded.image_grid_thw):
        prefix_ids = _expanded_prompt_ids(processor, fixed_prompt, grid)
        realized = encoded.input_ids[row, : len(prefix_ids)].tolist()
        if realized != prefix_ids:
            raise RuntimeError("assistant-loss boundary is not an exact token prefix")
        labels[row, : len(prefix_ids)] = -100
        target_count = int((labels[row] != -100).sum().item())
        if target_count <= 0:
            raise RuntimeError("assistant target produced no supervised tokens")
        target_token_counts.append(target_count)
    batch = {
        key: value for key, value in encoded.items() if isinstance(value, torch.Tensor)
    }
    batch["labels"] = labels
    return batch, target_token_counts


def open_degraded_images(records: Sequence[PairRecord]) -> list[Image.Image]:
    images: list[Image.Image] = []
    for record in records:
        if not record.degraded_path.is_file():
            raise FileNotFoundError(record.degraded_path)
        with Image.open(record.degraded_path) as handle:
            images.append(handle.convert("RGB"))
    return images


def evaluate_generated_outputs(
    records: Sequence[PairRecord], raw_outputs: Sequence[str]
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Evaluate strict structure, type, severity, and canonical serialization."""

    if not records or len(records) != len(raw_outputs):
        raise ValueError("records and outputs must have the same nonzero length")
    rows: list[dict[str, object]] = []
    confusion: Counter[tuple[int, int]] = Counter()
    valid_count = canonical_count = type_correct = severity_correct = 0
    absolute_error = 0
    support: Counter[int] = Counter()
    correct: Counter[int] = Counter()
    for record, raw_output in zip(records, raw_outputs, strict=True):
        true_level = record.level_index
        support[true_level] += 1
        predicted_level: int | None = None
        error: str | None = None
        try:
            predicted_level = parse_target(raw_output)
        except ValueError as exc:
            error = str(exc)
        valid = predicted_level is not None
        canonical = raw_output.strip() == canonical_target(true_level)
        expected_active = true_level > 1
        predicted_active = predicted_level is not None and predicted_level > 1
        type_match = valid and predicted_active == expected_active
        severity_match = valid and predicted_level == true_level
        valid_count += int(valid)
        canonical_count += int(canonical)
        type_correct += int(type_match)
        severity_correct += int(severity_match)
        if predicted_level is not None:
            confusion[(true_level, predicted_level)] += 1
            absolute_error += abs(predicted_level - true_level)
            correct[true_level] += int(predicted_level == true_level)
        else:
            absolute_error += 4
        rows.append(
            {
                "pair_id": record.pair_id,
                "source_id": record.source_id,
                "true_level": true_level,
                "target": canonical_target(true_level),
                "raw_output": raw_output,
                "predicted_level": predicted_level,
                "valid_structured_output": valid,
                "canonical_exact_match": canonical,
                "degradation_type_correct": type_match,
                "severity_correct": severity_match,
                "parse_error": error,
            }
        )
    total = len(records)
    metrics: dict[str, object] = {
        "samples": total,
        "valid_structured_output_rate": valid_count / total,
        "canonical_exact_match_rate": canonical_count / total,
        "degradation_type_accuracy": type_correct / total,
        "severity_accuracy": severity_correct / total,
        "combined_label_exact_match_accuracy": severity_correct / total,
        "severity_mean_absolute_error": absolute_error / total,
        "per_level": {
            str(level): {
                "support": support[level],
                "correct": correct[level],
                "accuracy": correct[level] / support[level] if support[level] else 0.0,
            }
            for level in sorted(VALID_LEVELS)
        },
        "confusion": {
            str(true): {
                str(predicted): confusion[(true, predicted)]
                for predicted in sorted(VALID_LEVELS)
            }
            for true in sorted(VALID_LEVELS)
        },
    }
    return rows, metrics


def parameter_group(name: str) -> str:
    """Map current Transformers parameter names to the audited module scope."""

    if name.startswith("model.visual.patch_embed."):
        return "visual.patch_embed"
    if name.startswith("model.visual.blocks."):
        return "visual.blocks"
    if name.startswith("model.visual.merger."):
        return "visual.merger"
    if name.startswith("model.language_model."):
        return "language_model"
    if name.startswith("lm_head."):
        return "lm_head"
    return "unexpected"


def count_parameter_scope(
    named_parameters: Sequence[tuple[str, torch.nn.Parameter]],
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for name, parameter in named_parameters:
        counts[parameter_group(name)] += parameter.numel()
    return dict(counts)
