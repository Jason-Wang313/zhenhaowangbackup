"""Pure contracts, schedules, and metrics for the supervised Qwen 8k pilot."""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Sequence

from .data import PairRecord
from .qwen_distributed import complete_source_groups
from .qwen_supervised import record_bundle_sha256


APPROVED_STATUS = (
    "approved_for_advisor_route_supervised_vlm_lowlight_8k_ddp8_pilot_only"
)
REQUIRED_SPLITS = ("train", "validation", "test")
WAIVER_TEXT = "skip scaling, prepare 8k"
WAIVER_SHA256 = hashlib.sha256(WAIVER_TEXT.encode("utf-8")).hexdigest()


def load_pilot_config(path: Path) -> tuple[dict[str, object], str, bytes]:
    raw = path.read_bytes()
    config = json.loads(raw)
    if not isinstance(config, dict) or config.get("schema_version") != 1:
        raise ValueError("bad supervised Qwen pilot config schema")
    if config.get("status") != APPROVED_STATUS:
        raise ValueError("config is not approved for the supervised Qwen pilot")
    return config, hashlib.sha256(raw).hexdigest(), raw


def validate_pilot_contract(config: dict[str, object]) -> None:
    required_sections = (
        "scaling_waiver",
        "model",
        "processor",
        "prompt",
        "data",
        "training",
        "distributed",
        "generation",
        "gate",
        "runtime",
        "telemetry",
    )
    sections = {name: config.get(name) for name in required_sections}
    if any(not isinstance(value, dict) for value in sections.values()):
        raise ValueError("supervised pilot config is missing a required section")
    waiver = sections["scaling_waiver"]
    model = sections["model"]
    processor = sections["processor"]
    data = sections["data"]
    training = sections["training"]
    distributed = sections["distributed"]
    generation = sections["generation"]
    gate = sections["gate"]
    runtime = sections["runtime"]
    telemetry = sections["telemetry"]
    assert all(
        isinstance(section, dict)
        for section in (
            waiver,
            model,
            processor,
            data,
            training,
            distributed,
            generation,
            gate,
            runtime,
            telemetry,
        )
    )

    if (
        waiver.get("formal_1_2_4_8_scaling_gate_skipped_by_user") is not True
        or waiver.get("user_instruction") != WAIVER_TEXT
        or waiver.get("user_instruction_sha256") != WAIVER_SHA256
        or waiver.get("selected_world_size") != 8
    ):
        raise ValueError("the explicit scaling waiver changed")
    for name in (
        "operational_config",
        "operational_config_sha256",
        "operational_evidence",
        "operational_evidence_sha256",
        "operational_final_visual_checkpoint",
        "operational_final_visual_checkpoint_sha256",
    ):
        if not isinstance(waiver.get(name), str) or not waiver[name]:
            raise ValueError(f"the DDP8 operational evidence is missing: {name}")
    throughput = waiver.get("observed_ddp8_global_samples_per_second")
    if (
        isinstance(throughput, bool)
        or not isinstance(throughput, (int, float))
        or not math.isfinite(float(throughput))
        or float(throughput) <= 0.0
    ):
        raise ValueError("the DDP8 throughput evidence is invalid")
    if model.get("architecture") != "Qwen2_5_VLForConditionalGeneration":
        raise ValueError("unexpected Qwen architecture")
    if model.get("trainable_module") != "model.visual" or model.get(
        "frozen_modules"
    ) != ["model.language_model", "lm_head"]:
        raise ValueError("Qwen trainable/frozen scope changed")
    groups = model.get("expected_parameter_groups")
    if not isinstance(groups, dict):
        raise ValueError("Qwen parameter-group contract is missing")
    visual = sum(
        int(groups.get(name, 0))
        for name in ("visual.patch_embed", "visual.blocks", "visual.merger")
    )
    frozen = sum(
        int(groups.get(name, 0)) for name in ("language_model", "lm_head")
    )
    if (
        visual != model.get("expected_trainable_parameters")
        or frozen != model.get("expected_frozen_parameters")
        or visual + frozen != model.get("expected_total_parameters")
    ):
        raise ValueError("Qwen parameter counts changed")
    if model.get("dtype") != "bfloat16" or model.get(
        "attention_implementation"
    ) != "sdpa":
        raise ValueError("Qwen dtype or attention implementation changed")
    if (
        processor.get("use_fast") is not False
        or processor.get("padding_side") != "right"
        or processor.get("min_pixels") != processor.get("max_pixels")
    ):
        raise ValueError("Qwen processor contract changed")

    splits = data.get("splits")
    if not isinstance(splits, dict) or tuple(splits) != REQUIRED_SPLITS:
        raise ValueError("pilot splits must be train, validation, and test")
    expected_split_counts = {
        "train": (8000, 1600, 1600),
        "validation": (1000, 200, 200),
        "test": (1000, 200, 200),
    }
    for split, expected in expected_split_counts.items():
        contract = splits.get(split)
        if not isinstance(contract, dict):
            raise ValueError(f"missing pilot split contract: {split}")
        realized = (
            contract.get("records"),
            contract.get("sources"),
            contract.get("records_per_level"),
        )
        if realized != expected or not isinstance(contract.get("bundle_sha256"), str):
            raise ValueError(f"pilot split cardinality changed: {split}")
    if data.get("source_disjoint") is not True or data.get(
        "test_never_used_for_training_or_selection"
    ) is not True:
        raise ValueError("source-disjoint/test protection was disabled")

    world_size = int(distributed.get("required_world_size", 0))
    if world_size != 8:
        raise ValueError("the waived pilot must use all eight H200s")
    expected_distributed = {
        "backend": "nccl",
        "required_world_size": 8,
        "keep_source_levels_in_one_microbatch": True,
        "broadcast_buffers": False,
        "ddp_init_sync": False,
        "gradient_as_bucket_view": True,
        "find_unused_parameters": False,
        "timeout_minutes": 30,
    }
    if distributed != expected_distributed:
        raise ValueError("distributed pilot contract changed")
    micro_batch = int(training.get("micro_batch_size_per_rank", 0))
    accumulation = int(training.get("gradient_accumulation_steps", 0))
    global_batch = int(training.get("global_batch_size", 0))
    if micro_batch != 5 or accumulation != 1:
        raise ValueError("each rank must process one complete L1--L5 source")
    if global_batch != micro_batch * accumulation * world_size:
        raise ValueError("pilot global batch is inconsistent")
    train_sources = int(splits["train"]["sources"])
    steps_per_epoch = int(training.get("steps_per_epoch", 0))
    epochs = int(training.get("epochs", 0))
    total_steps = int(training.get("total_steps", 0))
    if (
        training.get("source_groups_per_global_step") != world_size
        or steps_per_epoch != train_sources // world_size
        or epochs != 5
        or total_steps != steps_per_epoch * epochs
        or training.get("total_sample_presentations") != total_steps * global_batch
    ):
        raise ValueError("pilot epoch/step/sample semantics changed")
    if (
        training.get("fixed_source_shard_per_rank") is not True
        or training.get("shuffle_each_rank_source_shard_each_epoch") is not True
        or training.get("learning_rate_schedule")
        != "linear_warmup_cosine_decay"
        or training.get("gradient_checkpointing") is not True
        or training.get("fused_adamw") is not True
    ):
        raise ValueError("pilot training mechanics changed")
    warmup = int(training.get("warmup_steps", -1))
    if not 0 < warmup < total_steps:
        raise ValueError("pilot warmup steps are invalid")
    checkpoints = training.get("checkpoint_steps")
    validation_steps = training.get("validation_loss_steps")
    if (
        not isinstance(checkpoints, list)
        or checkpoints != sorted(set(checkpoints))
        or checkpoints[-1] != total_steps
        or any(not isinstance(step, int) or not 0 < step <= total_steps for step in checkpoints)
    ):
        raise ValueError("pilot checkpoint schedule changed")
    expected_validation = [steps_per_epoch * epoch for epoch in range(epochs + 1)]
    if validation_steps != expected_validation:
        raise ValueError("validation-loss schedule must cover step zero and every epoch")
    if (
        training.get("checkpoint_selection_metric")
        != "minimum_validation_teacher_forced_loss"
        or training.get("checkpoint_selection_tiebreaker") != "earlier_step"
    ):
        raise ValueError("validation-only checkpoint selection changed")

    if generation.get("do_sample") is not False or generation.get(
        "evaluate_validation_once_after_selection"
    ) is not True or generation.get("evaluate_test_once_after_selection") is not True:
        raise ValueError("pilot generation/evaluation contract changed")
    for name in (
        "minimum_valid_structured_output_rate",
        "minimum_degradation_type_accuracy",
        "minimum_degradation_type_macro_f1",
        "minimum_severity_accuracy",
        "minimum_severity_macro_f1",
        "minimum_per_level_accuracy",
    ):
        value = float(gate.get(name, -1.0))
        if not 0.0 < value <= 1.0:
            raise ValueError(f"bad pilot gate threshold: {name}")
    if not 0.0 <= float(gate.get("maximum_severity_mean_absolute_error", -1.0)) < 1.0:
        raise ValueError("bad severity-MAE threshold")
    if not 0.0 < float(
        gate.get("maximum_best_to_initial_validation_loss_ratio", -1.0)
    ) < 1.0:
        raise ValueError("bad validation-loss decrease threshold")
    for name in (
        "require_gradient_for_every_trainable_parameter",
        "require_no_frozen_gradients",
        "require_no_frozen_updates",
        "require_all_checkpoint_freeze_audits",
        "require_validation_and_test_thresholds",
    ):
        if gate.get(name) is not True:
            raise ValueError(f"mandatory pilot gate was disabled: {name}")
    if (
        runtime.get("offline") is not True
        or runtime.get("device") != "cuda"
        or runtime.get("required_visible_device_count") != world_size
        or runtime.get("deterministic_algorithms") is not True
        or runtime.get("sdpa_backend") != "math"
    ):
        raise ValueError("pilot runtime contract changed")
    if (
        telemetry.get("sample_interval_seconds") != 2
        or telemetry.get("monitor_all_selected_gpus") is not True
        or not isinstance(telemetry.get("csv"), str)
    ):
        raise ValueError("pilot telemetry contract changed")


def records_by_split(
    records: Sequence[PairRecord], config: dict[str, object]
) -> dict[str, list[PairRecord]]:
    data = config["data"]
    assert isinstance(data, dict)
    contracts = data["splits"]
    assert isinstance(contracts, dict)
    result: dict[str, list[PairRecord]] = {}
    source_sets: dict[str, set[str]] = {}
    for split in REQUIRED_SPLITS:
        selected = [record for record in records if record.split == split]
        contract = contracts[split]
        assert isinstance(contract, dict)
        groups = complete_source_groups(selected)
        level_counts = Counter(record.level_index for record in selected)
        expected_per_level = int(contract["records_per_level"])
        if (
            len(selected) != int(contract["records"])
            or len(groups) != int(contract["sources"])
            or level_counts != Counter({level: expected_per_level for level in range(1, 6)})
            or record_bundle_sha256(selected) != contract["bundle_sha256"]
        ):
            raise RuntimeError(f"pilot split evidence changed: {split}")
        result[split] = selected
        source_sets[split] = {record.source_id for record in selected}
    if any(
        source_sets[left] & source_sets[right]
        for index, left in enumerate(REQUIRED_SPLITS)
        for right in REQUIRED_SPLITS[index + 1 :]
    ):
        raise RuntimeError("pilot source split leakage detected")
    return result


def epoch_local_order(
    *, local_source_count: int, seed: int, epoch_index: int, rank: int
) -> list[int]:
    if local_source_count <= 0 or seed < 0 or epoch_index < 0 or rank < 0:
        raise ValueError("bad epoch-order argument")
    order = list(range(local_source_count))
    derived_seed = seed + epoch_index * 1009 + rank * 1000003
    random.Random(derived_seed).shuffle(order)
    return order


def training_position(step: int, *, steps_per_epoch: int) -> tuple[int, int]:
    if step <= 0 or steps_per_epoch <= 0:
        raise ValueError("training step and steps per epoch must be positive")
    zero_based = step - 1
    return zero_based // steps_per_epoch, zero_based % steps_per_epoch


def select_best_validation_checkpoint(
    entries: Sequence[dict[str, object]], *, allowed_steps: Sequence[int]
) -> dict[str, object]:
    allowed = set(allowed_steps)
    valid: list[dict[str, object]] = []
    seen: set[int] = set()
    for entry in entries:
        step = entry.get("step")
        loss = entry.get("teacher_forced_loss")
        if (
            isinstance(step, bool)
            or not isinstance(step, int)
            or step <= 0
            or step not in allowed
            or step in seen
            or isinstance(loss, bool)
            or not isinstance(loss, (int, float))
            or not math.isfinite(float(loss))
            or float(loss) < 0.0
        ):
            raise ValueError("invalid validation checkpoint evidence")
        seen.add(step)
        valid.append(entry)
    if seen != allowed:
        raise ValueError("validation evidence does not cover every selectable checkpoint")
    return min(valid, key=lambda entry: (float(entry["teacher_forced_loss"]), int(entry["step"])))


def _class_f1(rows: Sequence[dict[str, object]], class_value: int) -> float:
    true_positive = false_positive = false_negative = 0
    for row in rows:
        true_level = int(row["true_level"])
        predicted = row.get("predicted_level")
        true_positive += int(true_level == class_value and predicted == class_value)
        false_positive += int(true_level != class_value and predicted == class_value)
        false_negative += int(true_level == class_value and predicted != class_value)
    denominator = 2 * true_positive + false_positive + false_negative
    return 2 * true_positive / denominator if denominator else 0.0


def augment_understanding_metrics(
    prediction_rows: Sequence[dict[str, object]], base_metrics: dict[str, object]
) -> dict[str, object]:
    if not prediction_rows or int(base_metrics.get("samples", 0)) != len(prediction_rows):
        raise ValueError("prediction rows and base metrics differ")
    result = dict(base_metrics)
    type_rows = [
        {
            **row,
            "true_level": int(int(row["true_level"]) > 1),
            "predicted_level": (
                None
                if row.get("predicted_level") is None
                else int(int(row["predicted_level"]) > 1)
            ),
        }
        for row in prediction_rows
    ]
    type_f1 = {"clean": _class_f1(type_rows, 0), "low_light": _class_f1(type_rows, 1)}
    severity_f1 = {str(level): _class_f1(prediction_rows, level) for level in range(1, 6)}
    per_level_mae: dict[str, float] = {}
    for level in range(1, 6):
        rows = [row for row in prediction_rows if int(row["true_level"]) == level]
        if not rows:
            raise ValueError(f"prediction rows omit severity level {level}")
        errors = [
            4
            if row.get("predicted_level") is None
            else abs(int(row["predicted_level"]) - level)
            for row in rows
        ]
        per_level_mae[str(level)] = sum(errors) / len(errors)
    result["degradation_type_f1"] = type_f1
    result["degradation_type_macro_f1"] = sum(type_f1.values()) / len(type_f1)
    result["severity_f1"] = severity_f1
    result["severity_macro_f1"] = sum(severity_f1.values()) / len(severity_f1)
    result["per_level_mean_absolute_error"] = per_level_mae
    return result


def metric_gate_checks(
    metrics: dict[str, object], gate: dict[str, object], *, prefix: str
) -> dict[str, bool]:
    per_level = metrics.get("per_level")
    if not isinstance(per_level, dict) or set(per_level) != {str(level) for level in range(1, 6)}:
        raise ValueError("metrics have incomplete per-level evidence")
    if any(not isinstance(value, dict) for value in per_level.values()):
        raise ValueError("metrics have malformed per-level evidence")
    minimum_per_level = min(
        float(value["accuracy"])
        for value in per_level.values()
    )
    return {
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
        f"{prefix}_severity_mae_passed": float(
            metrics["severity_mean_absolute_error"]
        )
        <= float(gate["maximum_severity_mean_absolute_error"]),
        f"{prefix}_every_level_accuracy_passed": minimum_per_level
        >= float(gate["minimum_per_level_accuracy"]),
    }
