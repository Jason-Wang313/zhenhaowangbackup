"""Pure validation and selection helpers for Qwen GPU scaling evidence."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Sequence

from .qwen_distributed import validate_full_batch_contract


APPROVED_STATUS = "approved_for_qwen_lowlight_supervised_1_2_4_8_scaling_gate_only"
REQUIRED_WORLD_SIZES = (1, 2, 4, 8)


def load_scaling_config(path: Path) -> tuple[dict[str, object], str, bytes]:
    raw = path.read_bytes()
    config = json.loads(raw)
    if not isinstance(config, dict) or config.get("schema_version") != 1:
        raise ValueError("bad Qwen scaling config schema")
    if config.get("status") != APPROVED_STATUS:
        raise ValueError("config is not approved for the Qwen scaling gate")
    digest = hashlib.sha256(raw).hexdigest()
    return config, digest, raw


def expected_accumulation_steps(*, source_count: int, world_size: int) -> int:
    if world_size <= 0 or source_count <= 0 or source_count % world_size:
        raise ValueError("source count must be positive and divisible by world size")
    return source_count // world_size


def validate_scaling_contract(config: dict[str, object]) -> None:
    data = config.get("data")
    training = config.get("training")
    distributed = config.get("distributed")
    selection = config.get("selection")
    runtime = config.get("runtime")
    telemetry = config.get("telemetry")
    if not all(
        isinstance(section, dict)
        for section in (
            data,
            training,
            distributed,
            selection,
            runtime,
            telemetry,
        )
    ):
        raise ValueError("scaling config is missing a required section")
    assert isinstance(data, dict) and isinstance(training, dict)
    assert isinstance(distributed, dict) and isinstance(selection, dict)
    assert isinstance(runtime, dict) and isinstance(telemetry, dict)
    world_sizes = config.get("world_sizes")
    if world_sizes != list(REQUIRED_WORLD_SIZES):
        raise ValueError("scaling world sizes must be exactly 1, 2, 4, and 8")
    source_count = int(data.get("source_count", 0))
    sample_count = int(data.get("sample_count", 0))
    micro_batch = int(data.get("micro_batch_size_per_rank", 0))
    global_batch = int(data.get("global_batch_size", 0))
    expected = data.get("expected_gradient_accumulation_steps")
    if not isinstance(expected, dict):
        raise ValueError("missing accumulation-step map")
    for world_size in REQUIRED_WORLD_SIZES:
        accumulation = expected_accumulation_steps(
            source_count=source_count, world_size=world_size
        )
        if expected.get(str(world_size)) != accumulation:
            raise ValueError(f"bad accumulation steps for world size {world_size}")
        validate_full_batch_contract(
            sample_count=sample_count,
            source_count=source_count,
            world_size=world_size,
            micro_batch_size=micro_batch,
            gradient_accumulation_steps=accumulation,
            global_batch_size=global_batch,
        )
    if int(training.get("warmup_optimizer_steps", 0)) < 1:
        raise ValueError("scaling benchmark requires at least one warmup step")
    if int(training.get("measured_optimizer_steps", 0)) < 5:
        raise ValueError("scaling benchmark measurement is too short")
    if training.get("gradient_checkpointing") is not True:
        raise ValueError("gradient checkpointing contract changed")
    if training.get("precision") != "bfloat16" or training.get(
        "attention_backend"
    ) != "math_sdpa":
        raise ValueError("precision or attention backend changed")
    expected_distributed = {
        "backend": "nccl",
        "broadcast_buffers": False,
        "ddp_init_sync": False,
        "gradient_as_bucket_view": True,
        "find_unused_parameters": False,
        "timeout_minutes": 30,
    }
    if distributed != expected_distributed:
        raise ValueError("distributed scaling contract changed")
    if int(selection.get("projection_optimizer_steps", 0)) <= 0:
        raise ValueError("projection step count must be positive")
    minimum_speedup = float(
        selection.get("minimum_best_multi_gpu_speedup_over_one_gpu", 0.0)
    )
    if minimum_speedup <= 1.0:
        raise ValueError("multi-GPU speedup requirement must exceed one")
    maximum_loss_deviation = float(
        selection.get(
            "maximum_relative_measured_loss_deviation_from_one_gpu", -1.0
        )
    )
    if not 0.0 < maximum_loss_deviation <= 0.25:
        raise ValueError("cross-world loss-equivalence tolerance is invalid")
    if (
        selection.get("primary_criterion")
        != "minimum_projected_training_wall_seconds"
        or selection.get("tiebreaker")
        != "higher_measured_global_samples_per_second"
    ):
        raise ValueError("scaling-selection rule changed")
    if runtime.get("offline") is not True or runtime.get(
        "deterministic_algorithms"
    ) is not True:
        raise ValueError("runtime must remain offline and deterministic")
    if (
        telemetry.get("sample_interval_seconds") != 2
        or telemetry.get("monitor_all_physical_gpus") is not True
        or not isinstance(telemetry.get("csv"), str)
        or not isinstance(telemetry.get("events"), str)
    ):
        raise ValueError("GPU telemetry contract changed")


def build_scaling_decision(
    summaries: Sequence[dict[str, object]],
    *,
    projection_optimizer_steps: int,
    minimum_multi_gpu_speedup: float,
    maximum_relative_loss_deviation: float,
) -> dict[str, object]:
    if projection_optimizer_steps <= 0:
        raise ValueError("projection step count must be positive")
    if minimum_multi_gpu_speedup <= 1.0:
        raise ValueError("multi-GPU speedup requirement must exceed one")
    if not 0.0 < maximum_relative_loss_deviation <= 0.25:
        raise ValueError("cross-world loss-equivalence tolerance is invalid")
    by_world: dict[int, dict[str, object]] = {}
    for summary in summaries:
        world_size = summary.get("world_size")
        if isinstance(world_size, bool) or not isinstance(world_size, int):
            raise ValueError("scaling summary has no integer world size")
        if world_size in by_world:
            raise ValueError(f"duplicate scaling summary for world size {world_size}")
        if summary.get("status") != "passed":
            raise ValueError(f"world size {world_size} did not pass correctness")
        seconds_per_step = float(summary.get("measured_seconds_per_step", math.nan))
        samples_per_second = float(
            summary.get("measured_global_samples_per_second", math.nan)
        )
        startup = float(
            summary.get("shell_to_measurement_start_seconds", math.nan)
        )
        warmup = float(summary.get("warmup_seconds", math.nan))
        if not all(
            math.isfinite(value) and value > 0.0
            for value in (seconds_per_step, samples_per_second, startup, warmup)
        ):
            raise ValueError(f"world size {world_size} has invalid timing evidence")
        if warmup >= startup:
            raise ValueError(f"world size {world_size} has invalid startup timing")
        losses = summary.get("measured_losses")
        if (
            not isinstance(losses, list)
            or not losses
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < 0.0
                for value in losses
            )
        ):
            raise ValueError(f"world size {world_size} has invalid measured losses")
        by_world[world_size] = summary
    if tuple(sorted(by_world)) != REQUIRED_WORLD_SIZES:
        raise ValueError("scaling summaries do not cover 1, 2, 4, and 8 GPUs")

    one_gpu_throughput = float(by_world[1]["measured_global_samples_per_second"])
    rows: list[dict[str, object]] = []
    reference_losses = [float(value) for value in by_world[1]["measured_losses"]]
    for world_size in REQUIRED_WORLD_SIZES:
        summary = by_world[world_size]
        throughput = float(summary["measured_global_samples_per_second"])
        fixed_startup = float(summary["shell_to_measurement_start_seconds"]) - float(
            summary["warmup_seconds"]
        )
        projected = fixed_startup + (
            projection_optimizer_steps
            * float(summary["measured_seconds_per_step"])
        )
        measured_losses = [float(value) for value in summary["measured_losses"]]
        if len(measured_losses) != len(reference_losses):
            raise ValueError(
                f"world size {world_size} measured a different number of losses"
            )
        relative_loss_deviations = [
            abs(value - reference) / max(abs(reference), 1e-12)
            for value, reference in zip(measured_losses, reference_losses)
        ]
        rows.append(
            {
                "world_size": world_size,
                "gradient_accumulation_steps": summary[
                    "gradient_accumulation_steps"
                ],
                "measured_seconds_per_step": summary["measured_seconds_per_step"],
                "measured_global_samples_per_second": throughput,
                "throughput_speedup_over_one_gpu": throughput / one_gpu_throughput,
                "shell_to_measurement_start_seconds": summary[
                    "shell_to_measurement_start_seconds"
                ],
                "warmup_seconds": summary["warmup_seconds"],
                "projected_fixed_startup_seconds": fixed_startup,
                "projected_training_wall_seconds": projected,
                "maximum_relative_measured_loss_deviation_from_one_gpu": max(
                    relative_loss_deviations, default=0.0
                ),
                "maximum_allocated_gpu_memory_gib_by_rank": summary[
                    "maximum_allocated_gpu_memory_gib_by_rank"
                ],
            }
        )
    selected = min(
        rows,
        key=lambda row: (
            float(row["projected_training_wall_seconds"]),
            -float(row["measured_global_samples_per_second"]),
        ),
    )
    best_multi_speedup = max(
        float(row["throughput_speedup_over_one_gpu"])
        for row in rows
        if int(row["world_size"]) > 1
    )
    one_gpu_projected = float(rows[0]["projected_training_wall_seconds"])
    best_multi_projected = min(
        float(row["projected_training_wall_seconds"])
        for row in rows
        if int(row["world_size"]) > 1
    )
    best_multi_projected_speedup = one_gpu_projected / best_multi_projected
    maximum_observed_loss_deviation = max(
        float(row["maximum_relative_measured_loss_deviation_from_one_gpu"])
        for row in rows
    )
    return {
        "projection_optimizer_steps": projection_optimizer_steps,
        "minimum_required_multi_gpu_speedup": minimum_multi_gpu_speedup,
        "best_multi_gpu_throughput_speedup_over_one_gpu": best_multi_speedup,
        "multi_gpu_throughput_benefit_passed": best_multi_speedup
        >= minimum_multi_gpu_speedup,
        "best_multi_gpu_projected_wall_speedup_over_one_gpu": (
            best_multi_projected_speedup
        ),
        "multi_gpu_projected_wall_time_benefit_passed": (
            best_multi_projected_speedup > 1.0
        ),
        "maximum_allowed_relative_measured_loss_deviation": (
            maximum_relative_loss_deviation
        ),
        "maximum_observed_relative_measured_loss_deviation": (
            maximum_observed_loss_deviation
        ),
        "cross_world_loss_equivalence_passed": (
            maximum_observed_loss_deviation <= maximum_relative_loss_deviation
        ),
        "selected_world_size": selected["world_size"],
        "selection_criterion": "minimum_projected_training_wall_seconds",
        "rows": rows,
    }
