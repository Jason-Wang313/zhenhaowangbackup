#!/usr/bin/env python3
"""Preflight and select the fastest equivalent 1/2/4/8-GPU Qwen run."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lowlight_restoration.qwen_scaling import (
    REQUIRED_WORLD_SIZES,
    build_scaling_decision,
    load_scaling_config,
    validate_scaling_contract,
)
from smoke.qwen_lowlight_supervised_ddp_preflight import (
    load_ddp_config,
    run_preflight as run_ddp_preflight,
    validate_contract as validate_ddp_contract,
)
from smoke.qwen_lowlight_supervised_preflight import sha256, workspace_path
from smoke.train_qwen_lowlight_supervised import (
    atomic_write_new_bytes,
    atomic_write_new_json,
    read_json,
)

ATTEMPT_RE = re.compile(r"[0-9]{8}T[0-9]{6}Z-[0-9]+")
GIT_RE = re.compile(r"[0-9a-f]{40}")
SUMMARY_SCOPE = "qwen_lowlight_supervised_distributed_scaling_benchmark"
REQUIRED_GATE_CHECKS = (
    "gradient_scope_passed",
    "frozen_parameters_unchanged",
    "distributed_model_equivalence_passed",
    "global_batch_semantics_passed",
    "finite_measured_losses",
)


def load_contract(
    config_path: Path,
) -> tuple[
    dict[str, object],
    str,
    bytes,
    dict[str, object],
    Path,
    Path,
]:
    scaling_path = workspace_path(str(config_path))
    scaling, scaling_sha, scaling_raw = load_scaling_config(scaling_path)
    validate_scaling_contract(scaling)
    base_path = workspace_path(scaling.get("base_overfit_config"))
    base, base_sha, _ = load_ddp_config(base_path)
    validate_ddp_contract(base)
    if base_sha != scaling.get("base_overfit_config_sha256"):
        raise RuntimeError("base overfit config SHA-256 changed")
    runtime = scaling["runtime"]
    assert isinstance(runtime, dict)
    output_dir = workspace_path(runtime.get("output_dir"))
    return scaling, scaling_sha, scaling_raw, base, base_path, output_dir


def _canonical_base_preflight(result: dict[str, object]) -> dict[str, object]:
    canonical = dict(result)
    canonical.pop("cuda_visible_in_preflight", None)
    canonical["cuda_visibility_excluded_from_artifact_equivalence"] = True
    return canonical


def build_preflight(
    scaling: dict[str, object],
    scaling_sha: str,
    base_path: Path,
    *,
    skip_large_model_hashes: bool,
) -> dict[str, object]:
    base_result = _canonical_base_preflight(
        run_ddp_preflight(
            base_path, skip_large_model_hashes=skip_large_model_hashes
        )
    )
    if base_result.get("config_sha256") != scaling.get(
        "base_overfit_config_sha256"
    ):
        raise RuntimeError("base preflight config SHA-256 changed")
    if base_result.get("selection_bundle_sha256") != scaling.get(
        "selection_bundle_sha256"
    ):
        raise RuntimeError("scaling selection differs from the passing overfit gate")
    if base_result.get("selected_sources") != scaling["data"]["source_count"]:
        raise RuntimeError("scaling source count differs from base preflight")
    if base_result.get("selected_records") != scaling["data"]["sample_count"]:
        raise RuntimeError("scaling record count differs from base preflight")

    data = scaling["data"]
    assert isinstance(data, dict)
    accumulation = data["expected_gradient_accumulation_steps"]
    assert isinstance(accumulation, dict)
    world_contracts = [
        {
            "world_size": world_size,
            "source_groups_per_rank": int(data["source_count"]) // world_size,
            "micro_batch_size_per_rank": data["micro_batch_size_per_rank"],
            "gradient_accumulation_steps": accumulation[str(world_size)],
            "global_batch_size": data["global_batch_size"],
        }
        for world_size in REQUIRED_WORLD_SIZES
    ]
    return {
        "schema_version": 1,
        "status": "passed",
        "scope": "qwen_lowlight_supervised_scaling_preflight",
        "scaling_config_sha256": scaling_sha,
        "base_overfit_config_sha256": scaling["base_overfit_config_sha256"],
        "selection_bundle_sha256": scaling["selection_bundle_sha256"],
        "large_model_hashes_checked": not skip_large_model_hashes,
        "base_preflight": base_result,
        "world_contracts": world_contracts,
        "test_split_used": False,
        "official_foundir_used": False,
    }


def _compare_preflight_core(
    existing: dict[str, object], current_skip_hash: dict[str, object]
) -> None:
    if existing.get("large_model_hashes_checked") is not True:
        raise RuntimeError("authoritative scaling preflight omitted full model hashes")
    for key in (
        "schema_version",
        "status",
        "scope",
        "scaling_config_sha256",
        "base_overfit_config_sha256",
        "selection_bundle_sha256",
        "world_contracts",
    ):
        if existing.get(key) != current_skip_hash.get(key):
            raise RuntimeError(f"existing scaling preflight changed: {key}")
    existing_base = existing.get("base_preflight")
    current_base = current_skip_hash.get("base_preflight")
    if not isinstance(existing_base, dict) or not isinstance(current_base, dict):
        raise RuntimeError("scaling preflight has no base evidence")
    for key in (
        "config_sha256",
        "inventory_config_sha256",
        "model_revision",
        "model_files",
        "parameter_groups",
        "tensor_count",
        "selected_sources",
        "selected_records",
        "selected_per_level",
        "selection_bundle_sha256",
        "rank_source_assignments",
        "global_batch_size",
        "visual_tokens",
        "assistant_target_tokens",
        "processor",
        "image_processor",
        "tokenizer",
        "versions",
    ):
        if existing_base.get(key) != current_base.get(key):
            raise RuntimeError(f"existing base preflight changed: {key}")


def publish_or_verify_preflight(
    scaling: dict[str, object],
    scaling_sha: str,
    scaling_raw: bytes,
    base_path: Path,
    output_dir: Path,
    *,
    skip_large_model_hashes: bool,
) -> tuple[dict[str, object], Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_new_bytes(output_dir / "scaling_config.json", scaling_raw)
    authoritative_path = output_dir / "preflight.json"
    if not skip_large_model_hashes and authoritative_path.is_file():
        existing = read_json(authoritative_path)
        current = build_preflight(
            scaling,
            scaling_sha,
            base_path,
            skip_large_model_hashes=True,
        )
        _compare_preflight_core(existing, current)
        return existing, authoritative_path

    result = build_preflight(
        scaling,
        scaling_sha,
        base_path,
        skip_large_model_hashes=skip_large_model_hashes,
    )
    path = (
        output_dir / "preflight_skip_large_model_hashes.json"
        if skip_large_model_hashes
        else authoritative_path
    )
    atomic_write_new_json(path, result)
    return result, path


def verified_authoritative_preflight(
    scaling: dict[str, object],
    scaling_sha: str,
    base_path: Path,
    output_dir: Path,
) -> tuple[dict[str, object], Path]:
    path = output_dir / "preflight.json"
    if not path.is_file():
        raise RuntimeError("authoritative full-hash scaling preflight is missing")
    existing = read_json(path)
    current = build_preflight(
        scaling,
        scaling_sha,
        base_path,
        skip_large_model_hashes=True,
    )
    _compare_preflight_core(existing, current)
    return existing, path


def _finite_number(value: object, label: str, *, allow_zero: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"{label} is not numeric")
    number = float(value)
    if not math.isfinite(number) or (number < 0.0 if allow_zero else number <= 0.0):
        raise RuntimeError(f"{label} is not a valid finite value")
    return number


def validate_world_summary(
    summary: dict[str, object],
    *,
    world_size: int,
    scaling: dict[str, object],
    scaling_sha: str,
    base: dict[str, object],
    preflight: dict[str, object],
    preflight_sha: str,
) -> None:
    data = scaling["data"]
    training = scaling["training"]
    runtime = scaling["runtime"]
    base_model = base["model"]
    base_preflight = preflight["base_preflight"]
    assert all(
        isinstance(section, dict)
        for section in (data, training, runtime, base_model, base_preflight)
    )
    expected_accumulation = data["expected_gradient_accumulation_steps"]
    assert isinstance(expected_accumulation, dict)
    expected_exact = {
        "schema_version": 1,
        "status": "passed",
        "scope": SUMMARY_SCOPE,
        "scaling_config_sha256": scaling_sha,
        "base_overfit_config_sha256": scaling["base_overfit_config_sha256"],
        "preflight_sha256": preflight_sha,
        "model_revision": base_preflight["model_revision"],
        "selection_bundle_sha256": scaling["selection_bundle_sha256"],
        "parameter_groups": base_model["expected_parameter_groups"],
        "world_size": world_size,
        "physical_cuda_device_set": ",".join(
            str(index) for index in range(world_size)
        ),
        "micro_batch_size_per_rank": data["micro_batch_size_per_rank"],
        "gradient_accumulation_steps": expected_accumulation[str(world_size)],
        "global_batch_size": data["global_batch_size"],
        "warmup_optimizer_steps": training["warmup_optimizer_steps"],
        "measured_optimizer_steps": training["measured_optimizer_steps"],
        "measured_sample_presentations": int(
            training["measured_optimizer_steps"]
        )
        * int(data["global_batch_size"]),
        "automatic_gate_pass": True,
        "synthetic_internal_only": True,
        "scientific_pilot_result": False,
        "test_split_used": False,
        "official_foundir_used": False,
        "base_preflight_large_hashes_checked": True,
    }
    for key, expected in expected_exact.items():
        if summary.get(key) != expected:
            raise RuntimeError(
                f"world size {world_size} summary changed: {key} "
                f"expected={expected!r} actual={summary.get(key)!r}"
            )
    if GIT_RE.fullmatch(str(summary.get("git_revision", ""))) is None:
        raise RuntimeError(f"world size {world_size} has an invalid Git revision")
    attempt_id = str(summary.get("launcher_attempt_id", ""))
    if ATTEMPT_RE.fullmatch(attempt_id) is None:
        raise RuntimeError(f"world size {world_size} has no launcher attempt ID")

    measured_steps = int(training["measured_optimizer_steps"])
    measured_seconds = _finite_number(
        summary.get("measured_training_seconds"),
        f"world size {world_size} measured training seconds",
    )
    seconds_per_step = _finite_number(
        summary.get("measured_seconds_per_step"),
        f"world size {world_size} seconds per step",
    )
    samples_per_second = _finite_number(
        summary.get("measured_global_samples_per_second"),
        f"world size {world_size} samples per second",
    )
    warmup_seconds = _finite_number(
        summary.get("warmup_seconds"), f"world size {world_size} warmup seconds"
    )
    measurement_start = _finite_number(
        summary.get("shell_to_measurement_start_seconds"),
        f"world size {world_size} shell-to-measurement seconds",
    )
    shell_to_summary = _finite_number(
        summary.get("shell_to_summary_seconds"),
        f"world size {world_size} shell-to-summary seconds",
    )
    _finite_number(
        summary.get("initialization_seconds_this_process"),
        f"world size {world_size} initialization seconds",
    )
    _finite_number(
        summary.get("local_preprocessing_seconds_this_process"),
        f"world size {world_size} preprocessing seconds",
    )
    if measurement_start <= warmup_seconds:
        raise RuntimeError(f"world size {world_size} startup timing is inconsistent")
    if shell_to_summary + 0.1 < measurement_start + measured_seconds:
        raise RuntimeError(f"world size {world_size} total timing is inconsistent")
    if not math.isclose(
        seconds_per_step,
        measured_seconds / measured_steps,
        rel_tol=1e-9,
        abs_tol=1e-9,
    ):
        raise RuntimeError(f"world size {world_size} per-step timing changed")
    expected_throughput = (
        measured_steps * int(data["global_batch_size"]) / measured_seconds
    )
    if not math.isclose(
        samples_per_second, expected_throughput, rel_tol=1e-9, abs_tol=1e-9
    ):
        raise RuntimeError(f"world size {world_size} throughput is inconsistent")

    losses = summary.get("measured_losses")
    if not isinstance(losses, list) or len(losses) != measured_steps:
        raise RuntimeError(f"world size {world_size} measured loss count changed")
    numeric_losses = [
        _finite_number(
            value, f"world size {world_size} measured loss", allow_zero=True
        )
        for value in losses
    ]
    final_loss = _finite_number(
        summary.get("final_measured_loss"),
        f"world size {world_size} final loss",
        allow_zero=True,
    )
    if not math.isclose(final_loss, numeric_losses[-1], rel_tol=0.0, abs_tol=0.0):
        raise RuntimeError(f"world size {world_size} final loss is inconsistent")

    gate_checks = summary.get("gate_checks")
    if not isinstance(gate_checks, dict) or any(
        gate_checks.get(key) is not True for key in REQUIRED_GATE_CHECKS
    ):
        raise RuntimeError(f"world size {world_size} correctness gate failed")
    for audit_name in ("gradient_audit", "frozen_parameter_audit"):
        audit = summary.get(audit_name)
        if not isinstance(audit, dict) or audit.get(
            "distributed_all_ranks_passed"
        ) is not True:
            raise RuntimeError(f"world size {world_size} {audit_name} failed")
        rank_results = audit.get("rank_results")
        if not isinstance(rank_results, list) or len(rank_results) != world_size:
            raise RuntimeError(
                f"world size {world_size} {audit_name} rank evidence is incomplete"
            )
        if any(
            not isinstance(result, dict) or result.get("passed") is not True
            for result in rank_results
        ):
            raise RuntimeError(f"world size {world_size} {audit_name} rank failed")

    environments = summary.get("environment_by_rank")
    memories = summary.get("maximum_allocated_gpu_memory_gib_by_rank")
    if not isinstance(environments, list) or len(environments) != world_size:
        raise RuntimeError(f"world size {world_size} environment evidence is incomplete")
    if not isinstance(memories, list) or len(memories) != world_size:
        raise RuntimeError(f"world size {world_size} memory evidence is incomplete")
    seen_ranks: set[int] = set()
    for environment in environments:
        if not isinstance(environment, dict):
            raise RuntimeError(f"world size {world_size} has invalid environment evidence")
        rank = environment.get("rank")
        local_rank = environment.get("local_rank")
        if (
            isinstance(rank, bool)
            or not isinstance(rank, int)
            or isinstance(local_rank, bool)
            or not isinstance(local_rank, int)
        ):
            raise RuntimeError(f"world size {world_size} has an invalid rank")
        seen_ranks.add(rank)
        if local_rank != rank or environment.get("world_size") != world_size:
            raise RuntimeError(f"world size {world_size} rank mapping changed")
        if "H200" not in str(environment.get("gpu", "")):
            raise RuntimeError(f"world size {world_size} did not use an H200")
        if _finite_number(
            environment.get("gpu_total_memory_gib"),
            f"world size {world_size} GPU memory capacity",
        ) < float(runtime["minimum_gpu_memory_gib"]):
            raise RuntimeError(f"world size {world_size} GPU memory is too small")
        expected_versions = {
            "python": runtime["required_python"],
            "pytorch": runtime["required_pytorch"],
            "transformers": runtime["required_transformers"],
        }
        if environment.get("versions") != expected_versions:
            raise RuntimeError(f"world size {world_size} runtime versions changed")
    if seen_ranks != set(range(world_size)):
        raise RuntimeError(f"world size {world_size} rank set is incomplete")
    for rank, memory in enumerate(memories):
        _finite_number(memory, f"world size {world_size} rank {rank} peak memory")


def _mean(values: Iterable[float]) -> float:
    values_list = list(values)
    if not values_list:
        raise RuntimeError("cannot calculate telemetry mean from no samples")
    return sum(values_list) / len(values_list)


def validate_telemetry(
    path: Path, summaries: list[dict[str, object]]
) -> dict[str, object]:
    if not path.is_file():
        raise RuntimeError(f"GPU telemetry is missing: {path}")
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    with path.open(newline="") as handle:
        for row_number, row in enumerate(csv.reader(handle), start=1):
            if not row or row[0].strip() in {"attempt_id", "#"}:
                continue
            if row[0].lstrip().startswith("#"):
                continue
            if len(row) != 10:
                raise RuntimeError(f"bad GPU telemetry row {row_number}: {row!r}")
            fields = [value.strip() for value in row]
            attempt_id, stage = fields[0], fields[1]
            if ATTEMPT_RE.fullmatch(attempt_id) is None:
                raise RuntimeError(f"bad telemetry attempt ID at row {row_number}")
            try:
                parsed = {
                    "attempt_id": attempt_id,
                    "stage": stage,
                    "sample_utc": fields[2],
                    "index": int(fields[3]),
                    "name": fields[4],
                    "utilization_gpu_percent": float(fields[5]),
                    "memory_used_mib": float(fields[6]),
                    "memory_total_mib": float(fields[7]),
                    "power_draw_w": float(fields[8]),
                    "sm_clock_mhz": float(fields[9]),
                }
            except ValueError as exc:
                raise RuntimeError(
                    f"non-numeric GPU telemetry at row {row_number}"
                ) from exc
            if not all(
                math.isfinite(float(value))
                for key, value in parsed.items()
                if key
                in {
                    "utilization_gpu_percent",
                    "memory_used_mib",
                    "memory_total_mib",
                    "power_draw_w",
                    "sm_clock_mhz",
                }
            ):
                raise RuntimeError(f"non-finite GPU telemetry at row {row_number}")
            grouped[(attempt_id, stage)].append(parsed)

    world_evidence: dict[str, object] = {}
    for summary in summaries:
        world_size = int(summary["world_size"])
        attempt_id = str(summary["launcher_attempt_id"])
        stage = f"world_{world_size}"
        rows = grouped.get((attempt_id, stage), [])
        if not rows:
            raise RuntimeError(
                f"telemetry is missing for attempt {attempt_id} stage {stage}"
            )
        by_gpu: dict[int, list[dict[str, object]]] = defaultdict(list)
        for row in rows:
            by_gpu[int(row["index"])].append(row)
        if set(by_gpu) != set(range(8)):
            raise RuntimeError(
                f"telemetry for attempt {attempt_id} stage {stage} did not monitor all GPUs"
            )
        per_gpu: list[dict[str, object]] = []
        for index in range(8):
            gpu_rows = by_gpu[index]
            maximum_memory = max(float(row["memory_used_mib"]) for row in gpu_rows)
            active = index < world_size
            if active and maximum_memory < 1024.0:
                raise RuntimeError(
                    f"active GPU {index} has no model-memory evidence for {stage}"
                )
            per_gpu.append(
                {
                    "index": index,
                    "active_for_stage": active,
                    "samples": len(gpu_rows),
                    "mean_utilization_percent": _mean(
                        float(row["utilization_gpu_percent"]) for row in gpu_rows
                    ),
                    "maximum_memory_used_mib": maximum_memory,
                    "mean_power_draw_w": _mean(
                        float(row["power_draw_w"]) for row in gpu_rows
                    ),
                    "maximum_sm_clock_mhz": max(
                        float(row["sm_clock_mhz"]) for row in gpu_rows
                    ),
                }
            )
        active_rows = [row for row in rows if int(row["index"]) < world_size]
        world_evidence[str(world_size)] = {
            "launcher_attempt_id": attempt_id,
            "stage": stage,
            "telemetry_rows": len(rows),
            "all_eight_physical_gpus_monitored": True,
            "active_gpu_count": world_size,
            "active_gpu_mean_utilization_percent": _mean(
                float(row["utilization_gpu_percent"]) for row in active_rows
            ),
            "per_gpu": per_gpu,
        }
    return {
        "telemetry_path": str(path.relative_to(ROOT)),
        "telemetry_sha256": sha256(path),
        "worlds": world_evidence,
    }


def finalize_scaling(
    scaling: dict[str, object],
    scaling_sha: str,
    base: dict[str, object],
    base_path: Path,
    output_dir: Path,
) -> tuple[dict[str, object], int]:
    preflight, preflight_path = verified_authoritative_preflight(
        scaling, scaling_sha, base_path, output_dir
    )
    preflight_digest = sha256(preflight_path)
    summaries: list[dict[str, object]] = []
    summary_hashes: dict[str, str] = {}
    for world_size in REQUIRED_WORLD_SIZES:
        path = output_dir / f"world_{world_size}" / "summary.json"
        if not path.is_file():
            raise RuntimeError(f"world-size summary is missing: {path}")
        summary = read_json(path)
        validate_world_summary(
            summary,
            world_size=world_size,
            scaling=scaling,
            scaling_sha=scaling_sha,
            base=base,
            preflight=preflight,
            preflight_sha=preflight_digest,
        )
        summaries.append(summary)
        summary_hashes[str(world_size)] = sha256(path)
    git_revisions = {str(summary["git_revision"]) for summary in summaries}
    if len(git_revisions) != 1:
        raise RuntimeError("world-size benchmarks used different Git revisions")

    telemetry_config = scaling["telemetry"]
    selection = scaling["selection"]
    assert isinstance(telemetry_config, dict) and isinstance(selection, dict)
    telemetry_path = workspace_path(telemetry_config.get("csv"))
    telemetry = validate_telemetry(telemetry_path, summaries)
    decision = build_scaling_decision(
        summaries,
        projection_optimizer_steps=int(selection["projection_optimizer_steps"]),
        minimum_multi_gpu_speedup=float(
            selection["minimum_best_multi_gpu_speedup_over_one_gpu"]
        ),
        maximum_relative_loss_deviation=float(
            selection[
                "maximum_relative_measured_loss_deviation_from_one_gpu"
            ]
        ),
    )
    gate_checks = {
        "all_world_size_correctness_gates_passed": True,
        "all_world_size_telemetry_present": True,
        "multi_gpu_throughput_benefit_passed": decision[
            "multi_gpu_throughput_benefit_passed"
        ],
        "multi_gpu_projected_wall_time_benefit_passed": decision[
            "multi_gpu_projected_wall_time_benefit_passed"
        ],
        "cross_world_loss_equivalence_passed": decision[
            "cross_world_loss_equivalence_passed"
        ],
        "selected_configuration_uses_multiple_gpus": int(
            decision["selected_world_size"]
        )
        > 1,
    }
    automatic_pass = all(value is True for value in gate_checks.values())
    result = {
        "schema_version": 1,
        "status": "passed" if automatic_pass else "failed",
        "scope": "qwen_lowlight_supervised_1_2_4_8_scaling_gate",
        "scaling_config_sha256": scaling_sha,
        "base_overfit_config_sha256": scaling["base_overfit_config_sha256"],
        "selection_bundle_sha256": scaling["selection_bundle_sha256"],
        "preflight_sha256": preflight_digest,
        "git_revision": next(iter(git_revisions)),
        "world_summary_sha256": summary_hashes,
        "decision": decision,
        "telemetry": telemetry,
        "gate_checks": gate_checks,
        "automatic_gate_pass": automatic_pass,
        "selected_world_size_for_next_supervised_pilot": decision[
            "selected_world_size"
        ],
        "next_8k_supervised_pilot_authorized": automatic_pass,
        "next_8k_supervised_pilot_started": False,
        "synthetic_internal_only": True,
        "official_foundir_used": False,
    }
    atomic_write_new_json(output_dir / "scaling_summary.json", result)
    return result, 0 if automatic_pass else 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--contract-only", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--skip-large-model-hashes", action="store_true")
    args = parser.parse_args()
    modes = int(args.contract_only) + int(args.preflight_only)
    if modes > 1:
        parser.error("choose only one validation mode")
    if args.skip_large_model_hashes and not args.preflight_only:
        parser.error("--skip-large-model-hashes requires --preflight-only")

    scaling, scaling_sha, scaling_raw, base, base_path, output_dir = load_contract(
        args.config
    )
    if args.contract_only:
        print(
            "QWEN_SCALING_CONTRACT_PASS "
            + json.dumps(
                {
                    "scaling_config_sha256": scaling_sha,
                    "base_overfit_config_sha256": scaling[
                        "base_overfit_config_sha256"
                    ],
                    "world_sizes": list(REQUIRED_WORLD_SIZES),
                },
                sort_keys=True,
            )
        )
        return 0
    if args.preflight_only:
        result, path = publish_or_verify_preflight(
            scaling,
            scaling_sha,
            scaling_raw,
            base_path,
            output_dir,
            skip_large_model_hashes=args.skip_large_model_hashes,
        )
        marker = (
            "QWEN_SCALING_PREFLIGHT_SKIP_HASH_PASS"
            if args.skip_large_model_hashes
            else "QWEN_SCALING_PREFLIGHT_FULL_HASH_PASS"
        )
        print(
            marker
            + " "
            + json.dumps(
                {
                    "artifact": str(path.relative_to(ROOT)),
                    "scaling_config_sha256": scaling_sha,
                    "selection_bundle_sha256": result[
                        "selection_bundle_sha256"
                    ],
                    "large_model_hashes_checked": result[
                        "large_model_hashes_checked"
                    ],
                },
                sort_keys=True,
            )
        )
        return 0

    result, status = finalize_scaling(
        scaling, scaling_sha, base, base_path, output_dir
    )
    marker = "QWEN_SCALING_GATE_PASS" if status == 0 else "QWEN_SCALING_GATE_FAILED"
    print(
        marker
        + " "
        + json.dumps(
            {
                "selected_world_size": result[
                    "selected_world_size_for_next_supervised_pilot"
                ],
                "automatic_gate_pass": result["automatic_gate_pass"],
                "scaling_summary": str(
                    (output_dir / "scaling_summary.json").relative_to(ROOT)
                ),
            },
            sort_keys=True,
        )
    )
    return status


if __name__ == "__main__":
    raise SystemExit(main())
