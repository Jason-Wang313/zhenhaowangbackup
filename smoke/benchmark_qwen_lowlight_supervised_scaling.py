#!/usr/bin/env python3
"""Benchmark equivalent supervised Qwen updates on one, two, four, or eight GPUs."""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import platform
import re
import sys
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lowlight_restoration.qwen_distributed import (
    partition_source_groups,
    validate_full_batch_contract,
)
from lowlight_restoration.qwen_scaling import (
    REQUIRED_WORLD_SIZES,
    expected_accumulation_steps,
    load_scaling_config,
    validate_scaling_contract,
)
from lowlight_restoration.qwen_supervised import record_bundle_sha256
from smoke.qwen_lowlight_supervised_ddp_preflight import load_ddp_config
from smoke.qwen_lowlight_supervised_preflight import (
    load_inventory,
    select_records,
    sha256,
    workspace_path,
)
from smoke.train_qwen_lowlight_supervised import (
    atomic_write_new_json,
    frozen_snapshot,
    frozen_unchanged,
    git_revision,
    gradient_audit,
    load_model_and_processor,
    read_json,
    seed_everything,
)
from smoke.train_qwen_lowlight_supervised_ddp import (
    aggregate_audit,
    all_ranks_true,
    move_batch,
    prepare_local_batches,
    verify_model_sentinels,
)


def load_contract(
    scaling_path: Path,
) -> tuple[dict[str, object], str, dict[str, object], Path]:
    scaling, scaling_sha, _ = load_scaling_config(scaling_path)
    validate_scaling_contract(scaling)
    base_path = workspace_path(scaling.get("base_overfit_config"))
    base, base_sha, _ = load_ddp_config(base_path)
    if base_sha != scaling.get("base_overfit_config_sha256"):
        raise RuntimeError("base overfit config SHA-256 changed")
    return scaling, scaling_sha, base, base_path


def init_distributed(
    scaling: dict[str, object], requested_world_size: int
) -> tuple[int, int, int, torch.device]:
    distributed = scaling["distributed"]
    assert isinstance(distributed, dict)
    required_environment = ("RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE")
    missing = [name for name in required_environment if name not in os.environ]
    if missing:
        raise SystemExit(f"TORCHRUN_ENVIRONMENT_MISSING_{'_'.join(missing)}")
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
    if requested_world_size not in REQUIRED_WORLD_SIZES:
        raise SystemExit(f"UNSUPPORTED_SCALING_WORLD_SIZE_{requested_world_size}")
    if world_size != requested_world_size or local_world_size != requested_world_size:
        raise SystemExit(
            f"TORCHRUN_WORLD_SIZE_MISMATCH_{requested_world_size}_{world_size}_"
            f"{local_world_size}"
        )
    if not 0 <= rank < world_size or not 0 <= local_rank < world_size:
        raise SystemExit("INVALID_DISTRIBUTED_RANK")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA_NOT_AVAILABLE")
    if torch.cuda.device_count() != requested_world_size:
        raise SystemExit(
            f"VISIBLE_CUDA_COUNT_MUST_EQUAL_{requested_world_size}_GOT_"
            f"{torch.cuda.device_count()}"
        )
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dist.init_process_group(
        backend=str(distributed["backend"]),
        rank=rank,
        world_size=world_size,
        timeout=timedelta(minutes=int(distributed["timeout_minutes"])),
    )
    return rank, world_size, local_rank, device


def validate_runtime(
    scaling: dict[str, object], *, local_rank: int, world_size: int
) -> dict[str, object]:
    runtime = scaling["runtime"]
    assert isinstance(runtime, dict)
    required_workspace = str(runtime["required_cublas_workspace_config"])
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != required_workspace:
        raise SystemExit(f"CUBLAS_WORKSPACE_CONFIG_MUST_EQUAL_{required_workspace}")
    properties = torch.cuda.get_device_properties(local_rank)
    total_gib = properties.total_memory / 1024**3
    if total_gib < float(runtime["minimum_gpu_memory_gib"]):
        raise SystemExit(f"GPU_{local_rank}_MEMORY_TOO_SMALL_{total_gib:.2f}_GIB")
    import transformers

    actual = {
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "transformers": transformers.__version__,
    }
    expected = {
        "python": runtime["required_python"],
        "pytorch": runtime["required_pytorch"],
        "transformers": runtime["required_transformers"],
    }
    if actual != expected:
        raise SystemExit(f"RUNTIME_VERSION_MISMATCH_{actual}")
    return {
        "rank": dist.get_rank(),
        "local_rank": local_rank,
        "world_size": world_size,
        "gpu": torch.cuda.get_device_name(local_rank),
        "gpu_total_memory_gib": total_gib,
        "cuda": torch.version.cuda,
        "versions": actual,
    }


def verify_preflight(
    scaling: dict[str, object], scaling_sha: str, output_dir: Path
) -> dict[str, object]:
    path = output_dir / "preflight.json"
    if not path.is_file():
        raise RuntimeError("scaling preflight evidence is missing")
    evidence = read_json(path)
    if evidence.get("scaling_config_sha256") != scaling_sha:
        raise RuntimeError("scaling preflight belongs to another config")
    if evidence.get("base_overfit_config_sha256") != scaling.get(
        "base_overfit_config_sha256"
    ):
        raise RuntimeError("scaling preflight base config changed")
    base_preflight = evidence.get("base_preflight")
    if not isinstance(base_preflight, dict) or base_preflight.get(
        "large_model_hashes_checked"
    ) is not True:
        raise RuntimeError("scaling preflight omitted full model hashes")
    if base_preflight.get("selection_bundle_sha256") != scaling.get(
        "selection_bundle_sha256"
    ):
        raise RuntimeError("scaling preflight selection changed")
    return evidence


def completed_summary(
    path: Path, *, scaling_sha: str, world_size: int
) -> int | None:
    if not path.is_file():
        return None
    summary = read_json(path)
    if summary.get("scaling_config_sha256") != scaling_sha:
        raise RuntimeError("existing scaling summary belongs to another config")
    if summary.get("world_size") != world_size:
        raise RuntimeError("existing scaling summary has another world size")
    if summary.get("status") != "passed" or summary.get(
        "automatic_gate_pass"
    ) is not True:
        raise RuntimeError("existing scaling summary is not a verified pass")
    print(
        f"QWEN_SCALING_WORLD_{world_size}_PASS existing_verified_result",
        flush=True,
    )
    return 0


def main() -> int:
    process_started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    args = parser.parse_args()
    scaling, scaling_sha, base, _ = load_contract(args.config)
    data = scaling["data"]
    training = scaling["training"]
    distributed = scaling["distributed"]
    runtime = scaling["runtime"]
    assert all(
        isinstance(section, dict)
        for section in (data, training, distributed, runtime)
    )
    output_dir = workspace_path(runtime["output_dir"])
    summary_path = output_dir / f"world_{args.world_size}" / "summary.json"
    rank = world_size = local_rank = -1
    process_group_ready = False
    try:
        rank, world_size, local_rank, device = init_distributed(
            scaling, args.world_size
        )
        process_group_ready = True
        existing = completed_summary(
            summary_path, scaling_sha=scaling_sha, world_size=world_size
        )
        existing_tensor = torch.tensor(
            int(existing is not None), dtype=torch.int32, device=device
        )
        dist.all_reduce(existing_tensor, op=dist.ReduceOp.MIN)
        if bool(existing_tensor.item()):
            return 0

        launcher_attempt_id = os.environ.get("QWEN_SCALING_ATTEMPT_ID", "")
        if re.fullmatch(r"[0-9]{8}T[0-9]{6}Z-[0-9]+", launcher_attempt_id) is None:
            raise RuntimeError("QWEN_SCALING_ATTEMPT_ID is missing or malformed")

        preflight = verify_preflight(scaling, scaling_sha, output_dir)
        runtime_evidence = validate_runtime(
            scaling, local_rank=local_rank, world_size=world_size
        )
        inventory, _ = load_inventory(base)
        inventory_model = inventory["model"]
        assert isinstance(inventory_model, dict)
        selected, _ = select_records(base)
        selection_sha = record_bundle_sha256(selected)
        if selection_sha != scaling["selection_bundle_sha256"]:
            raise RuntimeError("scaling selection differs from the passing overfit gate")
        source_count = len({record.source_id for record in selected})
        accumulation_steps = expected_accumulation_steps(
            source_count=source_count, world_size=world_size
        )
        expected_accumulation = data["expected_gradient_accumulation_steps"]
        assert isinstance(expected_accumulation, dict)
        if accumulation_steps != expected_accumulation[str(world_size)]:
            raise RuntimeError("world-size accumulation contract changed")
        validate_full_batch_contract(
            sample_count=len(selected),
            source_count=source_count,
            world_size=world_size,
            micro_batch_size=int(data["micro_batch_size_per_rank"]),
            gradient_accumulation_steps=accumulation_steps,
            global_batch_size=int(data["global_batch_size"]),
        )
        local_groups = partition_source_groups(
            selected, world_size=world_size, rank=rank
        )

        os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        seed_everything(int(training["seed"]))
        if runtime["deterministic_algorithms"] is True:
            torch.use_deterministic_algorithms(True)
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        model, processor, parameter_groups = load_model_and_processor(
            base, inventory, device
        )
        bad_dropout = [
            name
            for name, module in model.named_modules()
            if isinstance(module, torch.nn.Dropout) and module.p != 0.0
        ]
        if bad_dropout:
            raise RuntimeError(f"nonzero Qwen dropout changed: {bad_dropout}")
        if training["gradient_checkpointing"] is True:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        model.train()
        initial_sentinels_match = verify_model_sentinels(
            model, world_size=world_size, device=device
        )
        if not initial_sentinels_match:
            raise RuntimeError("independently loaded model parameters differ by rank")
        ddp_model = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=bool(distributed["broadcast_buffers"]),
            init_sync=bool(distributed["ddp_init_sync"]),
            gradient_as_bucket_view=bool(distributed["gradient_as_bucket_view"]),
            find_unused_parameters=bool(distributed["find_unused_parameters"]),
        )
        preprocessing_started = time.monotonic()
        local_batches = prepare_local_batches(
            processor,
            local_groups,
            expected_micro_batch=int(data["micro_batch_size_per_rank"]),
        )
        preprocessing_seconds = time.monotonic() - preprocessing_started
        if len(local_batches) != accumulation_steps:
            raise RuntimeError("local source batches differ from accumulation steps")

        trainable_parameters = [
            parameter for parameter in model.parameters() if parameter.requires_grad
        ]
        optimizer = torch.optim.AdamW(
            trainable_parameters,
            lr=float(training["learning_rate"]),
            betas=(float(training["adam_beta1"]), float(training["adam_beta2"])),
            eps=float(training["adam_epsilon"]),
            weight_decay=float(training["weight_decay"]),
            fused=bool(training["fused_adamw"]),
        )
        optimizer_ids = {
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        }
        if optimizer_ids != {id(parameter) for parameter in trainable_parameters}:
            raise RuntimeError("optimizer scope differs from the visual path")
        frozen_versions, frozen_sentinels = frozen_snapshot(model)

        warmup_steps = int(training["warmup_optimizer_steps"])
        measured_steps = int(training["measured_optimizer_steps"])
        total_steps = warmup_steps + measured_steps
        global_batch_size = int(data["global_batch_size"])
        warmup_started = time.monotonic()
        initialization_seconds = warmup_started - process_started
        measured_started = 0.0
        measurement_wall_ns = 0
        measured_losses: list[float] = []
        distributed_gradient_audit: dict[str, object] | None = None
        for step in range(1, total_steps + 1):
            if step == warmup_steps + 1:
                dist.barrier()
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
                measured_started = time.monotonic()
                measurement_wall_ns = time.time_ns()
            optimizer.zero_grad(set_to_none=True)
            local_numerator = 0.0
            local_tokens = 0
            for micro_index, prepared in enumerate(local_batches):
                synchronize = micro_index == accumulation_steps - 1
                sync_context = (
                    contextlib.nullcontext()
                    if synchronize
                    else ddp_model.no_sync()
                )
                batch = prepared["batch"]
                assert isinstance(batch, dict)
                with sync_context:
                    output = ddp_model(**move_batch(batch, device), use_cache=False)
                    loss = output.loss
                    if loss is None or not torch.isfinite(loss):
                        raise RuntimeError(f"non-finite scaling loss at step {step}")
                    (loss / accumulation_steps).backward()
                target_tokens = int(prepared["target_tokens"])
                local_numerator += float(loss.detach().float().item()) * target_tokens
                local_tokens += target_tokens
            if step == 1:
                local_gradient = gradient_audit(model)
                distributed_gradient_audit = aggregate_audit(
                    local_gradient, rank=rank, world_size=world_size
                )
                if (
                    distributed_gradient_audit["distributed_all_ranks_passed"]
                    is not True
                ):
                    raise RuntimeError("distributed gradient audit failed")
            frozen_with_grad = [
                name
                for name, parameter in model.named_parameters()
                if not parameter.requires_grad and parameter.grad is not None
            ]
            if not all_ranks_true(not frozen_with_grad, device):
                raise RuntimeError("a frozen decoder received a gradient")
            grad_norm = torch.nn.utils.clip_grad_norm_(
                trainable_parameters, float(training["max_grad_norm"])
            )
            if not torch.isfinite(grad_norm):
                raise RuntimeError(f"non-finite scaling gradient norm at step {step}")
            optimizer.step()
            totals = torch.tensor(
                [local_numerator, local_tokens], dtype=torch.float64, device=device
            )
            dist.all_reduce(totals, op=dist.ReduceOp.SUM)
            global_loss = float((totals[0] / totals[1]).item())
            if step > warmup_steps:
                measured_losses.append(global_loss)
            if rank == 0:
                phase = "warmup" if step <= warmup_steps else "measured"
                print(
                    f"QWEN_SCALING_STEP world={world_size} phase={phase} "
                    f"step={step}/{total_steps} loss={global_loss:.8f} "
                    f"grad_norm={float(grad_norm.detach().float().item()):.6f} "
                    f"global_batch={global_batch_size}",
                    flush=True,
                )

        torch.cuda.synchronize(device)
        measured_seconds = time.monotonic() - measured_started
        if measured_started <= 0.0 or len(measured_losses) != measured_steps:
            raise RuntimeError("measured scaling interval is incomplete")
        warmup_seconds = measured_started - warmup_started
        local_frozen = frozen_unchanged(model, frozen_versions, frozen_sentinels)
        frozen_audit = aggregate_audit(
            local_frozen, rank=rank, world_size=world_size
        )
        final_sentinels_match = verify_model_sentinels(
            model, world_size=world_size, device=device
        )
        if distributed_gradient_audit is None:
            raise RuntimeError("gradient audit was not executed")
        gate_checks = {
            "gradient_scope_passed": distributed_gradient_audit[
                "distributed_all_ranks_passed"
            ]
            is True,
            "frozen_parameters_unchanged": frozen_audit[
                "distributed_all_ranks_passed"
            ]
            is True,
            "distributed_model_equivalence_passed": initial_sentinels_match
            and final_sentinels_match,
            "global_batch_semantics_passed": True,
            "finite_measured_losses": all(
                math.isfinite(value) for value in measured_losses
            ),
        }
        automatic_pass = all(gate_checks.values())
        runtime_evidence["maximum_allocated_gpu_memory_gib"] = (
            torch.cuda.max_memory_allocated(device) / 1024**3
        )
        gathered_runtime: list[object | None] = [None] * world_size
        dist.all_gather_object(gathered_runtime, runtime_evidence)
        max_memory = [
            float(item["maximum_allocated_gpu_memory_gib"])
            for item in gathered_runtime
            if isinstance(item, dict)
        ]
        if len(max_memory) != world_size:
            raise RuntimeError("per-rank memory evidence is incomplete")
        shell_start_ns = int(os.environ.get("QWEN_SCALING_SHELL_START_NS", "0"))
        if shell_start_ns <= 0 or measurement_wall_ns <= shell_start_ns:
            raise RuntimeError("shell-to-measurement timing marker is invalid")
        shell_to_measurement = (measurement_wall_ns - shell_start_ns) / 1e9
        shell_to_summary = (time.time_ns() - shell_start_ns) / 1e9
        measured_seconds_per_step = measured_seconds / measured_steps
        measured_samples_per_second = (
            measured_steps * global_batch_size / measured_seconds
        )
        if rank == 0:
            summary: dict[str, object] = {
                "schema_version": 1,
                "status": "passed" if automatic_pass else "failed",
                "scope": "qwen_lowlight_supervised_distributed_scaling_benchmark",
                "scaling_config_sha256": scaling_sha,
                "base_overfit_config_sha256": scaling[
                    "base_overfit_config_sha256"
                ],
                "preflight_sha256": sha256(output_dir / "preflight.json"),
                "git_revision": git_revision(),
                "launcher_attempt_id": launcher_attempt_id,
                "physical_cuda_device_set": os.environ.get(
                    "CUDA_VISIBLE_DEVICES", ""
                ),
                "model_revision": inventory_model["revision"],
                "selection_bundle_sha256": selection_sha,
                "parameter_groups": parameter_groups,
                "world_size": world_size,
                "micro_batch_size_per_rank": int(
                    data["micro_batch_size_per_rank"]
                ),
                "gradient_accumulation_steps": accumulation_steps,
                "global_batch_size": global_batch_size,
                "warmup_optimizer_steps": warmup_steps,
                "measured_optimizer_steps": measured_steps,
                "measured_sample_presentations": measured_steps
                * global_batch_size,
                "initialization_seconds_this_process": initialization_seconds,
                "local_preprocessing_seconds_this_process": preprocessing_seconds,
                "warmup_seconds": warmup_seconds,
                "measured_training_seconds": measured_seconds,
                "measured_seconds_per_step": measured_seconds_per_step,
                "measured_global_samples_per_second": measured_samples_per_second,
                "shell_to_measurement_start_seconds": shell_to_measurement,
                "shell_to_summary_seconds": shell_to_summary,
                "measured_losses": measured_losses,
                "final_measured_loss": measured_losses[-1],
                "gradient_audit": distributed_gradient_audit,
                "frozen_parameter_audit": frozen_audit,
                "maximum_allocated_gpu_memory_gib_by_rank": max_memory,
                "environment_by_rank": gathered_runtime,
                "gate_checks": gate_checks,
                "automatic_gate_pass": automatic_pass,
                "synthetic_internal_only": True,
                "scientific_pilot_result": False,
                "test_split_used": False,
                "official_foundir_used": False,
                "base_preflight_large_hashes_checked": preflight[
                    "base_preflight"
                ]["large_model_hashes_checked"],
            }
            atomic_write_new_json(summary_path, summary)
            marker = (
                f"QWEN_SCALING_WORLD_{world_size}_PASS"
                if automatic_pass
                else f"QWEN_SCALING_WORLD_{world_size}_FAILED"
            )
            print(
                marker
                + " "
                + json.dumps(
                    {
                        "seconds_per_step": measured_seconds_per_step,
                        "global_samples_per_second": measured_samples_per_second,
                        **gate_checks,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        result = torch.tensor(int(automatic_pass), dtype=torch.int32, device=device)
        dist.broadcast(result, src=0)
        return 0 if bool(result.item()) else 2
    finally:
        if process_group_ready and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
