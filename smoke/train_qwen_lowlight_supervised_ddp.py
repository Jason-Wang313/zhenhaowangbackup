#!/usr/bin/env python3
"""Train the Qwen2.5-VL visual path with a balanced full-batch DDP gate."""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import platform
import random
import sys
import tempfile
import time
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch
import torch.distributed as dist
from safetensors.torch import load_file, save_file
from torch.nn.parallel import DistributedDataParallel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lowlight_restoration.data import PairRecord
from lowlight_restoration.qwen_distributed import (
    cosine_learning_rate,
    merge_indexed_outputs,
    partition_source_groups,
    validate_full_batch_contract,
)
from lowlight_restoration.qwen_supervised import (
    canonical_target,
    evaluate_generated_outputs,
    open_degraded_images,
    prepare_supervised_batch,
    prompt_text,
    record_bundle_sha256,
)
from smoke.qwen_lowlight_supervised_ddp_preflight import (
    load_ddp_config,
    run_preflight,
)
from smoke.qwen_lowlight_supervised_preflight import (
    load_inventory,
    select_records,
    sha256,
    workspace_path,
)
from smoke.train_qwen_lowlight_supervised import (
    atomic_write_new_bytes,
    atomic_write_new_json,
    completed_result,
    cpu_tree,
    frozen_snapshot,
    frozen_unchanged,
    git_revision,
    gradient_audit,
    latest_checkpoint,
    load_model_and_processor,
    optimizer_to_parameter_devices,
    read_json,
    seed_everything,
    selection_payload,
)


def distributed_context(config: dict[str, object]) -> tuple[int, int, int, torch.device]:
    distributed = config["distributed"]
    runtime = config["runtime"]
    assert isinstance(distributed, dict) and isinstance(runtime, dict)
    required_world = int(distributed["required_world_size"])
    required_visible = int(runtime["required_visible_device_count"])
    required_environment = ("RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE")
    missing = [name for name in required_environment if name not in os.environ]
    if missing:
        raise SystemExit(f"TORCHRUN_ENVIRONMENT_MISSING_{'_'.join(missing)}")
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
    if world_size != required_world or local_world_size != required_world:
        raise SystemExit(
            f"DDP_WORLD_SIZE_MUST_EQUAL_{required_world}_GOT_{world_size}_{local_world_size}"
        )
    if rank < 0 or rank >= world_size or local_rank < 0 or local_rank >= world_size:
        raise SystemExit("INVALID_DDP_RANK_ASSIGNMENT")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA_NOT_AVAILABLE")
    if torch.cuda.device_count() != required_visible:
        raise SystemExit(
            f"VISIBLE_CUDA_DEVICE_COUNT_MUST_EQUAL_{required_visible}_GOT_"
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


def broadcast_rank0_call(
    rank: int, function: Callable[[], object]
) -> object:
    envelope: list[object | None] = [None]
    if rank == 0:
        try:
            envelope[0] = {"ok": True, "value": function()}
        except Exception as exc:  # propagate a useful rank-zero failure to every worker
            envelope[0] = {
                "ok": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
    dist.broadcast_object_list(envelope, src=0)
    payload = envelope[0]
    if not isinstance(payload, dict) or payload.get("ok") is not True:
        raise RuntimeError(f"rank-zero operation failed: {payload}")
    return payload["value"]


def all_ranks_true(value: bool, device: torch.device) -> bool:
    tensor = torch.tensor(int(value), dtype=torch.int32, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.MIN)
    return bool(tensor.item())


def validate_runtime(
    config: dict[str, object], *, local_rank: int, world_size: int
) -> dict[str, object]:
    runtime = config["runtime"]
    assert isinstance(runtime, dict)
    required_workspace = str(runtime["required_cublas_workspace_config"])
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != required_workspace:
        raise SystemExit(f"CUBLAS_WORKSPACE_CONFIG_MUST_EQUAL_{required_workspace}")
    properties = torch.cuda.get_device_properties(local_rank)
    total_gib = properties.total_memory / 1024**3
    if total_gib < float(runtime["minimum_gpu_memory_gib"]):
        raise SystemExit(f"GPU_{local_rank}_MEMORY_TOO_SMALL_{total_gib:.2f}_GIB")
    import transformers

    actual_versions = {
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "transformers": transformers.__version__,
    }
    expected_versions = {
        "python": runtime["required_python"],
        "pytorch": runtime["required_pytorch"],
        "transformers": runtime["required_transformers"],
    }
    if actual_versions != expected_versions:
        raise SystemExit(f"RUNTIME_VERSION_MISMATCH_{actual_versions}")
    return {
        "rank": dist.get_rank(),
        "local_rank": local_rank,
        "world_size": world_size,
        "gpu": torch.cuda.get_device_name(local_rank),
        "gpu_total_memory_gib": total_gib,
        "cuda": torch.version.cuda,
        "versions": actual_versions,
    }


def verify_model_sentinels(
    model: torch.nn.Module, *, world_size: int, device: torch.device
) -> bool:
    trainable = [
        parameter.detach().reshape(-1)
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    if not trainable:
        raise RuntimeError("distributed model has no trainable parameters")
    chosen = [trainable[0], trainable[len(trainable) // 2], trainable[-1]]
    values: list[torch.Tensor] = []
    for tensor in chosen:
        indices = sorted({0, max(0, tensor.numel() // 2), tensor.numel() - 1})
        values.append(
            tensor.index_select(
                0, torch.tensor(indices, dtype=torch.long, device=tensor.device)
            ).float()
        )
    local = torch.cat(values).to(device)
    gathered = [torch.empty_like(local) for _ in range(world_size)]
    dist.all_gather(gathered, local)
    return all(torch.equal(gathered[0], item) for item in gathered[1:])


def checkpoint_metadata(
    directory: Path,
    *,
    config_sha256: str,
    manifest_sha256: str,
    selection_sha256: str,
    model_revision: str,
    world_size: int,
    global_batch_size: int,
    expected_step: int | None = None,
) -> dict[str, object]:
    metadata_path = directory / "metadata.json"
    if not metadata_path.is_file():
        raise RuntimeError(f"checkpoint metadata missing: {directory}")
    metadata = read_json(metadata_path)
    expected = {
        "config_sha256": config_sha256,
        "manifest_sha256": manifest_sha256,
        "selection_bundle_sha256": selection_sha256,
        "model_revision": model_revision,
        "world_size": world_size,
        "global_batch_size": global_batch_size,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise RuntimeError(f"checkpoint {key} differs: {directory}")
    step = metadata.get("step")
    if isinstance(step, bool) or not isinstance(step, int) or step <= 0:
        raise RuntimeError(f"bad checkpoint step: {directory}")
    if expected_step is not None and step != expected_step:
        raise RuntimeError(f"checkpoint step differs: {directory}")
    for name, key in (
        ("visual.safetensors", "visual_sha256"),
        ("optimizer.pt", "optimizer_sha256"),
    ):
        artifact = directory / name
        if not artifact.is_file() or sha256(artifact) != metadata.get(key):
            raise RuntimeError(f"checkpoint artifact differs: {artifact}")
    return metadata


def save_checkpoint(
    *,
    output_dir: Path,
    step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    config_sha256: str,
    manifest_sha256: str,
    selection_sha256: str,
    model_revision: str,
    world_size: int,
    global_batch_size: int,
) -> tuple[Path, dict[str, object]]:
    checkpoint_root = output_dir / "checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    destination = checkpoint_root / f"step_{step:06d}"
    if destination.exists():
        return destination, checkpoint_metadata(
            destination,
            config_sha256=config_sha256,
            manifest_sha256=manifest_sha256,
            selection_sha256=selection_sha256,
            model_revision=model_revision,
            world_size=world_size,
            global_batch_size=global_batch_size,
            expected_step=step,
        )
    temporary = Path(
        tempfile.mkdtemp(prefix=f".step_{step:06d}.partial-", dir=checkpoint_root)
    )
    visual_path = temporary / "visual.safetensors"
    optimizer_path = temporary / "optimizer.pt"
    visual_state = {
        name: tensor.detach().cpu().contiguous()
        for name, tensor in model.visual.state_dict().items()
    }
    save_file(visual_state, visual_path)
    del visual_state
    torch.save(
        {
            "step": step,
            "optimizer": cpu_tree(optimizer.state_dict()),
            "python_random_state": random.getstate(),
            "numpy_random_state": np.random.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state(device),
        },
        optimizer_path,
    )
    metadata: dict[str, object] = {
        "schema_version": 2,
        "scope": "qwen2_5_vl_full_visual_path_only_balanced_ddp",
        "step": step,
        "config_sha256": config_sha256,
        "manifest_sha256": manifest_sha256,
        "selection_bundle_sha256": selection_sha256,
        "model_revision": model_revision,
        "world_size": world_size,
        "global_batch_size": global_batch_size,
        "visual_sha256": sha256(visual_path),
        "optimizer_sha256": sha256(optimizer_path),
    }
    atomic_write_new_json(temporary / "metadata.json", metadata)
    if destination.exists():
        raise RuntimeError(f"checkpoint appeared concurrently: {destination}")
    os.rename(temporary, destination)
    return destination, metadata


def load_checkpoint(
    *,
    directory: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    config_sha256: str,
    manifest_sha256: str,
    selection_sha256: str,
    model_revision: str,
    world_size: int,
    global_batch_size: int,
    verified_metadata: dict[str, object] | None = None,
) -> tuple[int, dict[str, object]]:
    metadata = verified_metadata
    if metadata is None:
        metadata = checkpoint_metadata(
            directory,
            config_sha256=config_sha256,
            manifest_sha256=manifest_sha256,
            selection_sha256=selection_sha256,
            model_revision=model_revision,
            world_size=world_size,
            global_batch_size=global_batch_size,
        )
    visual_state = load_file(directory / "visual.safetensors", device="cpu")
    missing, unexpected = model.visual.load_state_dict(visual_state, strict=True)
    if missing or unexpected:
        raise RuntimeError(
            f"visual checkpoint key mismatch missing={missing} unexpected={unexpected}"
        )
    del visual_state
    state = torch.load(
        directory / "optimizer.pt", map_location="cpu", weights_only=False
    )
    if not isinstance(state, dict) or state.get("step") != metadata["step"]:
        raise RuntimeError("optimizer checkpoint step mismatch")
    optimizer.load_state_dict(state["optimizer"])
    optimizer_to_parameter_devices(optimizer)
    random.setstate(state["python_random_state"])
    np.random.set_state(state["numpy_random_state"])
    torch.set_rng_state(state["torch_rng_state"])
    torch.cuda.set_rng_state(state["cuda_rng_state"], device=device)
    return int(metadata["step"]), metadata


def prepare_local_batches(
    processor: Any,
    groups: Sequence[Sequence[PairRecord]],
    *,
    expected_micro_batch: int,
) -> list[dict[str, object]]:
    prepared: list[dict[str, object]] = []
    for group in groups:
        records = list(group)
        if len(records) != expected_micro_batch:
            raise RuntimeError("local source group changed microbatch cardinality")
        images = open_degraded_images(records)
        batch, target_counts = prepare_supervised_batch(processor, records, images)
        prepared.append(
            {
                "source_id": records[0].source_id,
                "records": records,
                "batch": batch,
                "target_tokens": sum(target_counts),
            }
        )
    return prepared


def move_batch(
    batch: dict[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=False) for key, value in batch.items()}


def distributed_probe_loss(
    model: torch.nn.Module,
    local_batches: Sequence[dict[str, object]],
    *,
    device: torch.device,
) -> float:
    was_training = model.training
    model.eval()
    numerator = 0.0
    tokens = 0
    with torch.no_grad():
        for prepared in local_batches:
            batch = prepared["batch"]
            assert isinstance(batch, dict)
            output = model(**move_batch(batch, device), use_cache=False)
            loss = float(output.loss.detach().float().item())
            target_tokens = int(prepared["target_tokens"])
            if not math.isfinite(loss) or target_tokens <= 0:
                raise RuntimeError("non-finite distributed probe loss")
            numerator += loss * target_tokens
            tokens += target_tokens
    totals = torch.tensor([numerator, tokens], dtype=torch.float64, device=device)
    dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    if was_training:
        model.train()
    if totals[1].item() <= 0:
        raise RuntimeError("distributed probe has no supervised tokens")
    return float((totals[0] / totals[1]).item())


def aggregate_audit(
    local: dict[str, object], *, rank: int, world_size: int
) -> dict[str, object]:
    gathered: list[object | None] = [None] * world_size
    dist.all_gather_object(gathered, {"rank": rank, **local})
    results = [item for item in gathered if isinstance(item, dict)]
    if len(results) != world_size:
        raise RuntimeError("distributed audit did not return every rank")
    return {
        **local,
        "distributed_all_ranks_passed": all(
            item.get("passed") is True for item in results
        ),
        "rank_results": results,
    }


def generate_local_outputs(
    model: torch.nn.Module,
    processor: Any,
    records: Sequence[PairRecord],
    selection_indices: dict[str, int],
    *,
    rank: int,
    device: torch.device,
    max_new_tokens: int,
) -> list[tuple[int, str]]:
    model.eval()
    model.config.use_cache = True
    fixed_prompt = prompt_text(processor)
    outputs: list[tuple[int, str]] = []
    for local_index, record in enumerate(records, start=1):
        image = open_degraded_images([record])[0]
        encoded = processor(
            text=[fixed_prompt], images=[image], padding=True, return_tensors="pt"
        )
        inputs = {
            key: value.to(device)
            for key, value in encoded.items()
            if isinstance(value, torch.Tensor)
        }
        input_length = int(inputs["input_ids"].shape[1])
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                use_cache=True,
            )
        raw = processor.tokenizer.decode(
            generated[0, input_length:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()
        outputs.append((selection_indices[record.pair_id], raw))
        print(
            f"QWEN_SUPERVISED_DDP_GENERATE rank={rank} "
            f"local={local_index}/{len(records)} pair={record.pair_id} output={raw!r}",
            flush=True,
        )
    model.config.use_cache = False
    return outputs


def verified_preflight(
    config_path: Path, output_dir: Path
) -> dict[str, object]:
    existing_path = output_dir / "preflight.json"
    if not existing_path.is_file():
        return run_preflight(config_path, skip_large_model_hashes=False)
    existing = read_json(existing_path)
    if existing.get("large_model_hashes_checked") is not True:
        raise RuntimeError("existing DDP preflight omitted full model hashes")
    current = run_preflight(config_path, skip_large_model_hashes=True)
    for key in (
        "config_sha256",
        "inventory_config_sha256",
        "model_revision",
        "parameter_groups",
        "selection_bundle_sha256",
        "rank_source_assignments",
        "versions",
    ):
        if existing.get(key) != current.get(key):
            raise RuntimeError(f"existing DDP preflight changed: {key}")
    return existing


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", choices=("auto", "never"), default="auto")
    parser.add_argument("--stop-after-step", type=int)
    args = parser.parse_args()

    config, config_sha, config_raw = load_ddp_config(args.config)
    training = config["training"]
    data = config["data"]
    distributed = config["distributed"]
    generation = config["generation"]
    gate = config["gate"]
    runtime = config["runtime"]
    assert all(
        isinstance(section, dict)
        for section in (training, data, distributed, generation, gate, runtime)
    )
    output_dir = workspace_path(runtime["output_dir"])
    rank = world_size = local_rank = -1
    process_group_ready = False
    overall_started = time.monotonic()
    try:
        rank, world_size, local_rank, device = distributed_context(config)
        process_group_ready = True
        existing = broadcast_rank0_call(
            rank, lambda: completed_result(output_dir, config_sha)
        )
        if existing is not None:
            return int(existing)

        preflight = broadcast_rank0_call(
            rank, lambda: verified_preflight(args.config, output_dir)
        )
        if not isinstance(preflight, dict):
            raise RuntimeError("distributed preflight did not return an object")
        if rank == 0:
            print(
                "QWEN_SUPERVISED_DDP_EMBEDDED_PREFLIGHT_PASS "
                + json.dumps(preflight, sort_keys=True),
                flush=True,
            )
        inventory, _ = load_inventory(config)
        inventory_model = inventory["model"]
        assert isinstance(inventory_model, dict)
        selected, probe = select_records(config)
        selection_sha = record_bundle_sha256(selected)
        if selection_sha != preflight["selection_bundle_sha256"]:
            raise RuntimeError("preflight selection changed before DDP training")
        local_groups = partition_source_groups(
            selected, world_size=world_size, rank=rank
        )
        if rank == 0:
            output_dir.mkdir(parents=True, exist_ok=True)
            atomic_write_new_bytes(output_dir / "config.json", config_raw)
            atomic_write_new_json(output_dir / "preflight.json", preflight)
            atomic_write_new_json(
                output_dir / "selection.json", selection_payload(selected, config_sha)
            )
        dist.barrier()

        runtime_evidence = validate_runtime(
            config, local_rank=local_rank, world_size=world_size
        )
        os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        seed_everything(int(training["seed"]))
        if runtime["deterministic_algorithms"] is True:
            torch.use_deterministic_algorithms(True)
        if runtime["sdpa_backend"] == "math":
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)
        model, processor, parameter_groups = load_model_and_processor(
            config, inventory, device
        )
        bad_dropout = [
            name
            for name, module in model.named_modules()
            if isinstance(module, torch.nn.Dropout) and module.p != 0.0
        ]
        if bad_dropout:
            raise RuntimeError(f"nonzero Qwen dropout breaks exact resume: {bad_dropout}")
        if training["gradient_checkpointing"] is True:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        model.train()
        initial_sentinels_match = verify_model_sentinels(
            model, world_size=world_size, device=device
        )
        if not initial_sentinels_match:
            raise RuntimeError("independently loaded DDP model parameters differ")

        ddp_model = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=bool(distributed["broadcast_buffers"]),
            init_sync=bool(distributed["ddp_init_sync"]),
            gradient_as_bucket_view=bool(distributed["gradient_as_bucket_view"]),
            find_unused_parameters=bool(distributed["find_unused_parameters"]),
        )
        local_batches = prepare_local_batches(
            processor,
            local_groups,
            expected_micro_batch=int(training["micro_batch_size_per_rank"]),
        )
        if len(local_batches) != int(training["gradient_accumulation_steps"]):
            raise RuntimeError("rank-local batch count differs from accumulation steps")

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
            raise RuntimeError("optimizer scope differs from the trainable visual path")

        total_steps = int(training["total_steps"])
        global_batch_size = int(training["global_batch_size"])
        validate_full_batch_contract(
            sample_count=len(selected),
            source_count=len({record.source_id for record in selected}),
            world_size=world_size,
            micro_batch_size=int(training["micro_batch_size_per_rank"]),
            gradient_accumulation_steps=int(
                training["gradient_accumulation_steps"]
            ),
            global_batch_size=global_batch_size,
        )
        start_step = 0
        resumed_metadata: dict[str, object] | None = None
        if args.resume == "auto":
            checkpoint = latest_checkpoint(output_dir, total_steps)
            if checkpoint is not None:
                verified_checkpoint = broadcast_rank0_call(
                    rank,
                    lambda: checkpoint_metadata(
                        checkpoint,
                        config_sha256=config_sha,
                        manifest_sha256=str(data["manifest_sha256"]),
                        selection_sha256=selection_sha,
                        model_revision=str(inventory_model["revision"]),
                        world_size=world_size,
                        global_batch_size=global_batch_size,
                    ),
                )
                if not isinstance(verified_checkpoint, dict):
                    raise RuntimeError("rank-zero checkpoint verification was invalid")
                start_step, resumed_metadata = load_checkpoint(
                    directory=checkpoint,
                    model=model,
                    optimizer=optimizer,
                    device=device,
                    config_sha256=config_sha,
                    manifest_sha256=str(data["manifest_sha256"]),
                    selection_sha256=selection_sha,
                    model_revision=str(inventory_model["revision"]),
                    world_size=world_size,
                    global_batch_size=global_batch_size,
                    verified_metadata=verified_checkpoint,
                )
                if not verify_model_sentinels(
                    model, world_size=world_size, device=device
                ):
                    raise RuntimeError("resumed DDP model parameters differ by rank")
                if rank == 0:
                    event = {
                        "schema_version": 1,
                        "config_sha256": config_sha,
                        "resumed_from_step": start_step,
                        "world_size": world_size,
                        "checkpoint_metadata_sha256": sha256(
                            checkpoint / "metadata.json"
                        ),
                    }
                    atomic_write_new_json(
                        output_dir
                        / "resume_events"
                        / f"step_{start_step:06d}.json",
                        event,
                    )
                    print(
                        f"QWEN_SUPERVISED_DDP_RESUME_VERIFIED step={start_step} "
                        f"checkpoint={checkpoint}",
                        flush=True,
                    )
        elif latest_checkpoint(output_dir, total_steps) is not None:
            raise RuntimeError("checkpoints exist but --resume=never was requested")
        dist.barrier()

        frozen_versions, frozen_sentinels = frozen_snapshot(model)
        initial_probe_path = output_dir / "initial_probe.json"
        if initial_probe_path.exists():
            initial_probe = read_json(initial_probe_path)
            if initial_probe.get("config_sha256") != config_sha or initial_probe.get(
                "probe_bundle_sha256"
            ) != record_bundle_sha256(probe):
                raise RuntimeError("existing initial DDP probe differs")
        else:
            if start_step != 0:
                raise RuntimeError("resumed checkpoint has no initial probe evidence")
            initial_loss = distributed_probe_loss(
                model, local_batches, device=device
            )
            initial_probe = {
                "schema_version": 1,
                "config_sha256": config_sha,
                "probe_bundle_sha256": record_bundle_sha256(probe),
                "records": len(probe),
                "teacher_forced_loss": initial_loss,
                "world_size": world_size,
            }
            if rank == 0:
                atomic_write_new_json(initial_probe_path, initial_probe)
                print(
                    f"QWEN_SUPERVISED_DDP_INITIAL_PROBE loss={initial_loss:.8f}",
                    flush=True,
                )
        dist.barrier()

        if args.stop_after_step is not None:
            if args.stop_after_step <= 0 or args.stop_after_step > total_steps:
                raise ValueError("--stop-after-step is outside the configured run")
            if start_step >= args.stop_after_step:
                required_audit = (
                    output_dir
                    / f"frozen_parameter_audit_step_{args.stop_after_step:06d}.json"
                )
                if not required_audit.is_file():
                    raise RuntimeError(
                        "intentional-stop checkpoint lacks its frozen-parameter audit"
                    )
                if rank == 0:
                    print(
                        "QWEN_SUPERVISED_DDP_INTENTIONAL_STOP_ALREADY_REACHED "
                        f"step={start_step}",
                        flush=True,
                    )
                return 0

        checkpoint_steps = {int(value) for value in training["checkpoint_steps"]}
        gradient_path = output_dir / "gradient_audit.json"
        accumulation_steps = int(training["gradient_accumulation_steps"])
        run_started = time.monotonic()
        for step in range(start_step + 1, total_steps + 1):
            learning_rate = cosine_learning_rate(
                step=step,
                total_steps=total_steps,
                warmup_steps=int(training["warmup_steps"]),
                base_learning_rate=float(training["learning_rate"]),
                minimum_learning_rate=float(training["minimum_learning_rate"]),
            )
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
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
                        raise RuntimeError(f"non-finite training loss at step {step}")
                    (loss / accumulation_steps).backward()
                target_tokens = int(prepared["target_tokens"])
                local_numerator += float(loss.detach().float().item()) * target_tokens
                local_tokens += target_tokens

            if not gradient_path.exists():
                local_gradient_audit = gradient_audit(model)
                distributed_gradient_audit = aggregate_audit(
                    local_gradient_audit, rank=rank, world_size=world_size
                )
                if rank == 0:
                    atomic_write_new_json(
                        gradient_path,
                        {
                            "schema_version": 1,
                            "config_sha256": config_sha,
                            "step": step,
                            **distributed_gradient_audit,
                        },
                    )
                    print(
                        "QWEN_SUPERVISED_DDP_GRADIENT_SCOPE_PASS "
                        + json.dumps(distributed_gradient_audit, sort_keys=True),
                        flush=True,
                    )
                if (
                    distributed_gradient_audit["distributed_all_ranks_passed"]
                    is not True
                ):
                    raise RuntimeError("distributed gradient-scope audit failed")
            frozen_with_grad = [
                name
                for name, parameter in model.named_parameters()
                if not parameter.requires_grad and parameter.grad is not None
            ]
            if not all_ranks_true(not frozen_with_grad, device):
                raise RuntimeError("a frozen decoder received gradients on a DDP rank")
            grad_norm = torch.nn.utils.clip_grad_norm_(
                trainable_parameters, float(training["max_grad_norm"])
            )
            if not torch.isfinite(grad_norm):
                raise RuntimeError(f"non-finite gradient norm at step {step}")
            optimizer.step()

            totals = torch.tensor(
                [local_numerator, local_tokens], dtype=torch.float64, device=device
            )
            dist.all_reduce(totals, op=dist.ReduceOp.SUM)
            global_loss = float((totals[0] / totals[1]).item())
            elapsed = time.monotonic() - run_started
            if rank == 0:
                metric = {
                    "schema_version": 1,
                    "config_sha256": config_sha,
                    "step": step,
                    "world_size": world_size,
                    "micro_batch_size_per_rank": int(
                        training["micro_batch_size_per_rank"]
                    ),
                    "gradient_accumulation_steps": accumulation_steps,
                    "global_batch_size": global_batch_size,
                    "samples_seen_cumulative": step * global_batch_size,
                    "supervised_tokens": int(totals[1].item()),
                    "loss": global_loss,
                    "gradient_norm_before_clip": float(
                        grad_norm.detach().float().item()
                    ),
                    "learning_rate": learning_rate,
                }
                atomic_write_new_json(
                    output_dir / "step_metrics" / f"step_{step:06d}.json",
                    metric,
                )
                print(
                    f"QWEN_SUPERVISED_DDP_STEP step={step}/{total_steps} "
                    f"loss={global_loss:.8f} lr={learning_rate:.9f} "
                    f"grad_norm={metric['gradient_norm_before_clip']:.6f} "
                    f"global_batch={global_batch_size} elapsed_s={elapsed:.1f}",
                    flush=True,
                )

            if args.stop_after_step == step:
                early_local = frozen_unchanged(
                    model, frozen_versions, frozen_sentinels
                )
                early_audit = aggregate_audit(
                    early_local, rank=rank, world_size=world_size
                )
                if rank == 0:
                    atomic_write_new_json(
                        output_dir
                        / f"frozen_parameter_audit_step_{step:06d}.json",
                        {
                            "schema_version": 1,
                            "config_sha256": config_sha,
                            **early_audit,
                        },
                    )
                if early_audit["distributed_all_ranks_passed"] is not True:
                    raise RuntimeError("frozen parameters changed before DDP stop")
            if step in checkpoint_steps:
                dist.barrier()
                saved: list[object | None] = [None]
                if rank == 0:
                    directory, metadata = save_checkpoint(
                        output_dir=output_dir,
                        step=step,
                        model=model,
                        optimizer=optimizer,
                        device=device,
                        config_sha256=config_sha,
                        manifest_sha256=str(data["manifest_sha256"]),
                        selection_sha256=selection_sha,
                        model_revision=str(inventory_model["revision"]),
                        world_size=world_size,
                        global_batch_size=global_batch_size,
                    )
                    saved[0] = {
                        "directory": str(directory),
                        "metadata": metadata,
                    }
                dist.broadcast_object_list(saved, src=0)
                if rank == 0:
                    assert isinstance(saved[0], dict)
                    metadata = saved[0]["metadata"]
                    print(
                        f"QWEN_SUPERVISED_DDP_CHECKPOINT step={step} "
                        f"path={saved[0]['directory']} "
                        f"visual_sha256={metadata['visual_sha256']}",
                        flush=True,
                    )
                dist.barrier()
            if args.stop_after_step == step:
                if step not in checkpoint_steps:
                    raise RuntimeError("intentional DDP stop step was not checkpointed")
                if rank == 0:
                    atomic_write_new_json(
                        output_dir / f"intentional_stop_step_{step:06d}.json",
                        {
                            "schema_version": 1,
                            "config_sha256": config_sha,
                            "intentional_stop_step": step,
                            "world_size": world_size,
                            "checkpoint": f"checkpoints/step_{step:06d}",
                        },
                    )
                    print(
                        f"QWEN_SUPERVISED_DDP_INTENTIONAL_STOP step={step}",
                        flush=True,
                    )
                return 0

        training_elapsed = time.monotonic() - run_started
        required_resume_step = int(gate["required_resume_step"])
        resume_event_path = (
            output_dir / "resume_events" / f"step_{required_resume_step:06d}.json"
        )
        resume_verified = resume_event_path.is_file()
        if resume_verified:
            resume_event = read_json(resume_event_path)
            resume_verified = (
                resume_event.get("config_sha256") == config_sha
                and resume_event.get("resumed_from_step") == required_resume_step
                and resume_event.get("world_size") == world_size
            )
        early_frozen_path = (
            output_dir
            / f"frozen_parameter_audit_step_{required_resume_step:06d}.json"
        )
        early_frozen_verified = False
        if early_frozen_path.is_file():
            early_result = read_json(early_frozen_path)
            early_frozen_verified = (
                early_result.get("config_sha256") == config_sha
                and early_result.get("passed") is True
                and early_result.get("distributed_all_ranks_passed") is True
            )

        model.gradient_checkpointing_disable()
        final_loss = distributed_probe_loss(model, local_batches, device=device)
        local_records = [
            record for group in local_groups for record in group
        ]
        selection_indices = {
            record.pair_id: index for index, record in enumerate(selected)
        }
        local_outputs = generate_local_outputs(
            model,
            processor,
            local_records,
            selection_indices,
            rank=rank,
            device=device,
            max_new_tokens=int(generation["max_new_tokens"]),
        )
        gathered_outputs: list[object | None] = [None] * world_size
        dist.all_gather_object(gathered_outputs, local_outputs)
        raw_outputs = merge_indexed_outputs(
            [item for item in gathered_outputs if isinstance(item, list)],
            expected_count=len(selected),
        )
        prediction_rows, understanding_metrics = evaluate_generated_outputs(
            selected, raw_outputs
        )
        if rank == 0:
            atomic_write_new_json(
                output_dir / "predictions.json",
                {
                    "schema_version": 1,
                    "config_sha256": config_sha,
                    "selection_bundle_sha256": selection_sha,
                    "records": prediction_rows,
                },
            )

        local_frozen_audit = frozen_unchanged(
            model, frozen_versions, frozen_sentinels
        )
        frozen_audit = aggregate_audit(
            local_frozen_audit, rank=rank, world_size=world_size
        )
        if rank == 0:
            atomic_write_new_json(
                output_dir / "frozen_parameter_audit.json",
                {
                    "schema_version": 1,
                    "config_sha256": config_sha,
                    **frozen_audit,
                },
            )
        gradient_result = read_json(gradient_path)
        if gradient_result.get("config_sha256") != config_sha:
            raise RuntimeError("gradient audit belongs to another config")
        initial_loss = float(initial_probe["teacher_forced_loss"])
        final_to_initial = final_loss / initial_loss
        per_level_minimum = min(
            float(item["accuracy"])
            for item in understanding_metrics["per_level"].values()
        )
        final_sentinels_match = verify_model_sentinels(
            model, world_size=world_size, device=device
        )
        gate_checks = {
            "resume_checkpoint_verified": resume_verified,
            "gradient_scope_passed": gradient_result.get(
                "distributed_all_ranks_passed"
            )
            is True,
            "frozen_parameters_unchanged": early_frozen_verified
            and frozen_audit["distributed_all_ranks_passed"] is True,
            "distributed_model_equivalence_passed": initial_sentinels_match
            and final_sentinels_match,
            "full_global_batch_semantics_passed": True,
            "final_probe_loss_passed": final_loss
            <= float(gate["maximum_final_probe_loss"]),
            "probe_loss_ratio_passed": final_to_initial
            <= float(gate["maximum_final_to_initial_probe_loss_ratio"]),
            "structured_output_passed": float(
                understanding_metrics["valid_structured_output_rate"]
            )
            >= float(gate["minimum_valid_structured_output_rate"]),
            "degradation_type_passed": float(
                understanding_metrics["degradation_type_accuracy"]
            )
            >= float(gate["minimum_degradation_type_accuracy"]),
            "severity_memorization_passed": float(
                understanding_metrics["severity_accuracy"]
            )
            >= float(gate["minimum_severity_accuracy"]),
            "every_level_memorized": per_level_minimum
            >= float(gate["minimum_per_level_accuracy"]),
        }
        automatic_pass = all(gate_checks.values())
        final_checkpoint = output_dir / "checkpoints" / f"step_{total_steps:06d}"
        final_checkpoint_meta = broadcast_rank0_call(
            rank,
            lambda: checkpoint_metadata(
                final_checkpoint,
                config_sha256=config_sha,
                manifest_sha256=str(data["manifest_sha256"]),
                selection_sha256=selection_sha,
                model_revision=str(inventory_model["revision"]),
                world_size=world_size,
                global_batch_size=global_batch_size,
                expected_step=total_steps,
            ),
        )
        if not isinstance(final_checkpoint_meta, dict):
            raise RuntimeError("final checkpoint metadata broadcast was invalid")
        runtime_evidence["maximum_allocated_gpu_memory_gib"] = (
            torch.cuda.max_memory_allocated(device) / 1024**3
        )
        gathered_runtime: list[object | None] = [None] * world_size
        dist.all_gather_object(gathered_runtime, runtime_evidence)
        overall_elapsed = time.monotonic() - overall_started
        if rank == 0:
            summary: dict[str, object] = {
                "schema_version": 1,
                "status": "passed" if automatic_pass else "failed",
                "scope": "synthetic_lowlight_supervised_vlm_overfit40_ddp8_retry_gate",
                "config_sha256": config_sha,
                "git_revision": git_revision(),
                "model_repository": inventory_model["repository"],
                "model_revision": inventory_model["revision"],
                "selection_bundle_sha256": selection_sha,
                "manifest_sha256": data["manifest_sha256"],
                "parameter_groups": parameter_groups,
                "trainable_parameters": config["model"][
                    "expected_trainable_parameters"
                ],
                "frozen_parameters": config["model"][
                    "expected_frozen_parameters"
                ],
                "initial_probe_loss": initial_loss,
                "final_probe_loss": final_loss,
                "final_to_initial_probe_loss_ratio": final_to_initial,
                "understanding_metrics": understanding_metrics,
                "gradient_audit": gradient_result,
                "frozen_parameter_audit": frozen_audit,
                "early_frozen_parameter_audit_verified": early_frozen_verified,
                "resume_verified": resume_verified,
                "resumed_metadata": resumed_metadata,
                "world_size": world_size,
                "micro_batch_size_per_rank": int(
                    training["micro_batch_size_per_rank"]
                ),
                "gradient_accumulation_steps": accumulation_steps,
                "global_batch_size": global_batch_size,
                "optimizer_steps_this_process": total_steps - start_step,
                "samples_seen_this_process": (total_steps - start_step)
                * global_batch_size,
                "samples_seen_total": total_steps * global_batch_size,
                "training_elapsed_seconds_this_process": training_elapsed,
                "end_to_end_elapsed_seconds_this_process": overall_elapsed,
                "optimizer_steps_per_second_this_process": (
                    (total_steps - start_step) / training_elapsed
                ),
                "global_samples_per_second_this_process": (
                    (total_steps - start_step)
                    * global_batch_size
                    / training_elapsed
                ),
                "final_checkpoint": str(final_checkpoint.relative_to(output_dir)),
                "final_checkpoint_metadata": final_checkpoint_meta,
                "gate_checks": gate_checks,
                "automatic_gate_pass": automatic_pass,
                "environment_by_rank": gathered_runtime,
                "cublas_workspace_config": os.environ[
                    "CUBLAS_WORKSPACE_CONFIG"
                ],
                "sdpa_backend": runtime["sdpa_backend"],
                "synthetic_internal_only": True,
                "scientific_pilot_result": False,
                "test_split_used": False,
                "official_foundir_used": False,
            }
            atomic_write_new_json(output_dir / "training_summary.json", summary)
            marker = (
                "QWEN_SUPERVISED_DDP8_OVERFIT40_PASS"
                if automatic_pass
                else "QWEN_SUPERVISED_DDP8_OVERFIT40_FAILED"
            )
            print(marker + " " + json.dumps(gate_checks, sort_keys=True), flush=True)
        result_tensor = torch.tensor(
            int(automatic_pass), dtype=torch.int32, device=device
        )
        dist.broadcast(result_tensor, src=0)
        return 0 if bool(result_tensor.item()) else 2
    finally:
        if process_group_ready and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
