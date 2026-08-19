#!/usr/bin/env python3
"""Train and evaluate the source-disjoint Qwen Lowlight 8k DDP8 pilot."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.distributed as dist
from safetensors.torch import load_file
from torch.nn.parallel import DistributedDataParallel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lowlight_restoration.data import PairRecord, load_pair_manifest
from lowlight_restoration.qwen_distributed import (
    cosine_learning_rate,
    merge_indexed_outputs,
    partition_source_groups,
)
from lowlight_restoration.qwen_pilot import (
    augment_understanding_metrics,
    epoch_local_order,
    load_pilot_config,
    metric_gate_checks,
    records_by_split,
    select_best_validation_checkpoint,
    training_position,
)
from lowlight_restoration.qwen_supervised import (
    open_degraded_images,
    prepare_supervised_batch,
    prompt_text,
    record_bundle_sha256,
)
from smoke.qwen_lowlight_supervised_pilot_preflight import run_preflight
from smoke.qwen_lowlight_supervised_preflight import (
    load_inventory,
    sha256,
    workspace_path,
)
from smoke.train_qwen_lowlight_supervised import (
    atomic_write_new_bytes,
    atomic_write_new_json,
    frozen_snapshot,
    frozen_unchanged,
    git_revision,
    gradient_audit,
    latest_checkpoint,
    load_model_and_processor,
    read_json,
    seed_everything,
)
from smoke.train_qwen_lowlight_supervised_ddp import (
    aggregate_audit,
    all_ranks_true,
    broadcast_rank0_call,
    checkpoint_metadata,
    distributed_context,
    distributed_probe_loss,
    load_checkpoint,
    move_batch,
    save_checkpoint,
    validate_runtime,
    verify_model_sentinels,
)


def verified_preflight(config_path: Path, output_dir: Path) -> dict[str, object]:
    full_path = output_dir / "preflight_full.json"
    if not full_path.is_file():
        raise RuntimeError(
            "full pilot preflight is missing; run the checked launcher, which creates it"
        )
    full = read_json(full_path)
    if full.get("large_model_hashes_checked") is not True:
        raise RuntimeError("full pilot preflight omitted large model hashes")
    current = run_preflight(
        config_path,
        skip_large_model_hashes=True,
        write_artifact=True,
        show_progress=True,
    )
    stable_keys = (
        "config_sha256",
        "inventory_config_sha256",
        "manifest_sha256",
        "model_revision",
        "model_files",
        "model_hash_contract_sha256",
        "parameter_groups",
        "tensor_count",
        "overfit_gate_evidence",
        "split_evidence",
        "image_file_evidence",
        "world_size",
        "rank_source_assignments",
        "visual_tokens",
        "assistant_target_tokens",
        "processor",
        "image_processor",
        "tokenizer",
        "transformers_modeling_source",
        "versions",
        "test_split_content_verified_only",
        "test_split_used_for_training_or_selection",
        "official_foundir_used",
    )
    for key in stable_keys:
        if full.get(key) != current.get(key):
            raise RuntimeError(f"full pilot preflight evidence changed: {key}")
    return full


def _split_selection_payload(
    splits: dict[str, list[PairRecord]], config_sha256: str
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "scope": "synthetic_lowlight_source_disjoint_pilot_splits",
        "config_sha256": config_sha256,
        "splits": {
            split: {
                "bundle_sha256": record_bundle_sha256(records),
                "sources": sorted({record.source_id for record in records}),
                "records": [
                    {
                        "pair_id": record.pair_id,
                        "source_id": record.source_id,
                        "level": record.level_index,
                        "degraded_path": str(record.degraded_path.relative_to(ROOT)),
                        "degraded_sha256": record.degraded_sha256,
                    }
                    for record in records
                ],
            }
            for split, records in splits.items()
        },
        "test_split_used_for_training_or_selection": False,
        "official_foundir_used": False,
    }


def prepare_split_batches(
    processor: Any,
    groups: Sequence[Sequence[PairRecord]],
    *,
    split: str,
    rank: int,
    expected_micro_batch: int,
) -> list[dict[str, object]]:
    prepared: list[dict[str, object]] = []
    for index, group in enumerate(groups, start=1):
        records = list(group)
        if len(records) != expected_micro_batch:
            raise RuntimeError(f"{split} source group changed microbatch cardinality")
        images = open_degraded_images(records)
        batch, target_counts = prepare_supervised_batch(processor, records, images)
        del images
        prepared.append(
            {
                "source_id": records[0].source_id,
                "records": records,
                "batch": batch,
                "target_tokens": sum(target_counts),
            }
        )
        if index % 25 == 0 or index == len(groups):
            print(
                f"QWEN_PILOT_PREPROCESS rank={rank} split={split} "
                f"groups={index}/{len(groups)} records={index * len(records)}",
                flush=True,
            )
    return prepared


def _validation_loss_entry(
    *,
    model: torch.nn.Module,
    local_batches: Sequence[dict[str, object]],
    output_dir: Path,
    step: int,
    checkpoint_meta: dict[str, object] | None,
    config_sha256: str,
    validation_bundle_sha256: str,
    rank: int,
    device: torch.device,
) -> dict[str, object]:
    path = output_dir / "validation_losses" / f"step_{step:06d}.json"
    existing = broadcast_rank0_call(
        rank, lambda: read_json(path) if path.is_file() else None
    )
    if existing is not None:
        if not isinstance(existing, dict):
            raise RuntimeError("existing validation-loss evidence is not an object")
        expected_visual = (
            None if checkpoint_meta is None else checkpoint_meta.get("visual_sha256")
        )
        if (
            existing.get("config_sha256") != config_sha256
            or existing.get("validation_bundle_sha256")
            != validation_bundle_sha256
            or existing.get("step") != step
            or existing.get("checkpoint_visual_sha256") != expected_visual
            or isinstance(existing.get("teacher_forced_loss"), bool)
            or not isinstance(existing.get("teacher_forced_loss"), (int, float))
            or not math.isfinite(float(existing["teacher_forced_loss"]))
        ):
            raise RuntimeError(f"existing validation-loss evidence differs at step {step}")
        if rank == 0:
            print(
                f"QWEN_PILOT_VALIDATION_REUSED step={step} "
                f"loss={float(existing['teacher_forced_loss']):.8f}",
                flush=True,
            )
        return existing

    if rank == 0:
        print(f"QWEN_PILOT_VALIDATION_BEGIN step={step}", flush=True)
    started = time.monotonic()
    loss = distributed_probe_loss(model, local_batches, device=device)
    payload: dict[str, object] = {
        "schema_version": 1,
        "config_sha256": config_sha256,
        "validation_bundle_sha256": validation_bundle_sha256,
        "step": step,
        "checkpoint_visual_sha256": (
            None if checkpoint_meta is None else checkpoint_meta["visual_sha256"]
        ),
        "teacher_forced_loss": loss,
        "records": 1000,
        "world_size": dist.get_world_size(),
    }
    if rank == 0:
        atomic_write_new_json(path, payload)
        print(
            f"QWEN_PILOT_VALIDATION_END step={step} loss={loss:.8f} "
            f"elapsed_s={time.monotonic() - started:.1f}",
            flush=True,
        )
    dist.barrier()
    return payload


def _checkpoint_freeze_audit(
    *,
    model: torch.nn.Module,
    frozen_versions: dict[str, int],
    frozen_sentinels: dict[str, torch.Tensor],
    output_dir: Path,
    step: int,
    config_sha256: str,
    rank: int,
    world_size: int,
) -> dict[str, object]:
    local = frozen_unchanged(model, frozen_versions, frozen_sentinels)
    audit = aggregate_audit(local, rank=rank, world_size=world_size)
    payload = {
        "schema_version": 1,
        "config_sha256": config_sha256,
        "step": step,
        **audit,
    }
    if rank == 0:
        atomic_write_new_json(
            output_dir / "frozen_parameter_audits" / f"step_{step:06d}.json",
            payload,
        )
        print(
            f"QWEN_PILOT_FROZEN_AUDIT step={step} "
            f"passed={audit['distributed_all_ranks_passed']}",
            flush=True,
        )
    if audit["distributed_all_ranks_passed"] is not True:
        raise RuntimeError(f"frozen Qwen decoder changed at checkpoint step {step}")
    return payload


def _load_selected_visual(
    *,
    checkpoint: Path,
    model: torch.nn.Module,
    config_sha256: str,
    manifest_sha256: str,
    train_bundle_sha256: str,
    model_revision: str,
    world_size: int,
    global_batch_size: int,
    selected_step: int,
) -> dict[str, object]:
    metadata = checkpoint_metadata(
        checkpoint,
        config_sha256=config_sha256,
        manifest_sha256=manifest_sha256,
        selection_sha256=train_bundle_sha256,
        model_revision=model_revision,
        world_size=world_size,
        global_batch_size=global_batch_size,
        expected_step=selected_step,
    )
    state = load_file(checkpoint / "visual.safetensors", device="cpu")
    missing, unexpected = model.visual.load_state_dict(state, strict=True)
    if missing or unexpected:
        raise RuntimeError(
            f"selected visual checkpoint keys differ missing={missing} unexpected={unexpected}"
        )
    return metadata


def generate_local_outputs_batched(
    model: torch.nn.Module,
    processor: Any,
    groups: Sequence[Sequence[PairRecord]],
    selection_indices: dict[str, int],
    *,
    split: str,
    rank: int,
    device: torch.device,
    max_new_tokens: int,
) -> list[tuple[int, str]]:
    model.eval()
    model.config.use_cache = True
    fixed_prompt = prompt_text(processor)
    original_padding_side = processor.tokenizer.padding_side
    processor.tokenizer.padding_side = "left"
    outputs: list[tuple[int, str]] = []
    try:
        for group_index, group in enumerate(groups, start=1):
            records = list(group)
            if len(records) != 5:
                raise RuntimeError(f"{split} generation group is not complete L1--L5")
            images = open_degraded_images(records)
            encoded = processor(
                text=[fixed_prompt] * len(records),
                images=images,
                padding=True,
                return_tensors="pt",
            )
            del images
            inputs = {
                key: value.to(device)
                for key, value in encoded.items()
                if isinstance(value, torch.Tensor)
            }
            if not torch.all(inputs["attention_mask"][:, -1] == 1):
                raise RuntimeError("left-padded generation ends in a padding token")
            prompt_width = int(inputs["input_ids"].shape[1])
            with torch.no_grad():
                generated = model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=max_new_tokens,
                    use_cache=True,
                )
            if int(generated.shape[0]) != len(records):
                raise RuntimeError("batched generation cardinality changed")
            for row, record in enumerate(records):
                raw = processor.tokenizer.decode(
                    generated[row, prompt_width:],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                ).strip()
                outputs.append((selection_indices[record.pair_id], raw))
            print(
                f"QWEN_PILOT_GENERATE rank={rank} split={split} "
                f"groups={group_index}/{len(groups)} records={len(outputs)}",
                flush=True,
            )
    finally:
        processor.tokenizer.padding_side = original_padding_side
        model.config.use_cache = False
    return outputs


def _evaluate_split(
    *,
    model: torch.nn.Module,
    processor: Any,
    records: Sequence[PairRecord],
    local_groups: Sequence[Sequence[PairRecord]],
    output_dir: Path,
    split: str,
    config_sha256: str,
    selected_step: int,
    rank: int,
    world_size: int,
    device: torch.device,
    max_new_tokens: int,
    maximum_failures: int,
) -> tuple[dict[str, object], dict[str, str]]:
    if rank == 0:
        print(
            f"QWEN_PILOT_GENERATION_BEGIN split={split} records={len(records)}",
            flush=True,
        )
    started = time.monotonic()
    indices = {record.pair_id: index for index, record in enumerate(records)}
    local_outputs = generate_local_outputs_batched(
        model,
        processor,
        local_groups,
        indices,
        split=split,
        rank=rank,
        device=device,
        max_new_tokens=max_new_tokens,
    )
    gathered: list[object | None] = [None] * world_size
    dist.all_gather_object(gathered, local_outputs)
    raw_outputs = merge_indexed_outputs(
        [item for item in gathered if isinstance(item, list)],
        expected_count=len(records),
    )
    from lowlight_restoration.qwen_supervised import evaluate_generated_outputs

    prediction_rows, base_metrics = evaluate_generated_outputs(records, raw_outputs)
    metrics = augment_understanding_metrics(prediction_rows, base_metrics)
    metrics_payload: dict[str, object] = {
        "schema_version": 1,
        "config_sha256": config_sha256,
        "split": split,
        "split_bundle_sha256": record_bundle_sha256(records),
        "selected_checkpoint_step": selected_step,
        "metrics": metrics,
    }
    prediction_payload = {
        "schema_version": 1,
        "config_sha256": config_sha256,
        "split": split,
        "selected_checkpoint_step": selected_step,
        "records": prediction_rows,
    }
    failures = [row for row in prediction_rows if row["severity_correct"] is not True]
    failure_payload = {
        "schema_version": 1,
        "config_sha256": config_sha256,
        "split": split,
        "selected_checkpoint_step": selected_step,
        "total_failures": len(failures),
        "reported_failures": failures[:maximum_failures],
    }
    confusion_payload = {
        "schema_version": 1,
        "config_sha256": config_sha256,
        "split": split,
        "selected_checkpoint_step": selected_step,
        "severity_confusion": metrics["confusion"],
        "degradation_type_f1": metrics["degradation_type_f1"],
        "severity_f1": metrics["severity_f1"],
    }
    artifacts = {
        "metrics": output_dir / "evaluation" / f"{split}_metrics.json",
        "predictions": output_dir / "evaluation" / f"{split}_predictions.json",
        "failures": output_dir / "evaluation" / f"{split}_failures.json",
        "confusion": output_dir / "evaluation" / f"{split}_confusion.json",
    }
    if rank == 0:
        atomic_write_new_json(artifacts["metrics"], metrics_payload)
        atomic_write_new_json(artifacts["predictions"], prediction_payload)
        atomic_write_new_json(artifacts["failures"], failure_payload)
        atomic_write_new_json(artifacts["confusion"], confusion_payload)
        print(
            f"QWEN_PILOT_GENERATION_END split={split} "
            f"severity_accuracy={float(metrics['severity_accuracy']):.6f} "
            f"valid_json={float(metrics['valid_structured_output_rate']):.6f} "
            f"elapsed_s={time.monotonic() - started:.1f}",
            flush=True,
        )
    dist.barrier()
    hashes = broadcast_rank0_call(
        rank,
        lambda: {name: sha256(path) for name, path in artifacts.items()},
    )
    if not isinstance(hashes, dict):
        raise RuntimeError(f"{split} evaluation hashes are invalid")
    return metrics, {str(key): str(value) for key, value in hashes.items()}


def _validate_step_evidence(
    output_dir: Path, *, config_sha256: str, total_steps: int
) -> bool:
    directory = output_dir / "step_metrics"
    expected = {f"step_{step:06d}.json" for step in range(1, total_steps + 1)}
    actual = {path.name for path in directory.glob("step_*.json")}
    if actual != expected:
        return False
    for step in range(1, total_steps + 1):
        payload = read_json(directory / f"step_{step:06d}.json")
        if payload.get("config_sha256") != config_sha256 or payload.get("step") != step:
            return False
    return True


def completed_pilot_result(output_dir: Path, config_sha256: str) -> int | None:
    path = output_dir / "training_summary.json"
    if not path.is_file():
        return None
    summary = read_json(path)
    checks = summary.get("gate_checks")
    metadata = summary.get("selected_checkpoint_metadata")
    checkpoint_value = summary.get("selected_checkpoint")
    if (
        summary.get("config_sha256") != config_sha256
        or not isinstance(checks, dict)
        or not checks
        or not isinstance(metadata, dict)
        or not isinstance(checkpoint_value, str)
    ):
        raise RuntimeError("existing pilot summary is incomplete or belongs elsewhere")
    checkpoint = (output_dir / checkpoint_value).resolve()
    try:
        checkpoint.relative_to(output_dir.resolve())
    except ValueError as exc:
        raise RuntimeError("existing selected checkpoint escapes pilot output") from exc
    if read_json(checkpoint / "metadata.json") != metadata:
        raise RuntimeError("existing selected checkpoint metadata changed")
    if sha256(checkpoint / "visual.safetensors") != metadata.get("visual_sha256"):
        raise RuntimeError("existing selected visual checkpoint changed")
    status = summary.get("status")
    automatic_pass = summary.get("automatic_gate_pass")
    if status == "passed" and automatic_pass is True and all(checks.values()):
        print("QWEN_PILOT_8K_PASS existing_verified_result", flush=True)
        return 0
    if status == "failed" and automatic_pass is False and not all(checks.values()):
        print("QWEN_PILOT_8K_FAILED existing_verified_result", flush=True)
        return 2
    raise RuntimeError("existing pilot summary has inconsistent terminal status")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", choices=("auto", "never"), default="auto")
    parser.add_argument("--stop-after-step", type=int)
    args = parser.parse_args()

    config, config_sha, config_raw = load_pilot_config(args.config)
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
            rank, lambda: completed_pilot_result(output_dir, config_sha)
        )
        if existing is not None:
            return int(existing)

        preflight = broadcast_rank0_call(
            rank, lambda: verified_preflight(args.config, output_dir)
        )
        if not isinstance(preflight, dict):
            raise RuntimeError("pilot preflight broadcast is invalid")
        if rank == 0:
            print(
                "QWEN_PILOT_EMBEDDED_PREFLIGHT_PASS "
                + json.dumps(
                    {
                        "config_sha256": preflight["config_sha256"],
                        "manifest_sha256": preflight["manifest_sha256"],
                        "image_file_evidence": preflight["image_file_evidence"],
                        "model_revision": preflight["model_revision"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

        inventory, _ = load_inventory(config)
        inventory_model = inventory["model"]
        assert isinstance(inventory_model, dict)
        records = load_pair_manifest(workspace_path(data["manifest"]))
        splits = records_by_split(records, config)
        train_bundle = record_bundle_sha256(splits["train"])
        validation_bundle = record_bundle_sha256(splits["validation"])
        test_bundle = record_bundle_sha256(splits["test"])
        local_groups = {
            split: partition_source_groups(
                split_records, world_size=world_size, rank=rank
            )
            for split, split_records in splits.items()
        }
        if rank == 0:
            output_dir.mkdir(parents=True, exist_ok=True)
            atomic_write_new_bytes(output_dir / "config.json", config_raw)
            atomic_write_new_json(
                output_dir / "split_selection.json",
                _split_selection_payload(splits, config_sha),
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
            raise RuntimeError("independently loaded pilot model parameters differ")

        ddp_model = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=bool(distributed["broadcast_buffers"]),
            init_sync=bool(distributed["ddp_init_sync"]),
            gradient_as_bucket_view=bool(distributed["gradient_as_bucket_view"]),
            find_unused_parameters=bool(distributed["find_unused_parameters"]),
        )
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
            raise RuntimeError("pilot optimizer scope differs from full visual path")

        total_steps = int(training["total_steps"])
        global_batch_size = int(training["global_batch_size"])
        checkpoint_steps = [int(value) for value in training["checkpoint_steps"]]
        validation_steps = [int(value) for value in training["validation_loss_steps"]]
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
                        selection_sha256=train_bundle,
                        model_revision=str(inventory_model["revision"]),
                        world_size=world_size,
                        global_batch_size=global_batch_size,
                    ),
                )
                if not isinstance(verified_checkpoint, dict):
                    raise RuntimeError("pilot checkpoint verification was invalid")
                start_step, resumed_metadata = load_checkpoint(
                    directory=checkpoint,
                    model=model,
                    optimizer=optimizer,
                    device=device,
                    config_sha256=config_sha,
                    manifest_sha256=str(data["manifest_sha256"]),
                    selection_sha256=train_bundle,
                    model_revision=str(inventory_model["revision"]),
                    world_size=world_size,
                    global_batch_size=global_batch_size,
                    verified_metadata=verified_checkpoint,
                )
                if start_step not in checkpoint_steps:
                    raise RuntimeError("pilot resumed from a non-scheduled checkpoint")
                if not verify_model_sentinels(model, world_size=world_size, device=device):
                    raise RuntimeError("resumed pilot model parameters differ by rank")
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
                        output_dir / "resume_events" / f"step_{start_step:06d}.json",
                        event,
                    )
                    print(
                        f"QWEN_PILOT_RESUME_VERIFIED step={start_step} "
                        f"checkpoint={checkpoint}",
                        flush=True,
                    )
        elif latest_checkpoint(output_dir, total_steps) is not None:
            raise RuntimeError("pilot checkpoints exist but --resume=never was requested")
        dist.barrier()

        if args.stop_after_step is not None and (
            args.stop_after_step not in checkpoint_steps
            or args.stop_after_step <= start_step
        ):
            raise ValueError("--stop-after-step must be a future scheduled checkpoint")

        frozen_versions, frozen_sentinels = frozen_snapshot(model)
        expected_micro_batch = int(training["micro_batch_size_per_rank"])
        local_train_batches = prepare_split_batches(
            processor,
            local_groups["train"],
            split="train",
            rank=rank,
            expected_micro_batch=expected_micro_batch,
        )
        local_validation_batches = prepare_split_batches(
            processor,
            local_groups["validation"],
            split="validation",
            rank=rank,
            expected_micro_batch=expected_micro_batch,
        )
        if len(local_train_batches) != int(training["steps_per_epoch"]):
            raise RuntimeError("rank-local train source count differs from epoch steps")
        initial_validation_path = (
            output_dir / "validation_losses" / "step_000000.json"
        )
        if start_step > 0 and not initial_validation_path.is_file():
            raise RuntimeError(
                "resumed pilot checkpoint cannot reconstruct missing step-zero validation"
            )

        initial_validation = _validation_loss_entry(
            model=model,
            local_batches=local_validation_batches,
            output_dir=output_dir,
            step=0,
            checkpoint_meta=None,
            config_sha256=config_sha,
            validation_bundle_sha256=validation_bundle,
            rank=rank,
            device=device,
        )
        if start_step in validation_steps and start_step > 0:
            if resumed_metadata is None:
                raise RuntimeError("resumed validation step has no checkpoint metadata")
            _validation_loss_entry(
                model=model,
                local_batches=local_validation_batches,
                output_dir=output_dir,
                step=start_step,
                checkpoint_meta=resumed_metadata,
                config_sha256=config_sha,
                validation_bundle_sha256=validation_bundle,
                rank=rank,
                device=device,
            )
        gradient_path = output_dir / "gradient_audit.json"
        if start_step > 0 and not gradient_path.is_file():
            raise RuntimeError("resumed pilot checkpoint has no first-step gradient audit")

        training_started = time.monotonic()
        for step in range(start_step + 1, total_steps + 1):
            epoch_index, local_position = training_position(
                step, steps_per_epoch=int(training["steps_per_epoch"])
            )
            order = epoch_local_order(
                local_source_count=len(local_train_batches),
                seed=int(training["seed"]),
                epoch_index=epoch_index,
                rank=rank,
            )
            prepared = local_train_batches[order[local_position]]
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
            batch = prepared["batch"]
            assert isinstance(batch, dict)
            output = ddp_model(**move_batch(batch, device), use_cache=False)
            loss = output.loss
            if loss is None or not torch.isfinite(loss):
                raise RuntimeError(f"non-finite pilot training loss at step {step}")
            loss.backward()

            if step == 1:
                local_gradient = gradient_audit(model)
                distributed_gradient = aggregate_audit(
                    local_gradient, rank=rank, world_size=world_size
                )
                gradient_payload = {
                    "schema_version": 1,
                    "config_sha256": config_sha,
                    "step": 1,
                    **distributed_gradient,
                }
                if rank == 0:
                    atomic_write_new_json(gradient_path, gradient_payload)
                    print(
                        "QWEN_PILOT_GRADIENT_SCOPE "
                        + json.dumps(
                            {
                                "passed": distributed_gradient[
                                    "distributed_all_ranks_passed"
                                ],
                                "gradient_norm_by_group": distributed_gradient[
                                    "gradient_norm_by_group"
                                ],
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                if distributed_gradient["distributed_all_ranks_passed"] is not True:
                    raise RuntimeError("pilot distributed gradient-scope audit failed")
            frozen_with_grad = [
                name
                for name, parameter in model.named_parameters()
                if not parameter.requires_grad and parameter.grad is not None
            ]
            if not all_ranks_true(not frozen_with_grad, device):
                raise RuntimeError("a frozen decoder parameter received pilot gradients")
            grad_norm = torch.nn.utils.clip_grad_norm_(
                trainable_parameters, float(training["max_grad_norm"])
            )
            if not torch.isfinite(grad_norm):
                raise RuntimeError(f"non-finite pilot gradient norm at step {step}")
            optimizer.step()

            local_loss = torch.tensor(
                [
                    float(loss.detach().float().item())
                    * int(prepared["target_tokens"]),
                    int(prepared["target_tokens"]),
                ],
                dtype=torch.float64,
                device=device,
            )
            dist.all_reduce(local_loss, op=dist.ReduceOp.SUM)
            global_loss = float((local_loss[0] / local_loss[1]).item())
            if rank == 0:
                metric = {
                    "schema_version": 1,
                    "config_sha256": config_sha,
                    "step": step,
                    "epoch_index": epoch_index,
                    "position_in_epoch": local_position,
                    "world_size": world_size,
                    "micro_batch_size_per_rank": expected_micro_batch,
                    "gradient_accumulation_steps": 1,
                    "global_batch_size": global_batch_size,
                    "samples_seen_cumulative": step * global_batch_size,
                    "supervised_tokens": int(local_loss[1].item()),
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
                    f"QWEN_PILOT_STEP step={step}/{total_steps} "
                    f"epoch={epoch_index + 1}/{training['epochs']} "
                    f"position={local_position + 1}/{training['steps_per_epoch']} "
                    f"loss={global_loss:.8f} lr={learning_rate:.9f} "
                    f"grad_norm={metric['gradient_norm_before_clip']:.6f} "
                    f"elapsed_s={time.monotonic() - training_started:.1f}",
                    flush=True,
                )

            checkpoint_meta: dict[str, object] | None = None
            if step in checkpoint_steps:
                _checkpoint_freeze_audit(
                    model=model,
                    frozen_versions=frozen_versions,
                    frozen_sentinels=frozen_sentinels,
                    output_dir=output_dir,
                    step=step,
                    config_sha256=config_sha,
                    rank=rank,
                    world_size=world_size,
                )
                dist.barrier()
                saved: list[object | None] = [None]
                if rank == 0:
                    print(f"QWEN_PILOT_CHECKPOINT_BEGIN step={step}", flush=True)
                    directory, checkpoint_meta = save_checkpoint(
                        output_dir=output_dir,
                        step=step,
                        model=model,
                        optimizer=optimizer,
                        device=device,
                        config_sha256=config_sha,
                        manifest_sha256=str(data["manifest_sha256"]),
                        selection_sha256=train_bundle,
                        model_revision=str(inventory_model["revision"]),
                        world_size=world_size,
                        global_batch_size=global_batch_size,
                    )
                    saved[0] = {
                        "directory": str(directory),
                        "metadata": checkpoint_meta,
                    }
                dist.broadcast_object_list(saved, src=0)
                if not isinstance(saved[0], dict) or not isinstance(
                    saved[0].get("metadata"), dict
                ):
                    raise RuntimeError("pilot checkpoint broadcast is invalid")
                checkpoint_meta = saved[0]["metadata"]
                if rank == 0:
                    print(
                        f"QWEN_PILOT_CHECKPOINT_END step={step} "
                        f"visual_sha256={checkpoint_meta['visual_sha256']}",
                        flush=True,
                    )
                dist.barrier()
            if step in validation_steps:
                if checkpoint_meta is None:
                    raise RuntimeError("pilot validation step has no matching checkpoint")
                _validation_loss_entry(
                    model=model,
                    local_batches=local_validation_batches,
                    output_dir=output_dir,
                    step=step,
                    checkpoint_meta=checkpoint_meta,
                    config_sha256=config_sha,
                    validation_bundle_sha256=validation_bundle,
                    rank=rank,
                    device=device,
                )
            if args.stop_after_step == step:
                if rank == 0:
                    atomic_write_new_json(
                        output_dir / f"intentional_stop_step_{step:06d}.json",
                        {
                            "schema_version": 1,
                            "config_sha256": config_sha,
                            "intentional_stop_step": step,
                            "checkpoint": f"checkpoints/step_{step:06d}",
                        },
                    )
                    print(f"QWEN_PILOT_INTENTIONAL_STOP step={step}", flush=True)
                return 0

        training_elapsed = time.monotonic() - training_started
        validation_entries = broadcast_rank0_call(
            rank,
            lambda: [
                read_json(
                    output_dir / "validation_losses" / f"step_{step:06d}.json"
                )
                for step in validation_steps
            ],
        )
        if not isinstance(validation_entries, list) or not all(
            isinstance(entry, dict) for entry in validation_entries
        ):
            raise RuntimeError("pilot validation-loss series is invalid")
        selectable_steps = validation_steps[1:]
        selected = select_best_validation_checkpoint(
            validation_entries[1:], allowed_steps=selectable_steps
        )
        selected_step = int(selected["step"])
        selected_checkpoint = output_dir / "checkpoints" / f"step_{selected_step:06d}"
        selected_metadata = _load_selected_visual(
            checkpoint=selected_checkpoint,
            model=model,
            config_sha256=config_sha,
            manifest_sha256=str(data["manifest_sha256"]),
            train_bundle_sha256=train_bundle,
            model_revision=str(inventory_model["revision"]),
            world_size=world_size,
            global_batch_size=global_batch_size,
            selected_step=selected_step,
        )
        selected_sentinels_match = verify_model_sentinels(
            model, world_size=world_size, device=device
        )
        if not selected_sentinels_match:
            raise RuntimeError("selected pilot checkpoint differs across ranks")
        selection_payload = {
            "schema_version": 1,
            "config_sha256": config_sha,
            "selection_metric": "minimum_validation_teacher_forced_loss",
            "selection_tiebreaker": "earlier_step",
            "validation_bundle_sha256": validation_bundle,
            "validation_loss_entries": validation_entries,
            "selected_step": selected_step,
            "selected_teacher_forced_loss": selected["teacher_forced_loss"],
            "selected_checkpoint": f"checkpoints/step_{selected_step:06d}",
            "selected_checkpoint_metadata": selected_metadata,
            "test_bundle_sha256_recorded_but_not_evaluated": test_bundle,
            "test_split_used_for_selection": False,
        }
        if rank == 0:
            atomic_write_new_json(output_dir / "checkpoint_selection.json", selection_payload)
            print(
                f"QWEN_PILOT_CHECKPOINT_SELECTED step={selected_step} "
                f"validation_loss={float(selected['teacher_forced_loss']):.8f}",
                flush=True,
            )
        dist.barrier()

        model.gradient_checkpointing_disable()
        validation_metrics, validation_hashes = _evaluate_split(
            model=model,
            processor=processor,
            records=splits["validation"],
            local_groups=local_groups["validation"],
            output_dir=output_dir,
            split="validation",
            config_sha256=config_sha,
            selected_step=selected_step,
            rank=rank,
            world_size=world_size,
            device=device,
            max_new_tokens=int(generation["max_new_tokens"]),
            maximum_failures=int(gate["maximum_failure_examples"]),
        )
        test_metrics, test_hashes = _evaluate_split(
            model=model,
            processor=processor,
            records=splits["test"],
            local_groups=local_groups["test"],
            output_dir=output_dir,
            split="test",
            config_sha256=config_sha,
            selected_step=selected_step,
            rank=rank,
            world_size=world_size,
            device=device,
            max_new_tokens=int(generation["max_new_tokens"]),
            maximum_failures=int(gate["maximum_failure_examples"]),
        )

        final_local_audit = frozen_unchanged(
            model, frozen_versions, frozen_sentinels
        )
        final_frozen_audit = aggregate_audit(
            final_local_audit, rank=rank, world_size=world_size
        )
        if rank == 0:
            atomic_write_new_json(
                output_dir / "frozen_parameter_audit_final.json",
                {
                    "schema_version": 1,
                    "config_sha256": config_sha,
                    "selected_checkpoint_step": selected_step,
                    **final_frozen_audit,
                },
            )
        gradient_result = read_json(gradient_path)
        if (
            gradient_result.get("config_sha256") != config_sha
            or gradient_result.get("distributed_all_ranks_passed") is not True
        ):
            raise RuntimeError("pilot gradient audit evidence is invalid")
        checkpoint_audits = broadcast_rank0_call(
            rank,
            lambda: [
                read_json(
                    output_dir
                    / "frozen_parameter_audits"
                    / f"step_{step:06d}.json"
                )
                for step in checkpoint_steps
            ],
        )
        if not isinstance(checkpoint_audits, list):
            raise RuntimeError("pilot checkpoint freeze-audit series is invalid")
        all_checkpoint_audits = all(
            isinstance(audit, dict)
            and audit.get("config_sha256") == config_sha
            and audit.get("step") == step
            and audit.get("distributed_all_ranks_passed") is True
            for step, audit in zip(checkpoint_steps, checkpoint_audits, strict=True)
        )
        initial_loss = float(initial_validation["teacher_forced_loss"])
        best_loss = float(selected["teacher_forced_loss"])
        best_to_initial = best_loss / initial_loss
        validation_checks = metric_gate_checks(
            validation_metrics, gate, prefix="validation"
        )
        test_checks = metric_gate_checks(test_metrics, gate, prefix="test")
        step_evidence_complete = broadcast_rank0_call(
            rank,
            lambda: _validate_step_evidence(
                output_dir, config_sha256=config_sha, total_steps=total_steps
            ),
        )
        gate_checks: dict[str, bool] = {
            "validation_loss_decrease_passed": best_to_initial
            <= float(gate["maximum_best_to_initial_validation_loss_ratio"]),
            "gradient_scope_passed": True,
            "all_checkpoint_freeze_audits_passed": all_checkpoint_audits,
            "final_frozen_decoder_audit_passed": final_frozen_audit[
                "distributed_all_ranks_passed"
            ]
            is True,
            "distributed_model_equivalence_passed": initial_sentinels_match
            and selected_sentinels_match,
            "training_step_evidence_complete": step_evidence_complete is True,
            "validation_checkpoint_coverage_complete": [
                int(entry["step"]) for entry in validation_entries
            ]
            == validation_steps,
            "checkpoint_selection_used_validation_only": True,
            "test_evaluated_only_after_selection": True,
            **validation_checks,
            **test_checks,
        }
        automatic_pass = all(gate_checks.values())
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
                "scope": "synthetic_lowlight_supervised_vlm_pilot_8k_ddp8",
                "config_sha256": config_sha,
                "git_revision": git_revision(),
                "model_repository": inventory_model["repository"],
                "model_revision": inventory_model["revision"],
                "manifest_sha256": data["manifest_sha256"],
                "train_bundle_sha256": train_bundle,
                "validation_bundle_sha256": validation_bundle,
                "test_bundle_sha256": test_bundle,
                "parameter_groups": parameter_groups,
                "trainable_parameters": config["model"][
                    "expected_trainable_parameters"
                ],
                "frozen_parameters": config["model"]["expected_frozen_parameters"],
                "world_size": world_size,
                "micro_batch_size_per_rank": expected_micro_batch,
                "gradient_accumulation_steps": 1,
                "global_batch_size": global_batch_size,
                "optimizer_steps": total_steps,
                "sample_presentations": total_steps * global_batch_size,
                "initial_validation_teacher_forced_loss": initial_loss,
                "best_validation_teacher_forced_loss": best_loss,
                "best_to_initial_validation_loss_ratio": best_to_initial,
                "validation_loss_entries": validation_entries,
                "selected_checkpoint_step": selected_step,
                "selected_checkpoint": f"checkpoints/step_{selected_step:06d}",
                "selected_checkpoint_metadata": selected_metadata,
                "validation_metrics": validation_metrics,
                "test_metrics": test_metrics,
                "validation_artifact_sha256": validation_hashes,
                "test_artifact_sha256": test_hashes,
                "gradient_audit": gradient_result,
                "checkpoint_freeze_audits": checkpoint_audits,
                "final_frozen_parameter_audit": final_frozen_audit,
                "resumed_metadata": resumed_metadata,
                "training_elapsed_seconds_this_process": training_elapsed,
                "end_to_end_elapsed_seconds_this_process": overall_elapsed,
                "optimizer_steps_per_second_this_process": (
                    (total_steps - start_step) / training_elapsed
                    if total_steps > start_step
                    else 0.0
                ),
                "global_samples_per_second_this_process": (
                    (total_steps - start_step) * global_batch_size / training_elapsed
                    if total_steps > start_step
                    else 0.0
                ),
                "gate_checks": gate_checks,
                "automatic_gate_pass": automatic_pass,
                "environment_by_rank": gathered_runtime,
                "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
                "sdpa_backend": runtime["sdpa_backend"],
                "synthetic_internal_only": True,
                "scientific_pilot_result": True,
                "test_split_used_for_training": False,
                "test_split_used_for_checkpoint_selection": False,
                "test_split_evaluated_once_after_selection": True,
                "official_foundir_used": False,
            }
            atomic_write_new_json(output_dir / "training_summary.json", summary)
            marker = "QWEN_PILOT_8K_PASS" if automatic_pass else "QWEN_PILOT_8K_FAILED"
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
