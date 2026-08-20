#!/usr/bin/env python3
"""Continue Qwen visual-path training on real LOL-v2 data plus replay."""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import platform
import sys
import time
from collections import Counter
from datetime import timedelta
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.distributed as dist
from safetensors.torch import load_file
from torch.nn.parallel import DistributedDataParallel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lowlight_restoration.data import PairRecord  # noqa: E402
from lowlight_restoration.data import load_pair_manifest
from lowlight_restoration.qwen_distributed import (  # noqa: E402
    cosine_learning_rate,
    merge_indexed_outputs,
)
from lowlight_restoration.qwen_pilot import (  # noqa: E402
    augment_understanding_metrics,
    metric_gate_checks,
)
from lowlight_restoration.qwen_real_adaptation import (  # noqa: E402
    ALLOWED_WORLD_SIZES,
    global_training_groups,
    load_adaptation_config,
    partition_training_groups,
    records_by_real_split,
    select_best_adaptation_checkpoint,
    select_synthetic_replay,
    validate_adaptation_validation_entry,
)
from lowlight_restoration.qwen_real_validation import (  # noqa: E402
    augment_real_metrics,
    real_metric_gate_checks,
)
from lowlight_restoration.qwen_supervised import (  # noqa: E402
    evaluate_generated_outputs,
    open_degraded_images,
    prepare_supervised_batch,
    prompt_text,
    record_bundle_sha256,
)
from smoke.qwen_lowlight_real_adaptation_preflight import run_preflight  # noqa: E402
from smoke.qwen_lowlight_supervised_preflight import (  # noqa: E402
    load_inventory,
    sha256,
    workspace_path,
)
from smoke.train_qwen_lowlight_supervised import (  # noqa: E402
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
from smoke.train_qwen_lowlight_supervised_ddp import (  # noqa: E402
    aggregate_audit,
    all_ranks_true,
    broadcast_rank0_call,
    checkpoint_metadata,
    load_checkpoint,
    move_batch,
    save_checkpoint,
    verify_model_sentinels,
)


def distributed_context(
    config: dict[str, object],
) -> tuple[int, int, int, torch.device]:
    distributed = config["distributed"]
    runtime = config["runtime"]
    assert isinstance(distributed, dict) and isinstance(runtime, dict)
    required = ("RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE")
    missing = [name for name in required if name not in os.environ]
    if missing:
        raise SystemExit(f"TORCHRUN_ENVIRONMENT_MISSING_{'_'.join(missing)}")
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
    allowed = [int(value) for value in distributed["allowed_world_sizes"]]
    if world_size not in allowed or local_world_size != world_size:
        raise SystemExit(
            f"ADAPTATION_WORLD_SIZE_MUST_BE_2_OR_8_GOT_{world_size}_{local_world_size}"
        )
    if not 0 <= rank < world_size or not 0 <= local_rank < world_size:
        raise SystemExit("INVALID_ADAPTATION_DDP_RANK")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA_NOT_AVAILABLE")
    if torch.cuda.device_count() != world_size:
        raise SystemExit(
            f"VISIBLE_CUDA_COUNT_MUST_EQUAL_WORLD_SIZE_{world_size}_GOT_"
            f"{torch.cuda.device_count()}"
        )
    if world_size not in runtime["allowed_visible_device_counts"]:
        raise SystemExit("VISIBLE_CUDA_COUNT_OUTSIDE_CONFIG")
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


def verified_preflight(config_path: Path, output_dir: Path) -> dict[str, object]:
    full_path = output_dir / "preflight_full.json"
    if not full_path.is_file():
        raise RuntimeError(
            "full adaptation preflight is missing; use the checked launcher"
        )
    full = read_json(full_path)
    if full.get("large_model_hashes_checked") is not True:
        raise RuntimeError("full adaptation preflight omitted base-model hashes")
    current = run_preflight(
        config_path,
        skip_large_model_hashes=True,
        write_artifact=True,
        show_progress=True,
    )
    stable = (
        "config_sha256",
        "inventory_config_sha256",
        "model_revision",
        "model_files",
        "model_hash_contract_sha256",
        "parameter_groups",
        "tensor_count",
        "initialization",
        "preserved_failures",
        "real_manifest_sha256",
        "real_metadata_sha256",
        "real_split_evidence",
        "real_file_evidence",
        "synthetic_manifest_sha256",
        "synthetic_training_bundle_sha256",
        "synthetic_validation_bundle_sha256",
        "synthetic_file_evidence",
        "training_schedule",
        "visual_tokens",
        "assistant_target_tokens",
        "processor",
        "image_processor",
        "tokenizer",
        "transformers_modeling_source",
        "versions",
        "allowed_world_sizes",
        "real_test_content_verified_only",
        "real_test_used_for_training_or_selection",
        "previously_evaluated_real_records_used_for_training",
        "official_foundir_used",
    )
    for name in stable:
        if full.get(name) != current.get(name):
            raise RuntimeError(f"full adaptation preflight changed: {name}")
    return full


def _load_data(
    config: dict[str, object],
) -> tuple[
    dict[str, list[PairRecord]],
    list[PairRecord],
    list[PairRecord],
]:
    data = config["data"]
    assert isinstance(data, dict)
    real_config = data["real"]
    synthetic_config = data["synthetic_replay"]
    assert isinstance(real_config, dict) and isinstance(synthetic_config, dict)
    real_records = load_pair_manifest(workspace_path(real_config["manifest"]))
    real_splits = records_by_real_split(real_records, config)
    synthetic_records = load_pair_manifest(workspace_path(synthetic_config["manifest"]))
    synthetic_train = [
        record
        for record in synthetic_records
        if record.split == synthetic_config["training_split"]
    ]
    if (
        record_bundle_sha256(synthetic_train)
        != synthetic_config["training_bundle_sha256"]
    ):
        raise RuntimeError("synthetic adaptation training bundle changed")
    synthetic_validation = select_synthetic_replay(
        synthetic_records,
        source_count=int(synthetic_config["validation_source_count"]),
        seed=int(synthetic_config["validation_selection_seed"]),
    )
    if (
        record_bundle_sha256(synthetic_validation)
        != synthetic_config["validation_bundle_sha256"]
    ):
        raise RuntimeError("synthetic adaptation validation bundle changed")
    return real_splits, synthetic_train, synthetic_validation


def _selection_payload(
    real_splits: Mapping[str, Sequence[PairRecord]],
    synthetic_validation: Sequence[PairRecord],
    *,
    config_sha256: str,
    schedule_sha256: str,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "scope": "real_adaptation_source_disjoint_splits_and_replay",
        "config_sha256": config_sha256,
        "training_schedule_sha256": schedule_sha256,
        "real_splits": {
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
            for split, records in real_splits.items()
        },
        "synthetic_validation_replay": {
            "bundle_sha256": record_bundle_sha256(synthetic_validation),
            "sources": sorted({record.source_id for record in synthetic_validation}),
            "records": [record.pair_id for record in synthetic_validation],
        },
        "real_test_used_for_training_or_checkpoint_selection": False,
        "previously_evaluated_real_records_used": False,
        "official_foundir_used": False,
    }


def _prepare_batch(
    processor: Any,
    records: Sequence[PairRecord],
    *,
    domain: str,
    level: int | None = None,
) -> dict[str, object]:
    images = open_degraded_images(records)
    batch, target_counts = prepare_supervised_batch(processor, records, images)
    del images
    return {
        "domain": domain,
        "level": level,
        "records": list(records),
        "batch": batch,
        "target_tokens": sum(target_counts),
    }


def prepare_training_batches(
    processor: Any,
    real_train: Sequence[PairRecord],
    synthetic_train: Sequence[PairRecord],
    *,
    total_steps: int,
    seed: int,
    rank: int,
    world_size: int,
) -> dict[int, list[dict[str, object]]]:
    result: dict[int, list[dict[str, object]]] = {}
    groups_prepared = 0
    total_groups = total_steps * (8 // world_size)
    for step in range(1, total_steps + 1):
        global_groups = global_training_groups(
            real_train, synthetic_train, step=step, seed=seed
        )
        local = partition_training_groups(
            global_groups, world_size=world_size, rank=rank
        )
        prepared: list[dict[str, object]] = []
        for domain, records in local:
            if len(records) != 5 or {record.level_index for record in records} != set(
                range(1, 6)
            ):
                raise RuntimeError("adaptation training microbatch is not balanced")
            prepared.append(_prepare_batch(processor, records, domain=domain))
            groups_prepared += 1
        result[step] = prepared
        if step % 10 == 0 or step == total_steps:
            print(
                f"QWEN_REAL_ADAPTATION_PREPROCESS rank={rank} split=train "
                f"steps={step}/{total_steps} groups={groups_prepared}/{total_groups}",
                flush=True,
            )
    return result


def prepare_level_batches(
    processor: Any,
    records: Sequence[PairRecord],
    *,
    domain: str,
    rank: int,
    world_size: int,
    batch_size: int,
) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for level in range(1, 6):
        selected = sorted(
            (record for record in records if record.level_index == level),
            key=lambda record: record.pair_id,
        )[rank::world_size]
        for start in range(0, len(selected), batch_size):
            group = selected[start : start + batch_size]
            result.append(_prepare_batch(processor, group, domain=domain, level=level))
    print(
        f"QWEN_REAL_ADAPTATION_PREPROCESS rank={rank} split={domain} "
        f"batches={len(result)} records={sum(len(item['records']) for item in result)}",
        flush=True,
    )
    return result


def probe_loss_by_level(
    model: torch.nn.Module,
    batches: Sequence[dict[str, object]],
    *,
    device: torch.device,
) -> dict[str, object]:
    was_training = model.training
    model.eval()
    numerators = torch.zeros(5, dtype=torch.float64, device=device)
    tokens = torch.zeros(5, dtype=torch.float64, device=device)
    with torch.no_grad():
        for prepared in batches:
            level = prepared.get("level")
            if (
                isinstance(level, bool)
                or not isinstance(level, int)
                or level not in range(1, 6)
            ):
                raise RuntimeError("validation batch has no severity level")
            batch = prepared["batch"]
            assert isinstance(batch, dict)
            output = model(**move_batch(batch, device), use_cache=False)
            loss = float(output.loss.detach().float().item())
            target_tokens = int(prepared["target_tokens"])
            if not math.isfinite(loss) or target_tokens <= 0:
                raise RuntimeError("non-finite adaptation probe loss")
            numerators[level - 1] += loss * target_tokens
            tokens[level - 1] += target_tokens
    dist.all_reduce(numerators, op=dist.ReduceOp.SUM)
    dist.all_reduce(tokens, op=dist.ReduceOp.SUM)
    if torch.any(tokens <= 0):
        raise RuntimeError("adaptation validation probe omits a severity level")
    losses = numerators / tokens
    if was_training:
        model.train()
    return {
        "per_level_teacher_forced_loss": {
            str(level): float(losses[level - 1].item()) for level in range(1, 6)
        },
        "macro_level_teacher_forced_loss": float(losses.mean().item()),
        "overall_teacher_forced_loss": float((numerators.sum() / tokens.sum()).item()),
        "supervised_tokens_per_level": {
            str(level): int(tokens[level - 1].item()) for level in range(1, 6)
        },
    }


def validation_loss_entry(
    *,
    model: torch.nn.Module,
    real_batches: Sequence[dict[str, object]],
    synthetic_batches: Sequence[dict[str, object]],
    output_dir: Path,
    step: int,
    checkpoint_meta: Mapping[str, object] | None,
    config_sha256: str,
    real_bundle_sha256: str,
    synthetic_bundle_sha256: str,
    rank: int,
    device: torch.device,
) -> dict[str, object]:
    path = output_dir / "validation_losses" / f"step_{step:06d}.json"
    existing = broadcast_rank0_call(
        rank, lambda: read_json(path) if path.is_file() else None
    )
    expected_visual = (
        None if checkpoint_meta is None else checkpoint_meta.get("visual_sha256")
    )
    if existing is not None:
        if not isinstance(existing, dict):
            raise RuntimeError("existing adaptation validation is not an object")
        try:
            validate_adaptation_validation_entry(
                existing,
                config_sha256=config_sha256,
                step=step,
                checkpoint_visual_sha256=(
                    None if expected_visual is None else str(expected_visual)
                ),
                real_bundle_sha256=real_bundle_sha256,
                synthetic_bundle_sha256=synthetic_bundle_sha256,
                world_size=dist.get_world_size(),
            )
        except ValueError as exc:
            raise RuntimeError(
                f"existing adaptation validation differs at {step}: {exc}"
            ) from exc
        if rank == 0:
            print(
                f"QWEN_REAL_ADAPTATION_VALIDATION_REUSED step={step} "
                f"real_macro={float(existing['real_macro_level_teacher_forced_loss']):.8f} "
                "synthetic_macro="
                f"{float(existing['synthetic_macro_level_teacher_forced_loss']):.8f}",
                flush=True,
            )
        return existing
    if rank == 0:
        print(f"QWEN_REAL_ADAPTATION_VALIDATION_BEGIN step={step}", flush=True)
    started = time.monotonic()
    real = probe_loss_by_level(model, real_batches, device=device)
    synthetic = probe_loss_by_level(model, synthetic_batches, device=device)
    payload: dict[str, object] = {
        "schema_version": 1,
        "config_sha256": config_sha256,
        "step": step,
        "checkpoint_visual_sha256": expected_visual,
        "real_validation_bundle_sha256": real_bundle_sha256,
        "synthetic_validation_bundle_sha256": synthetic_bundle_sha256,
        "real_validation": real,
        "synthetic_validation": synthetic,
        "real_macro_level_teacher_forced_loss": real["macro_level_teacher_forced_loss"],
        "synthetic_macro_level_teacher_forced_loss": synthetic[
            "macro_level_teacher_forced_loss"
        ],
        "world_size": dist.get_world_size(),
    }
    validate_adaptation_validation_entry(
        payload,
        config_sha256=config_sha256,
        step=step,
        checkpoint_visual_sha256=(
            None if expected_visual is None else str(expected_visual)
        ),
        real_bundle_sha256=real_bundle_sha256,
        synthetic_bundle_sha256=synthetic_bundle_sha256,
        world_size=dist.get_world_size(),
    )
    if rank == 0:
        atomic_write_new_json(path, payload)
        print(
            f"QWEN_REAL_ADAPTATION_VALIDATION_END step={step} "
            f"real_macro={float(payload['real_macro_level_teacher_forced_loss']):.8f} "
            f"synthetic_macro={float(payload['synthetic_macro_level_teacher_forced_loss']):.8f} "
            f"elapsed_s={time.monotonic() - started:.1f}",
            flush=True,
        )
    dist.barrier()
    return payload


def checkpoint_freeze_audit(
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
            f"QWEN_REAL_ADAPTATION_FROZEN_AUDIT step={step} "
            f"passed={audit['distributed_all_ranks_passed']}",
            flush=True,
        )
    if audit["distributed_all_ranks_passed"] is not True:
        raise RuntimeError("frozen Qwen decoder changed during adaptation")
    return payload


def load_selected_visual(
    *,
    checkpoint: Path,
    model: torch.nn.Module,
    config_sha256: str,
    manifest_sha256: str,
    schedule_sha256: str,
    model_revision: str,
    world_size: int,
    global_batch_size: int,
    selected_step: int,
) -> dict[str, object]:
    metadata = checkpoint_metadata(
        checkpoint,
        config_sha256=config_sha256,
        manifest_sha256=manifest_sha256,
        selection_sha256=schedule_sha256,
        model_revision=model_revision,
        world_size=world_size,
        global_batch_size=global_batch_size,
        expected_step=selected_step,
    )
    state = load_file(checkpoint / "visual.safetensors", device="cpu")
    missing, unexpected = model.visual.load_state_dict(state, strict=True)
    if missing or unexpected:
        raise RuntimeError("selected adaptation visual checkpoint keys changed")
    return metadata


def generate_local_outputs_batched(
    model: torch.nn.Module,
    processor: Any,
    records: Sequence[PairRecord],
    selection_indices: Mapping[str, int],
    *,
    split: str,
    rank: int,
    world_size: int,
    batch_size: int,
    device: torch.device,
    max_new_tokens: int,
) -> list[tuple[int, str]]:
    model.eval()
    model.config.use_cache = True
    fixed_prompt = prompt_text(processor)
    original_padding = processor.tokenizer.padding_side
    processor.tokenizer.padding_side = "left"
    outputs: list[tuple[int, str]] = []
    local = list(records)[rank::world_size]
    try:
        for start in range(0, len(local), batch_size):
            group = local[start : start + batch_size]
            images = open_degraded_images(group)
            encoded = processor(
                text=[fixed_prompt] * len(group),
                images=images,
                padding=True,
                return_tensors="pt",
            )
            del images
            inputs = {
                name: value.to(device)
                for name, value in encoded.items()
                if isinstance(value, torch.Tensor)
            }
            if not torch.all(inputs["attention_mask"][:, -1] == 1):
                raise RuntimeError("left-padded generation ends in padding")
            prompt_width = int(inputs["input_ids"].shape[1])
            with torch.no_grad():
                generated = model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=max_new_tokens,
                    use_cache=True,
                )
            for row, record in enumerate(group):
                raw = processor.tokenizer.decode(
                    generated[row, prompt_width:],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                ).strip()
                outputs.append((selection_indices[record.pair_id], raw))
            print(
                f"QWEN_REAL_ADAPTATION_GENERATE rank={rank} split={split} "
                f"records={min(start + len(group), len(local))}/{len(local)}",
                flush=True,
            )
    finally:
        processor.tokenizer.padding_side = original_padding
        model.config.use_cache = False
    return outputs


def evaluate_split(
    *,
    model: torch.nn.Module,
    processor: Any,
    records: Sequence[PairRecord],
    output_dir: Path,
    split: str,
    metric_kind: str,
    config_sha256: str,
    selected_step: int,
    rank: int,
    world_size: int,
    device: torch.device,
    batch_size: int,
    max_new_tokens: int,
    maximum_failures: int,
) -> tuple[dict[str, object], dict[str, str]]:
    if rank == 0:
        print(
            f"QWEN_REAL_ADAPTATION_GENERATION_BEGIN split={split} "
            f"records={len(records)}",
            flush=True,
        )
    started = time.monotonic()
    indices = {record.pair_id: index for index, record in enumerate(records)}
    local_outputs = generate_local_outputs_batched(
        model,
        processor,
        records,
        indices,
        split=split,
        rank=rank,
        world_size=world_size,
        batch_size=batch_size,
        device=device,
        max_new_tokens=max_new_tokens,
    )
    gathered: list[object | None] = [None] * world_size
    dist.all_gather_object(gathered, local_outputs)
    raw_outputs = merge_indexed_outputs(
        [item for item in gathered if isinstance(item, list)],
        expected_count=len(records),
    )
    prediction_rows, base_metrics = evaluate_generated_outputs(records, raw_outputs)
    if metric_kind == "real":
        metrics = augment_real_metrics(prediction_rows, base_metrics)
    elif metric_kind == "synthetic":
        metrics = augment_understanding_metrics(prediction_rows, base_metrics)
    else:
        raise ValueError(f"unknown adaptation metric kind: {metric_kind}")
    metrics_payload = {
        "schema_version": 1,
        "config_sha256": config_sha256,
        "split": split,
        "metric_kind": metric_kind,
        "split_bundle_sha256": record_bundle_sha256(records),
        "selected_checkpoint_step": selected_step,
        "metrics": metrics,
    }
    predictions_payload = {
        "schema_version": 1,
        "config_sha256": config_sha256,
        "split": split,
        "metric_kind": metric_kind,
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
        atomic_write_new_json(artifacts["predictions"], predictions_payload)
        atomic_write_new_json(artifacts["failures"], failure_payload)
        atomic_write_new_json(artifacts["confusion"], confusion_payload)
        print(
            f"QWEN_REAL_ADAPTATION_GENERATION_END split={split} "
            f"severity_accuracy={float(metrics['severity_accuracy']):.6f} "
            f"valid_json={float(metrics['valid_structured_output_rate']):.6f} "
            f"elapsed_s={time.monotonic() - started:.1f}",
            flush=True,
        )
    dist.barrier()
    hashes = broadcast_rank0_call(
        rank, lambda: {name: sha256(path) for name, path in artifacts.items()}
    )
    if not isinstance(hashes, dict):
        raise RuntimeError("adaptation evaluation artifact hashes are invalid")
    return metrics, {str(name): str(value) for name, value in hashes.items()}


def _validate_step_evidence(
    output_dir: Path,
    *,
    config_sha256: str,
    training: Mapping[str, object],
    world_size: int,
) -> bool:
    total_steps = int(training["total_steps"])
    directory = output_dir / "step_metrics"
    expected = {f"step_{step:06d}.json" for step in range(1, total_steps + 1)}
    actual = {path.name for path in directory.glob("step_*.json")}
    if actual != expected:
        return False
    expected_fields = {
        "schema_version",
        "config_sha256",
        "step",
        "world_size",
        "micro_batch_size_per_rank",
        "gradient_accumulation_steps",
        "global_batch_size",
        "real_samples_per_step",
        "synthetic_samples_per_step",
        "samples_seen_cumulative",
        "real_supervised_tokens",
        "synthetic_replay_supervised_tokens",
        "supervised_tokens",
        "loss",
        "real_loss",
        "synthetic_replay_loss",
        "gradient_norm_before_clip",
        "learning_rate",
    }
    accumulation = int(training["global_batch_size"]) // (
        int(training["micro_batch_size_per_rank"]) * world_size
    )
    for step in range(1, total_steps + 1):
        try:
            payload = read_json(directory / f"step_{step:06d}.json")
        except (OSError, ValueError, TypeError):
            return False
        if (
            set(payload) != expected_fields
            or payload.get("schema_version") != 1
            or payload.get("config_sha256") != config_sha256
            or payload.get("step") != step
            or payload.get("world_size") != world_size
            or payload.get("micro_batch_size_per_rank")
            != training["micro_batch_size_per_rank"]
            or payload.get("gradient_accumulation_steps") != accumulation
            or payload.get("global_batch_size") != training["global_batch_size"]
            or payload.get("real_samples_per_step") != 20
            or payload.get("synthetic_samples_per_step") != 20
            or payload.get("samples_seen_cumulative")
            != step * int(training["global_batch_size"])
        ):
            return False
        token_names = (
            "real_supervised_tokens",
            "synthetic_replay_supervised_tokens",
            "supervised_tokens",
        )
        if any(
            isinstance(payload[name], bool)
            or not isinstance(payload[name], int)
            or int(payload[name]) <= 0
            for name in token_names
        ):
            return False
        real_tokens = int(payload["real_supervised_tokens"])
        synthetic_tokens = int(payload["synthetic_replay_supervised_tokens"])
        if int(payload["supervised_tokens"]) != real_tokens + synthetic_tokens:
            return False
        numeric_names = (
            "loss",
            "real_loss",
            "synthetic_replay_loss",
            "gradient_norm_before_clip",
            "learning_rate",
        )
        if any(
            isinstance(payload[name], bool)
            or not isinstance(payload[name], (int, float))
            or not math.isfinite(float(payload[name]))
            or float(payload[name]) < 0.0
            for name in numeric_names
        ):
            return False
        expected_loss = (
            float(payload["real_loss"]) * real_tokens
            + float(payload["synthetic_replay_loss"]) * synthetic_tokens
        ) / (real_tokens + synthetic_tokens)
        if not math.isclose(
            float(payload["loss"]), expected_loss, rel_tol=1e-12, abs_tol=1e-12
        ):
            return False
        expected_lr = cosine_learning_rate(
            step=step,
            total_steps=total_steps,
            warmup_steps=int(training["warmup_steps"]),
            base_learning_rate=float(training["learning_rate"]),
            minimum_learning_rate=float(training["minimum_learning_rate"]),
        )
        if not math.isclose(
            float(payload["learning_rate"]),
            expected_lr,
            rel_tol=1e-12,
            abs_tol=1e-15,
        ):
            return False
    return True


def _validation_checkpoint_visuals(
    output_dir: Path,
    *,
    checkpoint_steps: Sequence[int],
    config_sha256: str,
    real_manifest_sha256: str,
    schedule_sha256: str,
    model_revision: str,
    world_size: int,
    global_batch_size: int,
) -> dict[int, str | None]:
    result: dict[int, str | None] = {0: None}
    for step in checkpoint_steps:
        metadata = read_json(
            output_dir / "checkpoints" / f"step_{step:06d}" / "metadata.json"
        )
        expected = {
            "config_sha256": config_sha256,
            "manifest_sha256": real_manifest_sha256,
            "selection_bundle_sha256": schedule_sha256,
            "model_revision": model_revision,
            "world_size": world_size,
            "global_batch_size": global_batch_size,
            "step": step,
        }
        if any(metadata.get(name) != value for name, value in expected.items()):
            raise RuntimeError(
                f"adaptation validation checkpoint metadata changed at {step}"
            )
        visual_sha256 = metadata.get("visual_sha256")
        if not isinstance(visual_sha256, str) or len(visual_sha256) != 64:
            raise RuntimeError(
                f"adaptation validation checkpoint visual hash is invalid at {step}"
            )
        result[step] = visual_sha256
    return result


def completed_result(output_dir: Path, config_sha256: str) -> int | None:
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
        raise RuntimeError("existing adaptation summary is incomplete")
    checkpoint = (output_dir / checkpoint_value).resolve()
    try:
        checkpoint.relative_to(output_dir.resolve())
    except ValueError as exc:
        raise RuntimeError("selected adaptation checkpoint escapes output") from exc
    if read_json(checkpoint / "metadata.json") != metadata or sha256(
        checkpoint / "visual.safetensors"
    ) != metadata.get("visual_sha256"):
        raise RuntimeError("selected adaptation checkpoint evidence changed")
    if (
        summary.get("status") == "passed"
        and summary.get("automatic_gate_pass") is True
        and all(checks.values())
    ):
        print("QWEN_REAL_ADAPTATION_PASS existing_verified_result", flush=True)
        return 0
    if (
        summary.get("status") == "failed"
        and summary.get("automatic_gate_pass") is False
        and not all(checks.values())
    ):
        print("QWEN_REAL_ADAPTATION_FAILED existing_verified_result", flush=True)
        return 2
    raise RuntimeError("existing adaptation result has inconsistent status")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", choices=("auto", "never"), default="auto")
    parser.add_argument("--stop-after-step", type=int)
    args = parser.parse_args()

    config, config_sha, config_raw = load_adaptation_config(args.config)
    model_config = config["model"]
    data = config["data"]
    training = config["training"]
    distributed = config["distributed"]
    generation = config["generation"]
    gate = config["gate"]
    runtime = config["runtime"]
    assert all(
        isinstance(section, dict)
        for section in (
            model_config,
            data,
            training,
            distributed,
            generation,
            gate,
            runtime,
        )
    )
    real_config = data["real"]
    synthetic_config = data["synthetic_replay"]
    initialization = model_config["initialization"]
    assert all(
        isinstance(section, dict)
        for section in (real_config, synthetic_config, initialization)
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
            raise RuntimeError("adaptation preflight broadcast is invalid")
        if rank == 0:
            print(
                "QWEN_REAL_ADAPTATION_EMBEDDED_PREFLIGHT_PASS "
                + json.dumps(
                    {
                        "config_sha256": preflight["config_sha256"],
                        "initialization": preflight["initialization"],
                        "real_split_evidence": preflight["real_split_evidence"],
                        "training_schedule_sha256": preflight["training_schedule"][
                            "schedule_sha256"
                        ],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

        inventory, _ = load_inventory(config)
        inventory_model = inventory["model"]
        assert isinstance(inventory_model, dict)
        real_splits, synthetic_train, synthetic_validation = _load_data(config)
        schedule_sha = str(training["schedule_sha256"])
        if rank == 0:
            output_dir.mkdir(parents=True, exist_ok=True)
            atomic_write_new_bytes(output_dir / "config.json", config_raw)
            atomic_write_new_json(
                output_dir / "data_selection.json",
                _selection_payload(
                    real_splits,
                    synthetic_validation,
                    config_sha256=config_sha,
                    schedule_sha256=schedule_sha,
                ),
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
        initial_state = load_file(
            workspace_path(initialization["visual_checkpoint"]), device="cpu"
        )
        missing, unexpected = model.visual.load_state_dict(initial_state, strict=True)
        del initial_state
        if missing or unexpected:
            raise RuntimeError("step-600 initialization keys changed")
        bad_dropout = [
            name
            for name, module in model.named_modules()
            if isinstance(module, torch.nn.Dropout) and module.p != 0.0
        ]
        if bad_dropout:
            raise RuntimeError(
                f"nonzero dropout breaks adaptation resume: {bad_dropout}"
            )
        if training["gradient_checkpointing"] is True:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        model.train()
        initial_sentinels_match = verify_model_sentinels(
            model, world_size=world_size, device=device
        )
        if not initial_sentinels_match:
            raise RuntimeError("step-600 initialization differs across ranks")

        ddp_model = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=bool(distributed["broadcast_buffers"]),
            init_sync=bool(distributed["init_sync"]),
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
            raise RuntimeError("adaptation optimizer escaped full visual scope")

        total_steps = int(training["total_steps"])
        global_batch_size = int(training["global_batch_size"])
        checkpoint_steps = [int(value) for value in training["checkpoint_steps"]]
        validation_steps = [int(value) for value in training["validation_loss_steps"]]
        accumulation_steps = int(
            distributed["gradient_accumulation_by_world_size"][str(world_size)]
        )
        if (
            accumulation_steps * int(training["micro_batch_size_per_rank"]) * world_size
            != global_batch_size
        ):
            raise RuntimeError("adaptation effective batch changed at runtime")
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
                        manifest_sha256=str(real_config["manifest_sha256"]),
                        selection_sha256=schedule_sha,
                        model_revision=str(inventory_model["revision"]),
                        world_size=world_size,
                        global_batch_size=global_batch_size,
                    ),
                )
                if not isinstance(verified_checkpoint, dict):
                    raise RuntimeError("adaptation checkpoint verification failed")
                start_step, resumed_metadata = load_checkpoint(
                    directory=checkpoint,
                    model=model,
                    optimizer=optimizer,
                    device=device,
                    config_sha256=config_sha,
                    manifest_sha256=str(real_config["manifest_sha256"]),
                    selection_sha256=schedule_sha,
                    model_revision=str(inventory_model["revision"]),
                    world_size=world_size,
                    global_batch_size=global_batch_size,
                    verified_metadata=verified_checkpoint,
                )
                if start_step not in checkpoint_steps:
                    raise RuntimeError("adaptation resumed from unscheduled checkpoint")
                if not verify_model_sentinels(
                    model, world_size=world_size, device=device
                ):
                    raise RuntimeError("resumed adaptation differs across ranks")
                if rank == 0:
                    atomic_write_new_json(
                        output_dir / "resume_events" / f"step_{start_step:06d}.json",
                        {
                            "schema_version": 1,
                            "config_sha256": config_sha,
                            "resumed_from_step": start_step,
                            "world_size": world_size,
                            "checkpoint_metadata_sha256": sha256(
                                checkpoint / "metadata.json"
                            ),
                        },
                    )
                    print(
                        f"QWEN_REAL_ADAPTATION_RESUME_VERIFIED step={start_step}",
                        flush=True,
                    )
        elif latest_checkpoint(output_dir, total_steps) is not None:
            raise RuntimeError("adaptation checkpoint exists with --resume=never")
        dist.barrier()

        if args.stop_after_step is not None and (
            args.stop_after_step not in checkpoint_steps
            or args.stop_after_step <= start_step
        ):
            raise ValueError("--stop-after-step must be a future checkpoint")
        frozen_versions, frozen_sentinels = frozen_snapshot(model)
        prepared_training = prepare_training_batches(
            processor,
            real_splits["train"],
            synthetic_train,
            total_steps=total_steps,
            seed=int(training["seed"]),
            rank=rank,
            world_size=world_size,
        )
        if any(
            len(values) != accumulation_steps for values in prepared_training.values()
        ):
            raise RuntimeError("adaptation prepared accumulation count changed")
        real_validation_batches = prepare_level_batches(
            processor,
            real_splits["validation"],
            domain="real_validation",
            rank=rank,
            world_size=world_size,
            batch_size=int(generation["batch_size"]),
        )
        synthetic_validation_batches = prepare_level_batches(
            processor,
            synthetic_validation,
            domain="synthetic_validation",
            rank=rank,
            world_size=world_size,
            batch_size=int(generation["batch_size"]),
        )
        initial_path = output_dir / "validation_losses" / "step_000000.json"
        if start_step > 0 and not initial_path.is_file():
            raise RuntimeError("resumed adaptation lacks step-zero validation")
        initial_validation = validation_loss_entry(
            model=model,
            real_batches=real_validation_batches,
            synthetic_batches=synthetic_validation_batches,
            output_dir=output_dir,
            step=0,
            checkpoint_meta=None,
            config_sha256=config_sha,
            real_bundle_sha256=record_bundle_sha256(real_splits["validation"]),
            synthetic_bundle_sha256=record_bundle_sha256(synthetic_validation),
            rank=rank,
            device=device,
        )
        if start_step > 0:
            if resumed_metadata is None:
                raise RuntimeError("resumed adaptation has no metadata")
            validation_loss_entry(
                model=model,
                real_batches=real_validation_batches,
                synthetic_batches=synthetic_validation_batches,
                output_dir=output_dir,
                step=start_step,
                checkpoint_meta=resumed_metadata,
                config_sha256=config_sha,
                real_bundle_sha256=record_bundle_sha256(real_splits["validation"]),
                synthetic_bundle_sha256=record_bundle_sha256(synthetic_validation),
                rank=rank,
                device=device,
            )
        gradient_path = output_dir / "gradient_audit.json"
        if start_step > 0 and not gradient_path.is_file():
            raise RuntimeError("resumed adaptation lacks gradient audit")

        training_started = time.monotonic()
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
            domain_numerator = {
                "real": 0.0,
                "synthetic_replay": 0.0,
            }
            domain_tokens = {"real": 0, "synthetic_replay": 0}
            for micro_index, prepared in enumerate(prepared_training[step]):
                synchronize = micro_index == accumulation_steps - 1
                sync_context = (
                    contextlib.nullcontext() if synchronize else ddp_model.no_sync()
                )
                batch = prepared["batch"]
                assert isinstance(batch, dict)
                with sync_context:
                    output = ddp_model(**move_batch(batch, device), use_cache=False)
                    loss = output.loss
                    if loss is None or not torch.isfinite(loss):
                        raise RuntimeError(f"non-finite adaptation loss at step {step}")
                    (loss / accumulation_steps).backward()
                domain = str(prepared["domain"])
                target_tokens = int(prepared["target_tokens"])
                domain_numerator[domain] += (
                    float(loss.detach().float().item()) * target_tokens
                )
                domain_tokens[domain] += target_tokens

            if not gradient_path.exists():
                local_gradient = gradient_audit(model)
                distributed_gradient = aggregate_audit(
                    local_gradient, rank=rank, world_size=world_size
                )
                if rank == 0:
                    atomic_write_new_json(
                        gradient_path,
                        {
                            "schema_version": 1,
                            "config_sha256": config_sha,
                            "step": step,
                            **distributed_gradient,
                        },
                    )
                    print(
                        "QWEN_REAL_ADAPTATION_GRADIENT_SCOPE "
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
                    raise RuntimeError("adaptation gradient-scope audit failed")
            frozen_with_grad = [
                name
                for name, parameter in model.named_parameters()
                if not parameter.requires_grad and parameter.grad is not None
            ]
            if not all_ranks_true(not frozen_with_grad, device):
                raise RuntimeError("frozen decoder received adaptation gradients")
            grad_norm = torch.nn.utils.clip_grad_norm_(
                trainable_parameters, float(training["max_grad_norm"])
            )
            if not torch.isfinite(grad_norm):
                raise RuntimeError(f"non-finite adaptation grad norm at {step}")
            optimizer.step()

            totals = torch.tensor(
                [
                    domain_numerator["real"],
                    domain_tokens["real"],
                    domain_numerator["synthetic_replay"],
                    domain_tokens["synthetic_replay"],
                ],
                dtype=torch.float64,
                device=device,
            )
            dist.all_reduce(totals, op=dist.ReduceOp.SUM)
            real_loss = float((totals[0] / totals[1]).item())
            synthetic_loss = float((totals[2] / totals[3]).item())
            global_loss = float(
                ((totals[0] + totals[2]) / (totals[1] + totals[3])).item()
            )
            if rank == 0:
                metric = {
                    "schema_version": 1,
                    "config_sha256": config_sha,
                    "step": step,
                    "world_size": world_size,
                    "micro_batch_size_per_rank": training["micro_batch_size_per_rank"],
                    "gradient_accumulation_steps": accumulation_steps,
                    "global_batch_size": global_batch_size,
                    "real_samples_per_step": 20,
                    "synthetic_samples_per_step": 20,
                    "samples_seen_cumulative": step * global_batch_size,
                    "real_supervised_tokens": int(totals[1].item()),
                    "synthetic_replay_supervised_tokens": int(totals[3].item()),
                    "supervised_tokens": int(totals[1].item() + totals[3].item()),
                    "loss": global_loss,
                    "real_loss": real_loss,
                    "synthetic_replay_loss": synthetic_loss,
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
                    f"QWEN_REAL_ADAPTATION_STEP step={step}/{total_steps} "
                    f"loss={global_loss:.8f} real={real_loss:.8f} "
                    f"synthetic={synthetic_loss:.8f} lr={learning_rate:.9f} "
                    f"grad_norm={metric['gradient_norm_before_clip']:.6f} "
                    f"elapsed_s={time.monotonic() - training_started:.1f}",
                    flush=True,
                )

            checkpoint_meta: dict[str, object] | None = None
            if step in checkpoint_steps:
                checkpoint_freeze_audit(
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
                    print(
                        f"QWEN_REAL_ADAPTATION_CHECKPOINT_BEGIN step={step}",
                        flush=True,
                    )
                    directory, checkpoint_meta = save_checkpoint(
                        output_dir=output_dir,
                        step=step,
                        model=model,
                        optimizer=optimizer,
                        device=device,
                        config_sha256=config_sha,
                        manifest_sha256=str(real_config["manifest_sha256"]),
                        selection_sha256=schedule_sha,
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
                    raise RuntimeError("adaptation checkpoint broadcast failed")
                checkpoint_meta = saved[0]["metadata"]
                if rank == 0:
                    print(
                        f"QWEN_REAL_ADAPTATION_CHECKPOINT_END step={step} "
                        f"visual_sha256={checkpoint_meta['visual_sha256']}",
                        flush=True,
                    )
                dist.barrier()
            if step in validation_steps:
                if checkpoint_meta is None:
                    raise RuntimeError("adaptation validation lacks checkpoint")
                validation_loss_entry(
                    model=model,
                    real_batches=real_validation_batches,
                    synthetic_batches=synthetic_validation_batches,
                    output_dir=output_dir,
                    step=step,
                    checkpoint_meta=checkpoint_meta,
                    config_sha256=config_sha,
                    real_bundle_sha256=record_bundle_sha256(real_splits["validation"]),
                    synthetic_bundle_sha256=record_bundle_sha256(synthetic_validation),
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
                    print(
                        f"QWEN_REAL_ADAPTATION_INTENTIONAL_STOP step={step}",
                        flush=True,
                    )
                return 0

        training_elapsed = time.monotonic() - training_started
        validation_entries = broadcast_rank0_call(
            rank,
            lambda: [
                read_json(output_dir / "validation_losses" / f"step_{step:06d}.json")
                for step in validation_steps
            ],
        )
        if not isinstance(validation_entries, list) or not all(
            isinstance(entry, dict) for entry in validation_entries
        ):
            raise RuntimeError("adaptation validation-loss series is invalid")
        validation_visuals = broadcast_rank0_call(
            rank,
            lambda: _validation_checkpoint_visuals(
                output_dir,
                checkpoint_steps=checkpoint_steps,
                config_sha256=config_sha,
                real_manifest_sha256=str(real_config["manifest_sha256"]),
                schedule_sha256=schedule_sha,
                model_revision=str(inventory_model["revision"]),
                world_size=world_size,
                global_batch_size=global_batch_size,
            ),
        )
        if not isinstance(validation_visuals, dict):
            raise RuntimeError("adaptation validation checkpoint hashes are invalid")
        for expected_step, entry in zip(
            validation_steps, validation_entries, strict=True
        ):
            try:
                validate_adaptation_validation_entry(
                    entry,
                    config_sha256=config_sha,
                    step=expected_step,
                    checkpoint_visual_sha256=validation_visuals.get(expected_step),
                    real_bundle_sha256=record_bundle_sha256(real_splits["validation"]),
                    synthetic_bundle_sha256=record_bundle_sha256(synthetic_validation),
                    world_size=world_size,
                )
            except ValueError as exc:
                raise RuntimeError(
                    f"adaptation validation-loss series differs at "
                    f"{expected_step}: {exc}"
                ) from exc
        selected = select_best_adaptation_checkpoint(
            validation_entries[1:], allowed_steps=validation_steps[1:]
        )
        selected_step = int(selected["step"])
        selected_checkpoint = output_dir / "checkpoints" / f"step_{selected_step:06d}"
        selected_metadata = load_selected_visual(
            checkpoint=selected_checkpoint,
            model=model,
            config_sha256=config_sha,
            manifest_sha256=str(real_config["manifest_sha256"]),
            schedule_sha256=schedule_sha,
            model_revision=str(inventory_model["revision"]),
            world_size=world_size,
            global_batch_size=global_batch_size,
            selected_step=selected_step,
        )
        selected_sentinels_match = verify_model_sentinels(
            model, world_size=world_size, device=device
        )
        if not selected_sentinels_match:
            raise RuntimeError("selected adaptation checkpoint differs by rank")
        selection_payload = {
            "schema_version": 1,
            "config_sha256": config_sha,
            "selection_metric": training["checkpoint_selection_metric"],
            "selection_tiebreaker": training["checkpoint_selection_tiebreaker"],
            "real_validation_bundle_sha256": record_bundle_sha256(
                real_splits["validation"]
            ),
            "synthetic_validation_bundle_sha256": record_bundle_sha256(
                synthetic_validation
            ),
            "validation_loss_entries": validation_entries,
            "selected_step": selected_step,
            "selected_real_macro_level_teacher_forced_loss": selected[
                "real_macro_level_teacher_forced_loss"
            ],
            "selected_checkpoint": f"checkpoints/step_{selected_step:06d}",
            "selected_checkpoint_metadata": selected_metadata,
            "real_test_bundle_sha256_recorded_but_not_evaluated": record_bundle_sha256(
                real_splits["test"]
            ),
            "real_test_used_for_selection": False,
        }
        if rank == 0:
            atomic_write_new_json(
                output_dir / "checkpoint_selection.json", selection_payload
            )
            print(
                f"QWEN_REAL_ADAPTATION_CHECKPOINT_SELECTED step={selected_step} "
                f"real_macro_loss={float(selected['real_macro_level_teacher_forced_loss']):.8f}",
                flush=True,
            )
        dist.barrier()

        model.gradient_checkpointing_disable()
        real_validation_metrics, real_validation_hashes = evaluate_split(
            model=model,
            processor=processor,
            records=real_splits["validation"],
            output_dir=output_dir,
            split="real_validation",
            metric_kind="real",
            config_sha256=config_sha,
            selected_step=selected_step,
            rank=rank,
            world_size=world_size,
            device=device,
            batch_size=int(generation["batch_size"]),
            max_new_tokens=int(generation["max_new_tokens"]),
            maximum_failures=int(gate["maximum_failure_examples"]),
        )
        synthetic_metrics, synthetic_hashes = evaluate_split(
            model=model,
            processor=processor,
            records=synthetic_validation,
            output_dir=output_dir,
            split="synthetic_validation_replay",
            metric_kind="synthetic",
            config_sha256=config_sha,
            selected_step=selected_step,
            rank=rank,
            world_size=world_size,
            device=device,
            batch_size=int(generation["batch_size"]),
            max_new_tokens=int(generation["max_new_tokens"]),
            maximum_failures=int(gate["maximum_failure_examples"]),
        )
        # This is the first model access to the frozen real test records.
        real_test_metrics, real_test_hashes = evaluate_split(
            model=model,
            processor=processor,
            records=real_splits["test"],
            output_dir=output_dir,
            split="real_test",
            metric_kind="real",
            config_sha256=config_sha,
            selected_step=selected_step,
            rank=rank,
            world_size=world_size,
            device=device,
            batch_size=int(generation["batch_size"]),
            max_new_tokens=int(generation["max_new_tokens"]),
            maximum_failures=int(gate["maximum_failure_examples"]),
        )

        final_local_audit = frozen_unchanged(model, frozen_versions, frozen_sentinels)
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
            raise RuntimeError("adaptation gradient audit evidence is invalid")
        checkpoint_audits = broadcast_rank0_call(
            rank,
            lambda: [
                read_json(
                    output_dir / "frozen_parameter_audits" / f"step_{step:06d}.json"
                )
                for step in checkpoint_steps
            ],
        )
        if not isinstance(checkpoint_audits, list):
            raise RuntimeError("adaptation checkpoint audit series is invalid")
        all_checkpoint_audits = all(
            isinstance(audit, dict)
            and audit.get("config_sha256") == config_sha
            and audit.get("step") == step
            and audit.get("distributed_all_ranks_passed") is True
            for step, audit in zip(checkpoint_steps, checkpoint_audits, strict=True)
        )
        initial_loss = float(initial_validation["real_macro_level_teacher_forced_loss"])
        selected_loss = float(selected["real_macro_level_teacher_forced_loss"])
        selected_to_initial = selected_loss / initial_loss
        real_validation_checks = real_metric_gate_checks(
            real_validation_metrics,
            gate["real_validation"],
            prefix="real_validation",
        )
        real_test_checks = real_metric_gate_checks(
            real_test_metrics,
            gate["real_test"],
            prefix="real_test",
        )
        synthetic_checks = metric_gate_checks(
            synthetic_metrics,
            gate["synthetic_replay"],
            prefix="synthetic_replay",
        )
        step_evidence = broadcast_rank0_call(
            rank,
            lambda: _validate_step_evidence(
                output_dir,
                config_sha256=config_sha,
                training=training,
                world_size=world_size,
            ),
        )
        gate_checks: dict[str, bool] = {
            "real_validation_loss_decrease_passed": selected_to_initial
            <= float(gate["maximum_selected_to_initial_real_loss_ratio"]),
            "gradient_scope_passed": True,
            "all_checkpoint_freeze_audits_passed": all_checkpoint_audits,
            "final_frozen_decoder_audit_passed": final_frozen_audit[
                "distributed_all_ranks_passed"
            ]
            is True,
            "distributed_model_equivalence_passed": initial_sentinels_match
            and selected_sentinels_match,
            "training_step_evidence_complete": step_evidence is True,
            "validation_checkpoint_coverage_complete": [
                int(entry["step"]) for entry in validation_entries
            ]
            == validation_steps,
            "checkpoint_selection_used_validation_only": True,
            "real_test_evaluated_only_after_selection": True,
            "previously_evaluated_real_records_excluded": True,
            **real_validation_checks,
            **real_test_checks,
            **synthetic_checks,
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
                "scope": "source_disjoint_real_lowlight_Qwen_visual_adaptation",
                "config_sha256": config_sha,
                "git_revision": git_revision(),
                "model_repository": inventory_model["repository"],
                "model_revision": inventory_model["revision"],
                "initial_visual_checkpoint": initialization["visual_checkpoint"],
                "initial_visual_sha256": initialization["visual_sha256"],
                "real_manifest_sha256": real_config["manifest_sha256"],
                "real_train_bundle_sha256": record_bundle_sha256(real_splits["train"]),
                "real_validation_bundle_sha256": record_bundle_sha256(
                    real_splits["validation"]
                ),
                "real_test_bundle_sha256": record_bundle_sha256(real_splits["test"]),
                "synthetic_training_bundle_sha256": synthetic_config[
                    "training_bundle_sha256"
                ],
                "synthetic_validation_bundle_sha256": record_bundle_sha256(
                    synthetic_validation
                ),
                "training_schedule_sha256": schedule_sha,
                "parameter_groups": parameter_groups,
                "trainable_parameters": model_config["expected_trainable_parameters"],
                "frozen_parameters": model_config["expected_frozen_parameters"],
                "world_size": world_size,
                "micro_batch_size_per_rank": training["micro_batch_size_per_rank"],
                "gradient_accumulation_steps": accumulation_steps,
                "global_batch_size": global_batch_size,
                "optimizer_steps": total_steps,
                "sample_presentations": total_steps * global_batch_size,
                "real_sample_presentations": training["real_sample_presentations"],
                "synthetic_sample_presentations": training[
                    "synthetic_sample_presentations"
                ],
                "initial_real_macro_level_teacher_forced_loss": initial_loss,
                "selected_real_macro_level_teacher_forced_loss": selected_loss,
                "selected_to_initial_real_loss_ratio": selected_to_initial,
                "validation_loss_entries": validation_entries,
                "selected_checkpoint_step": selected_step,
                "selected_checkpoint": f"checkpoints/step_{selected_step:06d}",
                "selected_checkpoint_metadata": selected_metadata,
                "real_validation_metrics": real_validation_metrics,
                "real_test_metrics": real_test_metrics,
                "synthetic_validation_replay_metrics": synthetic_metrics,
                "real_validation_artifact_sha256": real_validation_hashes,
                "real_test_artifact_sha256": real_test_hashes,
                "synthetic_validation_artifact_sha256": synthetic_hashes,
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
                "annotation_kind": "reference_based_proxy_not_human_ground_truth",
                "real_holdout_is_source_disjoint": True,
                "real_holdout_used_for_training": False,
                "real_holdout_used_for_checkpoint_selection": False,
                "real_holdout_evaluated_once_after_selection": True,
                "previously_evaluated_real_records_used_for_training": False,
                "original_real_gate_remains_failed": True,
                "post_failure_constrained_diagnostic_remains_failed": True,
                "official_foundir_used": False,
                "diffusion_authorized_only_if_automatic_gate_pass": True,
            }
            atomic_write_new_json(output_dir / "training_summary.json", summary)
            marker = (
                "QWEN_REAL_ADAPTATION_PASS"
                if automatic_pass
                else "QWEN_REAL_ADAPTATION_FAILED"
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
