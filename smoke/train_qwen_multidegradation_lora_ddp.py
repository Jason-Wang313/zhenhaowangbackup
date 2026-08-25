#!/usr/bin/env python3
"""Train/evaluate the two/four-GPU Qwen multi-degradation LoRA ablation."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import random
import sys
import tempfile
import time
from datetime import timedelta
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image
from torch.nn.parallel import DistributedDataParallel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lowlight_restoration.multidegradation_data import (
    read_jsonl,
    sha256,
    stable_digest,
    write_new_or_verify,
)
from lowlight_restoration.qwen_lora import (
    configure_qwen_visual_lora,
    load_qwen_adapter,
    save_qwen_adapter,
)
from lowlight_restoration.qwen_multidegradation import (
    DegradationLabel,
    evaluate_outputs,
    make_label,
    prepare_supervised_batch,
    prompt_text,
)
from smoke.qwen_multidegradation_lora_preflight import (
    accumulation_steps_for_world,
    load_config,
    run_preflight,
    validate_contract,
    validate_software_runtime,
    workspace_path,
)


def atomic_json(path: Path, payload: Mapping[str, object]) -> str:
    return write_new_or_verify(
        path, (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    )


def setup_distributed(config: Mapping[str, object]) -> tuple[int, int, int, torch.device]:
    distributed = config["distributed"]
    assert isinstance(distributed, Mapping)
    required = ("RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE")
    if any(name not in os.environ for name in required):
        raise SystemExit("TORCHRUN_ENVIRONMENT_MISSING")
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    local_world = int(os.environ["LOCAL_WORLD_SIZE"])
    supported = [int(value) for value in distributed["supported_world_sizes"]]
    if world not in supported or local_world != world:
        raise SystemExit(f"QWEN_MULTIDEGRADATION_WORLD_SIZE_MUST_BE_ONE_OF_{supported}")
    if not torch.cuda.is_available() or torch.cuda.device_count() != world:
        raise SystemExit(f"QWEN_MULTIDEGRADATION_REQUIRES_{world}_VISIBLE_GPUS")
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dist.init_process_group(
        backend=str(distributed["backend"]),
        timeout=timedelta(minutes=int(distributed["timeout_minutes"])),
    )
    return rank, world, local_rank, device


def broadcast_rank0(rank: int, function: Any) -> object:
    envelope: list[object | None] = [None]
    if rank == 0:
        try:
            envelope[0] = {"ok": True, "value": function()}
        except Exception as exc:
            envelope[0] = {"ok": False, "type": type(exc).__name__, "error": str(exc)}
    dist.broadcast_object_list(envelope, src=0)
    payload = envelope[0]
    if not isinstance(payload, dict) or payload.get("ok") is not True:
        raise RuntimeError(f"rank-zero operation failed: {payload}")
    return payload["value"]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def label_for(record: Mapping[str, object]) -> DegradationLabel:
    levels = record["levels"]
    if not isinstance(levels, Mapping):
        raise ValueError(f"record levels are invalid: {record['record_id']}")
    return make_label(record["degradations"], levels)


def open_images(records: Sequence[Mapping[str, object]]) -> list[Image.Image]:
    images: list[Image.Image] = []
    for record in records:
        path = workspace_path(record["output_path"])
        with Image.open(path) as handle:
            images.append(handle.convert("RGB"))
    return images


def move_batch(batch: Mapping[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def load_model(
    config: Mapping[str, object],
    *,
    variant: str,
    device: torch.device,
) -> tuple[torch.nn.Module, Any, dict[str, object]]:
    model_config = config["model"]
    processor_config = config["processor"]
    assert isinstance(model_config, Mapping) and isinstance(processor_config, Mapping)
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    model_path = workspace_path(model_config["local_path"])
    processor = AutoProcessor.from_pretrained(
        model_path,
        local_files_only=True,
        use_fast=False,
        min_pixels=int(processor_config["min_pixels"]),
        max_pixels=int(processor_config["max_pixels"]),
    )
    processor.tokenizer.padding_side = str(processor_config["padding_side"])
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation=str(model_config["attention_implementation"]),
        low_cpu_mem_usage=True,
    ).to(device)
    scope = configure_qwen_visual_lora(
        model,
        rank=int(model_config["lora_rank"]),
        alpha=float(model_config["lora_alpha"]),
        dropout=float(model_config["lora_dropout"]),
        merger_trainable=variant == "merger_trainable",
    )
    if scope["matched_lora_module_count"] != model_config["expected_lora_modules"]:
        raise RuntimeError(f"loaded Qwen LoRA module count changed: {scope}")
    model.config.use_cache = False
    return model, processor, scope


def validate_runtime(config: Mapping[str, object], local_rank: int) -> dict[str, object]:
    runtime = config["runtime"]
    assert isinstance(runtime, Mapping)
    properties = torch.cuda.get_device_properties(local_rank)
    memory = properties.total_memory / 1024**3
    if memory < float(runtime["minimum_gpu_memory_gib"]):
        raise SystemExit(f"GPU_{local_rank}_MEMORY_TOO_SMALL_{memory:.2f}_GIB")
    actual = validate_software_runtime(dict(config))
    world = dist.get_world_size()
    accumulation = accumulation_steps_for_world(dict(config), world)
    training = config["training"]
    assert isinstance(training, Mapping)
    return {
        "rank": dist.get_rank(),
        "world_size": world,
        "gradient_accumulation_steps": accumulation,
        "effective_global_batch_size": (
            world * int(training["micro_batch_size_per_rank"]) * accumulation
        ),
        "gpu": torch.cuda.get_device_name(local_rank),
        "gpu_memory_gib": memory,
        "versions": actual,
    }


def learning_rate_scale(step: int, total: int, warmup: int, minimum: float) -> float:
    if step <= warmup:
        return step / warmup if warmup else 1.0
    progress = (step - warmup) / (total - warmup)
    return minimum + (1 - minimum) * 0.5 * (1 + math.cos(math.pi * progress))


def semantic_assistant_token_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    semantic_token_ids: Sequence[int],
    semantic_weight: float,
) -> torch.Tensor:
    """Causal assistant-token CE with extra weight on label decisions.

    Prompt and padding positions remain excluded through the existing ``-100``
    label mask.  The correction only changes the relative weight of frozen,
    explicitly listed target-token IDs; JSON syntax remains supervised.
    """

    if logits.ndim != 3 or labels.ndim != 2 or logits.shape[:2] != labels.shape:
        raise ValueError("semantic-loss logits and labels have incompatible shapes")
    if logits.shape[1] < 2 or logits.shape[2] <= 1:
        raise ValueError("semantic-loss logits have invalid dimensions")
    if not semantic_token_ids or len(set(semantic_token_ids)) != len(semantic_token_ids):
        raise ValueError("semantic token IDs must be nonempty and unique")
    if any(isinstance(token, bool) or not isinstance(token, int) or token < 0 for token in semantic_token_ids):
        raise ValueError("semantic token IDs must be nonnegative integers")
    if not math.isfinite(semantic_weight) or semantic_weight <= 1.0:
        raise ValueError("semantic token weight must be finite and greater than one")

    shifted_logits = logits[:, :-1, :]
    shifted_labels = labels[:, 1:].contiguous()
    valid = shifted_labels.ne(-100)
    if not torch.any(valid):
        raise ValueError("semantic loss has no supervised assistant tokens")
    active_labels = shifted_labels[valid]
    token_loss = F.cross_entropy(
        shifted_logits[valid].float().contiguous(),
        active_labels,
        reduction="none",
    )
    weights = torch.ones_like(token_loss)
    semantic = torch.zeros_like(active_labels, dtype=torch.bool)
    for token_id in semantic_token_ids:
        semantic.logical_or_(active_labels.eq(token_id))
    weights = torch.where(
        semantic,
        torch.full_like(weights, float(semantic_weight)),
        weights,
    )
    loss = (token_loss * weights).sum() / weights.sum()
    if not torch.isfinite(loss):
        raise RuntimeError("semantic assistant-token loss is non-finite")
    return loss


def training_loss(
    output_value: object,
    labels: torch.Tensor,
    training: Mapping[str, object],
) -> torch.Tensor:
    """Select the frozen baseline CE or the bounded semantic correction CE."""

    semantic = training.get("semantic_token_loss")
    if semantic is None:
        loss = getattr(output_value, "loss", None)
        if not isinstance(loss, torch.Tensor):
            raise RuntimeError("Qwen output omitted its language-model loss")
        return loss
    if not isinstance(semantic, Mapping) or semantic.get("enabled") is not True:
        raise ValueError("semantic-token loss contract is malformed")
    logits = getattr(output_value, "logits", None)
    if not isinstance(logits, torch.Tensor):
        raise RuntimeError("Qwen output omitted logits for semantic-token loss")
    token_ids = semantic.get("semantic_token_ids")
    if not isinstance(token_ids, list):
        raise ValueError("semantic-token loss IDs must be a list")
    return semantic_assistant_token_loss(
        logits,
        labels,
        semantic_token_ids=token_ids,
        semantic_weight=float(semantic["semantic_weight"]),
    )


def create_optimizer(
    model: torch.nn.Module, config: Mapping[str, object]
) -> torch.optim.Optimizer:
    training = config["training"]
    assert isinstance(training, Mapping)
    lora = []
    merger = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("model.visual.merger."):
            merger.append(parameter)
        else:
            lora.append(parameter)
    groups = [{"params": lora, "lr": float(training["lora_learning_rate"]), "kind": "lora"}]
    if merger:
        groups.append({"params": merger, "lr": float(training["merger_learning_rate"]), "kind": "merger"})
    optimizer = torch.optim.AdamW(
        groups,
        betas=(float(training["adam_beta1"]), float(training["adam_beta2"])),
        eps=float(training["adam_epsilon"]),
        weight_decay=float(training["weight_decay"]),
        fused=bool(training["fused_adamw"]),
    )
    expected = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    actual = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    if actual != expected:
        raise RuntimeError("optimizer scope differs from Qwen adapter scope")
    return optimizer


def optimizer_to_device(optimizer: torch.optim.Optimizer) -> None:
    for parameter, state in optimizer.state.items():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(parameter.device)


def checkpoint_directory(output: Path, stage_name: str) -> Path:
    return output / "checkpoints" / stage_name


def publish_stage_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    output: Path,
    stage_name: str,
    config_sha: str,
    variant: str,
    global_step: int,
    world_size: int,
    accumulation_steps: int,
) -> dict[str, object]:
    final = checkpoint_directory(output, stage_name)
    if final.is_dir():
        metadata = json.loads((final / "metadata.json").read_text())
        if (
            metadata.get("config_sha256") != config_sha
            or metadata.get("variant") != variant
            or metadata.get("world_size") != world_size
            or metadata.get("gradient_accumulation_steps") != accumulation_steps
        ):
            raise RuntimeError(f"checkpoint contract mismatch: {final}")
        return metadata
    final.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{stage_name}.", dir=final.parent))
    try:
        adapter_meta = save_qwen_adapter(
            model,
            temporary / "adapter.safetensors",
            {
                "config_sha256": config_sha,
                "variant": variant,
                "stage": stage_name,
                "global_step": global_step,
                "world_size": world_size,
                "gradient_accumulation_steps": accumulation_steps,
            },
        )
        torch.save(optimizer.state_dict(), temporary / "optimizer.pt")
        metadata = {
            "schema_version": 1,
            "config_sha256": config_sha,
            "variant": variant,
            "stage": stage_name,
            "global_step": global_step,
            "world_size": world_size,
            "gradient_accumulation_steps": accumulation_steps,
            "adapter_sha256": adapter_meta["adapter_sha256"],
            "optimizer_sha256": sha256(temporary / "optimizer.pt"),
        }
        (temporary / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
        os.rename(temporary, final)
        return metadata
    finally:
        if temporary.exists():
            for child in temporary.iterdir():
                child.unlink(missing_ok=True)
            temporary.rmdir()


def resume_stage(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    output: Path,
    stages: Sequence[Mapping[str, object]],
    config_sha: str,
    variant: str,
    world_size: int,
    accumulation_steps: int,
) -> tuple[int, int]:
    completed = 0
    global_step = 0
    for index, stage in enumerate(stages):
        directory = checkpoint_directory(output, str(stage["name"]))
        if not directory.is_dir():
            break
        metadata = json.loads((directory / "metadata.json").read_text())
        if (
            metadata.get("config_sha256") != config_sha
            or metadata.get("variant") != variant
            or metadata.get("world_size") != world_size
            or metadata.get("gradient_accumulation_steps") != accumulation_steps
        ):
            raise RuntimeError(f"resume checkpoint differs: {directory}")
        if sha256(directory / "adapter.safetensors") != metadata["adapter_sha256"]:
            raise RuntimeError(f"resume adapter hash failed: {directory}")
        if sha256(directory / "optimizer.pt") != metadata["optimizer_sha256"]:
            raise RuntimeError(f"resume optimizer hash failed: {directory}")
        completed = index + 1
        global_step = int(metadata["global_step"])
    for stage in stages[completed:]:
        if checkpoint_directory(output, str(stage["name"])).exists():
            raise RuntimeError("curriculum checkpoints are not consecutive")
    if completed:
        directory = checkpoint_directory(output, str(stages[completed - 1]["name"]))
        load_qwen_adapter(model, directory / "adapter.safetensors")
        optimizer.load_state_dict(torch.load(directory / "optimizer.pt", map_location="cpu", weights_only=False))
        optimizer_to_device(optimizer)
    return completed, global_step


def stage_records(
    records: Sequence[Mapping[str, object]],
    stage: Mapping[str, object],
    seed: int,
    *,
    epoch: int = 0,
) -> list[Mapping[str, object]]:
    selected = [
        record
        for record in records
        if record["split"] == "train"
        and int(record["composition_size"]) <= int(stage["maximum_composition_size"])
    ]
    selected.sort(
        key=lambda record: (
            stable_digest(seed, stage["name"], record["record_id"])
            if epoch == 0
            else stable_digest(seed, stage["name"], epoch, record["record_id"])
        )
    )
    if len(selected) != int(stage["records"]):
        raise RuntimeError(f"stage record count changed: {stage['name']}")
    return selected


def rank_microbatch(
    global_records: Sequence[Mapping[str, object]],
    *,
    rank: int,
    world: int,
    micro: int,
    accumulation_index: int,
) -> list[Mapping[str, object]]:
    start = (accumulation_index * world + rank) * micro
    batch = list(global_records[start : start + micro])
    if len(batch) != micro:
        raise RuntimeError("rank-local accumulated microbatch has wrong cardinality")
    return batch


def train_variant(
    model: torch.nn.Module,
    processor: Any,
    scope: Mapping[str, object],
    records: Sequence[Mapping[str, object]],
    config: Mapping[str, object],
    *,
    config_sha: str,
    variant: str,
    output: Path,
    rank: int,
    world: int,
    local_rank: int,
    device: torch.device,
) -> dict[str, object]:
    training = config["training"]
    data = config["data"]
    distributed = config["distributed"]
    assert isinstance(training, Mapping) and isinstance(data, Mapping) and isinstance(distributed, Mapping)
    stages = data["curriculum"]
    assert isinstance(stages, list)
    if training["gradient_checkpointing"]:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    ddp = DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        broadcast_buffers=False,
        find_unused_parameters=bool(distributed["find_unused_parameters"]),
    )
    optimizer = create_optimizer(model, config)
    micro = int(training["micro_batch_size_per_rank"])
    global_batch = int(training["global_batch_size"])
    accumulation = accumulation_steps_for_world(dict(config), world)
    completed_stages, global_step = resume_stage(
        model,
        optimizer,
        output=output,
        stages=stages,
        config_sha=config_sha,
        variant=variant,
        world_size=world,
        accumulation_steps=accumulation,
    )
    total_steps = int(training["total_optimizer_steps"])
    trainable_names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    gradient_seen: set[str] = set()
    frozen_gradient_names: set[str] = set()
    metrics: list[dict[str, object]] = []
    started = time.monotonic()

    for stage_index, stage in enumerate(stages):
        if stage_index < completed_stages:
            continue
        epochs = int(
            stage.get(
                "epochs", int(training["epochs_per_curriculum_stage"])
            )
        )
        if epochs <= 0:
            raise RuntimeError(f"stage epochs must be positive: {stage['name']}")
        records_per_epoch = int(stage["records"])
        steps_per_epoch = records_per_epoch // global_batch
        stage_steps = steps_per_epoch * epochs
        stage_started = time.monotonic()
        for epoch in range(epochs):
            selected = stage_records(
                records, stage, int(training["seed"]), epoch=epoch
            )
            for epoch_step in range(steps_per_epoch):
                stage_step = epoch * steps_per_epoch + epoch_step
                global_records = selected[
                    epoch_step * global_batch : (epoch_step + 1) * global_batch
                ]
                optimizer.zero_grad(set_to_none=True)
                loss_value = torch.zeros((), dtype=torch.float32, device=device)
                for accumulation_index in range(accumulation):
                    local_records = rank_microbatch(
                        global_records,
                        rank=rank,
                        world=world,
                        micro=micro,
                        accumulation_index=accumulation_index,
                    )
                    images = open_images(local_records)
                    labels = [label_for(record) for record in local_records]
                    batch, _ = prepare_supervised_batch(processor, labels, images)
                    synchronization = (
                        ddp.no_sync()
                        if accumulation_index + 1 < accumulation
                        else contextlib.nullcontext()
                    )
                    with synchronization:
                        device_batch = move_batch(batch, device)
                        forward_batch = (
                            {key: value for key, value in device_batch.items() if key != "labels"}
                            if training.get("semantic_token_loss") is not None
                            else device_batch
                        )
                        output_value = ddp(**forward_batch, use_cache=False)
                        loss = training_loss(output_value, device_batch["labels"], training)
                        if not torch.isfinite(loss):
                            raise RuntimeError(
                                f"non-finite Qwen loss at step {global_step + 1}"
                            )
                        (loss / accumulation).backward()
                    loss_value += loss.detach().float() / accumulation
                for name, parameter in model.named_parameters():
                    if parameter.requires_grad and parameter.grad is not None:
                        gradient_seen.add(name)
                    elif not parameter.requires_grad and parameter.grad is not None:
                        frozen_gradient_names.add(name)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    [
                        parameter
                        for parameter in model.parameters()
                        if parameter.requires_grad
                    ],
                    float(training["max_grad_norm"]),
                )
                global_step += 1
                scale = learning_rate_scale(
                    global_step,
                    total_steps,
                    int(training["warmup_steps"]),
                    float(training["minimum_learning_rate_ratio"]),
                )
                for group in optimizer.param_groups:
                    base = float(
                        training["merger_learning_rate"]
                        if group["kind"] == "merger"
                        else training["lora_learning_rate"]
                    )
                    group["lr"] = base * scale
                optimizer.step()
                dist.all_reduce(loss_value, op=dist.ReduceOp.SUM)
                loss_value /= world
                if rank == 0 and (
                    global_step <= 5
                    or global_step % 10 == 0
                    or stage_step + 1 == stage_steps
                ):
                    row = {
                        "global_step": global_step,
                        "stage": stage["name"],
                        "stage_epoch": epoch + 1,
                        "stage_step": stage_step + 1,
                        "loss": float(loss_value.item()),
                        "grad_norm": float(grad_norm.detach().float().item()),
                        "lora_learning_rate": float(optimizer.param_groups[0]["lr"]),
                    }
                    metrics.append(row)
                    print(
                        "QWEN_MULTIDEGRADATION_TRAIN "
                        + json.dumps(row, sort_keys=True),
                        flush=True,
                    )
        if rank == 0:
            publish_stage_checkpoint(
                model,
                optimizer,
                output=output,
                stage_name=str(stage["name"]),
                config_sha=config_sha,
                variant=variant,
                global_step=global_step,
                world_size=world,
                accumulation_steps=accumulation,
            )
            print(
                f"QWEN_MULTIDEGRADATION_STAGE_PASS variant={variant} stage={stage['name']} "
                f"steps={stage_steps} elapsed={time.monotonic()-stage_started:.2f}",
                flush=True,
            )
        dist.barrier()
    if global_step != total_steps:
        raise RuntimeError(f"optimizer step contract changed: {global_step} != {total_steps}")
    local_audit = {
        "all_trainable_received_gradient": gradient_seen == trainable_names or completed_stages == len(stages),
        "missing_trainable_gradients": sorted(trainable_names - gradient_seen),
        "frozen_gradient_names": sorted(frozen_gradient_names),
    }
    audits: list[object | None] = [None] * world
    dist.all_gather_object(audits, local_audit)
    audit_pass = all(
        isinstance(item, dict)
        and item["all_trainable_received_gradient"] is True
        and not item["frozen_gradient_names"]
        for item in audits
    )
    return {
        "training_elapsed_seconds": time.monotonic() - started,
        "global_step": global_step,
        "scope": dict(scope),
        "gradient_audit_pass": audit_pass,
        "rank_gradient_audits": audits,
        "training_metrics": metrics,
    }


def generate_outputs(
    model: torch.nn.Module,
    processor: Any,
    records: Sequence[Mapping[str, object]],
    config: Mapping[str, object],
    *,
    rank: int,
    world: int,
    device: torch.device,
) -> list[str]:
    generation = config["generation"]
    assert isinstance(generation, Mapping)
    model.eval()
    model.gradient_checkpointing_disable()
    model.config.use_cache = True
    prompt = prompt_text(processor)
    indexed = [(index, record) for index, record in enumerate(records) if index % world == rank]
    local: list[tuple[int, str]] = []
    batch_size = int(generation["batch_size_per_rank"])
    for offset in range(0, len(indexed), batch_size):
        chunk = indexed[offset : offset + batch_size]
        images = open_images([record for _, record in chunk])
        encoded = processor(
            text=[prompt] * len(chunk), images=images, padding=True, return_tensors="pt"
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
                max_new_tokens=int(generation["max_new_tokens"]),
                use_cache=True,
            )
        for row, (index, _) in enumerate(chunk):
            raw = processor.tokenizer.decode(
                generated[row, input_length:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            ).strip()
            local.append((index, raw))
        if (offset // batch_size + 1) % 25 == 0 or offset + batch_size >= len(indexed):
            print(
                f"QWEN_MULTIDEGRADATION_GENERATE rank={rank} "
                f"completed={min(offset+batch_size,len(indexed))}/{len(indexed)}",
                flush=True,
            )
    gathered: list[object | None] = [None] * world
    dist.all_gather_object(gathered, local)
    flattened = [item for rank_items in gathered if isinstance(rank_items, list) for item in rank_items]
    if len(flattened) != len(records) or sorted(index for index, _ in flattened) != list(range(len(records))):
        raise RuntimeError("distributed Qwen generation is incomplete")
    return [raw for _, raw in sorted(flattened)]


def gate_checks(metrics: Mapping[str, object], config: Mapping[str, object], gradient_pass: bool) -> dict[str, bool]:
    gate = config["gate"]
    assert isinstance(gate, Mapping)
    triple = metrics["by_composition_size"]["3"]
    return {
        "valid_structure": float(metrics["valid_structured_output_rate"]) >= float(gate["minimum_valid_structured_output_rate"]),
        "type_macro_f1": float(metrics["type_macro_f1"]) >= float(gate["minimum_type_macro_f1"]),
        "type_micro_f1": float(metrics["type_micro_f1"]) >= float(gate["minimum_type_micro_f1"]),
        "degradation_set_exact": float(metrics["degradation_set_exact_match_rate"]) >= float(gate["minimum_degradation_set_exact_match_rate"]),
        "combined_label_exact": float(metrics["combined_label_exact_match_rate"]) >= float(gate["minimum_combined_label_exact_match_rate"]),
        "severity_accuracy": float(metrics["severity_typed_accuracy"]) >= float(gate["minimum_severity_typed_accuracy"]),
        "severity_mae": float(metrics["severity_typed_mae_with_missing_penalty"]) <= float(gate["maximum_severity_typed_mae_with_missing_penalty"]),
        "triple_set_exact": float(triple["degradation_set_exact_match_rate"]) >= float(gate["minimum_triple_degradation_set_exact_match_rate"]),
        "gradient_scope": gradient_pass,
    }


def evaluate_split(
    model: torch.nn.Module,
    processor: Any,
    all_records: Sequence[Mapping[str, object]],
    split: str,
    config: Mapping[str, object],
    *,
    rank: int,
    world: int,
    device: torch.device,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    records = [record for record in all_records if record["split"] == split]
    raw = generate_outputs(model, processor, records, config, rank=rank, world=world, device=device)
    rows, metrics = evaluate_outputs([label_for(record) for record in records], raw)
    for row, record in zip(rows, records, strict=True):
        row["record_id"] = record["record_id"]
        row["source_id"] = record["source_id"]
    return rows, metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--variant", choices=("merger_frozen", "merger_trainable"), required=True)
    parser.add_argument("--test-only", action="store_true")
    args = parser.parse_args()
    config, config_sha = load_config(args.config)
    validate_contract(config)
    runtime = config["runtime"]
    data = config["data"]
    training = config["training"]
    assert isinstance(runtime, Mapping) and isinstance(data, Mapping) and isinstance(training, Mapping)
    output = workspace_path(runtime["output_root"]) / args.variant
    rank = world = local_rank = -1
    ready = False
    try:
        rank, world, local_rank, device = setup_distributed(config)
        ready = True
        existing_path = output / ("fixed_test_result.json" if args.test_only else "validation_result.json")
        existing = broadcast_rank0(
            rank,
            lambda: json.loads(existing_path.read_text()) if existing_path.is_file() else None,
        )
        if existing is not None:
            if existing.get("config_sha256") != config_sha or existing.get("variant") != args.variant:
                raise RuntimeError("existing result belongs to another contract")
            return 0 if existing.get("automatic_gate_pass") is True else 2
        preflight = broadcast_rank0(rank, lambda: run_preflight(args.config, full_hashes=False))
        runtime_evidence = validate_runtime(config, local_rank)
        os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        seed_everything(int(training["seed"]))
        all_records = read_jsonl(workspace_path(data["manifest"]))
        model, processor, scope = load_model(config, variant=args.variant, device=device)
        stages = data["curriculum"]
        assert isinstance(stages, list)
        final_checkpoint = checkpoint_directory(output, str(stages[-1]["name"]))
        if args.test_only:
            selection_path = workspace_path(runtime["output_root"]) / "checkpoint_selection.json"
            selection = broadcast_rank0(rank, lambda: json.loads(selection_path.read_text()))
            if selection.get("selected_variant") != args.variant:
                raise RuntimeError("test-only variant is not the frozen validation winner")
            if not final_checkpoint.is_dir():
                raise RuntimeError("selected final adapter does not exist")
            load_qwen_adapter(model, final_checkpoint / "adapter.safetensors")
            training_result = {"gradient_audit_pass": True, "fixed_adapter": True}
            split = "test"
        else:
            output.mkdir(parents=True, exist_ok=True)
            initialization_evidence = None
            initialization = config.get("initialization")
            if isinstance(initialization, Mapping):
                adapter = workspace_path(initialization["adapter"])
                if sha256(adapter) != initialization["adapter_sha256"]:
                    raise RuntimeError("initialization adapter hash changed")
                load_qwen_adapter(model, adapter)
                initialization_evidence = {
                    "adapter": initialization["adapter"],
                    "adapter_sha256": initialization["adapter_sha256"],
                    "validation_result": initialization["validation_result"],
                    "validation_result_sha256": initialization[
                        "validation_result_sha256"
                    ],
                    "optimizer": "fresh",
                }
                if rank == 0:
                    print(
                        "QWEN_MULTIDEGRADATION_INITIALIZATION_PASS "
                        + json.dumps(initialization_evidence, sort_keys=True),
                        flush=True,
                    )
            training_result = train_variant(
                model,
                processor,
                scope,
                all_records,
                config,
                config_sha=config_sha,
                variant=args.variant,
                output=output,
                rank=rank,
                world=world,
                local_rank=local_rank,
                device=device,
            )
            if initialization_evidence is not None:
                training_result["initialization"] = initialization_evidence
            split = "validation"
        rows, metrics = evaluate_split(
            model,
            processor,
            all_records,
            split,
            config,
            rank=rank,
            world=world,
            device=device,
        )
        checks = gate_checks(metrics, config, bool(training_result["gradient_audit_pass"]))
        automatic_pass = all(checks.values())
        if rank == 0:
            predictions_path = output / f"{split}_predictions.json"
            atomic_json(
                predictions_path,
                {"schema_version": 1, "config_sha256": config_sha, "variant": args.variant, "split": split, "records": rows},
            )
            result = {
                "schema_version": 1,
                "status": "passed" if automatic_pass else "failed",
                "automatic_gate_pass": automatic_pass,
                "config_sha256": config_sha,
                "variant": args.variant,
                "split": split,
                "metrics": metrics,
                "gate_checks": checks,
                "adapter": final_checkpoint.relative_to(ROOT).as_posix() + "/adapter.safetensors",
                "adapter_sha256": sha256(final_checkpoint / "adapter.safetensors"),
                "predictions": predictions_path.relative_to(ROOT).as_posix(),
                "predictions_sha256": sha256(predictions_path),
                "runtime": runtime_evidence,
                "training": training_result,
                "preflight_config_sha256": preflight["config_sha256"],
            }
            atomic_json(existing_path, result)
            print(
                ("QWEN_MULTIDEGRADATION_GATE_PASS " if automatic_pass else "QWEN_MULTIDEGRADATION_GATE_FAILED ")
                + json.dumps({"variant": args.variant, "split": split, "metrics": metrics}, sort_keys=True),
                flush=True,
            )
        pass_tensor = torch.tensor(int(automatic_pass), dtype=torch.int32, device=device)
        dist.broadcast(pass_tensor, src=0)
        return 0 if pass_tensor.item() else 2
    finally:
        if ready:
            dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
