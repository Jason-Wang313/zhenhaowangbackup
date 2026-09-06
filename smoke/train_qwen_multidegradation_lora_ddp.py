#!/usr/bin/env python3
"""Train/evaluate the two/four-GPU Qwen multi-degradation LoRA ablation."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import itertools
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
    DEGRADATIONS,
    FIXED_FIVE_SLOT_SCHEMA,
    PER_TYPE_DIALOGUE_SCHEMA,
    VARIABLE_ACTIVE_SCHEMA,
    DegradationLabel,
    aggregate_per_type_responses,
    evaluate_outputs,
    make_label,
    per_type_prompt_text,
    per_type_response,
    prepare_per_type_supervised_batch,
    prepare_supervised_batch,
    prompt_text,
)
from lowlight_restoration.qwen_type_calibration import (
    calibrated_level,
    fit_type_calibration,
    partition_records,
    validate_training_source_partition,
)
from smoke.qwen_multidegradation_lora_preflight import (
    accumulation_steps_for_world,
    load_config,
    micro_batch_size_for_world,
    run_preflight,
    validate_contract,
    validate_software_runtime,
    workspace_path,
)


def atomic_json(path: Path, payload: Mapping[str, object]) -> str:
    return write_new_or_verify(
        path, (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    )


def evaluation_artifact_paths(
    output: Path,
    *,
    split: str,
    validation_stage_name: str | None,
) -> tuple[Path, Path]:
    """Return immutable, non-colliding prediction and result paths."""
    if split == "test":
        if validation_stage_name is not None:
            raise ValueError("fixed-test evaluation cannot name a validation stage")
        return output / "test_predictions.json", output / "fixed_test_result.json"
    if split != "validation":
        raise ValueError(f"unsupported evaluation split: {split}")
    suffix = f"_{validation_stage_name}" if validation_stage_name else ""
    return (
        output / f"validation_predictions{suffix}.json",
        output / f"validation_result{suffix}.json",
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


@contextlib.contextmanager
def left_padded_generation(processor: Any):
    """Use decoder-safe left padding for a batched generation operation."""

    tokenizer = processor.tokenizer
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        yield
    finally:
        tokenizer.padding_side = original_padding_side


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
            image = handle.convert("RGB")
            crop_box = record.get("crop_box")
            if crop_box is not None:
                if (
                    not isinstance(crop_box, Sequence)
                    or isinstance(crop_box, (str, bytes))
                    or len(crop_box) != 4
                    or any(isinstance(value, bool) or not isinstance(value, int) for value in crop_box)
                ):
                    raise ValueError(f"invalid crop_box for {record.get('record_id')}")
                left, top, right, bottom = crop_box
                if left < 0 or top < 0 or right <= left or bottom <= top or right > image.width or bottom > image.height:
                    raise ValueError(f"out-of-bounds crop_box for {record.get('record_id')}")
                image = image.crop((left, top, right, bottom))
            images.append(image)
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
        language_rank=(
            int(model_config["language_lora_rank"])
            if "language_lora_rank" in model_config
            else None
        ),
        language_alpha=(
            float(model_config["language_lora_alpha"])
            if "language_lora_alpha" in model_config
            else None
        ),
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
            world * micro_batch_size_for_world(dict(config), world) * accumulation
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
    example_semantic_weights: Sequence[float] | None = None,
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
    if example_semantic_weights is None:
        semantic_weights = torch.full_like(weights, float(semantic_weight))
    else:
        if len(example_semantic_weights) != labels.shape[0]:
            raise ValueError("semantic example-weight cardinality changed")
        per_example = torch.as_tensor(
            example_semantic_weights,
            dtype=weights.dtype,
            device=weights.device,
        )
        if (
            not torch.all(torch.isfinite(per_example))
            or torch.any(per_example <= 1.0)
        ):
            raise ValueError("semantic example weights must be finite and greater than one")
        semantic_weights = per_example[:, None].expand_as(shifted_labels)[valid]
    weights = torch.where(semantic, semantic_weights, weights)
    loss = (token_loss * weights).sum() / weights.sum()
    if not torch.isfinite(loss):
        raise RuntimeError("semantic assistant-token loss is non-finite")
    return loss


def decoder_presence_auxiliary_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    level_token_ids: Sequence[int],
) -> torch.Tensor:
    """Binary absent/present loss derived only from Qwen decoder logits.

    L1 is the absent score. The present score is log-sum-exp over L2--L5.
    This adds no classifier head and preserves the decoder as the decision
    maker while directly penalizing omission of an active degradation.
    """

    if logits.ndim != 3 or labels.ndim != 2 or logits.shape[:2] != labels.shape:
        raise ValueError("presence-loss logits and labels have incompatible shapes")
    if (
        len(level_token_ids) != 5
        or len(set(level_token_ids)) != 5
        or any(
            isinstance(token, bool) or not isinstance(token, int) or token < 0
            for token in level_token_ids
        )
    ):
        raise ValueError("presence loss requires five unique L1--L5 token IDs")
    shifted_logits = logits[:, :-1, :]
    shifted_labels = labels[:, 1:].contiguous()
    decision = torch.zeros_like(shifted_labels, dtype=torch.bool)
    for token_id in level_token_ids:
        decision.logical_or_(shifted_labels.eq(token_id))
    if not torch.any(decision):
        raise ValueError("presence loss found no supervised L1--L5 decisions")
    decision_labels = shifted_labels[decision]
    level_scores = shifted_logits[decision].float()[:, list(level_token_ids)]
    binary_scores = torch.stack(
        (level_scores[:, 0], torch.logsumexp(level_scores[:, 1:], dim=1)),
        dim=1,
    )
    targets = decision_labels.ne(level_token_ids[0]).long()
    loss = F.cross_entropy(binary_scores, targets)
    if not torch.isfinite(loss):
        raise RuntimeError("decoder presence auxiliary loss is non-finite")
    return loss


def training_loss(
    output_value: object,
    labels: torch.Tensor,
    training: Mapping[str, object],
    *,
    query_degradations: Sequence[str] | None = None,
) -> torch.Tensor:
    """Select the frozen baseline CE or the bounded semantic correction CE."""

    semantic = training.get("semantic_token_loss")
    logits = getattr(output_value, "logits", None)
    if semantic is None:
        primary_loss = getattr(output_value, "loss", None)
        if not isinstance(primary_loss, torch.Tensor):
            raise RuntimeError("Qwen output omitted its language-model loss")
    else:
        if not isinstance(semantic, Mapping) or semantic.get("enabled") is not True:
            raise ValueError("semantic-token loss contract is malformed")
        if not isinstance(logits, torch.Tensor):
            raise RuntimeError("Qwen output omitted logits for semantic-token loss")
        token_ids = semantic.get("semantic_token_ids")
        if not isinstance(token_ids, list):
            raise ValueError("semantic-token loss IDs must be a list")
        type_weights = semantic.get("semantic_weight_by_degradation")
        example_weights: list[float] | None = None
        if type_weights is not None:
            if (
                not isinstance(type_weights, Mapping)
                or set(map(str, type_weights)) != set(DEGRADATIONS)
                or query_degradations is None
                or any(name not in DEGRADATIONS for name in query_degradations)
            ):
                raise ValueError("per-degradation semantic weights are malformed")
            example_weights = [float(type_weights[name]) for name in query_degradations]
        primary_loss = semantic_assistant_token_loss(
            logits,
            labels,
            semantic_token_ids=token_ids,
            semantic_weight=float(semantic["semantic_weight"]),
            example_semantic_weights=example_weights,
        )

    presence = training.get("presence_auxiliary_loss")
    if presence is None:
        return primary_loss
    if not isinstance(presence, Mapping) or presence.get("enabled") is not True:
        raise ValueError("presence auxiliary loss contract is malformed")
    if not isinstance(logits, torch.Tensor):
        raise RuntimeError("Qwen output omitted logits for presence auxiliary loss")
    level_token_ids = presence.get("level_token_ids")
    weight = float(presence.get("weight", 0.0))
    if not isinstance(level_token_ids, list) or not math.isfinite(weight) or weight <= 0:
        raise ValueError("presence auxiliary loss parameters are malformed")
    return primary_loss + weight * decoder_presence_auxiliary_loss(
        logits,
        labels,
        level_token_ids=level_token_ids,
    )


def create_optimizer(
    model: torch.nn.Module, config: Mapping[str, object]
) -> torch.optim.Optimizer:
    training = config["training"]
    assert isinstance(training, Mapping)
    lora = []
    language_lora = []
    merger = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("model.visual.merger."):
            merger.append(parameter)
        elif name.startswith("model.language_model."):
            language_lora.append(parameter)
        else:
            lora.append(parameter)
    groups = [{"params": lora, "lr": float(training["lora_learning_rate"]), "kind": "lora"}]
    if merger:
        groups.append({"params": merger, "lr": float(training["merger_learning_rate"]), "kind": "merger"})
    if language_lora:
        groups.append(
            {
                "params": language_lora,
                "lr": float(training["language_lora_learning_rate"]),
                "kind": "language_lora",
            }
        )
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
    global_batch_size: int,
) -> dict[str, object]:
    final = checkpoint_directory(output, stage_name)
    if final.is_dir():
        metadata = json.loads((final / "metadata.json").read_text())
        if (
            metadata.get("config_sha256") != config_sha
            or metadata.get("variant") != variant
            or metadata.get("world_size") != world_size
            or metadata.get("gradient_accumulation_steps") != accumulation_steps
            or metadata.get("global_batch_size") != global_batch_size
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
                "global_batch_size": global_batch_size,
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
            "global_batch_size": global_batch_size,
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
    global_batch_size: int,
    initialization: Mapping[str, object] | None = None,
    allow_world_size_change: bool = False,
) -> tuple[int, int]:
    completed = 0
    global_step = 0
    for index, stage in enumerate(stages):
        directory = checkpoint_directory(output, str(stage["name"]))
        if not directory.is_dir():
            break
        metadata = json.loads((directory / "metadata.json").read_text())
        distributed_contract_matches = (
            metadata.get("global_batch_size") == global_batch_size
            if allow_world_size_change
            else metadata.get("world_size") == world_size
            and metadata.get("gradient_accumulation_steps") == accumulation_steps
        )
        if (
            metadata.get("config_sha256") != config_sha
            or metadata.get("variant") != variant
            or not distributed_contract_matches
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
    elif initialization is not None and initialization.get("optimizer") is not None:
        adapter = workspace_path(initialization["adapter"])
        optimizer_path = workspace_path(initialization["optimizer"])
        if sha256(adapter) != initialization.get("adapter_sha256"):
            raise RuntimeError("external continuation adapter hash failed")
        if sha256(optimizer_path) != initialization.get("optimizer_sha256"):
            raise RuntimeError("external continuation optimizer hash failed")
        load_qwen_adapter(model, adapter)
        optimizer.load_state_dict(
            torch.load(optimizer_path, map_location="cpu", weights_only=False)
        )
        optimizer_to_device(optimizer)
        global_step = int(initialization.get("initial_optimizer_steps", 0))
        if global_step <= 0:
            raise RuntimeError("external continuation optimizer-step offset is invalid")
    return completed, global_step


def validate_completed_stage_checkpoints(
    *,
    output: Path,
    stages: Sequence[Mapping[str, object]],
    config_sha: str,
    variant: str,
    world_size: int,
    accumulation_steps: int,
    global_batch_size: int,
    initial_global_step: int = 0,
    allow_world_size_change: bool = False,
) -> list[dict[str, object]]:
    """Verify a completed run before evaluation-only recovery."""

    expected_global_step = initial_global_step
    evidence: list[dict[str, object]] = []
    for stage in stages:
        expected_global_step += (
            int(stage["records"])
            * int(stage.get("epochs", 1))
            // global_batch_size
        )
        directory = checkpoint_directory(output, str(stage["name"]))
        if not directory.is_dir():
            raise RuntimeError(f"validation-only checkpoint is missing: {directory}")
        metadata = json.loads((directory / "metadata.json").read_text())
        distributed_contract_matches = (
            metadata.get("global_batch_size") == global_batch_size
            if allow_world_size_change
            else metadata.get("world_size") == world_size
            and metadata.get("gradient_accumulation_steps") == accumulation_steps
        )
        if (
            metadata.get("config_sha256") != config_sha
            or metadata.get("variant") != variant
            or not distributed_contract_matches
            or metadata.get("global_step") != expected_global_step
        ):
            raise RuntimeError(f"validation-only checkpoint contract mismatch: {directory}")
        adapter_sha = sha256(directory / "adapter.safetensors")
        optimizer_sha = sha256(directory / "optimizer.pt")
        if (
            adapter_sha != metadata.get("adapter_sha256")
            or optimizer_sha != metadata.get("optimizer_sha256")
        ):
            raise RuntimeError(f"validation-only checkpoint hash failed: {directory}")
        evidence.append(
            {
                "stage": stage["name"],
                "global_step": expected_global_step,
                "adapter_sha256": adapter_sha,
                "optimizer_sha256": optimizer_sha,
            }
        )
    return evidence


def stage_records(
    records: Sequence[Mapping[str, object]],
    stage: Mapping[str, object],
    seed: int,
    *,
    epoch: int = 0,
    target_schema: str = VARIABLE_ACTIVE_SCHEMA,
    global_batch_size: int | None = None,
    excluded_source_ids: Sequence[str] = (),
) -> list[Mapping[str, object]]:
    excluded = set(map(str, excluded_source_ids))
    selected = [
        record
        for record in records
        if record["split"] == "train"
        and int(record["composition_size"]) <= int(stage["maximum_composition_size"])
        and str(record["source_id"]) not in excluded
    ]
    sampling = stage.get("source_sampling")
    if sampling is None:
        source_order_key = str(stage.get("source_order_key", stage["name"]))
        selected.sort(
            key=lambda record: (
                stable_digest(seed, source_order_key, record["record_id"])
                if epoch == 0
                else stable_digest(seed, source_order_key, epoch, record["record_id"])
            )
        )
    else:
        if not isinstance(sampling, Mapping) or sampling.get("schema") != "hard_triple_mixture_v1":
            raise ValueError("hard-triple sampling contract is malformed")
        composition_quotas = sampling.get("composition_source_quotas")
        triple_quotas = sampling.get("triple_combination_source_quotas")
        if not isinstance(composition_quotas, Mapping) or not isinstance(triple_quotas, Mapping):
            raise ValueError("hard-triple sampling quotas are missing")

        def deterministic_take(
            candidates: Sequence[Mapping[str, object]],
            quota: int,
            salt: str,
        ) -> list[Mapping[str, object]]:
            if quota <= 0 or not candidates:
                raise ValueError(f"invalid hard-triple pool or quota: {salt}")
            result: list[Mapping[str, object]] = []
            cycle = 0
            while len(result) < quota:
                ordered = sorted(
                    candidates,
                    key=lambda record: stable_digest(
                        seed,
                        stage["name"],
                        epoch,
                        "hard_triple",
                        salt,
                        cycle,
                        record["record_id"],
                    ),
                )
                result.extend(ordered[: quota - len(result)])
                cycle += 1
            return result

        sampled: list[Mapping[str, object]] = []
        for composition_size in range(3):
            quota = int(composition_quotas.get(str(composition_size), 0))
            pool = [
                record
                for record in selected
                if int(record["composition_size"]) == composition_size
            ]
            sampled.extend(deterministic_take(pool, quota, f"c{composition_size}"))
        expected_triples = {
            "+".join(combo)
            for combo in itertools.combinations(DEGRADATIONS, 3)
        }
        if set(map(str, triple_quotas)) != expected_triples:
            raise ValueError("hard-triple combination coverage changed")
        triple_total = 0
        for combo_text in sorted(expected_triples):
            combo = tuple(combo_text.split("+"))
            quota = int(triple_quotas[combo_text])
            pool = [
                record
                for record in selected
                if tuple(record["degradations"]) == combo
            ]
            sampled.extend(deterministic_take(pool, quota, combo_text))
            triple_total += quota
        if triple_total != int(composition_quotas.get("3", 0)):
            raise ValueError("hard-triple total differs from composition quota")
        if len(sampled) != int(stage.get("source_records", 0)):
            raise RuntimeError("hard-triple sampled source count changed")
        selected = sorted(
            sampled,
            key=lambda record: stable_digest(
                seed,
                stage["name"],
                epoch,
                "hard_triple_final",
                record["record_id"],
                len(str(record)),
            ),
        )
    if target_schema == PER_TYPE_DIALOGUE_SCHEMA:
        source_records = int(stage.get("source_records", 0))
        source_start = int(stage.get("source_start", 0))
        if source_start < 0 or len(selected) < source_start + source_records:
            raise RuntimeError(f"stage source record count changed: {stage['name']}")
        selected = selected[source_start : source_start + source_records]
        if global_batch_size is None or global_batch_size <= 0:
            raise ValueError("per-type dialogue stages require a global batch size")
        if global_batch_size % len(DEGRADATIONS):
            raise ValueError("per-type global batch must be divisible by five")
        images_per_batch = global_batch_size // len(DEGRADATIONS)
        if len(selected) % images_per_batch:
            raise ValueError("source records do not fill complete type-balanced batches")
        dialogues: list[Mapping[str, object]] = []
        for offset in range(0, len(selected), images_per_batch):
            chunk = selected[offset : offset + images_per_batch]
            units = [
                {**record, "query_degradation": degradation}
                for record in chunk
                for degradation in DEGRADATIONS
            ]
            units.sort(
                key=lambda record: stable_digest(
                    seed,
                    stage["name"],
                    epoch,
                    "dialogue",
                    record["record_id"],
                    record["query_degradation"],
                )
            )
            dialogues.extend(units)
        if len(dialogues) != int(stage["records"]):
            raise RuntimeError(f"stage dialogue record count changed: {stage['name']}")
        return dialogues
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
    micro = micro_batch_size_for_world(dict(config), world)
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
        global_batch_size=global_batch,
        initialization=(
            config.get("initialization")
            if isinstance(config.get("initialization"), Mapping)
            else None
        ),
        allow_world_size_change=bool(
            training.get("allow_world_size_change_on_resume", False)
        ),
    )
    total_steps = int(training["total_optimizer_steps"])
    trainable_names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    gradient_seen: set[str] = set()
    frozen_gradient_names: set[str] = set()
    metrics: list[dict[str, object]] = []
    epoch_summaries: list[dict[str, object]] = []
    started = time.monotonic()
    prompt_config = config["prompt"]
    assert isinstance(prompt_config, Mapping)
    target_schema = str(prompt_config["target_schema"])
    excluded_source_ids: Sequence[str] = ()
    partition_config = data.get("calibration_partition")
    if partition_config is not None:
        if not isinstance(partition_config, Mapping):
            raise ValueError("training-only calibration partition config is malformed")
        partition_path = workspace_path(partition_config["path"])
        if sha256(partition_path) != partition_config.get("sha256"):
            raise RuntimeError("training-only calibration partition changed")
        partition = json.loads(partition_path.read_text())
        validate_training_source_partition(
            partition,
            records,
            manifest_sha256=str(data["manifest_sha256"]),
        )
        excluded_source_ids = list(partition["calibration_source_ids"])

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
                records,
                stage,
                int(training["seed"]),
                epoch=epoch,
                target_schema=target_schema,
                global_batch_size=global_batch,
                excluded_source_ids=excluded_source_ids,
            )
            epoch_losses: list[float] = []
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
                    if target_schema == PER_TYPE_DIALOGUE_SCHEMA:
                        batch, _ = prepare_per_type_supervised_batch(
                            processor,
                            labels,
                            images,
                            [str(record["query_degradation"]) for record in local_records],
                        )
                    else:
                        batch, _ = prepare_supervised_batch(
                            processor,
                            labels,
                            images,
                            target_schema,
                        )
                    synchronization = (
                        ddp.no_sync()
                        if accumulation_index + 1 < accumulation
                        else contextlib.nullcontext()
                    )
                    with synchronization:
                        device_batch = move_batch(batch, device)
                        forward_batch = (
                            {key: value for key, value in device_batch.items() if key != "labels"}
                            if (
                                training.get("semantic_token_loss") is not None
                                or training.get("presence_auxiliary_loss") is not None
                            )
                            else device_batch
                        )
                        output_value = ddp(**forward_batch, use_cache=False)
                        loss = training_loss(
                            output_value,
                            device_batch["labels"],
                            training,
                            query_degradations=(
                                [str(record["query_degradation"]) for record in local_records]
                                if target_schema == PER_TYPE_DIALOGUE_SCHEMA
                                else None
                            ),
                        )
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
                initial_optimizer_steps = int(
                    training.get("initial_optimizer_steps", 0)
                )
                schedule_step = global_step - initial_optimizer_steps
                schedule_total_steps = int(
                    training.get("continuation_optimizer_steps", total_steps)
                )
                scale = learning_rate_scale(
                    schedule_step,
                    schedule_total_steps,
                    int(training["warmup_steps"]),
                    float(training["minimum_learning_rate_ratio"]),
                )
                for group in optimizer.param_groups:
                    rate_key = {
                        "lora": "lora_learning_rate",
                        "merger": "merger_learning_rate",
                        "language_lora": "language_lora_learning_rate",
                    }[group["kind"]]
                    base = float(training[rate_key])
                    group["lr"] = base * scale
                optimizer.step()
                dist.all_reduce(loss_value, op=dist.ReduceOp.SUM)
                loss_value /= world
                epoch_losses.append(float(loss_value.item()))
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
            window = max(1, len(epoch_losses) // 5)
            epoch_summary = {
                "stage": stage["name"],
                "stage_epoch": epoch + 1,
                "optimizer_steps": len(epoch_losses),
                "dialogue_presentations": len(epoch_losses) * global_batch,
                "mean_loss": sum(epoch_losses) / len(epoch_losses),
                "first_window_mean_loss": sum(epoch_losses[:window]) / window,
                "last_window_mean_loss": sum(epoch_losses[-window:]) / window,
                "loss_still_descending": (
                    sum(epoch_losses[-window:]) / window
                    < sum(epoch_losses[:window]) / window
                ),
            }
            epoch_summaries.append(epoch_summary)
            if rank == 0:
                print(
                    "QWEN_MULTIDEGRADATION_EPOCH_SUMMARY "
                    + json.dumps(epoch_summary, sort_keys=True),
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
                global_batch_size=global_batch,
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
        "epoch_summaries": epoch_summaries,
        "dialogue_presentations_this_contract": (
            global_step - int(training.get("initial_optimizer_steps", 0))
        )
        * global_batch,
        "dialogue_presentations_cumulative": int(
            config.get("initialization", {}).get("initial_dialogue_presentations", 0)
            if isinstance(config.get("initialization"), Mapping)
            else 0
        )
        + (
            global_step - int(training.get("initial_optimizer_steps", 0))
        )
        * global_batch,
    }


def generate_per_type_outputs(
    model: torch.nn.Module,
    processor: Any,
    records: Sequence[Mapping[str, object]],
    config: Mapping[str, object],
    *,
    rank: int,
    world: int,
    device: torch.device,
) -> tuple[list[str], list[dict[str, dict[str, object]]]]:
    """Ask Qwen five separate questions and aggregate its decoder answers."""

    generation = config["generation"]
    assert isinstance(generation, Mapping)
    model.eval()
    model.gradient_checkpointing_disable()
    model.config.use_cache = True
    indexed = [
        (record_index, degradation, record)
        for record_index, record in enumerate(records)
        for degradation in DEGRADATIONS
    ]
    local_indexed = [
        item for flat_index, item in enumerate(indexed) if flat_index % world == rank
    ]
    local: list[tuple[int, str, str]] = []
    batch_size = int(generation["batch_size_per_rank"])
    with left_padded_generation(processor):
        for offset in range(0, len(local_indexed), batch_size):
            chunk = local_indexed[offset : offset + batch_size]
            images = open_images([record for _, _, record in chunk])
            encoded = processor(
                text=[
                    per_type_prompt_text(processor, degradation)
                    for _, degradation, _ in chunk
                ],
                images=images,
                padding=True,
                return_tensors="pt",
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
            for row, (record_index, degradation, _) in enumerate(chunk):
                raw = processor.tokenizer.decode(
                    generated[row, input_length:],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                ).strip()
                local.append((record_index, degradation, raw))
            if (offset // batch_size + 1) % 25 == 0 or offset + batch_size >= len(local_indexed):
                print(
                    f"QWEN_MULTIDEGRADATION_PER_TYPE_GENERATE rank={rank} "
                    f"completed={min(offset+batch_size,len(local_indexed))}/{len(local_indexed)}",
                    flush=True,
                )
    gathered: list[object | None] = [None] * world
    dist.all_gather_object(gathered, local)
    flattened = [
        item
        for rank_items in gathered
        if isinstance(rank_items, list)
        for item in rank_items
    ]
    expected_count = len(records) * len(DEGRADATIONS)
    identities = [(index, degradation) for index, degradation, _ in flattened]
    expected_identities = [
        (index, degradation)
        for index in range(len(records))
        for degradation in DEGRADATIONS
    ]
    if len(flattened) != expected_count or sorted(identities) != sorted(expected_identities):
        raise RuntimeError("distributed per-type Qwen generation is incomplete")
    raw_lookup = {
        (index, degradation): raw
        for index, degradation, raw in flattened
    }
    aggregated: list[str] = []
    details: list[dict[str, dict[str, object]]] = []
    for index in range(len(records)):
        raw_by_type = {
            degradation: raw_lookup[(index, degradation)]
            for degradation in DEGRADATIONS
        }
        serialized, record_details = aggregate_per_type_responses(raw_by_type)
        aggregated.append(serialized if serialized is not None else "INVALID_PER_TYPE_DIALOGUE")
        details.append(record_details)
    return aggregated, details


def score_per_type_candidates(
    model: torch.nn.Module,
    processor: Any,
    records: Sequence[Mapping[str, object]],
    config: Mapping[str, object],
    *,
    rank: int,
    world: int,
    device: torch.device,
) -> list[dict[str, dict[str, object]]]:
    """Score L1--L5 at the decoder decision token without using ground truth."""

    calibration_config = config.get("decision_calibration")
    if not isinstance(calibration_config, Mapping):
        raise ValueError("decision-calibration contract is missing")
    digit_ids = calibration_config.get("level_token_ids")
    if (
        not isinstance(digit_ids, list)
        or len(digit_ids) != 5
        or any(not isinstance(value, int) for value in digit_ids)
    ):
        raise ValueError("L1--L5 decision token IDs are malformed")
    indexed = [
        (record_index, degradation, record)
        for record_index, record in enumerate(records)
        for degradation in DEGRADATIONS
    ]
    local_indexed = [
        item for flat_index, item in enumerate(indexed) if flat_index % world == rank
    ]
    local: list[tuple[int, str, list[float]]] = []
    batch_size = int(calibration_config["candidate_batch_size_per_rank"])
    placeholder = make_label((), {})
    model.eval()
    model.gradient_checkpointing_disable()
    model.config.use_cache = False
    for offset in range(0, len(local_indexed), batch_size):
        chunk = local_indexed[offset : offset + batch_size]
        images = open_images([record for _, _, record in chunk])
        queries = [degradation for _, degradation, _ in chunk]
        batch, _ = prepare_per_type_supervised_batch(
            processor,
            [placeholder] * len(chunk),
            images,
            queries,
        )
        labels = batch["labels"]
        positions: list[int] = []
        for row in range(labels.shape[0]):
            matches = torch.nonzero(labels[row].eq(int(digit_ids[0])), as_tuple=False)
            if matches.shape != (1, 1):
                raise RuntimeError("forced L1 response has an ambiguous decision token")
            position = int(matches[0, 0].item())
            if position <= 0:
                raise RuntimeError("decision token has no causal predecessor")
            positions.append(position)
        inputs = move_batch(
            {key: value for key, value in batch.items() if key != "labels"},
            device,
        )
        with torch.no_grad():
            output_value = model(**inputs, use_cache=False)
        logits = getattr(output_value, "logits", None)
        if not isinstance(logits, torch.Tensor):
            raise RuntimeError("Qwen candidate scoring omitted decoder logits")
        for row, ((record_index, degradation, _), position) in enumerate(
            zip(chunk, positions, strict=True)
        ):
            values = logits[row, position - 1, digit_ids].float().cpu().tolist()
            local.append((record_index, degradation, [float(value) for value in values]))
        if (offset // batch_size + 1) % 25 == 0 or offset + batch_size >= len(local_indexed):
            print(
                f"QWEN_MULTIDEGRADATION_CANDIDATE_SCORE rank={rank} "
                f"completed={min(offset+batch_size,len(local_indexed))}/{len(local_indexed)}",
                flush=True,
            )
    gathered: list[object | None] = [None] * world
    dist.all_gather_object(gathered, local)
    flattened = [
        item
        for rank_items in gathered
        if isinstance(rank_items, list)
        for item in rank_items
    ]
    identities = [(index, degradation) for index, degradation, _ in flattened]
    expected = [
        (index, degradation)
        for index in range(len(records))
        for degradation in DEGRADATIONS
    ]
    if len(flattened) != len(expected) or sorted(identities) != sorted(expected):
        raise RuntimeError("distributed L1--L5 candidate scoring is incomplete")
    lookup = {
        (index, degradation): scores
        for index, degradation, scores in flattened
    }
    return [
        {
            degradation: {"candidate_scores": lookup[(index, degradation)]}
            for degradation in DEGRADATIONS
        }
        for index in range(len(records))
    ]


def load_or_fit_training_calibration(
    model: torch.nn.Module,
    processor: Any,
    all_records: Sequence[Mapping[str, object]],
    config: Mapping[str, object],
    *,
    config_sha: str,
    adapter_sha: str,
    output: Path,
    rank: int,
    world: int,
    device: torch.device,
) -> dict[str, object]:
    """Fit once on reserved training sources, or verify and reuse that fit."""

    calibration_config = config.get("decision_calibration")
    data = config["data"]
    if not isinstance(calibration_config, Mapping) or not isinstance(data, Mapping):
        raise ValueError("decision calibration/data contract is malformed")
    partition_config = data.get("calibration_partition")
    if not isinstance(partition_config, Mapping):
        raise ValueError("training-only calibration partition is missing")
    partition_path = workspace_path(partition_config["path"])
    if sha256(partition_path) != partition_config.get("sha256"):
        raise RuntimeError("training-only calibration partition changed")
    partition = json.loads(partition_path.read_text())
    validate_training_source_partition(
        partition,
        all_records,
        manifest_sha256=str(data["manifest_sha256"]),
    )
    calibration_path = output / "training_only_type_calibration.json"
    score_path = output / "training_only_candidate_scores.json"

    def existing_calibration() -> dict[str, object] | None:
        if not calibration_path.is_file():
            return None
        payload = json.loads(calibration_path.read_text())
        if (
            payload.get("config_sha256") != config_sha
            or payload.get("adapter_sha256") != adapter_sha
            or payload.get("partition_sha256") != partition_config["sha256"]
            or payload.get("validation_or_test_used_for_fit") is not False
            or not score_path.is_file()
            or sha256(score_path) != payload.get("candidate_scores_sha256")
        ):
            raise RuntimeError("existing training-only calibration identity changed")
        return payload

    existing = broadcast_rank0(rank, existing_calibration)
    if existing is not None:
        return dict(existing)
    calibration_records, _ = partition_records(all_records, partition)
    scored = score_per_type_candidates(
        model,
        processor,
        calibration_records,
        config,
        rank=rank,
        world=world,
        device=device,
    )

    def fit_and_publish() -> dict[str, object]:
        rows: list[dict[str, object]] = []
        for record, details in zip(calibration_records, scored, strict=True):
            levels = record["levels"]
            if not isinstance(levels, Mapping):
                raise RuntimeError("calibration truth levels are malformed")
            for degradation in DEGRADATIONS:
                rows.append(
                    {
                        "record_id": record["record_id"],
                        "source_id": record["source_id"],
                        "degradation": degradation,
                        "true_level": int(levels.get(degradation, 1)),
                        "candidate_scores": details[degradation]["candidate_scores"],
                    }
                )
        score_payload = {
            "schema_version": 1,
            "status": "training_sources_only",
            "config_sha256": config_sha,
            "adapter_sha256": adapter_sha,
            "partition": partition_config["path"],
            "partition_sha256": partition_config["sha256"],
            "validation_or_test_used": False,
            "rows": rows,
        }
        atomic_json(score_path, score_payload)
        fitted = fit_type_calibration(
            rows,
            regularization=float(calibration_config["regularization"]),
            maximum_iterations=int(calibration_config["maximum_iterations"]),
        )
        fitted.update(
            {
                "config_sha256": config_sha,
                "adapter_sha256": adapter_sha,
                "partition": partition_config["path"],
                "partition_sha256": partition_config["sha256"],
                "candidate_scores": score_path.relative_to(ROOT).as_posix(),
                "candidate_scores_sha256": sha256(score_path),
            }
        )
        atomic_json(calibration_path, fitted)
        return fitted

    fitted = broadcast_rank0(rank, fit_and_publish)
    return dict(fitted)


def calibrated_per_type_outputs(
    scored: Sequence[Mapping[str, Mapping[str, object]]],
    calibration: Mapping[str, object],
) -> tuple[list[str], list[dict[str, dict[str, object]]]]:
    aggregated: list[str] = []
    all_details: list[dict[str, dict[str, object]]] = []
    for record_scores in scored:
        raw_by_type: dict[str, str] = {}
        details: dict[str, dict[str, object]] = {}
        for degradation in DEGRADATIONS:
            candidate_scores = record_scores[degradation]["candidate_scores"]
            level, adjusted = calibrated_level(
                degradation, candidate_scores, calibration
            )
            raw = per_type_response(degradation, level)
            raw_by_type[degradation] = raw
            details[degradation] = {
                "raw_output": raw,
                "predicted_level": level,
                "candidate_scores": list(candidate_scores),
                "calibrated_scores": adjusted,
                "decision_rule": "training_only_affine_calibrated_l1_l5_argmax",
            }
        serialized, parsed = aggregate_per_type_responses(raw_by_type)
        if serialized is None:
            raise RuntimeError("calibrated candidate decisions failed to aggregate")
        for degradation in DEGRADATIONS:
            if parsed[degradation]["predicted_level"] != details[degradation]["predicted_level"]:
                raise RuntimeError("calibrated decision changed during serialization")
        aggregated.append(serialized)
        all_details.append(details)
    return aggregated, all_details


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
    prompt_config = config["prompt"]
    assert isinstance(prompt_config, Mapping)
    target_schema = str(prompt_config["target_schema"])
    if target_schema == PER_TYPE_DIALOGUE_SCHEMA:
        aggregated, _ = generate_per_type_outputs(
            model,
            processor,
            records,
            config,
            rank=rank,
            world=world,
            device=device,
        )
        return aggregated
    prompt = prompt_text(processor, target_schema)
    indexed = [(index, record) for index, record in enumerate(records) if index % world == rank]
    local: list[tuple[int, str]] = []
    batch_size = int(generation["batch_size_per_rank"])
    with left_padded_generation(processor):
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


def gate_checks(
    metrics: Mapping[str, object],
    config: Mapping[str, object],
    gradient_pass: bool,
    *,
    split: str = "validation",
) -> dict[str, bool]:
    gate = config["gate"]
    assert isinstance(gate, Mapping)
    triple = metrics["by_composition_size"]["3"]
    if split not in {"validation", "test"}:
        raise ValueError(f"unsupported gate split: {split}")
    triple_threshold = float(
        gate.get(
            "minimum_protected_test_triple_degradation_set_exact_match_rate",
            gate["minimum_triple_degradation_set_exact_match_rate"],
        )
        if split == "test"
        else gate["minimum_triple_degradation_set_exact_match_rate"]
    )
    checks = {
        "valid_structure": float(metrics["valid_structured_output_rate"]) >= float(gate["minimum_valid_structured_output_rate"]),
        "type_macro_f1": float(metrics["type_macro_f1"]) >= float(gate["minimum_type_macro_f1"]),
        "type_micro_f1": float(metrics["type_micro_f1"]) >= float(gate["minimum_type_micro_f1"]),
        "degradation_set_exact": float(metrics["degradation_set_exact_match_rate"]) >= float(gate["minimum_degradation_set_exact_match_rate"]),
        "combined_label_exact": float(metrics["combined_label_exact_match_rate"]) >= float(gate["minimum_combined_label_exact_match_rate"]),
        "severity_accuracy": float(metrics["severity_typed_accuracy"]) >= float(gate["minimum_severity_typed_accuracy"]),
        "severity_mae": float(metrics["severity_typed_mae_with_missing_penalty"]) <= float(gate["maximum_severity_typed_mae_with_missing_penalty"]),
        "triple_set_exact": float(triple["degradation_set_exact_match_rate"]) >= triple_threshold,
        "gradient_scope": gradient_pass,
    }
    if "maximum_severity_typed_mse_with_missing_penalty" in gate:
        checks["severity_mse"] = float(
            metrics["severity_typed_mse_with_missing_penalty"]
        ) < float(gate["maximum_severity_typed_mse_with_missing_penalty"])
    if "minimum_triple_combined_label_exact_match_rate" in gate:
        checks["triple_combined_label_exact"] = float(
            triple["combined_label_exact_match_rate"]
        ) >= float(gate["minimum_triple_combined_label_exact_match_rate"])
    if "minimum_per_type_valid_response_rate" in gate:
        checks["per_type_valid_response"] = (
            float(metrics["per_type_valid_response_rate"])
            >= float(gate["minimum_per_type_valid_response_rate"])
        )
    if "minimum_per_type_level_accuracy" in gate:
        checks["per_type_level_accuracy"] = (
            float(metrics["per_type_level_accuracy"])
            >= float(gate["minimum_per_type_level_accuracy"])
        )
    return checks


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
    output: Path,
    config_sha: str,
    adapter_sha: str,
    evaluation_records: Sequence[Mapping[str, object]] | None = None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    records = (
        list(evaluation_records)
        if evaluation_records is not None
        else [record for record in all_records if record["split"] == split]
    )
    if not records or any(record.get("split") != split for record in records):
        raise RuntimeError("external evaluation records have the wrong split")
    prompt_config = config["prompt"]
    assert isinstance(prompt_config, Mapping)
    target_schema = str(prompt_config["target_schema"])
    per_type_details: list[dict[str, dict[str, object]]] | None = None
    if target_schema == PER_TYPE_DIALOGUE_SCHEMA:
        if config.get("decision_calibration") is not None:
            calibration = load_or_fit_training_calibration(
                model,
                processor,
                all_records,
                config,
                config_sha=config_sha,
                adapter_sha=adapter_sha,
                output=output,
                rank=rank,
                world=world,
                device=device,
            )
            scored = score_per_type_candidates(
                model,
                processor,
                records,
                config,
                rank=rank,
                world=world,
                device=device,
            )
            raw, per_type_details = calibrated_per_type_outputs(
                scored, calibration
            )
        else:
            calibration = None
            raw, per_type_details = generate_per_type_outputs(
                model,
                processor,
                records,
                config,
                rank=rank,
                world=world,
                device=device,
            )
        evaluation_schema = FIXED_FIVE_SLOT_SCHEMA
    else:
        raw = generate_outputs(
            model, processor, records, config, rank=rank, world=world, device=device
        )
        evaluation_schema = target_schema
    rows, metrics = evaluate_outputs(
        [label_for(record) for record in records],
        raw,
        target_schema=evaluation_schema,
    )
    for row, record in zip(rows, records, strict=True):
        row["record_id"] = record["record_id"]
        row["source_id"] = record["source_id"]
    if per_type_details is not None:
        total = valid = correct = active_total = active_correct = 0
        absent_total = absent_correct = 0
        per_type = {
            degradation: {"samples": 0, "valid": 0, "correct": 0}
            for degradation in DEGRADATIONS
        }
        for row, record, details in zip(
            rows, records, per_type_details, strict=True
        ):
            truth = label_for(record)
            row["per_type_responses"] = details
            for degradation in DEGRADATIONS:
                true_level = truth.level_map.get(degradation, 1)
                detail = details[degradation]
                predicted_level = detail["predicted_level"]
                is_valid = predicted_level is not None
                is_correct = predicted_level == true_level
                total += 1
                valid += int(is_valid)
                correct += int(is_correct)
                active = true_level > 1
                active_total += int(active)
                active_correct += int(active and is_correct)
                absent_total += int(not active)
                absent_correct += int(not active and is_correct)
                counts = per_type[degradation]
                counts["samples"] += 1
                counts["valid"] += int(is_valid)
                counts["correct"] += int(is_correct)
        metrics.update(
            {
                "per_type_query_count": total,
                "per_type_valid_response_rate": valid / total,
                "per_type_level_accuracy": correct / total,
                "per_type_active_level_accuracy": active_correct / active_total,
                "per_type_absent_level_accuracy": absent_correct / absent_total,
                "per_type_dialogue": {
                    degradation: {
                        **counts,
                        "valid_response_rate": counts["valid"] / counts["samples"],
                        "level_accuracy": counts["correct"] / counts["samples"],
                    }
                    for degradation, counts in per_type.items()
                },
            }
        )
        if calibration is not None:
            metrics["decision_calibration"] = {
                "schema": calibration["calibration_schema"],
                "fit_data": "reserved_training_sources_only",
                "fit_records": calibration["records"],
                "raw_nll": calibration["raw_nll"],
                "calibrated_nll": calibration["calibrated_nll"],
                "validation_or_test_used_for_fit": False,
            }
    return rows, metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--variant", choices=("merger_frozen", "merger_trainable"), required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--test-only", action="store_true")
    mode.add_argument("--validation-only", action="store_true")
    mode.add_argument("--validation-stage")
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
        validation_stage_name = str(args.validation_stage) if args.validation_stage else None
        requested_split = "test" if args.test_only else "validation"
        predictions_path, existing_path = evaluation_artifact_paths(
            output,
            split=requested_split,
            validation_stage_name=validation_stage_name,
        )
        existing = broadcast_rank0(
            rank,
            lambda: json.loads(existing_path.read_text()) if existing_path.is_file() else None,
        )
        if existing is not None:
            if existing.get("config_sha256") != config_sha or existing.get("variant") != args.variant:
                raise RuntimeError("existing result belongs to another contract")
            if (
                "validation_checkpoint_stage" in existing
                and existing.get("validation_checkpoint_stage")
                != (
                    validation_stage_name
                    or (
                        str(data["curriculum"][-1]["name"])
                        if not args.test_only
                        else None
                    )
                )
            ):
                raise RuntimeError("existing result belongs to another validation checkpoint")
            return 0 if existing.get("automatic_gate_pass") is True else 2
        preflight = broadcast_rank0(rank, lambda: run_preflight(args.config, full_hashes=False))
        runtime_evidence = validate_runtime(config, local_rank)
        os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        seed_everything(int(training["seed"]))
        all_records = read_jsonl(workspace_path(data["manifest"]))
        model, processor, scope = load_model(config, variant=args.variant, device=device)
        stages = data["curriculum"]
        assert isinstance(stages, list)
        selected_stage = str(stages[-1]["name"])
        stages_to_verify = stages
        if validation_stage_name is not None:
            candidates = data.get("validation_candidate_stages")
            if (
                not isinstance(candidates, list)
                or validation_stage_name not in candidates
            ):
                raise ValueError("validation stage is not a frozen candidate")
            candidate_index = next(
                index
                for index, stage in enumerate(stages)
                if str(stage["name"]) == validation_stage_name
            )
            selected_stage = validation_stage_name
            stages_to_verify = stages[: candidate_index + 1]
        final_checkpoint = checkpoint_directory(output, selected_stage)
        if args.test_only:
            selection_path = workspace_path(runtime["output_root"]) / "checkpoint_selection.json"
            selection = broadcast_rank0(rank, lambda: json.loads(selection_path.read_text()))
            if selection.get("selected_variant") != args.variant:
                raise RuntimeError("test-only variant is not the frozen validation winner")
            selected_adapter = workspace_path(selection["selected_adapter"])
            selected_adapter.relative_to(output.resolve())
            if sha256(selected_adapter) != selection.get("selected_adapter_sha256"):
                raise RuntimeError("selected validation adapter changed before fixed test")
            load_qwen_adapter(model, selected_adapter)
            final_checkpoint = selected_adapter.parent
            selected_stage = str(selection.get("selected_stage"))
            training_result = {"gradient_audit_pass": True, "fixed_adapter": True}
            split = "test"
        elif args.validation_only or validation_stage_name is not None:
            checkpoint_evidence = validate_completed_stage_checkpoints(
                output=output,
                stages=stages_to_verify,
                config_sha=config_sha,
                variant=args.variant,
                world_size=world,
                accumulation_steps=accumulation_steps_for_world(dict(config), world),
                global_batch_size=int(training["global_batch_size"]),
                initial_global_step=int(training.get("initial_optimizer_steps", 0)),
                allow_world_size_change=bool(
                    training.get("allow_world_size_change_on_resume", False)
                ),
            )
            expected_checkpoint_step = int(training.get("initial_optimizer_steps", 0)) + sum(
                int(stage["records"])
                * int(stage.get("epochs", training["epochs_per_curriculum_stage"]))
                // int(training["global_batch_size"])
                for stage in stages_to_verify
            )
            if int(checkpoint_evidence[-1]["global_step"]) != expected_checkpoint_step:
                raise RuntimeError("validation-only final optimizer step changed")
            load_qwen_adapter(model, final_checkpoint / "adapter.safetensors")
            training_result = {
                "gradient_audit_pass": True,
                "fixed_adapter": True,
                "validation_only_recovery": True,
                "global_step": expected_checkpoint_step,
                "completed_stage_checkpoints": checkpoint_evidence,
            }
            split = "validation"
        else:
            output.mkdir(parents=True, exist_ok=True)
            initialization_evidence = None
            initialization = config.get("initialization")
            if isinstance(initialization, Mapping):
                adapter = workspace_path(initialization["adapter"])
                if sha256(adapter) != initialization["adapter_sha256"]:
                    raise RuntimeError("initialization adapter hash changed")
                missing_prefixes = initialization.get(
                    "allow_missing_adapter_prefixes", []
                )
                if not isinstance(missing_prefixes, list) or not all(
                    isinstance(prefix, str) and prefix for prefix in missing_prefixes
                ):
                    raise ValueError("initialization missing-prefix contract is invalid")
                optimizer_path_value = initialization.get("optimizer")
                if optimizer_path_value is None:
                    load_qwen_adapter(
                        model,
                        adapter,
                        allow_missing_prefixes=tuple(missing_prefixes),
                    )
                    optimizer_evidence: object = "fresh"
                else:
                    optimizer_path = workspace_path(optimizer_path_value)
                    if sha256(optimizer_path) != initialization.get("optimizer_sha256"):
                        raise RuntimeError("initialization optimizer changed")
                    optimizer_evidence = {
                        "path": optimizer_path_value,
                        "sha256": initialization["optimizer_sha256"],
                        "initial_optimizer_steps": initialization[
                            "initial_optimizer_steps"
                        ],
                    }
                initialization_evidence = {
                    "adapter": initialization["adapter"],
                    "adapter_sha256": initialization["adapter_sha256"],
                    "allow_missing_adapter_prefixes": missing_prefixes,
                    "validation_result": initialization["validation_result"],
                    "validation_result_sha256": initialization[
                        "validation_result_sha256"
                    ],
                    "optimizer": optimizer_evidence,
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
        evaluation_records = None
        if split == "validation" and data.get("validation_manifest") is not None:
            validation_manifest = workspace_path(data["validation_manifest"])
            if sha256(validation_manifest) != data.get("validation_manifest_sha256"):
                raise RuntimeError("frozen external validation manifest changed")
            evaluation_records = read_jsonl(validation_manifest)
        rows, metrics = evaluate_split(
            model,
            processor,
            all_records,
            split,
            config,
            rank=rank,
            world=world,
            device=device,
            output=output,
            config_sha=config_sha,
            adapter_sha=sha256(final_checkpoint / "adapter.safetensors"),
            evaluation_records=evaluation_records,
        )
        checks = gate_checks(
            metrics,
            config,
            bool(training_result["gradient_audit_pass"]),
            split=split,
        )
        automatic_pass = all(checks.values())
        def publish_evaluation() -> bool:
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
                "validation_checkpoint_stage": (
                    selected_stage if split == "validation" else None
                ),
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
            return automatic_pass

        # Publish on rank zero and broadcast either the result or the exception.
        # This prevents nonzero ranks from hanging until the NCCL watchdog fires
        # when immutable artifact publication fails.
        published_pass = bool(broadcast_rank0(rank, publish_evaluation))
        return 0 if published_pass else 2
    finally:
        if ready:
            dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
