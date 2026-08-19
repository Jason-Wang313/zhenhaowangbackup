#!/usr/bin/env python3
"""Fail-closed CPU preflight for the balanced 8-GPU Qwen-VL retry gate."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import platform
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lowlight_restoration.qwen_distributed import (
    partition_source_groups,
    validate_full_batch_contract,
)
from lowlight_restoration.qwen_supervised import (
    DEGRADATION_QUESTION,
    SYSTEM_PROMPT,
    canonical_target,
    open_degraded_images,
    prepare_supervised_batch,
    record_bundle_sha256,
)
from smoke.qwen_lowlight_supervised_preflight import (
    inspect_checkpoint_parameters,
    load_inventory,
    select_records,
    sha256,
    validate_model_file_sizes,
    validate_selected_files,
    workspace_path,
)

APPROVED_STATUS = (
    "approved_for_advisor_route_supervised_vlm_overfit40_ddp8_retry_gate_only"
)
REQUIRED_SECTIONS = (
    "model",
    "processor",
    "prompt",
    "data",
    "training",
    "distributed",
    "generation",
    "gate",
    "runtime",
)


def load_ddp_config(path: Path) -> tuple[dict[str, object], str, bytes]:
    raw = path.read_bytes()
    config = json.loads(raw)
    if not isinstance(config, dict) or config.get("schema_version") != 1:
        raise ValueError("bad distributed supervised Qwen config schema")
    if config.get("status") != APPROVED_STATUS:
        raise ValueError("config is not approved for the DDP overfit retry gate")
    for section in REQUIRED_SECTIONS:
        if not isinstance(config.get(section), dict):
            raise ValueError(f"missing config section: {section}")
    return config, hashlib.sha256(raw).hexdigest(), raw


def _require_exact_parameter_contract(config: dict[str, object]) -> None:
    model = config["model"]
    assert isinstance(model, dict)
    if model.get("architecture") != "Qwen2_5_VLForConditionalGeneration":
        raise ValueError("unexpected Qwen architecture")
    if model.get("trainable_module") != "model.visual":
        raise ValueError("the complete visual path must be trainable")
    if model.get("frozen_modules") != ["model.language_model", "lm_head"]:
        raise ValueError("decoder/output freeze scope changed")
    groups = model.get("expected_parameter_groups")
    if not isinstance(groups, dict) or any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in groups.values()
    ):
        raise ValueError("bad expected Qwen parameter groups")
    visual = sum(
        int(groups.get(name, 0))
        for name in ("visual.patch_embed", "visual.blocks", "visual.merger")
    )
    frozen = sum(
        int(groups.get(name, 0)) for name in ("language_model", "lm_head")
    )
    if visual != model.get("expected_trainable_parameters"):
        raise ValueError("visual parameter-count contract changed")
    if frozen != model.get("expected_frozen_parameters"):
        raise ValueError("frozen parameter-count contract changed")
    if visual + frozen != model.get("expected_total_parameters"):
        raise ValueError("total parameter-count contract changed")


def validate_contract(config: dict[str, object]) -> None:
    model = config["model"]
    processor = config["processor"]
    prompt = config["prompt"]
    data = config["data"]
    training = config["training"]
    distributed = config["distributed"]
    generation = config["generation"]
    gate = config["gate"]
    runtime = config["runtime"]
    assert all(
        isinstance(section, dict)
        for section in (
            model,
            processor,
            prompt,
            data,
            training,
            distributed,
            generation,
            gate,
            runtime,
        )
    )
    _require_exact_parameter_contract(config)
    if model.get("dtype") != "bfloat16" or model.get(
        "attention_implementation"
    ) != "sdpa":
        raise ValueError("model dtype or attention implementation changed")
    if processor.get("use_fast") is not False or processor.get(
        "padding_side"
    ) != "right":
        raise ValueError("processor path or padding side changed")
    if int(processor.get("min_pixels", 0)) != int(
        processor.get("max_pixels", -1)
    ):
        raise ValueError("retry requires one fixed visual-token budget")

    prompt_hashes = {
        "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
        "degradation_question_sha256": hashlib.sha256(
            DEGRADATION_QUESTION.encode()
        ).hexdigest(),
    }
    for key, expected in prompt_hashes.items():
        if prompt.get(key) != expected:
            raise RuntimeError(f"frozen prompt changed: {key}")
    if prompt.get("target_schema") != "project_clean_l1_active_lowlight_l2_l5_v1":
        raise ValueError("target schema changed")
    manifest = workspace_path(data.get("manifest"))
    if sha256(manifest) != data.get("manifest_sha256"):
        raise RuntimeError("supervised manifest SHA-256 changed")
    if data.get("split") != "train":
        raise ValueError("overfit gate may use only the train split")
    source_count = int(data.get("source_count", 0))
    sample_count = int(data.get("sample_count", 0))
    if data.get("probe_source_count") != source_count:
        raise ValueError("the retry probe must cover every selected source")
    if data.get("expected_per_level") != source_count:
        raise ValueError("the selected levels are not balanced")

    world_size = int(distributed.get("required_world_size", 0))
    micro_batch = int(training.get("micro_batch_size_per_rank", 0))
    accumulation = int(training.get("gradient_accumulation_steps", 0))
    global_batch = int(training.get("global_batch_size", 0))
    validate_full_batch_contract(
        sample_count=sample_count,
        source_count=source_count,
        world_size=world_size,
        micro_batch_size=micro_batch,
        gradient_accumulation_steps=accumulation,
        global_batch_size=global_batch,
    )
    if training.get("samples_per_optimizer_step") != sample_count:
        raise ValueError("optimizer-step sample meaning changed")
    total_steps = int(training.get("total_steps", 0))
    if total_steps <= 2 or training.get("total_sample_presentations") != (
        total_steps * sample_count
    ):
        raise ValueError("total sample-presentation contract changed")
    if training.get("learning_rate_schedule") != "linear_warmup_cosine_decay":
        raise ValueError("retry learning-rate schedule changed")
    if not 0 <= int(training.get("warmup_steps", -1)) < total_steps:
        raise ValueError("bad warmup step count")
    base_lr = float(training.get("learning_rate", -1.0))
    minimum_lr = float(training.get("minimum_learning_rate", -1.0))
    if not 0.0 <= minimum_lr <= base_lr:
        raise ValueError("bad learning-rate bounds")
    if training.get("checkpoint_steps") != [
        gate.get("required_resume_step"),
        total_steps,
    ]:
        raise ValueError("checkpoint/resume demonstration contract changed")
    if training.get("gradient_checkpointing") is not True:
        raise ValueError("gradient checkpointing must remain enabled")

    expected_distributed = {
        "backend": "nccl",
        "required_world_size": 8,
        "full_selected_batch_per_optimizer_step": True,
        "keep_source_levels_in_one_microbatch": True,
        "broadcast_buffers": False,
        "ddp_init_sync": False,
        "gradient_as_bucket_view": True,
        "find_unused_parameters": False,
        "timeout_minutes": 30,
    }
    if distributed != expected_distributed:
        raise ValueError("distributed execution contract changed")
    if runtime.get("required_visible_device_count") != world_size:
        raise ValueError("visible-device and DDP world sizes differ")
    if runtime.get("offline") is not True or runtime.get("device") != "cuda":
        raise ValueError("runtime must be offline CUDA")
    if runtime.get("deterministic_algorithms") is not True or runtime.get(
        "sdpa_backend"
    ) != "math":
        raise ValueError("deterministic math-SDPA contract changed")
    workspace_path(runtime.get("output_dir"))
    if generation.get("do_sample") is not False:
        raise ValueError("generation must be deterministic")
    for requirement in (
        "require_gradient_for_every_trainable_parameter",
        "require_no_frozen_gradients",
        "require_no_frozen_updates",
    ):
        if gate.get(requirement) is not True:
            raise ValueError(f"mandatory gate requirement disabled: {requirement}")


def run_preflight(
    config_path: Path, *, skip_large_model_hashes: bool
) -> dict[str, object]:
    config, config_sha, _ = load_ddp_config(config_path)
    validate_contract(config)
    inventory, inventory_sha = load_inventory(config)
    inventory_model = inventory["model"]
    processor_config = config["processor"]
    runtime = config["runtime"]
    distributed = config["distributed"]
    assert isinstance(inventory_model, dict)
    assert isinstance(processor_config, dict)
    assert isinstance(runtime, dict) and isinstance(distributed, dict)

    expected_hashes = validate_model_file_sizes(inventory)
    if not skip_large_model_hashes:
        model_dir = workspace_path(inventory_model["local_path"])
        for name, expected in expected_hashes.items():
            if sha256(model_dir / name) != expected:
                raise RuntimeError(f"model file SHA-256 changed: {name}")
    parameter_groups, tensor_count = inspect_checkpoint_parameters(config, inventory)
    selected, probe = select_records(config)
    validate_selected_files(selected)
    world_size = int(distributed["required_world_size"])
    assignments = {
        str(rank): [
            group[0].source_id
            for group in partition_source_groups(
                selected, world_size=world_size, rank=rank
            )
        ]
        for rank in range(world_size)
    }
    assigned = [source for values in assignments.values() for source in values]
    if len(assigned) != len(set(assigned)) or set(assigned) != {
        record.source_id for record in selected
    }:
        raise RuntimeError("distributed source partition is not exact")

    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    import torch
    import transformers
    from transformers import (
        AutoConfig,
        AutoProcessor,
        Qwen2_5_VLForConditionalGeneration,
    )
    from transformers.models.qwen2_5_vl import modeling_qwen2_5_vl

    model_dir = workspace_path(inventory_model["local_path"])
    loaded_config = AutoConfig.from_pretrained(model_dir, local_files_only=True)
    if loaded_config.model_type != "qwen2_5_vl" or loaded_config.architectures != [
        "Qwen2_5_VLForConditionalGeneration"
    ]:
        raise RuntimeError("installed Transformers Qwen architecture changed")
    if float(getattr(loaded_config, "attention_dropout", -1.0)) != 0.0:
        raise RuntimeError("language-model attention dropout is not zero")
    if Qwen2_5_VLForConditionalGeneration.__name__ != config["model"][
        "architecture"
    ]:
        raise RuntimeError("installed Qwen conditional-generation class changed")
    processor = AutoProcessor.from_pretrained(
        model_dir,
        local_files_only=True,
        use_fast=False,
        min_pixels=int(processor_config["min_pixels"]),
        max_pixels=int(processor_config["max_pixels"]),
    )
    processor.tokenizer.padding_side = str(processor_config["padding_side"])
    if type(processor.image_processor).__name__ != "Qwen2VLImageProcessor":
        raise RuntimeError("the frozen slow Qwen image processor was not loaded")

    representatives = [probe[level - 1] for level in range(1, 6)]
    images = open_degraded_images(representatives)
    batch, target_counts = prepare_supervised_batch(processor, representatives, images)
    visual_tokens = [
        int(grid.prod().item()) // int(processor.image_processor.merge_size) ** 2
        for grid in batch["image_grid_thw"]
    ]
    if not all(
        int(processor_config["minimum_visual_tokens"])
        <= count
        <= int(processor_config["maximum_visual_tokens"])
        for count in visual_tokens
    ):
        raise RuntimeError(f"visual token budget changed: {visual_tokens}")
    for row, record in enumerate(representatives):
        supervised = batch["labels"][row]
        decoded = processor.tokenizer.decode(
            supervised[supervised != -100],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()
        if decoded != canonical_target(record.level_index):
            raise RuntimeError(
                f"teacher-forced target changed for {record.pair_id}: {decoded!r}"
            )

    expected_versions = {
        "python": runtime["required_python"],
        "pytorch": runtime["required_pytorch"],
        "transformers": runtime["required_transformers"],
    }
    actual_versions = {
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "transformers": transformers.__version__,
    }
    if actual_versions != expected_versions:
        raise RuntimeError(
            f"validated environment changed expected={expected_versions} "
            f"actual={actual_versions}"
        )
    level_counts = Counter(record.level_index for record in selected)
    return {
        "schema_version": 1,
        "config_sha256": config_sha,
        "inventory_config_sha256": inventory_sha,
        "model_revision": inventory_model["revision"],
        "model_files": len(expected_hashes),
        "large_model_hashes_checked": not skip_large_model_hashes,
        "parameter_groups": parameter_groups,
        "tensor_count": tensor_count,
        "selected_sources": len({record.source_id for record in selected}),
        "selected_records": len(selected),
        "selected_per_level": dict(sorted(level_counts.items())),
        "selection_bundle_sha256": record_bundle_sha256(selected),
        "probe_records": len(probe),
        "world_size": world_size,
        "rank_source_assignments": assignments,
        "global_batch_size": config["training"]["global_batch_size"],
        "visual_tokens": visual_tokens,
        "assistant_target_tokens": target_counts,
        "processor": type(processor).__name__,
        "image_processor": type(processor.image_processor).__name__,
        "tokenizer": type(processor.tokenizer).__name__,
        "transformers_modeling_source": str(
            inspect.getsourcefile(modeling_qwen2_5_vl)
        ),
        "versions": actual_versions,
        "cuda_visible_in_preflight": torch.cuda.is_available(),
        "test_split_used": False,
        "official_foundir_used": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--skip-large-model-hashes", action="store_true")
    args = parser.parse_args()
    result = run_preflight(
        args.config, skip_large_model_hashes=args.skip_large_model_hashes
    )
    print("QWEN_SUPERVISED_DDP8_PREFLIGHT_PASS " + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
