#!/usr/bin/env python3
"""Fail-closed CPU preflight for the advisor-route Qwen-VL overfit gate."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import platform
import re
import sys
from collections import Counter
from pathlib import Path

from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lowlight_restoration.data import PairRecord, load_pair_manifest
from lowlight_restoration.qwen_supervised import (
    DEGRADATION_QUESTION,
    SYSTEM_PROMPT,
    canonical_target,
    open_degraded_images,
    prepare_supervised_batch,
    record_bundle_sha256,
    select_complete_sources,
)

SHA256_RE = re.compile(r"[0-9a-f]{64}")
REQUIRED_SECTIONS = (
    "model",
    "processor",
    "prompt",
    "data",
    "training",
    "generation",
    "gate",
    "runtime",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def workspace_path(value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"bad workspace path: {value!r}")
    candidate = Path(value)
    resolved = (
        candidate.resolve() if candidate.is_absolute() else (ROOT / candidate).resolve()
    )
    try:
        resolved.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"path escapes workspace: {value}") from exc
    return resolved


def load_supervised_config(path: Path) -> tuple[dict[str, object], str, bytes]:
    raw = path.read_bytes()
    config = json.loads(raw)
    if not isinstance(config, dict) or config.get("schema_version") != 1:
        raise ValueError("bad supervised Qwen config schema")
    if (
        config.get("status")
        != "approved_for_advisor_route_supervised_vlm_overfit40_gate_only"
    ):
        raise ValueError("config is not approved for the supervised overfit gate")
    for section in REQUIRED_SECTIONS:
        if not isinstance(config.get(section), dict):
            raise ValueError(f"missing config section: {section}")
    digest = hashlib.sha256(raw).hexdigest()
    return config, digest, raw


def load_inventory(config: dict[str, object]) -> tuple[dict[str, object], str]:
    model = config["model"]
    assert isinstance(model, dict)
    path = workspace_path(model.get("inventory_config"))
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != model.get("inventory_config_sha256"):
        raise RuntimeError("Qwen inventory config SHA-256 changed")
    inventory = json.loads(raw)
    if not isinstance(inventory, dict) or not isinstance(inventory.get("model"), dict):
        raise ValueError("bad Qwen model inventory")
    return inventory, digest


def validate_prompt_and_data(config: dict[str, object]) -> None:
    prompt = config["prompt"]
    data = config["data"]
    assert isinstance(prompt, dict) and isinstance(data, dict)
    expected_prompt_hashes = {
        "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
        "degradation_question_sha256": hashlib.sha256(
            DEGRADATION_QUESTION.encode()
        ).hexdigest(),
    }
    for key, expected in expected_prompt_hashes.items():
        if prompt.get(key) != expected:
            raise RuntimeError(f"frozen prompt changed: {key}")
    manifest = workspace_path(data.get("manifest"))
    if sha256(manifest) != data.get("manifest_sha256"):
        raise RuntimeError("supervised manifest SHA-256 changed")


def validate_contract(config: dict[str, object]) -> None:
    model = config["model"]
    processor = config["processor"]
    data = config["data"]
    training = config["training"]
    generation = config["generation"]
    gate = config["gate"]
    runtime = config["runtime"]
    assert all(
        isinstance(section, dict)
        for section in (model, processor, data, training, generation, gate, runtime)
    )
    if model.get("architecture") != "Qwen2_5_VLForConditionalGeneration":
        raise ValueError("unexpected Qwen architecture")
    if model.get("trainable_module") != "model.visual":
        raise ValueError("the full visual path must be trainable")
    if model.get("frozen_modules") != ["model.language_model", "lm_head"]:
        raise ValueError("decoder/output freeze scope changed")
    expected_groups = model.get("expected_parameter_groups")
    if not isinstance(expected_groups, dict) or sum(
        expected_groups.values()
    ) != model.get("expected_total_parameters"):
        raise ValueError("bad parameter-count contract")
    if expected_groups.get("visual.patch_embed", 0) + expected_groups.get(
        "visual.blocks", 0
    ) + expected_groups.get("visual.merger", 0) != model.get(
        "expected_trainable_parameters"
    ):
        raise ValueError("bad visual trainable count")
    if expected_groups.get("language_model", 0) + expected_groups.get(
        "lm_head", 0
    ) != model.get("expected_frozen_parameters"):
        raise ValueError("bad frozen decoder count")
    if (
        processor.get("use_fast") is not False
        or processor.get("padding_side") != "right"
    ):
        raise ValueError("processor path or padding side changed")
    if int(data.get("sample_count", 0)) != int(data.get("source_count", 0)) * 5:
        raise ValueError("overfit sample count must be five levels per source")
    if int(data.get("expected_per_level", 0)) != int(data.get("source_count", 0)):
        raise ValueError("overfit per-level quota changed")
    if (
        training.get("micro_batch_size") != 1
        or training.get("gradient_accumulation_steps") != 1
    ):
        raise ValueError("the bounded smoke requires one image per optimizer step")
    checkpoint_steps = training.get("checkpoint_steps")
    if checkpoint_steps != [
        gate.get("required_resume_step"),
        training.get("total_steps"),
    ]:
        raise ValueError("checkpoint/resume demonstration contract changed")
    if training.get("gradient_checkpointing") is not True:
        raise ValueError("gradient checkpointing must remain enabled")
    for requirement in (
        "require_gradient_for_every_trainable_parameter",
        "require_no_frozen_gradients",
        "require_no_frozen_updates",
    ):
        if gate.get(requirement) is not True:
            raise ValueError(f"mandatory gate requirement disabled: {requirement}")
    if generation.get("do_sample") is not False:
        raise ValueError("generation must be deterministic")
    if runtime.get("offline") is not True or runtime.get("device") != "cuda":
        raise ValueError("runtime must be offline CUDA")
    validate_prompt_and_data(config)


def validate_model_file_sizes(inventory: dict[str, object]) -> dict[str, str]:
    model = inventory["model"]
    assert isinstance(model, dict)
    model_dir = workspace_path(model.get("local_path"))
    expected_files = model.get("expected_files")
    if not isinstance(expected_files, dict) or not expected_files:
        raise ValueError("model inventory has no expected files")
    actual = {path.name for path in model_dir.iterdir() if path.is_file()}
    if actual != set(expected_files):
        raise RuntimeError(
            f"model file set changed missing={sorted(set(expected_files) - actual)} "
            f"extra={sorted(actual - set(expected_files))}"
        )
    realized: dict[str, str] = {}
    for name, entry in expected_files.items():
        if not isinstance(entry, dict):
            raise ValueError(f"bad model inventory entry: {name}")
        path = model_dir / name
        if path.stat().st_size != entry.get("bytes"):
            raise RuntimeError(f"model file size changed: {name}")
        expected_sha = entry.get("sha256")
        if (
            not isinstance(expected_sha, str)
            or SHA256_RE.fullmatch(expected_sha) is None
        ):
            raise ValueError(f"bad expected model SHA-256: {name}")
        realized[name] = expected_sha
    return realized


def inspect_checkpoint_parameters(
    config: dict[str, object], inventory: dict[str, object]
) -> tuple[dict[str, int], int]:
    configured_model = config["model"]
    inventory_model = inventory["model"]
    assert isinstance(configured_model, dict) and isinstance(inventory_model, dict)
    model_dir = workspace_path(inventory_model.get("local_path"))
    index = json.loads((model_dir / "model.safetensors.index.json").read_text())
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or len(weight_map) != inventory_model.get(
        "expected_tensors"
    ):
        raise RuntimeError("bad Qwen SafeTensors index")
    counts: Counter[str] = Counter()
    tensors = 0
    for shard in sorted(set(weight_map.values())):
        expected_keys = {key for key, value in weight_map.items() if value == shard}
        with safe_open(model_dir / shard, framework="pt", device="cpu") as handle:
            if set(handle.keys()) != expected_keys:
                raise RuntimeError(f"SafeTensors index mismatch: {shard}")
            for key in handle.keys():
                shape = handle.get_slice(key).get_shape()
                elements = 1
                for dimension in shape:
                    elements *= dimension
                if key.startswith("visual.patch_embed."):
                    group = "visual.patch_embed"
                elif key.startswith("visual.blocks."):
                    group = "visual.blocks"
                elif key.startswith("visual.merger."):
                    group = "visual.merger"
                elif key.startswith("model."):
                    group = "language_model"
                elif key.startswith("lm_head."):
                    group = "lm_head"
                else:
                    group = "unexpected"
                counts[group] += elements
                tensors += 1
    expected = configured_model.get("expected_parameter_groups")
    if dict(counts) != expected:
        raise RuntimeError(f"Qwen parameter groups changed: {dict(counts)}")
    if sum(counts.values()) != inventory_model.get("expected_parameters"):
        raise RuntimeError("Qwen total parameter count changed")
    return dict(counts), tensors


def select_records(
    config: dict[str, object],
) -> tuple[list[PairRecord], list[PairRecord]]:
    data = config["data"]
    assert isinstance(data, dict)
    records = load_pair_manifest(workspace_path(data.get("manifest")))
    selected = select_complete_sources(
        records,
        split=str(data["split"]),
        source_count=int(data["source_count"]),
        seed=int(data["selection_seed"]),
    )
    levels = Counter(record.level_index for record in selected)
    expected = {level: int(data["expected_per_level"]) for level in range(1, 6)}
    if dict(levels) != expected or len(selected) != int(data["sample_count"]):
        raise RuntimeError(f"selected Lowlight composition changed: {dict(levels)}")
    probe_sources = int(data["probe_source_count"])
    probe = selected[: probe_sources * 5]
    if len({record.source_id for record in probe}) != probe_sources:
        raise RuntimeError("probe records are not complete source groups")
    return selected, probe


def validate_selected_files(records: list[PairRecord]) -> None:
    for record in records:
        if sha256(record.degraded_path) != record.degraded_sha256:
            raise RuntimeError(f"selected image SHA-256 changed: {record.pair_id}")


def run_preflight(
    config_path: Path, *, skip_large_model_hashes: bool
) -> dict[str, object]:
    config, config_sha, _ = load_supervised_config(config_path)
    validate_contract(config)
    inventory, inventory_sha = load_inventory(config)
    inventory_model = inventory["model"]
    processor_config = config["processor"]
    runtime = config["runtime"]
    assert isinstance(inventory_model, dict)
    assert isinstance(processor_config, dict) and isinstance(runtime, dict)
    expected_hashes = validate_model_file_sizes(inventory)
    if not skip_large_model_hashes:
        for name, expected in expected_hashes.items():
            if sha256(workspace_path(inventory_model["local_path"]) / name) != expected:
                raise RuntimeError(f"model file SHA-256 changed: {name}")
    parameter_groups, tensor_count = inspect_checkpoint_parameters(config, inventory)
    selected, probe = select_records(config)
    validate_selected_files(selected)

    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    import torch
    import transformers
    from transformers import (
        AutoConfig,
        AutoProcessor,
        Qwen2_5_VLForConditionalGeneration,
    )
    from transformers.models.qwen2_5_vl import modeling_qwen2_5_vl

    loaded_config = AutoConfig.from_pretrained(
        workspace_path(inventory_model["local_path"]), local_files_only=True
    )
    if loaded_config.model_type != "qwen2_5_vl" or loaded_config.architectures != [
        "Qwen2_5_VLForConditionalGeneration"
    ]:
        raise RuntimeError("installed Transformers Qwen architecture changed")
    if Qwen2_5_VLForConditionalGeneration.__name__ != config["model"]["architecture"]:
        raise RuntimeError("installed Qwen conditional-generation class changed")
    processor = AutoProcessor.from_pretrained(
        workspace_path(inventory_model["local_path"]),
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
        supervised_ids = batch["labels"][row]
        decoded = processor.tokenizer.decode(
            supervised_ids[supervised_ids != -100],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()
        if decoded != canonical_target(record.level_index):
            raise RuntimeError(
                f"teacher-forced target tokenization changed for {record.pair_id}: {decoded!r}"
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
            f"validated environment changed expected={expected_versions} actual={actual_versions}"
        )
    result: dict[str, object] = {
        "config_sha256": config_sha,
        "inventory_config_sha256": inventory_sha,
        "model_revision": inventory_model["revision"],
        "model_files": len(expected_hashes),
        "large_model_hashes_checked": not skip_large_model_hashes,
        "parameter_groups": parameter_groups,
        "tensor_count": tensor_count,
        "selected_sources": len({record.source_id for record in selected}),
        "selected_records": len(selected),
        "selection_bundle_sha256": record_bundle_sha256(selected),
        "probe_records": len(probe),
        "visual_tokens": visual_tokens,
        "assistant_target_tokens": target_counts,
        "processor": type(processor).__name__,
        "image_processor": type(processor.image_processor).__name__,
        "tokenizer": type(processor.tokenizer).__name__,
        "transformers_modeling_source": str(inspect.getsourcefile(modeling_qwen2_5_vl)),
        "versions": actual_versions,
        "cuda_visible_in_preflight": torch.cuda.is_available(),
        "test_split_used": False,
        "official_foundir_used": False,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--skip-large-model-hashes", action="store_true")
    args = parser.parse_args()
    result = run_preflight(
        args.config, skip_large_model_hashes=args.skip_large_model_hashes
    )
    print(
        "QWEN_SUPERVISED_OVERFIT40_PREFLIGHT_PASS " + json.dumps(result, sort_keys=True)
    )


if __name__ == "__main__":
    main()
