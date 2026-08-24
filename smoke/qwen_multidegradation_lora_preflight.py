#!/usr/bin/env python3
"""Fail-closed CPU preflight for the four-H100 Qwen LoRA ablation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lowlight_restoration.multidegradation_data import read_jsonl, sha256
from lowlight_restoration.qwen_multidegradation import (
    DEGRADATION_QUESTION,
    SYSTEM_PROMPT,
    make_label,
    prepare_supervised_batch,
)


def workspace_path(value: object) -> Path:
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise ValueError(f"expected workspace-relative path: {value!r}")
    path = (ROOT / value).resolve()
    path.relative_to(ROOT.resolve())
    return path


def load_config(path: Path) -> tuple[dict[str, object], str]:
    raw = path.read_bytes()
    config = json.loads(raw)
    if config.get("schema_version") != 1 or config.get("status") != (
        "approved_qwen_multidegradation_lora_frozen_merger_ablation"
    ):
        raise ValueError("Qwen multi-degradation LoRA config is not approved")
    return config, hashlib.sha256(raw).hexdigest()


def validate_contract(config: dict[str, object]) -> None:
    for section in (
        "contract", "model", "processor", "prompt", "data", "training",
        "distributed", "generation", "gate", "runtime",
    ):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"missing config section: {section}")
    contract = config["contract"]
    model = config["model"]
    processor = config["processor"]
    prompt = config["prompt"]
    data = config["data"]
    training = config["training"]
    distributed = config["distributed"]
    runtime = config["runtime"]
    assert all(isinstance(item, dict) for item in (contract, model, processor, prompt, data, training, distributed, runtime))
    if sha256(workspace_path(contract["path"])) != contract["sha256"]:
        raise RuntimeError("frozen advisor contract changed")
    if sha256(workspace_path(model["inventory_config"])) != model["inventory_config_sha256"]:
        raise RuntimeError("Qwen inventory changed")
    if sha256(workspace_path(data["manifest"])) != data["manifest_sha256"]:
        raise RuntimeError("pilot manifest changed")
    expected_prompt = {
        "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
        "degradation_question_sha256": hashlib.sha256(DEGRADATION_QUESTION.encode()).hexdigest(),
    }
    for key, expected in expected_prompt.items():
        if prompt.get(key) != expected:
            raise RuntimeError(f"frozen prompt changed: {key}")
    if model.get("variants") != ["merger_frozen", "merger_trainable"]:
        raise ValueError("merger ablation variants changed")
    if int(model.get("expected_visual_depth", 0)) != 32 or int(model.get("expected_lora_modules", 0)) != 160:
        raise ValueError("Qwen visual-LoRA scope changed")
    if int(distributed.get("required_world_size", 0)) != 4:
        raise ValueError("this contract requires four GPUs")
    micro = int(training.get("micro_batch_size_per_rank", 0))
    accumulation = int(training.get("gradient_accumulation_steps", 0))
    global_batch = int(training.get("global_batch_size", 0))
    if micro * accumulation * 4 != global_batch:
        raise ValueError("global batch does not match DDP semantics")
    stages = data.get("curriculum")
    if not isinstance(stages, list) or [stage.get("maximum_composition_size") for stage in stages] != [1, 2, 3]:
        raise ValueError("curriculum is not progressive clean/single/double/triple")
    steps = sum(int(stage["records"]) // global_batch for stage in stages)
    if steps != training.get("total_optimizer_steps"):
        raise ValueError("curriculum step count changed")
    if runtime.get("required_visible_devices") != "0,1,2,3":
        raise ValueError("visible-device contract changed")


def validate_model_inventory(config: dict[str, object], *, full_hashes: bool) -> dict[str, object]:
    model = config["model"]
    assert isinstance(model, dict)
    inventory = json.loads(workspace_path(model["inventory_config"]).read_text())
    inventory_model = inventory["model"]
    model_root = workspace_path(model["local_path"])
    if inventory_model["local_path"] != model["local_path"]:
        raise ValueError("model path differs from inventory")
    for name, expected in inventory_model["expected_files"].items():
        path = model_root / name
        if path.stat().st_size != expected["bytes"]:
            raise RuntimeError(f"Qwen file size changed: {name}")
        if full_hashes and sha256(path) != expected["sha256"]:
            raise RuntimeError(f"Qwen file hash changed: {name}")
    qwen_config = json.loads((model_root / "config.json").read_text())
    if qwen_config["vision_config"]["depth"] != model["expected_visual_depth"]:
        raise RuntimeError("Qwen visual depth changed")
    weight_map = json.loads((model_root / "model.safetensors.index.json").read_text())["weight_map"]
    required_suffixes = ("attn.qkv.weight", "attn.proj.weight", "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight")
    missing = [
        f"visual.blocks.{block}.{suffix}"
        for block in range(32)
        for suffix in required_suffixes
        if f"visual.blocks.{block}.{suffix}" not in weight_map
    ]
    if missing:
        raise RuntimeError(f"Qwen LoRA target weights missing: {missing[:5]}")
    return {"repository": inventory_model["repository"], "revision": inventory_model["revision"], "full_hashes": full_hashes}


def validate_data(config: dict[str, object], processor: object) -> dict[str, object]:
    data = config["data"]
    assert isinstance(data, dict)
    records = read_jsonl(workspace_path(data["manifest"]))
    counts = Counter(record["split"] for record in records)
    if dict(counts) != data["expected_split_records"]:
        raise RuntimeError(f"pilot split counts changed: {counts}")
    source_roles: dict[str, str] = {}
    for record in records:
        previous = source_roles.setdefault(record["source_id"], record["split"])
        if previous != record["split"]:
            raise RuntimeError(f"source leakage: {record['source_id']}")
        output = workspace_path(record["output_path"])
        if not output.is_file():
            raise FileNotFoundError(f"pilot image not generated: {output}")
    representatives = [
        next(record for record in records if record["split"] == "train" and record["composition_size"] == size)
        for size in range(4)
    ]
    images = []
    labels = []
    for record in representatives:
        with Image.open(workspace_path(record["output_path"])) as handle:
            images.append(handle.convert("RGB"))
        labels.append(make_label(record["degradations"], record["levels"]))
    batch, counts = prepare_supervised_batch(processor, labels, images)
    if len(counts) != 4 or any(count <= 0 for count in counts):
        raise RuntimeError("assistant-token supervision is empty")
    return {
        "records": len(records),
        "unique_sources_used": len(source_roles),
        "representative_target_token_counts": counts,
        "pixel_values_shape": list(batch["pixel_values"].shape),
    }


def run_preflight(config_path: Path, *, full_hashes: bool) -> dict[str, object]:
    config, config_sha = load_config(config_path)
    validate_contract(config)
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    from transformers import AutoProcessor

    model = config["model"]
    processor_config = config["processor"]
    assert isinstance(model, dict) and isinstance(processor_config, dict)
    processor = AutoProcessor.from_pretrained(
        workspace_path(model["local_path"]),
        local_files_only=True,
        use_fast=False,
        min_pixels=int(processor_config["min_pixels"]),
        max_pixels=int(processor_config["max_pixels"]),
    )
    processor.tokenizer.padding_side = "right"
    result = {
        "schema_version": 1,
        "status": "pass",
        "config_sha256": config_sha,
        "model": validate_model_inventory(config, full_hashes=full_hashes),
        "data": validate_data(config, processor),
    }
    print("QWEN_MULTIDEGRADATION_LORA_PREFLIGHT_PASS " + json.dumps(result, sort_keys=True), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/qwen_multidegradation_lora_v1.json")
    parser.add_argument("--full-hashes", action="store_true")
    args = parser.parse_args()
    run_preflight(args.config, full_hashes=args.full_hashes)


if __name__ == "__main__":
    main()
