#!/usr/bin/env python3
"""CPU-side offline preflight for the frozen Qwen-VL Lowlight conditioner."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from PIL import Image
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lowlight_restoration.data import load_pair_manifest
from lowlight_restoration.qwen_condition import SYSTEM_PROMPT, USER_PROMPT
from predict_qwen_lowlight import (
    load_config,
    select_records,
    validate_model_files,
    validate_prompt_config,
    workspace_path,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config, config_sha = load_config(args.config)
    model_config = config["model"]
    processor_config = config["processor"]
    data_config = config["data"]
    assert isinstance(model_config, dict)
    assert isinstance(processor_config, dict)
    assert isinstance(data_config, dict)

    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    if processor_config.get("use_fast") is not False:
        raise SystemExit("QWEN_PROCESSOR_MUST_USE_FROZEN_SLOW_PATH")
    validate_prompt_config(config)
    hashes = validate_model_files(config)
    local_model = workspace_path(model_config["local_path"])
    index = json.loads((local_model / "model.safetensors.index.json").read_text())
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise SystemExit("BAD_QWEN_WEIGHT_INDEX")
    expected_shards = {name for name in hashes if name.endswith(".safetensors")}
    if set(weight_map.values()) != expected_shards:
        raise SystemExit("BAD_QWEN_WEIGHT_SHARDS")
    tensor_count = 0
    parameter_count = 0
    realized_keys: set[str] = set()
    for shard in sorted(expected_shards):
        expected_keys = {key for key, value in weight_map.items() if value == shard}
        with safe_open(local_model / shard, framework="pt", device="cpu") as handle:
            keys = set(handle.keys())
            if keys != expected_keys:
                raise SystemExit(f"QWEN_SHARD_INDEX_MISMATCH {shard}")
            for key in keys:
                shape = handle.get_slice(key).get_shape()
                elements = 1
                for dimension in shape:
                    elements *= dimension
                parameter_count += elements
            tensor_count += len(keys)
            realized_keys.update(keys)
    if realized_keys != set(weight_map):
        raise SystemExit("BAD_QWEN_TENSOR_COVERAGE")
    expected_parameters = model_config.get("expected_parameters")
    expected_tensors = model_config.get("expected_tensors")
    if parameter_count != expected_parameters or tensor_count != expected_tensors:
        raise SystemExit(
            f"BAD_QWEN_MODEL_SHAPES parameters={parameter_count} tensors={tensor_count}"
        )

    from transformers import AutoConfig, AutoProcessor, Qwen2_5_VLForConditionalGeneration

    loaded_config = AutoConfig.from_pretrained(local_model, local_files_only=True)
    if (
        loaded_config.model_type != "qwen2_5_vl"
        or loaded_config.architectures != ["Qwen2_5_VLForConditionalGeneration"]
        or Qwen2_5_VLForConditionalGeneration.__name__ != model_config.get("architecture")
    ):
        raise SystemExit("BAD_QWEN_TRANSFORMERS_ARCHITECTURE")
    processor = AutoProcessor.from_pretrained(
        local_model,
        local_files_only=True,
        use_fast=bool(processor_config["use_fast"]),
        min_pixels=int(processor_config["min_pixels"]),
        max_pixels=int(processor_config["max_pixels"]),
    )

    records = load_pair_manifest(workspace_path(data_config["manifest"]))
    selected = select_records(
        records,
        split=str(data_config["split"]),
        sample_count=5,
        seed=int(data_config["selection_seed"]),
    )
    visual_tokens: list[int] = []
    for record in selected:
        with Image.open(record.degraded_path) as source:
            image = source.convert("RGB")
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": USER_PROMPT}],
            },
        ]
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[prompt], images=[image], padding=True, return_tensors="pt")
        if "pixel_values" not in inputs or "image_grid_thw" not in inputs:
            raise SystemExit("QWEN_PROCESSOR_MISSING_VISION_INPUT")
        grid = inputs.image_grid_thw[0]
        visual_tokens.append(int(grid.prod().item()) // 4)
    if not all(
        int(processor_config["minimum_visual_tokens"])
        <= count
        <= int(processor_config["maximum_visual_tokens"])
        for count in visual_tokens
    ):
        raise SystemExit(f"BAD_QWEN_VISUAL_TOKEN_COUNT {visual_tokens}")

    print(
        "QWEN_LOWLIGHT_PREFLIGHT_PASS "
        + json.dumps(
            {
                "config_sha256": config_sha,
                "revision": model_config["revision"],
                "files": len(hashes),
                "tensors": tensor_count,
                "parameters": parameter_count,
                "processor_visual_tokens": visual_tokens,
                "levels": [record.level_index for record in selected],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
