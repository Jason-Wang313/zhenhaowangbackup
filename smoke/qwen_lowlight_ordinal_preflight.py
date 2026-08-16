#!/usr/bin/env python3
"""CPU preflight for the frozen-Qwen Lowlight ordinal severity head."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lowlight_restoration.data import load_pair_manifest
from lowlight_restoration.degradation_assessment import (
    SUPPORTED_DEGRADATIONS,
    TASK_READINESS,
    parse_protocol,
    record_bundle_sha256,
    validate_source_isolation,
)
from lowlight_restoration.qwen_ordinal import (
    OrdinalSeverityHead,
    select_complete_source_sequences,
)
from predict_qwen_lowlight import sha256, validate_model_files, workspace_path
from predict_qwen_lowlight_fewshot import checked_image, manifest_rows
from train_qwen_lowlight_ordinal_head import (
    flatten,
    load_config,
    scalar_baseline_metrics,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config, config_sha, base_config = load_config(args.config)
    data = config["data"]
    features = config["features"]
    head_config = config["head"]
    assert isinstance(data, dict) and isinstance(features, dict) and isinstance(head_config, dict)
    selection_protocol = parse_protocol(config["head_selection_protocol"])
    if set(TASK_READINESS) != set(SUPPORTED_DEGRADATIONS):
        raise SystemExit("INCOMPLETE_CROSS_TASK_READINESS_REGISTRY")
    if any(
        TASK_READINESS[name] != "blocked_no_task_calibration"
        for name in ("noise", "blur", "haze", "rain")
    ):
        raise SystemExit("LATER_TASK_WAS_ACCIDENTALLY_AUTHORIZED")

    manifest = workspace_path(data["manifest"])
    if sha256(manifest) != data["manifest_sha256"]:
        raise SystemExit("BAD_ORDINAL_MANIFEST_HASH")
    records = load_pair_manifest(manifest)
    sequences = select_complete_source_sequences(
        records,
        split="train",
        source_count=int(data["selected_train_sources"]),
        seed=int(data["source_selection_seed"]),
    )
    train_sequences = sequences[: int(data["head_train_sources"])]
    selection_sequences = sequences[int(data["head_train_sources"]) :]
    train_records = flatten(train_sequences)
    selection_records = flatten(selection_sequences)
    validate_source_isolation(train_records, selection_records)
    if len(train_records) != int(data["head_train_sources"]) * 5:
        raise SystemExit("BAD_ORDINAL_TRAIN_RECORD_COUNT")
    if len(selection_records) != int(data["head_selection_sources"]) * 5:
        raise SystemExit("BAD_ORDINAL_SELECTION_RECORD_COUNT")
    if any(record.split in selection_protocol.protected_splits for record in train_records):
        raise SystemExit("ORDINAL_TRAIN_USED_PROTECTED_SPLIT")

    rows = manifest_rows(manifest)
    luminance = {row["pair_id"]: float(row["final_luminance"]) for row in rows}
    _, selection_baseline = scalar_baseline_metrics(
        train_records, selection_records, luminance
    )
    if float(selection_baseline["valid_structured_output_rate"]) != 1.0:
        raise SystemExit("BAD_SELECTION_BASELINE")

    model_config = base_config["model"]
    processor_config = base_config["processor"]
    assert isinstance(model_config, dict) and isinstance(processor_config, dict)
    hashes = validate_model_files(base_config)
    from transformers import AutoConfig, AutoProcessor

    local_model = workspace_path(model_config["local_path"])
    loaded = AutoConfig.from_pretrained(local_model, local_files_only=True)
    if loaded.model_type != "qwen2_5_vl" or loaded.vision_config.out_hidden_size != int(
        features["embedding_dim"]
    ):
        raise SystemExit("BAD_QWEN_ORDINAL_ARCHITECTURE")
    processor = AutoProcessor.from_pretrained(
        local_model,
        local_files_only=True,
        use_fast=bool(processor_config["use_fast"]),
        min_pixels=int(processor_config["min_pixels"]),
        max_pixels=int(processor_config["max_pixels"]),
    )
    images: list[Image.Image] = [checked_image(record, luminance) for record in train_sequences[0]]
    inputs = processor.image_processor(images=images, return_tensors="pt")
    if "pixel_values" not in inputs or "image_grid_thw" not in inputs:
        raise SystemExit("QWEN_ORDINAL_PROCESSOR_MISSING_VISION_INPUT")
    if inputs.image_grid_thw.shape[0] != int(features["extraction_batch_size"]):
        raise SystemExit("QWEN_ORDINAL_PROCESSOR_BAD_IMAGE_COUNT")
    visual_tokens = [int(grid.prod().item()) // 4 for grid in inputs.image_grid_thw]
    if not all(
        int(processor_config["minimum_visual_tokens"])
        <= value
        <= int(processor_config["maximum_visual_tokens"])
        for value in visual_tokens
    ):
        raise SystemExit("QWEN_ORDINAL_BAD_VISUAL_TOKEN_COUNT")
    head = OrdinalSeverityHead(int(features["feature_dim"]))
    parameters = sum(parameter.numel() for parameter in head.parameters())
    if parameters != int(head_config["expected_trainable_parameters"]):
        raise SystemExit("BAD_ORDINAL_PARAMETER_COUNT")

    print(
        "QWEN_LOWLIGHT_ORDINAL_PREFLIGHT_PASS "
        + json.dumps(
            {
                "config_sha256": config_sha,
                "model_revision": model_config["revision"],
                "model_files": len(hashes),
                "train_sources": len(train_sequences),
                "train_records": len(train_records),
                "selection_sources": len(selection_sequences),
                "selection_records": len(selection_records),
                "train_bundle_sha256": record_bundle_sha256(train_records),
                "selection_bundle_sha256": record_bundle_sha256(selection_records),
                "selection_baseline_accuracy": selection_baseline["severity_accuracy"],
                "selection_baseline_mae": selection_baseline["severity_mean_absolute_error"],
                "feature_dim": int(features["feature_dim"]),
                "head_parameters": parameters,
                "batch_visual_tokens": visual_tokens,
                "later_tasks": {
                    name: TASK_READINESS[name] for name in ("noise", "blur", "haze", "rain")
                },
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
