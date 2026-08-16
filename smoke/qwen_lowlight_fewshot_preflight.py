#!/usr/bin/env python3
"""CPU preflight for leakage-safe Lowlight Qwen visual calibration validation."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lowlight_restoration.data import load_pair_manifest
from lowlight_restoration.degradation_assessment import (
    SUPPORTED_DEGRADATIONS,
    TASK_READINESS,
    evaluate_severity,
    parse_protocol,
    predict_scalar_centroid,
    record_bundle_sha256,
    require_task_ready,
    validate_source_isolation,
)
from predict_qwen_lowlight import sha256, validate_model_files, workspace_path
from predict_qwen_lowlight_fewshot import (
    build_messages,
    calibration_records,
    checked_image,
    fit_luminance_baseline,
    load_v2_config,
    manifest_rows,
    prompt_sha256,
    select_unique_source_records,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    config, config_sha, base_config = load_v2_config(args.config)
    protocol_payload = config["assessment_protocol"]
    data_config = config["data"]
    assert isinstance(protocol_payload, dict) and isinstance(data_config, dict)
    protocol = parse_protocol(protocol_payload)
    require_task_ready(protocol.degradation)
    if set(TASK_READINESS) != set(SUPPORTED_DEGRADATIONS):
        raise SystemExit("INCOMPLETE_CROSS_TASK_READINESS_REGISTRY")
    if any(
        TASK_READINESS[name] != "blocked_no_task_calibration"
        for name in ("noise", "blur", "haze", "rain")
    ):
        raise SystemExit("LATER_TASK_WAS_ACCIDENTALLY_AUTHORIZED")

    manifest = workspace_path(data_config["manifest"])
    if sha256(manifest) != data_config["manifest_sha256"]:
        raise SystemExit("BAD_CALIBRATION_MANIFEST_HASH")
    records = load_pair_manifest(manifest)
    source_ids = data_config["calibration_source_ids"]
    if not isinstance(source_ids, list) or any(not isinstance(value, str) for value in source_ids):
        raise SystemExit("BAD_CALIBRATION_SOURCE_IDS")
    exemplars = calibration_records(records, source_ids, protocol.calibration_split)
    if data_config.get("require_unique_validation_sources") is not True:
        raise SystemExit("VALIDATION_MUST_REQUIRE_UNIQUE_SOURCES")
    selected = select_unique_source_records(
        records,
        split=protocol.validation_split,
        sample_count=int(data_config["validation_samples"]),
        seed=int(data_config["validation_selection_seed"]),
    )
    validate_source_isolation(exemplars, selected)
    if Counter(record.level_index for record in exemplars) != Counter({level: 3 for level in range(1, 6)}):
        raise SystemExit("BAD_CALIBRATION_LEVEL_COVERAGE")
    if Counter(record.level_index for record in selected) != Counter({level: 5 for level in range(1, 6)}):
        raise SystemExit("BAD_VALIDATION_LEVEL_COVERAGE")
    if len({record.source_id for record in selected}) != int(
        data_config["expected_validation_sources"]
    ):
        raise SystemExit("BAD_VALIDATION_SOURCE_COVERAGE")
    if any(record.split in protocol.protected_splits for record in [*exemplars, *selected]):
        raise SystemExit("PROTECTED_SPLIT_LEAKAGE")

    rows = manifest_rows(manifest)
    luminance = {row["pair_id"]: float(row["final_luminance"]) for row in rows}
    centroids = fit_luminance_baseline(rows, protocol.calibration_split)
    baseline_records: list[dict[str, object]] = []
    validation_images: list[Image.Image] = []
    for record in selected:
        image = checked_image(record, luminance)
        validation_images.append(image)
        baseline_records.append(
            {
                "true_level": record.level_index,
                "predicted_level": predict_scalar_centroid(
                    float(luminance[record.pair_id]), centroids
                ),
                "valid_structured_output": True,
            }
        )
    baseline_metrics = evaluate_severity(baseline_records)
    if (
        abs(float(baseline_metrics["severity_accuracy"]) - 0.64) > 1e-12
        or abs(float(baseline_metrics["severity_mean_absolute_error"]) - 0.40) > 1e-12
    ):
        raise SystemExit(f"FROZEN_BASELINE_RESULT_CHANGED {baseline_metrics}")

    model_config = base_config["model"]
    processor_config = base_config["processor"]
    assert isinstance(model_config, dict) and isinstance(processor_config, dict)
    hashes = validate_model_files(base_config)
    from transformers import AutoConfig, AutoProcessor

    local_model = workspace_path(model_config["local_path"])
    loaded_model_config = AutoConfig.from_pretrained(local_model, local_files_only=True)
    if (
        loaded_model_config.model_type != "qwen2_5_vl"
        or loaded_model_config.architectures != ["Qwen2_5_VLForConditionalGeneration"]
    ):
        raise SystemExit("BAD_QWEN_TRANSFORMERS_ARCHITECTURE")
    processor = AutoProcessor.from_pretrained(
        local_model,
        local_files_only=True,
        use_fast=bool(processor_config["use_fast"]),
        min_pixels=int(processor_config["min_pixels"]),
        max_pixels=int(processor_config["max_pixels"]),
    )
    exemplar_images = [checked_image(record, luminance) for record in exemplars]
    messages = build_messages(exemplars)
    rendered = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[rendered],
        images=[*exemplar_images, validation_images[0]],
        padding=True,
        return_tensors="pt",
    )
    expected_image_count = int(data_config["expected_calibration_records"]) + 1
    if "pixel_values" not in inputs or "image_grid_thw" not in inputs:
        raise SystemExit("QWEN_PROCESSOR_MISSING_VISION_INPUT")
    if inputs.image_grid_thw.shape[0] != expected_image_count:
        raise SystemExit(
            f"QWEN_PROCESSOR_IMAGE_COUNT_MISMATCH {inputs.image_grid_thw.shape[0]}"
        )
    visual_tokens = [int(grid.prod().item()) // 4 for grid in inputs.image_grid_thw]
    if not all(
        int(processor_config["minimum_visual_tokens"])
        <= count
        <= int(processor_config["maximum_visual_tokens"])
        for count in visual_tokens
    ):
        raise SystemExit(f"BAD_QWEN_VISUAL_TOKEN_COUNT {visual_tokens}")

    print(
        "QWEN_LOWLIGHT_FEWSHOT_PREFLIGHT_PASS "
        + json.dumps(
            {
                "config_sha256": config_sha,
                "model_revision": model_config["revision"],
                "model_files": len(hashes),
                "calibration_records": len(exemplars),
                "validation_records": len(selected),
                "calibration_bundle_sha256": record_bundle_sha256(exemplars),
                "validation_bundle_sha256": record_bundle_sha256(selected),
                "prompt_sha256": prompt_sha256(messages),
                "prompt_tokens": int(inputs.input_ids.shape[1]),
                "visual_tokens_total": sum(visual_tokens),
                "visual_tokens_per_image": visual_tokens,
                "baseline_accuracy": baseline_metrics["severity_accuracy"],
                "baseline_mae": baseline_metrics["severity_mean_absolute_error"],
                "later_tasks": {
                    name: TASK_READINESS[name] for name in ("noise", "blur", "haze", "rain")
                },
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
