#!/usr/bin/env python3
"""Evaluate frozen Qwen-VL with train-only visual calibration on Lowlight validation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import random
import subprocess
from collections import Counter
from pathlib import Path
from typing import Mapping

import numpy as np
from PIL import Image

from lowlight_restoration.data import PairRecord, load_pair_manifest
from lowlight_restoration.degradation_assessment import (
    assessment_gate,
    evaluate_severity,
    fit_scalar_centroids,
    parse_protocol,
    predict_scalar_centroid,
    record_bundle_sha256,
    require_task_ready,
    strict_condition_payload,
    validate_source_isolation,
)
from predict_qwen_lowlight import (
    atomic_write_new,
    load_config as load_base_config,
    sha256,
    validate_model_files,
    workspace_path,
)

ROOT = Path(__file__).resolve().parent

SYSTEM_PROMPT = (
    "You are a calibrated project degradation assessor. Use the labeled visual calibration "
    "sequences in this request to learn this project's ordered low-light levels. Judge only "
    "the project severity scale, not aesthetics or semantic content. Return exactly one compact "
    "JSON object with no markdown or extra text."
)

QUERY_PROMPT = """Classify the final, unlabeled image using the demonstrated project scale.

Output constraints:
- The top-level keys must be exactly "degradations" and "levels".
- "degradations" must be exactly ["low_light"].
- "levels" must be exactly {"low_light": N}, where N is integer 1, 2, 3, 4, or 5.
- Do not add explanations, markdown, confidence, captions, or any other field."""


def load_v2_config(path: Path) -> tuple[dict[str, object], str, dict[str, object]]:
    raw = path.read_bytes()
    config = json.loads(raw)
    if not isinstance(config, dict) or config.get("schema_version") != 2:
        raise ValueError("bad calibrated assessment config")
    for section in (
        "base_model_config",
        "assessment_protocol",
        "data",
        "prompt",
        "generation",
        "runtime",
    ):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"missing calibrated config section: {section}")
    base_reference = config["base_model_config"]
    assert isinstance(base_reference, dict)
    base_path = workspace_path(base_reference.get("path"))
    expected_base_hash = base_reference.get("sha256")
    if sha256(base_path) != expected_base_hash:
        raise RuntimeError("frozen Qwen base config hash mismatch")
    base_config, realized_base_hash = load_base_config(base_path)
    if realized_base_hash != expected_base_hash:
        raise RuntimeError("Qwen base config parser hash mismatch")
    protocol = parse_protocol(config["assessment_protocol"])
    require_task_ready(protocol.degradation)
    if protocol.degradation != "low_light":
        raise RuntimeError("this bounded runner is Lowlight-only")
    generation = config["generation"]
    runtime = config["runtime"]
    assert isinstance(generation, dict) and isinstance(runtime, dict)
    if generation.get("do_sample") is not False:
        raise ValueError("calibrated validation decoding must be deterministic")
    if runtime.get("device") != "cuda" or runtime.get("offline") is not True:
        raise ValueError("calibrated validation must use the frozen offline CUDA path")
    return config, hashlib.sha256(raw).hexdigest(), base_config


def calibration_records(
    records: list[PairRecord], source_ids: list[str], calibration_split: str
) -> list[PairRecord]:
    if not source_ids or len(source_ids) != len(set(source_ids)):
        raise ValueError("calibration_source_ids must be non-empty and unique")
    by_key = {(record.source_id, record.level_index): record for record in records}
    selected: list[PairRecord] = []
    for source_id in source_ids:
        for level in range(1, 6):
            record = by_key.get((source_id, level))
            if record is None:
                raise ValueError(f"missing calibration record: {source_id} L{level}")
            if record.split != calibration_split:
                raise ValueError(
                    f"calibration source {source_id} is in {record.split}, not {calibration_split}"
                )
            selected.append(record)
    return selected


def select_unique_source_records(
    records: list[PairRecord], *, split: str, sample_count: int, seed: int
) -> list[PairRecord]:
    """Select a balanced deterministic set with one record per source."""

    selected_split = [record for record in records if record.split == split]
    if sample_count <= 0 or sample_count > len(selected_split) or sample_count % 5:
        raise ValueError("sample_count must be positive, available, and divisible by five")
    per_level = sample_count // 5
    selected: list[PairRecord] = []
    used_sources: set[str] = set()
    for level in range(1, 6):
        candidates = sorted(
            (record for record in selected_split if record.level_index == level),
            key=lambda record: hashlib.sha256(f"{seed}:{record.pair_id}".encode()).hexdigest(),
        )
        realized = 0
        for record in candidates:
            if record.source_id in used_sources:
                continue
            selected.append(record)
            used_sources.add(record.source_id)
            realized += 1
            if realized == per_level:
                break
        if realized != per_level:
            raise ValueError(f"not enough unique validation sources for level {level}")
    return sorted(selected, key=lambda record: record.pair_id)


def build_messages(exemplars: list[PairRecord]) -> list[dict[str, object]]:
    content: list[dict[str, str]] = [
        {
            "type": "text",
            "text": (
                "The following are labeled project calibration sequences. Each sequence shows "
                "the same scene at levels 1 through 5, so compare exposure loss within and "
                "across scenes. Level 1 is clean; level 5 is the strongest low-light condition."
            ),
        }
    ]
    previous_source: str | None = None
    for record in exemplars:
        if record.source_id != previous_source:
            content.append(
                {
                    "type": "text",
                    "text": f"Calibration sequence {record.source_id} begins.",
                }
            )
            previous_source = record.source_id
        content.extend(
            (
                {
                    "type": "text",
                    "text": f"Labeled example: low_light project level {record.level_index}.",
                },
                {"type": "image"},
            )
        )
    content.extend(
        (
            {"type": "text", "text": "The final unlabeled validation image follows."},
            {"type": "image"},
            {"type": "text", "text": QUERY_PROMPT},
        )
    )
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": content},
    ]


def prompt_sha256(messages: list[dict[str, object]]) -> str:
    encoded = json.dumps(messages, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def image_luminance(image: Image.Image) -> float:
    rgb = np.asarray(image.convert("RGB"), dtype=np.float64)
    return float(np.mean(rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722))


def manifest_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "final_luminance" not in reader.fieldnames:
            raise ValueError("manifest lacks final_luminance for the calibration baseline")
        return list(reader)


def fit_luminance_baseline(
    rows: list[dict[str, str]], calibration_split: str
) -> dict[int, float]:
    values: list[tuple[int, float]] = []
    for row in rows:
        if row["split"] == calibration_split:
            values.append((int(row["project_level"][1:]), float(row["final_luminance"])))
    return fit_scalar_centroids(values)


def checked_image(record: PairRecord, manifest_luminance: Mapping[str, float]) -> Image.Image:
    if sha256(record.degraded_path) != record.degraded_sha256:
        raise RuntimeError(f"degraded image hash mismatch: {record.pair_id}")
    with Image.open(record.degraded_path) as source:
        image = source.convert("RGB")
    realized = image_luminance(image)
    expected = manifest_luminance[record.pair_id]
    if abs(realized - expected) > 0.02:
        raise RuntimeError(
            f"manifest/image luminance mismatch for {record.pair_id}: {expected} vs {realized}"
        )
    return image


def git_revision() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, text=True, capture_output=True
    )
    return result.stdout.strip()


def reusable_record(
    path: Path,
    *,
    pair: PairRecord,
    config_sha: str,
    model_revision: str,
    calibration_bundle_sha: str,
    frozen_prompt_sha: str,
) -> dict[str, object] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    expected = {
        "pair_id": pair.pair_id,
        "true_level": pair.level_index,
        "degraded_sha256": pair.degraded_sha256,
        "config_sha256": config_sha,
        "model_revision": model_revision,
        "calibration_bundle_sha256": calibration_bundle_sha,
        "prompt_sha256": frozen_prompt_sha,
    }
    if not isinstance(payload, dict) or any(payload.get(key) != value for key, value in expected.items()):
        raise RuntimeError(f"existing calibrated prediction differs: {path}")
    if payload.get("valid_structured_output") is True:
        parsed = strict_condition_payload(str(payload.get("raw_output")), "low_light")
        if payload.get("predicted_level") != parsed:
            raise RuntimeError(f"existing calibrated parsed result differs: {path}")
    elif payload.get("predicted_level") is not None:
        raise RuntimeError(f"existing invalid calibrated result differs: {path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    config, config_sha, base_config = load_v2_config(args.config)
    protocol_payload = config["assessment_protocol"]
    data_config = config["data"]
    generation_config = config["generation"]
    runtime_config = config["runtime"]
    assert isinstance(protocol_payload, dict)
    assert isinstance(data_config, dict)
    assert isinstance(generation_config, dict)
    assert isinstance(runtime_config, dict)
    protocol = parse_protocol(protocol_payload)
    required_cublas = runtime_config["required_cublas_workspace_config"]
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != required_cublas:
        raise SystemExit(f"CUBLAS_WORKSPACE_CONFIG_MUST_EQUAL_{required_cublas}")
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")

    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    if not torch.cuda.is_available():
        raise SystemExit("CUDA_NOT_AVAILABLE")
    seed = int(runtime_config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)

    manifest = workspace_path(data_config["manifest"])
    if sha256(manifest) != data_config.get("manifest_sha256"):
        raise RuntimeError("calibration manifest hash mismatch")
    records = load_pair_manifest(manifest)
    source_ids = data_config["calibration_source_ids"]
    if not isinstance(source_ids, list) or any(not isinstance(x, str) for x in source_ids):
        raise ValueError("bad calibration_source_ids")
    exemplars = calibration_records(records, source_ids, protocol.calibration_split)
    if len(exemplars) != int(data_config["expected_calibration_records"]):
        raise RuntimeError("unexpected calibration record count")
    if data_config.get("require_unique_validation_sources") is not True:
        raise RuntimeError("validation must require unique sources")
    selected = select_unique_source_records(
        records,
        split=protocol.validation_split,
        sample_count=int(data_config["validation_samples"]),
        seed=int(data_config["validation_selection_seed"]),
    )
    if Counter(record.level_index for record in selected) != Counter(
        {level: int(data_config["validation_samples_per_level"]) for level in range(1, 6)}
    ):
        raise RuntimeError("validation selection is not balanced across all five levels")
    if len({record.source_id for record in selected}) != int(
        data_config["expected_validation_sources"]
    ):
        raise RuntimeError("validation selection does not have the required unique sources")
    validate_source_isolation(exemplars, selected)
    calibration_bundle_sha = record_bundle_sha256(exemplars)
    validation_bundle_sha = record_bundle_sha256(selected)
    messages = build_messages(exemplars)
    frozen_prompt_sha = prompt_sha256(messages)

    rows = manifest_rows(manifest)
    row_luminance = {row["pair_id"]: float(row["final_luminance"]) for row in rows}
    centroids = fit_luminance_baseline(rows, protocol.calibration_split)
    baseline_records: list[dict[str, object]] = []
    for pair in selected:
        image = checked_image(pair, row_luminance)
        predicted = predict_scalar_centroid(image_luminance(image), centroids)
        baseline_records.append(
            {
                "true_level": pair.level_index,
                "predicted_level": predicted,
                "valid_structured_output": True,
            }
        )
    baseline_metrics = evaluate_severity(baseline_records)

    model_config = base_config["model"]
    processor_config = base_config["processor"]
    assert isinstance(model_config, dict) and isinstance(processor_config, dict)
    realized_hashes = validate_model_files(base_config)
    revision = model_config["revision"]
    local_model = workspace_path(model_config["local_path"])
    processor = AutoProcessor.from_pretrained(
        local_model,
        local_files_only=True,
        use_fast=bool(processor_config["use_fast"]),
        min_pixels=int(processor_config["min_pixels"]),
        max_pixels=int(processor_config["max_pixels"]),
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        local_model,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation=str(model_config["attention_implementation"]),
    ).cuda()
    model.eval().requires_grad_(False)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("Qwen-VL must remain frozen during visual calibration")

    exemplar_images = [checked_image(record, row_luminance) for record in exemplars]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    output_dir = workspace_path(runtime_config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions: list[dict[str, object]] = []
    print(
        "QWEN_LOWLIGHT_FEWSHOT_CUDA_READY "
        + json.dumps(
            {
                "gpu": torch.cuda.get_device_name(0),
                "calibration_images": len(exemplar_images),
                "validation_images": len(selected),
                "calibration_bundle_sha256": calibration_bundle_sha,
                "validation_bundle_sha256": validation_bundle_sha,
                "baseline_accuracy": baseline_metrics["severity_accuracy"],
                "baseline_mae": baseline_metrics["severity_mean_absolute_error"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    for index, pair in enumerate(selected):
        record_path = output_dir / "records" / f"{pair.pair_id}.json"
        reused = reusable_record(
            record_path,
            pair=pair,
            config_sha=config_sha,
            model_revision=str(revision),
            calibration_bundle_sha=calibration_bundle_sha,
            frozen_prompt_sha=frozen_prompt_sha,
        )
        if reused is not None:
            predictions.append(reused)
            print(f"QWEN_FEWSHOT_REUSED {index + 1}/25 pair_id={pair.pair_id}", flush=True)
            continue
        query_image = checked_image(pair, row_luminance)
        inputs = processor(
            text=[prompt],
            images=[*exemplar_images, query_image],
            padding=True,
            return_tensors="pt",
        ).to("cuda")
        if inputs.image_grid_thw.shape[0] != len(exemplar_images) + 1:
            raise RuntimeError("processor did not preserve every calibration/query image")
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=int(generation_config["max_new_tokens"]),
                use_cache=True,
            )
        trimmed = generated[:, inputs.input_ids.shape[1] :]
        raw_output = processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()
        parse_error = None
        try:
            predicted_level = strict_condition_payload(raw_output, "low_light")
            valid = True
        except ValueError as exc:
            predicted_level = None
            valid = False
            parse_error = str(exc)
        result = {
            "schema_version": 2,
            "index": index,
            "pair_id": pair.pair_id,
            "source_id": pair.source_id,
            "split": pair.split,
            "degraded_path": str(pair.degraded_path.relative_to(ROOT)),
            "degraded_sha256": pair.degraded_sha256,
            "true_level": pair.level_index,
            "raw_output": raw_output,
            "valid_structured_output": valid,
            "predicted_level": predicted_level,
            "parse_error": parse_error,
            "config_sha256": config_sha,
            "model_revision": revision,
            "calibration_bundle_sha256": calibration_bundle_sha,
            "validation_bundle_sha256": validation_bundle_sha,
            "prompt_sha256": frozen_prompt_sha,
        }
        atomic_write_new(record_path, result)
        predictions.append(result)
        print(
            "QWEN_FEWSHOT_PREDICTION "
            + json.dumps(
                {
                    "progress": f"{index + 1}/{len(selected)}",
                    "pair_id": pair.pair_id,
                    "true_level": pair.level_index,
                    "predicted_level": predicted_level,
                    "valid": valid,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    metrics = evaluate_severity(predictions)
    gate = assessment_gate(metrics, baseline_metrics, protocol)
    summary = {
        "schema_version": 2,
        "status": "passed" if gate["automatic_gate_pass"] else "failed",
        "scope": "lowlight_calibrated_validation_only",
        "test_split_used": False,
        "official_foundir_used": False,
        "config": config,
        "config_sha256": config_sha,
        "git_revision": git_revision(),
        "model_revision": revision,
        "model_file_sha256": realized_hashes,
        "calibration_bundle_sha256": calibration_bundle_sha,
        "validation_bundle_sha256": validation_bundle_sha,
        "prompt_sha256": frozen_prompt_sha,
        "system_prompt": SYSTEM_PROMPT,
        "query_prompt": QUERY_PROMPT,
        "calibration_examples": [
            {
                "pair_id": record.pair_id,
                "source_id": record.source_id,
                "level": record.level_index,
                "degraded_sha256": record.degraded_sha256,
            }
            for record in exemplars
        ],
        "baseline": {
            "name": protocol.baseline_name,
            "calibration_split": protocol.calibration_split,
            "centroids": {str(key): value for key, value in centroids.items()},
            "metrics": baseline_metrics,
        },
        "metrics": metrics,
        "gate": gate,
        "records": predictions,
        "environment": {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
            "max_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
        },
    }
    atomic_write_new(output_dir / "metrics.json", summary)
    marker = "QWEN_LOWLIGHT_CALIBRATED_VALIDATION_PASS" if gate["automatic_gate_pass"] else "QWEN_LOWLIGHT_CALIBRATED_VALIDATION_FAILED"
    print(marker + " " + json.dumps(gate, sort_keys=True), flush=True)
    if not gate["automatic_gate_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
