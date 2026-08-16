#!/usr/bin/env python3
"""Run frozen Qwen-VL Lowlight severity prediction with strict JSON outputs."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from lowlight_restoration.data import PairRecord, load_pair_manifest
from lowlight_restoration.qwen_condition import (
    FALLBACK_LEVEL,
    SYSTEM_PROMPT,
    USER_PROMPT,
    evaluate_predictions,
    parse_qwen_condition,
)

ROOT = Path(__file__).resolve().parent
SHA256_RE = re.compile(r"[0-9a-f]{64}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_new(path: Path, payload: object) -> None:
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    if path.exists():
        if not path.is_file() or path.read_bytes() != encoded:
            raise RuntimeError(f"existing output differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(encoded)


def load_config(path: Path) -> tuple[dict[str, object], str]:
    raw_bytes = path.read_bytes()
    payload = json.loads(raw_bytes)
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("bad Qwen conditioner config")
    for section in ("model", "processor", "prompt", "generation", "data", "runtime"):
        if not isinstance(payload.get(section), dict):
            raise ValueError(f"missing Qwen config section: {section}")
    return payload, hashlib.sha256(raw_bytes).hexdigest()


def workspace_path(value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"bad workspace path: {value!r}")
    path = (ROOT / value).resolve() if not Path(value).is_absolute() else Path(value).resolve()
    try:
        path.relative_to(ROOT)
    except ValueError as exc:
        raise ValueError(f"path escapes workspace: {value}") from exc
    return path


def validate_prompt_config(config: dict[str, object]) -> None:
    prompt = config.get("prompt")
    if not isinstance(prompt, dict):
        raise ValueError("missing Qwen prompt config")
    if (
        prompt.get("system_prompt_sha256") != hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest()
        or prompt.get("user_prompt_sha256") != hashlib.sha256(USER_PROMPT.encode()).hexdigest()
    ):
        raise RuntimeError("Qwen prompt hashes do not match the frozen prompts")


def validate_model_files(config: dict[str, object]) -> dict[str, str]:
    model = config["model"]
    assert isinstance(model, dict)
    local_path = workspace_path(model.get("local_path"))
    expected_files = model.get("expected_files")
    if not local_path.is_dir() or not isinstance(expected_files, dict) or not expected_files:
        raise RuntimeError("Qwen local model or expected file map is unavailable")
    actual_files = {path.name for path in local_path.iterdir() if path.is_file()}
    if actual_files != set(expected_files):
        raise RuntimeError(
            f"Qwen top-level file set mismatch missing={sorted(set(expected_files) - actual_files)} "
            f"extra={sorted(actual_files - set(expected_files))}"
        )
    realized: dict[str, str] = {}
    for name, expected in expected_files.items():
        if not isinstance(name, str) or not isinstance(expected, dict):
            raise ValueError("bad expected Qwen file entry")
        path = local_path / name
        expected_size = expected.get("bytes")
        expected_sha = expected.get("sha256")
        if not path.is_file() or path.stat().st_size != expected_size:
            raise RuntimeError(f"Qwen file size mismatch: {path}")
        if not isinstance(expected_sha, str) or SHA256_RE.fullmatch(expected_sha) is None:
            raise ValueError(f"bad expected SHA-256 for {name}")
        digest = sha256(path)
        if digest != expected_sha:
            raise RuntimeError(f"Qwen file SHA-256 mismatch: {path}")
        realized[name] = digest
    return realized


def select_records(
    records: list[PairRecord], *, split: str, sample_count: int, seed: int
) -> list[PairRecord]:
    selected_split = [record for record in records if record.split == split]
    if sample_count <= 0 or sample_count > len(selected_split) or sample_count % 5:
        raise ValueError("Qwen sample count must be positive, at most the split size, and divisible by 5")
    by_level: dict[int, list[PairRecord]] = defaultdict(list)
    for record in selected_split:
        by_level[record.level_index].append(record)
    per_level = sample_count // 5
    selected: list[PairRecord] = []
    for level in range(1, 6):
        candidates = sorted(
            by_level[level],
            key=lambda record: hashlib.sha256(f"{seed}:{record.pair_id}".encode()).hexdigest(),
        )
        if len(candidates) < per_level:
            raise ValueError(f"not enough level {level} records for stratified selection")
        selected.extend(candidates[:per_level])
    return sorted(selected, key=lambda record: record.pair_id)


def reusable_record(
    path: Path,
    *,
    pair: PairRecord,
    config_sha256: str,
    model_revision: str,
) -> dict[str, object] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    expected = {
        "pair_id": pair.pair_id,
        "true_level": pair.level_index,
        "degraded_sha256": pair.degraded_sha256,
        "config_sha256": config_sha256,
        "model_revision": model_revision,
    }
    if not isinstance(payload, dict) or any(payload.get(key) != value for key, value in expected.items()):
        raise RuntimeError(f"existing Qwen prediction record differs: {path}")
    if payload.get("valid_structured_output") is True:
        parsed = parse_qwen_condition(str(payload.get("raw_output")))
        if payload.get("predicted_level") != parsed.level or payload.get("used_level") != parsed.level:
            raise RuntimeError(f"existing parsed Qwen record differs: {path}")
    elif payload.get("predicted_level") is not None or payload.get("used_level") != FALLBACK_LEVEL:
        raise RuntimeError(f"existing fallback Qwen record differs: {path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    config, config_sha = load_config(args.config)
    model_config = config["model"]
    processor_config = config["processor"]
    prompt_config = config["prompt"]
    generation_config = config["generation"]
    data_config = config["data"]
    runtime_config = config["runtime"]
    assert isinstance(model_config, dict)
    assert isinstance(processor_config, dict)
    assert isinstance(prompt_config, dict)
    assert isinstance(generation_config, dict)
    assert isinstance(data_config, dict)
    assert isinstance(runtime_config, dict)

    if model_config.get("architecture") != "Qwen2_5_VLForConditionalGeneration":
        raise SystemExit("BAD_QWEN_ARCHITECTURE")
    if model_config.get("license") != "apache-2.0":
        raise SystemExit("BAD_QWEN_LICENSE")
    validate_prompt_config(config)
    revision = model_config.get("revision")
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise SystemExit("BAD_QWEN_REVISION")
    realized_hashes = validate_model_files(config)
    print(f"QWEN_MODEL_FILES_PASS files={len(realized_hashes)}", flush=True)

    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA_NOT_AVAILABLE")
    if generation_config.get("do_sample") is not False:
        raise SystemExit("QWEN_DECODING_MUST_BE_DETERMINISTIC")
    if processor_config.get("use_fast") is not False:
        raise SystemExit("QWEN_PROCESSOR_MUST_USE_FROZEN_SLOW_PATH")
    device = torch.device("cuda")
    seed = int(runtime_config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    records = load_pair_manifest(workspace_path(data_config["manifest"]))
    selected = select_records(
        records,
        split=str(data_config["split"]),
        sample_count=int(data_config["num_samples"]),
        seed=int(data_config["selection_seed"]),
    )
    expected_levels = {level: int(data_config["num_samples"]) // 5 for level in range(1, 6)}
    realized_levels = Counter(record.level_index for record in selected)
    if dict(realized_levels) != expected_levels:
        raise SystemExit(f"BAD_QWEN_SAMPLE_LEVELS {dict(realized_levels)}")

    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

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
        torch_dtype=torch.bfloat16,
        attn_implementation=str(model_config["attention_implementation"]),
    ).to(device)
    model.eval().requires_grad_(False)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("Qwen-VL must remain frozen")
    print(
        "QWEN_CUDA_READY "
        + json.dumps(
            {
                "gpu": torch.cuda.get_device_name(0),
                "model_revision": revision,
                "samples": len(selected),
                "max_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    output_dir = workspace_path(runtime_config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    result_records: list[dict[str, object]] = []
    for index, pair in enumerate(selected):
        result_path = output_dir / "records" / f"{pair.pair_id}.json"
        existing = reusable_record(
            result_path,
            pair=pair,
            config_sha256=config_sha,
            model_revision=revision,
        )
        if existing is not None:
            result_records.append(existing)
            print(f"QWEN_REUSED {index + 1}/{len(selected)} pair_id={pair.pair_id}", flush=True)
            continue
        if sha256(pair.degraded_path) != pair.degraded_sha256:
            raise RuntimeError(f"degraded image hash mismatch: {pair.pair_id}")
        with Image.open(pair.degraded_path) as source:
            image = source.convert("RGB")
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": USER_PROMPT},
                ],
            },
        ]
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[prompt], images=[image], padding=True, return_tensors="pt")
        inputs = inputs.to(device)
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=int(generation_config["max_new_tokens"]),
                use_cache=True,
            )
        trimmed = generated[:, inputs.input_ids.shape[1] :]
        raw_output = processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
        parse_error = None
        try:
            parsed = parse_qwen_condition(raw_output)
            predicted_level = parsed.level
            used_level = parsed.level
            valid = True
        except ValueError as exc:
            predicted_level = None
            used_level = FALLBACK_LEVEL
            valid = False
            parse_error = str(exc)
        result = {
            "schema_version": 1,
            "index": index,
            "pair_id": pair.pair_id,
            "split": pair.split,
            "degraded_path": str(pair.degraded_path.relative_to(ROOT)),
            "degraded_sha256": pair.degraded_sha256,
            "true_level": pair.level_index,
            "raw_output": raw_output,
            "valid_structured_output": valid,
            "predicted_level": predicted_level,
            "used_level": used_level,
            "parse_error": parse_error,
            "config_sha256": config_sha,
            "model_revision": revision,
        }
        atomic_write_new(result_path, result)
        result_records.append(result)
        print(
            "QWEN_PREDICTION "
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

    metrics = evaluate_predictions(result_records)
    summary = {
        "schema_version": 1,
        "status": "qwen_lowlight_prediction_complete",
        "config": config,
        "config_sha256": config_sha,
        "model_file_sha256": realized_hashes,
        "system_prompt": SYSTEM_PROMPT,
        "user_prompt": USER_PROMPT,
        "metrics": metrics,
        "records": result_records,
        "environment": {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "max_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
        },
    }
    atomic_write_new(output_dir / "metrics.json", summary)
    print("QWEN_LOWLIGHT_PREDICTION_COMPLETE " + json.dumps(metrics, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
