#!/usr/bin/env python3
"""Fail-closed CPU preflight for the two/four-H100 Qwen LoRA ablation."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import platform
import sys
from collections import Counter
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lowlight_restoration.multidegradation_data import read_jsonl, sha256
from lowlight_restoration.qwen_lora import configure_qwen_visual_lora
from lowlight_restoration.qwen_multidegradation import (
    DEGRADATION_QUESTION,
    SYSTEM_PROMPT,
    make_label,
    prepare_supervised_batch,
)


APPROVED_STATUSES = {
    "approved_qwen_multidegradation_lora_frozen_merger_ablation",
    "approved_qwen_multidegradation_lora_validation_correction",
    "approved_qwen_multidegradation_lora_semantic_loss_correction",
}

SEMANTIC_TARGET_TOKEN_IDS = [
    17, 18, 19, 20, 71, 1335, 2198, 3417, 8899, 9832, 10303, 28503,
    28663, 29093, 33542, 36799, 52218, 68882,
]


def workspace_path(value: object) -> Path:
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise ValueError(f"expected workspace-relative path: {value!r}")
    path = (ROOT / value).resolve()
    path.relative_to(ROOT.resolve())
    return path


def load_config(path: Path) -> tuple[dict[str, object], str]:
    raw = path.read_bytes()
    config = json.loads(raw)
    if (
        config.get("schema_version") != 1
        or config.get("status") not in APPROVED_STATUSES
    ):
        raise ValueError("Qwen multi-degradation LoRA config is not approved")
    return config, hashlib.sha256(raw).hexdigest()


def accumulation_steps_for_world(config: dict[str, object], world: int) -> int:
    training = config["training"]
    assert isinstance(training, dict)
    mapping = training.get("gradient_accumulation_steps_by_world_size")
    if not isinstance(mapping, dict) or str(world) not in mapping:
        raise ValueError(f"no accumulation contract for world size {world}")
    steps = int(mapping[str(world)])
    if steps <= 0:
        raise ValueError("gradient accumulation must be positive")
    return steps


def validate_software_runtime(config: dict[str, object]) -> dict[str, str]:
    import torch
    import transformers

    runtime = config["runtime"]
    assert isinstance(runtime, dict)
    actual = {
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "transformers": transformers.__version__,
    }
    expected = {
        "python": str(runtime["required_python"]),
        "pytorch": str(runtime["required_pytorch"]),
        "transformers": str(runtime["required_transformers"]),
    }
    if actual != expected:
        raise RuntimeError(f"software runtime mismatch: actual={actual} expected={expected}")
    return actual


def validate_contract(config: dict[str, object]) -> None:
    status = config.get("status")
    correction = status != "approved_qwen_multidegradation_lora_frozen_merger_ablation"
    semantic_correction = (
        status == "approved_qwen_multidegradation_lora_semantic_loss_correction"
    )
    for section in (
        "contract", "model", "processor", "prompt", "data", "training",
        "overfit", "distributed", "generation", "gate", "runtime",
    ):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"missing config section: {section}")
    contract = config["contract"]
    model = config["model"]
    processor = config["processor"]
    prompt = config["prompt"]
    data = config["data"]
    training = config["training"]
    overfit = config["overfit"]
    distributed = config["distributed"]
    runtime = config["runtime"]
    assert all(isinstance(item, dict) for item in (contract, model, processor, prompt, data, training, overfit, distributed, runtime))
    if sha256(workspace_path(contract["path"])) != contract["sha256"]:
        raise RuntimeError("frozen advisor contract changed")
    if sha256(workspace_path(model["inventory_config"])) != model["inventory_config_sha256"]:
        raise RuntimeError("Qwen inventory changed")
    if sha256(workspace_path(data["manifest"])) != data["manifest_sha256"]:
        raise RuntimeError("pilot manifest changed")
    generation_result_path = workspace_path(data["generation_result"])
    if sha256(generation_result_path) != data["generation_result_sha256"]:
        raise RuntimeError("pilot generation result changed")
    generation_result = json.loads(generation_result_path.read_text())
    if (
        generation_result.get("status") != "pass"
        or generation_result.get("manifest_sha256") != data["manifest_sha256"]
        or generation_result.get("completed_records") != 12000
    ):
        raise RuntimeError("pilot generation result is incomplete")
    real_manifest_path = workspace_path(data["real_sanity_manifest"])
    if sha256(real_manifest_path) != data["real_sanity_manifest_sha256"]:
        raise RuntimeError("frozen real-sanity manifest changed")
    real_manifest = json.loads(real_manifest_path.read_text())
    if (
        real_manifest.get("status")
        != "frozen_before_qwen_inference_official_sidd_bytes"
        or real_manifest.get("foundir_used") is not False
        or real_manifest.get("validation", {}).get("qwen_inference_before_freeze") is not False
        or real_manifest.get("validation", {}).get("records") != 25
    ):
        raise RuntimeError("real-sanity v3 is not a frozen pre-inference non-FoundIR pool")
    verification_config = data.get("sidd_official_verification")
    if not isinstance(verification_config, dict):
        raise ValueError("missing official SIDD verification contract")
    verification_path = workspace_path(verification_config["path"])
    if sha256(verification_path) != verification_config.get("sha256"):
        raise RuntimeError("official SIDD verification manifest changed")
    verification = json.loads(verification_path.read_text())
    noise_records = {
        str(record["record_id"]): record
        for record in real_manifest["records"]
        if record["expected_primary_type"] == "noise"
    }
    verified = verification.get("records")
    if (
        verification.get("status") != "all_five_exact_official_member_matches"
        or verification.get("foundir_used") is not False
        or verification.get("real_sanity_manifest") != data["real_sanity_manifest"]
        or verification.get("real_sanity_manifest_sha256")
        != data["real_sanity_manifest_sha256"]
        or not isinstance(verified, list)
        or len(verified) != 5
        or len(noise_records) != 5
    ):
        raise RuntimeError("official SIDD verification is incomplete")
    for result in verified:
        record = noise_records.get(str(result.get("record_id")))
        if (
            record is None
            or result.get("exact_match") is not True
            or result.get("local_path") != record["path"]
            or result.get("local_sha256") != record["sha256"]
            or result.get("official_member_sha256") != record["sha256"]
            or result.get("official_archive_url") != record["source_url"]
        ):
            raise RuntimeError("an official SIDD member does not bind to real-sanity v3")
    expected_prompt = {
        "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
        "degradation_question_sha256": hashlib.sha256(DEGRADATION_QUESTION.encode()).hexdigest(),
    }
    for key, expected in expected_prompt.items():
        if prompt.get(key) != expected:
            raise RuntimeError(f"frozen prompt changed: {key}")
    expected_variants = ["merger_trainable"] if correction else [
        "merger_frozen",
        "merger_trainable",
    ]
    if model.get("variants") != expected_variants:
        raise ValueError("merger variants differ from the experiment phase")
    if int(model.get("expected_visual_depth", 0)) != 32 or int(model.get("expected_lora_modules", 0)) != 160:
        raise ValueError("Qwen visual-LoRA scope changed")
    supported_worlds = distributed.get("supported_world_sizes")
    if supported_worlds != [2, 4]:
        raise ValueError("this contract must support two and four H100s")
    micro = int(training.get("micro_batch_size_per_rank", 0))
    global_batch = int(training.get("global_batch_size", 0))
    for world in supported_worlds:
        accumulation = accumulation_steps_for_world(config, world)
        if micro * accumulation * world != global_batch:
            raise ValueError(f"global batch does not match DDP semantics for world {world}")
    stages = data.get("curriculum")
    expected_maxima = [3] if correction else [1, 2, 3]
    if (
        not isinstance(stages, list)
        or [stage.get("maximum_composition_size") for stage in stages]
        != expected_maxima
    ):
        raise ValueError("curriculum differs from the approved experiment phase")
    default_epochs = int(training.get("epochs_per_curriculum_stage", 0))
    if default_epochs <= 0:
        raise ValueError("curriculum epochs must be positive")
    steps = sum(
        int(stage["records"])
        // global_batch
        * int(stage.get("epochs", default_epochs))
        for stage in stages
    )
    if steps != training.get("total_optimizer_steps"):
        raise ValueError("curriculum step count changed")
    if correction:
        expected_epochs = 1 if semantic_correction else 2
        if len(stages) != 1 or int(stages[0].get("epochs", 0)) != expected_epochs:
            raise ValueError(
                f"correction must use {expected_epochs} full-mixture consolidation epoch(s)"
            )
        if int(stages[0].get("records", 0)) != int(
            data["expected_split_records"]["train"]
        ):
            raise ValueError("correction must retain the complete training mixture")
        semantic_loss = training.get("semantic_token_loss")
        if semantic_correction:
            if (
                not isinstance(semantic_loss, dict)
                or semantic_loss.get("enabled") is not True
                or semantic_loss.get("semantic_token_ids") != SEMANTIC_TARGET_TOKEN_IDS
                or float(semantic_loss.get("semantic_weight", 0.0)) != 4.0
            ):
                raise ValueError("semantic correction loss contract changed")
        elif semantic_loss is not None:
            raise ValueError("baseline correction unexpectedly enables semantic loss")
        initialization = config.get("initialization")
        if not isinstance(initialization, dict):
            raise ValueError("correction initialization is missing")
        base_config_path = workspace_path(initialization.get("base_config"))
        if sha256(base_config_path) != initialization.get("base_config_sha256"):
            raise RuntimeError("correction base config changed")
        base_config = json.loads(base_config_path.read_text())
        expected_base_status = (
            "approved_qwen_multidegradation_lora_validation_correction"
            if semantic_correction
            else "approved_qwen_multidegradation_lora_frozen_merger_ablation"
        )
        if base_config.get("status") != expected_base_status:
            raise RuntimeError("correction base phase is inconsistent")
        adapter_path = workspace_path(initialization.get("adapter"))
        if sha256(adapter_path) != initialization.get("adapter_sha256"):
            raise RuntimeError("correction initialization adapter changed")
        result_path = workspace_path(initialization.get("validation_result"))
        if sha256(result_path) != initialization.get("validation_result_sha256"):
            raise RuntimeError("correction source validation result changed")
        result = json.loads(result_path.read_text())
        if (
            result.get("variant") != "merger_trainable"
            or result.get("split") != "validation"
            or result.get("config_sha256")
            != initialization.get("base_config_sha256")
            or result.get("adapter") != initialization.get("adapter")
            or result.get("adapter_sha256") != initialization.get("adapter_sha256")
        ):
            raise RuntimeError("correction initialization evidence is inconsistent")
        overfit_path = workspace_path(initialization.get("overfit_result"))
        if sha256(overfit_path) != initialization.get("overfit_result_sha256"):
            raise RuntimeError("correction source overfit result changed")
        overfit_result = json.loads(overfit_path.read_text())
        overfit_metrics = overfit_result.get("metrics", {})
        if (
            overfit_result.get("status") != "passed"
            or overfit_metrics.get("degradation_set_exact_match_rate") != 1.0
            or overfit_metrics.get("combined_label_exact_match_rate") != 1.0
            or overfit_metrics.get("severity_typed_accuracy") != 1.0
        ):
            raise RuntimeError("correction source did not pass the overfit gate")
        forbidden_fixed_tests = [initialization.get("base_fixed_test_result")]
        source_initialization = base_config.get("initialization")
        if isinstance(source_initialization, dict):
            forbidden_fixed_tests.append(
                source_initialization.get("base_fixed_test_result")
            )
        for fixed_test_path in forbidden_fixed_tests:
            if fixed_test_path is not None and workspace_path(fixed_test_path).exists():
                raise RuntimeError("a fixed test was accessed before correction selection")
    if runtime.get("supported_visible_devices") != {"0,1": 2, "0,1,2,3": 4}:
        raise ValueError("visible-device contract changed")
    if overfit.get("variant") != "merger_trainable":
        raise ValueError("overfit must exercise the larger approved trainable scope")
    if int(overfit.get("records", 0)) != global_batch or int(
        overfit.get("records_per_composition_size", 0)
    ) * 4 != global_batch:
        raise ValueError("overfit batch must cover eight clean/single/double/triple records")
    if int(overfit.get("total_optimizer_steps", 0)) < 100:
        raise ValueError("overfit gate is too short to demonstrate memorization")
    if overfit.get("require_all_twenty_type_level_pairs") is not True:
        raise ValueError("overfit type-level coverage requirement was disabled")


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


def validate_meta_architecture(config: dict[str, object]) -> dict[str, object]:
    import torch
    from transformers import AutoConfig, Qwen2_5_VLForConditionalGeneration

    model_config = config["model"]
    assert isinstance(model_config, dict)
    architecture_config = AutoConfig.from_pretrained(
        workspace_path(model_config["local_path"]), local_files_only=True
    )
    with torch.device("meta"):
        model = Qwen2_5_VLForConditionalGeneration(architecture_config)
    scope = configure_qwen_visual_lora(
        model,
        rank=int(model_config["lora_rank"]),
        alpha=float(model_config["lora_alpha"]),
        dropout=float(model_config["lora_dropout"]),
        merger_trainable=True,
    )
    expected = {
        "matched_lora_module_count": int(model_config["expected_lora_modules"]),
        "merger_trainable_parameters": int(
            model_config["expected_merger_trainable_parameters"]
        ),
        "trainable_parameters": int(
            model_config["expected_trainable_parameters_with_merger"]
        ),
    }
    actual = {key: int(scope[key]) for key in expected}
    lora_parameters = actual["trainable_parameters"] - actual["merger_trainable_parameters"]
    if actual != expected or lora_parameters != int(
        model_config["expected_lora_trainable_parameters"]
    ):
        raise RuntimeError(
            f"installed Transformers Qwen architecture changed: actual={actual} expected={expected}"
        )
    return {**actual, "lora_trainable_parameters": lora_parameters}


def validate_data(
    config: dict[str, object], processor: object, *, full_hashes: bool
) -> dict[str, object]:
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
    if full_hashes:
        def verify_generated(record: dict[str, object]) -> None:
            evidence_path = (
                ROOT
                / "data/qwen_multidegradation_pilot_v1/.generation"
                / str(record["split"])
                / f"{record['record_id']}.json"
            )
            if not evidence_path.is_file():
                raise FileNotFoundError(f"generation evidence missing: {evidence_path}")
            evidence = json.loads(evidence_path.read_text())
            record_hash = hashlib.sha256(
                json.dumps(record, separators=(",", ":"), sort_keys=True).encode()
            ).hexdigest()
            output = workspace_path(record["output_path"])
            if (
                evidence.get("record_id") != record["record_id"]
                or evidence.get("manifest_record_sha256") != record_hash
                or sha256(output) != evidence.get("output_sha256")
            ):
                raise RuntimeError(f"generated image evidence changed: {record['record_id']}")

        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
            list(executor.map(verify_generated, records))
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
        "all_generated_image_hashes_verified": full_hashes,
    }


def validate_real_sanity_files(config: dict[str, object]) -> dict[str, object]:
    data = config["data"]
    assert isinstance(data, dict)
    manifest = json.loads(workspace_path(data["real_sanity_manifest"]).read_text())
    records = manifest["records"]
    if len(records) != 25:
        raise RuntimeError("real-sanity record count changed")
    for record in records:
        path = workspace_path(record["path"])
        if sha256(path) != record["sha256"]:
            raise RuntimeError(f"real-sanity image hash changed: {path}")
    return {
        "records": len(records),
        "type_counts": dict(Counter(record["expected_primary_type"] for record in records)),
        "all_image_hashes_verified": True,
        "all_sidd_noise_bytes_officially_verified": True,
    }


def run_preflight(config_path: Path, *, full_hashes: bool) -> dict[str, object]:
    config, config_sha = load_config(config_path)
    validate_contract(config)
    software = validate_software_runtime(config)
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
        "software": software,
        "model": validate_model_inventory(config, full_hashes=full_hashes),
        "meta_architecture": (
            validate_meta_architecture(config) if full_hashes else {"performed": False}
        ),
        "data": validate_data(config, processor, full_hashes=full_hashes),
        "real_sanity": validate_real_sanity_files(config),
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
