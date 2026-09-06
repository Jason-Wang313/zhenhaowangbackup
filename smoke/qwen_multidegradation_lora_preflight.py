#!/usr/bin/env python3
"""Fail-closed CPU preflight for supported H100 Qwen LoRA experiments."""

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
    DEGRADATIONS,
    DEGRADATION_QUESTION,
    FIXED_FIVE_SLOT_QUESTION,
    FIXED_FIVE_SLOT_SCHEMA,
    PER_TYPE_DIALOGUE_SCHEMA,
    PER_TYPE_QUESTION_TEMPLATE,
    PER_TYPE_RESPONSE_TEMPLATE,
    PER_TYPE_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    VARIABLE_ACTIVE_SCHEMA,
    make_label,
    prepare_per_type_supervised_batch,
    prepare_supervised_batch,
)
from lowlight_restoration.qwen_type_calibration import (
    validate_training_source_partition,
)


APPROVED_STATUSES = {
    "approved_qwen_multidegradation_lora_frozen_merger_ablation",
    "approved_qwen_multidegradation_lora_validation_correction",
    "approved_qwen_multidegradation_lora_semantic_loss_correction",
    "approved_qwen_multidegradation_lora_crossmodal_decoder_correction",
    "approved_qwen_multidegradation_lora_fixed_slot_schema_correction",
    "approved_qwen_multidegradation_per_type_dialogue_rank32_retraining",
    "approved_qwen_multidegradation_per_type_dialogue_rank32_continuation",
    "approved_qwen_visibility_v2_per_type_rank32_decision_calibrated",
    "approved_qwen_visibility_v2b_hard_triple_continuation",
    "advisor_approved_qwen_100k_rank64_joint",
    "advisor_approved_qwen_100k_rank64_per_type",
    "advisor_directed_qwen_100k_rank64_joint_mild_v7",
    "advisor_directed_qwen_100k_rank64_per_type_mild_v7",
    "advisor_directed_qwen_100k_rank64_per_type_physical_v8",
}

RANK64_100K_STATUSES = {
    "advisor_approved_qwen_100k_rank64_joint",
    "advisor_approved_qwen_100k_rank64_per_type",
    "advisor_directed_qwen_100k_rank64_joint_mild_v7",
    "advisor_directed_qwen_100k_rank64_per_type_mild_v7",
    "advisor_directed_qwen_100k_rank64_per_type_physical_v8",
}
RANK64_100K_PER_TYPE_STATUSES = {
    "advisor_approved_qwen_100k_rank64_per_type",
    "advisor_directed_qwen_100k_rank64_per_type_mild_v7",
    "advisor_directed_qwen_100k_rank64_per_type_physical_v8",
}

SEMANTIC_TARGET_TOKEN_IDS = [
    17, 18, 19, 20, 71, 1335, 2198, 3417, 8899, 9832, 10303, 28503,
    28663, 29093, 33542, 36799, 52218, 68882,
]
FIXED_SLOT_SEMANTIC_TOKEN_IDS = [16, 17, 18, 19, 20]


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


def micro_batch_size_for_world(config: dict[str, object], world: int) -> int:
    training = config["training"]
    assert isinstance(training, dict)
    mapping = training.get("micro_batch_size_per_rank_by_world_size")
    if mapping is None:
        micro = int(training.get("micro_batch_size_per_rank", 0))
    else:
        if not isinstance(mapping, dict) or str(world) not in mapping:
            raise ValueError(f"no micro-batch contract for world size {world}")
        micro = int(mapping[str(world)])
    if micro <= 0:
        raise ValueError("micro batch size must be positive")
    return micro


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


def validate_rank64_100k_contract(config: dict[str, object]) -> None:
    """Validate the advisor-requested fair joint-vs-per-type comparison."""

    status = str(config.get("status"))
    per_type = status in RANK64_100K_PER_TYPE_STATUSES
    physical_v8 = status == "advisor_directed_qwen_100k_rank64_per_type_physical_v8"
    if status not in RANK64_100K_STATUSES:
        raise ValueError("unknown rank-64 100k comparison status")
    for section in (
        "contract",
        "data_contract",
        "model",
        "processor",
        "prompt",
        "data",
        "training",
        "overfit",
        "distributed",
        "generation",
        "gate",
        "runtime",
    ):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"missing rank-64 100k section: {section}")
    contract = config["contract"]
    data_contract = config["data_contract"]
    model = config["model"]
    prompt = config["prompt"]
    data = config["data"]
    training = config["training"]
    distributed = config["distributed"]
    generation = config["generation"]
    gate = config["gate"]
    runtime = config["runtime"]
    assert all(
        isinstance(item, dict)
        for item in (
            contract,
            data_contract,
            model,
            prompt,
            data,
            training,
            distributed,
            generation,
            gate,
            runtime,
        )
    )
    for section, description in (
        (contract, "advisor contract"),
        (data_contract, "100k data contract"),
    ):
        path = workspace_path(section["path"])
        if sha256(path) != section.get("sha256"):
            raise RuntimeError(f"{description} changed")
    data_contract_payload = json.loads(workspace_path(data_contract["path"]).read_text())
    expected_data_status = (
        "foundir_physics_blur_noise_v8_100k_contract"
        if physical_v8
        else (
            "advisor_directed_mild_blur_noise_v7_100k_contract"
            if status.startswith("advisor_directed_")
            else "human_approved_extreme_blur_v6_100k_contract"
        )
    )
    if data_contract_payload.get("status") != expected_data_status:
        raise RuntimeError("rank-64 config is bound to the wrong synthetic corpus")
    if sha256(workspace_path(model["inventory_config"])) != model.get(
        "inventory_config_sha256"
    ):
        raise RuntimeError("Qwen inventory changed")
    expected_model = {
        "lora_rank": 64,
        "lora_alpha": 64,
        "language_lora_rank": 64,
        "language_lora_alpha": 64,
        "expected_visual_lora_modules": 160,
        "expected_language_lora_modules": 112,
        "expected_lora_modules": 272,
        "expected_lora_trainable_parameters": 44605440,
        "expected_language_lora_trainable_parameters": 40370176,
        "expected_merger_trainable_parameters": 44574464,
        "expected_trainable_parameters_with_merger": 129550080,
    }
    for key, expected in expected_model.items():
        if model.get(key) != expected:
            raise ValueError(f"rank-64 model contract changed: {key}")
    if model.get("variants") != ["merger_trainable"]:
        raise ValueError("rank-64 comparison must train the visual merger")
    expected_prompt = (
        {
            "target_schema": PER_TYPE_DIALOGUE_SCHEMA,
            "system_prompt_sha256": hashlib.sha256(
                PER_TYPE_SYSTEM_PROMPT.encode()
            ).hexdigest(),
            "degradation_question_template_sha256": hashlib.sha256(
                PER_TYPE_QUESTION_TEMPLATE.encode()
            ).hexdigest(),
            "response_template_sha256": hashlib.sha256(
                PER_TYPE_RESPONSE_TEMPLATE.encode()
            ).hexdigest(),
            "query_degradations": list(DEGRADATIONS),
        }
        if per_type
        else {
            "target_schema": FIXED_FIVE_SLOT_SCHEMA,
            "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
            "degradation_question_sha256": hashlib.sha256(
                FIXED_FIVE_SLOT_QUESTION.encode()
            ).hexdigest(),
        }
    )
    for key, expected in expected_prompt.items():
        if prompt.get(key) != expected:
            raise RuntimeError(f"rank-64 prompt contract changed: {key}")
    manifest = workspace_path(data["manifest"])
    generation_result_path = workspace_path(data["generation_result"])
    if (
        sha256(manifest) != data.get("manifest_sha256")
        or sha256(generation_result_path) != data.get("generation_result_sha256")
    ):
        raise RuntimeError("rank-64 100k generated data changed")
    generation_result = json.loads(generation_result_path.read_text())
    if (
        generation_result.get("status") != "pass"
        or generation_result.get("completed_records") != 120000
        or generation_result.get("manifest_sha256") != data["manifest_sha256"]
        or generation_result.get("protected_test_inference_performed") is not False
        or generation_result.get("foundir_used") is not False
    ):
        raise RuntimeError("rank-64 100k generation is incomplete")
    expected_splits = {"train": 100000, "validation": 10000, "test": 10000}
    if data.get("expected_split_records") != expected_splits:
        raise ValueError("rank-64 100k split cardinality changed")
    multiplier = len(DEGRADATIONS) if per_type else 1
    expected_dialogues = {
        split: count * multiplier for split, count in expected_splits.items()
    }
    if data.get("expected_dialogue_split_records") != expected_dialogues:
        raise ValueError("rank-64 dialogue cardinality changed")
    stages = data.get("curriculum")
    if physical_v8:
        expected_names = [f"V8_physical_shard{index}" for index in range(1, 6)]
        if (
            not isinstance(stages, list)
            or len(stages) != 5
            or [stage.get("name") for stage in stages] != expected_names
            or any(stage.get("maximum_composition_size") != 3 for stage in stages)
            or any(int(stage.get("source_records", 0)) != 20000 for stage in stages)
            or any(int(stage.get("records", 0)) != 100000 for stage in stages)
            or any(int(stage.get("epochs", 0)) != 1 for stage in stages)
            or [int(stage.get("source_start", -1)) for stage in stages]
            != [0, 20000, 40000, 60000, 80000]
            or any(
                stage.get("source_order_key")
                != "V8_physical_full_mixture_epoch1"
                for stage in stages
            )
        ):
            raise ValueError(
                "v8 continuation must replay one complete resumable full mixture"
            )
    else:
        expected_source_stages = [40000, 70000, 100000]
        if not isinstance(stages, list) or len(stages) != 3:
            raise ValueError("rank-64 curriculum must have three stages")
        for index, (stage, source_records) in enumerate(
            zip(stages, expected_source_stages, strict=True), start=1
        ):
            if (
                stage.get("maximum_composition_size") != index
                or int(stage.get("source_records", source_records)) != source_records
                or int(stage.get("records", 0)) != source_records * multiplier
                or int(stage.get("epochs", 0)) != 1
            ):
                raise ValueError("rank-64 curriculum coverage changed")
    if distributed.get("supported_world_sizes") != [1, 2, 4, 8]:
        raise ValueError("rank-64 comparison must support 1/2/4/8 GPUs")
    global_batch = int(training.get("global_batch_size", 0))
    if global_batch != 200:
        raise ValueError("rank-64 comparison global batch changed")
    for world in (1, 2, 4, 8):
        if (
            micro_batch_size_for_world(config, world)
            * accumulation_steps_for_world(config, world)
            * world
            != global_batch
        ):
            raise ValueError(f"rank-64 DDP batch mismatch for world {world}")
    expected_steps = sum(
        int(stage["records"]) // global_batch for stage in stages
    )
    if (
        int(training.get("initial_optimizer_steps", -1)) != 0
        or int(training.get("total_optimizer_steps", -1)) != expected_steps
        or int(training.get("epochs_per_curriculum_stage", 0)) != 1
    ):
        raise ValueError("rank-64 optimizer schedule changed")
    semantic = training.get("semantic_token_loss")
    if (
        not isinstance(semantic, dict)
        or semantic.get("enabled") is not True
        or semantic.get("semantic_token_ids") != FIXED_SLOT_SEMANTIC_TOKEN_IDS
        or float(semantic.get("semantic_weight", 0.0)) != 8.0
        or semantic.get("semantic_weight_by_degradation") is not None
    ):
        raise ValueError("rank-64 comparison decision-token loss changed")
    initialization = config.get("initialization")
    if physical_v8:
        if not isinstance(initialization, dict):
            raise ValueError("v8 continuation initialization is missing")
        for path_key, hash_key in (
            ("base_config", "base_config_sha256"),
            ("adapter", "adapter_sha256"),
            ("validation_result", "validation_result_sha256"),
        ):
            path = workspace_path(initialization.get(path_key))
            if sha256(path) != initialization.get(hash_key):
                raise RuntimeError(f"v8 initialization changed: {path_key}")
        if (
            initialization.get("optimizer") is not None
            or initialization.get("allow_missing_adapter_prefixes") != []
            or int(initialization.get("initial_dialogue_presentations", 0)) != 1050000
        ):
            raise ValueError("v8 must initialize the v7 adapter with a fresh optimizer")
        base_config = json.loads(workspace_path(initialization["base_config"]).read_text())
        if (
            base_config.get("status")
            != "advisor_directed_qwen_100k_rank64_per_type_mild_v7"
            or base_config.get("model") != model
            or sha256(workspace_path(initialization["base_config"]))
            != initialization["base_config_sha256"]
        ):
            raise RuntimeError("v8 initialization architecture is not the v7 winner")
        validation = json.loads(
            workspace_path(initialization["validation_result"]).read_text()
        )
        if (
            validation.get("automatic_gate_pass") is not True
            or validation.get("adapter") != initialization["adapter"]
            or validation.get("adapter_sha256") != initialization["adapter_sha256"]
            or validation.get("config_sha256") != initialization["base_config_sha256"]
        ):
            raise RuntimeError("v8 initialization evidence is inconsistent")
        presence = training.get("presence_auxiliary_loss")
        if presence != {
            "enabled": True,
            "weight": 1.0,
            "level_token_ids": FIXED_SLOT_SEMANTIC_TOKEN_IDS,
        }:
            raise ValueError("v8 decoder presence auxiliary loss changed")
    else:
        if initialization is not None:
            raise ValueError("rank-64 fair comparison must start both methods from base Qwen")
        if training.get("presence_auxiliary_loss") is not None:
            raise ValueError("legacy rank-64 comparison unexpectedly enables presence loss")
    if float(gate.get("maximum_severity_typed_mse_with_missing_penalty", -1)) != 1.0:
        raise ValueError("rank-64 comparison requires severity MSE < 1")
    if float(gate.get("minimum_triple_combined_label_exact_match_rate", -1)) != 0.65:
        raise ValueError("rank-64 comparison requires strict triple label exact >= 0.65")
    if int(generation.get("batch_size_per_rank", 0)) <= 0:
        raise ValueError("rank-64 generation batch is invalid")
    if float(runtime.get("minimum_gpu_memory_gib", 0.0)) > 80.0:
        raise ValueError("rank-64 runtime would reject an 80GB H100")


def validate_contract(config: dict[str, object]) -> None:
    status = config.get("status")
    if status in RANK64_100K_STATUSES:
        validate_rank64_100k_contract(config)
        return
    correction = status != "approved_qwen_multidegradation_lora_frozen_merger_ablation"
    semantic_correction = (
        status == "approved_qwen_multidegradation_lora_semantic_loss_correction"
    )
    decoder_correction = (
        status
        == "approved_qwen_multidegradation_lora_crossmodal_decoder_correction"
    )
    fixed_slot_correction = (
        status
        == "approved_qwen_multidegradation_lora_fixed_slot_schema_correction"
    )
    per_type_dialogue = (
        status in {
            "approved_qwen_multidegradation_per_type_dialogue_rank32_retraining",
            "approved_qwen_multidegradation_per_type_dialogue_rank32_continuation",
            "approved_qwen_visibility_v2_per_type_rank32_decision_calibrated",
            "approved_qwen_visibility_v2b_hard_triple_continuation",
        }
    )
    per_type_continuation = (
        status
        == "approved_qwen_multidegradation_per_type_dialogue_rank32_continuation"
    )
    visibility_v2b = (
        status == "approved_qwen_visibility_v2b_hard_triple_continuation"
    )
    visibility_v2 = status in {
        "approved_qwen_visibility_v2_per_type_rank32_decision_calibrated",
        "approved_qwen_visibility_v2b_hard_triple_continuation",
    }
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
    if visibility_v2b:
        for path_key, hash_key in (
            ("validation_manifest", "validation_manifest_sha256"),
            ("validation_metadata", "validation_metadata_sha256"),
            ("validation_generation_result", "validation_generation_result_sha256"),
        ):
            path = workspace_path(data.get(path_key))
            if sha256(path) != data.get(hash_key):
                raise RuntimeError(f"v2b frozen validation input changed: {path_key}")
        validation_metadata = json.loads(
            workspace_path(data["validation_metadata"]).read_text()
        )
        validation_generation = json.loads(
            workspace_path(data["validation_generation_result"]).read_text()
        )
        if (
            validation_metadata.get("status")
            != "frozen_validation_manifest_before_model_correction_or_inference"
            or validation_metadata.get("manifest_sha256")
            != data["validation_manifest_sha256"]
            or validation_metadata.get("model_checkpoint_accessed") is not False
            or validation_metadata.get("protected_test_inference_performed") is not False
            or validation_generation.get("status") != "pass"
            or validation_generation.get("manifest_sha256")
            != data["validation_manifest_sha256"]
            or validation_generation.get("completed_records") != 1200
            or validation_generation.get("model_checkpoint_accessed") is not False
            or validation_generation.get("protected_test_inference_performed") is not False
        ):
            raise RuntimeError("v2b external validation was not frozen before correction")
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
    expected_schema = (
        PER_TYPE_DIALOGUE_SCHEMA
        if per_type_dialogue
        else FIXED_FIVE_SLOT_SCHEMA
        if fixed_slot_correction
        else VARIABLE_ACTIVE_SCHEMA
    )
    if per_type_dialogue:
        expected_prompt = {
            "system_prompt_sha256": hashlib.sha256(
                PER_TYPE_SYSTEM_PROMPT.encode()
            ).hexdigest(),
            "degradation_question_template_sha256": hashlib.sha256(
                PER_TYPE_QUESTION_TEMPLATE.encode()
            ).hexdigest(),
            "response_template_sha256": hashlib.sha256(
                PER_TYPE_RESPONSE_TEMPLATE.encode()
            ).hexdigest(),
        }
        if prompt.get("query_degradations") != list(DEGRADATIONS):
            raise RuntimeError("per-type query order changed")
    else:
        expected_question = (
            FIXED_FIVE_SLOT_QUESTION
            if fixed_slot_correction
            else DEGRADATION_QUESTION
        )
        expected_prompt = {
            "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
            "degradation_question_sha256": hashlib.sha256(expected_question.encode()).hexdigest(),
        }
    for key, expected in expected_prompt.items():
        if prompt.get(key) != expected:
            raise RuntimeError(f"frozen prompt changed: {key}")
    if prompt.get("target_schema") != expected_schema:
        raise RuntimeError("Qwen target schema differs from the experiment phase")
    expected_variants = ["merger_trainable"] if correction else [
        "merger_frozen",
        "merger_trainable",
    ]
    if model.get("variants") != expected_variants:
        raise ValueError("merger variants differ from the experiment phase")
    expected_modules = 272 if (decoder_correction or fixed_slot_correction or per_type_dialogue) else 160
    if (
        int(model.get("expected_visual_depth", 0)) != 32
        or int(model.get("expected_lora_modules", 0)) != expected_modules
    ):
        raise ValueError("Qwen visual-LoRA scope changed")
    if per_type_dialogue:
        if (
            int(model.get("expected_visual_lora_modules", 0)) != 160
            or int(model.get("expected_language_lora_modules", 0)) != 112
            or int(model.get("lora_rank", 0)) != 32
            or float(model.get("lora_alpha", 0.0)) != 64.0
            or int(model.get("language_lora_rank", 0)) != 32
            or float(model.get("language_lora_alpha", 0.0)) != 64.0
        ):
            raise ValueError("advisor-directed rank-32 Qwen LoRA scope changed")
    elif decoder_correction or fixed_slot_correction:
        if (
            int(model.get("expected_visual_lora_modules", 0)) != 160
            or int(model.get("expected_language_lora_modules", 0)) != 112
            or int(model.get("language_lora_rank", 0)) != 8
            or float(model.get("language_lora_alpha", 0.0)) != 16.0
        ):
            raise ValueError("cross-modal decoder LoRA scope changed")
    elif "language_lora_rank" in model or "language_lora_alpha" in model:
        raise ValueError("visual-only phase unexpectedly enables decoder LoRA")
    supported_worlds = distributed.get("supported_world_sizes")
    expected_worlds = [2, 8] if (per_type_continuation or visibility_v2) else [2, 4]
    if supported_worlds != expected_worlds:
        raise ValueError(f"this contract must support H100 world sizes {expected_worlds}")
    micro = int(training.get("micro_batch_size_per_rank", 0))
    global_batch = int(training.get("global_batch_size", 0))
    for world in supported_worlds:
        accumulation = accumulation_steps_for_world(config, world)
        world_micro = micro_batch_size_for_world(config, world)
        if world_micro * accumulation * world != global_batch:
            raise ValueError(f"global batch does not match DDP semantics for world {world}")
    stages = data.get("curriculum")
    expected_maxima = (
        [3, 3, 3]
        if visibility_v2b
        else [3, 3, 3, 3, 3]
        if (per_type_continuation or visibility_v2)
        else [3, 3, 3]
        if per_type_dialogue
        else [3]
        if correction
        else [1, 2, 3]
    )
    if (
        not isinstance(stages, list)
        or [stage.get("maximum_composition_size") for stage in stages]
        != expected_maxima
    ):
        raise ValueError("curriculum differs from the approved experiment phase")
    default_epochs = int(training.get("epochs_per_curriculum_stage", 0))
    if default_epochs <= 0:
        raise ValueError("curriculum epochs must be positive")
    continuation_steps = sum(
        int(stage["records"])
        // global_batch
        * int(stage.get("epochs", default_epochs))
        for stage in stages
    )
    initial_steps = int(training.get("initial_optimizer_steps", 0))
    expected_total_steps = initial_steps + continuation_steps
    if expected_total_steps != training.get("total_optimizer_steps"):
        raise ValueError("curriculum step count changed")
    if per_type_continuation:
        if (
            initial_steps != 1440
            or int(training.get("continuation_optimizer_steps", 0))
            != continuation_steps
            or int(training.get("warmup_steps", -1)) != 0
        ):
            raise ValueError("rank-32 continuation optimizer schedule changed")
    elif visibility_v2b:
        if (
            initial_steps != 1050
            or int(training.get("continuation_optimizer_steps", 0))
            != continuation_steps
            or int(training.get("warmup_steps", -1)) != 0
        ):
            raise ValueError("visibility-v2b continuation optimizer schedule changed")
    elif initial_steps != 0:
        raise ValueError("non-continuation experiment has an optimizer-step offset")
    if per_type_dialogue:
        expected_images = (
            int(data["gradient_records_per_epoch"])
            if visibility_v2
            else int(data["expected_split_records"]["train"])
        )
        expected_dialogues = expected_images * len(DEGRADATIONS)
        expected_stage_count = (
            3 if visibility_v2b else 5 if (per_type_continuation or visibility_v2) else 3
        )
        if len(stages) != expected_stage_count:
            raise ValueError(
                f"per-type phase must preserve {expected_stage_count} full epochs"
            )
        for stage in stages:
            if (
                int(stage.get("epochs", 0)) != 1
                or int(stage.get("source_records", 0)) != expected_images
                or int(stage.get("records", 0)) != expected_dialogues
            ):
                raise ValueError("per-type stage does not cover every image/type pair")
        expected_dialogue_splits = {
            split: (
                expected_dialogues
                if visibility_v2 and split == "train"
                else int(count) * len(DEGRADATIONS)
            )
            for split, count in data["expected_split_records"].items()
        }
        if data.get("expected_dialogue_split_records") != expected_dialogue_splits:
            raise ValueError("per-type dialogue split counts changed")
        semantic_loss = training.get("semantic_token_loss")
        if visibility_v2:
            if (
                not isinstance(semantic_loss, dict)
                or semantic_loss.get("enabled") is not True
                or semantic_loss.get("semantic_token_ids")
                != FIXED_SLOT_SEMANTIC_TOKEN_IDS
                or float(semantic_loss.get("semantic_weight", 0.0)) != 8.0
            ):
                raise ValueError("visibility-v2 decision-token loss contract changed")
            expected_type_weights = (
                {
                    "blur": 12.0,
                    "noise": 8.0,
                    "rain": 12.0,
                    "haze": 12.0,
                    "low_light": 8.0,
                }
                if visibility_v2b
                else None
            )
            if semantic_loss.get("semantic_weight_by_degradation") != expected_type_weights:
                raise ValueError("visibility-v2 per-type decision weights changed")
        elif semantic_loss is not None:
            raise ValueError("advisor-directed per-type dialogue must use plain assistant CE")
        if per_type_continuation:
            validate_per_type_continuation_initialization(config)
            if (
                [stage.get("name") for stage in stages]
                != [
                    "H4_per_type_dialogue_epoch4",
                    "H5_per_type_dialogue_epoch5",
                    "H6_per_type_dialogue_epoch6",
                    "H7_per_type_dialogue_epoch7",
                    "H8_per_type_dialogue_epoch8",
                ]
                or micro != 25
                or global_batch != 200
                or training.get("gradient_accumulation_steps_by_world_size")
                != {"2": 4, "8": 1}
                or float(training.get("lora_learning_rate", 0.0)) != 0.000005
                or float(training.get("merger_learning_rate", 0.0)) != 0.0000005
                or float(training.get("language_lora_learning_rate", 0.0))
                != 0.000002
                or training.get("allow_world_size_change_on_resume") is not True
            ):
                raise ValueError("rank-32 continuation training contract changed")
            if data.get("validation_candidate_stages") != [
                "H5_per_type_dialogue_epoch5",
                "H6_per_type_dialogue_epoch6",
                "H8_per_type_dialogue_epoch8",
            ]:
                raise ValueError("continuation validation checkpoint schedule changed")
            if (
                float(
                    config["gate"].get(
                        "minimum_triple_degradation_set_exact_match_rate", 0.0
                    )
                )
                != 0.8
                or config.get("selection", {}).get("candidate_epochs")
                != [5, 6, 8]
            ):
                raise ValueError("continuation validation-selection contract changed")
        elif visibility_v2:
            if visibility_v2b:
                validate_visibility_v2b_initialization(config)
                expected_names = [
                    "V2B_C1_hard_triple_epoch6",
                    "V2B_C2_hard_triple_epoch7",
                    "V2B_C3_hard_triple_epoch8",
                ]
                expected_sampling = {
                    "schema": "hard_triple_mixture_v1",
                    "composition_source_quotas": {
                        "0": 600,
                        "1": 2200,
                        "2": 2240,
                        "3": 3360,
                    },
                    "triple_combination_source_quotas": {
                        "blur+noise+rain": 480,
                        "blur+noise+haze": 240,
                        "blur+noise+low_light": 480,
                        "blur+rain+haze": 240,
                        "blur+rain+low_light": 240,
                        "blur+haze+low_light": 240,
                        "noise+rain+haze": 240,
                        "noise+rain+low_light": 480,
                        "noise+haze+low_light": 480,
                        "rain+haze+low_light": 240,
                    },
                }
                if (
                    [stage.get("name") for stage in stages] != expected_names
                    or any(stage.get("source_sampling") != expected_sampling for stage in stages)
                    or micro != 25
                    or global_batch != 200
                    or training.get("micro_batch_size_per_rank_by_world_size")
                    != {"2": 100, "8": 25}
                    or training.get("gradient_accumulation_steps_by_world_size")
                    != {"2": 1, "8": 1}
                    or float(training.get("lora_learning_rate", 0.0)) != 0.00001
                    or float(training.get("merger_learning_rate", 0.0)) != 0.000001
                    or float(training.get("language_lora_learning_rate", 0.0))
                    != 0.000004
                    or training.get("allow_world_size_change_on_resume") is not True
                    or float(
                        config["gate"].get(
                            "minimum_triple_degradation_set_exact_match_rate", 0.0
                        )
                    )
                    != 0.83
                    or float(
                        config["gate"].get(
                            "minimum_protected_test_triple_degradation_set_exact_match_rate",
                            0.0,
                        )
                    )
                    != 0.80
                    or config.get("selection", {}).get("checkpoint")
                    != "V2B_C3_hard_triple_epoch8"
                ):
                    raise ValueError("visibility-v2b correction contract changed")
            else:
                if config.get("initialization") is not None:
                    raise ValueError("visibility-v2 retraining must start from the Qwen base")
                expected_names = [
                    f"V2_H{epoch}_per_type_dialogue_epoch{epoch}"
                    for epoch in range(1, 6)
                ]
            partition_config = data.get("calibration_partition")
            decision = config.get("decision_calibration")
            if (
                [stage.get("name") for stage in stages] != expected_names
                or micro != 25
                or global_batch != 200
                or training.get("gradient_accumulation_steps_by_world_size")
                != ({"2": 1, "8": 1} if visibility_v2b else {"2": 4, "8": 1})
                or float(training.get("lora_learning_rate", 0.0))
                != (0.00001 if visibility_v2b else 0.00005)
                or float(training.get("merger_learning_rate", 0.0))
                != (0.000001 if visibility_v2b else 0.000005)
                or float(training.get("language_lora_learning_rate", 0.0))
                != (0.000004 if visibility_v2b else 0.00002)
                or not isinstance(partition_config, dict)
                or not isinstance(decision, dict)
                or decision.get("level_token_ids") != FIXED_SLOT_SEMANTIC_TOKEN_IDS
                or decision.get("fit_split") != "reserved_training_sources_only"
                or decision.get("validation_or_test_used_for_fit") is not False
                or int(decision.get("candidate_batch_size_per_rank", 0)) <= 0
                or config.get("selection", {}).get("checkpoint")
                != expected_names[-1]
                or float(
                    config["gate"].get(
                        "minimum_triple_degradation_set_exact_match_rate", 0.0
                    )
                )
                != (0.83 if visibility_v2b else 0.8)
            ):
                raise ValueError("visibility-v2 rank-32 training contract changed")
        elif config.get("initialization") is not None:
            raise ValueError("rank-32 per-type retraining must start from the frozen Qwen base")
        advisor_image = config.get("advisor_method_image")
        if not isinstance(advisor_image, dict):
            raise ValueError("advisor per-type framework image is not bound")
        advisor_image_path = workspace_path(advisor_image.get("path"))
        if sha256(advisor_image_path) != advisor_image.get("sha256"):
            raise RuntimeError("advisor per-type framework image changed")
    elif correction:
        expected_epochs = (
            2
            if fixed_slot_correction
            else 1
            if (semantic_correction or decoder_correction)
            else 2
        )
        if len(stages) != 1 or int(stages[0].get("epochs", 0)) != expected_epochs:
            raise ValueError(
                f"correction must use {expected_epochs} full-mixture consolidation epoch(s)"
            )
        if int(stages[0].get("records", 0)) != int(
            data["expected_split_records"]["train"]
        ):
            raise ValueError("correction must retain the complete training mixture")
        semantic_loss = training.get("semantic_token_loss")
        if semantic_correction or decoder_correction or fixed_slot_correction:
            expected_semantic_ids = (
                FIXED_SLOT_SEMANTIC_TOKEN_IDS
                if fixed_slot_correction
                else SEMANTIC_TARGET_TOKEN_IDS
            )
            expected_semantic_weight = 8.0 if fixed_slot_correction else 4.0
            if (
                not isinstance(semantic_loss, dict)
                or semantic_loss.get("enabled") is not True
                or semantic_loss.get("semantic_token_ids") != expected_semantic_ids
                or float(semantic_loss.get("semantic_weight", 0.0))
                != expected_semantic_weight
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
            "approved_qwen_multidegradation_lora_semantic_loss_correction"
            if (decoder_correction or fixed_slot_correction)
            else (
                "approved_qwen_multidegradation_lora_validation_correction"
                if semantic_correction
                else "approved_qwen_multidegradation_lora_frozen_merger_ablation"
            )
        )
        if base_config.get("status") != expected_base_status:
            raise RuntimeError("correction base phase is inconsistent")
        adapter_path = workspace_path(initialization.get("adapter"))
        if sha256(adapter_path) != initialization.get("adapter_sha256"):
            raise RuntimeError("correction initialization adapter changed")
        missing_prefixes = initialization.get("allow_missing_adapter_prefixes", [])
        expected_missing_prefixes = (
            ["model.language_model."]
            if (decoder_correction or fixed_slot_correction)
            else []
        )
        if missing_prefixes != expected_missing_prefixes:
            raise ValueError("correction adapter missing-prefix contract changed")
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
        source_config = base_config
        for _ in range(4):
            source_initialization = source_config.get("initialization")
            if not isinstance(source_initialization, dict):
                break
            forbidden_fixed_tests.append(
                source_initialization.get("base_fixed_test_result")
            )
            parent_value = source_initialization.get("base_config")
            parent_sha = source_initialization.get("base_config_sha256")
            if parent_value is None:
                break
            parent_path = workspace_path(parent_value)
            if sha256(parent_path) != parent_sha:
                raise RuntimeError("correction ancestry config changed")
            source_config = json.loads(parent_path.read_text())
        for fixed_test_path in forbidden_fixed_tests:
            if fixed_test_path is not None and workspace_path(fixed_test_path).exists():
                raise RuntimeError("a fixed test was accessed before correction selection")
    expected_devices = (
        {"0,1": 2, "0,1,2,3,4,5,6,7": 8}
        if (per_type_continuation or visibility_v2)
        else {"0,1": 2, "0,1,2,3": 4}
    )
    if runtime.get("supported_visible_devices") != expected_devices:
        raise ValueError("visible-device contract changed")
    if overfit.get("variant") != "merger_trainable":
        raise ValueError("overfit must exercise the larger approved trainable scope")
    if per_type_continuation:
        if (
            overfit.get("reused") is not True
            or overfit.get("result")
            != config["initialization"]["overfit_result"]
            or overfit.get("result_sha256")
            != config["initialization"]["overfit_result_sha256"]
        ):
            raise ValueError("continuation must reuse the hash-bound passing overfit result")
    elif per_type_dialogue:
        overfit_micro = int(overfit.get("micro_batch_size_per_rank", 0))
        overfit_mapping = overfit.get("gradient_accumulation_steps_by_world_size")
        if not isinstance(overfit_mapping, dict):
            raise ValueError("per-type overfit accumulation contract is missing")
        for world in supported_worlds:
            if overfit_micro * world * int(overfit_mapping[str(world)]) != int(
                overfit["global_batch_size"]
            ):
                raise ValueError("per-type overfit global batch changed")
        if (
            int(overfit.get("records", 0)) != 32
            or int(overfit.get("records_per_composition_size", 0)) != 8
        ):
            raise ValueError("per-type overfit image selection changed")
    elif int(overfit.get("records", 0)) != global_batch or int(
        overfit.get("records_per_composition_size", 0)
    ) * 4 != global_batch:
        raise ValueError("overfit batch must cover eight clean/single/double/triple records")
    if not per_type_continuation:
        minimum_overfit_steps = 50 if per_type_dialogue else 100
        if int(overfit.get("total_optimizer_steps", 0)) < minimum_overfit_steps:
            raise ValueError("overfit gate is too short to demonstrate memorization")
        if overfit.get("require_all_twenty_type_level_pairs") is not True:
            raise ValueError("overfit type-level coverage requirement was disabled")


def validate_per_type_continuation_initialization(
    config: dict[str, object],
) -> None:
    """Bind continued training to the untouched epoch-3 adapter and optimizer."""

    initialization = config.get("initialization")
    if not isinstance(initialization, dict):
        raise ValueError("rank-32 continuation initialization is missing")
    required_hashes = (
        ("base_config", "base_config_sha256"),
        ("adapter", "adapter_sha256"),
        ("optimizer", "optimizer_sha256"),
        ("checkpoint_metadata", "checkpoint_metadata_sha256"),
        ("validation_result", "validation_result_sha256"),
        ("overfit_result", "overfit_result_sha256"),
    )
    for path_key, hash_key in required_hashes:
        path = workspace_path(initialization.get(path_key))
        if sha256(path) != initialization.get(hash_key):
            raise RuntimeError(f"continuation initialization changed: {path_key}")
    base_config = json.loads(workspace_path(initialization["base_config"]).read_text())
    if (
        base_config.get("status")
        != "approved_qwen_multidegradation_per_type_dialogue_rank32_retraining"
        or base_config.get("model") != config.get("model")
        or base_config.get("prompt") != config.get("prompt")
    ):
        raise RuntimeError("continuation architecture differs from the advisor-approved base")
    metadata = json.loads(
        workspace_path(initialization["checkpoint_metadata"]).read_text()
    )
    if (
        metadata.get("config_sha256") != initialization["base_config_sha256"]
        or metadata.get("variant") != "merger_trainable"
        or metadata.get("stage") != "H3_per_type_dialogue_epoch3"
        or metadata.get("global_step") != 1440
        or metadata.get("adapter_sha256") != initialization["adapter_sha256"]
        or metadata.get("optimizer_sha256") != initialization["optimizer_sha256"]
    ):
        raise RuntimeError("epoch-3 continuation metadata is inconsistent")
    validation = json.loads(
        workspace_path(initialization["validation_result"]).read_text()
    )
    if (
        validation.get("config_sha256") != initialization["base_config_sha256"]
        or validation.get("variant") != "merger_trainable"
        or validation.get("split") != "validation"
        or validation.get("adapter") != initialization["adapter"]
        or validation.get("adapter_sha256") != initialization["adapter_sha256"]
        or float(
            validation.get("metrics", {})
            .get("by_composition_size", {})
            .get("3", {})
            .get("degradation_set_exact_match_rate", -1.0)
        )
        != 0.55
    ):
        raise RuntimeError("continuation source validation evidence is inconsistent")
    overfit = json.loads(workspace_path(initialization["overfit_result"]).read_text())
    if overfit.get("status") != "passed" or overfit.get("automatic_gate_pass") is not True:
        raise RuntimeError("continuation source overfit gate did not pass")
    if (
        initialization.get("initial_completed_epochs") != 3
        or initialization.get("initial_optimizer_steps") != 1440
        or initialization.get("initial_dialogue_presentations") != 144000
        or initialization.get("optimizer_resume") != "exact_adamw_state"
        or initialization.get("allow_missing_adapter_prefixes") != []
    ):
        raise ValueError("continuation offset contract changed")
    if workspace_path(initialization["base_fixed_test_result"]).exists():
        raise RuntimeError("the protected fixed test was accessed before continuation selection")


def validate_visibility_v2b_initialization(config: dict[str, object]) -> None:
    """Bind v2b correction to the immutable failed epoch-5 v2 checkpoint."""

    initialization = config.get("initialization")
    if not isinstance(initialization, dict):
        raise ValueError("visibility-v2b initialization is missing")
    required_hashes = (
        ("base_config", "base_config_sha256"),
        ("adapter", "adapter_sha256"),
        ("optimizer", "optimizer_sha256"),
        ("checkpoint_metadata", "checkpoint_metadata_sha256"),
        ("validation_result", "validation_result_sha256"),
        ("overfit_result", "overfit_result_sha256"),
    )
    for path_key, hash_key in required_hashes:
        path = workspace_path(initialization.get(path_key))
        if sha256(path) != initialization.get(hash_key):
            raise RuntimeError(f"visibility-v2b initialization changed: {path_key}")
    base_config = json.loads(workspace_path(initialization["base_config"]).read_text())
    if (
        base_config.get("status")
        != "approved_qwen_visibility_v2_per_type_rank32_decision_calibrated"
        or base_config.get("model") != config.get("model")
        or base_config.get("prompt") != config.get("prompt")
    ):
        raise RuntimeError("visibility-v2b architecture differs from frozen v2")
    metadata = json.loads(
        workspace_path(initialization["checkpoint_metadata"]).read_text()
    )
    if (
        metadata.get("config_sha256") != initialization["base_config_sha256"]
        or metadata.get("variant") != "merger_trainable"
        or metadata.get("stage") != "V2_H5_per_type_dialogue_epoch5"
        or metadata.get("global_step") != 1050
        or metadata.get("adapter_sha256") != initialization["adapter_sha256"]
        or metadata.get("optimizer_sha256") != initialization["optimizer_sha256"]
    ):
        raise RuntimeError("visibility-v2b epoch-5 metadata is inconsistent")
    validation = json.loads(
        workspace_path(initialization["validation_result"]).read_text()
    )
    triple_exact = float(
        validation.get("metrics", {})
        .get("by_composition_size", {})
        .get("3", {})
        .get("degradation_set_exact_match_rate", -1.0)
    )
    if (
        validation.get("status") != "failed"
        or validation.get("automatic_gate_pass") is not False
        or validation.get("config_sha256") != initialization["base_config_sha256"]
        or validation.get("adapter") != initialization["adapter"]
        or validation.get("adapter_sha256") != initialization["adapter_sha256"]
        or abs(triple_exact - 0.7694444444444445) > 1e-12
    ):
        raise RuntimeError("visibility-v2b source validation evidence is inconsistent")
    overfit = json.loads(workspace_path(initialization["overfit_result"]).read_text())
    if overfit.get("status") != "passed" or overfit.get("automatic_gate_pass") is not True:
        raise RuntimeError("visibility-v2b source overfit gate did not pass")
    if (
        initialization.get("initial_completed_epochs") != 5
        or initialization.get("initial_optimizer_steps") != 1050
        or initialization.get("initial_dialogue_presentations") != 210000
        or initialization.get("optimizer_resume") != "exact_adamw_state"
        or initialization.get("allow_missing_adapter_prefixes") != []
    ):
        raise ValueError("visibility-v2b continuation offsets changed")
    if workspace_path(initialization["base_fixed_test_result"]).exists():
        raise RuntimeError("the protected fixed test was accessed before v2b validation")


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
    expected = {
        "matched_lora_module_count": int(model_config["expected_lora_modules"]),
        "merger_trainable_parameters": int(
            model_config["expected_merger_trainable_parameters"]
        ),
        "trainable_parameters": int(
            model_config["expected_trainable_parameters_with_merger"]
        ),
    }
    if "expected_language_lora_trainable_parameters" in model_config:
        expected["language_lora_trainable_parameters"] = int(
            model_config["expected_language_lora_trainable_parameters"]
        )
    actual = {key: int(scope[key]) for key in expected}
    all_lora_parameters = (
        actual["trainable_parameters"] - actual["merger_trainable_parameters"]
    )
    language_lora_parameters = int(
        scope.get("language_lora_trainable_parameters", 0)
    )
    visual_lora_parameters = all_lora_parameters - language_lora_parameters
    if actual != expected or visual_lora_parameters != int(
        model_config["expected_lora_trainable_parameters"]
    ):
        raise RuntimeError(
            f"installed Transformers Qwen architecture changed: actual={actual} expected={expected}"
        )
    return {
        **actual,
        "lora_trainable_parameters": all_lora_parameters,
        "visual_lora_trainable_parameters": visual_lora_parameters,
    }


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
    external_validation: list[dict[str, object]] = []
    if config.get("status") == "approved_qwen_visibility_v2b_hard_triple_continuation":
        external_validation = read_jsonl(workspace_path(data["validation_manifest"]))
        if (
            len(external_validation) != 1200
            or any(record.get("split") != "validation" for record in external_validation)
        ):
            raise RuntimeError("v2b external validation cardinality changed")
        old_source_ids = {str(record["source_id"]) for record in records}
        old_source_paths = {str(record["source_path"]) for record in records}
        old_source_hashes = {str(record["source_sha256"]) for record in records}
        fresh_source_ids: set[str] = set()
        for record in external_validation:
            if (
                str(record["source_id"]) in old_source_ids
                or str(record["source_path"]) in old_source_paths
                or str(record["source_sha256"]) in old_source_hashes
            ):
                raise RuntimeError("v2b validation source overlaps training/protected data")
            fresh_source_ids.add(str(record["source_id"]))
            output = workspace_path(record["output_path"])
            if not output.is_file():
                raise FileNotFoundError(f"v2b validation image not generated: {output}")
        if len(fresh_source_ids) < 180:
            raise RuntimeError("v2b validation does not use the frozen fresh source pool")
    if full_hashes:
        evidence_root = workspace_path(
            data.get(
                "generation_evidence_root",
                "data/qwen_multidegradation_pilot_v1/.generation",
            )
        )
        def verify_generated(record: dict[str, object]) -> None:
            evidence_path = (
                evidence_root
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
        if external_validation:
            external_evidence_root = workspace_path(data["validation_generation_evidence_root"])

            def verify_external_generated(record: dict[str, object]) -> None:
                evidence_path = (
                    external_evidence_root
                    / "validation"
                    / f"{record['record_id']}.json"
                )
                if not evidence_path.is_file():
                    raise FileNotFoundError(
                        f"v2b generation evidence missing: {evidence_path}"
                    )
                evidence = json.loads(evidence_path.read_text())
                record_hash = hashlib.sha256(
                    json.dumps(record, separators=(",", ":"), sort_keys=True).encode()
                ).hexdigest()
                output = workspace_path(record["output_path"])
                if (
                    evidence.get("record_id") != record["record_id"]
                    or evidence.get("manifest_sha256")
                    != data["validation_manifest_sha256"]
                    or evidence.get("manifest_record_sha256") != record_hash
                    or sha256(output) != evidence.get("output_sha256")
                ):
                    raise RuntimeError(
                        f"v2b generated image evidence changed: {record['record_id']}"
                    )

            with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
                list(executor.map(verify_external_generated, external_validation))
    partition_evidence: dict[str, object] | None = None
    partition_config = data.get("calibration_partition")
    if partition_config is not None:
        if not isinstance(partition_config, dict):
            raise ValueError("calibration partition contract is malformed")
        partition_path = workspace_path(partition_config.get("path"))
        if sha256(partition_path) != partition_config.get("sha256"):
            raise RuntimeError("calibration partition hash changed")
        partition = json.loads(partition_path.read_text())
        partition_evidence = validate_training_source_partition(
            partition,
            records,
            manifest_sha256=str(data["manifest_sha256"]),
        )
        if (
            int(partition_evidence["gradient_records_per_epoch"])
            != int(data["gradient_records_per_epoch"])
            or int(partition_evidence["calibration_records"])
            != int(data["calibration_records"])
        ):
            raise RuntimeError("calibration partition counts changed")
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
    prompt = config["prompt"]
    assert isinstance(prompt, dict)
    target_schema = str(prompt["target_schema"])
    if target_schema == PER_TYPE_DIALOGUE_SCHEMA:
        expanded_labels = [label for label in labels for _ in DEGRADATIONS]
        expanded_images = [image.copy() for image in images for _ in DEGRADATIONS]
        queries = [degradation for _ in labels for degradation in DEGRADATIONS]
        batch, counts = prepare_per_type_supervised_batch(
            processor, expanded_labels, expanded_images, queries
        )
        expected_count = len(labels) * len(DEGRADATIONS)
    else:
        batch, counts = prepare_supervised_batch(
            processor, labels, images, target_schema
        )
        expected_count = len(labels)
    if len(counts) != expected_count or any(count <= 0 for count in counts):
        raise RuntimeError("assistant-token supervision is empty")
    decision_token_positions_verified = False
    decision = config.get("decision_calibration")
    if decision is not None:
        if not isinstance(decision, dict):
            raise ValueError("decision calibration is malformed")
        token_ids = decision.get("level_token_ids")
        if token_ids != FIXED_SLOT_SEMANTIC_TOKEN_IDS:
            raise RuntimeError("L1--L5 decoder token identities changed")
        labels_tensor = batch["labels"]
        for row in labels_tensor:
            matches = sum(int(row.eq(token_id).sum().item()) for token_id in token_ids)
            if matches != 1:
                raise RuntimeError("assistant response does not contain one decision token")
        for level, token_id in enumerate(token_ids, start=1):
            encoded = processor.tokenizer(
                str(level), add_special_tokens=False
            )["input_ids"]
            if encoded != [token_id]:
                raise RuntimeError("frozen L1--L5 single-token mapping changed")
        decision_token_positions_verified = True
    semantic_token_positions_verified = False
    if config.get("status") in RANK64_100K_STATUSES:
        semantic = config["training"].get("semantic_token_loss")
        if not isinstance(semantic, dict):
            raise RuntimeError("rank-64 semantic-token loss is missing")
        token_ids = semantic.get("semantic_token_ids")
        if token_ids != FIXED_SLOT_SEMANTIC_TOKEN_IDS:
            raise RuntimeError("rank-64 L1--L5 token IDs changed")
        for level, token_id in enumerate(token_ids, start=1):
            if processor.tokenizer(
                str(level), add_special_tokens=False
            )["input_ids"] != [token_id]:
                raise RuntimeError("rank-64 L1--L5 tokens are no longer atomic")
        expected_per_row = 1 if target_schema == PER_TYPE_DIALOGUE_SCHEMA else 5
        for row in batch["labels"]:
            matches = sum(
                int(row.eq(token_id).sum().item()) for token_id in token_ids
            )
            if matches != expected_per_row:
                raise RuntimeError(
                    "rank-64 assistant answer has the wrong decision-token count"
                )
        semantic_token_positions_verified = True
    presence_token_positions_verified = False
    presence = config["training"].get("presence_auxiliary_loss")
    if presence is not None:
        if (
            not isinstance(presence, dict)
            or presence.get("level_token_ids") != FIXED_SLOT_SEMANTIC_TOKEN_IDS
        ):
            raise RuntimeError("decoder presence-loss L1--L5 token IDs changed")
        expected_per_row = 1 if target_schema == PER_TYPE_DIALOGUE_SCHEMA else 5
        for row in batch["labels"]:
            matches = sum(
                int(row.eq(token_id).sum().item())
                for token_id in FIXED_SLOT_SEMANTIC_TOKEN_IDS
            )
            if matches != expected_per_row:
                raise RuntimeError(
                    "presence-loss assistant answer has the wrong decision-token count"
                )
        presence_token_positions_verified = True
    return {
        "records": len(records),
        "unique_sources_used": len(source_roles),
        "representative_target_token_counts": counts,
        "pixel_values_shape": list(batch["pixel_values"].shape),
        "all_generated_image_hashes_verified": full_hashes,
        "external_validation": (
            {
                "records": len(external_validation),
                "unique_sources": len(
                    {str(record["source_id"]) for record in external_validation}
                ),
                "source_disjoint_from_training_and_protected_test": True,
                "all_generated_image_hashes_verified": full_hashes,
            }
            if external_validation
            else None
        ),
        "decision_token_positions_verified": decision_token_positions_verified,
        "semantic_token_positions_verified": semantic_token_positions_verified,
        "presence_token_positions_verified": presence_token_positions_verified,
        "training_only_calibration_partition": (
            {
                "calibration_records": partition_evidence["calibration_records"],
                "gradient_eligible_records": partition_evidence[
                    "gradient_eligible_records"
                ],
                "gradient_records_per_epoch": partition_evidence[
                    "gradient_records_per_epoch"
                ],
                "source_disjoint": partition_evidence["source_disjoint"],
            }
            if partition_evidence is not None
            else None
        ),
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
