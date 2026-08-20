#!/usr/bin/env python3
"""Fail-closed CPU/GPU preflight for source-disjoint real Qwen adaptation."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import platform
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lowlight_restoration.data import PairRecord  # noqa: E402
from lowlight_restoration.data import load_pair_manifest
from lowlight_restoration.qwen_real_adaptation import (  # noqa: E402
    ALLOWED_WORLD_SIZES,
    global_training_groups,
    load_adaptation_config,
    partition_training_groups,
    records_by_real_split,
    select_synthetic_replay,
    training_schedule_evidence,
    validate_adaptation_contract,
)
from lowlight_restoration.qwen_supervised import (  # noqa: E402
    DEGRADATION_QUESTION,
    SYSTEM_PROMPT,
    canonical_target,
    open_degraded_images,
    prepare_supervised_batch,
    record_bundle_sha256,
)
from smoke.qwen_lowlight_supervised_preflight import (  # noqa: E402
    inspect_checkpoint_parameters,
    load_inventory,
    sha256,
    validate_model_file_sizes,
    workspace_path,
)


def _atomic_write_new_json(path: Path, payload: object) -> None:
    content = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    if path.exists():
        if not path.is_file() or path.read_bytes() != content:
            raise RuntimeError(f"existing adaptation preflight differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(content)


def _load_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise RuntimeError(f"JSON evidence is not an object: {path}")
    return payload


def _prompt_contract(config: dict[str, object]) -> None:
    prompt = config["prompt"]
    assert isinstance(prompt, dict)
    expected = {
        "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
        "degradation_question_sha256": hashlib.sha256(
            DEGRADATION_QUESTION.encode()
        ).hexdigest(),
        "target_schema": "project_clean_l1_active_lowlight_l2_l5_v1",
    }
    for name, value in expected.items():
        if prompt.get(name) != value:
            raise RuntimeError(f"adaptation prompt contract changed: {name}")


def _verify_preserved_failures(config: dict[str, object]) -> dict[str, object]:
    failures = config["preserved_failures"]
    assert isinstance(failures, dict)
    original_path = workspace_path(failures["original_free_generation_result"])
    constrained_path = workspace_path(failures["post_failure_constrained_result"])
    for path, expected in (
        (original_path, failures["original_free_generation_result_sha256"]),
        (constrained_path, failures["post_failure_constrained_result_sha256"]),
    ):
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"preserved real-domain failure changed: {path}")
    original = _load_json(original_path)
    constrained = _load_json(constrained_path)
    if (
        original.get("status") != "failed"
        or original.get("automatic_gate_pass") is not False
        or constrained.get("status") != "failed"
        or constrained.get("automatic_gate_pass") is not False
        or constrained.get("original_gate_remains_failed") is not True
        or constrained.get("independent_validation_claim_allowed") is not False
    ):
        raise RuntimeError("preserved failures were reclassified")
    return {
        "original_result_sha256": failures["original_free_generation_result_sha256"],
        "constrained_result_sha256": failures["post_failure_constrained_result_sha256"],
        "original_status": "failed",
        "constrained_status": "failed",
    }


def _verify_initialization(
    config: dict[str, object], *, hash_visual: bool
) -> dict[str, object]:
    model = config["model"]
    assert isinstance(model, dict)
    initialization = model["initialization"]
    assert isinstance(initialization, dict)
    visual = workspace_path(initialization["visual_checkpoint"])
    metadata_path = workspace_path(initialization["checkpoint_metadata"])
    summary_path = workspace_path(initialization["pilot_summary"])
    for path, expected in (
        (metadata_path, initialization["checkpoint_metadata_sha256"]),
        (summary_path, initialization["pilot_summary_sha256"]),
    ):
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"adaptation initialization evidence changed: {path}")
    if not visual.is_file():
        raise RuntimeError("adaptation initialization visual checkpoint is missing")
    if hash_visual and sha256(visual) != initialization["visual_sha256"]:
        raise RuntimeError("adaptation initialization visual SHA-256 changed")
    metadata = _load_json(metadata_path)
    summary = _load_json(summary_path)
    checks = summary.get("gate_checks")
    selected_metadata = summary.get("selected_checkpoint_metadata")
    if (
        metadata.get("step") != 600
        or metadata.get("visual_sha256") != initialization["visual_sha256"]
        or summary.get("status") != "passed"
        or summary.get("automatic_gate_pass") is not True
        or summary.get("selected_checkpoint_step") != 600
        or summary.get("selected_checkpoint") != "checkpoints/step_000600"
        or not isinstance(checks, dict)
        or not checks
        or not all(value is True for value in checks.values())
        or selected_metadata != metadata
        or summary.get("official_foundir_used") is not False
    ):
        raise RuntimeError("step-600 pilot initialization evidence is inconsistent")
    tensor_count = 0
    parameters = 0
    with safe_open(visual, framework="pt", device="cpu") as handle:
        for name in handle.keys():
            shape = handle.get_slice(name).get_shape()
            elements = 1
            for dimension in shape:
                elements *= dimension
            parameters += elements
            tensor_count += 1
    if parameters != model["expected_trainable_parameters"] or tensor_count != 390:
        raise RuntimeError("step-600 visual tensor inventory changed")
    return {
        "selected_checkpoint_step": 600,
        "visual_sha256": initialization["visual_sha256"],
        "visual_hash_checked": hash_visual,
        "visual_tensors": tensor_count,
        "visual_parameters": parameters,
        "checkpoint_metadata_sha256": initialization["checkpoint_metadata_sha256"],
        "pilot_summary_sha256": initialization["pilot_summary_sha256"],
        "pilot_automatic_gate_pass": True,
        "optimizer_reinitialized": True,
    }


def validate_static_contract(
    config_path: Path,
) -> tuple[dict[str, object], str, dict[str, object]]:
    config, config_sha, _ = load_adaptation_config(config_path)
    validate_adaptation_contract(config)
    _prompt_contract(config)
    data = config["data"]
    assert isinstance(data, dict)
    real = data["real"]
    synthetic = data["synthetic_replay"]
    assert isinstance(real, dict) and isinstance(synthetic, dict)
    for path_value, expected, label in (
        (real["manifest"], real["manifest_sha256"], "real manifest"),
        (real["metadata"], real["metadata_sha256"], "real metadata"),
        (
            synthetic["manifest"],
            synthetic["manifest_sha256"],
            "synthetic manifest",
        ),
    ):
        path = workspace_path(path_value)
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"adaptation {label} changed")
    metadata = _load_json(workspace_path(real["metadata"]))
    if (
        metadata.get("status") != "frozen_before_qwen_real_adaptation_inference"
        or metadata.get("manifest_sha256") != real["manifest_sha256"]
        or metadata.get("record_bundle_sha256") != real["record_bundle_sha256"]
        or metadata.get("split_contracts") != real["splits"]
        or metadata.get("model_predictions_consulted_during_split") is not False
        or metadata.get("previously_evaluated_real_records_used") is not False
        or metadata.get("official_foundir_used") is not False
        or metadata.get("holdout_protection", {}).get(
            "predictions_consulted_before_manifest_freeze"
        )
        is not False
    ):
        raise RuntimeError("real-adaptation metadata contract changed")
    failures = _verify_preserved_failures(config)
    return config, config_sha, failures


def _hash_model_files(
    inventory: dict[str, object], expected_hashes: dict[str, str]
) -> None:
    inventory_model = inventory["model"]
    assert isinstance(inventory_model, dict)
    model_dir = workspace_path(inventory_model["local_path"])

    def verify(item: tuple[str, str]) -> None:
        name, expected = item
        if sha256(model_dir / name) != expected:
            raise RuntimeError(f"base Qwen model file changed: {name}")

    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(verify, sorted(expected_hashes.items())))


def _load_data(
    config: dict[str, object],
) -> tuple[
    dict[str, list[PairRecord]],
    list[PairRecord],
    list[PairRecord],
    list[PairRecord],
]:
    data = config["data"]
    assert isinstance(data, dict)
    real_config = data["real"]
    synthetic_config = data["synthetic_replay"]
    assert isinstance(real_config, dict) and isinstance(synthetic_config, dict)
    real_records = load_pair_manifest(workspace_path(real_config["manifest"]))
    if record_bundle_sha256(real_records) != real_config["record_bundle_sha256"]:
        raise RuntimeError("real-adaptation full record bundle changed")
    real_splits = records_by_real_split(real_records, config)
    synthetic_records = load_pair_manifest(workspace_path(synthetic_config["manifest"]))
    synthetic_train = [
        record
        for record in synthetic_records
        if record.split == synthetic_config["training_split"]
    ]
    if (
        len(synthetic_train) != synthetic_config["training_records"]
        or len({record.source_id for record in synthetic_train})
        != synthetic_config["training_sources"]
        or Counter(record.level_index for record in synthetic_train)
        != Counter(
            {
                level: synthetic_config["training_records_per_level"]
                for level in range(1, 6)
            }
        )
        or record_bundle_sha256(synthetic_train)
        != synthetic_config["training_bundle_sha256"]
    ):
        raise RuntimeError("synthetic training replay pool changed")
    synthetic_validation = select_synthetic_replay(
        synthetic_records,
        source_count=int(synthetic_config["validation_source_count"]),
        seed=int(synthetic_config["validation_selection_seed"]),
    )
    if (
        len(synthetic_validation) != synthetic_config["validation_records"]
        or record_bundle_sha256(synthetic_validation)
        != synthetic_config["validation_bundle_sha256"]
    ):
        raise RuntimeError("synthetic validation replay selection changed")
    return real_splits, synthetic_records, synthetic_train, synthetic_validation


def _verify_real_source_inventory(
    config: dict[str, object], *, show_progress: bool
) -> dict[str, object]:
    data = config["data"]
    assert isinstance(data, dict)
    real = data["real"]
    assert isinstance(real, dict)
    metadata = _load_json(workspace_path(real["metadata"]))
    source_files = metadata.get("source_files")
    if not isinstance(source_files, list) or len(source_files) != 1_378:
        raise RuntimeError("real adaptation source-file inventory changed")

    def verify(item: object) -> tuple[str, str, int]:
        if not isinstance(item, dict):
            raise RuntimeError("malformed real source-file entry")
        path = workspace_path(item.get("path"))
        if (
            not path.is_file()
            or path.stat().st_size != item.get("bytes")
            or sha256(path) != item.get("sha256")
        ):
            raise RuntimeError(f"real adaptation source file changed: {path}")
        return str(item["path"]), str(item["sha256"]), int(item["bytes"])

    digest = hashlib.sha256()
    total_bytes = 0
    with ThreadPoolExecutor(max_workers=8) as executor:
        for index, (path, actual, size) in enumerate(
            executor.map(verify, source_files), start=1
        ):
            digest.update(f"{path}\0{actual}\0{size}\n".encode())
            total_bytes += size
            if show_progress and (index % 250 == 0 or index == len(source_files)):
                print(
                    f"QWEN_REAL_ADAPTATION_PREFLIGHT_REAL_FILES "
                    f"{index}/{len(source_files)}",
                    flush=True,
                )
    if digest.hexdigest() != metadata["source_file_inventory_sha256"]:
        # The metadata inventory digest is canonical JSON, while this streaming
        # digest independently proves the files checked in this preflight.
        canonical = hashlib.sha256(
            json.dumps(source_files, separators=(",", ":"), sort_keys=True).encode()
        ).hexdigest()
        if canonical != metadata["source_file_inventory_sha256"]:
            raise RuntimeError("real source inventory metadata digest changed")
    source = metadata["source"]
    archive = workspace_path(source["archive"])
    if (
        not archive.is_file()
        or archive.stat().st_size != source["archive_bytes"]
        or sha256(archive) != source["archive_sha256"]
    ):
        raise RuntimeError("official LOL-v2 archive changed")
    separation = metadata["separation_from_previously_evaluated_real_data"]
    if (
        separation.get("exact_sha256_overlap") != 0
        or separation.get("train_test_normal_perceptual_edges_hamming_le_4") != 0
        or separation.get("previously_evaluated_records_used_for_training") is not False
    ):
        raise RuntimeError("real adaptation separation evidence changed")
    return {
        "files": len(source_files),
        "total_bytes": total_bytes,
        "verified_file_bundle_sha256": digest.hexdigest(),
        "metadata_inventory_sha256": metadata["source_file_inventory_sha256"],
        "archive_sha256": source["archive_sha256"],
        "exact_overlap_with_previously_evaluated_sources": 0,
        "perceptual_overlap_with_LOLv2_Test_Normal": 0,
    }


def _scheduled_synthetic_records(
    real_train: list[PairRecord],
    synthetic_train: list[PairRecord],
    *,
    total_steps: int,
    seed: int,
) -> list[PairRecord]:
    selected: dict[str, PairRecord] = {}
    for step in range(1, total_steps + 1):
        for domain, group in global_training_groups(
            real_train, synthetic_train, step=step, seed=seed
        ):
            if domain == "synthetic_replay":
                for record in group:
                    selected[record.pair_id] = record
    return [selected[pair_id] for pair_id in sorted(selected)]


def _verify_synthetic_files(
    training: list[PairRecord],
    validation: list[PairRecord],
    *,
    show_progress: bool,
) -> dict[str, object]:
    unique = {record.pair_id: record for record in [*training, *validation]}

    def verify(record: PairRecord) -> tuple[str, str, int]:
        path = record.degraded_path
        if not path.is_file() or sha256(path) != record.degraded_sha256:
            raise RuntimeError(f"synthetic replay image changed: {record.pair_id}")
        return record.pair_id, record.degraded_sha256, path.stat().st_size

    digest = hashlib.sha256()
    total_bytes = 0
    values = [unique[pair_id] for pair_id in sorted(unique)]
    with ThreadPoolExecutor(max_workers=8) as executor:
        for index, (pair_id, actual, size) in enumerate(
            executor.map(verify, values), start=1
        ):
            digest.update(f"{pair_id}\0{actual}\0{size}\n".encode())
            total_bytes += size
            if show_progress and (index % 500 == 0 or index == len(values)):
                print(
                    f"QWEN_REAL_ADAPTATION_PREFLIGHT_SYNTHETIC_FILES "
                    f"{index}/{len(values)}",
                    flush=True,
                )
    return {
        "unique_files": len(values),
        "scheduled_training_files": len(training),
        "validation_replay_files": len(validation),
        "total_bytes": total_bytes,
        "verified_file_bundle_sha256": digest.hexdigest(),
    }


def _schedule_contract(
    config: dict[str, object],
    real_train: list[PairRecord],
    synthetic_train: list[PairRecord],
) -> dict[str, object]:
    training = config["training"]
    distributed = config["distributed"]
    assert isinstance(training, dict) and isinstance(distributed, dict)
    evidence = training_schedule_evidence(
        real_train,
        synthetic_train,
        total_steps=int(training["total_steps"]),
        seed=int(training["seed"]),
    )
    if (
        evidence["schedule_sha256"] != training["schedule_sha256"]
        or evidence["total_sample_presentations"]
        != training["total_sample_presentations"]
        or evidence["domain_presentations"]
        != {
            "real": training["real_sample_presentations"],
            "synthetic_replay": training["synthetic_sample_presentations"],
        }
    ):
        raise RuntimeError("world-size-invariant adaptation schedule changed")
    first_groups = global_training_groups(
        real_train, synthetic_train, step=1, seed=int(training["seed"])
    )
    assignments: dict[str, dict[str, object]] = {}
    for world_size in ALLOWED_WORLD_SIZES:
        local = {
            str(rank): [
                {
                    "domain": domain,
                    "pair_ids": [record.pair_id for record in group],
                }
                for domain, group in partition_training_groups(
                    first_groups, world_size=world_size, rank=rank
                )
            ]
            for rank in range(world_size)
        }
        flattened = [
            tuple(item["pair_ids"])
            for rank_values in local.values()
            for item in rank_values
        ]
        expected = [
            tuple(record.pair_id for record in group) for _, group in first_groups
        ]
        if Counter(flattened) != Counter(expected):
            raise RuntimeError(
                f"adaptation rank partition changed for {world_size} GPUs"
            )
        assignments[str(world_size)] = {
            "gradient_accumulation_steps": distributed[
                "gradient_accumulation_by_world_size"
            ][str(world_size)],
            "first_step_rank_assignments": local,
        }
    return {**evidence, "world_size_assignments": assignments}


def run_preflight(
    config_path: Path,
    *,
    skip_large_model_hashes: bool,
    write_artifact: bool = True,
    show_progress: bool = True,
    require_gpu: bool = False,
) -> dict[str, object]:
    config, config_sha, failure_evidence = validate_static_contract(config_path)
    inventory, inventory_sha = load_inventory(config)
    inventory_model = inventory["model"]
    processor_config = config["processor"]
    runtime = config["runtime"]
    training_config = config["training"]
    assert isinstance(inventory_model, dict)
    assert all(
        isinstance(section, dict)
        for section in (processor_config, runtime, training_config)
    )
    expected_model_hashes = validate_model_file_sizes(inventory)
    if not skip_large_model_hashes:
        _hash_model_files(inventory, expected_model_hashes)
    parameter_groups, tensor_count = inspect_checkpoint_parameters(config, inventory)
    initialization = _verify_initialization(config, hash_visual=True)
    real_splits, _, synthetic_train, synthetic_validation = _load_data(config)
    schedule = _schedule_contract(config, real_splits["train"], synthetic_train)
    scheduled_synthetic = _scheduled_synthetic_records(
        real_splits["train"],
        synthetic_train,
        total_steps=int(training_config["total_steps"]),
        seed=int(training_config["seed"]),
    )
    real_file_evidence = _verify_real_source_inventory(
        config, show_progress=show_progress
    )
    synthetic_file_evidence = _verify_synthetic_files(
        scheduled_synthetic,
        synthetic_validation,
        show_progress=show_progress,
    )

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
        raise RuntimeError("installed Qwen architecture changed")
    if float(getattr(loaded_config, "attention_dropout", -1.0)) != 0.0:
        raise RuntimeError("Qwen attention dropout is not zero")
    if Qwen2_5_VLForConditionalGeneration.__name__ != config["model"]["architecture"]:
        raise RuntimeError("installed Qwen generation class changed")
    processor = AutoProcessor.from_pretrained(
        model_dir,
        local_files_only=True,
        use_fast=False,
        min_pixels=int(processor_config["min_pixels"]),
        max_pixels=int(processor_config["max_pixels"]),
    )
    processor.tokenizer.padding_side = str(processor_config["padding_side"])
    if type(processor.image_processor).__name__ != "Qwen2VLImageProcessor":
        raise RuntimeError("frozen slow Qwen image processor was not loaded")
    representatives = [
        next(record for record in real_splits["train"] if record.level_index == level)
        for level in range(1, 6)
    ]
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
        raise RuntimeError(f"adaptation visual-token budget changed: {visual_tokens}")
    for row, record in enumerate(representatives):
        target_ids = batch["labels"][row]
        decoded = processor.tokenizer.decode(
            target_ids[target_ids != -100],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()
        if decoded != canonical_target(record.level_index):
            raise RuntimeError(
                f"adaptation target tokenization changed: {record.pair_id}"
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
            f"adaptation environment changed expected={expected_versions} "
            f"actual={actual_versions}"
        )
    data_config = config["data"]
    assert isinstance(data_config, dict)
    real_config = data_config["real"]
    synthetic_config = data_config["synthetic_replay"]
    assert isinstance(real_config, dict) and isinstance(synthetic_config, dict)
    split_evidence = {
        split: {
            "records": len(records),
            "sources": len({record.source_id for record in records}),
            "per_level": {
                str(level): Counter(record.level_index for record in records)[level]
                for level in range(1, 6)
            },
            "bundle_sha256": record_bundle_sha256(records),
        }
        for split, records in real_splits.items()
    }
    result: dict[str, object] = {
        "schema_version": 1,
        "scope": "qwen_real_lowlight_adaptation_preflight",
        "config_sha256": config_sha,
        "inventory_config_sha256": inventory_sha,
        "model_revision": inventory_model["revision"],
        "model_files": len(expected_model_hashes),
        "model_hash_contract_sha256": hashlib.sha256(
            json.dumps(
                expected_model_hashes, separators=(",", ":"), sort_keys=True
            ).encode()
        ).hexdigest(),
        "large_model_hashes_checked": not skip_large_model_hashes,
        "parameter_groups": parameter_groups,
        "tensor_count": tensor_count,
        "initialization": initialization,
        "preserved_failures": failure_evidence,
        "real_manifest_sha256": real_config["manifest_sha256"],
        "real_metadata_sha256": real_config["metadata_sha256"],
        "real_split_evidence": split_evidence,
        "real_file_evidence": real_file_evidence,
        "synthetic_manifest_sha256": synthetic_config["manifest_sha256"],
        "synthetic_training_bundle_sha256": synthetic_config["training_bundle_sha256"],
        "synthetic_validation_bundle_sha256": synthetic_config[
            "validation_bundle_sha256"
        ],
        "synthetic_file_evidence": synthetic_file_evidence,
        "training_schedule": schedule,
        "visual_tokens": visual_tokens,
        "assistant_target_tokens": target_counts,
        "processor": type(processor).__name__,
        "image_processor": type(processor.image_processor).__name__,
        "tokenizer": type(processor.tokenizer).__name__,
        "transformers_modeling_source": str(inspect.getsourcefile(modeling_qwen2_5_vl)),
        "versions": actual_versions,
        "cuda_visible_in_preflight": torch.cuda.is_available(),
        "allowed_world_sizes": list(ALLOWED_WORLD_SIZES),
        "real_test_content_verified_only": True,
        "real_test_used_for_training_or_selection": False,
        "previously_evaluated_real_records_used_for_training": False,
        "official_foundir_used": False,
    }
    if require_gpu and result["cuda_visible_in_preflight"] is not True:
        raise RuntimeError("adaptation GPU preflight requires visible CUDA devices")
    if write_artifact:
        output_dir = workspace_path(runtime["output_dir"])
        name = (
            "preflight_skip_large_model_hashes.json"
            if skip_large_model_hashes
            else "preflight_full.json"
        )
        _atomic_write_new_json(output_dir / name, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--skip-large-model-hashes", action="store_true")
    parser.add_argument("--contract-only", action="store_true")
    parser.add_argument("--no-write-artifact", action="store_true")
    parser.add_argument("--require-gpu", action="store_true")
    args = parser.parse_args()
    if args.contract_only:
        if not args.skip_large_model_hashes:
            parser.error("--contract-only requires --skip-large-model-hashes")
        if args.require_gpu:
            parser.error("--contract-only cannot require a GPU")
        _, config_sha, failures = validate_static_contract(args.config)
        initialization = _verify_initialization(
            load_adaptation_config(args.config)[0], hash_visual=False
        )
        print(
            "QWEN_REAL_ADAPTATION_CONTRACT_PASS "
            + json.dumps(
                {
                    "config_sha256": config_sha,
                    "initialization": initialization,
                    "preserved_failures": failures,
                },
                sort_keys=True,
            )
        )
        return
    result = run_preflight(
        args.config,
        skip_large_model_hashes=args.skip_large_model_hashes,
        write_artifact=not args.no_write_artifact,
        require_gpu=args.require_gpu,
    )
    marker = (
        "QWEN_REAL_ADAPTATION_PREFLIGHT_SKIP_MODEL_HASHES_PASS"
        if args.skip_large_model_hashes
        else "QWEN_REAL_ADAPTATION_PREFLIGHT_FULL_PASS"
    )
    concise = {
        "config_sha256": result["config_sha256"],
        "model_revision": result["model_revision"],
        "initialization": result["initialization"],
        "real_split_evidence": result["real_split_evidence"],
        "real_file_evidence": result["real_file_evidence"],
        "synthetic_file_evidence": result["synthetic_file_evidence"],
        "schedule_sha256": result["training_schedule"]["schedule_sha256"],
        "allowed_world_sizes": result["allowed_world_sizes"],
        "versions": result["versions"],
    }
    print(marker + " " + json.dumps(concise, sort_keys=True))


if __name__ == "__main__":
    main()
