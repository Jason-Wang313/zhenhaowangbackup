#!/usr/bin/env python3
"""Fail-closed CPU preflight for the source-disjoint Qwen 8k DDP8 pilot."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import platform
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lowlight_restoration.data import PairRecord, load_pair_manifest
from lowlight_restoration.qwen_distributed import (
    complete_source_groups,
    partition_source_groups,
)
from lowlight_restoration.qwen_pilot import (
    REQUIRED_SPLITS,
    load_pilot_config,
    records_by_split,
    validate_pilot_contract,
)
from lowlight_restoration.qwen_supervised import (
    DEGRADATION_QUESTION,
    SYSTEM_PROMPT,
    canonical_target,
    open_degraded_images,
    prepare_supervised_batch,
)
from smoke.qwen_lowlight_supervised_preflight import (
    inspect_checkpoint_parameters,
    load_inventory,
    sha256,
    validate_model_file_sizes,
    workspace_path,
)


def _atomic_write_new_json(path: Path, payload: object) -> None:
    content = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    if path.exists():
        if not path.is_file() or path.read_bytes() != content:
            raise RuntimeError(f"existing immutable preflight artifact differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(content)


def _load_json_object(path: Path) -> dict[str, object]:
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
    }
    for key, digest in expected.items():
        if prompt.get(key) != digest:
            raise RuntimeError(f"frozen supervised prompt changed: {key}")
    if prompt.get("target_schema") != "project_clean_l1_active_lowlight_l2_l5_v1":
        raise RuntimeError("frozen target schema changed")


def _validate_overfit_evidence(config: dict[str, object]) -> dict[str, object]:
    waiver = config["scaling_waiver"]
    assert isinstance(waiver, dict)
    config_path = workspace_path(waiver.get("operational_config"))
    summary_path = workspace_path(waiver.get("operational_evidence"))
    visual_path = workspace_path(waiver.get("operational_final_visual_checkpoint"))
    expected_paths = (
        (config_path, waiver.get("operational_config_sha256")),
        (summary_path, waiver.get("operational_evidence_sha256")),
        (visual_path, waiver.get("operational_final_visual_checkpoint_sha256")),
    )
    for path, expected in expected_paths:
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"passing DDP8 overfit evidence changed: {path}")

    overfit_config = _load_json_object(config_path)
    summary = _load_json_object(summary_path)
    checks = summary.get("gate_checks")
    metadata = summary.get("final_checkpoint_metadata")
    if (
        overfit_config.get("status")
        != "approved_for_advisor_route_supervised_vlm_overfit40_ddp8_retry_gate_only"
        or summary.get("status") != "passed"
        or summary.get("automatic_gate_pass") is not True
        or not isinstance(checks, dict)
        or not checks
        or not all(value is True for value in checks.values())
        or summary.get("config_sha256") != waiver["operational_config_sha256"]
        or summary.get("world_size") != 8
        or summary.get("model_revision") != "cc594898137f460bfe9f0759e9844b3ce807cfb5"
        or summary.get("test_split_used") is not False
        or summary.get("official_foundir_used") is not False
        or not isinstance(metadata, dict)
        or metadata.get("visual_sha256")
        != waiver["operational_final_visual_checkpoint_sha256"]
        or metadata.get("world_size") != 8
        or metadata.get("global_batch_size") != 40
    ):
        raise RuntimeError("passing DDP8 overfit summary is internally inconsistent")
    throughput = summary.get("global_samples_per_second_this_process")
    if (
        isinstance(throughput, bool)
        or not isinstance(throughput, (int, float))
        or not math.isclose(
            float(throughput),
            float(waiver["observed_ddp8_global_samples_per_second"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise RuntimeError("passing DDP8 throughput evidence changed")
    expected_visual = (
        summary_path.parent / str(summary.get("final_checkpoint")) / "visual.safetensors"
    ).resolve()
    if expected_visual != visual_path.resolve():
        raise RuntimeError("passing DDP8 final-checkpoint path changed")
    return {
        "config_sha256": waiver["operational_config_sha256"],
        "summary_sha256": waiver["operational_evidence_sha256"],
        "final_visual_sha256": waiver[
            "operational_final_visual_checkpoint_sha256"
        ],
        "world_size": 8,
        "global_batch_size": 40,
        "global_samples_per_second": float(throughput),
        "automatic_gate_pass": True,
    }


def _hash_model_files(
    inventory: dict[str, object], expected_hashes: dict[str, str]
) -> None:
    inventory_model = inventory["model"]
    assert isinstance(inventory_model, dict)
    model_dir = workspace_path(inventory_model["local_path"])

    def verify(item: tuple[str, str]) -> str:
        name, expected = item
        if sha256(model_dir / name) != expected:
            raise RuntimeError(f"model file SHA-256 changed: {name}")
        return name

    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(verify, sorted(expected_hashes.items())))


def _validate_all_images(
    splits: dict[str, list[PairRecord]], *, show_progress: bool
) -> dict[str, object]:
    records = [record for split in REQUIRED_SPLITS for record in splits[split]]
    if len(records) != 10_000 or len({record.pair_id for record in records}) != 10_000:
        raise RuntimeError("pilot image selection is not exactly 10,000 unique records")
    paths = [record.degraded_path for record in records]
    if len(set(paths)) != len(paths):
        raise RuntimeError("pilot records do not map to 10,000 unique image files")

    def inspect_record(record: PairRecord) -> tuple[str, str, int]:
        if not record.degraded_path.is_file():
            raise RuntimeError(f"pilot image is missing: {record.pair_id}")
        actual = sha256(record.degraded_path)
        if actual != record.degraded_sha256:
            raise RuntimeError(f"pilot image SHA-256 changed: {record.pair_id}")
        return record.pair_id, actual, record.degraded_path.stat().st_size

    digest = hashlib.sha256()
    total_bytes = 0
    with ThreadPoolExecutor(max_workers=8) as executor:
        for index, (pair_id, actual, size) in enumerate(
            executor.map(inspect_record, records), start=1
        ):
            digest.update(f"{pair_id}\0{actual}\0{size}\n".encode())
            total_bytes += size
            if show_progress and (index % 1000 == 0 or index == len(records)):
                print(
                    f"QWEN_PILOT_PREFLIGHT_IMAGES {index}/{len(records)}",
                    flush=True,
                )
    return {
        "records": len(records),
        "unique_files": len(paths),
        "total_bytes": total_bytes,
        "verified_file_bundle_sha256": digest.hexdigest(),
    }


def _rank_assignments(
    splits: dict[str, list[PairRecord]], *, world_size: int
) -> dict[str, dict[str, list[str]]]:
    result: dict[str, dict[str, list[str]]] = {}
    for split in REQUIRED_SPLITS:
        assignments = {
            str(rank): [
                group[0].source_id
                for group in partition_source_groups(
                    splits[split], world_size=world_size, rank=rank
                )
            ]
            for rank in range(world_size)
        }
        assigned = [source for values in assignments.values() for source in values]
        expected = {record.source_id for record in splits[split]}
        if len(assigned) != len(set(assigned)) or set(assigned) != expected:
            raise RuntimeError(f"pilot {split} rank partition is not exact")
        result[split] = assignments
    return result


def validate_static_contract(config_path: Path) -> tuple[dict[str, object], str]:
    config, config_sha, _ = load_pilot_config(config_path)
    validate_pilot_contract(config)
    _prompt_contract(config)
    data = config["data"]
    assert isinstance(data, dict)
    manifest = workspace_path(data.get("manifest"))
    if not manifest.is_file() or sha256(manifest) != data.get("manifest_sha256"):
        raise RuntimeError("pilot manifest SHA-256 changed")
    return config, config_sha


def run_preflight(
    config_path: Path,
    *,
    skip_large_model_hashes: bool,
    write_artifact: bool = True,
    show_progress: bool = True,
) -> dict[str, object]:
    config, config_sha = validate_static_contract(config_path)
    inventory, inventory_sha = load_inventory(config)
    inventory_model = inventory["model"]
    data = config["data"]
    processor_config = config["processor"]
    distributed = config["distributed"]
    runtime = config["runtime"]
    assert isinstance(inventory_model, dict)
    assert all(
        isinstance(section, dict)
        for section in (data, processor_config, distributed, runtime)
    )

    overfit_evidence = _validate_overfit_evidence(config)
    expected_model_hashes = validate_model_file_sizes(inventory)
    if not skip_large_model_hashes:
        _hash_model_files(inventory, expected_model_hashes)
    parameter_groups, tensor_count = inspect_checkpoint_parameters(config, inventory)

    records = load_pair_manifest(workspace_path(data["manifest"]))
    splits = records_by_split(records, config)
    image_evidence = _validate_all_images(splits, show_progress=show_progress)
    world_size = int(distributed["required_world_size"])
    assignments = _rank_assignments(splits, world_size=world_size)

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
        raise RuntimeError("installed Transformers Qwen architecture changed")
    if float(getattr(loaded_config, "attention_dropout", -1.0)) != 0.0:
        raise RuntimeError("Qwen language-model attention dropout is not zero")
    if Qwen2_5_VLForConditionalGeneration.__name__ != config["model"][
        "architecture"
    ]:
        raise RuntimeError("installed Qwen conditional-generation class changed")
    processor = AutoProcessor.from_pretrained(
        model_dir,
        local_files_only=True,
        use_fast=False,
        min_pixels=int(processor_config["min_pixels"]),
        max_pixels=int(processor_config["max_pixels"]),
    )
    processor.tokenizer.padding_side = str(processor_config["padding_side"])
    if type(processor.image_processor).__name__ != "Qwen2VLImageProcessor":
        raise RuntimeError("the frozen slow Qwen image processor was not loaded")

    representatives = complete_source_groups(splits["train"])[0]
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
        raise RuntimeError(f"pilot visual-token budget changed: {visual_tokens}")
    for row, record in enumerate(representatives):
        target_ids = batch["labels"][row]
        decoded = processor.tokenizer.decode(
            target_ids[target_ids != -100],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()
        if decoded != canonical_target(record.level_index):
            raise RuntimeError(
                f"pilot teacher-forced target changed for {record.pair_id}: {decoded!r}"
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
            f"pilot environment changed expected={expected_versions} actual={actual_versions}"
        )

    split_evidence = {
        split: {
            "records": len(values),
            "sources": len({record.source_id for record in values}),
            "per_level": {
                str(level): Counter(record.level_index for record in values)[level]
                for level in range(1, 6)
            },
            "bundle_sha256": data["splits"][split]["bundle_sha256"],
        }
        for split, values in splits.items()
    }
    model_hash_contract_sha = hashlib.sha256(
        json.dumps(expected_model_hashes, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    result: dict[str, object] = {
        "schema_version": 1,
        "scope": "synthetic_lowlight_supervised_vlm_pilot_8k_ddp8_preflight",
        "config_sha256": config_sha,
        "inventory_config_sha256": inventory_sha,
        "manifest_sha256": data["manifest_sha256"],
        "model_revision": inventory_model["revision"],
        "model_files": len(expected_model_hashes),
        "model_hash_contract_sha256": model_hash_contract_sha,
        "large_model_hashes_checked": not skip_large_model_hashes,
        "parameter_groups": parameter_groups,
        "tensor_count": tensor_count,
        "overfit_gate_evidence": overfit_evidence,
        "split_evidence": split_evidence,
        "image_file_evidence": image_evidence,
        "world_size": world_size,
        "rank_source_assignments": assignments,
        "visual_tokens": visual_tokens,
        "assistant_target_tokens": target_counts,
        "processor": type(processor).__name__,
        "image_processor": type(processor.image_processor).__name__,
        "tokenizer": type(processor.tokenizer).__name__,
        "transformers_modeling_source": str(
            inspect.getsourcefile(modeling_qwen2_5_vl)
        ),
        "versions": actual_versions,
        "test_split_content_verified_only": True,
        "test_split_used_for_training_or_selection": False,
        "official_foundir_used": False,
    }
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
    args = parser.parse_args()
    if args.contract_only:
        if args.skip_large_model_hashes:
            parser.error("--contract-only and --skip-large-model-hashes are exclusive")
        _, config_sha = validate_static_contract(args.config)
        print(f"QWEN_PILOT_8K_CONTRACT_PASS config_sha256={config_sha}")
        return
    result = run_preflight(
        args.config,
        skip_large_model_hashes=args.skip_large_model_hashes,
    )
    marker = (
        "QWEN_PILOT_8K_PREFLIGHT_SKIP_MODEL_HASHES_PASS"
        if args.skip_large_model_hashes
        else "QWEN_PILOT_8K_PREFLIGHT_FULL_PASS"
    )
    concise = {
        "config_sha256": result["config_sha256"],
        "manifest_sha256": result["manifest_sha256"],
        "model_revision": result["model_revision"],
        "large_model_hashes_checked": result["large_model_hashes_checked"],
        "model_files": result["model_files"],
        "image_file_evidence": result["image_file_evidence"],
        "split_evidence": result["split_evidence"],
        "world_size": result["world_size"],
        "overfit_gate_evidence": result["overfit_gate_evidence"],
        "versions": result["versions"],
    }
    print(marker + " " + json.dumps(concise, sort_keys=True))


if __name__ == "__main__":
    main()
