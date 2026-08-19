#!/usr/bin/env python3
"""Fail-closed CPU/GPU preflight for frozen-Qwen real Lowlight validation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lowlight_restoration.data import PairRecord, load_pair_manifest  # noqa: E402
from lowlight_restoration.qwen_real_validation import (  # noqa: E402
    PRIMARY_SUBSET,
    STRESS_SUBSET,
    load_real_validation_config,
    validate_real_validation_contract,
)
from lowlight_restoration.qwen_supervised import (  # noqa: E402
    DEGRADATION_QUESTION,
    SYSTEM_PROMPT,
    prompt_text,
    record_bundle_sha256,
)
from smoke.qwen_lowlight_supervised_preflight import (  # noqa: E402
    inspect_checkpoint_parameters,
    load_inventory,
    sha256,
    validate_model_file_sizes,
    workspace_path,
)
from smoke.train_qwen_lowlight_supervised import (  # noqa: E402
    atomic_write_new_json,
    read_json,
)


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def _verify_prompt(config: dict[str, object]) -> None:
    prompt = config["prompt"]
    assert isinstance(prompt, dict)
    expected = {
        "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
        "degradation_question_sha256": hashlib.sha256(
            DEGRADATION_QUESTION.encode()
        ).hexdigest(),
        "target_schema": "project_clean_l1_active_lowlight_l2_l5_v1",
    }
    for key, value in expected.items():
        if prompt.get(key) != value:
            raise RuntimeError(f"frozen real-validation prompt changed: {key}")


def _verify_selected_checkpoint(config: dict[str, object]) -> dict[str, object]:
    model = config["model"]
    assert isinstance(model, dict)
    pilot_config_path = workspace_path(model.get("pilot_config"))
    pilot_summary_path = workspace_path(model.get("pilot_summary"))
    checkpoint_path = workspace_path(model.get("selected_visual_checkpoint"))
    checkpoint_metadata_path = workspace_path(model.get("selected_checkpoint_metadata"))
    required = (
        (pilot_config_path, model.get("pilot_config_sha256")),
        (pilot_summary_path, model.get("pilot_summary_sha256")),
        (checkpoint_path, model.get("selected_visual_sha256")),
        (checkpoint_metadata_path, model.get("selected_checkpoint_metadata_sha256")),
    )
    for path, expected in required:
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"frozen step-600 evidence changed: {path}")

    pilot_config = read_json(pilot_config_path)
    summary = read_json(pilot_summary_path)
    metadata = read_json(checkpoint_metadata_path)
    checks = summary.get("gate_checks")
    if (
        pilot_config.get("status")
        != "approved_for_advisor_route_supervised_vlm_lowlight_8k_ddp8_pilot_only"
        or summary.get("status") != "passed"
        or summary.get("automatic_gate_pass") is not True
        or not isinstance(checks, dict)
        or not checks
        or not all(value is True for value in checks.values())
        or summary.get("config_sha256") != model.get("pilot_config_sha256")
        or summary.get("selected_checkpoint") != "checkpoints/step_000600"
        or summary.get("selected_checkpoint_metadata") != metadata
        or metadata.get("step") != model.get("selected_checkpoint_step")
        or metadata.get("visual_sha256") != model.get("selected_visual_sha256")
        or metadata.get("model_revision") != model.get("model_revision")
    ):
        raise RuntimeError("passing synthetic pilot evidence is inconsistent")
    expected_checkpoint = (
        pilot_summary_path.parent
        / str(summary["selected_checkpoint"])
        / "visual.safetensors"
    ).resolve()
    if expected_checkpoint != checkpoint_path.resolve():
        raise RuntimeError("selected step-600 checkpoint path changed")
    return {
        "pilot_config_sha256": model["pilot_config_sha256"],
        "pilot_summary_sha256": model["pilot_summary_sha256"],
        "checkpoint_metadata_sha256": model["selected_checkpoint_metadata_sha256"],
        "selected_checkpoint_step": metadata["step"],
        "selected_visual_sha256": metadata["visual_sha256"],
        "synthetic_pilot_automatic_gate_pass": True,
    }


def _load_manifest_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise RuntimeError("real-validation manifest has no header")
        rows = list(reader)
    required = {
        "pair_id",
        "project_level",
        "subset",
        "annotation_kind",
        "degraded_path",
        "degraded_sha256",
        "reference_path",
        "reference_sha256",
    }
    if not rows or not required.issubset(reader.fieldnames):
        raise RuntimeError("real-validation manifest schema changed")
    return rows


def _verify_manifest_and_sources(
    config: dict[str, object], *, verify_source_hashes: bool
) -> tuple[list[PairRecord], dict[str, object]]:
    data = config["data"]
    assert isinstance(data, dict)
    manifest_path = workspace_path(data.get("manifest"))
    metadata_path = workspace_path(data.get("metadata"))
    if sha256(manifest_path) != data.get("manifest_sha256"):
        raise RuntimeError("real-validation CSV SHA-256 changed")
    if sha256(metadata_path) != data.get("metadata_sha256"):
        raise RuntimeError("real-validation metadata SHA-256 changed")
    metadata = read_json(metadata_path)
    if (
        metadata.get("status") != "frozen_before_qwen_real_domain_inference"
        or metadata.get("manifest_sha256") != data.get("manifest_sha256")
        or metadata.get("annotation_kind") != data.get("annotation_kind")
        or metadata.get("official_foundir_used") is not False
        or metadata.get("model_predictions_consulted_during_selection") is not False
    ):
        raise RuntimeError("real-validation metadata contract changed")

    rows = _load_manifest_rows(manifest_path)
    records = load_pair_manifest(manifest_path)
    if len(rows) != len(records) or len({row["pair_id"] for row in rows}) != len(rows):
        raise RuntimeError("real-validation record cardinality changed")
    by_pair = {record.pair_id: record for record in records}
    subset_records: dict[str, list[PairRecord]] = {PRIMARY_SUBSET: [], STRESS_SUBSET: []}
    subset_levels: dict[str, Counter[int]] = {
        PRIMARY_SUBSET: Counter(),
        STRESS_SUBSET: Counter(),
    }
    annotation = "reference_based_proxy_not_human_ground_truth"
    for row in rows:
        subset = row["subset"]
        if subset not in subset_records or row["annotation_kind"] != annotation:
            raise RuntimeError(f"bad real-validation annotation/subset: {row['pair_id']}")
        record = by_pair[row["pair_id"]]
        subset_records[subset].append(record)
        subset_levels[subset][record.level_index] += 1
        if row["degraded_sha256"] != record.degraded_sha256:
            raise RuntimeError(f"real-validation degraded hash differs: {record.pair_id}")

    subset_evidence: dict[str, object] = {}
    subset_contracts = data["subsets"]
    assert isinstance(subset_contracts, dict)
    for subset in (PRIMARY_SUBSET, STRESS_SUBSET):
        contract = subset_contracts[subset]
        assert isinstance(contract, dict)
        selected = subset_records[subset]
        realized_counts = {
            str(level): subset_levels[subset][level] for level in sorted(subset_levels[subset])
        }
        if (
            len(selected) != contract.get("records")
            or realized_counts != contract.get("records_per_level")
            or record_bundle_sha256(selected) != contract.get("record_bundle_sha256")
        ):
            raise RuntimeError(f"real-validation subset evidence changed: {subset}")
        subset_evidence[subset] = {
            "records": len(selected),
            "records_per_level": realized_counts,
            "record_bundle_sha256": record_bundle_sha256(selected),
        }

    inventory = metadata.get("source_files")
    if not isinstance(inventory, list) or len(inventory) != metadata.get(
        "source_file_count"
    ):
        raise RuntimeError("real-validation source inventory changed")
    if _canonical_sha256(inventory) != metadata.get("source_file_inventory_sha256"):
        raise RuntimeError("real-validation source inventory digest changed")
    sources = metadata.get("sources")
    if not isinstance(sources, dict):
        raise RuntimeError("real-validation acquisition evidence is missing")
    sice_source = sources.get("sice")
    lolv2_source = sources.get("lolv2")
    if (
        not isinstance(sice_source, dict)
        or sice_source.get("mirror_revision")
        != "1a65a70ad5ae9feef5f8402dfa42347652ce1f2c"
        or sice_source.get("held_out_sources") != 47
        or not isinstance(lolv2_source, dict)
        or lolv2_source.get("held_out_pairs") != 100
    ):
        raise RuntimeError("real-validation acquisition revision/count changed")
    lolv2_archive = workspace_path(lolv2_source.get("archive"))
    if (
        not lolv2_archive.is_file()
        or lolv2_archive.stat().st_size != lolv2_source.get("archive_bytes")
    ):
        raise RuntimeError("official LOL-v2 archive size changed")
    acquisition_evidence: dict[str, object] = {
        "sice_mirror_revision": sice_source["mirror_revision"],
        "sice_held_out_sources": sice_source["held_out_sources"],
        "lolv2_held_out_pairs": lolv2_source["held_out_pairs"],
        "lolv2_archive": lolv2_source["archive"],
        "lolv2_archive_bytes": lolv2_source["archive_bytes"],
        "lolv2_archive_sha256": lolv2_source["archive_sha256"],
        "lolv2_archive_hash_checked": False,
    }
    if verify_source_hashes:
        if sha256(lolv2_archive) != lolv2_source.get("archive_sha256"):
            raise RuntimeError("official LOL-v2 archive SHA-256 changed")
        acquisition_evidence["lolv2_archive_hash_checked"] = True

        def verify(entry: object) -> tuple[str, int]:
            if not isinstance(entry, dict):
                raise RuntimeError("bad real-validation source inventory entry")
            path = workspace_path(entry.get("path"))
            if (
                not path.is_file()
                or path.stat().st_size != entry.get("bytes")
                or sha256(path) != entry.get("sha256")
            ):
                raise RuntimeError(f"real-validation source changed: {path}")
            with Image.open(path) as image:
                image.verify()
            return str(entry["path"]), int(entry["bytes"])

        with ThreadPoolExecutor(max_workers=8) as executor:
            verified = list(executor.map(verify, inventory))
        source_evidence = {
            "hashes_checked": True,
            "files": len(verified),
            "bytes": sum(size for _, size in verified),
            "verified_bundle_sha256": _canonical_sha256(verified),
        }
    else:
        source_evidence = {
            "hashes_checked": False,
            "files": len(inventory),
            "reason": "contract_only",
        }
    return records, {
        "manifest_sha256": data["manifest_sha256"],
        "metadata_sha256": data["metadata_sha256"],
        "annotation_kind": annotation,
        "official_foundir_used": False,
        "model_predictions_consulted_during_selection": False,
        "subsets": subset_evidence,
        "sources": source_evidence,
        "acquisition": acquisition_evidence,
    }


def _processor_dry_run(
    config: dict[str, object], inventory: dict[str, object], records: list[PairRecord]
) -> dict[str, object]:
    from transformers import AutoProcessor

    processor_config = config["processor"]
    inventory_model = inventory["model"]
    assert isinstance(processor_config, dict) and isinstance(inventory_model, dict)
    processor = AutoProcessor.from_pretrained(
        workspace_path(inventory_model["local_path"]),
        local_files_only=True,
        use_fast=False,
        min_pixels=int(processor_config["min_pixels"]),
        max_pixels=int(processor_config["max_pixels"]),
    )
    processor.tokenizer.padding_side = "right"
    chosen: list[PairRecord] = []
    for level in range(1, 6):
        chosen.append(next(record for record in records if record.level_index == level))
    images = []
    for record in chosen:
        with Image.open(record.degraded_path) as image:
            images.append(image.convert("RGB"))
    encoded = processor(
        text=[prompt_text(processor)] * len(images),
        images=images,
        padding=True,
        return_tensors="pt",
    )
    visual_tokens = [
        int(grid.prod().item()) // int(processor.image_processor.merge_size) ** 2
        for grid in encoded.image_grid_thw
    ]
    if any(
        value < int(processor_config["minimum_visual_tokens"])
        or value > int(processor_config["maximum_visual_tokens"])
        for value in visual_tokens
    ):
        raise RuntimeError("real-validation processor visual-token count changed")
    return {
        "records": [record.pair_id for record in chosen],
        "visual_tokens": visual_tokens,
        "input_shape": list(encoded.input_ids.shape),
        "pixel_values_shape": list(encoded.pixel_values.shape),
        "tokenizer_padding_side": processor.tokenizer.padding_side,
    }


def _verify_runtime(config: dict[str, object], *, require_gpu: bool) -> dict[str, object]:
    import torch
    import transformers

    runtime = config["runtime"]
    assert isinstance(runtime, dict)
    versions = {
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "transformers": transformers.__version__,
    }
    expected = {
        "python": runtime["required_python"],
        "pytorch": runtime["required_pytorch"],
        "transformers": runtime["required_transformers"],
    }
    if versions != expected:
        raise RuntimeError(f"real-validation runtime version mismatch: {versions}")
    evidence: dict[str, object] = {
        "versions": versions,
        "cuda_available": torch.cuda.is_available(),
        "visible_cuda_devices": torch.cuda.device_count(),
        "gpu_required": require_gpu,
    }
    if require_gpu:
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != runtime.get(
            "required_cublas_workspace_config"
        ):
            raise RuntimeError("real-validation CUBLAS_WORKSPACE_CONFIG changed")
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("real validation requires exactly one visible CUDA GPU")
        properties = torch.cuda.get_device_properties(0)
        total_gib = properties.total_memory / 1024**3
        if total_gib < float(runtime["minimum_gpu_memory_gib"]):
            raise RuntimeError(f"real-validation GPU memory is too small: {total_gib}")
        evidence.update(
            gpu=torch.cuda.get_device_name(0),
            gpu_total_memory_gib=total_gib,
            cuda=torch.version.cuda,
        )
    return evidence


def run_preflight(
    config_path: Path,
    *,
    contract_only: bool,
    skip_large_model_hashes: bool,
    require_gpu: bool,
    write_artifact: bool,
) -> dict[str, object]:
    config, config_sha, _ = load_real_validation_config(config_path)
    validate_real_validation_contract(config)
    _verify_prompt(config)
    checkpoint_evidence = _verify_selected_checkpoint(config)
    records, data_evidence = _verify_manifest_and_sources(
        config, verify_source_hashes=not contract_only
    )
    inventory, inventory_sha = load_inventory(config)
    expected_model_hashes = validate_model_file_sizes(inventory)
    parameter_groups, tensor_count = inspect_checkpoint_parameters(config, inventory)
    if parameter_groups != config["model"]["expected_parameter_groups"]:
        raise RuntimeError("real-validation model parameter groups changed")
    if not skip_large_model_hashes and not contract_only:
        inventory_model = inventory["model"]
        assert isinstance(inventory_model, dict)
        model_dir = workspace_path(inventory_model["local_path"])

        def verify_model(item: tuple[str, str]) -> str:
            name, expected = item
            if sha256(model_dir / name) != expected:
                raise RuntimeError(f"Qwen model file SHA-256 changed: {name}")
            return name

        with ThreadPoolExecutor(max_workers=4) as executor:
            list(executor.map(verify_model, sorted(expected_model_hashes.items())))
    model_evidence: dict[str, object] = {
        "inventory_config_sha256": inventory_sha,
        "model_revision": inventory["model"]["revision"],
        "model_files": len(expected_model_hashes),
        "large_model_hashes_checked": not skip_large_model_hashes and not contract_only,
        "parameter_groups": parameter_groups,
        "tensor_count": tensor_count,
        **checkpoint_evidence,
    }
    runtime_evidence = _verify_runtime(config, require_gpu=require_gpu)
    processor_evidence = (
        {"skipped": True, "reason": "contract_only"}
        if contract_only
        else _processor_dry_run(config, inventory, records)
    )
    payload: dict[str, object] = {
        "schema_version": 1,
        "status": "passed",
        "scope": "frozen_step600_non_FoundIR_real_Lowlight_validation_preflight",
        "config_sha256": config_sha,
        "contract_only": contract_only,
        "model": model_evidence,
        "data": data_evidence,
        "processor": processor_evidence,
        "runtime": runtime_evidence,
    }
    if write_artifact:
        output_dir = workspace_path(config["runtime"]["output_dir"])
        name = (
            "preflight_contract_only.json"
            if contract_only
            else "preflight_full.json"
            if not skip_large_model_hashes and require_gpu
            else "preflight_cpu.json"
            if skip_large_model_hashes
            else "preflight_cpu_full_model_hashes.json"
        )
        atomic_write_new_json(output_dir / name, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--contract-only", action="store_true")
    parser.add_argument("--skip-large-model-hashes", action="store_true")
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--no-write-artifact", action="store_true")
    args = parser.parse_args()
    payload = run_preflight(
        args.config,
        contract_only=args.contract_only,
        skip_large_model_hashes=args.skip_large_model_hashes,
        require_gpu=args.require_gpu,
        write_artifact=not args.no_write_artifact,
    )
    print(
        "QWEN_REAL_VALIDATION_PREFLIGHT_PASS",
        json.dumps(payload, sort_keys=True),
        flush=True,
    )


if __name__ == "__main__":
    main()
