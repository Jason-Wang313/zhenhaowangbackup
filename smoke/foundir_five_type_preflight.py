#!/usr/bin/env python3
"""Fail-closed CPU/GPU preflight for the five-type FoundIR FLUX.2 phase."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lowlight_restoration.foundir_five_type import category_counts, load_five_type_metadata
from lowlight_restoration.foundir_flux2_protocol import SOW_STAGE2_THRESHOLDS


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def workspace_path(value: str) -> Path:
    path = (ROOT / value).resolve()
    path.relative_to(ROOT.resolve())
    return path


def run(config_path: Path, *, require_data: bool, skip_large_hashes: bool) -> dict[str, object]:
    config = json.loads(config_path.read_text())
    if config.get("status") != "advisor_approved_five_type_foundir_dit_readiness":
        raise RuntimeError("five-type DiT config is not approved")
    data = config["data"]
    metadata = workspace_path(data["official_metadata"])
    if sha256(metadata) != data["official_metadata_sha256"]:
        raise RuntimeError("official FoundIR training metadata changed")
    records, all_counts = load_five_type_metadata(metadata)
    if sum(all_counts.values()) != data["expected_official_records"]:
        raise RuntimeError("official FoundIR metadata cardinality changed")
    observed = category_counts(records)
    if len(records) != data["expected_five_type_records"] or observed != data["expected_categories"]:
        raise RuntimeError("five-type category contract changed")

    for section, path_key, hash_key in (
        (config["qwen_condition"], "training_config", "training_config_sha256"),
        (config["qwen_condition"], "adapter", "adapter_sha256"),
        (config["qwen_condition"], "validation_result", "validation_result_sha256"),
        (config["model"], "flow_weights", "flow_weights_sha256"),
        (config["model"], "autoencoder_weights", "autoencoder_weights_sha256"),
    ):
        path = workspace_path(section[path_key])
        if not path.is_file():
            raise FileNotFoundError(path)
        if not skip_large_hashes or path.stat().st_size < 1024**3:
            if sha256(path) != section[hash_key]:
                raise RuntimeError(f"frozen artifact changed: {path}")

    qwen_condition = config["qwen_condition"]
    for path_key, hash_key in (
        ("foundir_training_diagnostic", "foundir_training_diagnostic_sha256"),
        ("foundir_training_views", "foundir_training_views_sha256"),
    ):
        if path_key in qwen_condition:
            path = workspace_path(qwen_condition[path_key])
            if not path.is_file() or sha256(path) != qwen_condition[hash_key]:
                raise RuntimeError(f"frozen Qwen training-only diagnostic changed: {path}")

    prompt_cache = config["prompt_cache"]
    for name, expected in prompt_cache["qwen3_weight_sha256"].items():
        path = workspace_path(str(Path(config["model"]["text_encoder"]) / name))
        if not path.is_file() or (not skip_large_hashes and sha256(path) != expected):
            raise RuntimeError(f"frozen FLUX Qwen3 text-encoder weight changed: {path}")
    baseline = config["released_baseline"]
    for path_key, hash_key in (
        ("generalist", "generalist_sha256"),
        ("lowlight_specialist", "lowlight_specialist_sha256"),
        ("weather_specialist", "weather_specialist_sha256"),
    ):
        path = workspace_path(baseline[path_key])
        if not path.is_file() or (not skip_large_hashes and sha256(path) != baseline[hash_key]):
            raise RuntimeError(f"released FoundIR baseline changed: {path}")

    distributed = config["distributed"]
    flow = config["flow_training"]
    if (
        distributed.get("required_world_size") != 8
        or distributed.get("micro_batch_size_per_rank") != 1
        or int(distributed.get("global_batch_size", -1))
        != 8 * int(distributed.get("gradient_accumulation_steps", -1))
        or config["lora"].get("rank") != 32
        or float(config["lora"].get("alpha", -1)) != 32.0
        or flow.get("objective") != "degraded_anchored_residual_flow"
        or not 0.0 <= float(flow.get("anchor_noise_scale", -1)) <= 0.25
    ):
        raise RuntimeError("eight-GPU FLUX training contract changed")
    full = config["full_training"]
    curriculum = full.get("curriculum")
    hardware = full.get("hardware_schedule")
    total_steps = sum(int(stage.get("presentations", 0)) for stage in curriculum or []) // int(distributed["global_batch_size"])
    stage_end_steps = []
    cumulative_presentations = 0
    for stage in curriculum or []:
        cumulative_presentations += int(stage.get("presentations", 0))
        stage_end_steps.append(cumulative_presentations // int(distributed["global_batch_size"]))
    candidate_steps = config["evaluation"].get("candidate_steps")
    no_ablation_v2 = config["qwen_condition"].get("method") == "per_type_dialogue_center_crop_448"
    if no_ablation_v2 and (
        config["scope"].get("exclude_jpeg") is not True
        or config["evaluation"].get("condition_arms") != ["qwen"]
        or float(full.get("generic_condition_probability", -1)) != 0.0
    ):
        raise RuntimeError("physical-v8 route must remain no-JPEG and no-ablation")
    if (
        not isinstance(curriculum, list)
        or [stage.get("maximum_composition_size") for stage in curriculum] != [1, 2, 3]
        or any(int(stage.get("presentations", 0)) % int(distributed["global_batch_size"]) for stage in curriculum)
        or set(hardware) != {"H100", "H200"}
        or any(
            8 * int(schedule["micro_batch_size_per_rank"]) * int(schedule["gradient_accumulation_steps"])
            != int(distributed["global_batch_size"])
            for schedule in hardware.values()
        )
        or not 0.0 <= float(full.get("identity_pair_probability", -1)) <= 0.1
        or not 0.0 <= float(full.get("generic_condition_probability", -1)) <= 0.1
        or config["evaluation"].get("sow_stage2_thresholds") != SOW_STAGE2_THRESHOLDS
        or not isinstance(candidate_steps, list)
        or candidate_steps != sorted(set(candidate_steps))
        or any(
            int(step) <= 0
            or int(step) > total_steps
            or (
                int(step) not in stage_end_steps
                and int(step) % int(full["checkpoint_every_steps"]) != 0
            )
            for step in candidate_steps
        )
    ):
        raise RuntimeError("full curriculum/evaluation contract changed")

    training_root = workspace_path(data["training_root"])
    missing_examples: list[str] = []
    if training_root.is_dir() and require_data:
        for record in records:
            for relative in (record.gt_relative, record.lq_relative):
                path = training_root / relative
                if not path.is_file():
                    missing_examples.append(relative)
                    if len(missing_examples) >= 20:
                        break
            if len(missing_examples) >= 20:
                break
    elif require_data:
        raise FileNotFoundError(
            "FoundIR official training root is missing: "
            f"{training_root}. Obtain the authorized ~4.85 TB training archive "
            "from the official FoundIR request form and extract it at this exact path."
        )
    if require_data and missing_examples:
        raise FileNotFoundError(f"FoundIR training pairs are incomplete: {missing_examples}")

    result = {
        "status": "pass" if training_root.is_dir() and not missing_examples else "metadata_only_pass_data_missing",
        "official_metadata_records": sum(all_counts.values()),
        "five_type_records": len(records),
        "category_counts": observed,
        "training_root": str(training_root),
        "training_root_present": training_root.is_dir(),
        "qwen_method": config["qwen_condition"]["method"],
        "lora_rank": config["lora"]["rank"],
        "lora_alpha": config["lora"]["alpha"],
        "protected_foundir_test_accessed": False,
    }
    print("FOUNDIR_FIVE_TYPE_PREFLIGHT " + json.dumps(result, sort_keys=True), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/foundir_five_type_flux2_dit_v1.json",
    )
    parser.add_argument("--require-data", action="store_true")
    parser.add_argument("--skip-large-hashes", action="store_true")
    args = parser.parse_args()
    run(args.config, require_data=args.require_data, skip_large_hashes=args.skip_large_hashes)


if __name__ == "__main__":
    main()
