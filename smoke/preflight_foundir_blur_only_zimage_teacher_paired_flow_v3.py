#!/usr/bin/env python3
"""Fail-closed preflight for the solver-aligned paired-flow v3 continuation."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party/diffsynth-studio"))

from lowlight_restoration.foundir_zimage_paired_flow_v3 import (  # noqa: E402
    load_config,
    load_frozen_rows,
    parent_checkpoint,
    sample_assignment,
    sha256,
    validate_gpu_inventory,
    verify_model_artifacts,
    write_once,
)
from smoke.preflight_foundir_blur_only_zimage_teacher_paired_flow import (  # noqa: E402
    validate_cuda_attention,
)


def git_revision(path: Path) -> str:
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--skip-large-hashes", action="store_true")
    parser.add_argument("--skip-model-artifacts", action="store_true")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--write-artifact", action="store_true")
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = load_config(config_path)
    train, visual = load_frozen_rows(config, verify_images=True)
    revisions = {
        "z_image": git_revision(ROOT / "third_party/z-image"),
        "diffsynth": git_revision(ROOT / "third_party/diffsynth-studio"),
    }
    if revisions["z_image"] != config["implementation"]["z_image_revision"]:
        raise RuntimeError("official Z-Image source revision changed")
    if revisions["diffsynth"] != config["implementation"]["diffsynth_revision"]:
        raise RuntimeError("DiffSynth-Studio source revision changed")
    artifacts = {"skipped": True, "controlnet_loaded": False}
    if not args.skip_model_artifacts:
        artifacts = verify_model_artifacts(config, large_hashes=not args.skip_large_hashes)
    parent = parent_checkpoint(config, verify_hash=True)

    training = config["training"]
    total_presentations = int(training["additional_steps"]) * int(config["distributed"]["required_world_size"])
    coverage = {
        (sample_assignment(index, len(train), int(training["inference_grid_steps"])))
        for index in range(total_presentations)
    }
    expected_pairs = len(train) * int(training["inference_grid_steps"])
    if len(coverage) != expected_pairs:
        raise RuntimeError("v3 presentation schedule does not cover every record/timestep pair")
    presentations_per_pair = total_presentations // expected_pairs
    if total_presentations % expected_pairs or presentations_per_pair != 8:
        raise RuntimeError("v3 record/timestep presentation balance changed")

    hardware = None
    attention_backend = None
    if args.require_cuda:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for the runtime preflight")
        names = [torch.cuda.get_device_properties(index).name for index in range(torch.cuda.device_count())]
        memory = [
            torch.cuda.get_device_properties(index).total_memory // (1024 * 1024)
            for index in range(torch.cuda.device_count())
        ]
        hardware = validate_gpu_inventory(names, memory, config)
        attention_backend = validate_cuda_attention(torch)
    result = {
        "schema_version": 3,
        "status": "foundir_blur_only_zimage_teacher_paired_flow_v3_preflight_pass",
        "config": str(config_path.relative_to(ROOT)),
        "config_sha256": sha256(config_path),
        "training_records": len(train),
        "visual_development_records": len(visual),
        "train_visual_source_group_overlap": 0,
        "initial_checkpoint": str(parent.relative_to(ROOT)),
        "initial_checkpoint_sha256": config["training"]["initial_checkpoint_sha256"],
        "initial_step": int(training["initial_step"]),
        "optimizer_resume": False,
        "additional_steps": int(training["additional_steps"]),
        "final_cumulative_step": int(training["final_cumulative_step"]),
        "timestep_sampling": training["timestep_sampling"],
        "loss_weighting": training["loss_weighting"],
        "record_timestep_pairs": expected_pairs,
        "presentations_per_record_timestep_pair": presentations_per_pair,
        "scheduler_shift": float(training["scheduler_shift"]),
        "cfg_scale": float(config["inference"]["cfg_scale"]),
        "initialization": config["inference"]["initialization"],
        "revisions": revisions,
        "model_artifacts": artifacts,
        "controlnet_loaded": False,
        "hardware_profile": hardware,
        "attention_backend": attention_backend,
        "sealed_internal_accessed": False,
        "official_foundir_test_accessed": False,
    }
    if args.write_artifact:
        output = ROOT / str(config["runtime"]["output_dir"]) / "preflight.json"
        write_once(output, (json.dumps(result, indent=2, sort_keys=True) + "\n").encode())
    print("FOUNDIR_ZIMAGE_PAIRED_FLOW_V3_PREFLIGHT_PASS " + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
