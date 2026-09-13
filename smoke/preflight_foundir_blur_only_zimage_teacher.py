#!/usr/bin/env python3
"""Fail-closed preflight for the advisor-approved Z-Image Blur teacher gate."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lowlight_restoration.foundir_zimage import (  # noqa: E402
    load_config,
    load_frozen_rows,
    sha256,
    validate_gpu_inventory,
    verify_model_artifacts,
    write_once,
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

    artifacts = {"skipped": True}
    if not args.skip_model_artifacts:
        artifacts = verify_model_artifacts(config, large_hashes=not args.skip_large_hashes)

    hardware = None
    if args.require_cuda:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for the runtime preflight")
        names = [torch.cuda.get_device_properties(index).name for index in range(torch.cuda.device_count())]
        memory = [torch.cuda.get_device_properties(index).total_memory // (1024 * 1024) for index in range(torch.cuda.device_count())]
        hardware = validate_gpu_inventory(names, memory, config)

    result = {
        "schema_version": 1,
        "status": "foundir_blur_only_zimage_teacher_preflight_pass",
        "config": str(config_path.relative_to(ROOT)),
        "config_sha256": sha256(config_path),
        "training_records": len(train),
        "visual_development_records": len(visual),
        "train_visual_source_group_overlap": 0,
        "revisions": revisions,
        "model_artifacts": artifacts,
        "hardware_profile": hardware,
        "sealed_internal_accessed": False,
        "official_foundir_test_accessed": False,
    }
    if args.write_artifact:
        output = ROOT / str(config["runtime"]["output_dir"]) / "preflight.json"
        write_once(output, (json.dumps(result, indent=2, sort_keys=True) + "\n").encode())
    print("FOUNDIR_ZIMAGE_TEACHER_PREFLIGHT_PASS " + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
