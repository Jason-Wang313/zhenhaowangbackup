#!/usr/bin/env python3
"""Fail-closed preflight for the VAE ceiling and solver-step diagnostic."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party/diffsynth-studio"))

from lowlight_restoration.foundir_zimage_vae_ceiling import (  # noqa: E402
    load_config,
    load_frozen_rows,
    sha256,
    teacher_checkpoint,
    validate_gpu_inventory,
    verify_model_artifacts,
)
from smoke.preflight_foundir_blur_only_zimage_teacher_paired_flow import (  # noqa: E402
    validate_cuda_attention,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--skip-large-hashes", action="store_true")
    parser.add_argument("--skip-model-artifacts", action="store_true")
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = load_config(config_path)
    train, visual = load_frozen_rows(config, verify_images=True)
    artifacts = {"skipped": True, "controlnet_loaded": False}
    if not args.skip_model_artifacts:
        artifacts = verify_model_artifacts(config, large_hashes=not args.skip_large_hashes)
    checkpoint = teacher_checkpoint(config, verify_hash=True)
    hardware = None
    attention_backend = None
    if args.require_cuda:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for the diagnostic")
        names = [torch.cuda.get_device_properties(index).name for index in range(torch.cuda.device_count())]
        memory = [
            torch.cuda.get_device_properties(index).total_memory // (1024 * 1024)
            for index in range(torch.cuda.device_count())
        ]
        hardware = validate_gpu_inventory(names, memory, config)
        attention_backend = validate_cuda_attention(torch)
    result = {
        "schema_version": 1,
        "status": "zimage_blur_vae_ceiling_solver_sweep_preflight_pass",
        "config": str(config_path.relative_to(ROOT)),
        "config_sha256": sha256(config_path),
        "teacher_checkpoint": str(checkpoint.relative_to(ROOT)),
        "teacher_checkpoint_sha256": config["teacher"]["checkpoint_sha256"],
        "training_records": len(train),
        "visual_development_records": len(visual),
        "teacher_inference_steps": config["evaluation"]["teacher_inference_steps"],
        "model_artifacts": artifacts,
        "hardware_profile": hardware,
        "attention_backend": attention_backend,
        "decoder_finetuned": False,
        "distillation_performed": False,
        "sealed_internal_accessed": False,
        "official_foundir_test_accessed": False,
    }
    print("FOUNDIR_ZIMAGE_VAE_CEILING_SOLVER_SWEEP_PREFLIGHT_PASS " + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
