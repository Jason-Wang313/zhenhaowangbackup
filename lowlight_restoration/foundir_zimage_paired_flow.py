"""Contracts and math for the Blur-only Z-Image paired-latent-flow teacher."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

import torch

from lowlight_restoration.foundir_zimage import (
    ROOT,
    load_aligned_pair,
    load_frozen_rows as _load_frozen_rows_v1,
    sha256,
    write_once,
)


EXPECTED_STATUS = "advisor_authorized_zimage_blur_teacher_paired_flow_tiny_overfit"
EXPECTED_HOLD_ID = "zimage_blur_teacher_tiny_overfit-a6becc1eff058fec"
EXPECTED_CONDITIONING = "base_native_paired_lq_to_gt_latent_rectified_flow"


def paired_flow_state(gt_latent: torch.Tensor, lq_latent: torch.Tensor, sigma: torch.Tensor | float) -> torch.Tensor:
    """Return x_sigma=(1-sigma)*GT+sigma*LQ on the paired restoration path."""
    return (1 - sigma) * gt_latent + sigma * lq_latent


def paired_flow_target(gt_latent: torch.Tensor, lq_latent: torch.Tensor) -> torch.Tensor:
    """Velocity dx/dsigma used by the descending-sigma Euler solver."""
    return lq_latent - gt_latent


def paired_flow_euler_step(
    sample: torch.Tensor,
    velocity: torch.Tensor,
    sigma: torch.Tensor | float,
    next_sigma: torch.Tensor | float,
) -> torch.Tensor:
    return sample + velocity * (next_sigma - sigma)


def load_config(path: Path) -> dict[str, object]:
    config = json.loads(path.read_text())
    validate_config(config)
    config["_config_path"] = str(path.resolve())
    return config


def validate_config(config: Mapping[str, object]) -> None:
    if config.get("schema_version") != 2 or config.get("status") != EXPECTED_STATUS:
        raise RuntimeError("config is not the authorized paired-flow Z-Image teacher gate")
    scope = config["scope"]
    if scope.get("category") != "01Blur" or scope.get("teacher_only") is not True:
        raise RuntimeError("paired-flow teacher must remain isolated Blur only")
    forbidden = (
        "distillation",
        "decoder_finetuning",
        "qwen_degradation_conditioning",
        "mixed_degradations",
        "sealed_internal_accessed",
        "official_foundir_test_accessed",
    )
    if any(scope.get(key) is not False for key in forbidden):
        raise RuntimeError("a forbidden route or protected split is enabled")
    auth = config["authorization"]
    if auth.get("hold_id") != EXPECTED_HOLD_ID or auth.get("decision") != "reports/advisor_governance/advisor_decision.json":
        raise RuntimeError("corrected teacher authorization identity changed")
    model = config["model"]
    if model.get("name") != "Tongyi-MAI/Z-Image" or model.get("license") != "apache-2.0":
        raise RuntimeError("the undistilled official Z-Image Base foundation is required")
    if "controlnet" in model:
        raise RuntimeError("paired-flow v2 must not load the Turbo-only ControlNet")
    impl = config["implementation"]
    if impl.get("conditioning") != EXPECTED_CONDITIONING or impl.get("controlnet") != "disabled":
        raise RuntimeError("paired-flow conditioning contract changed")
    training = config["training"]
    if training.get("objective") != "paired_lq_to_gt_latent_rectified_flow":
        raise RuntimeError("paired-flow training objective changed")
    if float(training.get("scheduler_shift", -1)) != 6.0:
        raise RuntimeError("official Z-Image Base static shift 6 is required")
    distributed = config["distributed"]
    if int(distributed.get("required_world_size", 0)) != 8:
        raise RuntimeError("teacher gate requires eight ranks")
    effective = (
        int(distributed["required_world_size"])
        * int(training["micro_batch_size_per_rank"])
        * int(training["gradient_accumulation_steps"])
    )
    if effective != int(training["global_batch_size"]):
        raise RuntimeError("global batch arithmetic changed")
    if int(training["max_steps"]) != 400 or training["save_steps"] != [200, 400]:
        raise RuntimeError("bounded tiny-overfit schedule changed")
    inference = config["inference"]
    if int(inference["teacher_steps"]) != 50 or float(inference["cfg_scale"]) != 1.0:
        raise RuntimeError("paired-flow teacher requires 50 steps and CFG 1")
    if float(inference.get("scheduler_shift", -1)) != 6.0:
        raise RuntimeError("training/inference scheduler shift mismatch")
    if inference.get("initialization") != "exact_lq_vae_latent_at_sigma_1":
        raise RuntimeError("inference must begin from the exact LQ latent")
    if inference["evaluate_steps"] != [0, 400]:
        raise RuntimeError("tiny evaluation must compare initialization and final checkpoint")


def load_frozen_rows(config: Mapping[str, object], *, verify_images: bool):
    """Reuse the immutable v1 crop manifest after validating its exact identity."""
    return _load_frozen_rows_v1(config, verify_images=verify_images)


def model_paths(config: Mapping[str, object]) -> dict[str, object]:
    model = config["model"]
    root = ROOT / str(model["root"])
    return {
        "transformer": sorted(root.glob(str(model["transformer_glob"]))),
        "text_encoder": sorted(root.glob(str(model["text_encoder_glob"]))),
        "vae": root / str(model["vae"]),
        "tokenizer": root / str(model["tokenizer"]),
    }


def verify_model_artifacts(config: Mapping[str, object], *, large_hashes: bool) -> dict[str, object]:
    paths = model_paths(config)
    if len(paths["transformer"]) != 2 or len(paths["text_encoder"]) != 3:
        raise RuntimeError("Z-Image Base shard count is incomplete")
    required = [*paths["transformer"], *paths["text_encoder"], paths["vae"]]
    if not paths["tokenizer"].is_dir() or any(not path.is_file() for path in required):
        raise RuntimeError("Z-Image Base runtime artifacts are incomplete")
    checked: dict[str, str] = {}
    if large_hashes:
        for relative, expected in config["model"]["expected_blob_sha256"].items():
            path = ROOT / str(config["model"]["root"]) / relative
            actual = sha256(path)
            if actual != expected:
                raise RuntimeError(f"Z-Image Base blob identity changed: {relative}")
            checked[relative] = actual
    return {"files": len(required), "large_hashes": large_hashes, "sha256": checked, "controlnet_loaded": False}


def validate_gpu_inventory(names: Sequence[str], memory_mib: Sequence[int], config: Mapping[str, object]) -> str:
    required = int(config["distributed"]["required_world_size"])
    if len(names) != required or len(memory_mib) != required:
        raise RuntimeError("exactly eight visible GPUs are required")
    families = []
    for name in names:
        matches = [fragment for fragment in config["distributed"]["allowed_gpu_name_fragments"] if fragment in name]
        if len(matches) != 1:
            raise RuntimeError(f"unsupported GPU: {name}")
        families.append(matches[0])
    if len(set(families)) != 1:
        raise RuntimeError("mixed H100/H200 ranks are not supported")
    minimum = int(config["distributed"]["minimum_memory_gib"]) * 1024
    if any(value < minimum for value in memory_mib):
        raise RuntimeError("visible GPU memory is below the frozen minimum")
    return families[0]


__all__ = [
    "ROOT",
    "load_aligned_pair",
    "load_config",
    "load_frozen_rows",
    "model_paths",
    "paired_flow_euler_step",
    "paired_flow_state",
    "paired_flow_target",
    "sha256",
    "validate_config",
    "validate_gpu_inventory",
    "verify_model_artifacts",
    "write_once",
]
