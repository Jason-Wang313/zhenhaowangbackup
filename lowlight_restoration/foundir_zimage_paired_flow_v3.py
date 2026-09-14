"""Contracts for the solver-aligned Z-Image paired-flow teacher continuation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import torch
from safetensors import safe_open

from lowlight_restoration.foundir_zimage_paired_flow import (
    ROOT,
    load_aligned_pair,
    load_frozen_rows,
    model_paths,
    paired_flow_euler_step,
    paired_flow_state,
    paired_flow_target,
    sha256,
    validate_gpu_inventory,
    verify_model_artifacts,
    write_once,
)


EXPECTED_STATUS = "advisor_authorized_zimage_blur_teacher_paired_flow_continuation_v3"
EXPECTED_HOLD_ID = "zimage_blur_teacher_paired_flow_tiny_overfit_v2-ea895f0c24c75257"
EXPECTED_CONDITIONING = "base_native_paired_lq_to_gt_latent_rectified_flow"
EXPECTED_TIMESTEP_SAMPLING = "uniform_exact_50_step_shift_6_inference_grid"
EXPECTED_LOSS_WEIGHTING = "plain_unweighted_velocity_mse"


def load_config(path: Path) -> dict[str, object]:
    config = json.loads(path.read_text())
    validate_config(config)
    config["_config_path"] = str(path.resolve())
    return config


def validate_config(config: Mapping[str, object]) -> None:
    if config.get("schema_version") != 3 or config.get("status") != EXPECTED_STATUS:
        raise RuntimeError("config is not the authorized v3 paired-flow continuation")
    scope = config["scope"]
    if scope.get("category") != "01Blur" or scope.get("teacher_only") is not True:
        raise RuntimeError("v3 teacher continuation must remain isolated Blur only")
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
    if auth.get("hold_id") != EXPECTED_HOLD_ID:
        raise RuntimeError("v3 continuation authorization identity changed")
    if auth.get("decision") != "reports/advisor_governance/advisor_decision.json":
        raise RuntimeError("v3 continuation decision path changed")
    model = config["model"]
    if model.get("name") != "Tongyi-MAI/Z-Image" or model.get("license") != "apache-2.0":
        raise RuntimeError("the undistilled official Z-Image Base foundation is required")
    if "controlnet" in model:
        raise RuntimeError("v3 must not load the Turbo-only ControlNet")
    implementation = config["implementation"]
    if implementation.get("conditioning") != EXPECTED_CONDITIONING:
        raise RuntimeError("paired-flow conditioning contract changed")
    if implementation.get("controlnet") != "disabled":
        raise RuntimeError("v3 ControlNet must remain disabled")
    training = config["training"]
    if training.get("objective") != "paired_lq_to_gt_latent_rectified_flow":
        raise RuntimeError("paired-flow objective changed")
    if training.get("timestep_sampling") != EXPECTED_TIMESTEP_SAMPLING:
        raise RuntimeError("training must use the exact inference grid")
    if training.get("loss_weighting") != EXPECTED_LOSS_WEIGHTING:
        raise RuntimeError("empirical T2I timestep weighting is forbidden in v3")
    if int(training.get("inference_grid_steps", 0)) != 50:
        raise RuntimeError("v3 training grid must have 50 steps")
    if float(training.get("scheduler_shift", -1)) != 6.0:
        raise RuntimeError("official Z-Image Base static shift 6 is required")
    if training.get("optimizer_resume") is not False:
        raise RuntimeError("v3 must explicitly record optimizer reset")
    if int(training.get("initial_step", -1)) != 400:
        raise RuntimeError("v3 must initialize at cumulative step 400")
    if int(training.get("additional_steps", -1)) != 800:
        raise RuntimeError("v3 bounded continuation must run 800 additional steps")
    if int(training.get("final_cumulative_step", -1)) != 1200:
        raise RuntimeError("v3 final cumulative step must be 1200")
    if training.get("save_cumulative_steps") != [800, 1200]:
        raise RuntimeError("v3 cumulative checkpoint schedule changed")
    distributed = config["distributed"]
    if int(distributed.get("required_world_size", 0)) != 8:
        raise RuntimeError("v3 teacher gate requires eight ranks")
    effective = (
        int(distributed["required_world_size"])
        * int(training["micro_batch_size_per_rank"])
        * int(training["gradient_accumulation_steps"])
    )
    if effective != int(training["global_batch_size"]):
        raise RuntimeError("global batch arithmetic changed")
    inference = config["inference"]
    if int(inference.get("teacher_steps", 0)) != 50 or float(inference.get("cfg_scale", -1)) != 1.0:
        raise RuntimeError("v3 teacher requires 50 steps and CFG 1")
    if float(inference.get("scheduler_shift", -1)) != 6.0:
        raise RuntimeError("training/inference scheduler shift mismatch")
    if inference.get("initialization") != "exact_lq_vae_latent_at_sigma_1":
        raise RuntimeError("inference must begin from the exact LQ latent")
    if inference.get("evaluate_cumulative_steps") != [400, 800, 1200]:
        raise RuntimeError("v3 must compare parent, midpoint, and final checkpoints")


def parent_checkpoint(config: Mapping[str, object], *, verify_hash: bool) -> Path:
    parent = ROOT / str(config["training"]["initial_checkpoint"])
    if not parent.is_file():
        raise RuntimeError(f"missing immutable v2 parent checkpoint: {parent}")
    expected = str(config["training"]["initial_checkpoint_sha256"])
    if verify_hash and sha256(parent) != expected:
        raise RuntimeError("v2 parent checkpoint identity changed")
    with safe_open(parent, framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
        keys = list(handle.keys())
    if metadata.get("step") != "400" or len(keys) != 476:
        raise RuntimeError("v2 parent checkpoint metadata or LoRA tensor count changed")
    return parent


def sample_assignment(index: int, record_count: int, grid_steps: int) -> tuple[int, int]:
    """Map presentations so every record sees every solver timestep evenly."""
    if index < 0 or record_count <= 0 or grid_steps <= 0:
        raise ValueError("invalid solver-aligned sample assignment")
    return index % record_count, (index // record_count) % grid_steps


def plain_velocity_mse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Solver-aligned v3 loss with no empirical text-to-image timestep weight."""
    return torch.nn.functional.mse_loss(prediction.float(), target.float())


def evaluation_checkpoint(config: Mapping[str, object], cumulative_step: int) -> Path:
    if cumulative_step == int(config["training"]["initial_step"]):
        return ROOT / str(config["training"]["initial_checkpoint"])
    if cumulative_step not in config["training"]["save_cumulative_steps"]:
        raise RuntimeError(f"unregistered v3 evaluation step: {cumulative_step}")
    return ROOT / str(config["runtime"]["output_dir"]) / f"step_{cumulative_step:06d}.safetensors"


__all__ = [
    "ROOT",
    "evaluation_checkpoint",
    "load_aligned_pair",
    "load_config",
    "load_frozen_rows",
    "model_paths",
    "paired_flow_euler_step",
    "paired_flow_state",
    "paired_flow_target",
    "parent_checkpoint",
    "plain_velocity_mse",
    "sample_assignment",
    "sha256",
    "validate_config",
    "validate_gpu_inventory",
    "verify_model_artifacts",
    "write_once",
]
