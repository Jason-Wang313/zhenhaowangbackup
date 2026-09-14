"""Contract for the fixed Blur-only Z-Image VAE ceiling diagnostic."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

from lowlight_restoration.foundir_zimage_paired_flow import (
    ROOT,
    load_aligned_pair,
    load_frozen_rows,
    model_paths,
    sha256,
    validate_gpu_inventory,
    verify_model_artifacts,
    write_once,
)


EXPECTED_STATUS = "advisor_authorized_zimage_blur_teacher_vae_ceiling_and_solver_sweep"
EXPECTED_HOLD_ID = "zimage_blur_teacher_paired_flow_tiny_overfit_v3-b6567614820dbc4e"


def load_config(path: Path) -> dict[str, object]:
    config = json.loads(path.read_text())
    validate_config(config)
    config["_config_path"] = str(path.resolve())
    return config


def validate_config(config: Mapping[str, object]) -> None:
    if config.get("schema_version") != 1 or config.get("status") != EXPECTED_STATUS:
        raise RuntimeError("config is not the authorized Z-Image ceiling and solver diagnostic")
    scope = config["scope"]
    if scope.get("category") != "01Blur" or scope.get("diagnostic_only") is not True:
        raise RuntimeError("VAE ceiling must remain a Blur-only diagnostic")
    forbidden = (
        "training",
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
        raise RuntimeError("VAE ceiling authorization identity changed")
    if auth.get("decision") != "reports/advisor_governance/advisor_decision.json":
        raise RuntimeError("VAE ceiling decision path changed")
    model = config["model"]
    if model.get("name") != "Tongyi-MAI/Z-Image":
        raise RuntimeError("unexpected Z-Image foundation")
    teacher = config["teacher"]
    if int(teacher.get("cumulative_step", 0)) != 1200:
        raise RuntimeError("solver sweep must use the terminal v3 teacher")
    if teacher.get("initialization") != "exact_lq_vae_latent_at_sigma_1":
        raise RuntimeError("solver sweep must initialize from exact LQ latent")
    if float(teacher.get("scheduler_shift", -1)) != 6.0 or float(teacher.get("cfg_scale", -1)) != 1.0:
        raise RuntimeError("solver sweep must preserve shift 6 and CFG 1")
    evaluation = config["evaluation"]
    required = (
        "measure_lq_roundtrip",
        "measure_gt_roundtrip",
        "require_gt_roundtrip_psnr_above_input",
        "require_gt_roundtrip_lpips_below_input",
        "save_fixed_triptychs",
    )
    if any(evaluation.get(key) is not True for key in required):
        raise RuntimeError("VAE ceiling measurements or gates changed")
    if evaluation.get("teacher_inference_steps") != [1, 2, 4, 8, 16, 50]:
        raise RuntimeError("teacher solver sweep changed")
    if int(config["distributed"].get("required_world_size", 0)) != 8:
        raise RuntimeError("VAE ceiling requires eight ranks")


def teacher_checkpoint(config: Mapping[str, object], *, verify_hash: bool) -> Path:
    path = ROOT / str(config["teacher"]["checkpoint"])
    if not path.is_file():
        raise RuntimeError("v3 teacher checkpoint is missing")
    if verify_hash and sha256(path) != config["teacher"]["checkpoint_sha256"]:
        raise RuntimeError("v3 teacher checkpoint identity changed")
    return path


__all__ = [
    "ROOT",
    "load_aligned_pair",
    "load_config",
    "load_frozen_rows",
    "model_paths",
    "sha256",
    "teacher_checkpoint",
    "validate_config",
    "validate_gpu_inventory",
    "verify_model_artifacts",
    "write_once",
]
