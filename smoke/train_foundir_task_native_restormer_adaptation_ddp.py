#!/usr/bin/env python3
"""Source-group-balanced FoundIR adaptation of a pinned task-native Restormer."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
from collections import Counter, defaultdict
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image
from torch.nn.parallel import DistributedDataParallel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lowlight_restoration.flux_prompt_cache import file_sha256  # noqa: E402
from lowlight_restoration.foundir_flux2_protocol import (  # noqa: E402
    resolve_workspace_path,
    stable_digest,
)
from lowlight_restoration.task_native_restormer import (  # noqa: E402
    Restormer,
    TASK_NATIVE_RESTORMER_SPECS,
    load_task_native_restormer,
)
from lowlight_restoration.blur_restormer_refinement import (  # noqa: E402
    load_mlx_motion_restormer_weights,
)
from smoke.evaluate_foundir_five_type_flux2_ddp import lpips_value, write_once  # noqa: E402
from smoke.evaluate_foundir_noise_rain_full_development_ddp import (  # noqa: E402
    finite_metric,
    load_audit_crop,
    source_group_bootstrap,
    strong_gate_checks,
)
from smoke.train_foundir_five_type_flux2_full_ddp import prepare_aligned_pair  # noqa: E402
from smoke.train_foundir_five_type_flux2_overfit_ddp import psnr  # noqa: E402


TASK_CONTRACTS = {
    "blur_motion_mse_lpips": {
        "category": "01Blur",
        "checkpoint": "motion_deblurring",
        "initial_checkpoint_sha256":
            "c465f48759cf155253c2c32ba4d9dd54c930fccda9b1aac1f85e2568ce65b65c",
        "records": {"train": 71393, "validation": 44, "internal_test": 0},
        "groups": {"train": 2782, "validation": 44, "internal_test": 0},
        "manifest_records": 71437,
        "validation_crop_seed": 2026092112,
        "steps": 2000,
        "milestones": [1000, 2000],
        "progressive_crops": [[0, 256]],
        "lr": 2e-5,
        "seed": 2026092113,
        "lpips_weight": 1e-2,
        "hard_crop_candidates": 4,
        "strong_gate": {
            "mean_psnr_minimum": 25.9,
            "mean_lpips_maximum": 0.18,
            "group_bootstrap_psnr_lower_minimum": 22.0,
            "group_bootstrap_lpips_upper_maximum": 0.35,
            "bootstrap_resamples": 5000,
            "bootstrap_confidence": 0.95,
            "bootstrap_seed": 2026092114,
        },
    },
    "noise_mse_lpips": {
        "category": "05Noise",
        "checkpoint": "real_denoising",
        "initial_checkpoint_sha256":
            "88389753bd72f0096e7a4f957137bfa6ed755680c9261bb22fba48c80b875180",
        "records": {"train": 54585, "validation": 1859, "internal_test": 1571},
        "groups": {"train": 1755, "validation": 34, "internal_test": 32},
        "steps": 250,
        "milestones": [250],
        "progressive_crops": [[0, 256]],
        "lr": 5e-6,
        "seed": 2026092103,
        "lpips_weight": 2e-4,
        "strong_gate": {
            "mean_psnr_minimum": 38.9,
            "mean_lpips_maximum": 0.10,
            "group_bootstrap_psnr_lower_minimum": 37.0,
            "group_bootstrap_lpips_upper_maximum": 0.13,
            "bootstrap_resamples": 5000,
            "bootstrap_confidence": 0.95,
            "bootstrap_seed": 2026092104,
        },
    },
    "noise_mse_24gpu": {
        "category": "05Noise",
        "checkpoint": "real_denoising",
        "initial_checkpoint_sha256":
            "b3fca23f8271477834c6b3eba2a6dfdf6fcbd480e53c43570e42960a13044429",
        "records": {"train": 54585, "validation": 1859, "internal_test": 1571},
        "groups": {"train": 1755, "validation": 34, "internal_test": 32},
        "steps": 1334,
        "milestones": [1334],
        "progressive_crops": [[0, 192], [667, 256]],
        "lr": 2e-5,
        "seed": 2026092101,
        "strong_gate": {
            "mean_psnr_minimum": 38.9,
            "mean_lpips_maximum": 0.10,
            "group_bootstrap_psnr_lower_minimum": 37.0,
            "group_bootstrap_lpips_upper_maximum": 0.13,
            "bootstrap_resamples": 5000,
            "bootstrap_confidence": 0.95,
            "bootstrap_seed": 2026092102,
        },
    },
    "noise_mse": {
        "category": "05Noise",
        "checkpoint": "real_denoising",
        "initial_checkpoint_sha256":
            "b3fca23f8271477834c6b3eba2a6dfdf6fcbd480e53c43570e42960a13044429",
        "records": {"train": 54585, "validation": 1859, "internal_test": 1571},
        "groups": {"train": 1755, "validation": 34, "internal_test": 32},
        "steps": 1000,
        "milestones": [1000],
        "progressive_crops": [[0, 192], [500, 256]],
        "lr": 2e-5,
        "seed": 2026092101,
        "strong_gate": {
            "mean_psnr_minimum": 38.9,
            "mean_lpips_maximum": 0.10,
            "group_bootstrap_psnr_lower_minimum": 37.0,
            "group_bootstrap_lpips_upper_maximum": 0.13,
            "bootstrap_resamples": 5000,
            "bootstrap_confidence": 0.95,
            "bootstrap_seed": 2026092102,
        },
    },
    "noise": {
        "category": "05Noise",
        "checkpoint": "real_denoising",
        "records": {"train": 54585, "validation": 1859, "internal_test": 1571},
        "groups": {"train": 1755, "validation": 34, "internal_test": 32},
        "steps": 3000,
        "milestones": [250, 750, 1500, 3000],
        "progressive_crops": [[0, 128], [1000, 192], [2500, 256]],
        "lr": 5e-5,
        "seed": 2026091003,
        "strong_gate": {
            "mean_psnr_minimum": 39.9,
            "mean_lpips_maximum": 0.08,
            "group_bootstrap_psnr_lower_minimum": 39.4,
            "group_bootstrap_lpips_upper_maximum": 0.09,
            "bootstrap_resamples": 5000,
            "bootstrap_confidence": 0.95,
            "bootstrap_seed": 2026091004,
        },
    },
    "rain": {
        "category": "10Rain",
        "checkpoint": "deraining",
        "records": {"train": 38334, "validation": 835, "internal_test": 731},
        "groups": {"train": 5780, "validation": 117, "internal_test": 117},
        "steps": 5000,
        "milestones": [500, 1500, 3000, 5000],
        "progressive_crops": [[0, 128], [1000, 192], [3000, 256]],
        "lr": 1e-4,
        "seed": 2026091005,
        "strong_gate": {
            "mean_psnr_minimum": 35.2,
            "mean_lpips_maximum": 0.06,
            "group_bootstrap_psnr_lower_minimum": 34.7,
            "group_bootstrap_lpips_upper_maximum": 0.07,
            "bootstrap_resamples": 5000,
            "bootstrap_confidence": 0.95,
            "bootstrap_seed": 2026091006,
        },
    },
    "blur_noise": {
        "category": "02Blur_Noise",
        "checkpoint": "real_denoising",
        "records": {"train": 2048, "validation": 128, "internal_test": 128},
        "groups": {"train": 768, "validation": 83, "internal_test": 83},
        "manifest_records": 2304,
        "validation_crop_seed": 2026091814,
        "steps": 1000,
        "milestones": [500, 1000],
        "progressive_crops": [[0, 128], [500, 192]],
        "lr": 5e-5,
        "seed": 2026091815,
        "strong_gate": {
            "mean_psnr_minimum": 39.13182482402203,
            "mean_lpips_maximum": 0.044947589149160194,
            "group_bootstrap_psnr_lower_minimum": 38.0,
            "group_bootstrap_lpips_upper_maximum": 0.07,
            "bootstrap_resamples": 5000,
            "bootstrap_confidence": 0.95,
            "bootstrap_seed": 2026091816,
        },
    },
    "blur": {
        "category": "01Blur",
        "checkpoint": "real_denoising",
        "initial_checkpoint_sha256":
            "faea0aa38d42e97925b99e619646eba61a464243c4ae90a29d3d1d5d334bd200",
        "records": {"train": 101907, "validation": 128, "internal_test": 128},
        "groups": {"train": 7083, "validation": 128, "internal_test": 128},
        "manifest_records": 102163,
        "validation_crop_seed": 2026092006,
        "steps": 1000,
        "milestones": [250, 500, 1000],
        "progressive_crops": [[0, 128], [500, 192]],
        "lr": 5e-5,
        "seed": 2026092007,
        "strong_gate": {
            "mean_psnr_minimum": 25.9,
            "mean_lpips_maximum": 0.18,
            "group_bootstrap_psnr_lower_minimum": 22.0,
            "group_bootstrap_lpips_upper_maximum": 0.35,
            "bootstrap_resamples": 5000,
            "bootstrap_confidence": 0.95,
            "bootstrap_seed": 2026092008,
        },
    },
    "blur_motion": {
        "category": "01Blur",
        "checkpoint": "motion_deblurring",
        "initial_checkpoint_sha256":
            "c465f48759cf155253c2c32ba4d9dd54c930fccda9b1aac1f85e2568ce65b65c",
        "records": {"train": 101907, "validation": 128, "internal_test": 128},
        "groups": {"train": 7083, "validation": 128, "internal_test": 128},
        "manifest_records": 102163,
        "validation_crop_seed": 2026092006,
        "steps": 9000,
        "milestones": [0, 1000, 3000, 6000, 9000],
        "progressive_crops": [[0, 192]],
        "lr": 2e-5,
        "seed": 2026092009,
        "strong_gate": {
            "mean_psnr_minimum": 25.9,
            "mean_lpips_maximum": 0.18,
            "group_bootstrap_psnr_lower_minimum": 22.0,
            "group_bootstrap_lpips_upper_maximum": 0.35,
            "bootstrap_resamples": 5000,
            "bootstrap_confidence": 0.95,
            "bootstrap_seed": 2026092010,
        },
    },
}


def validate_config(path: Path) -> dict:
    config = json.loads(path.read_text())
    task = config.get("task")
    expected = TASK_CONTRACTS.get(task)
    if expected is None:
        raise RuntimeError("unknown FoundIR task-native adaptation task")
    data = config.get("data", {})
    optimization = config.get("optimization", {})
    checkpoint_path = resolve_workspace_path(ROOT, config["initial_checkpoint"])
    capacity_path = resolve_workspace_path(ROOT, config["capacity_result"])
    manifest = resolve_workspace_path(ROOT, config["manifest"])
    motion_lpips = task == "blur_motion_mse_lpips"
    motion = task in ("blur_motion", "blur_motion_mse_lpips")
    noise_mse = task in ("noise_mse", "noise_mse_24gpu", "noise_mse_lpips")
    noise_mse_24gpu = task == "noise_mse_24gpu"
    noise_mse_lpips = task == "noise_mse_lpips"
    spec = None if motion else TASK_NATIVE_RESTORMER_SPECS[expected["checkpoint"]]
    scaleup = config.get("status") == "frozen_h200_eight_gpu_scaleup_from_v1_step1500"
    blur_noise = task == "blur_noise"
    blur = task == "blur"
    expected_world = 24 if noise_mse_24gpu else 8 if scaleup or blur_noise or blur or motion or noise_mse else 4
    expected_micro = 1 if noise_mse_24gpu else 4 if scaleup or blur_noise or blur or motion or noise_mse else 2
    expected_accumulation = 1 if scaleup or blur_noise or blur or motion or noise_mse else 4
    expected_global_batch = 24 if noise_mse_24gpu else 32
    expected_milestones = [1500, 3000, 5000] if scaleup else expected["milestones"]
    expected_initial_sha = expected.get(
        "initial_checkpoint_sha256", None if spec is None else spec.sha256
    )
    if (
        config.get("schema_version") != 1
        or config.get("status")
        not in (
            "frozen_after_task_native_capacity_failure",
            "frozen_h200_eight_gpu_scaleup_from_v1_step1500",
            "user_authorized_blur_noise_training_only_pilot",
            "user_authorized_blur_direct_training_only_pilot",
            "user_authorized_blur_motion_direct_scaleup",
            "user_authorized_blur_motion_mse_lpips_heavy_training",
            "user_authorized_noise_mse_training_only_pilot",
            "user_authorized_noise_mse_24gpu_training_only_pilot",
            "user_authorized_noise_mse_lpips_continuation",
        )
        or (blur_noise and config.get("status") != "user_authorized_blur_noise_training_only_pilot")
        or (blur and config.get("status") != "user_authorized_blur_direct_training_only_pilot")
        or (
            task == "blur_motion"
            and config.get("status") != "user_authorized_blur_motion_direct_scaleup"
        )
        or (
            motion_lpips
            and config.get("status")
            != "user_authorized_blur_motion_mse_lpips_heavy_training"
        )
        or (
            noise_mse
            and config.get("status")
            != (
                "user_authorized_noise_mse_24gpu_training_only_pilot"
                if noise_mse_24gpu
                else "user_authorized_noise_mse_lpips_continuation"
                if noise_mse_lpips
                else "user_authorized_noise_mse_training_only_pilot"
            )
        )
        or (scaleup and task != "rain")
        or config.get("category") != expected["category"]
        or config.get("initial_checkpoint_name") != expected["checkpoint"]
        or config.get("initial_checkpoint_sha256") != expected_initial_sha
        or not checkpoint_path.is_file()
        or file_sha256(checkpoint_path) != expected_initial_sha
        or not manifest.is_file()
        or file_sha256(manifest) != config.get("manifest_sha256")
        or int(config.get("manifest_expected_records", -1))
        != int(expected.get("manifest_records", 685495))
        or not capacity_path.is_file()
        or file_sha256(capacity_path) != config.get("capacity_result_sha256")
        or int(data.get("train_records", -1)) != expected["records"]["train"]
        or int(data.get("train_source_groups", -1)) != expected["groups"]["train"]
        or int(data.get("validation_records", -1)) != expected["records"]["validation"]
        or int(data.get("validation_source_groups", -1)) != expected["groups"]["validation"]
        or int(data.get("internal_test_records", -1))
        != expected["records"]["internal_test"]
        or int(data.get("internal_test_source_groups", -1))
        != expected["groups"]["internal_test"]
        or data.get("sampling") != "uniform_source_group_then_uniform_record"
        or data.get("geometric_augmentation") is not True
        or int(data.get("validation_crop_size", -1)) != 512
        or int(data.get("validation_crop_seed", -1))
        != int(expected.get("validation_crop_seed", 2026083001))
        or int(optimization.get("world_size", -1)) != expected_world
        or int(optimization.get("global_batch", -1)) != expected_global_batch
        or int(optimization.get("micro_batch_per_gpu", -1)) != expected_micro
        or int(optimization.get("gradient_accumulation", -1)) != expected_accumulation
        or int(optimization.get("total_steps", -1)) != expected["steps"]
        or optimization.get("milestones") != expected_milestones
        or float(optimization.get("learning_rate", -1)) != expected["lr"]
        or float(optimization.get("minimum_learning_rate", -1)) != 1e-6
        or int(optimization.get("warmup_steps", -1)) != 100
        or optimization.get("betas") != [0.9, 0.999]
        or float(optimization.get("weight_decay", -1)) != 1e-4
        or float(optimization.get("gradient_clip_norm", -1)) != 1.0
        or int(optimization.get("seed", -1)) != expected["seed"]
        or optimization.get("optimizer") != "AdamW"
        or optimization.get("loss")
        != (
            "rgb_mse_plus_lpips"
            if noise_mse_lpips or motion_lpips
            else "rgb_mse"
            if noise_mse
            else "rgb_l1"
        )
        or (
            (noise_mse_lpips or motion_lpips)
            and float(optimization.get("lpips_weight", -1)) != float(expected["lpips_weight"])
        )
        or (
            motion_lpips
            and int(optimization.get("hard_crop_candidates", -1))
            != int(expected["hard_crop_candidates"])
        )
        or optimization.get("precision") != "bfloat16_autocast_fp32_parameters"
        or optimization.get("progressive_crops") != expected["progressive_crops"]
        or config.get("strong_gate") != expected["strong_gate"]
        or config.get("selection_rule")
        != "maximize_worst_normalized_strong_gate_margin_then_earlier_step"
        or config.get("capacity_only_not_final_deployable_routing") is not True
        or config.get("sealed_internal_test_accessed") is not False
        or config.get("protected_foundir_test_accessed") is not False
        or (blur_noise and float(optimization.get("identity_mix_probability", -1)) != 0.25)
        or (blur and float(optimization.get("identity_mix_probability", -1)) != 0.0)
        or (motion and float(optimization.get("identity_mix_probability", -1)) != 0.0)
        or (noise_mse and float(optimization.get("identity_mix_probability", -1)) != 0.0)
    ):
        raise RuntimeError("FoundIR task-native adaptation contract changed")
    if int(optimization["global_batch"]) != (
        int(optimization["world_size"])
        * int(optimization["micro_batch_per_gpu"])
        * int(optimization["gradient_accumulation"])
    ):
        raise RuntimeError("FoundIR task-native global batch arithmetic changed")
    crops = optimization.get("progressive_crops")
    if (
        not isinstance(crops, list)
        or not crops
        or crops[0][0] != 0
        or any(
            len(item) != 2
            or isinstance(item[0], bool)
            or isinstance(item[1], bool)
            or not isinstance(item[0], int)
            or not isinstance(item[1], int)
            or item[0] < 0
            or item[1] < 8
            or item[1] % 8
            for item in crops
        )
        or any(crops[index][0] >= crops[index + 1][0] for index in range(len(crops) - 1))
    ):
        raise RuntimeError("invalid progressive crop contract")
    capacity = json.loads(capacity_path.read_text())
    if blur_noise:
        if (
            capacity.get("status") != "failed"
            or capacity.get("automatic_metric_gate_pass") is not False
            or capacity.get("model_weight_updates") is not False
            or capacity.get("development_pixels_accessed") is not False
            or capacity.get("official_foundir_test_accessed") is not False
        ):
            raise RuntimeError("Blur+Noise pilot requires verified sequential-specialist failure")
    elif blur:
        initial = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if (
            capacity.get("status")
            != "foundir_task_native_adaptation_development_selection"
            or capacity.get("task") != "blur_noise"
            or capacity.get("selected_step") != 1000
            or capacity.get("checkpoint_sha256") != expected_initial_sha
            or capacity.get("strong_gate_pass") is not True
            or capacity.get("sealed_internal_test_accessed") is not False
            or capacity.get("protected_foundir_test_accessed") is not False
            or initial.get("status") != "foundir_task_native_adaptation_checkpoint"
            or initial.get("task") != "blur_noise"
            or initial.get("step") != 1000
            or initial.get("sealed_internal_test_accessed") is not False
            or initial.get("protected_foundir_test_accessed") is not False
        ):
            raise RuntimeError("Blur direct pilot requires the frozen Blur+Noise adaptation")
    elif motion_lpips:
        if (
            capacity.get("status") != "failed"
            or capacity.get("checks", {}).get("01Blur", {}).get("psnr") is not False
            or capacity.get("checks", {}).get("01Blur", {}).get("lpips") is not False
            or capacity.get("routing", {}).get("wrong") != 0
            or config.get("official_feedback_used_for_design") is not True
        ):
            raise RuntimeError("Blur MSE+LPIPS training requires the fixed official v6 failure")
    elif motion:
        if (
            capacity.get("status") != "blur_restormer_refinement_gate_complete"
            or capacity.get("gate_pass") is not False
            or capacity.get("completed_optimizer_steps") != 0
            or capacity.get("selected_residual_scale") != 0.0
            or capacity.get("sealed_internal_test_accessed") is not False
            or capacity.get("protected_foundir_test_accessed") is not False
        ):
            raise RuntimeError("Blur motion scale-up requires the frozen step-0 domain audit")
        continuation = config.get("continuation", {})
        if continuation != {
            "stop_on_gate_pass": True,
            "progress_step": 1000,
            "minimum_psnr_gain_db": 0.25,
            "minimum_lpips_reduction": 0.01,
        }:
            raise RuntimeError("Blur motion continuation contract changed")
    elif noise_mse_lpips:
        initial = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if (
            capacity.get("status") != "failed"
            or capacity.get("checks", {}).get("05Noise", {}).get("psnr") is not True
            or capacity.get("checks", {}).get("05Noise", {}).get("lpips") is not False
            or initial.get("status") != "foundir_task_native_adaptation_checkpoint"
            or initial.get("task") != "noise_mse"
            or initial.get("step") != 1000
            or initial.get("sealed_internal_test_accessed") is not False
            or initial.get("protected_foundir_test_accessed") is not False
            or config.get("official_feedback_used_for_design") is not True
        ):
            raise RuntimeError("Noise LPIPS continuation requires the fixed v4 LPIPS-only failure and MSE step-1000")
    elif noise_mse:
        initial = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if (
            capacity.get("status") != "failed"
            or capacity.get("checks", {}).get("05Noise", {}).get("psnr") is not False
            or capacity.get("checks", {}).get("05Noise", {}).get("lpips") is not False
            or initial.get("status") != "foundir_task_native_adaptation_checkpoint"
            or initial.get("task") != "noise"
            or initial.get("step") != 250
            or initial.get("sealed_internal_test_accessed") is not False
            or initial.get("protected_foundir_test_accessed") is not False
            or config.get("official_feedback_used_for_design") is not True
        ):
            raise RuntimeError("Noise MSE pilot requires the failed official v3 evidence and project step-250 initialization")
    else:
        candidate_key = f"{task}:{expected['checkpoint']}"
        if (
            capacity.get("status") != "task_native_capacity_audit_complete"
            or capacity.get("candidate_capacity", {}).get(candidate_key, {}).get("strong_gate_pass")
            is not False
            or capacity.get("sealed_internal_test_accessed") is not False
            or capacity.get("protected_foundir_test_accessed") is not False
        ):
            raise RuntimeError("adaptation may run only after verified task-native capacity failure")
    if scaleup:
        resume = config.get("scaleup_resume", {})
        source_config = resolve_workspace_path(ROOT, resume.get("source_config", ""))
        source_checkpoint = resolve_workspace_path(ROOT, resume.get("checkpoint", ""))
        source_result = resolve_workspace_path(ROOT, resume.get("development_result", ""))
        if (
            resume.get("policy")
            != "resume_exact_v1_model_optimizer_at_step1500_with_equivalent_global_batch_v1"
            or int(resume.get("step", -1)) != 1500
            or file_sha256(source_config) != resume.get("source_config_sha256")
            or file_sha256(source_checkpoint) != resume.get("checkpoint_sha256")
            or file_sha256(source_result) != resume.get("development_result_sha256")
        ):
            raise RuntimeError("invalid Rain eight-GPU scale-up resume contract")
        source_payload = torch.load(source_checkpoint, map_location="cpu", weights_only=False)
        result_payload = json.loads(source_result.read_text())
        if (
            source_payload.get("status") != "foundir_task_native_adaptation_checkpoint"
            or source_payload.get("config_sha256") != resume.get("source_config_sha256")
            or source_payload.get("task") != "rain"
            or source_payload.get("step") != 1500
            or source_payload.get("sealed_internal_test_accessed") is not False
            or source_payload.get("protected_foundir_test_accessed") is not False
            or result_payload.get("status")
            != "foundir_task_native_adaptation_development_milestone"
            or result_payload.get("config_sha256") != resume.get("source_config_sha256")
            or result_payload.get("checkpoint_sha256") != resume.get("checkpoint_sha256")
            or result_payload.get("task") != "rain"
            or result_payload.get("step") != 1500
            or result_payload.get("sealed_internal_test_accessed") is not False
            or result_payload.get("protected_foundir_test_accessed") is not False
        ):
            raise RuntimeError("Rain eight-GPU scale-up source artifact identity changed")
    return config


def load_task_rows(config: Mapping[str, object]) -> dict[str, list[dict[str, str]]]:
    manifest = resolve_workspace_path(ROOT, config["manifest"])
    selected: dict[str, list[dict[str, str]]] = defaultdict(list)
    total = 0
    with manifest.open(newline="") as handle:
        for row in csv.DictReader(handle):
            total += 1
            if row["category"] == config["category"]:
                selected[row["split"]].append(dict(row))
    if total != int(config["manifest_expected_records"]):
        raise RuntimeError("FoundIR task-native source manifest count changed")
    expected = TASK_CONTRACTS[str(config["task"])]
    for split in ("train", "validation", "internal_test"):
        rows = selected[split]
        if len(rows) != expected["records"][split]:
            raise RuntimeError(f"FoundIR task-native {split} record count changed")
        if len({row["source_group"] for row in rows}) != expected["groups"][split]:
            raise RuntimeError(f"FoundIR task-native {split} source-group count changed")
    if set(selected) != {"train", "validation", "internal_test"}:
        raise RuntimeError("FoundIR task-native split set changed")
    all_rows = [row for split in ("train", "validation", "internal_test") for row in selected[split]]
    if len({row["record_id"] for row in all_rows}) != len(all_rows):
        raise RuntimeError("FoundIR task-native record IDs are not unique")
    group_splits: dict[str, str] = {}
    for split, rows in selected.items():
        for row in rows:
            previous = group_splits.setdefault(row["source_group"], split)
            if previous != split:
                raise RuntimeError("FoundIR task-native source group crosses splits")
    return dict(selected)


def crop_size_for_step(config: Mapping[str, object], step: int) -> int:
    result = None
    for start, size in config["optimization"]["progressive_crops"]:
        if step >= int(start):
            result = int(size)
    if result is None:
        raise RuntimeError("progressive crop schedule has no active size")
    return result


def sample_global_batch(
    grouped: Mapping[str, Sequence[Mapping[str, str]]],
    *,
    task: str,
    step: int,
    seed: int,
    count: int,
) -> list[tuple[Mapping[str, str], int]]:
    groups = sorted(grouped)
    if not groups or count <= 0:
        raise RuntimeError("FoundIR task-native sampling needs groups and a positive batch")
    generator = random.Random(int(stable_digest(seed, task, "train-step", step)[:16], 16))
    result = []
    for position in range(count):
        group = groups[generator.randrange(len(groups))]
        rows = grouped[group]
        row = rows[generator.randrange(len(rows))]
        augmentation_seed = int(
            stable_digest(seed, task, step, position, row["record_id"])[:16], 16
        )
        result.append((row, augmentation_seed))
    return result


def load_training_patch(
    row: Mapping[str, str],
    *,
    crop_size: int,
    augmentation_seed: int,
    verified: set[str],
    hard_crop_candidates: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    arrays = []
    for key, hash_key in (("lq_path", "lq_sha256"), ("gt_path", "gt_sha256")):
        path = resolve_workspace_path(ROOT, row[key])
        identity = path.as_posix()
        if identity not in verified:
            if file_sha256(path) != row[hash_key]:
                raise RuntimeError(f"FoundIR task-native training pair changed: {row['record_id']}")
            verified.add(identity)
        with Image.open(path) as handle:
            arrays.append(np.asarray(handle.convert("RGB"), dtype=np.uint8).copy())
    lq, gt = prepare_aligned_pair(arrays[0], arrays[1], crop_size, row["record_id"])
    generator = random.Random(augmentation_seed)
    candidates = [
        (
            generator.randrange(lq.shape[0] - crop_size + 1),
            generator.randrange(lq.shape[1] - crop_size + 1),
        )
        for _ in range(hard_crop_candidates)
    ]
    y, x = max(
        candidates,
        key=lambda position: float(
            np.mean(
                (
                    lq[position[0] : position[0] + crop_size,
                       position[1] : position[1] + crop_size].astype(np.float32)
                    - gt[position[0] : position[0] + crop_size,
                         position[1] : position[1] + crop_size].astype(np.float32)
                ) ** 2
            )
        ),
    )
    lq = lq[y : y + crop_size, x : x + crop_size]
    gt = gt[y : y + crop_size, x : x + crop_size]
    rotation = generator.randrange(4)
    if rotation:
        lq, gt = np.rot90(lq, rotation), np.rot90(gt, rotation)
    if generator.randrange(2):
        lq, gt = np.flip(lq, axis=1), np.flip(gt, axis=1)

    def tensor(array: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.ascontiguousarray(array)).permute(2, 0, 1).float().div_(255)

    return tensor(lq), tensor(gt)


def learning_rate(config: Mapping[str, object], step: int) -> float:
    optimization = config["optimization"]
    peak = float(optimization["learning_rate"])
    minimum = float(optimization["minimum_learning_rate"])
    warmup = int(optimization["warmup_steps"])
    total = int(optimization["total_steps"])
    if step < 1 or step > total:
        raise ValueError("adaptation learning-rate step is outside the training schedule")
    if step <= warmup:
        return peak * step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return minimum + (peak - minimum) * 0.5 * (1 + math.cos(math.pi * progress))


def move_optimizer_state(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def save_checkpoint(
    path: Path,
    *,
    config_sha: str,
    task: str,
    step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> None:
    payload = {
        "schema_version": 1,
        "status": "foundir_task_native_adaptation_checkpoint",
        "config_sha256": config_sha,
        "task": task,
        "step": step,
        "model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "optimizer": optimizer.state_dict(),
        "sealed_internal_test_accessed": False,
        "protected_foundir_test_accessed": False,
    }
    if path.exists():
        existing = torch.load(path, map_location="cpu", weights_only=False)
        if (
            existing.get("status") != payload["status"]
            or existing.get("config_sha256") != config_sha
            or existing.get("task") != task
            or existing.get("step") != step
        ):
            raise RuntimeError(f"existing adaptation checkpoint identity changed: {path}")
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise RuntimeError(f"orphan adaptation checkpoint temporary exists: {temporary}")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def resume_checkpoint(
    output: Path,
    *,
    config_sha: str,
    task: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> int:
    paths = sorted(output.glob("step_*.pt"))
    temporaries = sorted(output.glob("step_*.pt.tmp"))
    if temporaries:
        raise RuntimeError(f"orphan adaptation checkpoint temporaries exist: {temporaries}")
    if not paths:
        return 0
    path = paths[-1]
    payload = torch.load(path, map_location="cpu", weights_only=False)
    step = int(path.stem.removeprefix("step_"))
    allowed_steps = set(TASK_CONTRACTS[task]["milestones"])
    if (
        payload.get("status") != "foundir_task_native_adaptation_checkpoint"
        or payload.get("config_sha256") != config_sha
        or payload.get("task") != task
        or payload.get("step") != step
        or payload.get("sealed_internal_test_accessed") is not False
        or payload.get("protected_foundir_test_accessed") is not False
        or step not in allowed_steps
        or not isinstance(payload.get("model"), Mapping)
        or not isinstance(payload.get("optimizer"), Mapping)
    ):
        raise RuntimeError(f"invalid adaptation resume checkpoint: {path}")
    model.load_state_dict(payload["model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    move_optimizer_state(optimizer, device)
    return step


def resume_scaleup_source(
    config: Mapping[str, object],
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> int:
    resume = config.get("scaleup_resume")
    if not isinstance(resume, Mapping):
        return 0
    path = resolve_workspace_path(ROOT, resume["checkpoint"])
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    move_optimizer_state(optimizer, device)
    return int(resume["step"])


def raise_if_any_rank_failed(
    local_error: str | None,
    *,
    device: torch.device,
    world: int,
    operation: str,
) -> None:
    """Make recoverable per-rank failures visible before the next DDP collective."""

    failed = torch.tensor([int(local_error is not None)], device=device, dtype=torch.int32)
    dist.all_reduce(failed, op=dist.ReduceOp.MAX)
    if not int(failed.item()):
        return
    errors: list[object | None] = [None] * world
    dist.all_gather_object(errors, local_error)
    details = "; ".join(
        f"rank={rank}: {error}" for rank, error in enumerate(errors) if error is not None
    )
    raise RuntimeError(f"FoundIR task-native {operation} failed: {details}")


def validation_payload(
    rows: Sequence[Mapping[str, object]],
    *,
    config: Mapping[str, object],
    config_sha: str,
    checkpoint: Path,
    step: int,
) -> dict[str, object]:
    if len(rows) != int(config["data"]["validation_records"]):
        raise RuntimeError("adaptation validation coverage changed")
    metrics = {
        "records": len(rows),
        "source_groups": len({str(row["source_group"]) for row in rows}),
        "psnr": float(np.mean([row["restored"]["psnr"] for row in rows])),
        "lpips": float(np.mean([row["restored"]["lpips"] for row in rows])),
    }
    gate = config["strong_gate"]
    bootstrap = source_group_bootstrap(
        rows,
        resamples=int(gate["bootstrap_resamples"]),
        confidence=float(gate["bootstrap_confidence"]),
        seed=int(stable_digest(int(gate["bootstrap_seed"]), config["task"], step)[:16], 16),
    )
    checks = strong_gate_checks(metrics, bootstrap, gate)
    normalized = [
        metrics["psnr"] - float(gate["mean_psnr_minimum"]),
        (float(gate["mean_lpips_maximum"]) - metrics["lpips"]) / 0.02,
        float(bootstrap["lower"]["psnr"])
        - float(gate["group_bootstrap_psnr_lower_minimum"]),
        (float(gate["group_bootstrap_lpips_upper_maximum"])
        - float(bootstrap["upper"]["lpips"]))
        / 0.02,
    ]
    if not all(finite_metric(value) for value in normalized):
        raise RuntimeError("adaptation validation score is non-finite")
    return {
        "schema_version": 1,
        "status": "foundir_task_native_adaptation_development_milestone",
        "config_sha256": config_sha,
        "task": config["task"],
        "split": "validation",
        "step": step,
        "checkpoint": checkpoint.relative_to(ROOT).as_posix(),
        "checkpoint_sha256": file_sha256(checkpoint),
        "metrics": metrics,
        "source_group_bootstrap": bootstrap,
        "strong_gate_checks": checks,
        "strong_gate_pass": all(checks.values()),
        "selection_score": min(normalized),
        "capacity_only_not_final_deployable_routing": True,
        "sealed_internal_test_accessed": False,
        "protected_foundir_test_accessed": False,
    }


def evaluate(
    model: torch.nn.Module,
    rows: Sequence[Mapping[str, str]],
    metric,
    *,
    config: Mapping[str, object],
    config_sha: str,
    checkpoint: Path,
    step: int,
    rank: int,
    world: int,
    device: torch.device,
) -> dict[str, object] | None:
    model.eval()
    local = []
    local_error = None
    try:
        for index in range(rank, len(rows), world):
            row = rows[index]
            lq_m11, gt_m11, _, _ = load_audit_crop(
                row,
                crop_size=int(config["data"]["validation_crop_size"]),
                seed=int(config["data"]["validation_crop_seed"]),
            )
            source = ((lq_m11 + 1) * 0.5).unsqueeze(0).to(device)
            target = gt_m11.unsqueeze(0).to(device)
            with torch.inference_mode():
                restored = model(source).float().clamp(0, 1)
            array = restored[0].cpu().permute(1, 2, 0).mul(255).round().byte().numpy()
            restored_m11 = (
                torch.from_numpy(array.copy())
                .permute(2, 0, 1)
                .float()
                .div_(127.5)
                .sub_(1)
                .unsqueeze(0)
            )
            local.append(
                {
                    "record_id": row["record_id"],
                    "source_group": row["source_group"],
                    "restored": {
                        "psnr": psnr(target, restored_m11.to(device)),
                        "lpips": lpips_value(metric, target, restored_m11, device),
                    },
                }
            )
            metric.reset()
    except Exception as error:  # synchronize ordinary data/metric failures across ranks
        local_error = f"{type(error).__name__}: {error}"
    raise_if_any_rank_failed(
        local_error,
        device=device,
        world=world,
        operation=f"development evaluation at step {step}",
    )
    gathered: list[object | None] = [None] * world
    dist.all_gather_object(gathered, local)
    payload = None
    if rank == 0:
        complete = sorted(
            [item for shard in gathered for item in shard], key=lambda item: item["record_id"]
        )
        payload = validation_payload(
            complete,
            config=config,
            config_sha=config_sha,
            checkpoint=checkpoint,
            step=step,
        )
        result_path = checkpoint.parent / f"development_step_{step:06d}.json"
        write_once(result_path, (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode())
        print(
            "FOUNDIR_TASK_NATIVE_ADAPTATION_MILESTONE " + json.dumps(payload, sort_keys=True),
            flush=True,
        )
    model.train()
    return payload


def finalize_selection(
    config: Mapping[str, object], config_sha: str, output: Path
) -> dict[str, object]:
    candidates = []
    allow_early = config.get("task") == "blur_motion"
    for step in config["optimization"]["milestones"]:
        result_path = output / f"development_step_{int(step):06d}.json"
        if not result_path.is_file():
            if allow_early:
                continue
            raise RuntimeError(f"missing adaptation development milestone: {result_path}")
        payload = json.loads(result_path.read_text())
        if (
            payload.get("status") != "foundir_task_native_adaptation_development_milestone"
            or payload.get("config_sha256") != config_sha
            or payload.get("step") != int(step)
            or payload.get("task") != config["task"]
            or payload.get("split") != "validation"
            or payload.get("capacity_only_not_final_deployable_routing") is not True
            or payload.get("sealed_internal_test_accessed") is not False
            or payload.get("protected_foundir_test_accessed") is not False
            or not finite_metric(payload.get("selection_score"))
            or not isinstance(payload.get("strong_gate_pass"), bool)
        ):
            raise RuntimeError(f"invalid adaptation milestone: {result_path}")
        checkpoint = resolve_workspace_path(ROOT, payload.get("checkpoint", ""))
        if (
            not checkpoint.is_file()
            or file_sha256(checkpoint) != payload.get("checkpoint_sha256")
        ):
            raise RuntimeError(f"adaptation milestone checkpoint changed: {checkpoint}")
        candidates.append((payload, result_path))
    if not candidates:
        raise RuntimeError("adaptation selection has no completed milestones")
    selected, result_path = max(
        candidates, key=lambda item: (float(item[0]["selection_score"]), -int(item[0]["step"]))
    )
    selection = {
        "schema_version": 1,
        "status": "foundir_task_native_adaptation_development_selection",
        "config_sha256": config_sha,
        "task": config["task"],
        "selection_rule": config["selection_rule"],
        "selected_step": selected["step"],
        "checkpoint": selected["checkpoint"],
        "checkpoint_sha256": selected["checkpoint_sha256"],
        "development_result": result_path.relative_to(ROOT).as_posix(),
        "development_result_sha256": file_sha256(result_path),
        "metrics": selected["metrics"],
        "source_group_bootstrap": selected["source_group_bootstrap"],
        "strong_gate_checks": selected["strong_gate_checks"],
        "strong_gate_pass": selected["strong_gate_pass"],
        "capacity_only_not_final_deployable_routing": True,
        "sealed_internal_test_accessed": False,
        "protected_foundir_test_accessed": False,
    }
    write_once(
        output / "development_selection.json",
        (json.dumps(selection, indent=2, sort_keys=True) + "\n").encode(),
    )
    print("FOUNDIR_TASK_NATIVE_ADAPTATION_COMPLETE " + json.dumps(selection, sort_keys=True), flush=True)
    return selection


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    config_path = resolve_workspace_path(ROOT, args.config)
    config = validate_config(config_path)
    config_sha = file_sha256(config_path)
    rows = load_task_rows(config)
    if args.preflight:
        print(
            "FOUNDIR_TASK_NATIVE_ADAPTATION_PREFLIGHT_PASS "
            + json.dumps(
                {
                    "task": config["task"],
                    "config_sha256": config_sha,
                    "train_records": len(rows["train"]),
                    "validation_records": len(rows["validation"]),
                    "sealed_internal_test_accessed": False,
                    "protected_foundir_test_accessed": False,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return

    optimization = config["optimization"]
    world = int(optimization["world_size"])
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    expected_local_world = 8 if world == 24 else world
    if (
        int(os.environ["WORLD_SIZE"]) != world
        or int(os.environ.get("LOCAL_WORLD_SIZE", -1)) != expected_local_world
        or torch.cuda.device_count() != expected_local_world
    ):
        raise RuntimeError(
            f"FoundIR task-native adaptation requires world={world} "
            f"and local_world={expected_local_world}"
        )
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if not any(name in torch.cuda.get_device_properties(device).name for name in ("H100", "H200")):
        raise RuntimeError("FoundIR task-native adaptation requires H100/H200 GPUs")
    dist.init_process_group("nccl", timeout=timedelta(minutes=30), device_id=device)
    seed = int(optimization["seed"])
    torch.random.default_generator.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    initial_directory = resolve_workspace_path(ROOT, Path(config["initial_checkpoint"]).parent)
    if config["task"] in ("blur", "noise_mse", "noise_mse_24gpu", "noise_mse_lpips"):
        initial_directory = ROOT / "models/restormer-task-native"
    if config["task"] in ("blur_motion", "blur_motion_mse_lpips"):
        model = Restormer(layer_norm_type="WithBias")
        metadata = load_mlx_motion_restormer_weights(
            model, resolve_workspace_path(ROOT, config["initial_checkpoint"])
        )
        metadata = dict(metadata)
        metadata["name"] = "motion_deblurring"
    else:
        model, metadata = load_task_native_restormer(
            config["initial_checkpoint_name"], initial_directory
        )
    if config["task"] in ("blur", "noise_mse", "noise_mse_24gpu", "noise_mse_lpips"):
        initial = torch.load(
            resolve_workspace_path(ROOT, config["initial_checkpoint"]),
            map_location="cpu",
            weights_only=False,
        )
        model.load_state_dict(initial["model"], strict=True)
        metadata = dict(metadata)
        metadata["adaptation_initialization"] = (
            "noise_mse_step_1000"
            if config["task"] == "noise_mse_lpips"
            else "noise_step_250"
            if config["task"] in ("noise_mse", "noise_mse_24gpu")
            else "blur_noise_step_1000"
        )
    model = model.to(device)
    distributed = DistributedDataParallel(
        model, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False
    )
    optimizer = torch.optim.AdamW(
        distributed.parameters(),
        lr=float(optimization["learning_rate"]),
        betas=tuple(float(value) for value in optimization["betas"]),
        weight_decay=float(optimization["weight_decay"]),
    )
    output = resolve_workspace_path(ROOT, config["output"])
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
    dist.barrier()
    current_step = resume_checkpoint(
        output,
        config_sha=config_sha,
        task=config["task"],
        model=distributed.module,
        optimizer=optimizer,
        device=device,
    )
    if current_step == 0:
        current_step = resume_scaleup_source(
            config,
            model=distributed.module,
            optimizer=optimizer,
            device=device,
        )
    starts = torch.tensor([current_step, current_step], device=device, dtype=torch.int64)
    dist.all_reduce(starts[:1], op=dist.ReduceOp.MIN)
    dist.all_reduce(starts[1:], op=dist.ReduceOp.MAX)
    if int(starts[0].item()) != current_step or int(starts[1].item()) != current_step:
        raise RuntimeError("adaptation ranks resumed different steps")
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

    metric = (
        LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True)
        .to(device)
        .eval()
        .requires_grad_(False)
    )
    grouped: dict[str, list[Mapping[str, str]]] = defaultdict(list)
    for row in rows["train"]:
        grouped[row["source_group"]].append(row)
    verified: set[str] = set()
    micro = int(optimization["micro_batch_per_gpu"])
    accumulation = int(optimization["gradient_accumulation"])
    global_batch = int(optimization["global_batch"])
    milestones = {int(value) for value in optimization["milestones"]}
    total = int(optimization["total_steps"])
    completed_step = current_step
    started = __import__("time").perf_counter()
    print(
        "FOUNDIR_TASK_NATIVE_ADAPTATION_START "
        + json.dumps(
            {
                "task": config["task"],
                "rank": rank,
                "world": world,
                "resume_step": current_step,
                "model": metadata,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if current_step in milestones:
        checkpoint = output / f"step_{current_step:06d}.pt"
        if rank == 0 and not checkpoint.is_file():
            save_checkpoint(
                checkpoint,
                config_sha=config_sha,
                task=config["task"],
                step=current_step,
                model=distributed.module,
                optimizer=optimizer,
            )
        dist.barrier()
        result_path = output / f"development_step_{current_step:06d}.json"
        if not result_path.is_file():
            evaluate(
                distributed.module,
                rows["validation"],
                metric,
                config=config,
                config_sha=config_sha,
                checkpoint=checkpoint,
                step=current_step,
                rank=rank,
                world=world,
                device=device,
            )
            dist.barrier()
    stopped_early = False
    for step in range(current_step + 1, total + 1):
        completed_step = step
        lr = learning_rate(config, step)
        for group in optimizer.param_groups:
            group["lr"] = lr
        assignments = sample_global_batch(
            grouped,
            task=config["task"],
            step=step,
            seed=int(optimization["seed"]),
            count=global_batch,
        )
        local_assignments = assignments[rank::world]
        if len(local_assignments) != micro * accumulation:
            raise RuntimeError("adaptation rank batch arithmetic changed")
        crop_size = crop_size_for_step(config, step)
        optimizer.zero_grad(set_to_none=True)
        loss_value = 0.0
        for accumulation_index in range(accumulation):
            batch = local_assignments[
                accumulation_index * micro : (accumulation_index + 1) * micro
            ]
            pairs = None
            local_error = None
            try:
                pairs = [
                    load_training_patch(
                        row,
                        crop_size=crop_size,
                        augmentation_seed=augmentation_seed,
                        verified=verified,
                        hard_crop_candidates=int(optimization.get("hard_crop_candidates", 1)),
                    )
                    for row, augmentation_seed in batch
                ]
            except Exception as error:  # keep a bad input from stranding peer ranks in DDP
                local_error = f"{type(error).__name__}: {error}"
            raise_if_any_rank_failed(
                local_error,
                device=device,
                world=world,
                operation=f"training batch load at step {step} accumulation {accumulation_index}",
            )
            assert pairs is not None
            source = torch.stack([pair[0] for pair in pairs]).to(device)
            target = torch.stack([pair[1] for pair in pairs]).to(device)
            if config["task"] == "blur_noise":
                identity_mask = torch.tensor(
                    [
                        int(stable_digest("blur_noise_identity", step, seed)[0:16], 16) % 4
                        == 0
                        for _, seed in batch
                    ],
                    device=device,
                    dtype=torch.bool,
                )
                source[identity_mask] = target[identity_mask]
            sync = nullcontext() if accumulation_index + 1 == accumulation else distributed.no_sync()
            with sync:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    restored = distributed(source)
                    if config["task"] in ("noise_mse_lpips", "blur_motion_mse_lpips"):
                        pixel_loss = F.mse_loss(restored.float(), target.float())
                        perceptual_loss = metric(restored.float().clamp(0, 1), target.float())
                        loss = (
                            pixel_loss
                            + float(optimization["lpips_weight"]) * perceptual_loss
                        ) / accumulation
                    else:
                        loss = (
                            F.mse_loss(restored.float(), target.float())
                            if config["task"] in ("noise_mse", "noise_mse_24gpu")
                            else F.l1_loss(restored, target)
                        ) / accumulation
                nonfinite = torch.tensor(
                    [int(not bool(torch.isfinite(loss).item()))], device=device, dtype=torch.int32
                )
                dist.all_reduce(nonfinite, op=dist.ReduceOp.MAX)
                if int(nonfinite.item()):
                    raise RuntimeError(f"non-finite adaptation loss at step {step}")
                loss.backward()
                if config["task"] in ("noise_mse_lpips", "blur_motion_mse_lpips"):
                    metric.reset()
            loss_value += float(loss.detach().item())
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            distributed.parameters(), float(optimization["gradient_clip_norm"])
        )
        if not bool(torch.isfinite(gradient_norm).item()):
            raise RuntimeError(f"non-finite adaptation gradient norm at step {step}")
        optimizer.step()
        if step % 50 == 0 and rank == 0:
            elapsed = __import__("time").perf_counter() - started
            print(
                f"FOUNDIR_TASK_NATIVE_ADAPTATION_PROGRESS task={config['task']} "
                f"step={step}/{total} crop={crop_size} loss={loss_value:.8f} "
                f"lr={lr:.8g} elapsed_seconds={elapsed:.1f}",
                flush=True,
            )
        if step in milestones:
            checkpoint = output / f"step_{step:06d}.pt"
            if rank == 0:
                save_checkpoint(
                    checkpoint,
                    config_sha=config_sha,
                    task=config["task"],
                    step=step,
                    model=distributed.module,
                    optimizer=optimizer,
                )
            dist.barrier()
            evaluate(
                distributed.module,
                rows["validation"],
                metric,
                config=config,
                config_sha=config_sha,
                checkpoint=checkpoint,
                step=step,
                rank=rank,
                world=world,
                device=device,
            )
            dist.barrier()
            if config["task"] in ("blur_motion", "blur_motion_mse_lpips"):
                current_result = json.loads(
                    (output / f"development_step_{step:06d}.json").read_text()
                )
                stop = bool(current_result["strong_gate_pass"])
                if (
                    config["task"] == "blur_motion"
                    and step == int(config["continuation"]["progress_step"])
                ):
                    initial_result = json.loads(
                        (output / "development_step_000000.json").read_text()
                    )
                    psnr_gain = float(current_result["metrics"]["psnr"]) - float(
                        initial_result["metrics"]["psnr"]
                    )
                    lpips_reduction = float(initial_result["metrics"]["lpips"]) - float(
                        current_result["metrics"]["lpips"]
                    )
                    stop = stop or (
                        psnr_gain
                        < float(config["continuation"]["minimum_psnr_gain_db"])
                        or lpips_reduction
                        < float(config["continuation"]["minimum_lpips_reduction"])
                    )
                    if rank == 0:
                        print(
                            "FOUNDIR_BLUR_MOTION_PROGRESS_GATE "
                            + json.dumps(
                                {
                                    "step": step,
                                    "psnr_gain_db": psnr_gain,
                                    "lpips_reduction": lpips_reduction,
                                    "continue_training": not stop,
                                },
                                sort_keys=True,
                            ),
                            flush=True,
                        )
                decision = torch.tensor([int(stop)], device=device, dtype=torch.int32)
                dist.all_reduce(decision, op=dist.ReduceOp.MAX)
                if int(decision.item()):
                    stopped_early = True
                    break
    if rank == 0:
        selection = finalize_selection(config, config_sha, output)
        if config["task"] in ("blur_motion", "blur_motion_mse_lpips"):
            print(
                "FOUNDIR_BLUR_MOTION_TERMINAL "
                + json.dumps(
                    {
                        "stopped_early": stopped_early,
                        "completed_step": completed_step,
                        "gate_pass": selection["strong_gate_pass"],
                        "selected_step": selection["selected_step"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
