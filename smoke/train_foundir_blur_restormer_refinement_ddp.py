#!/usr/bin/env python3
"""Train a Restormer to refine the frozen Qwen-routed Blur base output."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lowlight_restoration.flux_prompt_cache import file_sha256  # noqa: E402
from lowlight_restoration.blur_restormer_refinement import (  # noqa: E402
    BaseLQRestormerExpert,
    BaseLQRestormerRefiner,
    load_mlx_motion_restormer_weights,
)
from lowlight_restoration.foundir_flux2_protocol import resolve_workspace_path, stable_digest  # noqa: E402
from lowlight_restoration.pixel_refinement import spatial_highpass  # noqa: E402
from smoke.evaluate_foundir_five_type_flux2_ddp import (  # noqa: E402
    load_aligned_crop,
    selected_from_contract,
    torchrun_worker_identity,
    write_once,
)
from smoke.evaluate_foundir_five_type_qwen_routed_integrated_ddp import (  # noqa: E402
    route_decision,
    validate_router_config,
)
from smoke.evaluate_foundir_five_type_released_baselines_ddp import (  # noqa: E402
    released_config,
    run_generalist,
)
from smoke.run_foundir_lowlight_gate2_teachers import (  # noqa: E402
    SPECIALIST_DEPENDENCY_DIRECTORIES,
    build_weather_specialist,
    import_weather_specialist_architecture,
    verify_specialist_dependency_versions,
)
from smoke.train_foundir_blur_generalist_specialization_ddp import build_diffusion  # noqa: E402
from smoke.train_foundir_blur_pixel_expert_gate_ddp import (  # noqa: E402
    BlurPresentationDataset,
    RankPresentationSampler,
    collate,
    learning_rate,
    mean_metrics,
    scalar_metrics,
)
from smoke.train_foundir_five_type_flux2_full_ddp import (  # noqa: E402
    load_rows,
    synchronized_loader_batches,
    synchronized_rank_error,
)


FOUNDIR_RUNTIME = ROOT / ".tools/foundir-runtime"

FUSION_ARCHITECTURES = {
    "released_foundir_restormer_26m_base_lq_fusion",
    "released_foundir_restormer_26m_base_lq_fusion_corrective",
    "upstream_motion_deblur_restormer_26m_base_lq_fusion",
}

MOTION_ARCHITECTURE = "upstream_motion_deblur_restormer_26m_base_lq_fusion"


def residual_scale_key(value: float) -> str:
    return f"{float(value):.6f}"


def evaluation_residual_scales(config: Mapping[str, object]) -> list[float]:
    values = config["model"].get("evaluation_residual_scales", [1.0])
    scales = [float(value) for value in values]
    if (
        not scales
        or scales != sorted(set(scales))
        or any(not math.isfinite(value) or value < 0.0 or value > 2.0 for value in scales)
    ):
        raise RuntimeError("invalid Blur Restormer residual-scale candidates")
    return scales


class QwenLevelWeightedBlurDataset(Dataset[dict[str, object]]):
    """Sample only deployment-active Blur levels with a frozen level mixture."""

    def __init__(
        self,
        rows_by_level: Mapping[int, Sequence[Mapping[str, str]]],
        probabilities: Mapping[int, float],
        config: Mapping[str, object],
    ) -> None:
        self.seed = int(config["data"]["seed"])
        self.levels = tuple(sorted(probabilities))
        if set(rows_by_level) != set(self.levels) or any(
            not rows_by_level[level] for level in self.levels
        ):
            raise RuntimeError("weighted Blur level pool is incomplete")
        weights = [float(probabilities[level]) for level in self.levels]
        if any(weight <= 0.0 for weight in weights) or not math.isclose(
            sum(weights), 1.0, rel_tol=0.0, abs_tol=1e-12
        ):
            raise RuntimeError(
                "weighted Blur level probabilities must be positive and sum to one"
            )
        cumulative = 0.0
        self.cumulative = []
        for level, weight in zip(self.levels, weights, strict=True):
            cumulative += weight
            self.cumulative.append((cumulative, level))
        self.datasets = {
            level: BlurPresentationDataset(tuple(rows_by_level[level]), (), config)
            for level in self.levels
        }

    def __len__(self) -> int:
        return 2**31 - 1

    def level_for_presentation(self, presentation: int) -> int:
        draw = int(
            stable_digest(self.seed, "qwen-blur-level", presentation)[:16], 16
        ) / 2**64
        for cumulative, level in self.cumulative:
            if draw < cumulative:
                return level
        return self.levels[-1]

    def __getitem__(self, presentation: int) -> dict[str, object]:
        level = self.level_for_presentation(presentation)
        item = self.datasets[level][presentation]
        item["qwen_blur_level"] = level
        return item


def configure_foundir_runtime_paths() -> None:
    """Pin the released FoundIR runtime before importing its model code."""

    required = (FOUNDIR_RUNTIME, *SPECIALIST_DEPENDENCY_DIRECTORIES)
    for path in required:
        if not path.is_dir():
            raise RuntimeError(f"missing released FoundIR runtime dependency: {path}")
    paths = [ROOT / "third_party/foundir", FOUNDIR_RUNTIME]
    paths.extend(SPECIALIST_DEPENDENCY_DIRECTORIES)
    for path in paths:
        while str(path) in sys.path:
            sys.path.remove(str(path))
    sys.path.insert(0, str(ROOT / "third_party/foundir"))
    sys.path.insert(0, str(FOUNDIR_RUNTIME))
    for dependency in reversed(SPECIALIST_DEPENDENCY_DIRECTORIES):
        sys.path.insert(0, str(dependency))


def validate_gpu_hardware(world: int) -> list[str]:
    if torch.cuda.device_count() != world:
        raise RuntimeError(
            f"Blur Restormer gate requires exactly {world} visible GPUs; "
            f"found {torch.cuda.device_count()}"
        )
    names = [torch.cuda.get_device_properties(index).name for index in range(world)]
    families = ["H200" if "H200" in name else "H100" if "H100" in name else None for name in names]
    if any(family is None for family in families) or len(set(families)) != 1:
        raise RuntimeError(f"Blur Restormer gate requires homogeneous H100 or H200 GPUs: {names}")
    return names


def validate_config(config: Mapping[str, object]) -> tuple[dict, dict, dict, Path]:
    if (
        config.get("schema_version") != 1
        or config.get("status") != "development_only_blur_restormer_refinement_gate"
        or config.get("checkpoint_selection_uses_validation_only") is not True
        or config.get("sealed_internal_test_accessed") is not False
        or config.get("protected_foundir_test_accessed") is not False
    ):
        raise RuntimeError("Blur Restormer gate contract is invalid")
    for path_key, hash_key in (
        ("router_config", "router_config_sha256"),
        ("router_development_result", "router_development_result_sha256"),
        ("router_development_freeze", "router_development_freeze_sha256"),
        ("restormer_source", "restormer_source_sha256"),
        ("initial_checkpoint", "initial_checkpoint_sha256"),
    ):
        path = resolve_workspace_path(ROOT, config[path_key])
        if file_sha256(path) != config[hash_key]:
            raise RuntimeError(f"Blur Restormer gate identity changed: {path_key}")
    router_path = resolve_workspace_path(ROOT, config["router_config"])
    router, parent, specialization, blur_checkpoint = validate_router_config(router_path)
    if (
        router["routing"].get("policy") != "development_selected_confidence_guarded_v2"
        or int(router["routing"]["blur_level_minimum"]) != 4
        or router["protected_foundir_test_accessed"] is not False
    ):
        raise RuntimeError("Blur Restormer gate requires frozen integrated v2 routing")
    development = json.loads(
        resolve_workspace_path(ROOT, config["router_development_result"]).read_text()
    )
    freeze = json.loads(
        resolve_workspace_path(ROOT, config["router_development_freeze"]).read_text()
    )
    if (
        development.get("status") != "qwen_routed_integrated_development_complete"
        or development.get("records") != 660
        or development.get("protected_foundir_test_accessed") is not False
        or freeze.get("status") != "frozen_qwen_routed_integrated_development_result"
        or freeze.get("development_result_sha256")
        != config["router_development_result_sha256"]
        or freeze.get("sealed_internal_test_accessed") is not False
        or freeze.get("protected_foundir_test_accessed") is not False
    ):
        raise RuntimeError("Blur Restormer gate requires a valid frozen v2 development result")
    training = config["training"]
    if int(training["world_size"]) != 8 or int(training["global_batch_size"]) != (
        int(training["world_size"])
        * int(training["batch_size_per_rank"])
        * int(training["gradient_accumulation_steps"])
    ):
        raise RuntimeError("Blur Restormer global batch contract changed")
    if (
        int(training["num_workers_per_rank"]) < 1
        or int(training["data_loader_timeout_seconds"]) < 1
    ):
        raise RuntimeError("Blur Restormer bounded DataLoader contract changed")
    model = config["model"]
    architecture = model.get("architecture")
    model_contracts = {
        "released_foundir_restormer_26m": {
            "parameters": 26_126_644,
            "input": "frozen_selected_blur_base_output",
            "output": "full_refined_rgb_with_internal_residual_skip",
            "training_categories": [
                "01Blur",
                "02Blur_Noise",
                "15Lowlight_Blur",
                "18Lowlight_Blur_Noise",
            ],
            "expected_training_records": 244_998,
            "crop_size": 256,
        },
        "released_foundir_restormer_26m_base_lq_fusion": {
            "parameters": 26_126_887,
            "input": "frozen_selected_blur_base_and_aligned_lq",
            "output": "full_refined_rgb_with_internal_residual_skip",
            "training_categories": ["01Blur"],
            "expected_training_records": 105_057,
            "crop_size": 512,
        },
        "released_foundir_restormer_26m_base_lq_fusion_corrective": {
            "parameters": 26_126_887,
            "input": "frozen_selected_blur_base_and_aligned_lq",
            "output": "full_refined_rgb_with_internal_residual_skip",
            "training_categories": ["01Blur"],
            "expected_training_records": 85_153,
            "crop_size": 512,
        },
        MOTION_ARCHITECTURE: {
            "parameters": 26_126_887,
            "input": "frozen_selected_blur_base_and_aligned_lq",
            "output": "task_native_motion_deblur_rgb_blended_from_frozen_base",
            "training_categories": ["01Blur"],
            "expected_training_records": 85_153,
            "crop_size": 512,
        },
    }
    if architecture not in model_contracts:
        raise RuntimeError("unsupported Blur Restormer model architecture")
    expected_model = model_contracts[architecture]
    data = config["data"]
    if (
        list(data["training_categories"]) != expected_model["training_categories"]
        or int(data["expected_training_records"])
        != expected_model["expected_training_records"]
        or list(data["validation_categories"]) != ["01Blur", "05Noise"]
        or int(data["validation_records_per_category"]) != 60
        or int(data["crop_size"]) != expected_model["crop_size"]
        or float(data["noise_identity_presentation_probability"]) != 0.0
        or float(data["identity_probability"]) != 0.0
    ):
        raise RuntimeError("Blur Restormer data contract changed")
    allowed_initializations = {
        "released_weather_restormer",
        "released_weather_restormer_with_zero_output_residual_head",
        "v3_step_1000_corrective_finetune",
        "upstream_motion_deblurring_mlx_fp32_lq_fusion",
    }
    if (
        int(model["parameters"]) != expected_model["parameters"]
        or model.get("input") != expected_model["input"]
        or model.get("output") != expected_model["output"]
        or model.get("initialization") not in allowed_initializations
    ):
        raise RuntimeError("Blur Restormer model contract changed")
    loss = training["loss"]
    common_optimization = (
        int(training["batch_size_per_rank"]) == 2
        and int(training["gradient_accumulation_steps"]) == 2
        and int(training["global_batch_size"]) == 32
        and float(training["weight_decay"]) == 1e-4
        and float(training["max_grad_norm"]) == 1.0
        and float(loss["edge_l1_weight"]) == 0.05
        and float(loss["lpips_weight"]) == 0.1
        and int(loss["lpips_every_steps"]) == 1
    )
    legacy_optimization = (
        architecture
        not in {
            "released_foundir_restormer_26m_base_lq_fusion_corrective",
            MOTION_ARCHITECTURE,
        }
        and int(training["optimizer_steps"]) == 12_000
        and float(training["learning_rate"]) == 1e-4
        and float(training["minimum_learning_rate"]) == 1e-6
        and int(training["warmup_steps"]) == 200
        and float(loss["charbonnier_weight"]) == 1.0
        and float(loss["rgb_mse_weight"]) == 1.0
        and float(loss.get("psnr_surrogate_weight", 0.0)) == 0.0
    )
    corrective_optimization = (
        architecture == "released_foundir_restormer_26m_base_lq_fusion_corrective"
        and int(training["optimizer_steps"]) == 3_000
        and float(training["learning_rate"]) == 2e-5
        and float(training["minimum_learning_rate"]) == 2e-6
        and int(training["warmup_steps"]) == 50
        and float(loss["charbonnier_weight"]) == 0.25
        and float(loss["rgb_mse_weight"]) == 10.0
        and float(loss["psnr_surrogate_weight"]) == 0.02
        and float(loss["psnr_surrogate_epsilon"]) == 1e-4
    )
    motion_optimization = (
        architecture == MOTION_ARCHITECTURE
        and int(training["optimizer_steps"]) == 3_000
        and float(training["learning_rate"]) == 2e-5
        and float(training["minimum_learning_rate"]) == 2e-6
        and int(training["warmup_steps"]) == 50
        and float(training["core_learning_rate_scale"]) == 0.1
        and float(training["fusion_learning_rate_scale"]) == 1.0
        and float(loss["charbonnier_weight"]) == 1.0
        and float(loss["rgb_mse_weight"]) == 1.0
        and float(loss.get("psnr_surrogate_weight", 0.0)) == 0.0
    )
    if not common_optimization or not (
        legacy_optimization or corrective_optimization or motion_optimization
    ):
        raise RuntimeError("Blur Restormer optimization contract changed")
    if architecture == "released_foundir_restormer_26m_base_lq_fusion_corrective":
        source = config.get("correction_source", {})
        for path_key, hash_key in (
            ("config", "config_sha256"),
            ("checkpoint", "checkpoint_sha256"),
            ("development_result", "development_result_sha256"),
        ):
            path = resolve_workspace_path(ROOT, source[path_key])
            if file_sha256(path) != source[hash_key]:
                raise RuntimeError(f"Blur Restormer corrective source changed: {path_key}")
        payload = torch.load(
            resolve_workspace_path(ROOT, source["checkpoint"]),
            map_location="cpu",
            weights_only=False,
        )
        if (
            payload.get("schema_version") != 1
            or payload.get("step") != 1_000
            or payload.get("config_sha256") != source["config_sha256"]
            or payload.get("sealed_internal_test_accessed") is not False
            or payload.get("protected_foundir_test_accessed") is not False
        ):
            raise RuntimeError("invalid Blur Restormer corrective checkpoint")
        result = json.loads(
            resolve_workspace_path(ROOT, source["development_result"]).read_text()
        )
        if (
            result.get("step") != 1_000
            or result.get("checkpoint_sha256") != source["checkpoint_sha256"]
            or result.get("config_sha256") != source["config_sha256"]
            or result.get("sealed_internal_test_accessed") is not False
            or result.get("protected_foundir_test_accessed") is not False
        ):
            raise RuntimeError("invalid Blur Restormer corrective development result")
        levels = [int(value) for value in data.get("training_qwen_blur_levels", [])]
        expected_by_level = {
            int(key): int(value)
            for key, value in data.get(
                "expected_training_records_by_qwen_blur_level", {}
            ).items()
        }
        probabilities = {
            int(key): float(value)
            for key, value in data.get(
                "qwen_blur_level_sampling_probabilities", {}
            ).items()
        }
        if (
            levels != [4, 5]
            or expected_by_level != {4: 36_396, 5: 48_757}
            or probabilities != {4: 0.25, 5: 0.75}
            or model.get("initialization") != "v3_step_1000_corrective_finetune"
            or evaluation_residual_scales(config) != [0.0, 0.5, 0.75, 1.0, 1.25]
            or model.get("residual_scale_selection")
            != "maximize_worst_gate_margin_then_closest_to_one_then_lower_scale"
        ):
            raise RuntimeError("Blur Restormer corrective data/model contract changed")
        continuation = config.get("continuation", {})
        if (
            continuation.get("policy")
            != "fail_closed_improve_worst_margin_with_psnr_base_and_v3_lpips_floors"
            or not math.isclose(
                float(continuation.get("minimum_blur_psnr")),
                float(result["blur"]["base"]["psnr"]),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or not math.isclose(
                float(continuation.get("maximum_blur_lpips")),
                float(result["blur"]["refined_routed"]["lpips"]),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or not math.isclose(
                float(continuation.get("initial_selection_score")),
                selection_score(result, config["gate"]),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or float(continuation.get("minimum_selection_score_gain")) != 0.0
            or continuation.get("stop_on_gate_pass") is not True
        ):
            raise RuntimeError("Blur Restormer corrective continuation contract changed")
    if architecture == MOTION_ARCHITECTURE:
        levels = [int(value) for value in data.get("training_qwen_blur_levels", [])]
        expected_by_level = {
            int(key): int(value)
            for key, value in data.get(
                "expected_training_records_by_qwen_blur_level", {}
            ).items()
        }
        probabilities = {
            int(key): float(value)
            for key, value in data.get(
                "qwen_blur_level_sampling_probabilities", {}
            ).items()
        }
        provenance = config.get("pretrained_source", {})
        if (
            levels != [4, 5]
            or expected_by_level != {4: 36_396, 5: 48_757}
            or probabilities != {4: 0.2, 5: 0.8}
            or model.get("initialization")
            != "upstream_motion_deblurring_mlx_fp32_lq_fusion"
            or evaluation_residual_scales(config)
            != [0.0, 0.25, 0.5, 0.75, 1.0, 1.25]
            or model.get("residual_scale_selection")
            != "maximize_balanced_worst_gate_margin_then_closest_to_one_then_lower_scale"
            or provenance
            != {
                "repository": "mlx-community/Restormer-motion-deblurring-fp32",
                "revision": "210447d2bdbec96ecdc35bc4cc0bb5d42daefbc3",
                "filename": "model.safetensors",
                "upstream_repository": "swz30/Restormer",
                "upstream_checkpoint": "motion_deblurring.pth",
                "license": "MIT",
            }
            or float(config["gate"].get("psnr_margin_unit_db")) != 0.1
            or float(config["gate"].get("lpips_margin_unit")) != 0.01
        ):
            raise RuntimeError("motion Restormer data/model/provenance contract changed")
        base_reference = {
            "blur": {"refined_routed": development["task_metrics"]["blur"]},
            "noise": {"refined_routed": development["task_metrics"]["noise"]},
        }
        continuation = config.get("continuation", {})
        base_blur = development["task_metrics"]["blur"]
        if (
            continuation.get("policy")
            != "fail_closed_balanced_progress_from_frozen_base"
            or not math.isclose(
                float(continuation.get("minimum_blur_psnr")),
                float(base_blur["psnr"]) - 0.1,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or not math.isclose(
                float(continuation.get("maximum_blur_lpips")),
                float(base_blur["lpips"]),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or not math.isclose(
                float(continuation.get("initial_selection_score")),
                selection_score(base_reference, config["gate"]),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or float(continuation.get("minimum_selection_score_gain")) != 0.05
            or continuation.get("stop_on_gate_pass") is not True
        ):
            raise RuntimeError("motion Restormer continuation contract changed")
    checkpoints = [int(value) for value in training["checkpoint_steps"]]
    if checkpoints != sorted(set(checkpoints)) or checkpoints[-1] != int(
        training["optimizer_steps"]
    ):
        raise RuntimeError("Blur Restormer checkpoint schedule changed")
    return router, parent, specialization, blur_checkpoint


def load_restormer(config: Mapping[str, object], parent: Mapping[str, object], device):
    checkpoint = resolve_workspace_path(ROOT, config["initial_checkpoint"])
    architecture = config["model"]["architecture"]
    if architecture == MOTION_ARCHITECTURE:
        verify_specialist_dependency_versions()
        Restormer = import_weather_specialist_architecture(
            ROOT / "third_party/foundir/specialist_model"
        )
        core = Restormer()
        load_mlx_motion_restormer_weights(core, checkpoint)
    else:
        core = build_weather_specialist(
            released_config(dict(parent), checkpoint), device
        )
    if not hasattr(core, "output") or not isinstance(core.output, torch.nn.Conv2d):
        raise RuntimeError("released Restormer residual output head changed")
    if config["model"]["initialization"] == (
        "released_weather_restormer_with_zero_output_residual_head"
    ):
        torch.nn.init.zeros_(core.output.weight)
        if core.output.bias is not None:
            torch.nn.init.zeros_(core.output.bias)
        if torch.count_nonzero(core.output.weight).item() != 0:
            raise RuntimeError("Restormer residual output head did not zero-initialize")
    if architecture == MOTION_ARCHITECTURE:
        model = BaseLQRestormerExpert(core)
    elif architecture in FUSION_ARCHITECTURES:
        model = BaseLQRestormerRefiner(core)
    else:
        model = core
    model = model.to(device)
    if config["model"]["initialization"] == "v3_step_1000_corrective_finetune":
        source = config["correction_source"]
        payload = torch.load(
            resolve_workspace_path(ROOT, source["checkpoint"]),
            map_location="cpu",
            weights_only=False,
        )
        model.load_state_dict(payload["model"], strict=True)
    parameter_devices = {parameter.device for parameter in model.parameters()}
    if parameter_devices != {torch.device(device)}:
        raise RuntimeError(
            f"Blur Restormer parameters span unexpected devices: {parameter_devices}"
        )
    model.requires_grad_(True).train()
    parameters = sum(parameter.numel() for parameter in model.parameters())
    if parameters != int(config["model"]["parameters"]):
        raise RuntimeError(f"Restormer parameter count changed: {parameters}")
    return model


def residual_output_head(model) -> torch.nn.Conv2d:
    core = (
        model.restormer
        if isinstance(model, (BaseLQRestormerRefiner, BaseLQRestormerExpert))
        else model
    )
    if not hasattr(core, "output") or not isinstance(core.output, torch.nn.Conv2d):
        raise RuntimeError("Restormer residual output head changed")
    return core.output


def refiner_prediction(
    model,
    base: torch.Tensor,
    lq: torch.Tensor,
    config: Mapping[str, object],
    *,
    residual_scale: float = 1.0,
) -> torch.Tensor:
    if config["model"]["architecture"] in FUSION_ARCHITECTURES:
        full = model(base, lq)
    else:
        full = model(base)
    return base + (full - base) * float(residual_scale)


def disable_foundir_sample_progress() -> None:
    """Prevent four-line tqdm output for every frozen-base training batch."""

    import src.model as foundir_model

    foundir_model.tqdm = lambda iterable, *args, **kwargs: iterable


def frozen_base_output(diffusion, source: torch.Tensor, task: str) -> torch.Tensor:
    with torch.inference_mode():
        output = diffusion.sample(
            source,
            batch_size=source.shape[0],
            last=True,
            task=task,
        )[-1]
    return output.float().clamp(0, 1)


def quantized(value: torch.Tensor) -> torch.Tensor:
    return value.float().clamp(0, 1).mul(255).round().div(255)


def loss_terms(
    prediction: torch.Tensor,
    target: torch.Tensor,
    config: Mapping[str, object],
    lpips_metric,
    *,
    step: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    contract = config["training"]["loss"]
    difference = prediction.float() - target.float()
    charbonnier = torch.sqrt(
        difference.square() + float(contract["charbonnier_epsilon"]) ** 2
    ).mean()
    mse = difference.square().mean()
    per_image_mse = difference.square().flatten(1).mean(1)
    psnr_surrogate = torch.log1p(
        per_image_mse / float(contract.get("psnr_surrogate_epsilon", 1e-4))
    ).mean()
    edge = F.l1_loss(
        spatial_highpass(prediction.float()), spatial_highpass(target.float())
    )
    lpips_active = step % int(contract["lpips_every_steps"]) == 0
    if lpips_active:
        with torch.autocast(device_type=prediction.device.type, enabled=False):
            perceptual = lpips_metric(prediction.float().clamp(0, 1), target.float())
    else:
        perceptual = prediction.new_zeros((), dtype=torch.float32)
    loss = (
        charbonnier * float(contract["charbonnier_weight"])
        + mse * float(contract["rgb_mse_weight"])
        + edge * float(contract["edge_l1_weight"])
        + perceptual * float(contract["lpips_weight"])
        + psnr_surrogate * float(contract.get("psnr_surrogate_weight", 0.0))
    )
    return loss, {
        "loss": float(loss.detach()),
        "charbonnier": float(charbonnier.detach()),
        "mse": float(mse.detach()),
        "psnr_surrogate": float(psnr_surrogate.detach()),
        "edge_l1": float(edge.detach()),
        "lpips": float(perceptual.detach()),
        "lpips_active": float(lpips_active),
    }


def save_checkpoint(path: Path, model, optimizer, *, step: int, config_sha: str) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite Blur Restormer checkpoint: {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        torch.save(
            {
                "schema_version": 1,
                "config_sha256": config_sha,
                "step": step,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "sealed_internal_test_accessed": False,
                "protected_foundir_test_accessed": False,
            },
            temporary,
        )
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def resume(
    output: Path,
    model,
    optimizer,
    config_sha: str,
    checkpoint_steps: Sequence[int],
) -> int:
    candidates = sorted(output.glob("step_*.pt"))
    if not candidates:
        return 0
    path = candidates[-1]
    payload = torch.load(path, map_location="cpu", weights_only=False)
    step = int(path.stem.split("_", 1)[1])
    if (
        step not in checkpoint_steps
        or payload.get("schema_version") != 1
        or payload.get("config_sha256") != config_sha
        or payload.get("step") != step
        or payload.get("sealed_internal_test_accessed") is not False
        or payload.get("protected_foundir_test_accessed") is not False
    ):
        raise RuntimeError(f"invalid Blur Restormer resume checkpoint: {path}")
    model.load_state_dict(payload["model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    return step


def evaluation_seed(router: Mapping[str, object], record_id: str) -> int:
    return int(
        stable_digest(
            int(router["evaluation"]["seed"]),
            "qwen-routed-integrated",
            record_id,
        )[:16],
        16,
    )


def evaluate(
    refiner,
    generalist,
    blur_model,
    rows: Sequence[Mapping[str, str]],
    config: Mapping[str, object],
    router: Mapping[str, object],
    parent: Mapping[str, object],
    metric,
    device: torch.device,
    rank: int,
    world: int,
) -> list[dict[str, object]]:
    refiner.eval()
    local = []
    error = None
    try:
        for index in range(rank, len(rows), world):
            row = rows[index]
            lq, gt, _ = load_aligned_crop(
                row,
                crop_size=int(router["evaluation"]["crop_size"]),
                seed=int(parent["data"]["split_seed"]),
            )
            source = ((lq.unsqueeze(0).to(device) + 1.0) * 0.5).float()
            target = ((gt.unsqueeze(0).to(device) + 1.0) * 0.5).float()
            decision = route_decision(json.loads(row["qwen_json"]), router["routing"])
            seed = evaluation_seed(router, row["record_id"])
            random.seed(seed)
            np.random.seed(seed % (2**32 - 1))
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            base_model = (
                blur_model if decision["base"] == "blur_specialization" else generalist
            )
            base = frozen_base_output(base_model, source, row["record_id"])
            scales = evaluation_residual_scales(config)
            if decision["base"] == "blur_specialization":
                with torch.inference_mode(), torch.autocast(
                    device_type="cuda", dtype=torch.bfloat16
                ):
                    full = refiner_prediction(refiner, base, source, config).float()
            base_q = quantized(base)
            candidates = {}
            for scale in scales:
                restored = (
                    base + (full - base) * scale
                    if decision["base"] == "blur_specialization"
                    else base
                )
                candidates[residual_scale_key(scale)] = scalar_metrics(
                    metric, target, quantized(restored)
                )
            local.append(
                {
                    "record_id": row["record_id"],
                    "category": row["category"],
                    "route": decision["route"],
                    "base": scalar_metrics(metric, target, base_q),
                    "refined_routed_by_scale": candidates,
                }
            )
    except Exception as exc:  # propagated collectively before all_gather_object
        error = f"{type(exc).__name__}: {exc}"
    synchronized_rank_error(error, device=device, label="Blur Restormer evaluation failure")
    gathered: list[list[dict[str, object]] | None] = [None for _ in range(world)]
    dist.all_gather_object(gathered, local)
    refiner.train()
    return [row for shard in gathered if shard is not None for row in shard]


def summary(
    rows: Sequence[Mapping[str, object]],
    *,
    step: int,
    config: Mapping[str, object],
    router_development: Mapping[str, object],
) -> dict[str, object]:
    scales = evaluation_residual_scales(config)
    candidate_summaries = {}
    for scale in scales:
        key = residual_scale_key(scale)
        if all("refined_routed_by_scale" in row for row in rows):
            projected = [
                {**row, "refined_routed": row["refined_routed_by_scale"][key]}
                for row in rows
            ]
        else:
            if scales != [1.0]:
                raise RuntimeError("missing Blur Restormer residual-scale metrics")
            projected = list(rows)
        candidate_summaries[key] = {
            "residual_scale": scale,
            "blur": mean_metrics(projected, "01Blur", "refined_routed"),
            "noise": mean_metrics(projected, "05Noise", "refined_routed"),
        }
    winner = max(
        candidate_summaries.values(),
        key=lambda item: (
            selection_score(
                {
                    "blur": {"refined_routed": item["blur"]},
                    "noise": {"refined_routed": item["noise"]},
                },
                config["gate"],
            ),
            -abs(float(item["residual_scale"]) - 1.0),
            -float(item["residual_scale"]),
        ),
    )
    result = {
        "schema_version": 1,
        "status": "blur_restormer_refinement_development_complete",
        "step": step,
        "records": len(rows),
        "blur": {
            "base": mean_metrics(rows, "01Blur", "base"),
            "refined_routed": winner["blur"],
            "qwen_blur_l4plus_records": sum(
                row["route"].startswith("blur_specialization")
                for row in rows
                if row["category"] == "01Blur"
            ),
        },
        "noise": {
            "base": mean_metrics(rows, "05Noise", "base"),
            "refined_routed": winner["noise"],
            "qwen_false_blur_l4plus_records": sum(
                row["route"].startswith("blur_specialization")
                for row in rows
                if row["category"] == "05Noise"
            ),
        },
        "split": "validation",
        "selected_residual_scale": winner["residual_scale"],
        "residual_scale_candidates": candidate_summaries,
        "sealed_internal_test_accessed": False,
        "protected_foundir_test_accessed": False,
    }
    if config["gate"]["require_exact_v2_base_metric_reproduction"]:
        tolerance = float(config["gate"]["base_metric_tolerance"])
        for task in ("blur", "noise"):
            expected = router_development["task_metrics"][task]
            actual = result[task]["base"]
            for metric in ("psnr", "lpips"):
                if abs(float(actual[metric]) - float(expected[metric])) > tolerance:
                    raise RuntimeError(
                        f"Blur Restormer {task} base {metric} does not reproduce integrated v2"
                    )
    return result


def selection_score(result: Mapping[str, object], gate: Mapping[str, object]) -> float:
    blur = result["blur"]["refined_routed"]
    noise = result["noise"]["refined_routed"]
    psnr_unit = float(gate.get("psnr_margin_unit_db", 1.0))
    lpips_unit = float(gate.get("lpips_margin_unit", 0.01))
    if (
        not math.isfinite(psnr_unit)
        or not math.isfinite(lpips_unit)
        or psnr_unit <= 0.0
        or lpips_unit <= 0.0
    ):
        raise RuntimeError("invalid Blur Restormer gate margin units")
    return min(
        (float(blur["psnr"]) - float(gate["blur_psnr_minimum"])) / psnr_unit,
        (float(gate["blur_lpips_maximum"]) - float(blur["lpips"])) / lpips_unit,
        (float(noise["psnr"]) - float(gate["noise_psnr_minimum"])) / psnr_unit,
        (float(gate["noise_lpips_maximum"]) - float(noise["lpips"])) / lpips_unit,
    )


def gate_checks(result: Mapping[str, object], gate: Mapping[str, object]) -> dict[str, bool]:
    blur = result["blur"]["refined_routed"]
    noise = result["noise"]["refined_routed"]
    return {
        "blur_psnr": float(blur["psnr"]) >= float(gate["blur_psnr_minimum"]),
        "blur_lpips": float(blur["lpips"]) <= float(gate["blur_lpips_maximum"]),
        "noise_psnr": float(noise["psnr"]) >= float(gate["noise_psnr_minimum"]),
        "noise_lpips": float(noise["lpips"]) <= float(gate["noise_lpips_maximum"]),
    }


def continuation_decision(
    result: Mapping[str, object],
    config: Mapping[str, object],
    *,
    previous_selection_score: float,
) -> dict[str, object]:
    checks = gate_checks(result, config["gate"])
    score = selection_score(result, config["gate"])
    step = int(result["step"])
    total_steps = int(config["training"]["optimizer_steps"])
    contract = config.get("continuation")
    reasons = []
    if all(checks.values()) and contract and contract["stop_on_gate_pass"]:
        reasons.append("gate_pass")
    elif step >= total_steps:
        reasons.append("final_milestone")
    elif contract:
        blur = result["blur"]["refined_routed"]
        if float(blur["psnr"]) < float(contract["minimum_blur_psnr"]):
            reasons.append(
                "blur_psnr_below_allowed_floor"
                if contract.get("policy")
                == "fail_closed_balanced_progress_from_frozen_base"
                else "blur_psnr_below_frozen_base"
            )
        if float(blur["lpips"]) > float(contract["maximum_blur_lpips"]):
            reasons.append(
                "blur_lpips_worse_than_frozen_base"
                if contract.get("policy")
                == "fail_closed_balanced_progress_from_frozen_base"
                else "blur_lpips_worse_than_v3_source"
            )
        required = previous_selection_score + float(
            contract["minimum_selection_score_gain"]
        )
        if score <= required:
            reasons.append("worst_normalized_margin_did_not_improve")
    continue_training = not reasons
    return {
        "schema_version": 1,
        "status": "blur_restormer_milestone_continuation_decision",
        "step": step,
        "selection_score": score,
        "previous_selection_score": previous_selection_score,
        "checks": checks,
        "continue_training": continue_training,
        "reasons": reasons,
        "sealed_internal_test_accessed": False,
        "protected_foundir_test_accessed": False,
    }


def cpu_preflight(
    config_path: Path,
    config: Mapping[str, object],
    config_sha: str,
    parent: Mapping[str, object],
    specialization: Mapping[str, object],
    blur_checkpoint: Path,
) -> None:
    """Strictly load both model families without accessing any evaluation pixels."""

    configure_foundir_runtime_paths()
    device = torch.device("cpu")
    refiner = load_restormer(config, parent, device)
    refiner_parameters = sum(parameter.numel() for parameter in refiner.parameters())
    output_head = residual_output_head(refiner)
    refiner_output_head_nonzero = int(torch.count_nonzero(output_head.weight).item())
    probe = torch.linspace(0.0, 1.0, steps=3 * 16 * 16).reshape(1, 3, 16, 16)
    with torch.inference_mode():
        identity_initialization = torch.equal(
            refiner_prediction(refiner, probe, probe.flip(-1), config), probe
        )
    if (
        config["model"]["initialization"]
        == "released_weather_restormer_with_zero_output_residual_head"
        and not identity_initialization
    ):
        raise RuntimeError("zero-head Restormer is not an exact identity")
    del refiner
    pretrained = resolve_workspace_path(ROOT, specialization["pretrained_checkpoint"])
    blur_model = build_diffusion(specialization, pretrained, device)
    payload = torch.load(blur_checkpoint, map_location="cpu", weights_only=False)
    blur_model.model.load_state_dict(payload["model"], strict=True)
    blur_parameters = sum(parameter.numel() for parameter in blur_model.model.parameters())
    del payload, blur_model
    print(
        "FOUNDIR_BLUR_RESTORMER_PREFLIGHT_PASS "
        + json.dumps(
            {
                "config": config_path.relative_to(ROOT).as_posix(),
                "config_sha256": config_sha,
                "restormer_parameters": refiner_parameters,
                "restormer_output_head_nonzero": refiner_output_head_nonzero,
                "identity_initialization": identity_initialization,
                "initialization": config["model"]["initialization"],
                "blur_base_parameters": blur_parameters,
                "sealed_internal_test_accessed": False,
                "protected_foundir_test_accessed": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    config_path = resolve_workspace_path(ROOT, args.config)
    config = json.loads(config_path.read_text())
    config_sha = file_sha256(config_path)
    router, parent, specialization, blur_checkpoint = validate_config(config)
    if args.preflight:
        cpu_preflight(
            config_path,
            config,
            config_sha,
            parent,
            specialization,
            blur_checkpoint,
        )
        return
    configure_foundir_runtime_paths()

    rank, world, local_rank = torchrun_worker_identity(
        int(config["training"]["world_size"])
    )
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl")
    validate_gpu_hardware(world)

    rows = load_rows(
        resolve_workspace_path(ROOT, router["conditions"]),
        int(parent["data"]["expected_five_type_records"]),
    )
    training_categories = set(config["data"]["training_categories"])
    training_rows = [
        row
        for row in rows
        if row["split"] == config["data"]["training_split"]
        and row["category"] in training_categories
    ]
    weighted_probabilities = config["data"].get(
        "qwen_blur_level_sampling_probabilities"
    )
    rows_by_level = None
    if weighted_probabilities is not None:
        levels = set(
            int(value) for value in config["data"]["training_qwen_blur_levels"]
        )
        rows_by_level = {level: [] for level in levels}
        for row in training_rows:
            level = int(json.loads(row["qwen_json"])["blur"])
            if level in rows_by_level:
                rows_by_level[level].append(row)
        expected = {
            int(key): int(value)
            for key, value in config["data"][
                "expected_training_records_by_qwen_blur_level"
            ].items()
        }
        actual = {
            level: len(level_rows) for level, level_rows in rows_by_level.items()
        }
        if actual != expected:
            raise RuntimeError(
                f"Blur Restormer routed training coverage changed: {actual}"
            )
        training_rows = [
            row for level in sorted(rows_by_level) for row in rows_by_level[level]
        ]
    if len(training_rows) != int(config["data"]["expected_training_records"]):
        raise RuntimeError(f"Blur Restormer training coverage changed: {len(training_rows)}")
    validation = selected_from_contract(
        parent,
        rows,
        split=config["data"]["validation_split"],
        config_sha=file_sha256(resolve_workspace_path(ROOT, router["parent_config"])),
        conditions_sha=file_sha256(resolve_workspace_path(ROOT, router["conditions"])),
    )
    validation_categories = set(config["data"]["validation_categories"])
    validation = [row for row in validation if row["category"] in validation_categories]
    counts = defaultdict(int)
    for row in validation:
        counts[row["category"]] += 1
    if any(
        counts[category] != int(config["data"]["validation_records_per_category"])
        for category in validation_categories
    ):
        raise RuntimeError(f"Blur Restormer validation coverage changed: {dict(counts)}")

    disable_foundir_sample_progress()
    pretrained = resolve_workspace_path(ROOT, specialization["pretrained_checkpoint"])
    generalist = build_diffusion(specialization, pretrained, device).eval().requires_grad_(False)
    blur_model = build_diffusion(specialization, pretrained, device)
    payload = torch.load(blur_checkpoint, map_location="cpu", weights_only=False)
    blur_model.model.load_state_dict(payload["model"], strict=True)
    del payload
    blur_model.eval().requires_grad_(False)
    refiner = load_restormer(config, parent, device)
    wrapped = DistributedDataParallel(refiner, device_ids=[local_rank], broadcast_buffers=False)
    optimizer_parameters = wrapped.parameters()
    if config["model"]["architecture"] == MOTION_ARCHITECTURE:
        core_scale = float(config["training"]["core_learning_rate_scale"])
        fusion_scale = float(config["training"]["fusion_learning_rate_scale"])
        peak_lr = float(config["training"]["learning_rate"])
        optimizer_parameters = [
            {
                "params": wrapped.module.restormer.parameters(),
                "lr": peak_lr * core_scale,
                "lr_scale": core_scale,
                "component": "motion_restormer_core",
            },
            {
                "params": wrapped.module.fusion.parameters(),
                "lr": peak_lr * fusion_scale,
                "lr_scale": fusion_scale,
                "component": "base_lq_fusion",
            },
        ]
    optimizer = torch.optim.AdamW(
        optimizer_parameters,
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
        fused=True,
    )
    output = resolve_workspace_path(ROOT, config["output"])
    output.mkdir(parents=True, exist_ok=True)
    checkpoints = set(int(value) for value in config["training"]["checkpoint_steps"])
    step = resume(output, wrapped.module, optimizer, config_sha, sorted(checkpoints))
    resumed = torch.tensor([step], device=device)
    dist.all_reduce(resumed)
    if int(resumed.item()) != step * world:
        raise RuntimeError("Blur Restormer ranks resumed different steps")
    initial_checkpoint = output / "step_000000.pt"
    if 0 in checkpoints and not initial_checkpoint.exists():
        if rank == 0:
            save_checkpoint(
                initial_checkpoint,
                wrapped.module,
                optimizer,
                step=0,
                config_sha=config_sha,
            )
        dist.barrier()

    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

    metric = (
        LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True)
        .to(device)
        .eval()
        .requires_grad_(False)
    )

    def evaluate_milestone(current_step: int) -> dict[str, object]:
        checkpoint = output / f"step_{current_step:06d}.pt"
        result_path = output / f"development_step_{current_step:06d}.json"
        if result_path.exists():
            existing = json.loads(result_path.read_text())
            if (
                existing.get("config_sha256") != config_sha
                or existing.get("step") != current_step
                or existing.get("checkpoint_sha256") != file_sha256(checkpoint)
                or existing.get("sealed_internal_test_accessed") is not False
                or existing.get("protected_foundir_test_accessed") is not False
            ):
                raise RuntimeError(f"invalid existing Blur Restormer milestone: {result_path}")
            return existing
        result_rows = evaluate(
            wrapped.module,
            generalist,
            blur_model,
            validation,
            config,
            router,
            parent,
            metric,
            device,
            rank,
            world,
        )
        router_development = json.loads(
            resolve_workspace_path(ROOT, config["router_development_result"]).read_text()
        )
        result = summary(
            result_rows,
            step=current_step,
            config=config,
            router_development=router_development,
        )
        result.update(
            config_sha256=config_sha,
            checkpoint=checkpoint.relative_to(ROOT).as_posix(),
            checkpoint_sha256=file_sha256(checkpoint),
            model_parameters=sum(parameter.numel() for parameter in wrapped.module.parameters()),
        )
        if rank == 0:
            write_once(
                result_path,
                (json.dumps(result, indent=2, sort_keys=True) + "\n").encode(),
            )
            print(
                "FOUNDIR_BLUR_RESTORMER_EVAL " + json.dumps(result, sort_keys=True),
                flush=True,
            )
        return result

    def milestone_continuation(
        milestone_result: Mapping[str, object],
    ) -> dict[str, object] | None:
        contract = config.get("continuation")
        if not contract:
            return None
        current_step = int(milestone_result["step"])
        earlier_steps = [
            value
            for value in sorted(checkpoints)
            if value < current_step
            and (output / f"development_step_{value:06d}.json").is_file()
        ]
        if earlier_steps:
            previous = json.loads(
                (output / f"development_step_{earlier_steps[-1]:06d}.json").read_text()
            )
            previous_score = selection_score(previous, config["gate"])
        else:
            previous_score = float(contract["initial_selection_score"])
        decision = continuation_decision(
            milestone_result,
            config,
            previous_selection_score=previous_score,
        )
        decision_error = None
        if rank == 0:
            try:
                decision_path = output / f"continuation_step_{current_step:06d}.json"
                content = (
                    json.dumps(decision, indent=2, sort_keys=True) + "\n"
                ).encode()
                if decision_path.exists():
                    if decision_path.read_bytes() != content:
                        raise RuntimeError(
                            "existing Blur Restormer continuation decision changed: "
                            f"{decision_path}"
                        )
                else:
                    write_once(decision_path, content)
                marker = (
                    "FOUNDIR_BLUR_RESTORMER_CONTINUE"
                    if decision["continue_training"]
                    else "FOUNDIR_BLUR_RESTORMER_AUTO_STOP"
                )
                print(marker + " " + json.dumps(decision, sort_keys=True), flush=True)
            except Exception as exc:
                decision_error = f"{type(exc).__name__}: {exc}"
        synchronized_rank_error(
            decision_error,
            device=device,
            label=f"Blur Restormer continuation failure at step {current_step}",
        )
        return decision

    stop_training = False
    if step in checkpoints:
        milestone_result = evaluate_milestone(step)
        dist.barrier()
        decision = milestone_continuation(milestone_result)
        dist.barrier()
        stop_training = decision is not None and not bool(decision["continue_training"])

    if rows_by_level is not None:
        dataset = QwenLevelWeightedBlurDataset(
            rows_by_level,
            {
                int(key): float(value)
                for key, value in weighted_probabilities.items()
            },
            config,
        )
    else:
        dataset = BlurPresentationDataset(training_rows, (), config)
    total_steps = int(config["training"]["optimizer_steps"])
    global_batch = int(config["training"]["global_batch_size"])
    sampler = RankPresentationSampler(
        start=step * global_batch,
        stop=total_steps * global_batch,
        rank=rank,
        world=world,
    )
    loader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=int(config["training"]["batch_size_per_rank"]),
        num_workers=int(config["training"]["num_workers_per_rank"]),
        pin_memory=True,
        persistent_workers=int(config["training"]["num_workers_per_rank"]) > 0,
        timeout=int(config["training"]["data_loader_timeout_seconds"]),
        collate_fn=collate,
    )
    iterator = iter(loader)
    accumulation = int(config["training"]["gradient_accumulation_steps"])
    optimizer.zero_grad(set_to_none=True)
    while step < total_steps and not stop_training:
        started = time.perf_counter()
        totals = defaultdict(float)
        step_batches, data_seconds = synchronized_loader_batches(
            iterator,
            accumulation,
            device=device,
        )
        totals["data_seconds"] = data_seconds
        for micro, batch in enumerate(step_batches):
            source = batch["source"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            train_seed = int(
                stable_digest(
                    int(config["data"]["seed"]),
                    "frozen-base-train",
                    step + 1,
                    micro,
                    rank,
                )[:16],
                16,
            )
            torch.manual_seed(train_seed)
            torch.cuda.manual_seed_all(train_seed)
            sync = wrapped.no_sync() if micro + 1 < accumulation else torch.enable_grad()
            forward_error = None
            with sync:
                try:
                    base = frozen_base_output(blur_model, source, "blur-refiner-train")
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        prediction = refiner_prediction(wrapped, base, source, config)
                    loss, terms = loss_terms(
                        prediction,
                        target,
                        config,
                        metric,
                        step=step + 1,
                    )
                except Exception as exc:  # synchronize before any rank enters DDP backward
                    forward_error = f"{type(exc).__name__}: {exc}"
                synchronized_rank_error(
                    forward_error,
                    device=device,
                    label=f"Blur Restormer forward/loss failure at step {step + 1}",
                )
                (loss / accumulation).backward()
            metric.reset()
            for key, value in terms.items():
                totals[key] += value / accumulation
        torch.nn.utils.clip_grad_norm_(
            wrapped.parameters(), float(config["training"]["max_grad_norm"])
        )
        step += 1
        lr = learning_rate(step, config)
        for group in optimizer.param_groups:
            group["lr"] = lr * float(group.get("lr_scale", 1.0))
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if step % 25 == 0 or step in checkpoints:
            values = torch.tensor([totals[key] for key in sorted(totals)], device=device)
            dist.all_reduce(values)
            values /= world
            if rank == 0:
                fields = " ".join(
                    f"{key}={float(value):.6f}"
                    for key, value in zip(sorted(totals), values, strict=True)
                )
                print(
                    f"FOUNDIR_BLUR_RESTORMER step={step}/{total_steps} "
                    f"lr={lr:.8f} seconds={time.perf_counter()-started:.3f} {fields}",
                    flush=True,
                )
        if step in checkpoints:
            checkpoint = output / f"step_{step:06d}.pt"
            if rank == 0:
                save_checkpoint(
                    checkpoint,
                    wrapped.module,
                    optimizer,
                    step=step,
                    config_sha=config_sha,
                )
            dist.barrier()
            milestone_result = evaluate_milestone(step)
            dist.barrier()
            decision = milestone_continuation(milestone_result)
            dist.barrier()
            stop_training = decision is not None and not bool(
                decision["continue_training"]
            )

    gate_pass = torch.zeros((), dtype=torch.int64, device=device)
    if rank == 0:
        evaluated_steps = [
            value
            for value in sorted(checkpoints)
            if (output / f"development_step_{value:06d}.json").is_file()
        ]
        if not evaluated_steps:
            raise RuntimeError("Blur Restormer stopped without a development milestone")
        candidates = [
            json.loads((output / f"development_step_{value:06d}.json").read_text())
            for value in evaluated_steps
        ]
        winner = max(
            candidates,
            key=lambda item: (selection_score(item, config["gate"]), -int(item["step"])),
        )
        blur = winner["blur"]["refined_routed"]
        noise = winner["noise"]["refined_routed"]
        checks = gate_checks(winner, config["gate"])
        continuation_paths = [
            (output / f"continuation_step_{value:06d}.json")
            for value in evaluated_steps
            if (output / f"continuation_step_{value:06d}.json").is_file()
        ]
        result = {
            "schema_version": 1,
            "status": "blur_restormer_refinement_gate_complete",
            "config": config_path.relative_to(ROOT).as_posix(),
            "config_sha256": config_sha,
            "training_records": len(training_rows),
            "sample_presentations": step * global_batch,
            "planned_sample_presentations": total_steps * global_batch,
            "completed_optimizer_steps": step,
            "planned_optimizer_steps": total_steps,
            "stopped_early": step < total_steps,
            "evaluated_steps": evaluated_steps,
            "continuation_decisions": [
                path.relative_to(ROOT).as_posix() for path in continuation_paths
            ],
            "model_parameters": sum(parameter.numel() for parameter in wrapped.module.parameters()),
            "selection_rule": config["gate"]["selection"],
            "selected_step": winner["step"],
            "checkpoint": winner["checkpoint"],
            "checkpoint_sha256": winner["checkpoint_sha256"],
            "selected_residual_scale": winner["selected_residual_scale"],
            "residual_scale_candidates": winner["residual_scale_candidates"],
            "selected_development_metrics": {"blur": winner["blur"], "noise": winner["noise"]},
            "checks": checks,
            "gate_pass": all(checks.values()),
            "sealed_internal_test_accessed": False,
            "protected_foundir_test_accessed": False,
        }
        write_once(output / "result.json", (json.dumps(result, indent=2, sort_keys=True) + "\n").encode())
        gate_pass.fill_(int(result["gate_pass"]))
        marker = "FOUNDIR_BLUR_RESTORMER_GATE_PASS" if result["gate_pass"] else "FOUNDIR_BLUR_RESTORMER_GATE_FAIL"
        print(marker + " " + json.dumps(result, sort_keys=True), flush=True)
    dist.broadcast(gate_pass, src=0)
    dist.barrier()
    dist.destroy_process_group()
    if not bool(gate_pass.item()):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
