#!/usr/bin/env python3
"""Train the terminal Noise-family cascade: frozen denoiser plus exposure corrector."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path
from typing import Mapping

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.nn.parallel import DistributedDataParallel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lowlight_restoration.flux_prompt_cache import file_sha256  # noqa: E402
from lowlight_restoration.foundir_flux2_protocol import resolve_workspace_path  # noqa: E402
from lowlight_restoration.task_native_restormer import load_task_native_restormer  # noqa: E402
from smoke.train_foundir_noise_family_restormer_pilot_ddp import (  # noqa: E402
    CATEGORIES,
    category_balanced_batch,
    evaluate,
    finalize,
    load_rows,
)
from smoke.train_foundir_task_native_restormer_adaptation_ddp import (  # noqa: E402
    crop_size_for_step,
    learning_rate,
    load_training_patch,
    move_optimizer_state,
    raise_if_any_rank_failed,
)


class NoiseExposureCascade(nn.Module):
    def __init__(self, noise: nn.Module, exposure: nn.Module) -> None:
        super().__init__()
        self.noise = noise.eval().requires_grad_(False)
        self.exposure = exposure

    def train(self, mode: bool = True):
        super().train(mode)
        self.noise.eval()
        return self

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            denoised = self.noise(image).clamp(0, 1)
        return self.exposure(denoised)


def initialize_identity_exposure(model: nn.Module) -> None:
    if not hasattr(model, "output") or not isinstance(model.output, nn.Conv2d):
        raise RuntimeError("exposure Restormer output layer changed")
    with torch.no_grad():
        model.output.weight.zero_()
        if model.output.bias is not None:
            model.output.bias.zero_()
    if int(torch.count_nonzero(model.output.weight)) != 0:
        raise RuntimeError("exposure corrector is not exact identity initialized")


def validate_config(path: Path) -> dict:
    config = json.loads(path.read_text())
    data = config.get("data", {})
    optimization = config.get("optimization", {})
    expected_gates = {
        "05Noise": {"mean_psnr_minimum": 39.9, "mean_lpips_maximum": 0.08},
        "16Lowlight_Noise": {
            "mean_psnr_minimum": 18.733316631721234,
            "mean_lpips_maximum": 0.2582512099761516,
        },
    }
    if (
        config.get("schema_version") != 1
        or config.get("status") != "user_authorized_noise_family_cascade_terminal_round"
        or config.get("task") != "noise_family"
        or config.get("method")
        != "frozen_noise_restormer_then_identity_initialized_exposure_restormer"
        or config.get("categories") != list(CATEGORIES)
        or int(config.get("manifest_expected_records", -1)) != 2304
        or config.get("base_checkpoint_name") != "real_denoising"
        or int(data.get("train_records_per_category", -1)) != 1024
        or int(data.get("validation_records_per_category", -1)) != 128
        or int(data.get("train_source_groups", -1)) != 768
        or int(data.get("validation_source_groups", -1)) != 128
        or data.get("sampling") != "equal_category_then_uniform_source_group_then_uniform_record"
        or data.get("geometric_augmentation") is not True
        or int(data.get("validation_crop_size", -1)) != 512
        or int(data.get("validation_crop_seed", -1)) != 2026091902
        or int(optimization.get("world_size", -1)) != 8
        or int(optimization.get("global_batch", -1)) != 32
        or int(optimization.get("micro_batch_per_gpu", -1)) != 4
        or int(optimization.get("gradient_accumulation", -1)) != 1
        or int(optimization.get("total_steps", -1)) != 1000
        or optimization.get("milestones") != [500, 1000]
        or optimization.get("progressive_crops") != [[0, 128], [500, 192]]
        or optimization.get("optimizer") != "AdamW"
        or float(optimization.get("learning_rate", -1)) != 5e-5
        or float(optimization.get("minimum_learning_rate", -1)) != 1e-6
        or int(optimization.get("warmup_steps", -1)) != 100
        or optimization.get("betas") != [0.9, 0.999]
        or float(optimization.get("weight_decay", -1)) != 1e-4
        or float(optimization.get("gradient_clip_norm", -1)) != 1.0
        or optimization.get("loss") != "rgb_l1"
        or optimization.get("precision") != "bfloat16_autocast_fp32_parameters"
        or int(optimization.get("seed", -1)) != 2026091914
        or config.get("per_category_gates") != expected_gates
        or config.get("selection_rule")
        != "maximize_worst_per_category_normalized_gate_margin_then_earlier_step"
        or config.get("terminal_round_no_followup_tuning") is not True
        or config.get("development_pixels_accessed") is not False
        or config.get("official_foundir_test_accessed") is not False
    ):
        raise RuntimeError("Noise-family cascade terminal contract changed")
    artifacts = (
        ("manifest", "manifest_sha256"),
        ("base_checkpoint", "base_checkpoint_sha256"),
        ("frozen_noise_checkpoint", "frozen_noise_checkpoint_sha256"),
        ("failed_shared_model_selection", "failed_shared_model_selection_sha256"),
    )
    for path_key, sha_key in artifacts:
        artifact = resolve_workspace_path(ROOT, config[path_key])
        if not artifact.is_file() or file_sha256(artifact) != config[sha_key]:
            raise RuntimeError(f"Noise-family cascade artifact changed: {path_key}")
    noise = torch.load(
        resolve_workspace_path(ROOT, config["frozen_noise_checkpoint"]),
        map_location="cpu",
        weights_only=False,
    )
    failed = json.loads(
        resolve_workspace_path(ROOT, config["failed_shared_model_selection"]).read_text()
    )
    if (
        noise.get("status") != "foundir_task_native_adaptation_checkpoint"
        or noise.get("task") != "noise"
        or noise.get("step") != 250
        or not isinstance(noise.get("model"), Mapping)
        or noise.get("sealed_internal_test_accessed") is not False
        or noise.get("protected_foundir_test_accessed") is not False
        or failed.get("status") != "foundir_noise_family_restormer_development_selection"
        or failed.get("joint_gate_pass") is not False
        or failed.get("official_foundir_test_accessed") is not False
    ):
        raise RuntimeError("Noise-family cascade prerequisite evidence changed")
    return config


def save_checkpoint(
    path: Path,
    *,
    config_sha: str,
    step: int,
    cascade: NoiseExposureCascade,
    optimizer: torch.optim.Optimizer,
    frozen_noise_sha: str,
) -> None:
    payload = {
        "schema_version": 1,
        "status": "foundir_noise_family_cascade_checkpoint",
        "config_sha256": config_sha,
        "task": "noise_family_cascade",
        "step": step,
        "frozen_noise_checkpoint_sha256": frozen_noise_sha,
        "exposure": {key: value.detach().cpu() for key, value in cascade.exposure.state_dict().items()},
        "optimizer": optimizer.state_dict(),
        "development_pixels_accessed": False,
        "official_foundir_test_accessed": False,
    }
    if path.exists():
        existing = torch.load(path, map_location="cpu", weights_only=False)
        if existing.get("config_sha256") != config_sha or existing.get("step") != step:
            raise RuntimeError(f"existing cascade checkpoint changed: {path}")
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def resume_checkpoint(
    output: Path,
    *,
    config_sha: str,
    cascade: NoiseExposureCascade,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> int:
    if list(output.glob("step_*.pt.tmp")):
        raise RuntimeError("orphan cascade checkpoint temporary exists")
    paths = sorted(output.glob("step_*.pt"))
    if not paths:
        return 0
    path = paths[-1]
    payload = torch.load(path, map_location="cpu", weights_only=False)
    step = int(path.stem.removeprefix("step_"))
    if (
        payload.get("status") != "foundir_noise_family_cascade_checkpoint"
        or payload.get("config_sha256") != config_sha
        or payload.get("task") != "noise_family_cascade"
        or payload.get("step") != step
        or step not in {500, 1000}
        or payload.get("development_pixels_accessed") is not False
        or payload.get("official_foundir_test_accessed") is not False
    ):
        raise RuntimeError("invalid cascade resume checkpoint")
    cascade.exposure.load_state_dict(payload["exposure"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    move_optimizer_state(optimizer, device)
    return step


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    config_path = resolve_workspace_path(ROOT, args.config)
    config = validate_config(config_path)
    config_sha = file_sha256(config_path)
    rows = load_rows(config)
    if args.preflight:
        print(
            "FOUNDIR_NOISE_FAMILY_CASCADE_PREFLIGHT_PASS "
            + json.dumps(
                {
                    "config_sha256": config_sha,
                    "train_records": len(rows["train"]),
                    "validation_records": len(rows["validation"]),
                    "method": config["method"],
                    "development_pixels_accessed": False,
                    "official_foundir_test_accessed": False,
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
    if int(os.environ["WORLD_SIZE"]) != world or torch.cuda.device_count() != world:
        raise RuntimeError("Noise-family cascade requires exactly eight visible GPUs")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if not any(name in torch.cuda.get_device_properties(device).name for name in ("H100", "H200")):
        raise RuntimeError("Noise-family cascade requires H100/H200 GPUs")
    dist.init_process_group("nccl", timeout=timedelta(minutes=30))
    torch.manual_seed(int(optimization["seed"]))
    torch.cuda.manual_seed_all(int(optimization["seed"]))

    noise, noise_metadata = load_task_native_restormer(
        config["base_checkpoint_name"],
        resolve_workspace_path(ROOT, Path(config["base_checkpoint"]).parent),
    )
    noise_payload = torch.load(
        resolve_workspace_path(ROOT, config["frozen_noise_checkpoint"]),
        map_location="cpu",
        weights_only=False,
    )
    noise.load_state_dict(noise_payload["model"], strict=True)
    exposure, exposure_metadata = load_task_native_restormer(
        config["base_checkpoint_name"],
        resolve_workspace_path(ROOT, Path(config["base_checkpoint"]).parent),
    )
    initialize_identity_exposure(exposure)
    cascade = NoiseExposureCascade(noise, exposure).to(device)
    distributed = DistributedDataParallel(
        cascade, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False
    )
    trainable = [parameter for parameter in distributed.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
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
        cascade=distributed.module,
        optimizer=optimizer,
        device=device,
    )
    starts = torch.tensor([current_step, current_step], device=device, dtype=torch.int64)
    dist.all_reduce(starts[:1], op=dist.ReduceOp.MIN)
    dist.all_reduce(starts[1:], op=dist.ReduceOp.MAX)
    if int(starts[0].item()) != current_step or int(starts[1].item()) != current_step:
        raise RuntimeError("cascade ranks resumed different steps")

    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

    metric = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(device).eval()
    grouped: dict[str, dict[str, list[Mapping[str, str]]]] = {
        category: defaultdict(list) for category in CATEGORIES
    }
    for row in rows["train"]:
        grouped[row["category"]][row["source_group"]].append(row)
    verified: set[str] = set()
    micro = int(optimization["micro_batch_per_gpu"])
    accumulation = int(optimization["gradient_accumulation"])
    global_batch = int(optimization["global_batch"])
    milestones = {int(value) for value in optimization["milestones"]}
    total = int(optimization["total_steps"])
    started = time.perf_counter()
    print(
        "FOUNDIR_NOISE_FAMILY_CASCADE_START "
        + json.dumps(
            {
                "rank": rank,
                "world": world,
                "resume_step": current_step,
                "noise": noise_metadata,
                "exposure": exposure_metadata,
                "exposure_initialization": "exact_identity_zero_output_convolution",
            },
            sort_keys=True,
        ),
        flush=True,
    )
    for step in range(current_step + 1, total + 1):
        lr = learning_rate(config, step)
        for group in optimizer.param_groups:
            group["lr"] = lr
        assignments = category_balanced_batch(
            grouped, step=step, seed=int(optimization["seed"]), count=global_batch
        )
        local_assignments = assignments[rank::world]
        if len(local_assignments) != micro * accumulation:
            raise RuntimeError("cascade rank batch arithmetic changed")
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
                    )
                    for row, augmentation_seed in batch
                ]
            except Exception as error:
                local_error = f"{type(error).__name__}: {error}"
            raise_if_any_rank_failed(
                local_error, device=device, world=world, operation=f"cascade batch load step {step}"
            )
            assert pairs is not None
            source = torch.stack([pair[0] for pair in pairs]).to(device)
            target = torch.stack([pair[1] for pair in pairs]).to(device)
            sync = nullcontext() if accumulation_index + 1 == accumulation else distributed.no_sync()
            with sync:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    restored = distributed(source)
                    loss = F.l1_loss(restored, target) / accumulation
                if not bool(torch.isfinite(loss).item()):
                    raise RuntimeError(f"non-finite cascade loss at step {step}")
                loss.backward()
            loss_value += float(loss.detach())
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            trainable, float(optimization["gradient_clip_norm"])
        )
        if not bool(torch.isfinite(gradient_norm).item()):
            raise RuntimeError(f"non-finite cascade gradient at step {step}")
        optimizer.step()
        if step % 50 == 0 and rank == 0:
            print(
                f"FOUNDIR_NOISE_FAMILY_CASCADE_PROGRESS step={step}/{total} crop={crop_size} "
                f"loss={loss_value:.8f} lr={lr:.8g} elapsed_seconds={time.perf_counter()-started:.1f}",
                flush=True,
            )
        if step in milestones:
            checkpoint = output / f"step_{step:06d}.pt"
            if rank == 0:
                save_checkpoint(
                    checkpoint,
                    config_sha=config_sha,
                    step=step,
                    cascade=distributed.module,
                    optimizer=optimizer,
                    frozen_noise_sha=config["frozen_noise_checkpoint_sha256"],
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
    if rank == 0:
        finalize(config, config_sha, output)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
