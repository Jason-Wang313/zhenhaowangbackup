#!/usr/bin/env python3
"""Train a deployable 8x8 amplitude map for the frozen fidelity residual."""

from __future__ import annotations

import json
import math
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lowlight_restoration.flux_prompt_cache import file_sha256  # noqa: E402
from lowlight_restoration.proposal_residual_basis import ProposalResidualBasisRestormer  # noqa: E402
from lowlight_restoration.spatial_residual_amplitude import (  # noqa: E402
    SpatialResidualAmplitudePredictor,
    analytic_patch_amplitude,
    apply_patch_amplitude,
    parameter_count,
)
from smoke.train_foundir_proposal_relative_energy_noise_v1_ddp import (  # noqa: E402
    atomic_json,
    crop_for_step,
    fixed_validation_rows,
    load_audit_crop,
    load_frozen_proposal,
    load_task_rows,
    load_training_patch,
    sample_global_batch,
)
from smoke.train_foundir_proposal_relative_energy_noise_v2_ddp import (  # noqa: E402
    build_lpips,
    lpips_values,
    run_frozen,
)

CONFIG_PATH = ROOT / "configs/foundir_spatial_residual_amplitude_noise_v1.json"


def learning_rate(config: dict, step: int) -> float:
    schedule = config["schedule"]
    peak, minimum = float(schedule["learning_rate"]), float(schedule["minimum_learning_rate"])
    warmup, horizon = int(schedule["warmup_steps"]), int(schedule["primary_steps"])
    if step <= warmup:
        return peak * step / warmup
    progress = (step - warmup) / max(1, horizon - warmup)
    return minimum + .5 * (peak - minimum) * (1 + math.cos(math.pi * progress))


def load_residual_basis(config: dict, device: torch.device) -> ProposalResidualBasisRestormer:
    specification = config["frozen_residual_basis"]
    path = ROOT / specification["checkpoint"]
    if file_sha256(path) != specification["checkpoint_sha256"]:
        raise RuntimeError("frozen residual-basis checkpoint identity changed")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if int(payload["step"]) != int(specification["step"]):
        raise RuntimeError("frozen residual-basis step changed")
    basis_config = json.loads((ROOT / "configs/foundir_proposal_residual_basis_noise_v1.json").read_text())
    model = ProposalResidualBasisRestormer(
        residual_bound=float(basis_config["model"]["residual_bound"]),
        initial_output_scale=float(basis_config["model"]["initial_output_scale"]),
    )
    model.load_state_dict(payload["model"], strict=True)
    return model.to(device).eval().requires_grad_(False)


def frozen_outputs(
    proposal_model: torch.nn.Module,
    residual_model: ProposalResidualBasisRestormer,
    observation: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        proposal = run_frozen(proposal_model, observation).add(1).mul(.5)
        fidelity, _ = residual_model(observation, proposal)
    # Clone outside inference_mode so the trainable predictor may save these
    # tensors for its backward pass.
    return proposal.float().clone(), fidelity.float().clone()


def perceptual(network: torch.nn.Module, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return lpips_values(network, (output.mul(2).sub(1),), target.mul(2).sub(1))[:, 0].mean()


def objective(
    amplitude: torch.Tensor,
    output: torch.Tensor,
    fidelity: torch.Tensor,
    proposal: torch.Tensor,
    target: torch.Tensor,
    lpips_network: torch.nn.Module,
    config: dict,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    target_amplitude, confidence = analytic_patch_amplitude(proposal, fidelity, target)
    teacher = (F.smooth_l1_loss(amplitude, target_amplitude, reduction="none") * confidence).mean()
    output_mse = F.mse_loss(output, target)
    output_lpips = perceptual(lpips_network, output, target)
    patch = int(config["loss"]["patch_size"])
    output_error = F.avg_pool2d((output - target).square().mean(1, keepdim=True), patch, patch)
    fidelity_error = F.avg_pool2d((fidelity - target).square().mean(1, keepdim=True), patch, patch)
    no_harm = F.relu(output_error - fidelity_error).mean()
    weights = config["loss"]
    loss = (
        output_mse
        + float(weights["teacher_weight"]) * teacher
        + float(weights["lpips_weight"]) * output_lpips
        + float(weights["local_no_harm_weight"]) * no_harm
    )
    return loss, {
        "output_mse": output_mse,
        "output_lpips": output_lpips,
        "teacher": teacher,
        "no_harm": no_harm,
        "amplitude_mean": amplitude.mean(),
        "target_amplitude_mean": target_amplitude.mean(),
    }


def validate(
    predictor: SpatialResidualAmplitudePredictor,
    proposal_model: torch.nn.Module,
    residual_model: ProposalResidualBasisRestormer,
    rows: list[dict[str, str]],
    config: dict,
    lpips_network: torch.nn.Module,
    device: torch.device,
    rank: int,
    world: int,
) -> dict | None:
    predictor.eval()
    names = ("proposal", "fidelity", "spatial")
    sums = torch.zeros(len(names), 3, dtype=torch.float64, device=device)
    amplitude_sums = torch.zeros(4, dtype=torch.float64, device=device)
    for index in range(rank, len(rows), world):
        lq, gt, _, _ = load_audit_crop(
            rows[index], crop_size=int(config["data"]["validation_crop_size"]),
            seed=int(config["data"]["validation_crop_seed"]),
        )
        observation = lq[None].to(device).add(1).mul(.5)
        target = gt[None].to(device).add(1).mul(.5)
        proposal, fidelity = frozen_outputs(proposal_model, residual_model, observation)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            amplitude = predictor(observation, proposal, fidelity)
            spatial = apply_patch_amplitude(proposal, fidelity, amplitude).float()
        arms = (proposal, fidelity, spatial)
        with torch.no_grad():
            lpips = lpips_values(
                lpips_network, tuple(value.mul(2).sub(1) for value in arms), target.mul(2).sub(1)
            )[0]
            for arm, value in enumerate(arms):
                mse = (value - target).square().mean()
                sums[arm, 0] += (-10 * torch.log10(mse.clamp_min(1e-12))).double()
                sums[arm, 1] += lpips[arm].double()
                sums[arm, 2] += 1
            amplitude_sums[0] += amplitude.double().sum()
            amplitude_sums[1] += amplitude.double().square().sum()
            amplitude_sums[2] += amplitude.numel()
            amplitude_sums[3] += ((amplitude < .05) | (amplitude > 1.95)).double().sum()
    dist.all_reduce(sums)
    dist.all_reduce(amplitude_sums)
    predictor.train()
    if rank:
        return None
    proposal_psnr = float(sums[0, 0] / sums[0, 2])
    proposal_lpips = float(sums[0, 1] / sums[0, 2])
    result = {}
    for index, name in enumerate(names):
        psnr = float(sums[index, 0] / sums[index, 2])
        lpips = float(sums[index, 1] / sums[index, 2])
        result[name] = {
            "records": int(sums[index, 2]), "psnr": psnr, "lpips": lpips,
            "psnr_gain_db": psnr - proposal_psnr,
            "lpips_reduction": proposal_lpips - lpips,
        }
    mean = amplitude_sums[0] / amplitude_sums[2]
    result["amplitude"] = {
        "mean": float(mean),
        "standard_deviation": float((amplitude_sums[1] / amplitude_sums[2] - mean.square()).clamp_min(0).sqrt()),
        "boundary_fraction": float(amplitude_sums[3] / amplitude_sums[2]),
    }
    return result


def save_checkpoint(path: Path, model: SpatialResidualAmplitudePredictor, optimizer: torch.optim.Optimizer, **metadata) -> None:
    payload = {
        "schema_version": 1,
        "status": "foundir_spatial_residual_amplitude_noise_v1_checkpoint",
        "model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "optimizer": optimizer.state_dict(),
        **metadata,
        "internal_test_accessed": False,
        "official_foundir_test_accessed": False,
        "sealed_internal_accessed": False,
        "frozen_660_accessed": False,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def main() -> None:
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    config = json.loads(CONFIG_PATH.read_text())
    config_sha = file_sha256(CONFIG_PATH)
    if world != int(config["hardware"]["world_size"]):
        raise RuntimeError("spatial amplitude training requires exactly eight local GPUs")
    names: list[str | None] = [None] * world
    dist.all_gather_object(names, torch.cuda.get_device_name(local_rank))
    families = [family for family in config["hardware"]["supported_families"] if all(family in str(name).upper() for name in names)]
    if len(families) != 1:
        raise RuntimeError(f"requires homogeneous H100 or H200 GPUs, got {names}")
    family = families[0]
    profile = config["hardware"]["profiles"][family]
    micro, accumulation = int(profile["micro_batch_per_gpu"]), int(profile["gradient_accumulation"])
    if micro * accumulation * world != int(config["hardware"]["global_batch"]):
        raise RuntimeError("spatial amplitude global batch changed")
    if file_sha256(ROOT / config["data"]["manifest"]) != config["data"]["manifest_sha256"]:
        raise RuntimeError("spatial amplitude manifest identity changed")

    rows = load_task_rows(config)
    validation_rows = fixed_validation_rows(
        rows["validation"], int(config["data"]["validation_subset_records"]),
        int(config["data"]["validation_crop_seed"]),
    )
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows["train"]:
        grouped.setdefault(row["source_group"], []).append(row)
    proposal_model = load_frozen_proposal(config, device)
    residual_model = load_residual_basis(config, device)
    predictor = SpatialResidualAmplitudePredictor(
        channels=int(config["model"]["channels"]), blocks=int(config["model"]["blocks"])
    ).to(device)
    distributed = DDP(predictor, device_ids=[local_rank], gradient_as_bucket_view=False)
    lpips_network = build_lpips(device)
    optimizer = torch.optim.AdamW(
        distributed.parameters(), lr=float(config["schedule"]["learning_rate"]),
        betas=(.9, .999), weight_decay=float(config["schedule"]["weight_decay"]),
    )
    run_id = os.environ.get("SPATIAL_AMPLITUDE_RUN_ID", "qualification_v1")
    total = int(os.environ.get("SPATIAL_AMPLITUDE_TOTAL_STEPS", "25"))
    if total not in {int(value) for value in config["schedule"]["milestones"]}:
        raise RuntimeError(f"total steps must be a frozen milestone, got {total}")
    output = ROOT / config["runtime"]["output_root"] / run_id
    output.mkdir(parents=True, exist_ok=True)
    hardware = {
        "gpu_family": family, "gpu_names": names, "world_size": world,
        "micro_batch_per_gpu": micro, "gradient_accumulation": accumulation,
        "global_batch": micro * accumulation * world,
    }
    if rank == 0:
        atomic_json(output / "launch.json", {
            "status": "foundir_spatial_residual_amplitude_noise_v1_started",
            "run_id": run_id, "total_steps": total, "config_sha256": config_sha,
            "parameters": parameter_count(predictor), "hardware": hardware, **config["access"],
        })

    start_step = 0
    checkpoints = sorted(output.glob("step_*.pt"))
    if checkpoints:
        payload = torch.load(checkpoints[-1], map_location="cpu", weights_only=False)
        if payload.get("config_sha256") != config_sha:
            raise RuntimeError("spatial amplitude resume config changed")
        predictor.load_state_dict(payload["model"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(device)
        start_step = int(payload["step"])

    term_names = ("output_mse", "output_lpips", "teacher", "no_harm", "amplitude_mean", "target_amplitude_mean")
    verified: set[str] = set()
    started = time.monotonic()
    distributed.train()
    global_batch = int(config["hardware"]["global_batch"])
    for step in range(start_step + 1, total + 1):
        lr = learning_rate(config, step)
        for group in optimizer.param_groups:
            group["lr"] = lr
        selected = sample_global_batch(
            grouped, task="spatial_residual_amplitude_noise_v1", step=step,
            seed=int(config["schedule"]["seed"]), count=global_batch,
        )
        local = selected[rank * (global_batch // world):(rank + 1) * (global_batch // world)]
        optimizer.zero_grad(set_to_none=True)
        reduced = torch.zeros(1 + len(term_names), device=device)
        for accumulation_index in range(accumulation):
            chunk = local[accumulation_index * micro:(accumulation_index + 1) * micro]
            pairs = [
                load_training_patch(row, crop_size=crop_for_step(config, step), augmentation_seed=seed, verified=verified)
                for row, seed in chunk
            ]
            observation = torch.stack([pair[0] for pair in pairs]).to(device)
            target = torch.stack([pair[1] for pair in pairs]).to(device)
            proposal, fidelity = frozen_outputs(proposal_model, residual_model, observation)
            sync = nullcontext() if accumulation_index + 1 == accumulation else distributed.no_sync()
            with sync:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    amplitude = distributed(observation, proposal, fidelity)
                    restored = apply_patch_amplitude(proposal, fidelity, amplitude).float()
                loss, terms = objective(amplitude.float(), restored, fidelity, proposal, target, lpips_network, config)
                (loss / accumulation).backward()
            reduced[0] += loss.detach() / accumulation
            for index, name in enumerate(term_names, 1):
                reduced[index] += terms[name].detach() / accumulation
        norm = torch.nn.utils.clip_grad_norm_(distributed.parameters(), float(config["schedule"]["gradient_clip_norm"]))
        if not bool(torch.isfinite(norm)) or not bool(torch.isfinite(reduced).all()):
            raise RuntimeError("non-finite spatial amplitude training state")
        optimizer.step()
        if step % 5 == 0:
            dist.all_reduce(reduced)
            if rank == 0:
                values = (reduced / world).tolist()
                record = {"step": step, "lr": lr, "loss": values[0]}
                record.update({name: values[index] for index, name in enumerate(term_names, 1)})
                record["elapsed_seconds"] = time.monotonic() - started
                print(json.dumps(record), flush=True)
        if step == total or step % int(config["schedule"]["steps_per_effective_pass"]) == 0:
            dist.barrier()
            validation = validate(
                predictor, proposal_model, residual_model, validation_rows, config,
                lpips_network, device, rank, world,
            )
            dist.barrier()
            if rank == 0:
                elapsed = time.monotonic() - started
                save_checkpoint(
                    output / f"step_{step:06d}.pt", predictor, optimizer, step=step,
                    config_sha256=config_sha, validation=validation, hardware=hardware,
                    elapsed_seconds=elapsed,
                )
                atomic_json(output / f"validation_step_{step:06d}.json", {
                    "status": "foundir_spatial_residual_amplitude_noise_v1_development_validation",
                    "step": step, "validation": validation,
                    "seconds_per_optimizer_step": elapsed / max(1, step - start_step),
                    "hardware": hardware, **config["access"],
                })
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
