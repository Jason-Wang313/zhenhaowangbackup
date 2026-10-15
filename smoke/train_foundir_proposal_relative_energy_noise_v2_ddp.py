#!/usr/bin/env python3
"""Train the max-complete PPT proposal-relative energy V2 on Noise data."""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch import Tensor
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lowlight_restoration.flux_prompt_cache import file_sha256  # noqa: E402
from lowlight_restoration.proposal_relative_energy_v2 import (  # noqa: E402
    EnergyAndDirectControlsV2,
    parameter_counts,
)
from lowlight_restoration.task_native_restormer import (  # noqa: E402
    TASK_NATIVE_RESTORMER_SPECS,
    load_task_native_restormer,
)
from smoke.train_foundir_proposal_relative_energy_noise_v1_ddp import (  # noqa: E402
    atomic_json,
    crop_for_step,
    edge_loss,
    fixed_validation_rows,
    load_audit_crop,
    load_frozen_proposal,
    load_task_rows,
    load_training_patch,
    sample_global_batch,
)


CONFIG_PATH = ROOT / "configs/foundir_proposal_relative_energy_noise_v2.json"
DEGRADATIONS = ("blur", "noise", "rain", "haze", "low_light")


def descriptor_for_row(row: dict[str, str], config: dict, device: torch.device) -> Tensor:
    field = str(config["data"]["vlm_levels_field"])
    try:
        levels = json.loads(row[field])
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid frozen VLM levels for {row['record_id']}") from exc
    minimum = int(config["data"]["vlm_level_minimum"])
    maximum = int(config["data"]["vlm_level_maximum"])
    if set(levels) != set(DEGRADATIONS) or maximum <= minimum:
        raise RuntimeError("frozen VLM severity schema changed")
    values = []
    for name in DEGRADATIONS:
        value = int(levels[name])
        if not minimum <= value <= maximum:
            raise RuntimeError(f"VLM severity out of range for {row['record_id']}")
        values.append((value - minimum) / (maximum - minimum))
    return torch.tensor(values, device=device, dtype=torch.float32)


def learning_rate(config: dict, step: int) -> float:
    peak = float(config["schedule"]["learning_rate"])
    minimum = float(config["schedule"]["minimum_learning_rate"])
    warmup = int(config["schedule"]["warmup_steps"])
    horizon = int(config["schedule"]["learning_rate_horizon_steps"])
    if horizon != int(config["schedule"]["primary_steps"]):
        raise RuntimeError("V2 learning-rate horizon must equal the frozen five-pass endpoint")
    if step <= warmup:
        return peak * step / warmup
    progress = (step - warmup) / max(1, horizon - warmup)
    return minimum + 0.5 * (peak - minimum) * (1 + math.cos(math.pi * progress))


def load_frozen_candidate_models(config: dict, device: torch.device) -> dict[str, torch.nn.Module]:
    models: dict[str, torch.nn.Module] = {"proposal": load_frozen_proposal(config, device)}
    directory = ROOT / config["frozen_contrastive_models"]["checkpoint_dir"]
    for role in ("under_restored", "wrong_expert", "detail_damaged"):
        name = str(config["frozen_contrastive_models"][role])
        spec = TASK_NATIVE_RESTORMER_SPECS[name]
        if file_sha256(directory / spec.filename) != spec.sha256:
            raise RuntimeError(f"frozen {role} checkpoint identity changed")
        model, _ = load_task_native_restormer(name, directory)
        models[role] = model.requires_grad_(False).eval().to(device)
    return models


def run_frozen(model: torch.nn.Module, value_01: Tensor) -> Tensor:
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        output = model(value_01).clamp(0, 1)
    return output.float().mul(2).sub(1).detach()


def frozen_candidates(models: dict[str, torch.nn.Module], observation_01: Tensor) -> tuple[Tensor, Tensor]:
    proposal = run_frozen(models["proposal"], observation_01)
    under = run_frozen(models["under_restored"], observation_01)
    wrong = run_frozen(models["wrong_expert"], observation_01)
    detail = run_frozen(models["detail_damaged"], observation_01)
    over = run_frozen(models["proposal"], proposal.add(1).mul(0.5))
    return proposal, torch.stack((wrong, under, over, detail), dim=1)


def build_lpips(device: torch.device) -> torch.nn.Module:
    metric = LearnedPerceptualImagePatchSimilarity(net_type="alex").to(device).eval()
    metric.requires_grad_(False)
    return metric.net


def lpips_values(network: torch.nn.Module, candidates: tuple[Tensor, ...], target: Tensor) -> Tensor:
    values = []
    for candidate in candidates:
        value = network(candidate, target, normalize=False)
        values.append(value.reshape(value.shape[0], -1).mean(1))
    return torch.stack(values, dim=1)


def reconstruction(candidate: Tensor, target: Tensor, edge_weight: float) -> Tensor:
    return F.l1_loss(candidate, target) + edge_weight * edge_loss(candidate, target)


def pairwise_ranking(energies: Tensor, qualities: Tensor, margin: float) -> Tensor:
    better = qualities[:, :, None] + 1e-8 < qualities[:, None, :]
    gap = (qualities[:, None, :] - qualities[:, :, None]).clamp_min(0)
    scale = qualities.mean(1, keepdim=True)[:, :, None].clamp_min(1e-6)
    violations = F.relu(margin * gap / scale + energies[:, :, None] - energies[:, None, :])
    return violations[better].mean() if bool(better.any()) else energies.new_zeros(())


def spatial_ranking(
    energy_maps: Tensor,
    candidates: tuple[Tensor, ...],
    target: Tensor,
    *,
    kernel: int,
    margin: float,
) -> Tensor:
    candidate_tensor = torch.stack(candidates, dim=1)
    local_quality = (candidate_tensor - target[:, None]).square().mean(2)
    batch, count, height, width = local_quality.shape
    local_quality = F.avg_pool2d(local_quality.flatten(0, 1)[:, None], kernel, kernel)
    local_quality = local_quality.reshape(batch, count, *local_quality.shape[-2:])
    local_energy = F.avg_pool2d(energy_maps.flatten(0, 1), kernel, kernel)
    local_energy = local_energy.reshape(batch, count, *local_energy.shape[-2:])
    better = local_quality[:, :, None] + 1e-8 < local_quality[:, None, :]
    gap = (local_quality[:, None, :] - local_quality[:, :, None]).clamp_min(0)
    scale = local_quality.mean(1, keepdim=True)[:, :, None].clamp_min(1e-6)
    violations = F.relu(
        margin * gap / scale + local_energy[:, :, None] - local_energy[:, None, :]
    )
    return violations[better].mean() if bool(better.any()) else local_energy.new_zeros(())


def training_objective(
    output: dict[str, Tensor | tuple[Tensor, ...]],
    proposal: Tensor,
    target: Tensor,
    lpips_network: torch.nn.Module,
    config: dict,
) -> tuple[Tensor, dict[str, Tensor]]:
    weights = config["loss"]
    trajectory = tuple(output["trajectory_candidates"])
    gradients = tuple(output["trajectory_gradients"])
    states = (proposal,) + trajectory
    state_mse = torch.stack(
        [(candidate - target).square().flatten(1).mean(1) for candidate in states], dim=1
    )
    state_lpips = lpips_values(lpips_network, states, target)
    trajectory_reconstruction = torch.stack(
        [reconstruction(candidate, target, float(weights["edge"])) for candidate in trajectory]
    ).mean()
    trajectory_lpips = state_lpips[:, 1:].mean()
    no_harm_mse = F.relu(state_mse[:, 1:] - state_mse[:, :-1]).mean()
    no_harm_lpips = F.relu(state_lpips[:, 1:] - state_lpips[:, :-1]).mean()

    directions = []
    for previous, gradient in zip(states[:-1], gradients, strict=True):
        desired = (target - previous).flatten(1)
        update = (-gradient).flatten(1)
        directions.append((1.0 - F.cosine_similarity(update, desired, dim=1, eps=1e-8)).mean())
    direction = torch.stack(directions).mean()

    contrastive = tuple(output["contrastive_candidates"])
    energies = output["contrastive_energies"]
    energy_maps = output["contrastive_energy_maps"]
    with torch.no_grad():
        perceptual_quality = lpips_values(lpips_network, tuple(v.detach() for v in contrastive), target)
        pixel_quality = torch.stack(
            [(candidate.detach() - target).square().flatten(1).mean(1) for candidate in contrastive],
            dim=1,
        )
        quality = pixel_quality + float(weights["ranking_lpips_quality"]) * perceptual_quality
    scalar_rank = pairwise_ranking(energies, quality, float(weights["ranking_margin"]))
    local_rank = spatial_ranking(
        energy_maps,
        contrastive,
        target,
        kernel=int(weights["spatial_quality_kernel"]),
        margin=float(weights["spatial_margin"]),
    )

    direct = output["direct"]
    direct_lpips = lpips_values(lpips_network, (proposal, direct), target)
    direct_reconstruction = reconstruction(direct, target, float(weights["edge"]))
    direct_no_harm_mse = F.relu(
        (direct - target).square().flatten(1).mean(1)
        - (proposal - target).square().flatten(1).mean(1)
    ).mean()
    direct_no_harm_lpips = F.relu(direct_lpips[:, 1] - direct_lpips[:, 0]).mean()

    terms = {
        "trajectory_reconstruction": trajectory_reconstruction,
        "trajectory_lpips": trajectory_lpips,
        "direction": direction,
        "scalar_ranking": scalar_rank,
        "spatial_ranking": local_rank,
        "no_harm_mse": no_harm_mse,
        "no_harm_lpips": no_harm_lpips,
        "direct_reconstruction": direct_reconstruction,
        "direct_lpips": direct_lpips[:, 1].mean(),
        "direct_no_harm_mse": direct_no_harm_mse,
        "direct_no_harm_lpips": direct_no_harm_lpips,
    }
    loss = (
        float(weights["trajectory_reconstruction"]) * trajectory_reconstruction
        + float(weights["trajectory_lpips"]) * trajectory_lpips
        + float(weights["gradient_direction"]) * direction
        + float(weights["scalar_ranking"]) * scalar_rank
        + float(weights["spatial_ranking"]) * local_rank
        + float(weights["no_harm_mse"]) * no_harm_mse
        + float(weights["no_harm_lpips"]) * no_harm_lpips
        + float(weights["direct_reconstruction"]) * direct_reconstruction
        + float(weights["direct_lpips"]) * direct_lpips[:, 1].mean()
        + float(weights["no_harm_mse"]) * direct_no_harm_mse
        + float(weights["no_harm_lpips"]) * direct_no_harm_lpips
    )
    return loss, terms


def validate(
    controls: EnergyAndDirectControlsV2,
    frozen_models: dict[str, torch.nn.Module],
    rows: list[dict[str, str]],
    config: dict,
    lpips_network: torch.nn.Module,
    device: torch.device,
    rank: int,
    world: int,
) -> dict | None:
    controls.eval()
    names = ("proposal", "direct", "energy_1", "energy_2", "energy_3", "adaptive")
    sums = torch.zeros(len(names), 3, device=device, dtype=torch.float64)
    for index in range(rank, len(rows), world):
        row = rows[index]
        lq, gt, _, _ = load_audit_crop(
            row,
            crop_size=int(config["data"]["validation_crop_size"]),
            seed=int(config["data"]["validation_crop_seed"]),
        )
        observation_01 = lq[None].to(device).add(1).mul(0.5)
        target = gt[None].to(device)
        observation = observation_01.mul(2).sub(1)
        descriptor = descriptor_for_row(row, config, device)[None]
        # Validation needs only the frozen proposal.  The four contrastive
        # experts are training supervision and must not add avoidable
        # validation time or memory pressure.
        proposal = run_frozen(frozen_models["proposal"], observation_01)
        with torch.no_grad():
            direct = controls.direct_model(observation, proposal, descriptor)
        with torch.enable_grad():
            trajectory = controls.energy_model.refine(
                observation,
                proposal.detach(),
                descriptor,
                steps=3,
                create_graph=False,
                detach_between_steps=True,
            )
        candidates = tuple(value.detach() for value in trajectory.candidates)
        with torch.no_grad():
            adaptive, _, _ = controls.energy_model.select_lowest_energy(
                observation, proposal, candidates, descriptor
            )
        arms = (proposal, direct) + candidates + (adaptive,)
        with torch.no_grad():
            perceptual = lpips_values(lpips_network, arms, target)[0]
            for arm, candidate in enumerate(arms):
                mse = (candidate.add(1).mul(0.5) - target.add(1).mul(0.5)).square().mean()
                sums[arm, 0] += (-10 * torch.log10(mse.clamp_min(1e-12))).double()
                sums[arm, 1] += perceptual[arm].double()
                sums[arm, 2] += 1
    dist.all_reduce(sums)
    controls.train()
    if rank:
        return None
    return {
        name: {
            "records": int(sums[index, 2].item()),
            "psnr": float((sums[index, 0] / sums[index, 2]).item()),
            "lpips": float((sums[index, 1] / sums[index, 2]).item()),
        }
        for index, name in enumerate(names)
    }


def save_checkpoint(
    path: Path,
    model: EnergyAndDirectControlsV2,
    optimizer: torch.optim.Optimizer,
    *,
    step: int,
    config_sha: str,
    validation: dict | None,
    hardware: dict,
    elapsed_seconds: float,
) -> None:
    payload = {
        "schema_version": 2,
        "status": "proposal_relative_energy_noise_v2_checkpoint",
        "config_sha256": config_sha,
        "step": step,
        "model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "optimizer": optimizer.state_dict(),
        "validation": validation,
        "hardware": hardware,
        "elapsed_seconds": elapsed_seconds,
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
    if world != 8:
        raise RuntimeError("V2 requires exactly eight local GPUs")
    names = [None] * world
    dist.all_gather_object(names, torch.cuda.get_device_name(local_rank))
    families = {
        family
        for name in names
        for family in config["hardware"]["supported_families"]
        if family in str(name).upper()
    }
    if len(families) != 1 or any(not any(f in str(name).upper() for f in families) for name in names):
        raise RuntimeError(f"V2 requires homogeneous 8xH100 or 8xH200, got {names}")
    family = next(iter(families))
    profile = config["hardware"]["profiles"][family]
    micro = int(profile["micro_batch_per_gpu"])
    accumulation = int(profile["gradient_accumulation"])
    global_batch = micro * accumulation * world
    if global_batch != int(config["hardware"]["global_batch"]):
        raise RuntimeError("V2 hardware profile changed the frozen global batch")

    rows = load_task_rows(config)
    if file_sha256(ROOT / config["data"]["manifest"]) != config["data"]["manifest_sha256"]:
        raise RuntimeError("V2 Noise manifest hash changed")
    validation_rows = fixed_validation_rows(
        rows["validation"],
        int(config["data"]["validation_subset_records"]),
        int(config["data"]["validation_crop_seed"]),
    )
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows["train"]:
        grouped.setdefault(row["source_group"], []).append(row)

    frozen_models = load_frozen_candidate_models(config, device)
    lpips_network = build_lpips(device)
    controls = EnergyAndDirectControlsV2().to(device)
    counts = parameter_counts(controls)
    ddp = DDP(controls, device_ids=[local_rank], gradient_as_bucket_view=False)
    optimizer = torch.optim.AdamW(
        ddp.parameters(),
        lr=float(config["schedule"]["learning_rate"]),
        betas=(0.9, 0.999),
        weight_decay=float(config["schedule"]["weight_decay"]),
    )
    run_id = os.environ.get("PROPOSAL_ENERGY_V2_RUN_ID", "qualification_v1")
    total = int(os.environ.get("PROPOSAL_ENERGY_V2_TOTAL_STEPS", "25"))
    allowed = {int(value) for value in config["schedule"]["milestones"]}
    if total not in allowed:
        raise RuntimeError(f"V2 total steps must be a frozen milestone, got {total}")
    output_dir = ROOT / config["runtime"]["output_root"] / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    hardware = {
        "gpu_family": family,
        "gpu_names": names,
        "world_size": world,
        "micro_batch_per_gpu": micro,
        "gradient_accumulation": accumulation,
        "global_batch": global_batch,
    }
    if rank == 0:
        atomic_json(output_dir / "launch.json", {
            "status": "proposal_relative_energy_noise_v2_started",
            "run_id": run_id,
            "total_steps": total,
            "config_sha256": config_sha,
            "parameter_counts": counts,
            "six_changes_only": True,
            "hardware": hardware,
            "internal_test_accessed": False,
            "official_foundir_test_accessed": False,
            "sealed_internal_accessed": False,
            "frozen_660_accessed": False,
        })

    start_step = 0
    checkpoints = sorted(output_dir.glob("step_*.pt"))
    if checkpoints:
        payload = torch.load(checkpoints[-1], map_location="cpu", weights_only=False)
        if payload.get("config_sha256") != config_sha:
            raise RuntimeError("V2 resume config identity changed")
        controls.load_state_dict(payload["model"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(device)
        start_step = int(payload["step"])

    verified: set[str] = set()
    started = time.monotonic()
    ddp.train()
    term_names = (
        "trajectory_reconstruction", "trajectory_lpips", "direction", "scalar_ranking",
        "spatial_ranking", "no_harm_mse", "no_harm_lpips", "direct_reconstruction",
        "direct_lpips", "direct_no_harm_mse", "direct_no_harm_lpips",
    )
    for step in range(start_step + 1, total + 1):
        lr = learning_rate(config, step)
        for group in optimizer.param_groups:
            group["lr"] = lr
        selected = sample_global_batch(
            grouped,
            task="proposal_relative_energy_noise_v2",
            step=step,
            seed=int(config["schedule"]["seed"]),
            count=global_batch,
        )
        local = selected[rank * (global_batch // world):(rank + 1) * (global_batch // world)]
        optimizer.zero_grad(set_to_none=True)
        reduced = torch.zeros(1 + len(term_names), device=device)
        for accumulation_index in range(accumulation):
            chunk = local[accumulation_index * micro:(accumulation_index + 1) * micro]
            pairs = [
                load_training_patch(
                    row,
                    crop_size=crop_for_step(config, step),
                    augmentation_seed=seed,
                    verified=verified,
                )
                for row, seed in chunk
            ]
            observation_01 = torch.stack([pair[0] for pair in pairs]).to(device)
            target = torch.stack([pair[1] for pair in pairs]).to(device).mul(2).sub(1)
            observation = observation_01.mul(2).sub(1)
            descriptor = torch.stack(
                [descriptor_for_row(row, config, device) for row, _ in chunk]
            )
            proposal, hard_negatives = frozen_candidates(frozen_models, observation_01)
            output = ddp(observation, proposal, target, descriptor, hard_negatives)
            loss, terms = training_objective(
                output, proposal, target, lpips_network, config
            )
            (loss / accumulation).backward()
            reduced[0] += loss.detach() / accumulation
            for index, name in enumerate(term_names, start=1):
                reduced[index] += terms[name].detach() / accumulation
        torch.nn.utils.clip_grad_norm_(ddp.parameters(), float(config["schedule"]["gradient_clip_norm"]))
        optimizer.step()
        if not bool(torch.isfinite(reduced).all()):
            raise RuntimeError("non-finite V2 objective")
        if step % 5 == 0:
            dist.all_reduce(reduced)
            if rank == 0:
                values = (reduced / world).tolist()
                payload = {"step": step, "lr": lr, "loss": values[0]}
                payload.update({name: values[index] for index, name in enumerate(term_names, start=1)})
                payload["elapsed_seconds"] = time.monotonic() - started
                print(json.dumps(payload), flush=True)

        if step in allowed and (step == total or step % int(config["schedule"]["steps_per_effective_pass"]) == 0):
            dist.barrier()
            validation = validate(
                controls,
                frozen_models,
                validation_rows,
                config,
                lpips_network,
                device,
                rank,
                world,
            )
            dist.barrier()
            if rank == 0:
                elapsed = time.monotonic() - started
                save_checkpoint(
                    output_dir / f"step_{step:06d}.pt",
                    controls,
                    optimizer,
                    step=step,
                    config_sha=config_sha,
                    validation=validation,
                    hardware=hardware,
                    elapsed_seconds=elapsed,
                )
                seconds_per_step = elapsed / max(1, step - start_step)
                atomic_json(output_dir / f"validation_step_{step:06d}.json", {
                    "status": "proposal_relative_energy_noise_v2_development_validation",
                    "step": step,
                    "validation": validation,
                    "seconds_per_optimizer_step": seconds_per_step,
                    "hardware": hardware,
                    "internal_test_accessed": False,
                    "official_foundir_test_accessed": False,
                    "sealed_internal_accessed": False,
                    "frozen_660_accessed": False,
                })

    if rank == 0:
        wall_hours = (time.monotonic() - started) / 3600
        atomic_json(output_dir / "terminal_result.json", {
            "status": "proposal_relative_energy_noise_v2_training_complete",
            "completed_optimizer_steps": total,
            "wall_clock_hours": wall_hours,
            "allocated_gpu_hours": wall_hours * world,
            "hardware": hardware,
            "internal_test_accessed": False,
            "official_foundir_test_accessed": False,
            "sealed_internal_accessed": False,
            "frozen_660_accessed": False,
        })
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
