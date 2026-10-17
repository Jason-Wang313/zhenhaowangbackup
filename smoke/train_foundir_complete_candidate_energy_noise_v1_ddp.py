#!/usr/bin/env python3
"""Train the complete candidate-conditioned Noise restoration energy."""

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
from torch import Tensor
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lowlight_restoration.complete_candidate_energy import (  # noqa: E402
    CompleteCandidateEnergy,
    compose_selected_candidates,
    parameter_count,
    relative_risk_targets,
    select_candidates_fail_closed,
    tie_aware_gap_ranking,
)
from lowlight_restoration.flux_prompt_cache import file_sha256  # noqa: E402
from lowlight_restoration.proposal_residual_basis import ProposalResidualBasisRestormer  # noqa: E402
from lowlight_restoration.task_native_restormer import (  # noqa: E402
    TASK_NATIVE_RESTORMER_SPECS,
    load_task_native_restormer,
)
from smoke.evaluate_foundir_complete_energy_candidate_oracle_noise_v1_ddp import (  # noqa: E402
    deployable_candidates,
)
from smoke.train_foundir_proposal_relative_energy_noise_v1_ddp import (  # noqa: E402
    atomic_json,
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

CONFIG_PATH = ROOT / "configs/foundir_complete_candidate_energy_noise_v1.json"
SEVERITY_NAMES = ("blur", "noise", "rain", "haze", "low_light")
NEGATIVE_ROLES = ("under_restored", "wrong_expert", "detail_damaged", "repeat_proposal")


def learning_rate(config: dict, step: int) -> float:
    schedule = config["schedule"]
    peak, minimum = float(schedule["learning_rate"]), float(schedule["minimum_learning_rate"])
    warmup, horizon = int(schedule["warmup_steps"]), int(schedule["primary_steps"])
    if step <= warmup:
        return peak * step / warmup
    progress = (step - warmup) / max(1, horizon - warmup)
    return minimum + .5 * (peak - minimum) * (1 + math.cos(math.pi * progress))


def crop_for_step(config: dict, step: int) -> int:
    selected = 0
    for start, size in config["schedule"]["progressive_crops"]:
        if step > int(start):
            selected = int(size)
    if not selected or selected % int(config["model"]["patch_size"]):
        raise RuntimeError("complete energy crop/patch contract changed")
    return selected


def severity_from_rows(rows: list[dict[str, str]], device: torch.device) -> Tensor:
    values = []
    for row in rows:
        parsed = json.loads(row["qwen_levels_json"])
        if set(parsed) != set(SEVERITY_NAMES):
            raise RuntimeError("structured VLM severity schema changed")
        levels = [int(parsed[name]) for name in SEVERITY_NAMES]
        if any(level < 1 or level > 5 for level in levels):
            raise RuntimeError("structured VLM severity outside L1-L5")
        values.append([(level - 1) / 4 for level in levels])
    return torch.tensor(values, device=device, dtype=torch.float32)


def load_residual_basis(config: dict, base: dict, device: torch.device) -> ProposalResidualBasisRestormer:
    specification = config["frozen_residual_basis"]
    path = ROOT / specification["checkpoint"]
    if file_sha256(path) != specification["checkpoint_sha256"]:
        raise RuntimeError("complete energy residual checkpoint identity changed")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if int(payload["step"]) != int(specification["step"]):
        raise RuntimeError("complete energy residual checkpoint step changed")
    model = ProposalResidualBasisRestormer(
        residual_bound=float(base["model"]["residual_bound"]),
        initial_output_scale=float(base["model"]["initial_output_scale"]),
    )
    model.load_state_dict(payload["model"], strict=True)
    return model.to(device).eval().requires_grad_(False)


def load_negative_models(config: dict, device: torch.device) -> dict[str, torch.nn.Module]:
    specification = config["negative_models"]
    directory = ROOT / specification["checkpoint_dir"]
    result = {}
    for role in NEGATIVE_ROLES[:3]:
        name = specification[role]
        model_spec = TASK_NATIVE_RESTORMER_SPECS[name]
        if file_sha256(directory / model_spec.filename) != model_spec.sha256:
            raise RuntimeError(f"complete energy {role} checkpoint identity changed")
        model, _ = load_task_native_restormer(name, directory)
        result[role] = model.to(device).eval().requires_grad_(False)
    return result


def frozen_deployable(proposal_model, basis, observation: Tensor) -> tuple[Tensor, Tensor]:
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        proposal = run_frozen(proposal_model, observation).add(1).mul(.5)
        fidelity, perceptual = basis(observation, proposal)
    mapping = deployable_candidates(proposal.float(), fidelity.float(), perceptual.float())
    return proposal.float(), torch.stack(tuple(mapping.values()), dim=1)


def frozen_negative(
    role: str,
    models: dict[str, torch.nn.Module],
    proposal_model: torch.nn.Module,
    observation: Tensor,
    proposal: Tensor,
) -> Tensor:
    model = proposal_model if role == "repeat_proposal" else models[role]
    model_input = proposal if role == "repeat_proposal" else observation
    return run_frozen(model, model_input).add(1).mul(.5)


def spatial_lpips_maps(network: torch.nn.Module, candidates: Tensor, target: Tensor) -> Tensor:
    batch, count, channels, height, width = candidates.shape
    flat = candidates.flatten(0, 1).mul(2).sub(1)
    repeated_target = target[:, None].expand(-1, count, -1, -1, -1).flatten(0, 1).mul(2).sub(1)
    with torch.no_grad():
        values = network(flat, repeated_target, normalize=False)
    if values.shape != (batch * count, 1, height, width):
        raise RuntimeError(f"spatial LPIPS output changed: {tuple(values.shape)}")
    return values[:, 0].reshape(batch, count, height, width).float()


def objective(
    risks: Tensor,
    candidates: Tensor,
    target: Tensor,
    spatial_lpips_network: torch.nn.Module,
    config: dict,
) -> tuple[Tensor, dict[str, Tensor]]:
    weights = config["loss"]
    patch = int(config["model"]["patch_size"])
    with torch.no_grad():
        perceptual_maps = spatial_lpips_maps(spatial_lpips_network, candidates, target)
        target_mse, target_perceptual = relative_risk_targets(
            candidates, target, perceptual_maps,
            safe_index=int(config["safe_index"]), patch_size=patch,
            epsilon=float(weights["target_epsilon"]), log_clip=float(weights["target_log_clip"]),
        )
    predicted_mse, predicted_perceptual, log_variance = risks.float().unbind(2)
    log_variance = log_variance.clamp(-8, 8)
    squared_error = (predicted_mse - target_mse).square() + (predicted_perceptual - target_perceptual).square()
    risk_regression = (.5 * torch.exp(-log_variance) * squared_error + .5 * log_variance).mean()
    mse_ranking = tie_aware_gap_ranking(
        predicted_mse, target_mse, tie_epsilon=float(weights["tie_epsilon"])
    )
    perceptual_ranking = tie_aware_gap_ranking(
        predicted_perceptual, target_perceptual, tie_epsilon=float(weights["tie_epsilon"])
    )
    negative_margin = F.relu(
        float(weights["negative_relative_mse_margin"]) - predicted_mse[:, -1]
    ).mean()
    loss = (
        float(weights["risk_regression_weight"]) * risk_regression
        + float(weights["ranking_weight"]) * (mse_ranking + perceptual_ranking)
        + float(weights["negative_margin_weight"]) * negative_margin
    )
    return loss, {
        "risk_regression": risk_regression,
        "mse_ranking": mse_ranking,
        "perceptual_ranking": perceptual_ranking,
        "negative_margin": negative_margin,
        "mean_log_variance": log_variance.mean(),
        "oracle_non_safe_fraction": (target_mse[:, :-1].argmin(1) != int(config["safe_index"])).float().mean(),
    }


def validate(
    model,
    proposal_model,
    residual_model,
    rows,
    config,
    global_lpips,
    device,
    rank,
    world,
):
    model.eval()
    names = ("proposal", "fidelity", "selected")
    sums = torch.zeros(len(names), 3, dtype=torch.float64, device=device)
    counts = torch.zeros(len(config["candidate_names"]) + 2, dtype=torch.float64, device=device)
    for index in range(rank, len(rows), world):
        row = rows[index]
        lq, gt, _, _ = load_audit_crop(
            row, crop_size=int(config["data"]["validation_crop_size"]),
            seed=int(config["data"]["validation_crop_seed"]),
        )
        observation = lq[None].to(device).add(1).mul(.5)
        target = gt[None].to(device).add(1).mul(.5)
        proposal, candidates = frozen_deployable(proposal_model, residual_model, observation)
        severity = severity_from_rows([row], device)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            risks = model(observation, proposal, candidates, severity).float()
        selected, accepted = select_candidates_fail_closed(
            risks, safe_index=int(config["safe_index"]),
            uncertainty_scale=float(config["inference"]["uncertainty_scale"]),
            minimum_mse_advantage=float(config["inference"]["minimum_mse_advantage"]),
            maximum_relative_perceptual_risk=float(config["inference"]["maximum_relative_perceptual_risk"]),
        )
        output = compose_selected_candidates(
            candidates, selected, patch_size=int(config["model"]["patch_size"])
        )
        arms = (proposal, candidates[:, int(config["safe_index"])], output)
        with torch.no_grad():
            lpips = lpips_values(
                global_lpips, tuple(value.mul(2).sub(1) for value in arms), target.mul(2).sub(1)
            )[0]
            for arm, value in enumerate(arms):
                mse = (value - target).square().mean()
                sums[arm, 0] += (-10 * torch.log10(mse.clamp_min(1e-12))).double()
                sums[arm, 1] += lpips[arm].double()
                sums[arm, 2] += 1
            for candidate_index in range(candidates.shape[1]):
                counts[candidate_index] += (selected == candidate_index).sum().double()
            counts[-2] += accepted.sum().double()
            counts[-1] += selected.numel()
    dist.all_reduce(sums)
    dist.all_reduce(counts)
    model.train()
    if rank:
        return None
    proposal_psnr, proposal_lpips = float(sums[0, 0] / sums[0, 2]), float(sums[0, 1] / sums[0, 2])
    result = {}
    for arm, name in enumerate(names):
        psnr, lpips = float(sums[arm, 0] / sums[arm, 2]), float(sums[arm, 1] / sums[arm, 2])
        result[name] = {
            "records": int(sums[arm, 2]), "psnr": psnr, "lpips": lpips,
            "psnr_gain_db": psnr - proposal_psnr, "lpips_reduction": proposal_lpips - lpips,
        }
    result["selection"] = {
        "accepted_fraction": float(counts[-2] / counts[-1]),
        "fractions": {
            name: float(counts[i] / counts[-1]) for i, name in enumerate(config["candidate_names"])
        },
    }
    result["joint_gate_pass"] = (
        result["selected"]["psnr"] >= float(config["gate"]["minimum_psnr"])
        and result["selected"]["lpips"] <= float(config["gate"]["maximum_lpips"])
    )
    return result


def save_checkpoint(path: Path, model, optimizer, **metadata) -> None:
    payload = {
        "schema_version": 1, "status": "foundir_complete_candidate_energy_noise_v1_checkpoint",
        "model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "optimizer": optimizer.state_dict(), **metadata,
        "internal_test_accessed": False, "official_foundir_test_accessed": False,
        "sealed_internal_accessed": False, "frozen_660_accessed": False,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def main() -> None:
    dist.init_process_group("nccl")
    rank, world, local_rank = dist.get_rank(), dist.get_world_size(), int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    config = json.loads(CONFIG_PATH.read_text())
    base = json.loads((ROOT / config["base_config"]).read_text())
    config["data"], config["frozen_proposal"] = base["data"], base["frozen_proposal"]
    config_sha = file_sha256(CONFIG_PATH)
    if world != int(config["hardware"]["world_size"]):
        raise RuntimeError("complete candidate energy requires exactly eight local GPUs")
    names = [None] * world
    dist.all_gather_object(names, torch.cuda.get_device_name(local_rank))
    families = [x for x in config["hardware"]["supported_families"] if all(x in str(n).upper() for n in names)]
    if len(families) != 1:
        raise RuntimeError(f"requires homogeneous H100 or H200 GPUs, got {names}")
    family = families[0]
    profile = config["hardware"]["profiles"][family]
    micro, accumulation = int(profile["micro_batch_per_gpu"]), int(profile["gradient_accumulation"])
    if micro * accumulation * world != int(config["hardware"]["global_batch"]):
        raise RuntimeError("complete candidate energy global batch changed")
    if file_sha256(ROOT / config["data"]["manifest"]) != config["data"]["manifest_sha256"]:
        raise RuntimeError("complete candidate energy manifest identity changed")
    rows = load_task_rows(config)
    validation_rows = fixed_validation_rows(
        rows["validation"], int(config["data"]["validation_subset_records"]),
        int(config["data"]["validation_crop_seed"]),
    )
    grouped = {}
    for row in rows["train"]:
        grouped.setdefault(row["source_group"], []).append(row)
    proposal_model = load_frozen_proposal(config, device)
    residual_model = load_residual_basis(config, base, device)
    negative_models = load_negative_models(config, device)
    model = CompleteCandidateEnergy(
        condition_dim=int(config["model"]["condition_dim"]),
        widths=tuple(int(value) for value in config["model"]["widths"]),
        blocks=int(config["model"]["blocks"]),
        initial_log_variance=float(config["model"]["initial_log_variance"]),
    ).to(device)
    distributed = DDP(model, device_ids=[local_rank], gradient_as_bucket_view=False)
    global_lpips = build_lpips(device)
    spatial_lpips = build_lpips(device)
    spatial_lpips.spatial = True
    optimizer = torch.optim.AdamW(
        distributed.parameters(), lr=float(config["schedule"]["learning_rate"]),
        betas=(.9, .999), weight_decay=float(config["schedule"]["weight_decay"]),
    )
    run_id = os.environ.get("COMPLETE_CANDIDATE_ENERGY_RUN_ID", "qualification_v1")
    total = int(os.environ.get("COMPLETE_CANDIDATE_ENERGY_TOTAL_STEPS", "25"))
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
            "status": "foundir_complete_candidate_energy_noise_v1_started",
            "run_id": run_id, "total_steps": total, "config_sha256": config_sha,
            "parameters": parameter_count(model), "hardware": hardware, **config["access"],
        })
    start_step = 0
    checkpoints = sorted(output.glob("step_*.pt"))
    if checkpoints:
        payload = torch.load(checkpoints[-1], map_location="cpu", weights_only=False)
        if payload.get("config_sha256") != config_sha:
            raise RuntimeError("complete candidate energy resume config changed")
        model.load_state_dict(payload["model"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(device)
        start_step = int(payload["step"])
    term_names = (
        "risk_regression", "mse_ranking", "perceptual_ranking", "negative_margin",
        "mean_log_variance", "oracle_non_safe_fraction",
    )
    verified, started = set(), time.monotonic()
    distributed.train()
    global_batch = int(config["hardware"]["global_batch"])
    for step in range(start_step + 1, total + 1):
        lr = learning_rate(config, step)
        for group in optimizer.param_groups:
            group["lr"] = lr
        selected_rows = sample_global_batch(
            grouped, task="complete_candidate_energy_noise_v1", step=step,
            seed=int(config["schedule"]["seed"]), count=global_batch,
        )
        local = selected_rows[rank * (global_batch // world):(rank + 1) * (global_batch // world)]
        optimizer.zero_grad(set_to_none=True)
        reduced = torch.zeros(1 + len(term_names), device=device)
        negative_role = NEGATIVE_ROLES[(step - 1) % len(NEGATIVE_ROLES)]
        for accumulation_index in range(accumulation):
            chunk = local[accumulation_index * micro:(accumulation_index + 1) * micro]
            batch_rows = [row for row, _ in chunk]
            pairs = [
                load_training_patch(
                    row, crop_size=crop_for_step(config, step), augmentation_seed=seed, verified=verified
                ) for row, seed in chunk
            ]
            observation = torch.stack([pair[0] for pair in pairs]).to(device)
            target = torch.stack([pair[1] for pair in pairs]).to(device)
            proposal, deployable = frozen_deployable(proposal_model, residual_model, observation)
            negative = frozen_negative(
                negative_role, negative_models, proposal_model, observation, proposal
            )
            training_candidates = torch.cat((deployable, negative[:, None]), dim=1)
            severity = severity_from_rows(batch_rows, device)
            sync = nullcontext() if accumulation_index + 1 == accumulation else distributed.no_sync()
            with sync:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    risks = distributed(observation, proposal, training_candidates, severity)
                loss, terms = objective(
                    risks, training_candidates, target, spatial_lpips, config
                )
                (loss / accumulation).backward()
            reduced[0] += loss.detach() / accumulation
            for index, name in enumerate(term_names, 1):
                reduced[index] += terms[name].detach() / accumulation
        norm = torch.nn.utils.clip_grad_norm_(
            distributed.parameters(), float(config["schedule"]["gradient_clip_norm"])
        )
        if not bool(torch.isfinite(norm)) or not bool(torch.isfinite(reduced).all()):
            raise RuntimeError("non-finite complete candidate energy training state")
        optimizer.step()
        if step % 5 == 0:
            dist.all_reduce(reduced)
            if rank == 0:
                values = (reduced / world).tolist()
                record = {
                    "step": step, "lr": lr, "loss": values[0],
                    "negative_role": negative_role, "elapsed_seconds": time.monotonic() - started,
                }
                record.update({name: values[index] for index, name in enumerate(term_names, 1)})
                print(json.dumps(record), flush=True)
        if step == total or step % int(config["schedule"]["steps_per_effective_pass"]) == 0:
            dist.barrier()
            validation = validate(
                model, proposal_model, residual_model, validation_rows, config,
                global_lpips, device, rank, world,
            )
            dist.barrier()
            if rank == 0:
                elapsed = time.monotonic() - started
                save_checkpoint(
                    output / f"step_{step:06d}.pt", model, optimizer, step=step,
                    config_sha256=config_sha, validation=validation, hardware=hardware,
                    elapsed_seconds=elapsed,
                )
                atomic_json(output / f"validation_step_{step:06d}.json", {
                    "status": "foundir_complete_candidate_energy_noise_v1_development_validation",
                    "step": step, "validation": validation,
                    "seconds_per_optimizer_step": elapsed / max(1, step - start_step),
                    "projected_primary_wall_hours": elapsed / max(1, step - start_step) * int(config["schedule"]["primary_steps"]) / 3600,
                    "projected_primary_gpu_hours": elapsed / max(1, step - start_step) * int(config["schedule"]["primary_steps"]) / 3600 * world,
                    "hardware": hardware, **config["access"],
                })
    if rank == 0:
        elapsed = time.monotonic() - started
        atomic_json(output / "terminal_result.json", {
            "status": "foundir_complete_candidate_energy_noise_v1_training_complete",
            "completed_optimizer_steps": total, "wall_clock_hours": elapsed / 3600,
            "allocated_gpu_hours": elapsed / 3600 * world, "hardware": hardware,
            **config["access"],
        })
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
