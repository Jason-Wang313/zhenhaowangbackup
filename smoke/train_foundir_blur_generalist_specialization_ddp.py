#!/usr/bin/env python3
"""Specialize the released FoundIR backbone for Blur on development data only."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party/foundir"))

from lowlight_restoration.flux_prompt_cache import file_sha256  # noqa: E402
from lowlight_restoration.foundir_flux2_protocol import resolve_workspace_path, stable_digest  # noqa: E402
from lowlight_restoration.pixel_refinement import spatial_highpass  # noqa: E402
from smoke.evaluate_foundir_five_type_flux2_ddp import (  # noqa: E402
    load_aligned_crop,
    selected_from_contract,
    torchrun_worker_identity,
    write_once,
)
from smoke.train_foundir_blur_pixel_expert_gate_ddp import (  # noqa: E402
    BlurPresentationDataset,
    RankPresentationSampler,
    collate,
    learning_rate,
    mean_metrics,
    scalar_metrics,
)
from smoke.train_foundir_five_type_flux2_full_ddp import load_rows  # noqa: E402


def validate_config(config: Mapping[str, object]) -> tuple[dict, Path, Path]:
    if (
        config.get("schema_version") != 1
        or config.get("status") != "development_only_pretrained_blur_specialization_gate"
        or config.get("checkpoint_selection_uses_validation_only") is not True
        or config.get("sealed_internal_test_accessed") is not False
        or config.get("protected_foundir_test_accessed") is not False
    ):
        raise RuntimeError("pretrained Blur specialization contract is invalid")
    parent_path = resolve_workspace_path(ROOT, config["parent_config"])
    conditions = resolve_workspace_path(ROOT, config["conditions"])
    pretrained = resolve_workspace_path(ROOT, config["pretrained_checkpoint"])
    if file_sha256(parent_path) != config.get("parent_config_sha256"):
        raise RuntimeError("parent config changed")
    if file_sha256(conditions) != config.get("conditions_sha256"):
        raise RuntimeError("condition cache changed")
    if file_sha256(pretrained) != config.get("pretrained_checkpoint_sha256"):
        raise RuntimeError("released pretrained checkpoint changed")
    training = config["training"]
    if int(training["global_batch_size"]) != (
        int(training["world_size"])
        * int(training["batch_size_per_rank"])
        * int(training["gradient_accumulation_steps"])
    ):
        raise RuntimeError("specialization global batch contract changed")
    checkpoints = [int(value) for value in training["checkpoint_steps"]]
    if checkpoints != sorted(set(checkpoints)) or checkpoints[-1] != int(training["optimizer_steps"]):
        raise RuntimeError("specialization checkpoint schedule is invalid")
    return json.loads(parent_path.read_text()), conditions, pretrained


def build_diffusion(config: Mapping[str, object], pretrained: Path, device: torch.device):
    from src.model import ResidualDiffusion, UnetRes

    model_config = config["model"]
    network = UnetRes(
        dim=int(model_config["dim"]),
        dim_mults=tuple(model_config["dim_mults"]),
        num_unet=1,
        condition=True,
        objective="pred_res",
        test_res_or_noise="res",
    )
    diffusion = ResidualDiffusion(
        network,
        image_size=1024,
        timesteps=int(model_config["timesteps"]),
        delta_end=float(model_config["delta_end"]),
        sampling_timesteps=int(model_config["sampling_steps"]),
        ddim_sampling_eta=float(model_config["ddim_eta"]),
        objective="pred_res",
        loss_type="l1",
        condition=True,
        sum_scale=float(model_config["sum_scale"]),
        test_res_or_noise="res",
    )
    payload = torch.load(pretrained, map_location="cpu", weights_only=False)
    state = {
        key[len("ema_model.") :]: value
        for key, value in payload["ema"].items()
        if key.startswith("ema_model.")
    }
    diffusion.load_state_dict(state, strict=True)
    diffusion.init()
    return diffusion.to(device)


def compute_loss(
    wrapped: DistributedDataParallel,
    diffusion,
    source: torch.Tensor,
    target: torch.Tensor,
    config: Mapping[str, object],
    lpips_metric,
    *,
    step: int,
    rng_seed: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    source_m11 = source * 2.0 - 1.0
    target_m11 = target * 2.0 - 1.0
    target_residual = source_m11 - target_m11
    generator = torch.Generator(device=source.device).manual_seed(rng_seed)
    timestep = torch.randint(
        0,
        diffusion.num_timesteps,
        (source.shape[0],),
        device=source.device,
        generator=generator,
    )
    noise = torch.randn(
        source_m11.shape,
        dtype=source_m11.dtype,
        device=source.device,
        generator=generator,
    )
    noisy = diffusion.q_sample(target_m11, target_residual, source_m11, timestep, noise=noise)
    model_input = torch.cat((noisy, source_m11), dim=1)
    time_coordinates = [
        diffusion.alphas_cumsum[timestep] * diffusion.num_timesteps,
        diffusion.betas_cumsum[timestep] * diffusion.num_timesteps,
    ]
    predicted_residual = wrapped(model_input, time_coordinates)[0]
    predicted_clean = ((source_m11 - predicted_residual).clamp(-1, 1) + 1.0) * 0.5
    contract = config["training"]["loss"]
    residual_l1 = F.l1_loss(predicted_residual.float(), target_residual.float())
    clean_l1 = F.l1_loss(predicted_clean.float(), target.float())
    clean_mse = F.mse_loss(predicted_clean.float(), target.float())
    edge_l1 = F.l1_loss(spatial_highpass(predicted_clean.float()), spatial_highpass(target.float()))
    lpips_active = step % int(contract["lpips_every_steps"]) == 0
    if lpips_active:
        # Keep the frozen AlexNet feature loss in FP32.  The surrounding model
        # forward uses BF16 autocast, but LPIPS is both numerically sensitive
        # and part of checkpoint selection.
        with torch.autocast(device_type=source.device.type, enabled=False):
            perceptual = lpips_metric(predicted_clean.float(), target.float())
    else:
        perceptual = predicted_clean.new_zeros((), dtype=torch.float32)
    loss = (
        residual_l1 * float(contract["residual_l1_weight"])
        + clean_l1 * float(contract["clean_l1_weight"])
        + clean_mse * float(contract["clean_mse_weight"])
        + edge_l1 * float(contract["edge_l1_weight"])
        + perceptual * float(contract["lpips_weight"])
    )
    return loss, {
        "loss": float(loss.detach()),
        "residual_l1": float(residual_l1.detach()),
        "clean_l1": float(clean_l1.detach()),
        "clean_mse": float(clean_mse.detach()),
        "edge_l1": float(edge_l1.detach()),
        "lpips": float(perceptual.detach()),
        "lpips_active": float(lpips_active),
    }


def save_checkpoint(path: Path, model, optimizer, *, step: int, config_sha: str) -> None:
    temporary = path.with_suffix(".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": 1,
            "config_sha256": config_sha,
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "protected_foundir_test_accessed": False,
        },
        temporary,
    )
    os.replace(temporary, path)


def resume(output: Path, model, optimizer, config_sha: str) -> int:
    candidates = sorted(output.glob("step_*.pt"))
    if not candidates:
        return 0
    path = candidates[-1]
    payload = torch.load(path, map_location="cpu", weights_only=False)
    step = int(path.stem.split("_", 1)[1])
    if (
        payload.get("config_sha256") != config_sha
        or payload.get("step") != step
        or payload.get("protected_foundir_test_accessed") is not False
    ):
        raise RuntimeError(f"invalid specialization resume checkpoint: {path}")
    model.load_state_dict(payload["model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    return step


def evaluate(diffusion, rows, metric, device, rank: int, world: int, seed: int) -> list[dict[str, object]]:
    diffusion.eval()
    local = []
    for index in range(rank, len(rows), world):
        row = rows[index]
        lq, gt, _ = load_aligned_crop(row, crop_size=512, seed=seed)
        source = ((lq.unsqueeze(0).to(device) + 1.0) * 0.5).float()
        target = ((gt.unsqueeze(0).to(device) + 1.0) * 0.5).float()
        torch.manual_seed(int(stable_digest(seed, "specialization-eval", row["record_id"])[:16], 16))
        with torch.inference_mode():
            prediction = diffusion.sample(source, batch_size=1, last=True, task=row["record_id"])[-1].float().clamp(0, 1)
        local.append(
            {
                "record_id": row["record_id"],
                "category": row["category"],
                "qwen_blur_active": int(json.loads(row["qwen_json"])["blur"]) > 1,
                "input": scalar_metrics(metric, target, source),
                "specialized": scalar_metrics(metric, target, prediction),
            }
        )
    gathered: list[list[dict[str, object]] | None] = [None for _ in range(world)]
    dist.all_gather_object(gathered, local)
    diffusion.train()
    return [row for shard in gathered if shard is not None for row in shard]


def selection_score(summary: Mapping[str, object], gate: Mapping[str, object]) -> float:
    metrics = summary["blur"]["specialized"]
    return min(
        float(metrics["psnr"]) - float(gate["blur_psnr_minimum"]),
        (float(gate["blur_lpips_maximum"]) - float(metrics["lpips"])) / 0.01,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config_path = resolve_workspace_path(ROOT, args.config)
    config = json.loads(config_path.read_text())
    config_sha = file_sha256(config_path)
    parent, conditions_path, pretrained = validate_config(config)
    rank, world, local_rank = torchrun_worker_identity(int(config["training"]["world_size"]))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl")
    rows = load_rows(conditions_path, int(parent["data"]["expected_five_type_records"]))
    blur_rows = [row for row in rows if row["split"] == "train" and row["category"] == "01Blur"]
    noise_rows = [row for row in rows if row["split"] == "train" and row["category"] == "05Noise"]
    validation = selected_from_contract(
        parent,
        rows,
        split="validation",
        config_sha=file_sha256(resolve_workspace_path(ROOT, config["parent_config"])),
        conditions_sha=file_sha256(conditions_path),
    )
    validation = [row for row in validation if row["category"] in {"01Blur", "05Noise"}]
    if len(blur_rows) != 105057 or len(noise_rows) != 54585 or len(validation) != 120:
        raise RuntimeError("specialization data cardinality changed")

    torch.manual_seed(int(config["data"]["seed"]))
    diffusion = build_diffusion(config, pretrained, device)
    network = diffusion.model
    wrapped = DistributedDataParallel(network, device_ids=[local_rank], broadcast_buffers=False)
    diffusion.model = wrapped.module
    optimizer = torch.optim.AdamW(
        wrapped.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
        fused=True,
    )
    output = resolve_workspace_path(ROOT, config["output"])
    output.mkdir(parents=True, exist_ok=True)
    step = resume(output, wrapped.module, optimizer, config_sha)
    steps = torch.tensor([step], device=device)
    dist.all_reduce(steps)
    if int(steps.item()) != step * world:
        raise RuntimeError("specialization ranks resumed different steps")

    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

    metric = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(device).eval().requires_grad_(False)
    dataset = BlurPresentationDataset(blur_rows, noise_rows, config)
    total_steps = int(config["training"]["optimizer_steps"])
    global_batch = int(config["training"]["global_batch_size"])
    sampler = RankPresentationSampler(start=step * global_batch, stop=total_steps * global_batch, rank=rank, world=world)
    loader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=int(config["training"]["batch_size_per_rank"]),
        num_workers=int(config["training"]["num_workers_per_rank"]),
        pin_memory=True,
        persistent_workers=True,
        collate_fn=collate,
    )
    iterator = iter(loader)
    accumulation = int(config["training"]["gradient_accumulation_steps"])
    checkpoints = set(int(value) for value in config["training"]["checkpoint_steps"])
    optimizer.zero_grad(set_to_none=True)
    while step < total_steps:
        started = time.perf_counter()
        totals = defaultdict(float)
        for micro in range(accumulation):
            batch = next(iterator)
            source = batch["source"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            sync = wrapped.no_sync() if micro + 1 < accumulation else torch.enable_grad()
            with sync, torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss, terms = compute_loss(
                    wrapped,
                    diffusion,
                    source,
                    target,
                    config,
                    metric,
                    step=step + 1,
                    rng_seed=int(
                        stable_digest(
                            int(config["data"]["seed"]),
                            "training-rng",
                            step + 1,
                            micro,
                            rank,
                        )[:16],
                        16,
                    ),
                )
                (loss / accumulation).backward()
            metric.reset()
            for key, value in terms.items():
                totals[key] += value / accumulation
        torch.nn.utils.clip_grad_norm_(wrapped.parameters(), float(config["training"]["max_grad_norm"]))
        step += 1
        lr = learning_rate(step, config)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if step % 25 == 0 or step in checkpoints:
            values = torch.tensor([totals[key] for key in sorted(totals)], device=device)
            dist.all_reduce(values)
            if rank == 0:
                values /= world
                fields = " ".join(f"{key}={float(value):.6f}" for key, value in zip(sorted(totals), values))
                print(f"FOUNDIR_BLUR_SPECIALIZE step={step}/{total_steps} lr={lr:.8f} seconds={time.perf_counter()-started:.3f} {fields}", flush=True)
        if step in checkpoints:
            if rank == 0:
                save_checkpoint(output / f"step_{step:06d}.pt", wrapped.module, optimizer, step=step, config_sha=config_sha)
            dist.barrier()
            result_path = output / f"development_step_{step:06d}.json"
            if not result_path.exists():
                result_rows = evaluate(diffusion, validation, metric, device, rank, world, int(parent["data"]["split_seed"]))
                summary = {
                    "schema_version": 1,
                    "status": "pretrained_blur_specialization_development_complete",
                    "step": step,
                    "blur": {
                        "input": mean_metrics(result_rows, "01Blur", "input"),
                        "specialized": mean_metrics(result_rows, "01Blur", "specialized"),
                        "qwen_active_records": sum(row["qwen_blur_active"] for row in result_rows if row["category"] == "01Blur"),
                    },
                    "noise": {
                        "input": mean_metrics(result_rows, "05Noise", "input"),
                        "specialized_false_positive_diagnostic": mean_metrics(result_rows, "05Noise", "specialized"),
                        "qwen_false_blur_records": sum(row["qwen_blur_active"] for row in result_rows if row["category"] == "05Noise"),
                    },
                    "config_sha256": config_sha,
                    "checkpoint": (output / f"step_{step:06d}.pt").relative_to(ROOT).as_posix(),
                    "checkpoint_sha256": file_sha256(output / f"step_{step:06d}.pt"),
                    "split": "validation",
                    "sealed_internal_test_accessed": False,
                    "protected_foundir_test_accessed": False,
                }
                if rank == 0:
                    write_once(result_path, (json.dumps(summary, indent=2, sort_keys=True) + "\n").encode())
                    print("FOUNDIR_BLUR_SPECIALIZE_EVAL " + json.dumps(summary, sort_keys=True), flush=True)
            dist.barrier()

    if rank == 0:
        candidates = [json.loads((output / f"development_step_{value:06d}.json").read_text()) for value in sorted(checkpoints)]
        winner = max(candidates, key=lambda item: (selection_score(item, config["gate"]), -int(item["step"])))
        metrics = winner["blur"]["specialized"]
        checks = {
            "blur_psnr": float(metrics["psnr"]) >= float(config["gate"]["blur_psnr_minimum"]),
            "blur_lpips": float(metrics["lpips"]) <= float(config["gate"]["blur_lpips_maximum"]),
        }
        result = {
            "schema_version": 1,
            "status": "pretrained_blur_specialization_gate_complete",
            "config_sha256": config_sha,
            "selected_step": winner["step"],
            "checkpoint": winner["checkpoint"],
            "checkpoint_sha256": winner["checkpoint_sha256"],
            "selected_development_metrics": winner,
            "checks": checks,
            "gate_pass": all(checks.values()),
            "sealed_internal_test_accessed": False,
            "protected_foundir_test_accessed": False,
        }
        write_once(output / "result.json", (json.dumps(result, indent=2, sort_keys=True) + "\n").encode())
        print("FOUNDIR_BLUR_SPECIALIZATION_GATE_PASS " + json.dumps(result, sort_keys=True), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
