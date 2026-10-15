#!/usr/bin/env python3
"""Resumable formal development training for the VLM-energy intersection model."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import sys
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lowlight_restoration.vlm_energy_objectives import (  # noqa: E402
    edge_reconstruction,
    multi_scale_reconstruction,
    trajectory_calibrated_objectives,
)
from lowlight_restoration.vlm_energy_restoration import DEGRADATIONS  # noqa: E402
from smoke.preflight_foundir_vlm_energy_intersection_v1_8gpu import build_model, load_jsonl, sha256  # noqa: E402


CONFIG_PATH = ROOT / "configs/foundir_vlm_energy_intersection_v1_train.json"


def stable_int(*parts: object) -> int:
    return int.from_bytes(hashlib.sha256(":".join(map(str, parts)).encode()).digest()[:8], "big")


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_bytes(encoded)
    os.replace(temporary, path)


class PairDataset(Dataset[tuple[Tensor, Tensor, Tensor, str, int, str]]):
    def __init__(
        self,
        rows: list[dict[str, Any]],
        targets: dict[str, dict[str, Any]],
        *,
        crop: int,
        seed: int,
        augment: bool,
    ) -> None:
        self.rows = rows
        self.targets = targets
        self.crop = int(crop)
        self.seed = int(seed)
        self.augment = bool(augment)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor, str, int, str]:
        row = self.rows[index]
        record_id = str(row["record_id"])
        with Image.open(ROOT / str(row["output_path"])) as handle:
            degraded = np.asarray(handle.convert("RGB"), dtype=np.uint8)
        with Image.open(ROOT / str(self.targets[record_id]["target_path"])) as handle:
            clean = np.asarray(handle.convert("RGB"), dtype=np.uint8)
        if degraded.shape != clean.shape or min(degraded.shape[:2]) < self.crop:
            raise RuntimeError(f"bad training pair: {record_id}")
        if self.augment:
            rng = random.Random(stable_int(self.seed, self.epoch, record_id))
            y = rng.randrange(degraded.shape[0] - self.crop + 1)
            x = rng.randrange(degraded.shape[1] - self.crop + 1)
        else:
            y = (degraded.shape[0] - self.crop) // 2
            x = (degraded.shape[1] - self.crop) // 2
        degraded = degraded[y:y+self.crop, x:x+self.crop]
        clean = clean[y:y+self.crop, x:x+self.crop]
        if self.augment:
            if rng.randrange(2):
                degraded, clean = np.flip(degraded, 1), np.flip(clean, 1)
            if rng.randrange(2):
                degraded, clean = np.flip(degraded, 0), np.flip(clean, 0)
            if rng.randrange(2):
                degraded, clean = np.swapaxes(degraded, 0, 1), np.swapaxes(clean, 0, 1)
        degraded, clean = np.ascontiguousarray(degraded), np.ascontiguousarray(clean)
        present = set(map(str, row["degradations"]))
        descriptor = torch.tensor([float(name in present) for name in DEGRADATIONS])
        combination = "+".join(sorted(present))
        return (
            torch.from_numpy(degraded).permute(2, 0, 1).float().div_(127.5).sub_(1),
            torch.from_numpy(clean).permute(2, 0, 1).float().div_(127.5).sub_(1),
            descriptor,
            record_id,
            int(row["composition_size"]),
            combination,
        )


class ProposalWarmup(nn.Module):
    """Expose only proposal-path parameters to DDP during warmup."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.condition_encoder = model.condition_encoder
        self.proposal = model.proposal

    def forward(self, observation: Tensor, descriptor: Tensor) -> Tensor:
        return self.proposal(observation, self.condition_encoder(descriptor))


class ResidualReplay:
    def __init__(self, capacity: int, spatial_size: int) -> None:
        self.values: deque[Tensor] = deque(maxlen=int(capacity))
        self.spatial_size = int(spatial_size)

    def add(self, residual: Tensor) -> None:
        reduced = F.interpolate(
            residual.detach().float(), size=(self.spatial_size, self.spatial_size),
            mode="bilinear", align_corners=False,
        ).to(device="cpu", dtype=torch.float16)
        self.values.extend(reduced.unbind(0))

    def sample(
        self, batch: int, height: int, width: int, probability: float,
        *, device: torch.device, generator: torch.Generator,
    ) -> Tensor:
        output = torch.zeros(batch, 3, height, width, device=device)
        if not self.values:
            return output
        use = torch.rand(batch, generator=generator, device=device) < float(probability)
        indices = torch.randint(len(self.values), (batch,), generator=generator, device=device).cpu().tolist()
        selected = torch.stack([self.values[index] for index in indices]).float().to(device)
        selected = F.interpolate(selected, size=(height, width), mode="bilinear", align_corners=False)
        return torch.where(use[:, None, None, None], selected, output)


def select_validation(rows: list[dict[str, Any]], per_combination: int, seed: int) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, tuple[str, ...]], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["composition_size"]), tuple(sorted(map(str, row["degradations"]))))].append(row)
    selected = []
    for key, values in sorted(grouped.items()):
        values.sort(key=lambda row: hashlib.sha256(f"{seed}:{row['record_id']}".encode()).hexdigest())
        if len(values) < per_combination:
            raise RuntimeError(f"insufficient validation records for {key}")
        selected.extend(values[:per_combination])
    return sorted(selected, key=lambda row: str(row["record_id"]))


def make_loader(
    dataset: PairDataset,
    *,
    batch: int,
    workers: int,
    sampler: DistributedSampler | None,
    shuffle: bool = False,
) -> DataLoader:
    return DataLoader(
        dataset, batch_size=batch, sampler=sampler, shuffle=shuffle,
        num_workers=workers, pin_memory=True, drop_last=False,
        persistent_workers=False,
        prefetch_factor=2 if workers else None,
    )


def validate(
    model: nn.Module,
    loader: DataLoader,
    metric: LearnedPerceptualImagePatchSimilarity,
    *,
    device: torch.device,
    rank: int,
    world: int,
    epoch: int,
    steps: int,
) -> dict[str, Any] | None:
    model.eval()
    rows = []
    for degraded, clean, descriptor, record_ids, orders, combinations in loader:
        degraded, clean = degraded.to(device, non_blocking=True), clean.to(device, non_blocking=True)
        descriptor = descriptor.to(device, non_blocking=True)
        with torch.enable_grad():
            trajectory = model(
                degraded, descriptor, steps=steps, create_graph=False,
                detach_between_steps=True,
            )
        output = trajectory.output.detach()
        mse = (output.add(1).mul(0.5) - clean.add(1).mul(0.5)).square().flatten(1).mean(dim=1)
        psnr = -10.0 * torch.log10(mse.clamp_min(1e-12))
        with torch.no_grad(), torch.autocast(device_type="cuda", enabled=False):
            lpips = metric(output.float(), clean.float()).reshape(-1)
        for index, record_id in enumerate(record_ids):
            rows.append({
                "record_id": record_id,
                "composition_size": int(orders[index]),
                "combination": combinations[index],
                "psnr": float(psnr[index]),
                "lpips": float(lpips[index]),
            })
    gathered: list[list[dict[str, Any]] | None] = [None] * world if rank == 0 else []
    dist.gather_object(rows, gathered if rank == 0 else None, dst=0)
    model.train()
    if rank != 0:
        return None
    unique = {row["record_id"]: row for part in gathered if part for row in part}
    records = list(unique.values())
    by_order: dict[str, Any] = {}
    for order in (1, 2, 3):
        values = [row for row in records if row["composition_size"] == order]
        by_order[str(order)] = {
            "records": len(values),
            "psnr": sum(row["psnr"] for row in values) / len(values),
            "lpips": sum(row["lpips"] for row in values) / len(values),
        }
    return {
        "epoch": epoch,
        "records": len(records),
        "overall": {
            "psnr": sum(row["psnr"] for row in records) / len(records),
            "lpips": sum(row["lpips"] for row in records) / len(records),
        },
        "by_composition_size": by_order,
        "per_record": records,
    }


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    phase: str,
    config_sha: str,
    validation: dict[str, Any] | None,
    world_size: int,
    global_batch: int,
) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    torch.save({
        "schema_version": 1,
        "config_sha256": config_sha,
        "epoch": epoch,
        "global_step": global_step,
        "phase": phase,
        "world_size": world_size,
        "global_batch": global_batch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "validation": validation,
    }, temporary)
    os.replace(temporary, path)
    return sha256(path)


def main() -> None:
    dist.init_process_group("nccl")
    rank, world, local_rank = dist.get_rank(), dist.get_world_size(), int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    config = json.loads(CONFIG_PATH.read_text())
    config_sha = sha256(CONFIG_PATH)
    base_path = ROOT / config["base_config"]
    if sha256(base_path) != config["base_config_sha256"]:
        raise RuntimeError("base config changed")
    if sha256(ROOT / config["autotune_result"]) != config["autotune_result_sha256"]:
        raise RuntimeError("autotune result changed")
    if sha256(ROOT / "lowlight_restoration/vlm_energy_restoration.py") != config["implementation_sha256"]:
        raise RuntimeError("model implementation changed")
    if sha256(ROOT / "lowlight_restoration/vlm_energy_objectives.py") != config["objectives_sha256"]:
        raise RuntimeError("objective implementation changed")
    base = json.loads(base_path.read_text())
    autotune = json.loads((ROOT / config["autotune_result"]).read_text())
    if autotune["status"] != "foundir_vlm_energy_intersection_v1_autotune_pass":
        raise RuntimeError("autotune did not pass")
    profiles = config["distributed"]["profiles"]
    if str(world) not in profiles:
        raise RuntimeError(f"unsupported world size {world}; supported={sorted(map(int, profiles))}")
    profile = profiles[str(world)]
    expected_local_world = int(config["distributed"]["gpus_per_node"])
    local_world = int(os.environ.get("LOCAL_WORLD_SIZE", "0"))
    if local_world != expected_local_world:
        raise RuntimeError(f"requires {expected_local_world} GPUs per node, got {local_world}")
    expected_nodes = int(profile["nodes"])
    if world != expected_nodes * expected_local_world:
        raise RuntimeError("distributed profile is internally inconsistent")
    selection = autotune["selection"]
    if selection["precision"] != "float32":
        raise RuntimeError("formal precision diverged from autotune selection")
    batch = int(profile["batch_per_gpu"])
    global_batch = batch * world
    if global_batch != int(profile["global_batch"]):
        raise RuntimeError("distributed batch profile is internally inconsistent")
    if world == 8:
        if batch != int(selection["batch_per_gpu"]):
            raise RuntimeError("single-node runtime diverged from autotune selection")
    elif bool(config["distributed"]["require_measured_multinode_benchmark"]):
        benchmark_text = os.environ.get("VLM_ENERGY_MULTINODE_BENCHMARK_RESULT", "")
        if not benchmark_text:
            raise RuntimeError("set VLM_ENERGY_MULTINODE_BENCHMARK_RESULT to the matching PASS result")
        benchmark_path = Path(benchmark_text)
        if not benchmark_path.is_absolute():
            benchmark_path = ROOT / benchmark_path
        benchmark = json.loads(benchmark_path.read_text())
        if benchmark.get("status") != "foundir_vlm_energy_multinode_benchmark_pass":
            raise RuntimeError("multi-node benchmark did not pass")
        if int(benchmark.get("world_size", -1)) != world or int(benchmark.get("batch_per_gpu", -1)) != batch:
            raise RuntimeError("multi-node benchmark profile does not match this launch")
    gpu = torch.cuda.get_device_name(local_rank)
    if not any(name in gpu for name in ("H100", "H200")):
        raise RuntimeError(f"unsupported GPU: {gpu}")

    data_manifest = ROOT / base["data"]["manifest"]
    target_manifest = ROOT / base["data"]["target_manifest"]
    if sha256(data_manifest) != config["data"]["manifest_sha256"] or sha256(target_manifest) != config["data"]["target_manifest_sha256"]:
        raise RuntimeError("formal data identity changed")
    rows = load_jsonl(data_manifest)
    targets = {str(row["record_id"]): row for row in load_jsonl(target_manifest)}
    train_rows = [row for row in rows if row["split"] == "train"]
    validation_rows = select_validation(
        [row for row in rows if row["split"] == "validation"],
        int(config["data"]["validation_records_per_combination"]),
        int(base["model"]["initialization_seed"]) + 1,
    )
    if len(train_rows) != 90000 or len(validation_rows) != int(config["data"]["validation_records"]):
        raise RuntimeError("formal train/validation count changed")

    crop = int(config["data"]["crop_size"])
    seed = int(base["model"]["initialization_seed"])
    train_dataset = PairDataset(train_rows, targets, crop=crop, seed=seed, augment=True)
    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world, rank=rank, shuffle=True, seed=seed, drop_last=False
    )
    validation_local = validation_rows[rank::world]
    validation_dataset = PairDataset(validation_local, targets, crop=crop, seed=seed + 1, augment=False)
    workers = int(config["data"]["workers_per_rank"])
    validation_loader = make_loader(validation_dataset, batch=batch, workers=workers, sampler=None)

    model = build_model(base).to(device)
    perceptual = LearnedPerceptualImagePatchSimilarity(
        net_type="alex", reduction="none", normalize=False
    ).to(device).eval().requires_grad_(False)
    output_root = ROOT / config["runtime"]["output_root"] / os.environ.get("VLM_ENERGY_TRAIN_RUN_ID", "formal_v1")
    last_path = output_root / "last.pt"
    start_epoch = global_step = 0
    resume: dict[str, Any] | None = None
    if last_path.is_file():
        resume = torch.load(last_path, map_location="cpu", weights_only=False)
        if resume.get("config_sha256") != config_sha:
            raise RuntimeError("resume checkpoint config changed")
        if int(resume.get("world_size", -1)) != world or int(resume.get("global_batch", -1)) != global_batch:
            raise RuntimeError("resume world size/global batch differs from the original run")
        model.load_state_dict(resume["model_state_dict"], strict=True)
        start_epoch = int(resume["epoch"])
        global_step = int(resume["global_step"])
    if rank == 0:
        print("VLM_ENERGY_FORMAL_SETUP " + json.dumps({
            "gpu": gpu, "world": world, "batch_per_gpu": batch, "global_batch": batch * world,
            "train_records": len(train_rows), "validation_records": len(validation_rows),
            "start_epoch": start_epoch, "global_step": global_step,
            "output_root": output_root.relative_to(ROOT).as_posix(),
        }, sort_keys=True), flush=True)

    schedule = config["schedule"]
    warmup_epochs = int(schedule["proposal_warmup_epochs"])
    total_epochs = int(schedule["total_epochs"])
    validation_epochs = set(map(int, schedule["validation_epochs"]))
    replay = ResidualReplay(
        int(config["landscape"]["replay_capacity"]),
        int(config["landscape"]["replay_spatial_size"]),
    )
    started = time.perf_counter()
    steps_per_epoch = math.ceil(len(train_rows) / global_batch)

    # Recreate DDP/optimizer at the phase boundary so warmup has no unused
    # energy parameters and joint training retains strict unused checks.
    current_phase = "proposal_warmup" if start_epoch < warmup_epochs else "joint"
    if current_phase == "proposal_warmup":
        wrapper: nn.Module = ProposalWarmup(model).to(device)
        distributed: DDP = DDP(wrapper, device_ids=[local_rank], broadcast_buffers=False)
        optimizer = torch.optim.AdamW(
            distributed.parameters(), lr=float(schedule["warmup_learning_rate"]),
            betas=tuple(map(float, schedule["betas"])), weight_decay=float(schedule["weight_decay"]),
        )
        if resume is not None and resume.get("phase") == current_phase:
            optimizer.load_state_dict(resume["optimizer_state_dict"])
    else:
        distributed = DDP(model, device_ids=[local_rank], broadcast_buffers=False)
        optimizer = torch.optim.AdamW(
            distributed.parameters(), lr=float(schedule["joint_peak_learning_rate"]),
            betas=tuple(map(float, schedule["betas"])), weight_decay=float(schedule["weight_decay"]),
        )
        if resume is not None and resume.get("phase") == current_phase:
            optimizer.load_state_dict(resume["optimizer_state_dict"])

    for epoch in range(start_epoch + 1, total_epochs + 1):
        target_phase = "proposal_warmup" if epoch <= warmup_epochs else "joint"
        if target_phase != current_phase:
            # Rebuild before changing DDP topology. Rewrapping the exact same
            # Parameter objects can retain reducer hooks from the warmup
            # wrapper on some PyTorch versions.
            transition_state = {
                key: value.detach().clone() for key, value in model.state_dict().items()
            }
            del distributed, optimizer, wrapper
            torch.cuda.empty_cache()
            model = build_model(base).to(device)
            model.load_state_dict(transition_state, strict=True)
            del transition_state
            current_phase = target_phase
            distributed = DDP(model, device_ids=[local_rank], broadcast_buffers=False)
            optimizer = torch.optim.AdamW(
                distributed.parameters(), lr=float(schedule["joint_peak_learning_rate"]),
                betas=tuple(map(float, schedule["betas"])), weight_decay=float(schedule["weight_decay"]),
            )
        train_dataset.set_epoch(epoch)
        train_sampler.set_epoch(epoch)
        train_loader = make_loader(
            train_dataset, batch=batch, workers=workers, sampler=train_sampler
        )
        if len(train_loader) != steps_per_epoch:
            raise RuntimeError(
                f"loader steps changed: expected {steps_per_epoch}, got {len(train_loader)}"
            )
        distributed.train()
        epoch_loss = 0.0
        epoch_started = time.perf_counter()
        for epoch_step, batch_data in enumerate(train_loader, start=1):
            degraded, clean, descriptor = (value.to(device, non_blocking=True) for value in batch_data[:3])
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            full = global_step % int(schedule["full_objective_interval"]) == 0
            if current_phase == "proposal_warmup":
                output = distributed(degraded, descriptor)
                loss = multi_scale_reconstruction(output, clean) + 0.1 * edge_reconstruction(output, clean)
                lpips_value = loss.new_zeros(())
                if full:
                    with torch.autocast(device_type="cuda", enabled=False):
                        lpips_value = perceptual(output.float(), clean.float()).mean()
                    loss = loss + float(base["loss"]["lpips_alex"]) * lpips_value
                progress = direction = ranking = loss.new_zeros(())
            else:
                joint_step = global_step - warmup_epochs * steps_per_epoch
                warm = int(schedule["joint_lr_warmup_steps"])
                total_joint = int(schedule["joint_epochs"]) * steps_per_epoch
                if joint_step <= warm:
                    lr = float(schedule["joint_peak_learning_rate"]) * joint_step / max(1, warm)
                else:
                    ratio = min(1.0, (joint_step - warm) / max(1, total_joint - warm))
                    cosine = 0.5 * (1.0 + math.cos(math.pi * ratio))
                    lr = float(schedule["joint_minimum_learning_rate"]) + (
                        float(schedule["joint_peak_learning_rate"]) - float(schedule["joint_minimum_learning_rate"])
                    ) * cosine
                for group in optimizer.param_groups:
                    group["lr"] = lr
                descent_steps = config["landscape"]["descent_steps_cycle"][(joint_step - 1) % 3]
                generator = torch.Generator(device=device).manual_seed(seed + rank * 10**8 + global_step)
                scale_low, scale_high = map(float, config["landscape"]["step_scale"])
                step_scale = torch.rand(degraded.shape[0], generator=generator, device=device).mul_(scale_high - scale_low).add_(scale_low)
                sigma_low, sigma_high = map(float, config["landscape"]["langevin_sigma"])
                sigma = float(torch.rand((), generator=generator, device=device).mul_(sigma_high - sigma_low).add_(sigma_low))
                perturbation = torch.randn(degraded.shape, generator=generator, device=device) * sigma
                perturbation = perturbation + replay.sample(
                    degraded.shape[0], degraded.shape[2], degraded.shape[3],
                    float(config["landscape"]["replay_probability"]), device=device, generator=generator,
                )
                trajectory = distributed(
                    degraded, descriptor, steps=int(descent_steps), create_graph=True,
                    step_scale=step_scale, initial_perturbation=perturbation,
                )
                terms = trajectory_calibrated_objectives(
                    distributed.module, degraded, clean, descriptor, trajectory,
                    ranking_margin_scale=float(base["landscape_training"]["ranking_margin_scale"]),
                    compute_quality_ranking=full,
                )
                loss = (
                    float(base["loss"]["multi_scale_reconstruction"]) * terms.reconstruction
                    + float(base["loss"]["edge_reconstruction"]) * edge_reconstruction(trajectory.output, clean)
                    + float(base["loss"]["vector_direction"]) * terms.vector_direction
                    + float(base["loss"]["vector_progress"]) * terms.vector_progress
                    + float(base["loss"]["quality_ranking"]) * terms.quality_ranking
                )
                lpips_value = loss.new_zeros(())
                if full:
                    with torch.autocast(device_type="cuda", enabled=False):
                        lpips_value = perceptual(trajectory.output.float(), clean.float()).mean()
                    loss = loss + float(base["loss"]["lpips_alex"]) * lpips_value
                progress, direction, ranking = terms.vector_progress, terms.vector_direction, terms.quality_ranking
                replay.add(trajectory.output - trajectory.proposal)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(f"non-finite loss at epoch {epoch} step {epoch_step}")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(distributed.parameters(), float(schedule["gradient_clip_norm"]))
            if not bool(torch.isfinite(norm)):
                raise RuntimeError(f"non-finite gradient at epoch {epoch} step {epoch_step}")
            optimizer.step()
            epoch_loss += float(loss.detach())
            if rank == 0 and (epoch_step == 1 or epoch_step % 25 == 0 or epoch_step == len(train_loader)):
                elapsed = time.perf_counter() - started
                completed = (epoch - 1) * len(train_loader) + epoch_step
                total = total_epochs * len(train_loader)
                print("VLM_ENERGY_FORMAL_PROGRESS " + json.dumps({
                    "epoch": epoch, "total_epochs": total_epochs, "phase": current_phase,
                    "epoch_step": epoch_step, "steps_per_epoch": len(train_loader),
                    "global_step": global_step, "loss": float(loss.detach()),
                    "lpips": float(lpips_value.detach()), "vector_direction": float(direction.detach()),
                    "vector_progress": float(progress.detach()), "quality_ranking": float(ranking.detach()),
                    "gradient_norm": float(norm), "elapsed_seconds": elapsed,
                    "estimated_total_seconds": elapsed / max(1, completed) * total,
                }, sort_keys=True), flush=True)

        dist.barrier()
        validation = None
        if epoch in validation_epochs:
            validation_model = distributed.module if current_phase == "joint" else model
            validation = validate(
                validation_model, validation_loader, perceptual, device=device,
                rank=rank, world=world, epoch=epoch,
                steps=(
                    0 if current_phase == "proposal_warmup"
                    else int(config["validation"]["fixed_descent_steps"])
                ),
            )
            if rank == 0 and validation is not None:
                write_json_atomic(output_root / f"validation_epoch_{epoch:03d}.json", validation)
                print("VLM_ENERGY_VALIDATION " + json.dumps({
                    "epoch": epoch,
                    "overall": validation["overall"],
                    "by_composition_size": validation["by_composition_size"],
                }, sort_keys=True), flush=True)
        if rank == 0:
            checkpoint_model = distributed.module if current_phase == "joint" else model
            digest = save_checkpoint(
                last_path, model=checkpoint_model, optimizer=optimizer,
                epoch=epoch, global_step=global_step, phase=current_phase,
                config_sha=config_sha, validation=validation,
                world_size=world, global_batch=global_batch,
            )
            sidecar = {
                "epoch": epoch, "global_step": global_step, "phase": current_phase,
                "mean_epoch_loss": epoch_loss / len(train_loader),
                "elapsed_epoch_seconds": time.perf_counter() - epoch_started,
                "checkpoint": last_path.relative_to(ROOT).as_posix(),
                "checkpoint_sha256": digest,
                "validation": validation,
                "parent_test_accessed": False,
                "official_foundir_test_accessed": False,
                "sealed_internal_accessed": False,
            }
            write_json_atomic(output_root / f"epoch_{epoch:03d}.json", sidecar)
            if validation is not None:
                candidate = output_root / f"model_epoch_{epoch:03d}.pt"
                temporary = candidate.with_name(candidate.name + f".tmp.{os.getpid()}")
                torch.save({
                    "schema_version": 1, "config_sha256": config_sha,
                    "epoch": epoch, "global_step": global_step,
                    "model_state_dict": checkpoint_model.state_dict(),
                    "validation": validation,
                }, temporary)
                os.replace(temporary, candidate)
        dist.barrier()

    if rank == 0:
        result = {
            "schema_version": 1,
            "status": "foundir_vlm_energy_intersection_v1_training_complete",
            "config_sha256": config_sha,
            "epochs": total_epochs,
            "global_step": global_step,
            "elapsed_seconds": time.perf_counter() - started,
            "last_checkpoint": last_path.relative_to(ROOT).as_posix(),
            "last_checkpoint_sha256": sha256(last_path),
            "parent_test_accessed": False,
            "official_foundir_test_accessed": False,
            "sealed_internal_accessed": False,
        }
        write_json_atomic(output_root / "result.json", result)
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
