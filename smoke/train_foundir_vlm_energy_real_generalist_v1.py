#!/usr/bin/env python3
"""Real-train-split continuation with balanced sampling and raw/EMA validation."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import socket
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from torch import Tensor
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.utils.data import DataLoader, Dataset, Sampler
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lowlight_restoration.vlm_energy_objectives import (  # noqa: E402
    edge_reconstruction,
    trajectory_calibrated_objectives,
)
from lowlight_restoration.vlm_energy_restoration import DEGRADATIONS  # noqa: E402
from smoke.preflight_foundir_vlm_energy_intersection_v1_8gpu import build_model, sha256  # noqa: E402
from smoke.train_foundir_vlm_energy_intersection_v1_8gpu import (  # noqa: E402
    ResidualReplay,
    stable_int,
    validate,
    write_json_atomic,
)


CONFIG_PATH = ROOT / "configs/foundir_vlm_energy_real_generalist_v1.json"


def resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def load_rows(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    train: list[dict[str, Any]] = []
    validation: list[dict[str, Any]] = []
    counts: dict[str, int] = defaultdict(int)
    with path.open(newline="") as handle:
        for source in csv.DictReader(handle):
            split = source["split"]
            counts[split] += 1
            if split not in {"train", "validation"}:
                continue
            degradations = list(map(str, json.loads(source["degradations_json"])))
            row = {
                "record_id": source["record_id"],
                "category": source["category"],
                "composition_size": int(source["composition_size"]),
                "degradations": degradations,
                "lq_path": source["lq_path"],
                "gt_path": source["gt_path"],
            }
            (train if split == "train" else validation).append(row)
    return train, validation, dict(counts)


class RealPairDataset(Dataset[tuple[Tensor, Tensor, Tensor, str, int, str]]):
    def __init__(self, rows: list[dict[str, Any]], *, crop: int, seed: int, augment: bool) -> None:
        self.rows = rows
        self.crop = int(crop)
        self.seed = int(seed)
        self.augment = bool(augment)
        self.pass_index = 0

    def set_pass(self, pass_index: int) -> None:
        self.pass_index = int(pass_index)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor, str, int, str]:
        row = self.rows[index]
        record_id = str(row["record_id"])
        with Image.open(resolve(str(row["lq_path"]))) as handle:
            degraded = np.asarray(handle.convert("RGB"), dtype=np.uint8)
        with Image.open(resolve(str(row["gt_path"]))) as handle:
            clean = np.asarray(handle.convert("RGB"), dtype=np.uint8)
        if degraded.shape != clean.shape or min(degraded.shape[:2]) < self.crop:
            raise RuntimeError(f"bad real pair: {record_id}")
        if self.augment:
            rng = random.Random(stable_int(self.seed, self.pass_index, record_id))
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
        present = set(map(str, row["degradations"]))
        descriptor = torch.tensor([float(name in present) for name in DEGRADATIONS])
        return (
            torch.from_numpy(np.ascontiguousarray(degraded)).permute(2, 0, 1).float().div_(127.5).sub_(1),
            torch.from_numpy(np.ascontiguousarray(clean)).permute(2, 0, 1).float().div_(127.5).sub_(1),
            descriptor,
            record_id,
            int(row["composition_size"]),
            "+".join(sorted(present)),
        )


class HierarchicalBalancedDistributedSampler(Sampler[int]):
    """Equalize composition sizes, then categories, with deterministic cycling."""

    def __init__(self, rows: list[dict[str, Any]], *, world: int, rank: int, seed: int) -> None:
        self.world, self.rank, self.seed = int(world), int(rank), int(seed)
        grouped: dict[int, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
        for index, row in enumerate(rows):
            grouped[int(row["composition_size"])][str(row["category"])].append(index)
        if sorted(grouped) != [1, 2, 3]:
            raise RuntimeError(f"expected composition tiers 1/2/3, got {sorted(grouped)}")
        self.grouped = {order: dict(values) for order, values in grouped.items()}
        self.total = math.ceil(len(rows) / self.world) * self.world
        self.pass_index = 0

    def set_pass(self, pass_index: int) -> None:
        self.pass_index = int(pass_index)

    def __len__(self) -> int:
        return self.total // self.world

    def __iter__(self) -> Iterator[int]:
        pools: dict[tuple[int, str], list[int]] = {}
        positions: dict[tuple[int, str], int] = {}
        cycles: dict[tuple[int, str], int] = defaultdict(int)
        categories = {order: sorted(groups) for order, groups in self.grouped.items()}

        def refill(key: tuple[int, str]) -> None:
            order, category = key
            generator = torch.Generator().manual_seed(
                stable_int(self.seed, self.pass_index, order, category, cycles[key])
            )
            source = self.grouped[order][category]
            permutation = torch.randperm(len(source), generator=generator).tolist()
            pools[key] = [source[index] for index in permutation]
            positions[key] = 0
            cycles[key] += 1

        local: list[int] = []
        category_positions: dict[int, int] = defaultdict(int)
        for global_position in range(self.total):
            order = 1 + global_position % 3
            choices = categories[order]
            category = choices[category_positions[order] % len(choices)]
            category_positions[order] += 1
            key = (order, category)
            if key not in pools or positions[key] >= len(pools[key]):
                refill(key)
            value = pools[key][positions[key]]
            positions[key] += 1
            if global_position % self.world == self.rank:
                local.append(value)
        if len(local) != len(self):
            raise RuntimeError("balanced sampler length changed")
        return iter(local)


def make_loader(dataset: Dataset, *, batch: int, workers: int, sampler: Sampler[int] | None) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch,
        sampler=sampler,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=bool(workers),
        prefetch_factor=2 if workers else None,
    )


def save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    ema: AveragedModel,
    optimizer: torch.optim.Optimizer,
    pass_index: int,
    real_global_step: int,
    config_sha: str,
    init_path: Path,
    init_sha: str,
    world: int,
    global_batch: int,
    validation: dict[str, Any] | None,
) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    torch.save({
        "schema_version": 1,
        "config_sha256": config_sha,
        "pass_index": pass_index,
        "real_global_step": real_global_step,
        "world_size": world,
        "global_batch": global_batch,
        "initial_checkpoint": init_path.relative_to(ROOT).as_posix(),
        "initial_checkpoint_sha256": init_sha,
        "model_state_dict": model.state_dict(),
        "ema_model_state_dict": ema.module.state_dict(),
        "ema_n_averaged": int(ema.n_averaged.item()),
        "optimizer_state_dict": optimizer.state_dict(),
        "validation": validation,
        "internal_test_accessed": False,
        "parent_test_accessed": False,
        "official_foundir_test_accessed": False,
        "sealed_internal_accessed": False,
    }, temporary)
    os.replace(temporary, path)
    return sha256(path)


def main() -> None:
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    cfg = json.loads(CONFIG_PATH.read_text())
    config_sha = sha256(CONFIG_PATH)
    base_path = ROOT / cfg["base_config"]
    if sha256(base_path) != cfg["base_config_sha256"]:
        raise RuntimeError("base config changed")
    if sha256(ROOT / "lowlight_restoration/vlm_energy_restoration.py") != cfg["implementation_sha256"]:
        raise RuntimeError("model implementation changed")
    if sha256(ROOT / "lowlight_restoration/vlm_energy_objectives.py") != cfg["objectives_sha256"]:
        raise RuntimeError("objective implementation changed")
    profile = cfg["distributed"]["profiles"].get(str(world))
    if profile is None:
        raise RuntimeError(f"unsupported world size: {world}")
    if int(os.environ.get("LOCAL_WORLD_SIZE", "0")) != int(cfg["distributed"]["gpus_per_node"]):
        raise RuntimeError("requires exactly eight GPUs per node")
    batch = int(profile["batch_per_gpu"])
    global_batch = batch * world
    if global_batch != int(profile["global_batch"]):
        raise RuntimeError("global batch profile changed")
    properties = torch.cuda.get_device_properties(local_rank)
    gpu = properties.name
    gpu_family = next((name for name in ("H200", "H100") if name in gpu.upper()), "UNSUPPORTED")
    identity = {
        "rank": rank,
        "local_rank": local_rank,
        "host": socket.gethostname(),
        "gpu": gpu,
        "gpu_family": gpu_family,
        "total_memory_bytes": properties.total_memory,
        "compute_capability": f"{properties.major}.{properties.minor}",
    }
    identities: list[dict[str, Any] | None] = [None] * world
    dist.all_gather_object(identities, identity)
    if rank == 0:
        print(
            "VLM_ENERGY_REAL_GPU_IDENTITIES " + json.dumps(identities, sort_keys=True),
            flush=True,
        )
    host_counts: dict[str, int] = {}
    for item in identities:
        assert item is not None
        host_counts[item["host"]] = host_counts.get(item["host"], 0) + 1
    expected_nodes = int(profile["nodes"])
    if len(host_counts) != expected_nodes or set(host_counts.values()) != {8}:
        raise RuntimeError(f"physical-node layout mismatch: {host_counts}")
    families = {item["gpu_family"] for item in identities if item}
    capabilities = {item["compute_capability"] for item in identities if item}
    minimum_memory = min(int(item["total_memory_bytes"]) for item in identities if item)
    if "UNSUPPORTED" in families:
        raise RuntimeError(
            "unsupported GPU family: "
            + json.dumps(identities, sort_keys=True)
        )
    if len(capabilities) != 1 or minimum_memory < 80_000_000_000:
        raise RuntimeError(
            "incompatible accelerator capability or memory: "
            + json.dumps(identities, sort_keys=True)
        )

    output_root = ROOT / cfg["runtime"]["output_root"] / os.environ.get(
        "VLM_ENERGY_REAL_RUN_ID", "real_generalist_v1"
    )
    if rank == 0:
        write_json_atomic(output_root / "hardware_identity.json", {
            "schema_version": 1,
            "world_size": world,
            "nodes": host_counts,
            "gpu_families": sorted(families),
            "heterogeneous_gpu_families": len(families) > 1,
            "gpu_identities": identities,
            "internal_test_accessed": False,
            "parent_test_accessed": False,
            "official_foundir_test_accessed": False,
            "sealed_internal_accessed": False,
        })

    data = cfg["data"]
    manifest = ROOT / data["manifest"]
    if sha256(manifest) != data["manifest_sha256"]:
        raise RuntimeError("real manifest identity changed")
    train_rows, validation_rows, split_counts = load_rows(manifest)
    expected_counts = {
        "train": int(data["expected_train_records"]),
        "validation": int(data["expected_validation_records"]),
        "internal_test": int(data["expected_internal_test_records"]),
    }
    if split_counts != expected_counts:
        raise RuntimeError(f"real split counts changed: {split_counts}")
    max_validation = int(os.environ.get("VLM_ENERGY_REAL_MAX_VALIDATION_RECORDS", "0"))
    if max_validation:
        grouped_validation: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in validation_rows:
            grouped_validation[int(row["composition_size"])].append(row)
        selected_validation: list[dict[str, Any]] = []
        quota = max(1, max_validation // 3)
        for order in (1, 2, 3):
            values = sorted(
                grouped_validation[order],
                key=lambda row: hashlib.sha256(str(row["record_id"]).encode()).hexdigest(),
            )
            selected_validation.extend(values[:quota])
        validation_rows = selected_validation[:max_validation]

    base = json.loads(base_path.read_text())
    seed = int(base["model"]["initialization_seed"])
    crop = int(data["crop_size"])
    train_dataset = RealPairDataset(train_rows, crop=crop, seed=seed + 101, augment=True)
    train_sampler = HierarchicalBalancedDistributedSampler(
        train_rows, world=world, rank=rank, seed=seed + 103
    )
    validation_local = validation_rows[rank::world]
    validation_dataset = RealPairDataset(
        validation_local, crop=crop, seed=seed + 107, augment=False
    )
    workers = int(data["workers_per_rank"])
    validation_loader = make_loader(validation_dataset, batch=batch, workers=workers, sampler=None)

    init_text = os.environ.get("VLM_ENERGY_REAL_INIT_CHECKPOINT", "")
    if not init_text:
        raise RuntimeError("set VLM_ENERGY_REAL_INIT_CHECKPOINT")
    init_path = resolve(init_text)
    init_sha = sha256(init_path)
    model = build_model(base).to(device)
    last_path = output_root / "last.pt"
    start_pass = real_global_step = 0
    resume = None
    if last_path.is_file():
        resume = torch.load(last_path, map_location="cpu", weights_only=False)
        if resume.get("config_sha256") != config_sha:
            raise RuntimeError("real continuation config changed")
        if int(resume.get("world_size", -1)) != world or int(resume.get("global_batch", -1)) != global_batch:
            raise RuntimeError("real continuation world/global batch changed")
        if resume.get("initial_checkpoint_sha256") != init_sha:
            raise RuntimeError("real continuation initial checkpoint changed")
        model.load_state_dict(resume["model_state_dict"], strict=True)
        start_pass = int(resume["pass_index"])
        real_global_step = int(resume["real_global_step"])
    else:
        initial = torch.load(init_path, map_location="cpu", weights_only=False)
        if initial.get("phase") != "joint" or not isinstance(initial.get("model_state_dict"), dict):
            raise RuntimeError("initial checkpoint is not a joint Generalist checkpoint")
        model.load_state_dict(initial["model_state_dict"], strict=True)

    distributed = DDP(model, device_ids=[local_rank], broadcast_buffers=False)
    schedule = cfg["schedule"]
    optimizer = torch.optim.AdamW(
        distributed.parameters(),
        lr=float(schedule["learning_rate"]),
        betas=tuple(map(float, schedule["betas"])),
        weight_decay=float(schedule["weight_decay"]),
    )
    ema = AveragedModel(
        distributed.module,
        device=device,
        multi_avg_fn=get_ema_multi_avg_fn(float(cfg["ema"]["decay"])),
        use_buffers=True,
    )
    if resume is not None:
        optimizer.load_state_dict(resume["optimizer_state_dict"])
        ema.module.load_state_dict(resume["ema_model_state_dict"], strict=True)
        ema.n_averaged.fill_(int(resume["ema_n_averaged"]))
    perceptual = LearnedPerceptualImagePatchSimilarity(
        net_type="alex", reduction="none", normalize=False
    ).to(device).eval().requires_grad_(False)
    replay = ResidualReplay(
        int(cfg["landscape"]["replay_capacity"]),
        int(cfg["landscape"]["replay_spatial_size"]),
    )

    passes = int(schedule["passes"])
    steps_per_pass = math.ceil(len(train_sampler) / batch)
    total_steps = passes * steps_per_pass
    max_steps = int(os.environ.get("VLM_ENERGY_REAL_MAX_STEPS", "0"))
    qualification = max_steps > 0
    if rank == 0:
        print("VLM_ENERGY_REAL_SETUP " + json.dumps({
            "gpu": gpu,
            "gpu_families": sorted(families),
            "heterogeneous_gpu_families": len(families) > 1,
            "nodes": host_counts,
            "world": world,
            "batch_per_gpu": batch,
            "global_batch": global_batch,
            "train_records": len(train_rows),
            "validation_records": len(validation_rows),
            "steps_per_pass": steps_per_pass,
            "start_pass": start_pass,
            "real_global_step": real_global_step,
            "initial_checkpoint_sha256": init_sha,
            "balanced_sampling": data["sampling"],
            "ema_decay": cfg["ema"]["decay"],
            "qualification_max_steps": max_steps,
        }, sort_keys=True), flush=True)

    started = time.perf_counter()
    stopped_for_qualification = False
    for pass_index in range(start_pass + 1, passes + 1):
        train_dataset.set_pass(pass_index)
        train_sampler.set_pass(pass_index)
        train_loader = make_loader(
            train_dataset, batch=batch, workers=workers, sampler=train_sampler
        )
        distributed.train()
        pass_loss = 0.0
        pass_started = time.perf_counter()
        completed_in_pass = 0
        for pass_step, batch_data in enumerate(train_loader, start=1):
            degraded, clean, descriptor = (
                value.to(device, non_blocking=True) for value in batch_data[:3]
            )
            optimizer.zero_grad(set_to_none=True)
            real_global_step += 1
            completed_in_pass += 1
            warmup = int(schedule["warmup_steps"])
            if real_global_step <= warmup:
                lr = float(schedule["learning_rate"]) * real_global_step / max(1, warmup)
            else:
                ratio = min(1.0, (real_global_step - warmup) / max(1, total_steps - warmup))
                cosine = 0.5 * (1.0 + math.cos(math.pi * ratio))
                lr = float(schedule["minimum_learning_rate"]) + (
                    float(schedule["learning_rate"]) - float(schedule["minimum_learning_rate"])
                ) * cosine
            for group in optimizer.param_groups:
                group["lr"] = lr

            full = real_global_step % int(schedule["full_objective_interval"]) == 0
            cycle = list(map(int, cfg["landscape"]["descent_steps_cycle"]))
            descent_steps = cycle[(real_global_step - 1) % len(cycle)]
            generator = torch.Generator(device=device).manual_seed(
                seed + 2 * 10**8 + rank * 10**6 + real_global_step
            )
            scale_low, scale_high = map(float, cfg["landscape"]["step_scale"])
            step_scale = torch.rand(
                degraded.shape[0], generator=generator, device=device
            ).mul_(scale_high - scale_low).add_(scale_low)
            sigma_low, sigma_high = map(float, cfg["landscape"]["langevin_sigma"])
            sigma = float(torch.rand((), generator=generator, device=device).mul_(sigma_high - sigma_low).add_(sigma_low))
            perturbation = torch.randn(degraded.shape, generator=generator, device=device) * sigma
            perturbation = perturbation + replay.sample(
                degraded.shape[0], degraded.shape[2], degraded.shape[3],
                float(cfg["landscape"]["replay_probability"]),
                device=device,
                generator=generator,
            )
            trajectory = distributed(
                degraded,
                descriptor,
                steps=descent_steps,
                create_graph=True,
                step_scale=step_scale,
                initial_perturbation=perturbation,
            )
            terms = trajectory_calibrated_objectives(
                distributed.module,
                degraded,
                clean,
                descriptor,
                trajectory,
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
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(f"non-finite loss at pass {pass_index} step {pass_step}")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(
                distributed.parameters(), float(schedule["gradient_clip_norm"])
            )
            if not bool(torch.isfinite(norm)):
                raise RuntimeError(f"non-finite gradient at pass {pass_index} step {pass_step}")
            optimizer.step()
            if real_global_step % int(cfg["ema"]["update_every_steps"]) == 0:
                ema.update_parameters(distributed.module)
            replay.add(trajectory.output - trajectory.proposal)
            pass_loss += float(loss.detach())
            if rank == 0 and (pass_step == 1 or pass_step % 25 == 0):
                elapsed = time.perf_counter() - started
                print("VLM_ENERGY_REAL_PROGRESS " + json.dumps({
                    "pass": pass_index,
                    "pass_step": pass_step,
                    "steps_per_pass": steps_per_pass,
                    "real_global_step": real_global_step,
                    "loss": float(loss.detach()),
                    "lpips": float(lpips_value.detach()),
                    "gradient_norm": float(norm),
                    "learning_rate": lr,
                    "ema_updates": int(ema.n_averaged.item()),
                    "elapsed_seconds": elapsed,
                }, sort_keys=True), flush=True)
            if qualification and real_global_step >= max_steps:
                stopped_for_qualification = True
                break

        dist.barrier()
        validation = None
        should_validate = stopped_for_qualification or pass_index in set(map(int, schedule["validation_after_passes"]))
        if should_validate:
            raw_result = validate(
                distributed.module,
                validation_loader,
                perceptual,
                device=device,
                rank=rank,
                world=world,
                epoch=pass_index,
                steps=int(cfg["validation"]["fixed_descent_steps"]),
            )
            ema_result = validate(
                ema.module,
                validation_loader,
                perceptual,
                device=device,
                rank=rank,
                world=world,
                epoch=pass_index,
                steps=int(cfg["validation"]["fixed_descent_steps"]),
            )
            if rank == 0:
                validation = {"raw": raw_result, "ema": ema_result}
                write_json_atomic(
                    output_root / f"validation_pass_{pass_index:02d}.json", validation
                )
                print("VLM_ENERGY_REAL_VALIDATION " + json.dumps({
                    "pass": pass_index,
                    "raw": raw_result["overall"],
                    "ema": ema_result["overall"],
                }, sort_keys=True), flush=True)
        if rank == 0:
            digest = save_checkpoint(
                last_path,
                model=distributed.module,
                ema=ema,
                optimizer=optimizer,
                pass_index=pass_index,
                real_global_step=real_global_step,
                config_sha=config_sha,
                init_path=init_path,
                init_sha=init_sha,
                world=world,
                global_batch=global_batch,
                validation=validation,
            )
            write_json_atomic(output_root / f"pass_{pass_index:02d}.json", {
                "pass": pass_index,
                "completed_steps": completed_in_pass,
                "real_global_step": real_global_step,
                "mean_loss": pass_loss / max(1, completed_in_pass),
                "elapsed_pass_seconds": time.perf_counter() - pass_started,
                "checkpoint": last_path.relative_to(ROOT).as_posix(),
                "checkpoint_sha256": digest,
                "validation": validation,
                "qualification": qualification,
                "internal_test_accessed": False,
                "official_foundir_test_accessed": False,
                "sealed_internal_accessed": False,
            })
        dist.barrier()
        if stopped_for_qualification:
            break

    if rank == 0:
        status = (
            "foundir_vlm_energy_real_generalist_qualification_pass"
            if qualification
            else "foundir_vlm_energy_real_generalist_training_complete"
        )
        result = {
            "schema_version": 1,
            "status": status,
            "config_sha256": config_sha,
            "world_size": world,
            "global_batch": global_batch,
            "gpu_families": sorted(families),
            "heterogeneous_gpu_families": len(families) > 1,
            "nodes": host_counts,
            "completed_pass": pass_index,
            "real_global_step": real_global_step,
            "elapsed_seconds": time.perf_counter() - started,
            "initial_checkpoint": init_path.relative_to(ROOT).as_posix(),
            "initial_checkpoint_sha256": init_sha,
            "last_checkpoint": last_path.relative_to(ROOT).as_posix(),
            "last_checkpoint_sha256": sha256(last_path),
            "internal_test_accessed": False,
            "parent_test_accessed": False,
            "official_foundir_test_accessed": False,
            "sealed_internal_accessed": False,
        }
        write_json_atomic(output_root / "result.json", result)
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
