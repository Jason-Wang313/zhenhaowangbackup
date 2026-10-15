#!/usr/bin/env python3
"""Train frozen-proposal energy and matched direct control on Noise development data."""

from __future__ import annotations

import hashlib
import csv
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lowlight_restoration.flux_prompt_cache import file_sha256  # noqa: E402
from lowlight_restoration.proposal_relative_energy import (  # noqa: E402
    EnergyAndDirectControls,
    parameter_counts,
)
from lowlight_restoration.task_native_restormer import load_task_native_restormer  # noqa: E402

CONFIG_PATH = ROOT / "configs/foundir_proposal_relative_energy_noise_v1.json"


def stable_int(*parts: object) -> int:
    return int.from_bytes(hashlib.sha256(":".join(map(str, parts)).encode()).digest()[:8], "big")


def load_task_rows(config: dict) -> dict[str, list[dict[str, str]]]:
    selected: dict[str, list[dict[str, str]]] = defaultdict(list)
    total = 0
    with (ROOT / config["data"]["manifest"]).open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"record_id", "split", "source_group", "category", "lq_path", "gt_path", "lq_sha256", "gt_sha256"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise RuntimeError("Noise manifest schema changed")
        for row in reader:
            total += 1
            if row["category"] == config["data"]["category"]:
                selected[row["split"]].append(dict(row))
    if total != 685495:
        raise RuntimeError("Noise source manifest cardinality changed")
    expected = {
        "train": (int(config["data"]["train_records"]), int(config["data"]["train_source_groups"])),
        "validation": (int(config["data"]["validation_records"]), int(config["data"]["validation_source_groups"])),
        "internal_test": (1571, 32),
    }
    if set(selected) != set(expected):
        raise RuntimeError("Noise split set changed")
    owners: dict[str, str] = {}
    identifiers: set[str] = set()
    for split, (record_count, group_count) in expected.items():
        rows = selected[split]
        if len(rows) != record_count or len({row["source_group"] for row in rows}) != group_count:
            raise RuntimeError(f"Noise {split} count changed")
        for row in rows:
            if row["record_id"] in identifiers:
                raise RuntimeError("duplicate Noise record ID")
            identifiers.add(row["record_id"])
            previous = owners.setdefault(row["source_group"], split)
            if previous != split:
                raise RuntimeError("Noise source group crosses splits")
    return dict(selected)


def prepare_aligned_pair(lq: np.ndarray, gt: np.ndarray, crop: int, record_id: str) -> tuple[np.ndarray, np.ndarray]:
    if lq.shape != gt.shape or lq.ndim != 3 or lq.shape[2] != 3:
        raise RuntimeError(f"unaligned/non-RGB Noise pair: {record_id}")
    height, width = gt.shape[:2]
    if min(height, width) >= crop:
        return lq, gt
    scale = crop / min(height, width)
    size = (max(crop, round(width * scale)), max(crop, round(height * scale)))
    return (
        np.asarray(Image.fromarray(lq).resize(size, Image.Resampling.BICUBIC), dtype=np.uint8).copy(),
        np.asarray(Image.fromarray(gt).resize(size, Image.Resampling.BICUBIC), dtype=np.uint8).copy(),
    )


def load_pair_arrays(row: dict[str, str], verified: set[str] | None = None) -> tuple[np.ndarray, np.ndarray]:
    arrays = []
    for path_key, hash_key in (("lq_path", "lq_sha256"), ("gt_path", "gt_sha256")):
        path = ROOT / row[path_key]
        identity = path.as_posix()
        if verified is None or identity not in verified:
            if file_sha256(path) != row[hash_key]:
                raise RuntimeError(f"Noise pair hash changed: {row['record_id']}")
            if verified is not None:
                verified.add(identity)
        with Image.open(path) as handle:
            arrays.append(np.asarray(handle.convert("RGB"), dtype=np.uint8).copy())
    return arrays[0], arrays[1]


def load_training_patch(
    row: dict[str, str], *, crop_size: int, augmentation_seed: int, verified: set[str]
) -> tuple[torch.Tensor, torch.Tensor]:
    lq, gt = load_pair_arrays(row, verified)
    lq, gt = prepare_aligned_pair(lq, gt, crop_size, row["record_id"])
    generator = random.Random(augmentation_seed)
    y = generator.randrange(lq.shape[0] - crop_size + 1)
    x = generator.randrange(lq.shape[1] - crop_size + 1)
    lq, gt = lq[y:y + crop_size, x:x + crop_size], gt[y:y + crop_size, x:x + crop_size]
    rotation = generator.randrange(4)
    if rotation:
        lq, gt = np.rot90(lq, rotation), np.rot90(gt, rotation)
    if generator.randrange(2):
        lq, gt = np.flip(lq, 1), np.flip(gt, 1)
    def tensor(value: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.ascontiguousarray(value)).permute(2, 0, 1).float().div_(255)
    return tensor(lq), tensor(gt)


def sample_global_batch(
    grouped: dict[str, list[dict[str, str]]], *, task: str, step: int, seed: int, count: int
) -> list[tuple[dict[str, str], int]]:
    groups = sorted(grouped)
    generator = random.Random(stable_int(seed, task, "train-step", step))
    result = []
    for position in range(count):
        group = groups[generator.randrange(len(groups))]
        row = grouped[group][generator.randrange(len(grouped[group]))]
        result.append((row, stable_int(seed, task, step, position, row["record_id"])))
    return result


def load_audit_crop(
    row: dict[str, str], *, crop_size: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor, list[int], bool]:
    lq, gt = load_pair_arrays(row)
    original = gt.shape
    lq, gt = prepare_aligned_pair(lq, gt, crop_size, row["record_id"])
    digest = hashlib.sha256(f"{seed}:evaluation_crop:{row['record_id']}".encode()).digest()
    y = int.from_bytes(digest[:8], "big") % (gt.shape[0] - crop_size + 1)
    x = int.from_bytes(digest[8:16], "big") % (gt.shape[1] - crop_size + 1)
    def tensor(value: np.ndarray) -> torch.Tensor:
        value = value[y:y + crop_size, x:x + crop_size]
        return torch.from_numpy(value.copy()).permute(2, 0, 1).float().div_(127.5).sub_(1)
    return tensor(lq), tensor(gt), [x, y], gt.shape != original


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def edge_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    px = prediction[..., :, 1:] - prediction[..., :, :-1]
    tx = target[..., :, 1:] - target[..., :, :-1]
    py = prediction[..., 1:, :] - prediction[..., :-1, :]
    ty = target[..., 1:, :] - target[..., :-1, :]
    return F.l1_loss(px, tx) + F.l1_loss(py, ty)


def crop_for_step(config: dict, step: int) -> int:
    selected = 0
    for start, size in config["schedule"]["progressive_crops"]:
        if step > int(start):
            selected = int(size)
    if not selected:
        raise RuntimeError("no progressive crop selected")
    return selected


def learning_rate(config: dict, step: int, total: int) -> float:
    peak = float(config["schedule"]["learning_rate"])
    minimum = float(config["schedule"]["minimum_learning_rate"])
    warmup = int(config["schedule"]["warmup_steps"])
    if step <= warmup:
        return peak * step / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return minimum + 0.5 * (peak - minimum) * (1 + math.cos(math.pi * progress))


def load_frozen_proposal(config: dict, device: torch.device) -> torch.nn.Module:
    spec = config["frozen_proposal"]
    checkpoint = ROOT / spec["checkpoint"]
    if file_sha256(checkpoint) != spec["checkpoint_sha256"]:
        raise RuntimeError("frozen Noise proposal checkpoint hash changed")
    model, _ = load_task_native_restormer(
        spec["base_checkpoint_name"], ROOT / spec["base_checkpoint_dir"]
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if (
        payload.get("status") != "foundir_task_native_adaptation_checkpoint"
        or payload.get("task") != "noise_mse_lpips"
        or int(payload.get("step", -1)) != 250
        or payload.get("sealed_internal_test_accessed") is not False
        or payload.get("protected_foundir_test_accessed") is not False
    ):
        raise RuntimeError("frozen Noise proposal checkpoint contract changed")
    model.load_state_dict(payload["model"], strict=True)
    model.requires_grad_(False).eval().to(device)
    return model


def make_proposal(model: torch.nn.Module, observation_01: torch.Tensor) -> torch.Tensor:
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        proposal = model(observation_01).clamp(0, 1)
    return proposal.float().mul(2).sub(1).detach()


def fixed_validation_rows(rows: list[dict[str, str]], count: int, seed: int) -> list[dict[str, str]]:
    ordered = sorted(
        rows,
        key=lambda row: hashlib.sha256(
            f"{seed}:{row['source_group']}:{row['record_id']}".encode()
        ).hexdigest(),
    )
    # Round-robin source groups prevents a large group from dominating the subset.
    selected, seen = [], set()
    for row in ordered:
        if row["source_group"] not in seen:
            selected.append(row)
            seen.add(row["source_group"])
            if len(selected) == count:
                return selected
    for row in ordered:
        if row not in selected:
            selected.append(row)
            if len(selected) == count:
                break
    return selected


def validate(
    controls: EnergyAndDirectControls,
    proposal_model: torch.nn.Module,
    rows: list[dict[str, str]],
    config: dict,
    device: torch.device,
    rank: int,
    world: int,
) -> dict | None:
    controls.eval()
    metric = LearnedPerceptualImagePatchSimilarity(net_type="alex").to(device).eval()
    names = ("proposal", "direct", "energy_1", "energy_2", "energy_3")
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
        descriptor = torch.tensor([[0, 1, 0, 0, 0]], device=device, dtype=torch.float32)
        proposal = make_proposal(proposal_model, observation_01)
        with torch.no_grad():
            direct = controls.direct_model(observation, proposal, descriptor)
        candidates = [proposal, direct]
        for steps in (1, 2, 3):
            with torch.enable_grad():
                output = controls.energy_model.refine(
                    observation,
                    proposal.detach(),
                    descriptor,
                    steps=steps,
                    create_graph=False,
                    detach_between_steps=True,
                ).output.detach()
            candidates.append(output)
        for arm, candidate in enumerate(candidates):
            mse = (candidate.add(1).mul(0.5) - target.add(1).mul(0.5)).square().mean()
            psnr = -10 * torch.log10(mse.clamp_min(1e-12))
            with torch.no_grad():
                lpips = metric(candidate, target)
            sums[arm, 0] += psnr.double()
            sums[arm, 1] += lpips.double()
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
    model: EnergyAndDirectControls,
    optimizer: torch.optim.Optimizer,
    *,
    step: int,
    config_sha: str,
    validation: dict | None,
    hardware: dict,
    elapsed_seconds: float,
) -> None:
    payload = {
        "schema_version": 1,
        "status": "proposal_relative_energy_noise_checkpoint",
        "config_sha256": config_sha,
        "step": step,
        "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
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
    if world != 8 or int(config["hardware"]["world_size"]) != 8:
        raise RuntimeError("this experiment requires exactly eight local GPUs")
    names = [None] * world
    dist.all_gather_object(names, torch.cuda.get_device_name(local_rank))
    families = {
        family
        for name in names
        for family in config["hardware"]["supported_families"]
        if family in str(name).upper()
    }
    if len(families) != 1 or any(not any(f in str(n).upper() for f in families) for n in names):
        raise RuntimeError(f"requires one homogeneous 8xH100 or 8xH200 node, got {names}")
    family = next(iter(families))
    profile = config["hardware"]["profiles"][family]
    micro = int(profile["micro_batch_per_gpu"])
    accumulation = int(profile["gradient_accumulation"])
    global_batch = micro * accumulation * world
    if global_batch != int(config["hardware"]["global_batch"]):
        raise RuntimeError("hardware profile changed the frozen global batch")

    # Local loader checks exact split counts and source-group disjointness
    # without importing the legacy OpenCV-based evaluation stack.
    rows = load_task_rows(config)
    if file_sha256(ROOT / config["data"]["manifest"]) != config["data"]["manifest_sha256"]:
        raise RuntimeError("Noise manifest hash changed")
    if len(rows["train"]) != int(config["data"]["train_records"]):
        raise RuntimeError("Noise train count changed")
    validation_rows = fixed_validation_rows(
        rows["validation"],
        int(config["data"]["validation_subset_records"]),
        int(config["data"]["validation_crop_seed"]),
    )
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows["train"]:
        grouped.setdefault(row["source_group"], []).append(row)

    proposal_model = load_frozen_proposal(config, device)
    controls = EnergyAndDirectControls().to(device)
    counts = parameter_counts(controls)
    ratio = max(counts.values()) / min(counts.values())
    if ratio > 1.25:
        raise RuntimeError(f"direct/energy controls are not parameter matched: {counts}")
    ddp = DDP(controls, device_ids=[local_rank], gradient_as_bucket_view=False)
    optimizer = torch.optim.AdamW(
        ddp.parameters(),
        lr=float(config["schedule"]["learning_rate"]),
        betas=(0.9, 0.999),
        weight_decay=float(config["schedule"]["weight_decay"]),
    )
    run_id = os.environ.get("PROPOSAL_ENERGY_RUN_ID", "qualification_v1")
    total = int(os.environ.get("PROPOSAL_ENERGY_TOTAL_STEPS", "500"))
    allowed = {int(v) for v in config["schedule"]["milestones"]}
    if total not in allowed:
        raise RuntimeError(f"total steps must be a frozen milestone, got {total}")
    output = ROOT / config["runtime"]["output_root"] / run_id
    output.mkdir(parents=True, exist_ok=True)
    hardware = {
        "gpu_family": family,
        "gpu_names": names,
        "world_size": world,
        "micro_batch_per_gpu": micro,
        "gradient_accumulation": accumulation,
        "global_batch": global_batch,
    }
    if rank == 0:
        atomic_json(output / "launch.json", {
            "status": "proposal_relative_energy_noise_started",
            "run_id": run_id,
            "total_steps": total,
            "config_sha256": config_sha,
            "parameter_counts": counts,
            "hardware": hardware,
            "internal_test_accessed": False,
            "official_foundir_test_accessed": False,
            "sealed_internal_accessed": False,
            "frozen_660_accessed": False,
        })

    start_step = 0
    checkpoints = sorted(output.glob("step_*.pt"))
    if checkpoints:
        payload = torch.load(checkpoints[-1], map_location="cpu", weights_only=False)
        if payload.get("config_sha256") != config_sha:
            raise RuntimeError("resume config identity changed")
        controls.load_state_dict(payload["model"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(device)
        start_step = int(payload["step"])

    verified: set[str] = set()
    weights = config["loss"]
    started = time.monotonic()
    ddp.train()
    for step in range(start_step + 1, total + 1):
        lr = learning_rate(config, step, total)
        for group in optimizer.param_groups:
            group["lr"] = lr
        selected = sample_global_batch(
            grouped,
            task="proposal_relative_energy_noise_v1",
            step=step,
            seed=int(config["schedule"]["seed"]),
            count=global_batch,
        )
        local = selected[rank * (global_batch // world):(rank + 1) * (global_batch // world)]
        optimizer.zero_grad(set_to_none=True)
        reduced = torch.zeros(5, device=device)
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
            target_01 = torch.stack([pair[1] for pair in pairs]).to(device)
            observation = observation_01.mul(2).sub(1)
            target = target_01.mul(2).sub(1)
            descriptor = torch.tensor([0, 1, 0, 0, 0], device=device).float()[None].repeat(micro, 1)
            proposal = make_proposal(proposal_model, observation_01)
            result = ddp(observation, proposal, target, descriptor)
            energy_recon = F.l1_loss(result["refined"], target) + float(weights["edge"]) * edge_loss(result["refined"], target)
            direct_recon = F.l1_loss(result["direct"], target) + float(weights["edge"]) * edge_loss(result["direct"], target)
            loss = (
                float(weights["energy_reconstruction"]) * energy_recon
                + float(weights["energy_direction"]) * result["direction"]
                + float(weights["energy_ranking"]) * result["ranking"]
                + float(weights["direct_reconstruction"]) * direct_recon
            ) / accumulation
            loss.backward()
            reduced += torch.stack((loss.detach() * accumulation, energy_recon.detach(), direct_recon.detach(), result["direction"].detach(), result["ranking"].detach())) / accumulation
        torch.nn.utils.clip_grad_norm_(ddp.parameters(), float(config["schedule"]["gradient_clip_norm"]))
        optimizer.step()
        if not bool(torch.isfinite(reduced).all()):
            raise RuntimeError("non-finite proposal-relative energy objective")
        if step % 25 == 0:
            dist.all_reduce(reduced)
            if rank == 0:
                values = (reduced / world).tolist()
                print(json.dumps({"step": step, "lr": lr, "loss": values[0], "energy_reconstruction": values[1], "direct_reconstruction": values[2], "direction": values[3], "ranking": values[4], "elapsed_seconds": time.monotonic() - started}), flush=True)

        if step in allowed and (step == total or step % int(config["schedule"]["steps_per_effective_pass"]) == 0):
            dist.barrier()
            validation = validate(controls, proposal_model, validation_rows, config, device, rank, world)
            dist.barrier()
            if rank == 0:
                elapsed = time.monotonic() - started
                save_checkpoint(
                    output / f"step_{step:06d}.pt",
                    controls,
                    optimizer,
                    step=step,
                    config_sha=config_sha,
                    validation=validation,
                    hardware=hardware,
                    elapsed_seconds=elapsed,
                )
                seconds_per_step = elapsed / max(1, step - start_step)
                atomic_json(output / f"validation_step_{step:06d}.json", {
                    "status": "proposal_relative_energy_noise_development_validation",
                    "step": step,
                    "validation": validation,
                    "seconds_per_optimizer_step": seconds_per_step,
                    "projected_primary_wall_hours": seconds_per_step * int(config["schedule"]["primary_steps"]) / 3600,
                    "projected_primary_gpu_hours": seconds_per_step * int(config["schedule"]["primary_steps"]) / 3600 * 8,
                    "hardware": hardware,
                    "internal_test_accessed": False,
                    "official_foundir_test_accessed": False,
                    "sealed_internal_accessed": False,
                    "frozen_660_accessed": False,
                })
            dist.barrier()
    if rank == 0:
        atomic_json(output / "terminal_result.json", {
            "status": "proposal_relative_energy_noise_training_complete",
            "completed_optimizer_steps": total,
            "hardware": hardware,
            "wall_clock_hours": (time.monotonic() - started) / 3600,
            "allocated_gpu_hours": (time.monotonic() - started) / 3600 * 8,
            "internal_test_accessed": False,
            "official_foundir_test_accessed": False,
            "sealed_internal_accessed": False,
            "frozen_660_accessed": False,
        })
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
