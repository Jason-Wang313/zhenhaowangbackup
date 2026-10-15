#!/usr/bin/env python3
"""Train the bounded double-only stage-trajectory compatibility tiny gate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image
from torch.nn.parallel import DistributedDataParallel


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lowlight_restoration.stage_trajectory_compatibility import (
    StageTrajectoryCompatibilityAdapter,
    parameter_count,
)


CATEGORIES = ("15Lowlight_Blur", "16Lowlight_Noise")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text())
    data, optimization, model = config["data"], config["optimization"], config["model"]
    manifest = resolve(data["manifest"])
    if (
        config.get("schema_version") != 1
        or config.get("status") != "user_authorized_stage_trajectory_compatibility_tiny_gate_v1"
        or not manifest.is_file()
        or sha256(manifest) != data.get("manifest_sha256")
        or data.get("train_categories") != list(CATEGORIES)
        or int(data.get("triple_records_in_gradients_or_selection", -1)) != 0
        or int(optimization.get("world_size", -1)) != 8
        or int(optimization.get("micro_batch_per_gpu", -1)) != 2
        or int(optimization.get("gradient_accumulation", -1)) != 2
        or int(optimization.get("global_batch", -1)) != 32
        or 8 * 2 * 2 != 32
        or optimization.get("milestones") != [250, 500, 1000]
        or int(optimization.get("total_steps", -1)) != 1000
        or model.get("identity_initialized") is not True
        or model.get("frozen_router") is not True
        or model.get("frozen_experts") is not True
        or config["selection"].get("triple_metrics_forbidden_before_selection_freeze") is not True
        or config["frozen_endpoint"].get("use_for_gradients_or_checkpoint_selection") is not False
        or config.get("official_foundir_test_accessed") is not False
    ):
        raise RuntimeError("frozen stage-trajectory config changed")
    return config


def read_rows(config: dict[str, Any]) -> dict[str, list[dict[str, str]]]:
    selected: dict[str, list[dict[str, str]]] = defaultdict(list)
    with resolve(config["data"]["manifest"]).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["split"] not in {"train", "validation"} or row["category"] not in CATEGORIES:
                raise RuntimeError("unexpected trajectory manifest row")
            selected[row["split"]].append(dict(row))
    if len(selected["train"]) != 89 or len(selected["validation"]) != 19:
        raise RuntimeError("trajectory split cardinality changed")
    train_groups = {row["source_group"] for row in selected["train"]}
    validation_groups = {row["source_group"] for row in selected["validation"]}
    if len(train_groups) != 51 or len(validation_groups) != 13 or train_groups & validation_groups:
        raise RuntimeError("trajectory source-group split changed")
    return dict(selected)


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8).copy()


def load_record(row: dict[str, str]) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    x, y = int(row["crop_x"]), int(row["crop_y"])
    lq_full = load_rgb(resolve(row["lq_path"]))
    gt_full = load_rgb(resolve(row["gt_path"]))
    lq = lq_full[y : y + 512, x : x + 512]
    gt = gt_full[y : y + 512, x : x + 512]
    if lq.shape != (512, 512, 3) or gt.shape != (512, 512, 3):
        raise RuntimeError(f"invalid frozen crop: {row['record_id']}")
    after_noise = load_rgb(resolve(row["after_noise_path"])) if row["after_noise_path"] else lq.copy()
    after_low_light = load_rgb(resolve(row["after_low_light_path"]))
    after_blur = load_rgb(resolve(row["after_blur_path"])) if row["after_blur_path"] else after_low_light.copy()
    states = [lq, after_noise, after_low_light, after_blur]
    if any(state.shape != (512, 512, 3) for state in states):
        raise RuntimeError(f"stage shape changed: {row['record_id']}")
    presence = np.asarray([float(row["presence_noise"]), float(row["presence_low_light"]), float(row["presence_blur"])], dtype=np.float32)
    return states, presence, gt


def tensor(array: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(array.copy()).permute(2, 0, 1).float().div_(255).to(device)


def psnr(reference: torch.Tensor, prediction: torch.Tensor) -> float:
    mse = float(F.mse_loss(prediction.float(), reference.float()).item())
    return math.inf if mse == 0 else -10.0 * math.log10(mse)


def validate(
    model: StageTrajectoryCompatibilityAdapter,
    rows: list[dict[str, str]],
    cache: dict[str, tuple[list[np.ndarray], np.ndarray, np.ndarray]],
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    records = []
    with torch.inference_mode():
        for row in rows:
            arrays, presence_array, gt_array = cache[row["record_id"]]
            states = [tensor(array, device).unsqueeze(0) for array in arrays]
            presence = torch.from_numpy(presence_array).unsqueeze(0).to(device)
            gt = tensor(gt_array, device).unsqueeze(0)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                output = model(*states, presence)
            records.append({
                "record_id": row["record_id"],
                "source_group": row["source_group"],
                "category": row["category"],
                "baseline_psnr": psnr(gt, states[-1]),
                "adapter_psnr": psnr(gt, output),
                "output_mean": float(output.mean().item()),
                "output_endpoint_fraction": float(((output <= 1 / 255) | (output >= 254 / 255)).float().mean().item()),
            })
    categories = {}
    for category in CATEGORIES:
        values = [item for item in records if item["category"] == category]
        categories[category] = {
            "records": len(values),
            "baseline_psnr": float(np.mean([item["baseline_psnr"] for item in values])),
            "adapter_psnr": float(np.mean([item["adapter_psnr"] for item in values])),
        }
        categories[category]["psnr_gain"] = categories[category]["adapter_psnr"] - categories[category]["baseline_psnr"]
    gains = [categories[category]["psnr_gain"] for category in CATEGORIES]
    return {
        "records": len(records),
        "categories": categories,
        "minimum_category_psnr_gain": min(gains),
        "overall_psnr_gain": float(np.mean([item["adapter_psnr"] - item["baseline_psnr"] for item in records])),
        "maximum_endpoint_fraction": max(item["output_endpoint_fraction"] for item in records),
        "per_record": records,
    }


def preflight(config_path: Path, *, write_evidence: Path | None = None) -> dict[str, Any]:
    config = load_config(config_path)
    rows = read_rows(config)
    for row in rows["train"] + rows["validation"]:
        if sha256(resolve(row["lq_path"])) != row["lq_sha256"] or sha256(resolve(row["gt_path"])) != row["gt_sha256"]:
            raise RuntimeError(f"source image hash changed: {row['record_id']}")
        load_record(row)
    model = StageTrajectoryCompatibilityAdapter(**{
        "channels": int(config["model"]["channels"]),
        "blocks": int(config["model"]["blocks"]),
        "maximum_residual": float(config["model"]["maximum_residual"]),
    })
    image = torch.rand(2, 3, 32, 32)
    presence = torch.tensor([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]])
    output = model(image, image, image, image, presence)
    if not torch.equal(output, image):
        raise RuntimeError("adapter is not exact identity at initialization")
    output.mean().backward()
    if model.head.weight.grad is None or not torch.isfinite(model.head.weight.grad).all():
        raise RuntimeError("adapter gradient rehearsal failed")
    payload = {
        "schema_version": 1,
        "status": "stage_trajectory_compatibility_cpu_preflight_pass",
        "config": config_path.relative_to(ROOT).as_posix(),
        "config_sha256": sha256(config_path),
        "manifest_sha256": config["data"]["manifest_sha256"],
        "records": {split: len(values) for split, values in rows.items()},
        "parameters": parameter_count(model),
        "identity_initialization_exact": True,
        "finite_gradient_rehearsal": True,
        "triple_records_in_gradients_or_selection": 0,
        "source_disjoint_endpoint": True,
        "development_pixels_accessed": True,
        "official_foundir_test_accessed": False,
        "checkpoint": "none_cpu_readiness",
        "checkpoint_sha256": "none",
    }
    if write_evidence is not None:
        if write_evidence.exists():
            raise FileExistsError(f"refusing to overwrite {write_evidence}")
        write_evidence.parent.mkdir(parents=True, exist_ok=True)
        write_evidence.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def learning_rate(config: dict[str, Any], step: int) -> float:
    optimization = config["optimization"]
    peak, floor = float(optimization["learning_rate"]), float(optimization["minimum_learning_rate"])
    warmup, total = int(optimization["warmup_steps"]), int(optimization["total_steps"])
    if step <= warmup:
        return peak * step / warmup
    progress = (step - warmup) / (total - warmup)
    return floor + 0.5 * (peak - floor) * (1.0 + math.cos(math.pi * progress))


def train(config_path: Path) -> None:
    config = load_config(config_path)
    rows = read_rows(config)
    rank, world, local_rank = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), int(os.environ["LOCAL_RANK"])
    if world != 8:
        raise RuntimeError("stage-trajectory gate requires exactly eight GPUs")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    name = torch.cuda.get_device_name(device)
    if not any(fragment in name for fragment in config["distributed"]["allowed_gpu_name_fragments"]):
        raise RuntimeError(f"unsupported GPU: {name}")
    dist.init_process_group("nccl", device_id=device)
    seed = int(config["optimization"]["seed"])
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed(seed + rank)
    random.seed(seed + rank)
    np.random.seed(seed + rank)

    bare = StageTrajectoryCompatibilityAdapter(
        channels=int(config["model"]["channels"]), blocks=int(config["model"]["blocks"]),
        maximum_residual=float(config["model"]["maximum_residual"]),
    ).to(device)
    model = DistributedDataParallel(bare, device_ids=[local_rank], output_device=local_rank, gradient_as_bucket_view=True)
    optimization = config["optimization"]
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(optimization["learning_rate"]), betas=tuple(optimization["betas"]),
        weight_decay=float(optimization["weight_decay"]),
    )
    output_dir = resolve(config["runtime"]["output_dir"])
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier()

    category_rows = {category: [row for row in rows["train"] if row["category"] == category] for category in CATEGORIES}
    cache = {
        row["record_id"]: load_record(row)
        for row in rows["train"] + rows["validation"]
    }
    if rank == 0:
        print(f"STAGE_TRAJECTORY_CACHE_READY records={len(cache)}", flush=True)
    micro = int(optimization["micro_batch_per_gpu"])
    accumulation = int(optimization["gradient_accumulation"])
    crop_size = int(config["data"]["crop_size"])
    milestones = set(map(int, optimization["milestones"]))
    config_sha = sha256(config_path)
    optimizer.zero_grad(set_to_none=True)
    for step in range(1, int(optimization["total_steps"]) + 1):
        model.train()
        step_loss = 0.0
        for accumulation_index in range(accumulation):
            batch_rows = []
            for local_index in range(micro):
                category = CATEGORIES[(step + accumulation_index + local_index + rank) % len(CATEGORIES)]
                pool = category_rows[category]
                batch_rows.append(pool[(step * world * micro + rank * micro + accumulation_index * micro + local_index) % len(pool)])
            state_batches = [[], [], [], []]
            presence_batch, target_batch = [], []
            for row in batch_rows:
                arrays, presence_array, target = cache[row["record_id"]]
                rng = random.Random(f"{seed}:{step}:{accumulation_index}:{rank}:{row['record_id']}")
                top, left = rng.randrange(513 - crop_size), rng.randrange(513 - crop_size)
                flip_h, flip_v, transpose = rng.randrange(2), rng.randrange(2), rng.randrange(2)
                augmented = [*arrays, target]
                for array_index, array in enumerate(augmented):
                    value = array[top : top + crop_size, left : left + crop_size]
                    if flip_h:
                        value = value[:, ::-1]
                    if flip_v:
                        value = value[::-1]
                    if transpose:
                        value = value.transpose(1, 0, 2)
                    if array_index < 4:
                        state_batches[array_index].append(value.copy())
                    else:
                        target_batch.append(value.copy())
                presence_batch.append(presence_array)
            states = [torch.stack([tensor(value, device) for value in values]) for values in state_batches]
            presence = torch.from_numpy(np.stack(presence_batch)).to(device)
            target = torch.stack([tensor(value, device) for value in target_batch])
            synchronize = accumulation_index == accumulation - 1
            context = torch.enable_grad() if synchronize else model.no_sync()
            with context:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    prediction = model(*states, presence)
                    difference = prediction.float() - target.float()
                    charbonnier = torch.sqrt(difference.square() + 1e-6).mean()
                    edge = (torch.abs(torch.diff(prediction.float(), dim=2) - torch.diff(target.float(), dim=2)).mean()
                            + torch.abs(torch.diff(prediction.float(), dim=3) - torch.diff(target.float(), dim=3)).mean()) / 2
                    loss = charbonnier + 0.1 * edge
                (loss / accumulation).backward()
                step_loss += float(loss.detach().item()) / accumulation
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(optimization["gradient_clip_norm"]))
        lr = learning_rate(config, step)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        loss_tensor = torch.tensor(step_loss, device=device)
        dist.all_reduce(loss_tensor)
        mean_loss = float(loss_tensor.item() / world)
        if not math.isfinite(mean_loss):
            raise RuntimeError("nonfinite trajectory-adapter loss")
        if rank == 0 and (step == 1 or step % 25 == 0):
            print(f"STAGE_TRAJECTORY_TRAIN step={step}/1000 loss={mean_loss:.8f} lr={lr:.8g}", flush=True)
        if step in milestones:
            dist.barrier()
            if rank == 0:
                checkpoint = output_dir / f"step_{step:06d}.pt"
                torch.save({"schema_version": 1, "step": step, "config_sha256": config_sha, "model": bare.state_dict()}, checkpoint)
                metrics = validate(bare, rows["validation"], cache, device)
                metrics.update({
                    "schema_version": 1, "status": "stage_trajectory_double_validation_complete", "step": step,
                    "checkpoint": checkpoint.relative_to(ROOT).as_posix(), "checkpoint_sha256": sha256(checkpoint),
                    "config_sha256": config_sha, "triple_metrics_accessed": False,
                    "official_foundir_test_accessed": False,
                })
                (output_dir / f"validation_step_{step:06d}.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
                print(f"STAGE_TRAJECTORY_VALIDATION step={step} min_gain={metrics['minimum_category_psnr_gain']:.6f} overall_gain={metrics['overall_psnr_gain']:.6f}", flush=True)
            dist.barrier()

    if rank == 0:
        candidates = [json.loads((output_dir / f"validation_step_{step:06d}.json").read_text()) for step in sorted(milestones)]
        selected = max(candidates, key=lambda item: (item["minimum_category_psnr_gain"], item["overall_psnr_gain"], -item["step"]))
        promoted = all(item["psnr_gain"] > 0.0 for item in selected["categories"].values())
        selection = {
            "schema_version": 1, "status": "stage_trajectory_checkpoint_selection_frozen",
            "selection_rule": config["selection"]["rule"], "selected_step": selected["step"],
            "selected_checkpoint": selected["checkpoint"], "selected_checkpoint_sha256": selected["checkpoint_sha256"],
            "validation": selected, "endpoint_promotion_pass": promoted,
            "triple_metrics_accessed": False, "official_foundir_test_accessed": False,
        }
        (output_dir / "development_selection.json").write_text(json.dumps(selection, indent=2, sort_keys=True) + "\n")
        print(f"STAGE_TRAJECTORY_TRAINING_END selected={selected['step']} promotion={promoted}", flush=True)
    dist.barrier()
    dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/foundir_stage_trajectory_compatibility_v1.json")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--write-evidence")
    args = parser.parse_args()
    config_path = resolve(args.config)
    if args.check:
        evidence = resolve(args.write_evidence) if args.write_evidence else None
        print(json.dumps(preflight(config_path, write_evidence=evidence), indent=2, sort_keys=True))
    else:
        train(config_path)


if __name__ == "__main__":
    main()
