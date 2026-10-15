#!/usr/bin/env python3
"""One bounded, category-balanced Restormer pilot for the FoundIR Noise family."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import Counter, defaultdict
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image
from torch.nn.parallel import DistributedDataParallel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lowlight_restoration.flux_prompt_cache import file_sha256  # noqa: E402
from lowlight_restoration.foundir_flux2_protocol import resolve_workspace_path, stable_digest  # noqa: E402
from lowlight_restoration.task_native_restormer import load_task_native_restormer  # noqa: E402
from smoke.evaluate_foundir_five_type_flux2_ddp import lpips_value, write_once  # noqa: E402
from smoke.evaluate_foundir_mixed_restoration_pilot_ddp import psnr  # noqa: E402
from smoke.evaluate_foundir_noise_rain_full_development_ddp import source_group_bootstrap  # noqa: E402
from smoke.train_foundir_task_native_restormer_adaptation_ddp import (  # noqa: E402
    crop_size_for_step,
    learning_rate,
    load_training_patch,
    move_optimizer_state,
    raise_if_any_rank_failed,
    sample_global_batch,
)


CATEGORIES = ("05Noise", "16Lowlight_Noise")
FORBIDDEN_FOUNDIR_BASELINE_SHAS = {
    "2603a0be47d0844d4c9cea542ecca039e0e4aaa39d45d41064bfbdc13c0830c9",
    "f834248e145d020d5a02f257b7ceb383b579add2522932938b3e1e0bd460abe6",
    "25fa90003de8fa6965d861fc66159fb0c188654723acc15e8c49ec8359ca8b93",
}


def validate_config(path: Path) -> dict:
    config = json.loads(path.read_text())
    data = config.get("data", {})
    optimization = config.get("optimization", {})
    gates = config.get("per_category_gates", {})
    expected_gates = {
        "05Noise": {"mean_psnr_minimum": 39.9, "mean_lpips_maximum": 0.08},
        "16Lowlight_Noise": {
            "mean_psnr_minimum": 18.733316631721234,
            "mean_lpips_maximum": 0.2582512099761516,
        },
    }
    required_hashes = (
        ("manifest", "manifest_sha256"),
        ("base_checkpoint", "base_checkpoint_sha256"),
        ("initial_adaptation_checkpoint", "initial_adaptation_checkpoint_sha256"),
        ("initial_adaptation_result", "initial_adaptation_result_sha256"),
        ("failed_mixed_compatibility_result", "failed_mixed_compatibility_result_sha256"),
    )
    if (
        config.get("schema_version") != 1
        or config.get("status") != "user_authorized_noise_family_training_only_pilot"
        or config.get("task") != "noise_family"
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
        or float(optimization.get("identity_mix_probability", -1)) != 0.0
        or optimization.get("precision") != "bfloat16_autocast_fp32_parameters"
        or int(optimization.get("seed", -1)) != 2026091912
        or gates != expected_gates
        or config.get("selection_rule")
        != "maximize_worst_per_category_normalized_gate_margin_then_earlier_step"
        or config.get("one_bounded_pilot_no_automatic_tuning") is not True
        or config.get("development_pixels_accessed") is not False
        or config.get("official_foundir_test_accessed") is not False
    ):
        raise RuntimeError("Noise-family pilot contract changed")
    if int(optimization["global_batch"]) != 8 * 4 or int(optimization["global_batch"]) % 2:
        raise RuntimeError("Noise-family batch arithmetic changed")
    for path_key, hash_key in required_hashes:
        artifact = resolve_workspace_path(ROOT, config[path_key])
        if not artifact.is_file() or file_sha256(artifact) != config[hash_key]:
            raise RuntimeError(f"Noise-family frozen artifact changed: {path_key}")
        if config[hash_key] in FORBIDDEN_FOUNDIR_BASELINE_SHAS:
            raise RuntimeError("refusing released FoundIR baseline/specialist weights")
    initial = torch.load(
        resolve_workspace_path(ROOT, config["initial_adaptation_checkpoint"]),
        map_location="cpu",
        weights_only=False,
    )
    initial_result = json.loads(
        resolve_workspace_path(ROOT, config["initial_adaptation_result"]).read_text()
    )
    failure = json.loads(
        resolve_workspace_path(ROOT, config["failed_mixed_compatibility_result"]).read_text()
    )
    if (
        initial.get("status") != "foundir_task_native_adaptation_checkpoint"
        or initial.get("task") != "noise"
        or initial.get("step") != 250
        or not isinstance(initial.get("model"), Mapping)
        or initial.get("sealed_internal_test_accessed") is not False
        or initial.get("protected_foundir_test_accessed") is not False
        or initial_result.get("strong_gate_pass") is not True
        or initial_result.get("task") != "noise"
        or initial_result.get("step") != 250
        or failure.get("status") != "noise_route_cross_compatibility_complete"
        or failure.get("gate_pass") is not False
        or failure.get("official_foundir_test_accessed") is not False
    ):
        raise RuntimeError("Noise-family prerequisite evidence changed")
    return config


def load_rows(config: Mapping[str, object]) -> dict[str, list[dict[str, str]]]:
    selected: dict[str, list[dict[str, str]]] = defaultdict(list)
    manifest = resolve_workspace_path(ROOT, config["manifest"])
    with manifest.open(newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    if len(rows) != int(config["manifest_expected_records"]):
        raise RuntimeError("Noise-family manifest cardinality changed")
    counts = Counter((row["split"], row["category"]) for row in rows)
    expected = {
        ("train", category): int(config["data"]["train_records_per_category"])
        for category in CATEGORIES
    } | {
        ("validation", category): int(config["data"]["validation_records_per_category"])
        for category in CATEGORIES
    }
    if counts != Counter(expected) or len({row["record_id"] for row in rows}) != len(rows):
        raise RuntimeError("Noise-family split/category coverage changed")
    groups_by_split: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        selected[row["split"]].append(row)
        groups_by_split[row["split"]].add(row["source_group"])
    if (
        len(groups_by_split["train"]) != int(config["data"]["train_source_groups"])
        or len(groups_by_split["validation"])
        != int(config["data"]["validation_source_groups"])
        or groups_by_split["train"] & groups_by_split["validation"]
    ):
        raise RuntimeError("Noise-family train/validation sources changed")
    for split in ("train", "validation"):
        category_groups = {
            category: {row["source_group"] for row in selected[split] if row["category"] == category}
            for category in CATEGORIES
        }
        if category_groups[CATEGORIES[0]] != category_groups[CATEGORIES[1]]:
            raise RuntimeError(f"Noise-family {split} categories are no longer source paired")
    return dict(selected)


def category_balanced_batch(
    grouped: Mapping[str, Mapping[str, Sequence[Mapping[str, str]]]],
    *,
    step: int,
    seed: int,
    count: int,
) -> list[tuple[Mapping[str, str], int]]:
    if count % len(CATEGORIES):
        raise RuntimeError("Noise-family batch cannot be category balanced")
    per_category = count // len(CATEGORIES)
    parts = [
        sample_global_batch(
            grouped[category],
            task=f"noise_family:{category}",
            step=step,
            seed=seed,
            count=per_category,
        )
        for category in CATEGORIES
    ]
    # Contiguous category halves plus rank-strided sharding give every one of
    # eight ranks two examples from each category in a global batch of 32.
    return [assignment for part in parts for assignment in part]


def load_validation_crop(
    row: Mapping[str, str], *, crop_size: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    arrays = []
    for key, hash_key in (("lq_path", "lq_sha256"), ("gt_path", "gt_sha256")):
        path = resolve_workspace_path(ROOT, row[key])
        if file_sha256(path) != row[hash_key]:
            raise RuntimeError(f"Noise-family validation pair changed: {row['record_id']}")
        with Image.open(path) as handle:
            arrays.append(np.asarray(handle.convert("RGB"), dtype=np.uint8).copy())
    if arrays[0].shape != arrays[1].shape or min(arrays[0].shape[:2]) < crop_size:
        raise RuntimeError(f"invalid Noise-family validation pair: {row['record_id']}")
    digest = bytes.fromhex(stable_digest(seed, "mixed_restoration_crop", row["record_id"]))
    height, width = arrays[0].shape[:2]
    y = int.from_bytes(digest[:8], "big") % (height - crop_size + 1)
    x = int.from_bytes(digest[8:16], "big") % (width - crop_size + 1)

    def tensor(array: np.ndarray) -> torch.Tensor:
        crop = array[y : y + crop_size, x : x + crop_size]
        return torch.from_numpy(crop.copy()).permute(2, 0, 1).float().div_(127.5).sub_(1)

    return tensor(arrays[0]), tensor(arrays[1])


def save_checkpoint(
    path: Path,
    *,
    config_sha: str,
    step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    initial_sha: str,
) -> None:
    payload = {
        "schema_version": 1,
        "status": "foundir_noise_family_restormer_checkpoint",
        "config_sha256": config_sha,
        "task": "noise_family",
        "step": step,
        "initial_adaptation_checkpoint_sha256": initial_sha,
        "model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "optimizer": optimizer.state_dict(),
        "development_pixels_accessed": False,
        "official_foundir_test_accessed": False,
    }
    if path.exists():
        existing = torch.load(path, map_location="cpu", weights_only=False)
        if existing.get("config_sha256") != config_sha or existing.get("step") != step:
            raise RuntimeError(f"existing Noise-family checkpoint changed: {path}")
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def resume_checkpoint(
    output: Path,
    *,
    config_sha: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> int:
    if list(output.glob("step_*.pt.tmp")):
        raise RuntimeError("orphan Noise-family checkpoint temporary exists")
    paths = sorted(output.glob("step_*.pt"))
    if not paths:
        return 0
    path = paths[-1]
    payload = torch.load(path, map_location="cpu", weights_only=False)
    step = int(path.stem.removeprefix("step_"))
    if (
        payload.get("status") != "foundir_noise_family_restormer_checkpoint"
        or payload.get("config_sha256") != config_sha
        or payload.get("task") != "noise_family"
        or payload.get("step") != step
        or step not in {500, 1000}
        or payload.get("development_pixels_accessed") is not False
        or payload.get("official_foundir_test_accessed") is not False
    ):
        raise RuntimeError("invalid Noise-family resume checkpoint")
    model.load_state_dict(payload["model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    move_optimizer_state(optimizer, device)
    return step


def validation_payload(
    records: Sequence[Mapping[str, object]],
    *,
    config: Mapping[str, object],
    config_sha: str,
    checkpoint: Path,
    step: int,
) -> dict[str, object]:
    per_category = {}
    checks = {}
    margins = []
    for category in CATEGORIES:
        subset = [row for row in records if row["category"] == category]
        expected = int(config["data"]["validation_records_per_category"])
        if len(subset) != expected:
            raise RuntimeError(f"Noise-family validation coverage changed for {category}")
        metrics = {
            "records": len(subset),
            "source_groups": len({row["source_group"] for row in subset}),
            "input_psnr": float(np.mean([row["input"]["psnr"] for row in subset])),
            "input_lpips": float(np.mean([row["input"]["lpips"] for row in subset])),
            "psnr": float(np.mean([row["restored"]["psnr"] for row in subset])),
            "lpips": float(np.mean([row["restored"]["lpips"] for row in subset])),
        }
        bootstrap = source_group_bootstrap(
            subset,
            resamples=int(config["bootstrap"]["resamples"]),
            confidence=float(config["bootstrap"]["confidence"]),
            seed=int(stable_digest(config["bootstrap"]["seed"], category, step)[:16], 16),
        )
        gate = config["per_category_gates"][category]
        category_checks = {
            "mean_psnr": metrics["psnr"] >= float(gate["mean_psnr_minimum"]),
            "mean_lpips": metrics["lpips"] <= float(gate["mean_lpips_maximum"]),
        }
        margins.extend(
            [
                metrics["psnr"] - float(gate["mean_psnr_minimum"]),
                (float(gate["mean_lpips_maximum"]) - metrics["lpips"]) / 0.02,
            ]
        )
        per_category[category] = {
            "metrics": metrics,
            "source_group_bootstrap": bootstrap,
            "gate": gate,
            "checks": category_checks,
            "gate_pass": all(category_checks.values()),
        }
        checks[category] = all(category_checks.values())
    return {
        "schema_version": 1,
        "status": "foundir_noise_family_restormer_development_milestone",
        "config_sha256": config_sha,
        "task": "noise_family",
        "step": step,
        "checkpoint": checkpoint.relative_to(ROOT).as_posix(),
        "checkpoint_sha256": file_sha256(checkpoint),
        "per_category": per_category,
        "per_category_gate_pass": checks,
        "joint_gate_pass": all(checks.values()),
        "selection_score": min(margins),
        "development_pixels_accessed": False,
        "official_foundir_test_accessed": False,
    }


def evaluate(
    model: torch.nn.Module,
    rows: Sequence[Mapping[str, str]],
    metric,
    *,
    config: Mapping[str, object],
    config_sha: str,
    checkpoint: Path,
    step: int,
    rank: int,
    world: int,
    device: torch.device,
) -> None:
    model.eval()
    local = []
    local_error = None
    try:
        for index in range(rank, len(rows), world):
            row = rows[index]
            lq, gt = load_validation_crop(
                row,
                crop_size=int(config["data"]["validation_crop_size"]),
                seed=int(config["data"]["validation_crop_seed"]),
            )
            source = ((lq + 1) * 0.5).unsqueeze(0).to(device)
            target = gt.unsqueeze(0).to(device)
            with torch.inference_mode():
                restored = model(source).float().clamp(0, 1)
            array = restored[0].cpu().permute(1, 2, 0).mul(255).round().byte().numpy()
            restored_m11 = (
                torch.from_numpy(array.copy()).permute(2, 0, 1).float().div_(127.5).sub_(1).unsqueeze(0)
            )
            input_batch = lq.unsqueeze(0).to(device)
            input_lpips = lpips_value(metric, target, input_batch, device)
            metric.reset()
            restored_lpips = lpips_value(metric, target, restored_m11, device)
            metric.reset()
            local.append(
                {
                    "record_id": row["record_id"],
                    "source_group": row["source_group"],
                    "category": row["category"],
                    "input": {
                        "psnr": psnr(target, input_batch),
                        "lpips": input_lpips,
                    },
                    "restored": {
                        "psnr": psnr(target, restored_m11.to(device)),
                        "lpips": restored_lpips,
                    },
                }
            )
    except Exception as error:
        local_error = f"{type(error).__name__}: {error}"
    raise_if_any_rank_failed(
        local_error, device=device, world=world, operation=f"Noise-family validation step {step}"
    )
    gathered: list[object | None] = [None] * world
    dist.all_gather_object(gathered, local)
    if rank == 0:
        complete = sorted(
            [item for shard in gathered for item in shard], key=lambda item: item["record_id"]
        )
        payload = validation_payload(
            complete,
            config=config,
            config_sha=config_sha,
            checkpoint=checkpoint,
            step=step,
        )
        path = checkpoint.parent / f"development_step_{step:06d}.json"
        write_once(path, (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode())
        print("FOUNDIR_NOISE_FAMILY_MILESTONE " + json.dumps(payload, sort_keys=True), flush=True)
    model.train()


def finalize(config: Mapping[str, object], config_sha: str, output: Path) -> None:
    candidates = []
    for step in config["optimization"]["milestones"]:
        path = output / f"development_step_{int(step):06d}.json"
        payload = json.loads(path.read_text())
        checkpoint = resolve_workspace_path(ROOT, payload.get("checkpoint", ""))
        if (
            payload.get("status") != "foundir_noise_family_restormer_development_milestone"
            or payload.get("config_sha256") != config_sha
            or payload.get("step") != int(step)
            or not checkpoint.is_file()
            or file_sha256(checkpoint) != payload.get("checkpoint_sha256")
            or payload.get("development_pixels_accessed") is not False
            or payload.get("official_foundir_test_accessed") is not False
        ):
            raise RuntimeError(f"invalid Noise-family milestone: {path}")
        candidates.append((payload, path))
    selected, result_path = max(
        candidates, key=lambda item: (float(item[0]["selection_score"]), -int(item[0]["step"]))
    )
    selection = {
        "schema_version": 1,
        "status": "foundir_noise_family_restormer_development_selection",
        "config_sha256": config_sha,
        "task": "noise_family",
        "selection_rule": config["selection_rule"],
        "selected_step": selected["step"],
        "checkpoint": selected["checkpoint"],
        "checkpoint_sha256": selected["checkpoint_sha256"],
        "development_result": result_path.relative_to(ROOT).as_posix(),
        "development_result_sha256": file_sha256(result_path),
        "per_category": selected["per_category"],
        "joint_gate_pass": selected["joint_gate_pass"],
        "one_bounded_pilot_no_automatic_tuning": True,
        "development_pixels_accessed": False,
        "official_foundir_test_accessed": False,
    }
    write_once(
        output / "development_selection.json",
        (json.dumps(selection, indent=2, sort_keys=True) + "\n").encode(),
    )
    print("FOUNDIR_NOISE_FAMILY_COMPLETE " + json.dumps(selection, sort_keys=True), flush=True)


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
            "FOUNDIR_NOISE_FAMILY_PREFLIGHT_PASS "
            + json.dumps(
                {
                    "config_sha256": config_sha,
                    "train_records": len(rows["train"]),
                    "validation_records": len(rows["validation"]),
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
        raise RuntimeError("Noise-family pilot requires exactly eight visible GPUs")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if not any(name in torch.cuda.get_device_properties(device).name for name in ("H100", "H200")):
        raise RuntimeError("Noise-family pilot requires H100/H200 GPUs")
    dist.init_process_group("nccl", timeout=timedelta(minutes=30))
    torch.manual_seed(int(optimization["seed"]))
    torch.cuda.manual_seed_all(int(optimization["seed"]))

    model, metadata = load_task_native_restormer(
        config["base_checkpoint_name"],
        resolve_workspace_path(ROOT, Path(config["base_checkpoint"]).parent),
    )
    initial = torch.load(
        resolve_workspace_path(ROOT, config["initial_adaptation_checkpoint"]),
        map_location="cpu",
        weights_only=False,
    )
    model.load_state_dict(initial["model"], strict=True)
    model = model.to(device)
    distributed = DistributedDataParallel(
        model, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False
    )
    optimizer = torch.optim.AdamW(
        distributed.parameters(),
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
        model=distributed.module,
        optimizer=optimizer,
        device=device,
    )
    starts = torch.tensor([current_step, current_step], device=device, dtype=torch.int64)
    dist.all_reduce(starts[:1], op=dist.ReduceOp.MIN)
    dist.all_reduce(starts[1:], op=dist.ReduceOp.MAX)
    if int(starts[0].item()) != current_step or int(starts[1].item()) != current_step:
        raise RuntimeError("Noise-family ranks resumed different steps")

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
        "FOUNDIR_NOISE_FAMILY_START "
        + json.dumps(
            {
                "rank": rank,
                "world": world,
                "resume_step": current_step,
                "model": metadata,
                "initial_adaptation_checkpoint_sha256": config[
                    "initial_adaptation_checkpoint_sha256"
                ],
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
            grouped,
            step=step,
            seed=int(optimization["seed"]),
            count=global_batch,
        )
        local_assignments = assignments[rank::world]
        if len(local_assignments) != micro * accumulation:
            raise RuntimeError("Noise-family rank batch arithmetic changed")
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
                local_error,
                device=device,
                world=world,
                operation=f"Noise-family batch load step {step}",
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
                    raise RuntimeError(f"non-finite Noise-family loss at step {step}")
                loss.backward()
            loss_value += float(loss.detach())
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            distributed.parameters(), float(optimization["gradient_clip_norm"])
        )
        if not bool(torch.isfinite(gradient_norm).item()):
            raise RuntimeError(f"non-finite Noise-family gradient at step {step}")
        optimizer.step()
        if step % 50 == 0 and rank == 0:
            print(
                f"FOUNDIR_NOISE_FAMILY_PROGRESS step={step}/{total} crop={crop_size} "
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
                    model=distributed.module,
                    optimizer=optimizer,
                    initial_sha=config["initial_adaptation_checkpoint_sha256"],
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
