#!/usr/bin/env python3
"""Train the frozen direct Low-light+Haze Restormer with full-image milestones."""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import smoke.train_foundir_task_native_restormer_adaptation_ddp as base
from lowlight_restoration.foundir_flux2_protocol import resolve_workspace_path, stable_digest
from smoke.evaluate_foundir_composition_whole_image_v1 import (
    full_psnr,
    tile_boxes,
    tiled_lpips,
)


TASK = "lowlight_haze_direct"
CATEGORY = "09Lowlight_Haze"
EXPECTED = {
    "records": 75498,
    "train_records": 75370,
    "validation_records": 128,
    "train_groups": 7514,
    "validation_groups": 128,
    "milestones": [1000, 3000, 6000],
}


def validate_config(path: Path) -> dict:
    config = json.loads(path.read_text())
    data = config.get("data", {})
    optimization = config.get("optimization", {})
    initialization = config.get("initialization", {})
    final = config.get("final_frozen_comparison", {})
    manifest = resolve_workspace_path(ROOT, config.get("manifest", ""))
    checkpoint = resolve_workspace_path(ROOT, initialization.get("checkpoint", ""))
    comparison = resolve_workspace_path(ROOT, final.get("manifest", ""))
    if (
        config.get("schema_version") != 1
        or config.get("status") != "user_authorized_lowlight_haze_direct_restormer_v1"
        or config.get("task") != TASK
        or config.get("category") != CATEGORY
        or not manifest.is_file()
        or base.file_sha256(manifest) != config.get("manifest_sha256")
        or int(config.get("manifest_expected_records", -1)) != EXPECTED["records"]
        or initialization.get("policy")
        != "public_task_native_restormer_deraining_then_project_only_adaptation"
        or initialization.get("checkpoint_name") != "deraining"
        or not checkpoint.is_file()
        or base.file_sha256(checkpoint) != initialization.get("checkpoint_sha256")
        or initialization.get("foundir_weights_used") is not False
        or int(data.get("train_records", -1)) != EXPECTED["train_records"]
        or int(data.get("validation_records", -1)) != EXPECTED["validation_records"]
        or int(data.get("train_source_groups", -1)) != EXPECTED["train_groups"]
        or int(data.get("validation_source_groups", -1)) != EXPECTED["validation_groups"]
        or int(data.get("excluded_comparison_records", -1)) != 660
        or int(data.get("excluded_comparison_source_groups", -1)) != 221
        or data.get("sampling") != "uniform_source_group_then_uniform_record"
        or data.get("geometric_augmentation") is not True
        or data.get("milestone_evaluation") != "full_resolution_tiled_rgb_uint8"
        or int(data.get("tile_size", -1)) != 512
        or int(data.get("tile_overlap", -1)) != 64
        or int(optimization.get("world_size", -1)) != 8
        or int(optimization.get("global_batch", -1)) != 32
        or int(optimization.get("micro_batch_per_gpu", -1)) != 4
        or int(optimization.get("gradient_accumulation", -1)) != 1
        or int(optimization.get("total_steps", -1)) != 6000
        or optimization.get("milestones") != EXPECTED["milestones"]
        or optimization.get("progressive_crops") != [[0, 128], [1500, 192], [4000, 256]]
        or optimization.get("optimizer") != "AdamW"
        or float(optimization.get("learning_rate", -1)) != 5e-5
        or float(optimization.get("minimum_learning_rate", -1)) != 1e-6
        or int(optimization.get("warmup_steps", -1)) != 100
        or optimization.get("betas") != [0.9, 0.999]
        or float(optimization.get("weight_decay", -1)) != 1e-4
        or float(optimization.get("gradient_clip_norm", -1)) != 1.0
        or optimization.get("loss") != "rgb_l1"
        or optimization.get("precision") != "bfloat16_autocast_fp32_parameters"
        or int(optimization.get("seed", -1)) != 2026092602
        or not comparison.is_file()
        or base.file_sha256(comparison) != final.get("manifest_sha256")
        or int(final.get("category_records", -1)) != 60
        or final.get("use_for_gradient_or_checkpoint_selection") is not False
        or config.get("selection_rule")
        != "maximize_worst_normalized_development_gate_margin_then_earlier_step"
        or config.get("one_bounded_run_no_automatic_follow_on") is not True
        or config.get("sealed_internal_test_accessed") is not False
        or config.get("protected_foundir_test_accessed") is not False
    ):
        raise RuntimeError("Low-light+Haze Restormer frozen config changed")
    if int(optimization["global_batch"]) != 8 * 4:
        raise RuntimeError("Low-light+Haze global batch arithmetic changed")
    config["initial_checkpoint_name"] = initialization["checkpoint_name"]
    config["initial_checkpoint"] = initialization["checkpoint"]
    config["strong_gate"] = config["development_gate"]
    return config


def load_task_rows(config: Mapping[str, object]) -> dict[str, list[dict[str, str]]]:
    manifest = resolve_workspace_path(ROOT, config["manifest"])
    selected: dict[str, list[dict[str, str]]] = defaultdict(list)
    with manifest.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["category"] != CATEGORY or row["split"] not in {"train", "validation"}:
                raise RuntimeError("Low-light+Haze manifest contains an unexpected row")
            selected[row["split"]].append(dict(row))
    if len(selected["train"]) != EXPECTED["train_records"] or len(selected["validation"]) != EXPECTED["validation_records"]:
        raise RuntimeError("Low-light+Haze manifest split cardinality changed")
    train_groups = {row["source_group"] for row in selected["train"]}
    validation_groups = {row["source_group"] for row in selected["validation"]}
    if len(train_groups) != EXPECTED["train_groups"] or len(validation_groups) != EXPECTED["validation_groups"]:
        raise RuntimeError("Low-light+Haze manifest source-group cardinality changed")
    if train_groups & validation_groups:
        raise RuntimeError("Low-light+Haze train/validation source groups overlap")
    return dict(selected)


def _load_rgb(row: Mapping[str, str], key: str, hash_key: str) -> np.ndarray:
    path = resolve_workspace_path(ROOT, row[key])
    if base.file_sha256(path) != row[hash_key]:
        raise RuntimeError(f"Low-light+Haze validation image changed: {row['record_id']} {key}")
    with Image.open(path) as handle:
        return np.asarray(handle.convert("RGB"), dtype=np.uint8).copy()


def restore_whole(
    model: torch.nn.Module,
    pixels: np.ndarray,
    *,
    tile: int,
    overlap: int,
    device: torch.device,
) -> np.ndarray:
    height, width = pixels.shape[:2]
    accumulator = np.zeros((height, width, 3), dtype=np.float32)
    counts = np.zeros((height, width, 1), dtype=np.float32)
    for y0, x0, y1, x1 in tile_boxes(height, width, tile, overlap):
        patch = torch.from_numpy(pixels[y0:y1, x0:x1].copy()).permute(2, 0, 1).float().div_(255).unsqueeze(0).to(device)
        patch_height, patch_width = patch.shape[-2:]
        pad_height, pad_width = (-patch_height) % 8, (-patch_width) % 8
        if pad_height or pad_width:
            patch = F.pad(patch, (0, pad_width, 0, pad_height), mode="reflect")
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            restored = model(patch).float().clamp(0, 1)[:, :, :patch_height, :patch_width]
        array = restored[0].mul(255).permute(1, 2, 0).cpu().numpy()
        accumulator[y0:y1, x0:x1] += array
        counts[y0:y1, x0:x1] += 1
    if np.any(counts == 0):
        raise RuntimeError("Low-light+Haze tiled inference coverage is incomplete")
    return np.rint(accumulator / counts).clip(0, 255).astype(np.uint8)


def validation_payload(
    rows: Sequence[Mapping[str, object]],
    *,
    config: Mapping[str, object],
    config_sha: str,
    checkpoint: Path,
    step: int,
) -> dict[str, object]:
    if len(rows) != EXPECTED["validation_records"]:
        raise RuntimeError("Low-light+Haze validation coverage changed")
    metrics = {
        "records": len(rows),
        "source_groups": len({str(row["source_group"]) for row in rows}),
        "psnr": float(np.mean([row["restored"]["psnr"] for row in rows])),
        "lpips": float(np.mean([row["restored"]["lpips"] for row in rows])),
    }
    input_metrics = {
        "psnr": float(np.mean([row["input"]["psnr"] for row in rows])),
        "lpips": float(np.mean([row["input"]["lpips"] for row in rows])),
    }
    gate = config["development_gate"]
    bootstrap = base.source_group_bootstrap(
        rows,
        resamples=int(gate["bootstrap_resamples"]),
        confidence=float(gate["bootstrap_confidence"]),
        seed=int(stable_digest(int(gate["bootstrap_seed"]), TASK, step)[:16], 16),
    )
    checks = base.strong_gate_checks(metrics, bootstrap, gate)
    normalized = [
        metrics["psnr"] - float(gate["mean_psnr_minimum"]),
        (float(gate["mean_lpips_maximum"]) - metrics["lpips"]) / 0.02,
        float(bootstrap["lower"]["psnr"]) - float(gate["group_bootstrap_psnr_lower_minimum"]),
        (float(gate["group_bootstrap_lpips_upper_maximum"]) - float(bootstrap["upper"]["lpips"])) / 0.02,
    ]
    if not all(math.isfinite(value) for value in normalized):
        raise RuntimeError("Low-light+Haze validation score is non-finite")
    return {
        "schema_version": 1,
        "status": "foundir_task_native_adaptation_development_milestone",
        "config_sha256": config_sha,
        "task": TASK,
        "category": CATEGORY,
        "split": "validation",
        "evaluation_protocol": "full_resolution_overlapping_512_tile_rgb_uint8",
        "step": step,
        "checkpoint": checkpoint.relative_to(ROOT).as_posix(),
        "checkpoint_sha256": base.file_sha256(checkpoint),
        "input_metrics": input_metrics,
        "metrics": metrics,
        "source_group_bootstrap": bootstrap,
        "strong_gate_checks": checks,
        "strong_gate_pass": all(checks.values()),
        "selection_score": min(normalized),
        "capacity_only_not_final_deployable_routing": True,
        "sealed_internal_test_accessed": False,
        "protected_foundir_test_accessed": False,
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
) -> dict[str, object] | None:
    model.eval()
    local: list[dict[str, object]] = []
    local_error = None
    try:
        for index in range(rank, len(rows), world):
            row = rows[index]
            lq = _load_rgb(row, "lq_path", "lq_sha256")
            gt = _load_rgb(row, "gt_path", "gt_sha256")
            if lq.shape != gt.shape:
                raise RuntimeError(f"unaligned validation pair: {row['record_id']}")
            restored = restore_whole(
                model,
                lq,
                tile=int(config["data"]["tile_size"]),
                overlap=int(config["data"]["tile_overlap"]),
                device=device,
            )
            local.append(
                {
                    "record_id": row["record_id"],
                    "source_group": row["source_group"],
                    "input": {
                        "psnr": full_psnr(gt, lq),
                        "lpips": tiled_lpips(metric, gt, lq, device),
                    },
                    "restored": {
                        "psnr": full_psnr(gt, restored),
                        "lpips": tiled_lpips(metric, gt, restored, device),
                    },
                }
            )
            metric.reset()
            print(
                f"FOUNDIR_LOWLIGHT_HAZE_VALIDATION rank={rank} step={step} record={index + 1}/{len(rows)}",
                flush=True,
            )
    except Exception as error:
        local_error = f"{type(error).__name__}: {error}"
    base.raise_if_any_rank_failed(
        local_error,
        device=device,
        world=world,
        operation=f"Low-light+Haze full-resolution evaluation at step {step}",
    )
    gathered: list[object | None] = [None] * world
    dist.all_gather_object(gathered, local)
    payload = None
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
        result_path = checkpoint.parent / f"development_step_{step:06d}.json"
        base.write_once(result_path, (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode())
        print("FOUNDIR_LOWLIGHT_HAZE_MILESTONE " + json.dumps(payload, sort_keys=True), flush=True)
    model.train()
    return payload


def main() -> None:
    base.TASK_CONTRACTS[TASK] = {
        "milestones": EXPECTED["milestones"],
        "progressive_crops": [[0, 128], [1500, 192], [4000, 256]],
    }
    base.validate_config = validate_config
    base.load_task_rows = load_task_rows
    base.validation_payload = validation_payload
    base.evaluate = evaluate
    base.main()


if __name__ == "__main__":
    main()
