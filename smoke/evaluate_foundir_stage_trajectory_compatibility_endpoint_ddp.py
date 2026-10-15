#!/usr/bin/env python3
"""One-shot full-resolution triple endpoint for the frozen compatibility adapter."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lowlight_restoration.stage_trajectory_compatibility import StageTrajectoryCompatibilityAdapter
from smoke.evaluate_foundir_composition_whole_image_v1 import full_psnr, tile_boxes, tiled_lpips
from smoke.train_foundir_stage_trajectory_compatibility_ddp import load_config, resolve, sha256


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8).copy()


def save_rgb(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array, mode="RGB").save(path, format="PNG", compress_level=4)


def trajectory(row: dict[str, str], config: dict) -> tuple[list[np.ndarray], np.ndarray]:
    root = resolve(config["frozen_endpoint"]["trajectory_root"])
    record_id = row["record_id"]
    lq = load_rgb(resolve(row["lq_path"]))
    if row["routing_status"] == "correct":
        states = [
            lq,
            load_rgb(root / "noise" / f"{record_id}.png"),
            load_rgb(root / "low_light" / f"{record_id}.png"),
            load_rgb(root / "blur" / f"{record_id}.png"),
        ]
        presence = np.asarray([1.0, 1.0, 1.0], dtype=np.float32)
    elif row["routing_status"] == "fallback":
        low_light = load_rgb(root / "low_light" / f"{record_id}.png")
        states = [lq, lq.copy(), low_light, low_light.copy()]
        presence = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
    else:
        raise RuntimeError(f"unexpected triple routing status: {record_id}")
    if len({state.shape for state in states}) != 1:
        raise RuntimeError(f"trajectory shape mismatch: {record_id}")
    return states, presence


def restore_tiled(model, states: list[np.ndarray], presence: np.ndarray, device: torch.device) -> np.ndarray:
    height, width = states[0].shape[:2]
    accumulator = np.zeros((height, width, 3), dtype=np.float32)
    counts = np.zeros((height, width, 1), dtype=np.float32)
    for y0, x0, y1, x1 in tile_boxes(height, width, 512, 64):
        tensors = []
        patch_height, patch_width = y1 - y0, x1 - x0
        for state in states:
            patch = torch.from_numpy(state[y0:y1, x0:x1].copy()).permute(2, 0, 1).float().div_(255).unsqueeze(0).to(device)
            pad_h, pad_w = (-patch_height) % 8, (-patch_width) % 8
            if pad_h or pad_w:
                patch = F.pad(patch, (0, pad_w, 0, pad_h), mode="reflect")
            tensors.append(patch)
        control = torch.from_numpy(presence).unsqueeze(0).to(device)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(*tensors, control)[0, :, :patch_height, :patch_width]
        array = output.float().mul(255).permute(1, 2, 0).cpu().numpy()
        accumulator[y0:y1, x0:x1] += array
        counts[y0:y1, x0:x1] += 1
    if np.any(counts == 0):
        raise RuntimeError("endpoint tiled coverage incomplete")
    return np.rint(accumulator / counts).clip(0, 255).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/foundir_stage_trajectory_compatibility_v1.json")
    args = parser.parse_args()
    config_path = resolve(args.config)
    config = load_config(config_path)
    output = resolve(config["runtime"]["output_dir"])
    result_path = output / "endpoint_result.json"
    if result_path.exists():
        raise FileExistsError("refusing to rerun frozen triple endpoint")
    selection = json.loads((output / "development_selection.json").read_text())
    if selection.get("status") != "stage_trajectory_checkpoint_selection_frozen" or selection.get("endpoint_promotion_pass") is not True:
        raise RuntimeError("double-only selection did not authorize triple endpoint")
    checkpoint = resolve(selection["selected_checkpoint"])
    if sha256(checkpoint) != selection["selected_checkpoint_sha256"]:
        raise RuntimeError("selected compatibility checkpoint changed")

    with resolve(config["frozen_endpoint"]["manifest"]).open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["category"] == config["frozen_endpoint"]["category"]]
    if len(rows) != 60:
        raise RuntimeError("frozen triple endpoint cardinality changed")

    rank, world, local_rank = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), int(os.environ["LOCAL_RANK"])
    if world != 8:
        raise RuntimeError("endpoint requires eight GPUs")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    model = StageTrajectoryCompatibilityAdapter(
        channels=int(config["model"]["channels"]), blocks=int(config["model"]["blocks"]),
        maximum_residual=float(config["model"]["maximum_residual"]),
    ).to(device)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    metric = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(device).eval()
    image_dir, sidecar_dir = output / "endpoint_images", output / "endpoint_per_record"
    image_dir.mkdir(parents=True, exist_ok=True)
    sidecar_dir.mkdir(parents=True, exist_ok=True)

    for index in range(rank, len(rows), world):
        row = rows[index]
        record_id = row["record_id"]
        states, presence = trajectory(row, config)
        gt = load_rgb(resolve(row["gt_path"]))
        restored = restore_tiled(model, states, presence, device)
        destination = image_dir / f"{record_id}.png"
        save_rgb(destination, restored)
        parent = json.loads((resolve("checkpoints/foundir_composition_whole_image_v1/per_record") / f"{record_id}.json").read_text())
        full = json.loads((resolve("checkpoints/foundir_composition_whole_image_full_pipeline_v1/per_record") / f"{record_id}.json").read_text())
        values = {
            "schema_version": 1, "status": "complete", "record_id": record_id,
            "source_group": row["source_group"], "routing_status": row["routing_status"],
            "adapter": {"psnr": full_psnr(gt, restored), "lpips": tiled_lpips(metric, gt, restored, device)},
            "project_baseline": parent["ours"],
            "released_generalist": parent["foundir"],
            "full_foundir": full["full_foundir_generalist_specialist"],
            "output_sha256": sha256(destination), "checkpoint_sha256": sha256(checkpoint),
            "official_foundir_test_accessed": False,
        }
        (sidecar_dir / f"{record_id}.json").write_text(json.dumps(values, indent=2, sort_keys=True) + "\n")
        print(f"STAGE_TRAJECTORY_ENDPOINT rank={rank} record={index + 1}/60", flush=True)
    dist.barrier()

    if rank == 0:
        records = [json.loads((sidecar_dir / f"{row['record_id']}.json").read_text()) for row in rows]
        groups = {
            "all_60": records,
            "correct_53": [item for item in records if item["routing_status"] == "correct"],
            "fallback_7": [item for item in records if item["routing_status"] == "fallback"],
        }
        aggregates = {}
        for name, values in groups.items():
            aggregate = {"records": len(values)}
            for system in ("adapter", "project_baseline", "released_generalist", "full_foundir"):
                aggregate[system] = {
                    metric_name: float(np.mean([item[system][metric_name] for item in values]))
                    for metric_name in ("psnr", "lpips")
                }
            aggregate["adapter_minus_project"] = {
                "psnr": aggregate["adapter"]["psnr"] - aggregate["project_baseline"]["psnr"],
                "lpips": aggregate["adapter"]["lpips"] - aggregate["project_baseline"]["lpips"],
            }
            aggregates[name] = aggregate
        result = {
            "schema_version": 1, "status": "stage_trajectory_compatibility_endpoint_complete",
            "config_sha256": sha256(config_path), "checkpoint": selection["selected_checkpoint"],
            "checkpoint_sha256": selection["selected_checkpoint_sha256"], "aggregates": aggregates,
            "model_weight_updates_during_endpoint": False, "official_foundir_test_accessed": False,
        }
        result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
