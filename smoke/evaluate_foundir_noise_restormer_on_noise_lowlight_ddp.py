#!/usr/bin/env python3
"""Test the frozen historical Noise Restormer on Noise+Low-light confirmation data."""
from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lowlight_restoration.task_native_restormer import load_task_native_restormer
from smoke.evaluate_foundir_mixed_restoration_pilot_ddp import (
    crop_pair, load_json, lpips_value, paired_source_group_bootstrap, psnr, sha256,
    stage_path, workspace, write_json_once,
)

CONFIG = ROOT / "configs/foundir_mixed_restoration_confirmation_v1.json"
CONFIG_SHA = "209290a19f4227b9ca947cc973bb64cdfd1292125c0d2f1951ebd9bc63f29dde"
CHECKPOINT = ROOT / "checkpoints/foundir_five_type_flux2_dit_v3/noise_restormer_adaptation_v1/step_000250.pt"
CHECKPOINT_SHA = "b3fca23f8271477834c6b3eba2a6dfdf6fcbd480e53c43570e42960a13044429"
PARENT = ROOT / "models/restormer-task-native/real_denoising.pth"
PARENT_SHA = "4cae18bed8a291b9a5deeeab48756bbe6f61fcf7f31a4cca0969b24608601387"
FORBIDDEN_FOUNDIR_BASELINE_SHAS = {
    "2603a0be47d0844d4c9cea542ecca039e0e4aaa39d45d41064bfbdc13c0830c9",  # generalist
    "f834248e145d020d5a02f257b7ceb383b579add2522932938b3e1e0bd460abe6",  # low-light specialist
    "25fa90003de8fa6965d861fc66159fb0c188654723acc15e8c49ec8359ca8b93",  # weather specialist
}
OUTPUT = ROOT / "checkpoints/foundir_noise_restormer_on_noise_lowlight_v1"


def main() -> None:
    parent_sha = sha256(PARENT)
    checkpoint_sha = sha256(CHECKPOINT)
    if sha256(CONFIG) != CONFIG_SHA or parent_sha != PARENT_SHA or checkpoint_sha != CHECKPOINT_SHA:
        raise RuntimeError("frozen input artifact changed")
    if parent_sha in FORBIDDEN_FOUNDIR_BASELINE_SHAS or checkpoint_sha in FORBIDDEN_FOUNDIR_BASELINE_SHAS:
        raise RuntimeError("refusing released FoundIR baseline or specialist checkpoint")
    config = load_json(CONFIG)
    if config["scope"].get("development_pixels_accessed") is not False or config["scope"].get("official_foundir_test_accessed") is not False:
        raise RuntimeError("protected data policy changed")
    manifest = workspace(config["data"]["manifest"])
    with manifest.open(newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["category"] == "16Lowlight_Noise"]
    if len(rows) != 128 or len({row["source_group"] for row in rows}) != 128:
        raise RuntimeError("confirmation identity changed")

    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    dist.init_process_group("nccl", device_id=device)
    rank, world = dist.get_rank(), dist.get_world_size()
    if world != 8 or torch.cuda.device_count() != 8:
        raise RuntimeError("requires exactly eight visible GPUs")
    started = time.monotonic()
    try:
        model, _ = load_task_native_restormer("real_denoising", PARENT.parent)
        payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
        if payload.get("task") != "noise" or payload.get("step") != 250:
            raise RuntimeError("historical Noise checkpoint metadata changed")
        model.load_state_dict(payload["model"], strict=True)
        model = model.to(device).eval().requires_grad_(False)
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
        metric = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(device).eval()
        sidecars = OUTPUT / "per_record"
        cached = workspace(config["runtime"]["output_dir"])
        for index in range(rank, len(rows), world):
            row = rows[index]
            sidecar = sidecars / f"{row['record_id']}.json"
            if sidecar.exists():
                continue
            lq, gt, crop_xy = crop_pair(row, config)
            source = ((lq + 1) * 0.5).unsqueeze(0).to(device)
            target = gt.unsqueeze(0).to(device)
            with torch.inference_mode():
                restored = model(source).float().clamp(0, 1)
            array = restored[0].cpu().permute(1, 2, 0).mul(255).round().byte().numpy()
            candidate = torch.from_numpy(array.copy()).permute(2, 0, 1).float().div_(127.5).sub_(1)
            with Image.open(stage_path(cached, "low_light", row["record_id"])) as handle:
                baseline_array = np.asarray(handle.convert("RGB"), dtype=np.uint8).copy()
            baseline = torch.from_numpy(baseline_array).permute(2, 0, 1).float().div_(127.5).sub_(1)
            values = {}
            for role, image in (("input", lq), ("restormer", candidate), ("existing_route", baseline)):
                prediction = image.unsqueeze(0).to(device)
                values[role] = {"psnr": psnr(target, prediction), "lpips": lpips_value(metric, target, prediction)}
                metric.reset()
            write_json_once(sidecar, {"schema_version": 1, "record_id": row["record_id"], "source_group": row["source_group"], "crop_xy": crop_xy, **values, "development_pixels_accessed": False, "official_foundir_test_accessed": False})
        dist.barrier()
        if rank == 0:
            items = [load_json(sidecars / f"{row['record_id']}.json") for row in rows]
            means = {role: {name: float(np.mean([item[role][name] for item in items])) for name in ("psnr", "lpips")} for role in ("input", "restormer", "existing_route")}
            paired = [{"source_group": item["source_group"], "input": item["input"], "restored": item["restormer"]} for item in items]
            bootstrap = paired_source_group_bootstrap(paired, resamples=5000, confidence=0.95, seed=20260927)
            checks = {
                "restormer_psnr_improves_over_input": means["restormer"]["psnr"] > means["input"]["psnr"],
                "restormer_lpips_improves_over_input": means["restormer"]["lpips"] < means["input"]["lpips"],
                "bootstrap_psnr_lower_positive": bootstrap["lower"]["psnr_gain_db"] > 0,
                "bootstrap_lpips_upper_negative": bootstrap["upper"]["lpips_delta"] < 0,
                "restormer_psnr_not_worse_than_existing_route": means["restormer"]["psnr"] >= means["existing_route"]["psnr"],
                "restormer_lpips_not_worse_than_existing_route": means["restormer"]["lpips"] <= means["existing_route"]["lpips"],
            }
            result = {"schema_version": 1, "status": "historical_noise_restormer_mixed_compatibility_complete", "gate_pass": all(checks.values()), "records": 128, "source_groups": 128, "checkpoint_sha256": CHECKPOINT_SHA, "metrics": means, "bootstrap_restormer_vs_input": bootstrap, "checks": checks, "wall_seconds": time.monotonic() - started, "world_size": world, "model_weight_updates": False, "development_pixels_accessed": False, "official_foundir_test_accessed": False}
            write_json_once(OUTPUT / "result.json", result)
            print("FOUNDIR_HISTORICAL_NOISE_RESTORMER_MIXED_RESULT " + json.dumps(result, sort_keys=True), flush=True)
        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
