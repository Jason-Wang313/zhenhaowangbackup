#!/usr/bin/env python3
"""Compare cached Noise-only and Noise->Low-light outputs on Noise+Low-light."""
from __future__ import annotations

import csv
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from smoke.evaluate_foundir_mixed_restoration_pilot_ddp import (
    crop_pair,
    load_json,
    lpips_value,
    paired_source_group_bootstrap,
    psnr,
    sha256,
    stage_path,
    workspace,
    write_json_once,
)

CONFIG = ROOT / "configs/foundir_mixed_restoration_confirmation_v1.json"
CONFIG_SHA = "209290a19f4227b9ca947cc973bb64cdfd1292125c0d2f1951ebd9bc63f29dde"
MANIFEST_SHA = "188e7c60e88c6fedfb2fa9a86604f58f7f83cb152435162206f4ec04e9b9255a"
OUTPUT = ROOT / "checkpoints/foundir_noise_route_cross_compatibility_v1"


def main() -> None:
    if sha256(CONFIG) != CONFIG_SHA:
        raise RuntimeError("confirmation config changed")
    config = load_json(CONFIG)
    manifest = workspace(config["data"]["manifest"])
    if sha256(manifest) != MANIFEST_SHA:
        raise RuntimeError("confirmation manifest changed")
    if config["scope"].get("official_foundir_test_accessed") is not False or config["scope"].get("development_pixels_accessed") is not False:
        raise RuntimeError("protected data policy changed")
    with manifest.open(newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["category"] == "16Lowlight_Noise"]
    if len(rows) != 128 or len({row["source_group"] for row in rows}) != 128:
        raise RuntimeError("Noise+Low-light confirmation identity changed")
    source_output = workspace(config["runtime"]["output_dir"])
    for row in rows:
        if not stage_path(source_output, "noise", row["record_id"]).is_file() or not stage_path(source_output, "low_light", row["record_id"]).is_file():
            raise RuntimeError(f"cached restoration missing: {row['record_id']}")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    rank, world = dist.get_rank(), dist.get_world_size()
    if world != 8 or torch.cuda.device_count() != 8:
        raise RuntimeError("requires exactly eight visible GPUs")
    started = time.monotonic()
    try:
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

        metric = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(device).eval()
        sidecars = OUTPUT / "per_record"
        for index in range(rank, len(rows), world):
            row = rows[index]
            sidecar = sidecars / f"{row['record_id']}.json"
            if sidecar.exists():
                continue
            lq, gt, crop_xy = crop_pair(row, config)
            target = gt.unsqueeze(0).to(device)
            values = {}
            candidates = {"input": lq}
            for role, task in (("noise_only", "noise"), ("noise_then_lowlight", "low_light")):
                with Image.open(stage_path(source_output, task, row["record_id"])) as handle:
                    array = np.asarray(handle.convert("RGB"), dtype=np.uint8).copy()
                candidates[role] = torch.from_numpy(array).permute(2, 0, 1).float().div_(127.5).sub_(1)
            for role, image in candidates.items():
                prediction = image.unsqueeze(0).to(device)
                values[role] = {"psnr": psnr(target, prediction), "lpips": lpips_value(metric, target, prediction)}
                metric.reset()
            write_json_once(sidecar, {
                "schema_version": 1, "record_id": row["record_id"], "source_group": row["source_group"],
                "crop_xy": crop_xy, **values, "official_foundir_test_accessed": False,
                "development_pixels_accessed": False,
            })
        dist.barrier()
        if rank == 0:
            payloads = [load_json(sidecars / f"{row['record_id']}.json") for row in rows]
            means = {
                role: {metric_name: float(np.mean([item[role][metric_name] for item in payloads])) for metric_name in ("psnr", "lpips")}
                for role in ("input", "noise_only", "noise_then_lowlight")
            }
            bootstrap_rows = [
                {"source_group": item["source_group"], "input": item["input"], "restored": item["noise_only"]}
                for item in payloads
            ]
            bootstrap = paired_source_group_bootstrap(bootstrap_rows, resamples=5000, confidence=0.95, seed=20260924)
            checks = {
                "noise_only_psnr_improves_over_input": means["noise_only"]["psnr"] > means["input"]["psnr"],
                "noise_only_lpips_improves_over_input": means["noise_only"]["lpips"] < means["input"]["lpips"],
                "bootstrap_psnr_gain_lower_positive": bootstrap["lower"]["psnr_gain_db"] > 0,
                "bootstrap_lpips_delta_upper_negative": bootstrap["upper"]["lpips_delta"] < 0,
                "noise_only_psnr_not_worse_than_existing_route": means["noise_only"]["psnr"] >= means["noise_then_lowlight"]["psnr"],
                "noise_only_lpips_not_worse_than_existing_route": means["noise_only"]["lpips"] <= means["noise_then_lowlight"]["lpips"],
            }
            result = {
                "schema_version": 1,
                "status": "noise_route_cross_compatibility_complete",
                "gate_pass": all(checks.values()),
                "records": len(rows),
                "source_groups": len(rows),
                "metrics": means,
                "bootstrap_noise_only_vs_input": bootstrap,
                "checks": checks,
                "wall_seconds": time.monotonic() - started,
                "world_size": world,
                "model_inference_reused": True,
                "model_weight_updates": False,
                "development_pixels_accessed": False,
                "official_foundir_test_accessed": False,
            }
            write_json_once(OUTPUT / "result.json", result)
            print("FOUNDIR_NOISE_ROUTE_CROSS_COMPATIBILITY " + json.dumps(result, sort_keys=True), flush=True)
        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
