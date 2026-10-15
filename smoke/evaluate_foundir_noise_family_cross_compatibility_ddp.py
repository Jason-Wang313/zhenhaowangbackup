#!/usr/bin/env python3
"""Test the existing Noise->Low-light route on paired pure-Noise sources."""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from smoke.evaluate_foundir_mixed_restoration_pilot_ddp import (
    crop_pair,
    initialize_distributed,
    load_json,
    lpips_value,
    paired_source_group_bootstrap,
    psnr,
    run_stage,
    sha256,
    stage_path,
    workspace,
    write_json_once,
)


def contract(path: Path) -> tuple[dict, list[dict[str, str]], str]:
    config = load_json(path)
    config_sha = sha256(path)
    if config.get("status") != "frozen_existing_route_cross_compatibility_v1":
        raise RuntimeError("bad cross-compatibility contract")
    scope = config["scope"]
    if scope.get("split") != "train" or scope.get("model_weight_updates") is not False or scope.get("development_pixels_accessed") is not False or scope.get("official_foundir_test_accessed") is not False:
        raise RuntimeError("protected data or mutation forbidden")
    manifest = workspace(config["data"]["manifest"])
    if sha256(manifest) != config["data"]["manifest_sha256"]:
        raise RuntimeError("manifest changed")
    with manifest.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 128 or len({row["source_group"] for row in rows}) != 128 or {row["category"] for row in rows} != {"05Noise"}:
        raise RuntimeError("manifest cardinality changed")
    if config["orders"] != {"05Noise": ["noise", "low_light"]}:
        raise RuntimeError("route changed")
    for expert in config["experts"].values():
        if sha256(workspace(expert["checkpoint"])) != expert["checkpoint_sha256"] or sha256(workspace(expert["parent_config"])) != expert["parent_config_sha256"]:
            raise RuntimeError("expert artifact changed")
    return config, rows, config_sha


def metrics(config: dict, rows: list[dict[str, str]], config_sha: str) -> None:
    rank, world, device, profile = initialize_distributed(config)
    started = time.monotonic()
    try:
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

        metric = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(device).eval()
        output = workspace(config["runtime"]["output_dir"])
        sidecars = output / "per_record"
        for index in range(rank, len(rows), world):
            row = rows[index]
            sidecar = sidecars / f"{row['record_id']}.json"
            if sidecar.exists():
                continue
            lq, gt, crop_xy = crop_pair(row, config)
            target = gt.unsqueeze(0).to(device)
            images = {"input": lq}
            for role, task in (("noise_only", "noise"), ("noise_then_lowlight", "low_light")):
                with Image.open(stage_path(output, task, row["record_id"])) as handle:
                    array = np.asarray(handle.convert("RGB"), dtype=np.uint8).copy()
                images[role] = torch.from_numpy(array).permute(2, 0, 1).float().div_(127.5).sub_(1)
            values = {}
            for role, image in images.items():
                prediction = image.unsqueeze(0).to(device)
                values[role] = {"psnr": psnr(target, prediction), "lpips": lpips_value(metric, target, prediction)}
                metric.reset()
            write_json_once(sidecar, {"schema_version": 1, "config_sha256": config_sha, "record_id": row["record_id"], "source_group": row["source_group"], "crop_xy": crop_xy, **values, "development_pixels_accessed": False, "official_foundir_test_accessed": False})
        dist.barrier()
        if rank == 0:
            payloads = [load_json(sidecars / f"{row['record_id']}.json") for row in rows]
            means = {
                role: {name: float(np.mean([item[role][name] for item in payloads])) for name in ("psnr", "lpips")}
                for role in ("input", "noise_only", "noise_then_lowlight")
            }
            paired = [{"source_group": item["source_group"], "input": item["noise_only"], "restored": item["noise_then_lowlight"]} for item in payloads]
            bootstrap = paired_source_group_bootstrap(paired, resamples=int(config["gate"]["bootstrap_resamples"]), confidence=float(config["gate"]["bootstrap_confidence"]), seed=int(config["gate"]["bootstrap_seed"]))
            checks = {
                "noise_only_meets_sow_psnr": means["noise_only"]["psnr"] >= config["gate"]["noise_sow_psnr_minimum"],
                "noise_only_meets_sow_lpips": means["noise_only"]["lpips"] <= config["gate"]["noise_sow_lpips_maximum"],
                "noise_then_lowlight_meets_sow_psnr": means["noise_then_lowlight"]["psnr"] >= config["gate"]["noise_sow_psnr_minimum"],
                "noise_then_lowlight_meets_sow_lpips": means["noise_then_lowlight"]["lpips"] <= config["gate"]["noise_sow_lpips_maximum"],
                "noise_then_lowlight_psnr_not_worse_than_noise_only": means["noise_then_lowlight"]["psnr"] >= means["noise_only"]["psnr"],
                "noise_then_lowlight_lpips_not_worse_than_noise_only": means["noise_then_lowlight"]["lpips"] <= means["noise_only"]["lpips"],
            }
            result = {"schema_version": 1, "status": "noise_family_reverse_cross_compatibility_complete", "gate_pass": all(checks.values()), "records": len(rows), "source_groups": len(rows), "metrics": means, "bootstrap_noise_then_lowlight_vs_noise_only": bootstrap, "checks": checks, "hardware_profile": profile, "world_size": world, "wall_seconds": time.monotonic() - started, "model_weight_updates": False, "development_pixels_accessed": False, "official_foundir_test_accessed": False}
            write_json_once(output / "result.json", result)
            print("FOUNDIR_NOISE_FAMILY_REVERSE_CROSS_COMPATIBILITY " + json.dumps(result, sort_keys=True), flush=True)
        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=("noise", "low_light", "metrics"), required=True)
    args = parser.parse_args()
    config, rows, config_sha = contract(args.config.resolve())
    if args.stage in {"noise", "low_light"}:
        run_stage(args.stage, config, rows, config_sha)
    else:
        metrics(config, rows, config_sha)


if __name__ == "__main__":
    main()
