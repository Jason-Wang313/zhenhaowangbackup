#!/usr/bin/env python3
"""Confirm frozen Noise-only alpha=.5 on 16 fresh Blur+Noise training records."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party/flux2/src"))

from lowlight_restoration.scalar_residual import quantized_scalar_residual  # noqa: E402
from smoke.evaluate_foundir_blur_only_flux2_full_development_ddp import restore_batch  # noqa: E402
from smoke.evaluate_foundir_mixed_restoration_pilot_ddp import (  # noqa: E402
    crop_pair,
    initialize_distributed,
    load_flux_expert,
    save_png_once,
    sha256,
    stable_digest,
    workspace,
    write_json_once,
)
from smoke.train_foundir_blur_only_flux2_overfit_ddp import lpips_value, psnr  # noqa: E402


def validate(path: Path, *, verify_images: bool) -> tuple[dict, list[dict[str, str]]]:
    config = json.loads(path.read_text())
    scope, data, evidence, expert = (
        config.get("scope", {}), config.get("data", {}),
        config.get("selection_evidence", {}), config.get("expert", {}),
    )
    if (
        config.get("status") != "user_authorized_frozen_blur_noise_noise_only_confirmation_v1"
        or scope.get("split") != "train"
        or scope.get("category") != "02Blur_Noise"
        or scope.get("model_weight_updates") is not False
        or scope.get("alpha_selection_reopened") is not False
        or scope.get("development_pixels_accessed") is not False
        or scope.get("official_foundir_test_accessed") is not False
        or float(evidence.get("selected_alpha", -1)) != 0.5
        or float(expert.get("alpha", -1)) != 0.5
        or int(config["inference"].get("num_steps", -1)) != 50
    ):
        raise RuntimeError("Blur+Noise confirmation contract changed")
    for file_path, expected in (
        (workspace(data["manifest"]), data["manifest_sha256"]),
        (workspace(evidence["path"]), evidence["sha256"]),
        (workspace(expert["parent_config"]), expert["parent_config_sha256"]),
        (workspace(expert["checkpoint"]), expert["checkpoint_sha256"]),
    ):
        if not file_path.is_file() or sha256(file_path) != expected:
            raise RuntimeError(f"frozen dependency changed: {file_path}")
    selection = json.loads(workspace(evidence["path"]).read_text())
    if selection.get("selected_alpha") != 0.5 or selection.get("status") != "fixed_alpha_selected":
        raise RuntimeError("alpha selection evidence changed")
    with workspace(data["manifest"]).open(newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    if (
        len(rows) != 16
        or len({row["source_group"] for row in rows}) != 16
        or any(row["split"] != "train" or row["category"] != "02Blur_Noise" for row in rows)
    ):
        raise RuntimeError("confirmation manifest contract changed")
    if verify_images:
        for row in rows:
            for field, hash_field in (("lq_path", "lq_sha256"), ("gt_path", "gt_sha256")):
                image = workspace(row[field])
                if not image.is_file() or sha256(image) != row[hash_field]:
                    raise RuntimeError(f"confirmation image changed: {row['record_id']} {field}")
    return config, rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    config_path = workspace(args.config)
    config, rows = validate(config_path, verify_images=args.preflight)
    if args.preflight:
        print(json.dumps({
            "status": "BLUR_NOISE_NOISE_ONLY_CONFIRMATION_PREFLIGHT_PASS",
            "records": 16,
            "source_groups": 16,
            "alpha": 0.5,
            "development_pixels_accessed": False,
            "official_foundir_test_accessed": False,
        }, sort_keys=True))
        return 0

    rank, world, device, profile = initialize_distributed(config)
    started = time.monotonic()
    try:
        output = workspace(config["runtime"]["output_dir"])
        sidecars, visuals = output / "per_record", output / "fixed_visuals"
        model, autoencoder, context, ids, pack_image = load_flux_expert(
            "noise", config["expert"], config, device
        )
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

        metric = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(device).eval()
        config_sha = sha256(config_path)
        local_rows = rows[rank::world]
        if len(local_rows) != 2:
            raise RuntimeError("confirmation rank partition changed")
        pending = [row for row in local_rows if not (sidecars / f"{row['record_id']}.json").exists()]
        if pending:
            pairs = [crop_pair(row, config) for row in pending]
            source = torch.stack([pair[0] for pair in pairs]).to(device)
            target = torch.stack([pair[1] for pair in pairs]).to(device)
            seeds = [
                int(stable_digest(config["inference"]["seed"], "noise_only_confirmation", row["record_id"])[:16], 16)
                for row in pending
            ]
            restored = restore_batch(
                model, autoencoder, source, context, ids, pack_image,
                seeds=seeds,
                steps=int(config["inference"]["num_steps"]),
                reference_time=int(config["inference"]["reference_time_coordinate"]),
            )
            arrays, candidate = quantized_scalar_residual(source, restored, 0.5)
            for index, row in enumerate(pending):
                values = {}
                for role, image in (("input", source[index:index+1]), ("candidate", candidate[index:index+1])):
                    values[role] = {
                        "psnr": psnr(target[index:index+1], image),
                        "lpips": lpips_value(metric, target[index:index+1], image),
                    }
                    metric.reset()
                write_json_once(sidecars / f"{row['record_id']}.json", {
                    "schema_version": 1,
                    "status": "complete",
                    "config_sha256": config_sha,
                    "record_id": row["record_id"],
                    "source_group": row["source_group"],
                    "alpha": 0.5,
                    **values,
                    "development_pixels_accessed": False,
                    "official_foundir_test_accessed": False,
                })
                triptych = np.concatenate([
                    value.detach().cpu().add(1).mul(127.5).round().clamp(0, 255).byte().permute(1, 2, 0).numpy()
                    for value in (source[index], candidate[index], target[index])
                ], axis=1)
                save_png_once(visuals / f"{row['record_id']}.png", triptych)
        dist.barrier()
        if rank == 0:
            payloads = [json.loads((sidecars / f"{row['record_id']}.json").read_text()) for row in rows]
            means = {
                role: {
                    metric_name: float(np.mean([row[role][metric_name] for row in payloads]))
                    for metric_name in ("psnr", "lpips")
                }
                for role in ("input", "candidate")
            }
            checks = {
                "mean_psnr_improves": means["candidate"]["psnr"] > means["input"]["psnr"],
                "mean_lpips_improves": means["candidate"]["lpips"] < means["input"]["lpips"],
            }
            result = {
                "schema_version": 1,
                "status": "awaiting_fixed_visual_review" if all(checks.values()) else "failed",
                "automatic_metric_gate_pass": all(checks.values()),
                "fixed_visual_review_required": True,
                "fixed_visual_review_complete": False,
                "records": 16,
                "source_groups": 16,
                "alpha": 0.5,
                "input": means["input"],
                "candidate": means["candidate"],
                "delta": {
                    "psnr_db": means["candidate"]["psnr"] - means["input"]["psnr"],
                    "lpips": means["candidate"]["lpips"] - means["input"]["lpips"],
                },
                "checks": checks,
                "psnr_win_fraction": float(np.mean([row["candidate"]["psnr"] > row["input"]["psnr"] for row in payloads])),
                "lpips_win_fraction": float(np.mean([row["candidate"]["lpips"] < row["input"]["lpips"] for row in payloads])),
                "hardware_profile": profile,
                "world_size": world,
                "wall_seconds": time.monotonic() - started,
                "model_weight_updates": False,
                "development_pixels_accessed": False,
                "official_foundir_test_accessed": False,
            }
            write_json_once(output / "result.json", result)
            print("BLUR_NOISE_NOISE_ONLY_CONFIRMATION_RESULT " + json.dumps(result, sort_keys=True), flush=True)
        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
