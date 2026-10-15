#!/usr/bin/env python3
"""Final frozen 60-image comparison for the direct Low-light+Haze Restormer."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lowlight_restoration.task_native_restormer import load_task_native_restormer  # noqa: E402
from smoke.evaluate_foundir_composition_whole_image_v1 import full_psnr, tiled_lpips  # noqa: E402
from smoke.train_foundir_lowlight_haze_restormer_ddp import (  # noqa: E402
    CATEGORY,
    restore_whole,
    validate_config,
)
import smoke.train_foundir_task_native_restormer_adaptation_ddp as base  # noqa: E402


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as handle:
        return np.asarray(handle.convert("RGB"), dtype=np.uint8).copy()


def save_png(path: Path, pixels: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        current = load_rgb(path)
        if current.shape != pixels.shape or not np.array_equal(current, pixels):
            raise RuntimeError(f"immutable Low-light+Haze output changed: {path}")
        return
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    Image.fromarray(pixels, mode="RGB").save(temporary, format="PNG")
    os.replace(temporary, path)


def read_contract(config_path: Path) -> tuple[dict, list[dict[str, str]], dict, Path]:
    config = validate_config(config_path)
    config_sha = base.file_sha256(config_path)
    selection_path = ROOT / config["output"] / "development_selection.json"
    if not selection_path.is_file():
        raise RuntimeError("Low-light+Haze development selection is missing")
    selection = json.loads(selection_path.read_text())
    checkpoint = ROOT / str(selection.get("checkpoint", ""))
    if (
        selection.get("status") != "foundir_task_native_adaptation_development_selection"
        or selection.get("config_sha256") != config_sha
        or selection.get("task") != config["task"]
        or not checkpoint.is_file()
        or base.file_sha256(checkpoint) != selection.get("checkpoint_sha256")
        or selection.get("sealed_internal_test_accessed") is not False
        or selection.get("protected_foundir_test_accessed") is not False
    ):
        raise RuntimeError("Low-light+Haze selected checkpoint contract changed")

    comparison_path = ROOT / config["final_frozen_comparison"]["manifest"]
    with comparison_path.open(newline="") as handle:
        comparison_rows = list(csv.DictReader(handle))
    rows = [row for row in comparison_rows if row["category"] == CATEGORY]
    if len(comparison_rows) != 660 or len(rows) != 60:
        raise RuntimeError("frozen Low-light+Haze comparison cardinality changed")

    training_groups: set[str] = set()
    with (ROOT / config["manifest"]).open(newline="") as handle:
        for row in csv.DictReader(handle):
            training_groups.add(row["source_group"])
    if training_groups & {row["source_group"] for row in comparison_rows}:
        raise RuntimeError("frozen comparison source group leaked into Restormer data")
    return config, rows, selection, checkpoint


def write_json(path: Path, payload: Mapping[str, object]) -> None:
    base.write_once(path, (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    config, rows, selection, checkpoint = read_contract(config_path)
    if args.preflight:
        print(
            "FOUNDIR_LOWLIGHT_HAZE_FINAL_PREFLIGHT_PASS "
            + json.dumps(
                {
                    "records": len(rows),
                    "selected_step": selection["selected_step"],
                    "checkpoint_sha256": selection["checkpoint_sha256"],
                    "official_foundir_test_accessed": False,
                },
                sort_keys=True,
            )
        )
        return

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    if world != 8 or int(os.environ.get("LOCAL_WORLD_SIZE", -1)) != 8 or torch.cuda.device_count() != 8:
        raise RuntimeError("Low-light+Haze final comparison requires exactly eight visible GPUs")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    gpu_name = torch.cuda.get_device_properties(device).name
    if "H100" not in gpu_name and "H200" not in gpu_name:
        raise RuntimeError(f"unsupported GPU: {gpu_name}")
    dist.init_process_group("nccl", device_id=device)

    model, metadata = load_task_native_restormer(
        config["initialization"]["checkpoint_name"],
        ROOT / Path(config["initialization"]["checkpoint"]).parent,
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("config_sha256") != base.file_sha256(config_path) or payload.get("task") != config["task"]:
        raise RuntimeError("Low-light+Haze checkpoint payload identity changed")
    model.load_state_dict(payload["model"], strict=True)
    model = model.to(device).eval().requires_grad_(False)
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

    metric = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(device).eval()
    output = ROOT / config["output"] / "final_comparison"
    sidecars = output / "per_record"
    local_error = None
    try:
        for index in range(rank, len(rows), world):
            row = rows[index]
            lq_path, gt_path = ROOT / row["lq_path"], ROOT / row["gt_path"]
            if base.file_sha256(lq_path) != row["lq_sha256"] or base.file_sha256(gt_path) != row["gt_sha256"]:
                raise RuntimeError(f"frozen comparison input changed: {row['record_id']}")
            lq, gt = load_rgb(lq_path), load_rgb(gt_path)
            restored = restore_whole(
                model,
                lq,
                tile=int(config["data"]["tile_size"]),
                overlap=int(config["data"]["tile_overlap"]),
                device=device,
            )
            image_path = output / "images" / f"{row['record_id']}.png"
            save_png(image_path, restored)
            baseline_path = (
                ROOT
                / "checkpoints/foundir_composition_whole_image_full_pipeline_v1/per_record"
                / f"{row['record_id']}.json"
            )
            baseline = json.loads(baseline_path.read_text())
            foundir = baseline.get("full_foundir_generalist_specialist", {})
            if baseline.get("category") != CATEGORY or baseline.get("official_foundir_test_accessed") is not False:
                raise RuntimeError(f"Full FoundIR baseline identity changed: {row['record_id']}")
            values = {
                "psnr": full_psnr(gt, restored),
                "lpips": tiled_lpips(metric, gt, restored, device),
            }
            metric.reset()
            write_json(
                sidecars / f"{row['record_id']}.json",
                {
                    "schema_version": 1,
                    "status": "complete",
                    "record_id": row["record_id"],
                    "source_group": row["source_group"],
                    "category": CATEGORY,
                    "project_restormer": values,
                    "full_foundir": {"psnr": float(foundir["psnr"]), "lpips": float(foundir["lpips"])},
                    "output": image_path.relative_to(ROOT).as_posix(),
                    "output_sha256": base.file_sha256(image_path),
                    "official_foundir_test_accessed": False,
                },
            )
            print(f"FOUNDIR_LOWLIGHT_HAZE_FINAL rank={rank} record={index + 1}/{len(rows)}", flush=True)
    except Exception as error:
        local_error = f"{type(error).__name__}: {error}"
    base.raise_if_any_rank_failed(local_error, device=device, world=world, operation="Low-light+Haze final comparison")
    dist.barrier()
    if rank == 0:
        records = [json.loads((sidecars / f"{row['record_id']}.json").read_text()) for row in rows]
        project = {
            key: float(np.mean([item["project_restormer"][key] for item in records]))
            for key in ("psnr", "lpips")
        }
        foundir = {
            key: float(np.mean([item["full_foundir"][key] for item in records]))
            for key in ("psnr", "lpips")
        }
        result = {
            "schema_version": 1,
            "status": "foundir_lowlight_haze_restormer_final_comparison_complete",
            "config_sha256": base.file_sha256(config_path),
            "records": len(records),
            "selected_step": selection["selected_step"],
            "checkpoint": selection["checkpoint"],
            "checkpoint_sha256": selection["checkpoint_sha256"],
            "initialization": metadata,
            "project_restormer": project,
            "full_foundir": foundir,
            "project_minus_full_foundir": {
                "psnr_db": project["psnr"] - foundir["psnr"],
                "lpips": project["lpips"] - foundir["lpips"],
            },
            "beats_full_foundir_both_metrics": project["psnr"] > foundir["psnr"] and project["lpips"] < foundir["lpips"],
            "model_weight_updates": True,
            "development_pixels_accessed": True,
            "official_foundir_test_accessed": False,
        }
        write_json(output / "result.json", result)
        print("FOUNDIR_LOWLIGHT_HAZE_FINAL_COMPLETE " + json.dumps(result, sort_keys=True), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
