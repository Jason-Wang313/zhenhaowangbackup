#!/usr/bin/env python3
"""Measure the Z-Image VAE ceiling and frozen v3 teacher solver-step curve."""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party/diffsynth-studio"))

from lowlight_restoration.foundir_zimage_vae_ceiling import (  # noqa: E402
    load_aligned_pair,
    load_config,
    load_frozen_rows,
    sha256,
    teacher_checkpoint,
    validate_gpu_inventory,
    verify_model_artifacts,
    write_once,
)
from smoke.evaluate_foundir_blur_only_zimage_teacher_paired_flow_ddp import (  # noqa: E402
    build_pipe,
    lpips_value,
    psnr,
    restore_from_exact_lq_latent,
)
from diffsynth.core import load_state_dict  # noqa: E402


def save_ceiling_triptych(path: Path, record_id: str, lq: Image.Image, ceiling: Image.Image, gt: Image.Image) -> None:
    images = [image.convert("RGB") for image in (lq, ceiling, gt)]
    labels = ("Input", "GT VAE Roundtrip Ceiling", "Ground Truth")
    canvas = Image.new("RGB", (images[0].width * 3, images[0].height + 28), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (label, image) in enumerate(zip(labels, images)):
        x = index * image.width
        canvas.paste(image, (x, 28))
        draw.text((x + 6, 7), f"{record_id} | {label}", fill="black")
    buffer = io.BytesIO()
    canvas.save(buffer, format="PNG")
    write_once(path, buffer.getvalue())


def save_solver_triptych(
    path: Path, record_id: str, steps: int, lq: Image.Image, output: Image.Image, gt: Image.Image
) -> None:
    images = [image.convert("RGB") for image in (lq, output, gt)]
    labels = ("Input", f"Frozen Teacher {steps}-step", "Ground Truth")
    canvas = Image.new("RGB", (images[0].width * 3, images[0].height + 28), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (label, image) in enumerate(zip(labels, images)):
        x = index * image.width
        canvas.paste(image, (x, 28))
        draw.text((x + 6, 7), f"{record_id} | {label}", fill="black")
    buffer = io.BytesIO()
    canvas.save(buffer, format="PNG")
    write_once(path, buffer.getvalue())


@torch.no_grad()
def vae_roundtrip(pipe, image: Image.Image) -> Image.Image:
    pipe.load_models_to_device(["vae_encoder"])
    latent = pipe.vae_encoder(pipe.preprocess_image(image))
    pipe.load_models_to_device(["vae_decoder"])
    decoded = pipe.vae_decoder(latent)
    output = pipe.vae_output_to_image(decoded).convert("RGB")
    pipe.load_models_to_device([])
    return output


def aggregate(rows: list[dict[str, object]]) -> dict[str, float | int]:
    if not rows:
        raise RuntimeError("cannot aggregate an empty VAE ceiling result")
    fields = (
        "input_psnr",
        "input_lpips",
        "lq_roundtrip_psnr",
        "lq_roundtrip_lpips",
        "gt_roundtrip_psnr",
        "gt_roundtrip_lpips",
    )
    result: dict[str, float | int] = {"records": len(rows)}
    for field in fields:
        values = np.asarray([float(row[field]) for row in rows], dtype=np.float64)
        if not np.isfinite(values).all():
            raise RuntimeError(f"non-finite VAE ceiling metric: {field}")
        result[field] = float(np.mean(values))
    result["gt_roundtrip_psnr_headroom_db"] = float(
        np.mean([float(row["gt_roundtrip_psnr"]) - float(row["input_psnr"]) for row in rows])
    )
    result["gt_roundtrip_lpips_headroom"] = float(
        np.mean([float(row["input_lpips"]) - float(row["gt_roundtrip_lpips"]) for row in rows])
    )
    return result


def aggregate_solver(rows: list[dict[str, object]]) -> dict[str, float | int]:
    if not rows:
        raise RuntimeError("cannot aggregate an empty solver result")
    result: dict[str, float | int] = {"records": len(rows)}
    for field in ("input_psnr", "input_lpips", "output_psnr", "output_lpips"):
        values = np.asarray([float(row[field]) for row in rows], dtype=np.float64)
        if not np.isfinite(values).all():
            raise RuntimeError(f"non-finite solver metric: {field}")
        result[field] = float(np.mean(values))
    result["psnr_gain_db"] = float(
        np.mean([float(row["output_psnr"]) - float(row["input_psnr"]) for row in rows])
    )
    result["lpips_improvement"] = float(
        np.mean([float(row["input_lpips"]) - float(row["output_lpips"]) for row in rows])
    )
    result["psnr_improved_fraction"] = sum(
        float(row["output_psnr"]) > float(row["input_psnr"]) for row in rows
    ) / len(rows)
    result["lpips_improved_fraction"] = sum(
        float(row["output_lpips"]) < float(row["input_lpips"]) for row in rows
    ) / len(rows)
    return result


def gather_rows(local_rows, world: int, expected_record_ids: set[str]):
    gathered = [None] * world
    dist.all_gather_object(gathered, local_rows)
    combined = sorted([item for part in gathered for item in part], key=lambda item: item["record_id"])
    actual_record_ids = [str(item["record_id"]) for item in combined]
    if len(actual_record_ids) != len(expected_record_ids):
        raise RuntimeError("distributed diagnostic record count mismatch")
    if len(set(actual_record_ids)) != len(actual_record_ids):
        raise RuntimeError("distributed diagnostic contains duplicate records")
    if set(actual_record_ids) != expected_record_ids:
        raise RuntimeError("distributed diagnostic record identity mismatch")
    return combined


def best_solver_steps(snapshots: list[dict[str, object]]) -> dict[str, dict[str, int]]:
    result = {}
    for role in ("tiny_overfit_train", "visual_development"):
        role_snapshots = [snapshot for snapshot in snapshots if snapshot["role"] == role]
        if not role_snapshots:
            raise RuntimeError(f"missing solver snapshots for {role}")
        result[role] = {
            "best_psnr_step": int(max(role_snapshots, key=lambda item: item["metrics"]["output_psnr"])["steps"]),
            "best_lpips_step": int(min(role_snapshots, key=lambda item: item["metrics"]["output_lpips"])["steps"]),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = load_config(config_path)
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    if world != int(config["distributed"]["required_world_size"]):
        raise RuntimeError("diagnostic world size changed")
    verify_model_artifacts(config, large_hashes=rank == 0)
    checkpoint = teacher_checkpoint(config, verify_hash=rank == 0)
    dist.barrier()
    train_rows, visual_rows = load_frozen_rows(config, verify_images=True)
    names = [torch.cuda.get_device_properties(index).name for index in range(torch.cuda.device_count())]
    memory = [
        torch.cuda.get_device_properties(index).total_memory // (1024 * 1024)
        for index in range(torch.cuda.device_count())
    ]
    validate_gpu_inventory(names, memory, config)

    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

    metric = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(device).eval()
    pipe = build_pipe(config, device)
    output_root = ROOT / str(config["runtime"]["output_dir"])
    if rank == 0:
        output_root.mkdir(parents=True, exist_ok=True)
        if any(output_root.iterdir()):
            raise RuntimeError("diagnostic output is nonempty; inspect it instead of overwriting")
    dist.barrier()

    ceiling_snapshots = []
    for role, rows in (("tiny_overfit_train", train_rows), ("visual_development", visual_rows)):
        local_results = []
        for index in range(rank, len(rows), world):
            row = rows[index]
            lq, gt = load_aligned_pair(row, int(config["data"]["crop_size"]))
            lq_roundtrip = vae_roundtrip(pipe, lq)
            gt_roundtrip = vae_roundtrip(pipe, gt)
            metrics = {
                "record_id": row["record_id"],
                "input_psnr": psnr(gt, lq),
                "input_lpips": lpips_value(metric, gt, lq, device),
                "lq_roundtrip_psnr": psnr(gt, lq_roundtrip),
                "lq_roundtrip_lpips": lpips_value(metric, gt, lq_roundtrip, device),
                "gt_roundtrip_psnr": psnr(gt, gt_roundtrip),
                "gt_roundtrip_lpips": lpips_value(metric, gt, gt_roundtrip, device),
            }
            local_results.append(metrics)
            save_ceiling_triptych(
                output_root / "vae_ceiling" / role / f"{index:02d}_{row['record_id']}.png",
                row["record_id"],
                lq,
                gt_roundtrip,
                gt,
            )
        combined = gather_rows(local_results, world, {str(row["record_id"]) for row in rows})
        dist.barrier()
        if rank == 0:
            payload = {
                "role": role,
                "metrics": aggregate(combined),
                "rows": combined,
            }
            write_once(
                output_root / "vae_ceiling" / role / "metrics.json",
                (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode(),
            )
            ceiling_snapshots.append(payload)
            print("ZIMAGE_VAE_CEILING_SNAPSHOT " + json.dumps(payload, sort_keys=True), flush=True)

    state = load_state_dict(str(checkpoint), torch_dtype=torch.bfloat16, device=device)
    pipe.load_lora(pipe.dit, state_dict=state, hotload=True)
    solver_snapshots = []
    for steps in config["evaluation"]["teacher_inference_steps"]:
        steps = int(steps)
        for role, rows in (("tiny_overfit_train", train_rows), ("visual_development", visual_rows)):
            local_results = []
            for index in range(rank, len(rows), world):
                row = rows[index]
                lq, gt = load_aligned_pair(row, int(config["data"]["crop_size"]))
                restored = restore_from_exact_lq_latent(
                    pipe,
                    lq,
                    str(config["model"]["prompt"]),
                    steps=steps,
                    shift=float(config["teacher"]["scheduler_shift"]),
                )
                metrics = {
                    "record_id": row["record_id"],
                    "input_psnr": psnr(gt, lq),
                    "input_lpips": lpips_value(metric, gt, lq, device),
                    "output_psnr": psnr(gt, restored),
                    "output_lpips": lpips_value(metric, gt, restored, device),
                }
                local_results.append(metrics)
                save_solver_triptych(
                    output_root / "solver_sweep" / f"steps_{steps:02d}" / role / f"{index:02d}_{row['record_id']}.png",
                    row["record_id"],
                    steps,
                    lq,
                    restored,
                    gt,
                )
            combined = gather_rows(local_results, world, {str(row["record_id"]) for row in rows})
            dist.barrier()
            if rank == 0:
                payload = {"steps": steps, "role": role, "metrics": aggregate_solver(combined), "rows": combined}
                write_once(
                    output_root / "solver_sweep" / f"steps_{steps:02d}" / role / "metrics.json",
                    (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode(),
                )
                solver_snapshots.append(payload)
                print("ZIMAGE_SOLVER_SWEEP_SNAPSHOT " + json.dumps(payload, sort_keys=True), flush=True)
    pipe.clear_lora()

    if rank == 0:
        ceiling_checks = {}
        for snapshot in ceiling_snapshots:
            role = snapshot["role"]
            ceiling_checks[f"{role}_psnr_ceiling_above_input"] = (
                snapshot["metrics"]["gt_roundtrip_psnr_headroom_db"] > 0
            )
            ceiling_checks[f"{role}_lpips_ceiling_below_input"] = (
                snapshot["metrics"]["gt_roundtrip_lpips_headroom"] > 0
            )
        candidate_steps = [
            snapshot["steps"]
            for snapshot in solver_snapshots
            if snapshot["role"] == "tiny_overfit_train"
            and snapshot["metrics"]["psnr_gain_db"] > 0
            and snapshot["metrics"]["lpips_improvement"] > 0
        ]
        result = {
            "schema_version": 1,
            "status": "zimage_blur_vae_ceiling_solver_sweep_complete",
            "config": str(config_path.relative_to(ROOT)),
            "config_sha256": sha256(config_path),
            "teacher_checkpoint": str(checkpoint.relative_to(ROOT)),
            "teacher_checkpoint_sha256": sha256(checkpoint),
            "ceiling_snapshots": ceiling_snapshots,
            "solver_snapshots": solver_snapshots,
            "best_solver_steps": best_solver_steps(solver_snapshots),
            "ceiling_checks": ceiling_checks,
            "latent_ceiling_gate_pass": all(ceiling_checks.values()),
            "teacher_metric_candidate_steps": candidate_steps,
            "solver_candidate_exists": bool(candidate_steps),
            "diagnostic_only": True,
            "decoder_finetuned": False,
            "distillation_performed": False,
            "sealed_internal_accessed": False,
            "official_foundir_test_accessed": False,
        }
        write_once(output_root / "result.json", (json.dumps(result, indent=2, sort_keys=True) + "\n").encode())
        print("ZIMAGE_VAE_CEILING_SOLVER_SWEEP_COMPLETE " + json.dumps(result, sort_keys=True), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
