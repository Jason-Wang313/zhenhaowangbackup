#!/usr/bin/env python3
"""Evaluate the solver-aligned Z-Image paired-flow teacher continuation."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party/diffsynth-studio"))

from lowlight_restoration.foundir_zimage_paired_flow_v3 import (  # noqa: E402
    evaluation_checkpoint,
    load_aligned_pair,
    load_config,
    load_frozen_rows,
    parent_checkpoint,
    sha256,
    validate_gpu_inventory,
    verify_model_artifacts,
    write_once,
)
from smoke.evaluate_foundir_blur_only_zimage_teacher_paired_flow_ddp import (  # noqa: E402
    aggregate,
    build_pipe,
    lpips_value,
    psnr,
    restore_from_exact_lq_latent,
    save_triptych,
)
from diffsynth.core import load_state_dict  # noqa: E402


def verify_checkpoint(config, config_path: Path, cumulative_step: int) -> tuple[Path, str]:
    checkpoint = evaluation_checkpoint(config, cumulative_step)
    if not checkpoint.is_file():
        raise RuntimeError(f"missing v3 evaluation checkpoint: {checkpoint}")
    digest = sha256(checkpoint)
    if cumulative_step == int(config["training"]["initial_step"]):
        if digest != config["training"]["initial_checkpoint_sha256"]:
            raise RuntimeError("v3 parent checkpoint identity changed")
        return checkpoint, digest
    with safe_open(checkpoint, framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
        tensor_count = len(list(handle.keys()))
    expected = {
        "config_sha256": sha256(config_path),
        "step": str(cumulative_step),
        "initial_step": str(config["training"]["initial_step"]),
        "parent_sha256": str(config["training"]["initial_checkpoint_sha256"]),
        "optimizer_reset": "true",
    }
    if any(metadata.get(key) != value for key, value in expected.items()) or tensor_count != 476:
        raise RuntimeError("v3 checkpoint metadata or LoRA tensor count changed")
    return checkpoint, digest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = load_config(config_path)
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if world != int(config["distributed"]["required_world_size"]):
        raise RuntimeError("v3 evaluator world size changed")
    verify_model_artifacts(config, large_hashes=rank == 0)
    parent_checkpoint(config, verify_hash=rank == 0)
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
    output_root = ROOT / str(config["runtime"]["output_dir"]) / "evaluation"
    all_aggregates = []
    for step_value in config["inference"]["evaluate_cumulative_steps"]:
        step_value = int(step_value)
        checkpoint, checkpoint_digest = verify_checkpoint(config, config_path, step_value)
        state = load_state_dict(str(checkpoint), torch_dtype=torch.bfloat16, device=device)
        pipe.load_lora(pipe.dit, state_dict=state, hotload=True)
        for role, rows in (("tiny_overfit_train", train_rows), ("visual_development", visual_rows)):
            local_rows = []
            for index in range(rank, len(rows), world):
                row = rows[index]
                lq, gt = load_aligned_pair(row, int(config["data"]["crop_size"]))
                restored = restore_from_exact_lq_latent(
                    pipe,
                    lq,
                    str(config["model"]["prompt"]),
                    steps=int(config["inference"]["teacher_steps"]),
                    shift=float(config["inference"]["scheduler_shift"]),
                )
                row_metrics = {
                    "record_id": row["record_id"],
                    "role": role,
                    "step": step_value,
                    "input_psnr": psnr(gt, lq),
                    "output_psnr": psnr(gt, restored),
                    "input_lpips": lpips_value(metric, gt, lq, device),
                    "output_lpips": lpips_value(metric, gt, restored, device),
                }
                local_rows.append(row_metrics)
                save_triptych(
                    output_root / f"step_{step_value:06d}" / role / f"{index:02d}_{row['record_id']}.png",
                    row["record_id"],
                    lq,
                    restored,
                    gt,
                )
            gathered = [None] * world
            dist.all_gather_object(gathered, local_rows)
            dist.barrier()
            if rank == 0:
                combined = sorted(
                    [item for part in gathered for item in part], key=lambda item: item["record_id"]
                )
                if len(combined) != len(rows):
                    raise RuntimeError("distributed v3 evaluation record count mismatch")
                payload = {
                    "schema_version": 3,
                    "status": "zimage_paired_flow_teacher_v3_snapshot_complete",
                    "step": step_value,
                    "role": role,
                    "checkpoint": str(checkpoint.relative_to(ROOT)),
                    "checkpoint_sha256": checkpoint_digest,
                    "config_sha256": sha256(config_path),
                    "initialization": "exact_lq_vae_latent_at_sigma_1",
                    "scheduler_shift": float(config["inference"]["scheduler_shift"]),
                    "cfg_scale": 1.0,
                    "controlnet_loaded": False,
                    "metrics": aggregate(combined),
                    "rows": combined,
                    "sealed_internal_accessed": False,
                    "official_foundir_test_accessed": False,
                }
                write_once(
                    output_root / f"step_{step_value:06d}" / role / "metrics.json",
                    (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode(),
                )
                all_aggregates.append(
                    {key: payload[key] for key in ("step", "role", "checkpoint", "checkpoint_sha256", "metrics")}
                )
                print("ZIMAGE_PAIRED_FLOW_V3_SNAPSHOT " + json.dumps(all_aggregates[-1], sort_keys=True), flush=True)
        pipe.clear_lora()

    if rank == 0:
        final_step = int(config["inference"]["evaluate_cumulative_steps"][-1])
        initial_step = int(config["training"]["initial_step"])
        final_train = next(
            item for item in all_aggregates
            if item["step"] == final_step and item["role"] == "tiny_overfit_train"
        )
        parent_train = next(
            item for item in all_aggregates
            if item["step"] == initial_step and item["role"] == "tiny_overfit_train"
        )
        metric_checks = {
            "psnr_improved_over_input": final_train["metrics"]["psnr_gain_db"] > 0,
            "lpips_improved_over_input": final_train["metrics"]["lpips_improvement"] > 0,
        }
        result = {
            "schema_version": 3,
            "status": "zimage_blur_teacher_paired_flow_v3_evaluation_complete",
            "config": str(config_path.relative_to(ROOT)),
            "config_sha256": sha256(config_path),
            "conditioning": config["implementation"]["conditioning"],
            "initial_checkpoint": str(parent_checkpoint(config, verify_hash=False).relative_to(ROOT)),
            "initial_checkpoint_sha256": config["training"]["initial_checkpoint_sha256"],
            "optimizer_resume": False,
            "timestep_sampling": config["training"]["timestep_sampling"],
            "loss_weighting": config["training"]["loss_weighting"],
            "initialization": config["inference"]["initialization"],
            "scheduler_shift": float(config["inference"]["scheduler_shift"]),
            "cfg_scale": 1.0,
            "controlnet_loaded": False,
            "snapshots": all_aggregates,
            "parent_to_final_train": {
                "psnr_change_db": final_train["metrics"]["output_psnr"] - parent_train["metrics"]["output_psnr"],
                "lpips_change": final_train["metrics"]["output_lpips"] - parent_train["metrics"]["output_lpips"],
            },
            "metric_checks": metric_checks,
            "automatic_metric_gate_pass": all(metric_checks.values()),
            "human_visual_review_required": True,
            "human_visual_review_complete": False,
            "sealed_internal_accessed": False,
            "official_foundir_test_accessed": False,
        }
        write_once(
            output_root / "result.json",
            (json.dumps(result, indent=2, sort_keys=True) + "\n").encode(),
        )
        print("ZIMAGE_PAIRED_FLOW_V3_EVALUATION_COMPLETE " + json.dumps(result, sort_keys=True), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
