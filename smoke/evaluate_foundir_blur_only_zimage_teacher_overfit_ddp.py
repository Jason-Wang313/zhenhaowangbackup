#!/usr/bin/env python3
"""Evaluate initial/final Z-Image Blur teacher states on fixed aligned crops."""

from __future__ import annotations

import argparse
import io
import json
import math
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

from lowlight_restoration.foundir_zimage import (  # noqa: E402
    load_aligned_pair,
    load_config,
    load_frozen_rows,
    model_paths,
    sha256,
    validate_gpu_inventory,
    verify_model_artifacts,
    write_once,
)
from diffsynth.core import ModelConfig, load_state_dict  # noqa: E402
from diffsynth.diffusion.base_pipeline import ControlNetInput  # noqa: E402
from diffsynth.pipelines.z_image import ZImagePipeline  # noqa: E402


def image_tensor(image: Image.Image, device: torch.device) -> torch.Tensor:
    array = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.float32).div_(255)


def psnr(reference: Image.Image, prediction: Image.Image) -> float:
    ref = np.asarray(reference.convert("RGB"), dtype=np.float64) / 255.0
    pred = np.asarray(prediction.convert("RGB"), dtype=np.float64) / 255.0
    mse = float(np.mean((ref - pred) ** 2))
    return 100.0 if mse <= 1e-12 else -10.0 * math.log10(mse)


def lpips_value(metric, reference: Image.Image, prediction: Image.Image, device: torch.device) -> float:
    with torch.inference_mode():
        return float(metric(image_tensor(prediction, device), image_tensor(reference, device)).item())


def save_triptych(path: Path, record_id: str, lq: Image.Image, output: Image.Image, gt: Image.Image) -> None:
    images = [image.convert("RGB") for image in (lq, output, gt)]
    labels = ("Input", "Z-Image Output", "Ground Truth")
    canvas = Image.new("RGB", (images[0].width * 3, images[0].height + 28), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (label, image) in enumerate(zip(labels, images)):
        x = index * image.width
        canvas.paste(image, (x, 28))
        draw.text((x + 6, 7), f"{record_id} | {label}", fill="black")
    buffer = io.BytesIO()
    canvas.save(buffer, format="PNG")
    write_once(path, buffer.getvalue())


def aggregate(rows):
    return {
        "records": len(rows),
        "input_psnr": float(np.mean([row["input_psnr"] for row in rows])),
        "output_psnr": float(np.mean([row["output_psnr"] for row in rows])),
        "psnr_gain_db": float(np.mean([row["output_psnr"] - row["input_psnr"] for row in rows])),
        "input_lpips": float(np.mean([row["input_lpips"] for row in rows])),
        "output_lpips": float(np.mean([row["output_lpips"] for row in rows])),
        "lpips_improvement": float(np.mean([row["input_lpips"] - row["output_lpips"] for row in rows])),
        "psnr_improved_fraction": sum(row["output_psnr"] > row["input_psnr"] for row in rows) / len(rows),
        "lpips_improved_fraction": sum(row["output_lpips"] < row["input_lpips"] for row in rows) / len(rows),
    }


def build_pipe(config, device):
    paths = model_paths(config)
    pipe = ZImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=[
            ModelConfig(path=[str(path) for path in paths["transformer"]]),
            ModelConfig(path=[str(path) for path in paths["text_encoder"]]),
            ModelConfig(path=str(paths["vae"])),
            ModelConfig(path=str(paths["controlnet"])),
        ],
        tokenizer_config=ModelConfig(path=str(paths["tokenizer"])),
        enable_npu_patch=False,
    )
    pipe.dit = pipe.enable_lora_hot_loading(pipe.dit)
    return pipe


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
    verify_model_artifacts(config, large_hashes=rank == 0)
    dist.barrier()
    train_rows, visual_rows = load_frozen_rows(config, verify_images=True)
    names = [torch.cuda.get_device_properties(index).name for index in range(torch.cuda.device_count())]
    memory = [torch.cuda.get_device_properties(index).total_memory // (1024 * 1024) for index in range(torch.cuda.device_count())]
    validate_gpu_inventory(names, memory, config)

    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

    metric = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(device).eval()
    pipe = build_pipe(config, device)
    output_root = ROOT / str(config["runtime"]["output_dir"]) / "evaluation"
    control_scale = float(config["model"]["controlnet"]["control_scale"])
    all_aggregates = []
    for step in config["inference"]["evaluate_steps"]:
        step = int(step)
        checkpoint = None
        if step > 0:
            checkpoint = ROOT / str(config["runtime"]["output_dir"]) / f"step_{step:06d}.safetensors"
            if not checkpoint.is_file():
                raise RuntimeError(f"missing teacher checkpoint: {checkpoint}")
            state = load_state_dict(str(checkpoint), torch_dtype=torch.bfloat16, device=device)
            pipe.load_lora(pipe.dit, state_dict=state, hotload=True)
        for role, rows in (("tiny_overfit_train", train_rows), ("visual_development", visual_rows)):
            local_rows = []
            for index in range(rank, len(rows), world):
                row = rows[index]
                lq, gt = load_aligned_pair(row, int(config["data"]["crop_size"]))
                restored = pipe(
                    prompt=str(config["model"]["prompt"]),
                    negative_prompt="",
                    cfg_scale=float(config["inference"]["cfg_scale"]),
                    height=lq.height,
                    width=lq.width,
                    seed=int(config["inference"]["seed"]) + index,
                    rand_device="cuda",
                    num_inference_steps=int(config["inference"]["teacher_steps"]),
                    controlnet_inputs=[ControlNetInput(image=lq, scale=control_scale)],
                ).convert("RGB")
                row_metrics = {
                    "record_id": row["record_id"],
                    "role": role,
                    "step": step,
                    "input_psnr": psnr(gt, lq),
                    "output_psnr": psnr(gt, restored),
                    "input_lpips": lpips_value(metric, gt, lq, device),
                    "output_lpips": lpips_value(metric, gt, restored, device),
                }
                local_rows.append(row_metrics)
                save_triptych(
                    output_root / f"step_{step:06d}" / role / f"{index:02d}_{row['record_id']}.png",
                    row["record_id"], lq, restored, gt,
                )
            gathered = [None] * world
            dist.all_gather_object(gathered, local_rows)
            dist.barrier()
            if rank == 0:
                combined = sorted(
                    [item for part in gathered for item in part], key=lambda item: item["record_id"]
                )
                if len(combined) != len(rows):
                    raise RuntimeError("distributed evaluation record count mismatch")
                payload = {
                    "schema_version": 1,
                    "status": "zimage_teacher_snapshot_complete",
                    "step": step,
                    "role": role,
                    "checkpoint": None if checkpoint is None else str(checkpoint.relative_to(ROOT)),
                    "checkpoint_sha256": None if checkpoint is None else sha256(checkpoint),
                    "config_sha256": sha256(config_path),
                    "metrics": aggregate(combined),
                    "rows": combined,
                    "sealed_internal_accessed": False,
                    "official_foundir_test_accessed": False,
                }
                write_once(
                    output_root / f"step_{step:06d}" / role / "metrics.json",
                    (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode(),
                )
                all_aggregates.append({key: payload[key] for key in ("step", "role", "checkpoint", "checkpoint_sha256", "metrics")})
                print("ZIMAGE_TEACHER_SNAPSHOT " + json.dumps(all_aggregates[-1], sort_keys=True), flush=True)
        if step > 0:
            pipe.clear_lora()

    if rank == 0:
        final_train = next(item for item in all_aggregates if item["step"] == 400 and item["role"] == "tiny_overfit_train")
        metric_checks = {
            "psnr_improved_over_input": final_train["metrics"]["psnr_gain_db"] > 0,
            "lpips_improved_over_input": final_train["metrics"]["lpips_improvement"] > 0,
        }
        result = {
            "schema_version": 1,
            "status": "zimage_blur_teacher_tiny_overfit_evaluation_complete",
            "config": str(config_path.relative_to(ROOT)),
            "config_sha256": sha256(config_path),
            "snapshots": all_aggregates,
            "metric_checks": metric_checks,
            "automatic_metric_gate_pass": all(metric_checks.values()),
            "human_visual_review_required": True,
            "human_visual_review_complete": False,
            "sealed_internal_accessed": False,
            "official_foundir_test_accessed": False,
        }
        write_once(output_root / "result.json", (json.dumps(result, indent=2, sort_keys=True) + "\n").encode())
        print("ZIMAGE_TEACHER_EVALUATION_COMPLETE " + json.dumps(result, sort_keys=True), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
