#!/usr/bin/env python3
"""Evaluate selected FLUX checkpoint at native resolution with fixed tiled blending."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as functional
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party/flux2/src"))

from lowlight_restoration.flux_prompt_cache import PromptContextCache, file_sha256  # noqa: E402
from lowlight_restoration.foundir_five_type import FIVE_TYPE_CATEGORIES  # noqa: E402
from lowlight_restoration.foundir_flux2_protocol import (  # noqa: E402
    evaluation_artifact_stem,
    resolve_workspace_path,
    stable_digest,
)
from lowlight_restoration.lora import inject_lora, load_lora_checkpoint  # noqa: E402
from lowlight_restoration.paired_development import rgb_structural_similarity  # noqa: E402
from smoke.evaluate_foundir_five_type_flux2_ddp import (  # noqa: E402
    aggregate,
    lpips_value,
    quantized_rgb,
    selected_from_contract,
    write_once,
)
from smoke.train_foundir_five_type_flux2_full_ddp import load_rows  # noqa: E402
from smoke.train_foundir_five_type_flux2_overfit_ddp import psnr, restore  # noqa: E402


def tile_starts(length: int, tile_size: int, overlap: int) -> list[int]:
    if length <= 0 or tile_size <= 0 or not 0 <= overlap < tile_size:
        raise ValueError("invalid tiled-inference dimensions")
    if length <= tile_size:
        return [0]
    stride = tile_size - overlap
    intervals = math.ceil((length - tile_size) / stride)
    return [round(index * (length - tile_size) / intervals) for index in range(intervals + 1)]


def axis_window(length: int, overlap: int, *, at_start: bool, at_end: bool) -> np.ndarray:
    window = np.ones(length, dtype=np.float32)
    ramp_length = min(overlap, length // 2)
    if ramp_length:
        ramp = np.sin(np.linspace(0.0, math.pi / 2.0, ramp_length, dtype=np.float32)) ** 2
        if not at_start:
            window[:ramp_length] = ramp
        if not at_end:
            window[-ramp_length:] = ramp[::-1]
    return window


def tiled_restore(
    source: np.ndarray,
    process_tile,
    *,
    tile_size: int,
    overlap: int,
) -> tuple[np.ndarray, dict[str, object]]:
    height, width = source.shape[:2]
    ys = tile_starts(height, tile_size, overlap)
    xs = tile_starts(width, tile_size, overlap)
    accumulation = np.zeros((height, width, 3), dtype=np.float64)
    weights = np.zeros((height, width, 1), dtype=np.float64)
    for y in ys:
        for x in xs:
            tile = source[y : min(y + tile_size, height), x : min(x + tile_size, width)]
            restored = process_tile(tile, x, y)
            if restored.shape != tile.shape:
                raise RuntimeError("tiled FLUX inference changed tile dimensions")
            wy = axis_window(tile.shape[0], overlap, at_start=y == 0, at_end=y + tile.shape[0] == height)
            wx = axis_window(tile.shape[1], overlap, at_start=x == 0, at_end=x + tile.shape[1] == width)
            window = (wy[:, None] * wx[None, :])[:, :, None]
            accumulation[y : y + tile.shape[0], x : x + tile.shape[1]] += restored * window
            weights[y : y + tile.shape[0], x : x + tile.shape[1]] += window
    if np.any(weights <= 0):
        raise RuntimeError("tiled blending left uncovered native-resolution pixels")
    return (accumulation / weights).astype(np.float32), {
        "tile_count": len(ys) * len(xs),
        "x_starts": xs,
        "y_starts": ys,
    }


def seam_error_ratio(prediction: np.ndarray, target: np.ndarray, metadata: dict[str, object], band: int) -> float:
    error = np.mean(np.abs(prediction.astype(np.float32) - target.astype(np.float32)), axis=2)
    mask = np.zeros(error.shape, dtype=bool)
    for x in metadata["x_starts"][1:]:
        mask[:, max(0, x - band) : min(error.shape[1], x + band + 1)] = True
    for y in metadata["y_starts"][1:]:
        mask[max(0, y - band) : min(error.shape[0], y + band + 1), :] = True
    if not mask.any() or (~mask).sum() == 0:
        return 1.0
    return float(error[mask].mean() / max(1e-8, float(error[~mask].mean())))


def load_native(row: dict[str, str]) -> tuple[np.ndarray, np.ndarray]:
    paths = [resolve_workspace_path(ROOT, row[key]) for key in ("lq_path", "gt_path")]
    if file_sha256(paths[0]) != row["lq_sha256"] or file_sha256(paths[1]) != row["gt_sha256"]:
        raise RuntimeError(f"native evaluation pair changed: {row['record_id']}")
    arrays = []
    for path in paths:
        with Image.open(path) as handle:
            arrays.append(np.asarray(handle.convert("RGB"), dtype=np.uint8).copy())
    if arrays[0].shape != arrays[1].shape:
        raise RuntimeError(f"native pair is unaligned: {row['record_id']}")
    return arrays[0], arrays[1]


def verify_sidecar(
    path: Path,
    *,
    record_id: str,
    config_sha: str,
    checkpoint_sha: str,
    conditions_sha: str,
) -> dict[str, object]:
    payload = json.loads(path.read_text())
    if (
        payload.get("status") != "complete"
        or payload.get("record_id") != record_id
        or payload.get("config_sha256") != config_sha
        or payload.get("checkpoint_sha256") != checkpoint_sha
        or payload.get("conditions_sha256") != conditions_sha
        or payload.get("protected_foundir_test_accessed") is not False
    ):
        raise RuntimeError(f"native-resolution sidecar identity changed: {path}")
    output = resolve_workspace_path(ROOT, payload["output_path"])
    if file_sha256(output) != payload["output_sha256"]:
        raise RuntimeError(f"native-resolution output changed: {output}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.config = resolve_workspace_path(ROOT, args.config)
    args.checkpoint = resolve_workspace_path(ROOT, args.checkpoint)
    args.selection = resolve_workspace_path(ROOT, args.selection)
    args.output = resolve_workspace_path(ROOT, args.output)
    config = json.loads(args.config.read_text())
    config_sha = file_sha256(args.config)
    checkpoint_sha = file_sha256(args.checkpoint)
    selection = json.loads(args.selection.read_text())
    if (
        selection.get("status") != "frozen_development_selected_checkpoint"
        or selection.get("config_sha256") != config_sha
        or selection.get("checkpoint_sha256") != checkpoint_sha
        or selection.get("sealed_internal_test_accessed") is not False
    ):
        raise RuntimeError("native-resolution evaluation requires the development-selected checkpoint")
    conditions = resolve_workspace_path(ROOT, config["full_conditions"]["csv"])
    conditions_sha = file_sha256(conditions)
    rows = load_rows(conditions, int(config["data"]["expected_five_type_records"]))
    development = selected_from_contract(
        config, rows, split="validation", config_sha=config_sha, conditions_sha=conditions_sha
    )
    quota = int(config["evaluation"]["full_resolution_records_per_category"])
    selected = []
    for category in sorted(FIVE_TYPE_CATEGORIES):
        candidates = sorted(
            (row for row in development if row["category"] == category),
            key=lambda row: stable_digest(int(config["data"]["split_seed"]), "native", row["record_id"]),
        )
        selected.extend(candidates[:quota])
    if len(selected) != quota * len(FIVE_TYPE_CATEGORIES):
        raise RuntimeError("native-resolution development subset cardinality changed")

    dist.init_process_group(backend="nccl")
    try:
        rank, world, local_rank = dist.get_rank(), dist.get_world_size(), int(os.environ["LOCAL_RANK"])
        if world != 8 or torch.cuda.device_count() != 8:
            raise RuntimeError("native-resolution FLUX evaluation requires exactly eight GPUs")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        if not any(name in torch.cuda.get_device_properties(device).name for name in ("H100", "H200")):
            raise RuntimeError("native-resolution evaluator requires H100 or H200")
        os.environ.update(
            HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
            KLEIN_4B_BASE_MODEL_PATH=str(resolve_workspace_path(ROOT, config["model"]["flow_weights"])),
            AE_MODEL_PATH=str(resolve_workspace_path(ROOT, config["model"]["autoencoder_weights"])),
        )
        from flux2.sampling import batched_prc_img
        from flux2.util import load_ae, load_flow_model
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

        cache = PromptContextCache(
            resolve_workspace_path(ROOT, config["prompt_cache"]["index"]),
            maximum_resident=int(config["prompt_cache"]["maximum_resident_contexts_per_rank"]),
        )
        autoencoder = load_ae(config["model"]["name"], device=device).eval().requires_grad_(False)
        model = load_flow_model(config["model"]["name"], device=device).eval().requires_grad_(False)
        names = inject_lora(model, list(config["lora"]["target_patterns"]), rank=32, alpha=32.0, dropout=0.0)
        if len(names) != int(config["lora"]["expected_module_count"]):
            raise RuntimeError("native-resolution LoRA module count changed")
        load_lora_checkpoint(model, args.checkpoint)
        lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(device).eval()
        images = args.output / "images"
        sidecars = args.output / "per_record"
        images.mkdir(parents=True, exist_ok=True)
        sidecars.mkdir(parents=True, exist_ok=True)
        tile_size = int(config["evaluation"]["tile_size"])
        overlap = int(config["evaluation"]["tile_overlap"])
        for index in range(rank, len(selected), world):
            row = selected[index]
            stem = evaluation_artifact_stem(row["record_id"])
            sidecar = sidecars / f"{stem}.json"
            if sidecar.exists():
                verify_sidecar(
                    sidecar,
                    record_id=row["record_id"],
                    config_sha=config_sha,
                    checkpoint_sha=checkpoint_sha,
                    conditions_sha=conditions_sha,
                )
                continue
            lq, gt = load_native(row)
            context, context_ids = cache.get(row["qwen_text"], device)

            def process_tile(tile: np.ndarray, x: int, y: int) -> np.ndarray:
                height, width = tile.shape[:2]
                tensor = torch.from_numpy(tile.copy()).permute(2, 0, 1).float().div_(127.5).sub_(1).unsqueeze(0)
                if height != tile_size or width != tile_size:
                    tensor = functional.pad(
                        tensor,
                        (0, tile_size - width, 0, tile_size - height),
                        mode="reflect" if height > tile_size - height and width > tile_size - width else "replicate",
                    )
                prediction = restore(
                    model,
                    autoencoder,
                    tensor.to(device),
                    context,
                    context_ids,
                    batched_prc_img,
                    seed=int(stable_digest(int(config["flow_training"]["seed"]), row["record_id"], x, y)[:16], 16),
                    steps=int(config["evaluation"]["inference_steps"]),
                    noise_scale=float(config["flow_training"]["anchor_noise_scale"]),
                    reference_time=int(config["flow_training"]["reference_time_coordinate"]),
                )[:, :, :height, :width]
                return (
                    ((prediction[0].detach().float().cpu() + 1.0) * 127.5)
                    .permute(1, 2, 0)
                    .numpy()
                    .clip(0, 255)
                    .astype(np.float32)
                )

            prediction_float, metadata = tiled_restore(
                lq, process_tile, tile_size=tile_size, overlap=overlap
            )
            prediction = np.rint(np.clip(prediction_float, 0, 255)).astype(np.uint8)
            output_path = images / f"{stem}.png"
            if output_path.exists():
                raise RuntimeError(f"orphan native output exists: {output_path}")
            Image.fromarray(prediction).save(output_path)
            target_t = torch.from_numpy(gt.copy()).permute(2, 0, 1).float().div_(127.5).sub_(1).unsqueeze(0)
            prediction_t = torch.from_numpy(prediction.copy()).permute(2, 0, 1).float().div_(127.5).sub_(1).unsqueeze(0)
            maximum = max(gt.shape[:2])
            if maximum > 1024:
                scale = 1024 / maximum
                size = (max(1, round(gt.shape[0] * scale)), max(1, round(gt.shape[1] * scale)))
                lpips_target = functional.interpolate(target_t, size=size, mode="bilinear", align_corners=False)
                lpips_prediction = functional.interpolate(prediction_t, size=size, mode="bilinear", align_corners=False)
            else:
                lpips_target, lpips_prediction = target_t, prediction_t
            result = {
                "status": "complete",
                "record_id": row["record_id"],
                "source_group": row["source_group"],
                "category": row["category"],
                "composition_size": int(row["composition_size"]),
                "evaluation_fold": int(row["evaluation_fold"]),
                "native_width": int(gt.shape[1]),
                "native_height": int(gt.shape[0]),
                "tile_count": int(metadata["tile_count"]),
                "psnr": psnr(target_t.to(device), prediction_t.to(device)),
                "lpips": lpips_value(lpips_metric, lpips_target, lpips_prediction, device),
                "ssim": rgb_structural_similarity(gt.astype(np.float32) / 255, prediction.astype(np.float32) / 255),
                "seam_error_ratio": seam_error_ratio(
                    prediction, gt, metadata, int(config["evaluation"]["seam_band_width"])
                ),
                "config_sha256": config_sha,
                "checkpoint_sha256": checkpoint_sha,
                "conditions_sha256": conditions_sha,
                "prompt_sha256": hashlib.sha256(row["qwen_text"].encode()).hexdigest(),
                "output_path": output_path.relative_to(ROOT).as_posix(),
                "output_sha256": file_sha256(output_path),
                "protected_foundir_test_accessed": False,
            }
            write_once(sidecar, (json.dumps(result, indent=2, sort_keys=True) + "\n").encode())
            print(f"FOUNDIR_FLUX_NATIVE rank={rank} record={index+1}/{len(selected)} tiles={metadata['tile_count']}", flush=True)
        dist.barrier()
        if rank == 0:
            result_rows = [
                verify_sidecar(
                    sidecars / f"{evaluation_artifact_stem(row['record_id'])}.json",
                    record_id=row["record_id"],
                    config_sha=config_sha,
                    checkpoint_sha=checkpoint_sha,
                    conditions_sha=conditions_sha,
                )
                for row in selected
            ]
            payload = {
                "schema_version": 1,
                "status": "native_resolution_development_complete",
                "config_sha256": config_sha,
                "checkpoint_sha256": checkpoint_sha,
                "tile_size": tile_size,
                "tile_overlap": overlap,
                "summary": aggregate(result_rows, config=config),
                "mean_seam_error_ratio": float(np.mean([row["seam_error_ratio"] for row in result_rows])),
                "records": len(result_rows),
                "protected_foundir_test_accessed": False,
            }
            write_once(args.output / "result.json", (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode())
            print("FOUNDIR_FLUX_NATIVE_PASS " + json.dumps(payload, sort_keys=True), flush=True)
        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
