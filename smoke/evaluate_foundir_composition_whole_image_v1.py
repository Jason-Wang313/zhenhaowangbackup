#!/usr/bin/env python3
"""Evaluate the routed final system and released FoundIR on 660 whole images."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party/flux2/src"))

from lowlight_restoration.scalar_residual import quantized_scalar_residual  # noqa: E402
from lowlight_restoration.task_native_restormer import load_task_native_restormer  # noqa: E402
from smoke.evaluate_foundir_five_type_flux2_ddp import (  # noqa: E402
    synchronize_rank_zero_finalization,
    torchrun_worker_identity,
)
from smoke.evaluate_foundir_five_type_released_baselines_ddp import (  # noqa: E402
    released_config,
    run_generalist,
)
from smoke.evaluate_foundir_mixed_restoration_pilot_ddp import (  # noqa: E402
    expert_order,
    load_flux_expert,
    validate_config,
)
from smoke.evaluate_foundir_blur_only_flux2_full_development_ddp import restore_batch  # noqa: E402
from smoke.run_foundir_lowlight_gate2_teachers import (  # noqa: E402
    SPECIALIST_DEPENDENCY_DIRECTORIES,
    build_generalist,
)

TRUTH_COMPOSITION_SIZE = {
    "01Blur": 1, "05Noise": 1, "08Haze": 1, "10Rain": 1, "14Lowlight": 1,
    "02Blur_Noise": 2, "09Lowlight_Haze": 2, "13Rain_Haze": 2,
    "15Lowlight_Blur": 2, "16Lowlight_Noise": 2,
    "18Lowlight_Blur_Noise": 3,
}


def workspace(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_digest(*parts: object) -> str:
    return hashlib.sha256("\0".join(map(str, parts)).encode()).hexdigest()


def save_png(path: Path, pixels: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with Image.open(path) as handle:
            current = np.asarray(handle.convert("RGB"), dtype=np.uint8)
        if current.shape != pixels.shape or not np.array_equal(current, pixels):
            raise RuntimeError(f"existing whole-image output changed: {path}")
        return
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    Image.fromarray(pixels, mode="RGB").save(temporary, format="PNG")
    os.replace(temporary, path)


def write_json(path: Path, payload: Mapping[str, object]) -> None:
    data = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise RuntimeError(f"existing whole-image artifact changed: {path}")
        return
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def load_rgb(path: str | Path) -> np.ndarray:
    with Image.open(workspace(path)) as handle:
        return np.asarray(handle.convert("RGB"), dtype=np.uint8).copy()


def tensor_m11(pixels: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(pixels.copy()).permute(2, 0, 1).float().div_(127.5).sub_(1)


def starts(length: int, tile: int, overlap: int) -> list[int]:
    if length <= tile:
        return [0]
    values = list(range(0, length - tile + 1, tile - overlap))
    if values[-1] != length - tile:
        values.append(length - tile)
    return values


def tile_boxes(height: int, width: int, tile: int, overlap: int) -> list[tuple[int, int, int, int]]:
    return [
        (y, x, min(y + tile, height), min(x + tile, width))
        for y in starts(height, tile, overlap)
        for x in starts(width, tile, overlap)
    ]


def stage_image(output: Path, task: str, record_id: str) -> Path:
    return output / "ours" / "stages" / task / f"{record_id}.png"


def source_for_stage(row: Mapping[str, str], task: str, config: Mapping[str, object], output: Path) -> np.ndarray:
    order = expert_order(row, config)
    position = order.index(task)
    return load_rgb(row["lq_path"] if position == 0 else stage_image(output, order[position - 1], row["record_id"]))


def merge_ours_tiles(
    source_pixels: np.ndarray,
    *,
    task: str,
    row: Mapping[str, str],
    config: Mapping[str, object],
    expert: Mapping[str, object],
    device: torch.device,
    profile: str,
    loaded: object,
) -> np.ndarray:
    tile = int(config["whole_image"]["ours_tile_size"])
    overlap = int(config["whole_image"]["ours_overlap"])
    height, width = source_pixels.shape[:2]
    boxes = tile_boxes(height, width, tile, overlap)
    accumulator = np.zeros((height, width, 3), dtype=np.float32)
    counts = np.zeros((height, width, 1), dtype=np.float32)
    batch_size = int(config["distributed"]["batch_size_per_rank"][profile])
    for offset in range(0, len(boxes), batch_size):
        batch_boxes = boxes[offset : offset + batch_size]
        source = torch.stack([
            tensor_m11(source_pixels[y0:y1, x0:x1]) for y0, x0, y1, x1 in batch_boxes
        ]).to(device)
        if expert["kind"] == "flux2":
            model, autoencoder, cfg_context, cfg_ids, pack_image = loaded
            seeds = [
                int(stable_digest(config["inference"]["seed"], "whole", task, row["record_id"], y0, x0)[:16], 16)
                for y0, x0, _, _ in batch_boxes
            ]
            restored = restore_batch(
                model, autoencoder, source, cfg_context, cfg_ids, pack_image,
                seeds=seeds,
                steps=int(config["inference"]["num_steps"]),
                reference_time=int(config["inference"]["reference_time_coordinate"]),
            )
        else:
            with torch.inference_mode():
                restored = loaded((source + 1) * 0.5).float().clamp(0, 1).mul(2).sub(1)
        arrays, _ = quantized_scalar_residual(source, restored, float(expert["alpha"]))
        for box, array in zip(batch_boxes, arrays, strict=True):
            y0, x0, y1, x1 = box
            pixels = array.permute(1, 2, 0).float().cpu().numpy()
            accumulator[y0:y1, x0:x1] += pixels[: y1 - y0, : x1 - x0]
            counts[y0:y1, x0:x1] += 1
    if np.any(counts == 0):
        raise RuntimeError(f"tile coverage incomplete: {row['record_id']} {task}")
    return np.rint(accumulator / counts).clip(0, 255).astype(np.uint8)


def run_ours_stage(config: Mapping[str, object], rows: Sequence[Mapping[str, str]], task: str) -> None:
    rank, world, local_rank = torchrun_worker_identity()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    profile = next((name for name in config["distributed"]["allowed_gpu_name_fragments"] if name in torch.cuda.get_device_properties(device).name), None)
    if profile is None:
        raise RuntimeError("whole-image evaluation requires H100 or H200")
    output = workspace(config["runtime"]["output_dir"])
    selected = [row for row in rows if task in expert_order(row, config)]
    expert = config["experts"][task]
    if expert["kind"] == "flux2":
        loaded = load_flux_expert(task, expert, config, device)
    else:
        model, _ = load_task_native_restormer(expert["base_checkpoint_name"], workspace(expert["base_checkpoint_dir"]))
        payload = torch.load(workspace(expert["checkpoint"]), map_location="cpu", weights_only=False)
        if payload.get("task") != task or int(payload.get("step", -1)) != int(expert["checkpoint_step"]):
            raise RuntimeError(f"{task} checkpoint metadata changed")
        model.load_state_dict(payload["model"], strict=True)
        loaded = model.to(device).eval().requires_grad_(False)
    for index in range(rank, len(selected), world):
        row = selected[index]
        destination = stage_image(output, task, row["record_id"])
        if destination.exists():
            continue
        source = source_for_stage(row, task, config, output)
        result = merge_ours_tiles(source, task=task, row=row, config=config, expert=expert, device=device, profile=profile, loaded=loaded)
        save_png(destination, result)
        print(f"WHOLE_OURS task={task} rank={rank} record={index+1}/{len(selected)} tiles_done", flush=True)


def pad_minimum(tensor: torch.Tensor, minimum: int) -> tuple[torch.Tensor, int, int]:
    height, width = tensor.shape[-2:]
    pad_h, pad_w = max(0, minimum - height), max(0, minimum - width)
    if pad_h or pad_w:
        tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode="reflect")
    return tensor, height, width


def run_foundir(config: Mapping[str, object], rows: Sequence[Mapping[str, str]]) -> None:
    rank, world, local_rank = torchrun_worker_identity()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    extra = ROOT / ".tools/foundir-runtime"
    sys.path.insert(0, str(extra))
    for dependency in reversed(SPECIALIST_DEPENDENCY_DIRECTORIES):
        sys.path.insert(0, str(dependency))
    foundir_path = workspace(config["whole_image"]["released_foundir_config"])
    if sha256(foundir_path) != config["whole_image"]["released_foundir_config_sha256"]:
        raise RuntimeError("released FoundIR config changed")
    foundir = json.loads(foundir_path.read_text())
    model = build_generalist(released_config(foundir), device)
    output = workspace(config["runtime"]["output_dir"])
    tile, overlap = int(config["whole_image"]["foundir_tile_size"]), int(config["whole_image"]["foundir_overlap"])
    for index in range(rank, len(rows), world):
        row = rows[index]
        destination = output / "foundir" / "images" / f"{row['record_id']}.png"
        if destination.exists():
            continue
        pixels = load_rgb(row["lq_path"])
        source = torch.from_numpy(pixels.copy()).permute(2, 0, 1).float().div_(255).unsqueeze(0)
        original_h, original_w = pixels.shape[:2]
        if original_h * original_w <= tile * tile:
            result = run_generalist(model, source, device, row["record_id"])[0]
            merged = result.detach().cpu().permute(1, 2, 0).mul(255).numpy()
        else:
            padded, keep_h, keep_w = pad_minimum(source, tile)
            height, width = padded.shape[-2:]
            accumulator = np.zeros((height, width, 3), dtype=np.float32)
            counts = np.zeros((height, width, 1), dtype=np.float32)
            for y0, x0, y1, x1 in tile_boxes(height, width, tile, overlap):
                restored = run_generalist(model, padded[:, :, y0:y1, x0:x1], device, row["record_id"])[0]
                array = restored.detach().cpu().permute(1, 2, 0).mul(255).numpy()
                accumulator[y0:y1, x0:x1] += array
                counts[y0:y1, x0:x1] += 1
            merged = (accumulator / counts)[:keep_h, :keep_w]
        save_png(destination, np.rint(merged).clip(0, 255).astype(np.uint8))
        print(f"WHOLE_FOUNDIR rank={rank} record={index+1}/{len(rows)}", flush=True)


def full_psnr(reference: np.ndarray, prediction: np.ndarray) -> float:
    mse = float(np.mean((reference.astype(np.float64) - prediction.astype(np.float64)) ** 2))
    return float("inf") if mse == 0 else float(20 * np.log10(255.0) - 10 * np.log10(mse))


def tiled_lpips(metric, reference: np.ndarray, prediction: np.ndarray, device: torch.device) -> float:
    height, width = reference.shape[:2]
    weighted, area = 0.0, 0
    for y0 in range(0, height, 512):
        for x0 in range(0, width, 512):
            ref = tensor_m11(reference[y0:min(y0+512, height), x0:min(x0+512, width)]).unsqueeze(0)
            pred = tensor_m11(prediction[y0:min(y0+512, height), x0:min(x0+512, width)]).unsqueeze(0)
            tile_area = ref.shape[-2] * ref.shape[-1]
            pad_h, pad_w = max(0, 64-ref.shape[-2]), max(0, 64-ref.shape[-1])
            if pad_h or pad_w:
                ref, pred = F.pad(ref, (0,pad_w,0,pad_h), mode="replicate"), F.pad(pred, (0,pad_w,0,pad_h), mode="replicate")
            with torch.inference_mode():
                value = float(metric((pred.to(device)+1)*0.5, (ref.to(device)+1)*0.5).item())
            metric.reset()
            weighted += value * tile_area
            area += tile_area
    return weighted / area


def run_metrics(config: Mapping[str, object], rows: Sequence[Mapping[str, str]], config_sha: str) -> None:
    rank, world, local_rank = torchrun_worker_identity()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    metric = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(device).eval()
    output = workspace(config["runtime"]["output_dir"])
    sidecars = output / "per_record"
    for index in range(rank, len(rows), world):
        row = rows[index]
        destination = sidecars / f"{row['record_id']}.json"
        if destination.exists():
            continue
        gt = load_rgb(row["gt_path"])
        order = expert_order(row, config)
        ours = (
            load_rgb(stage_image(output, order[-1], row["record_id"]))
            if order else load_rgb(row["lq_path"])
        )
        foundir = load_rgb(output / "foundir" / "images" / f"{row['record_id']}.png")
        if ours.shape != gt.shape or foundir.shape != gt.shape:
            raise RuntimeError(f"whole-image output shape mismatch: {row['record_id']}")
        values = {
            "ours": {"psnr": full_psnr(gt, ours), "lpips": tiled_lpips(metric, gt, ours, device)},
            "foundir": {"psnr": full_psnr(gt, foundir), "lpips": tiled_lpips(metric, gt, foundir, device)},
        }
        write_json(destination, {
            "schema_version": 1, "status": "complete", "config_sha256": config_sha,
            "record_id": row["record_id"], "source_group": row["source_group"],
            "category": row["category"], "composition_size": TRUTH_COMPOSITION_SIZE[row["category"]],
            "expert_order": order, "routing_status": row["routing_status"], **values,
            "official_foundir_test_accessed": False,
        })
        print(f"WHOLE_METRICS rank={rank} record={index+1}/{len(rows)}", flush=True)

    def finalize() -> None:
        payloads = [json.loads((sidecars / f"{row['record_id']}.json").read_text()) for row in rows]
        grouped: dict[int, list[dict]] = defaultdict(list)
        category: dict[str, list[dict]] = defaultdict(list)
        for payload in payloads:
            grouped[int(payload["composition_size"])].append(payload)
            category[payload["category"]].append(payload)
        def summary(values: Sequence[Mapping[str, object]]) -> dict[str, object]:
            result = {"records": len(values)}
            for role in ("foundir", "ours"):
                result[role] = {metric_name: float(np.mean([item[role][metric_name] for item in values])) for metric_name in ("psnr", "lpips")}
            result["ours_minus_foundir"] = {
                "psnr_db": result["ours"]["psnr"] - result["foundir"]["psnr"],
                "lpips": result["ours"]["lpips"] - result["foundir"]["lpips"],
            }
            result["ours_win_fraction"] = {
                "psnr": float(np.mean([item["ours"]["psnr"] > item["foundir"]["psnr"] for item in values])),
                "lpips": float(np.mean([item["ours"]["lpips"] < item["foundir"]["lpips"] for item in values])),
            }
            return result
        result = {
            "schema_version": 1, "status": "foundir_composition_whole_image_complete",
            "config_sha256": config_sha, "records": len(payloads),
            "composition_metrics": {str(key): summary(values) for key, values in sorted(grouped.items())},
            "category_metrics": {key: summary(values) for key, values in sorted(category.items())},
            "metric_protocol": config["whole_image"], "model_weight_updates": False,
            "development_pixels_accessed": True, "official_foundir_test_accessed": False,
        }
        write_json(output / "result.json", result)
        print("FOUNDIR_COMPOSITION_WHOLE_IMAGE_COMPLETE " + json.dumps(result["composition_metrics"], sort_keys=True), flush=True)
    synchronize_rank_zero_finalization(
        finalize, rank=rank, world=world, output=output,
        identity={"config_sha256": config_sha, "records": len(rows)},
        phase="whole-image metrics", timeout_seconds=3600,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mode", choices=["preflight", "foundir", "metrics", "ours"] , required=True)
    parser.add_argument("--task", choices=["rain", "blur_noise", "noise", "haze", "low_light", "blur"])
    args = parser.parse_args()
    config_path = workspace(args.config)
    config, rows = validate_config(config_path, verify_images=args.mode == "preflight")
    if config.get("whole_image", {}).get("ours_tile_size") != 512 or len(rows) != 660:
        raise RuntimeError("whole-image comparison contract changed")
    if args.mode == "preflight":
        print(json.dumps({"status":"FOUNDIR_COMPOSITION_WHOLE_IMAGE_PREFLIGHT_PASS","records":len(rows),"official_foundir_test_accessed":False}, sort_keys=True))
    elif args.mode == "ours":
        if args.task is None:
            raise RuntimeError("--task is required for ours mode")
        run_ours_stage(config, rows, args.task)
    elif args.mode == "foundir":
        run_foundir(config, rows)
    else:
        run_metrics(config, rows, sha256(config_path))


if __name__ == "__main__":
    main()
