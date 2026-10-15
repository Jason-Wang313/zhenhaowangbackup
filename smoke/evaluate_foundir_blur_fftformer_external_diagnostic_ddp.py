#!/usr/bin/env python3
"""Bounded FFTformer initialization diagnostic on routed official Blur pairs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_config(path: Path) -> tuple[dict, str]:
    config = json.loads(path.read_text())
    if config["status"] != "external_initialization_diagnostic_only_not_deployable":
        raise RuntimeError("FFTformer diagnostic deployability policy changed")
    if config["provenance"]["external_checkpoint_is_finally_deployable"] is not False:
        raise RuntimeError("external FFTformer checkpoint must remain diagnostic-only")
    if config["policy"]["official_gt_used_for_training"] is not False:
        raise RuntimeError("official GT must never be used for training")
    for field, sha_field in (
        ("architecture", "architecture_sha256"),
        ("checkpoint", "checkpoint_sha256"),
    ):
        artifact = resolve(config["provenance"][field])
        if sha256(artifact) != config["provenance"][sha_field]:
            raise RuntimeError(f"pinned FFTformer {field} changed")
    manifest = resolve(config["data"]["manifest"])
    if sha256(manifest) != config["data"]["manifest_sha256"]:
        raise RuntimeError("official manifest changed")
    return config, sha256(path)


def load_rows(config: dict) -> list[dict[str, str]]:
    with resolve(config["data"]["manifest"]).open(newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["category"] == config["data"]["category"]]
    with resolve(config["qwen"]["router_decisions"]).open(newline="") as handle:
        decisions = {row["record_id"]: row for row in csv.DictReader(handle)}
    if len(rows) != int(config["data"]["records"]):
        raise RuntimeError("official Blur record count changed")
    for row in rows:
        decision = decisions.get(row["record_id"])
        if decision is None or decision["status"] != config["qwen"]["required_status"]:
            raise RuntimeError(f"frozen Qwen route is not correct: {row['record_id']}")
        if decision["expert_plan"] != config["qwen"]["required_expert_plan"]:
            raise RuntimeError(f"frozen Qwen plan is not Blur: {row['record_id']}")
    return rows


def load_model(config: dict, device: torch.device) -> torch.nn.Module:
    architecture = resolve(config["provenance"]["architecture"])
    spec = importlib.util.spec_from_file_location("pinned_fftformer_arch", architecture)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import pinned FFTformer architecture")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.fftformer()
    state = torch.load(resolve(config["provenance"]["checkpoint"]), map_location="cpu", weights_only=True)
    if type(state).__name__ != "OrderedDict" or len(state) != 535:
        raise RuntimeError("FFTformer checkpoint tensor contract changed")
    if sum(tensor.numel() for tensor in state.values()) != 16_560_474:
        raise RuntimeError("FFTformer checkpoint parameter contract changed")
    model.load_state_dict(state, strict=True)
    return model.to(device).eval().requires_grad_(False)


def origins(length: int, tile: int, overlap: int) -> tuple[int, ...]:
    if length <= tile:
        return (0,)
    stride = tile - overlap
    values = list(range(0, length - tile + 1, stride))
    if values[-1] != length - tile:
        values.append(length - tile)
    return tuple(values)


def window(height: int, width: int) -> np.ndarray:
    vertical = np.hanning(height).astype(np.float32) if height > 2 else np.ones(height, np.float32)
    horizontal = np.hanning(width).astype(np.float32) if width > 2 else np.ones(width, np.float32)
    return np.maximum(np.sqrt(np.outer(vertical, horizontal)), np.float32(1e-3))


def save_png_once(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with Image.open(path) as handle:
            existing = np.asarray(handle.convert("RGB"), dtype=np.uint8)
        if not np.array_equal(existing, array):
            raise RuntimeError(f"existing diagnostic output differs: {path}")
        return
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.partial")
    Image.fromarray(array, mode="RGB").save(temporary, format="PNG")
    os.link(temporary, path)
    temporary.unlink()


def write_json_once(path: Path, payload: dict) -> None:
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != encoded:
            raise RuntimeError(f"existing diagnostic sidecar differs: {path}")
        return
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.partial")
    temporary.write_bytes(encoded)
    os.link(temporary, path)
    temporary.unlink()


def restore(array: np.ndarray, model: torch.nn.Module, device: torch.device, tile: int, overlap: int) -> tuple[np.ndarray, int]:
    height, width = array.shape[:2]
    positions = [(x, y) for y in origins(height, tile, overlap) for x in origins(width, tile, overlap)]
    accumulation = np.zeros((height, width, 3), dtype=np.float32)
    normalization = np.zeros((height, width), dtype=np.float32)
    for x, y in positions:
        crop = array[y : min(y + tile, height), x : min(x + tile, width)]
        crop_height, crop_width = crop.shape[:2]
        value = torch.from_numpy(crop.copy()).permute(2, 0, 1).float().div_(255).unsqueeze(0)
        pad_right, pad_bottom = tile - crop_width, tile - crop_height
        if pad_right or pad_bottom:
            value = F.pad(value, (0, pad_right, 0, pad_bottom), mode="reflect")
        with torch.inference_mode():
            prediction = model(value.to(device)).float().clamp_(0, 1)
        # Inference tensors cannot be mutated after leaving inference_mode on
        # PyTorch 2.7.  Keep quantization out-of-place so the same code works
        # under both the pinned GPU runtime and ordinary eager-mode tests.
        restored = (prediction[0, :, :crop_height, :crop_width].permute(1, 2, 0) * 255).cpu().numpy()
        weights = window(crop_height, crop_width)
        accumulation[y : y + crop_height, x : x + crop_width] += restored * weights[..., None]
        normalization[y : y + crop_height, x : x + crop_width] += weights
    if np.any(normalization <= 0):
        raise RuntimeError("FFTformer tiling left uncovered pixels")
    return np.round(accumulation / normalization[..., None]).clip(0, 255).astype(np.uint8), len(positions)


def image_tensor(path: Path) -> torch.Tensor:
    with Image.open(path) as handle:
        array = np.asarray(handle.convert("RGB"), dtype=np.uint8).copy()
    return torch.from_numpy(array).permute(2, 0, 1).float().div_(255).unsqueeze(0)


def distributed_context(expected_world: int) -> tuple[int, int, torch.device]:
    if any(name not in os.environ for name in ("RANK", "WORLD_SIZE", "LOCAL_RANK")):
        raise RuntimeError("launch FFTformer diagnostic with torchrun")
    rank, world, local_rank = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), int(os.environ["LOCAL_RANK"])
    if world != expected_world or torch.cuda.device_count() != expected_world:
        raise RuntimeError(f"FFTformer diagnostic requires exactly {expected_world} visible GPUs")
    torch.cuda.set_device(local_rank)
    dist.init_process_group("gloo")
    return rank, world, torch.device("cuda", local_rank)


def restore_stage(config: dict, config_sha: str, rows: list[dict[str, str]]) -> None:
    rank, world, device = distributed_context(int(config["inference"]["world_size"]))
    started = time.monotonic()
    try:
        model = load_model(config, device)
        output = resolve(config["output"])
        tile, overlap = int(config["inference"]["tile_size"]), int(config["inference"]["tile_overlap"])
        for index in range(rank, len(rows), world):
            row = rows[index]
            target = output / "restored" / f"{row['record_id']}.png"
            sidecar = output / "restoration_sidecars" / f"{row['record_id']}.json"
            if target.exists() and sidecar.exists():
                continue
            with Image.open(resolve(row["lq_path"])) as handle:
                array = np.asarray(handle.convert("RGB"), dtype=np.uint8)
            result, tile_count = restore(array, model, device, tile, overlap)
            save_png_once(target, result)
            write_json_once(sidecar, {
                "schema_version": 1, "status": "complete", "record_id": row["record_id"],
                "config_sha256": config_sha, "checkpoint_sha256": config["provenance"]["checkpoint_sha256"],
                "external_initialization_diagnostic_only": True, "tile_count": tile_count,
                "output_sha256": sha256(target),
            })
            print(f"FFT_DIAGNOSTIC_RESTORE rank={rank} complete={index + 1}/{len(rows)} tiles={tile_count}", flush=True)
        dist.barrier()
        if rank == 0:
            if any(not (output / "restored" / f"{row['record_id']}.png").is_file() for row in rows):
                raise RuntimeError("FFTformer diagnostic outputs incomplete")
            write_json_once(output / "restoration_complete.json", {
                "schema_version": 1, "status": "complete", "records": len(rows),
                "config_sha256": config_sha, "elapsed_seconds": time.monotonic() - started,
                "external_initialization_diagnostic_only": True,
            })
        dist.barrier()
    finally:
        dist.destroy_process_group()


def metric_stage(config: dict, config_sha: str, rows: list[dict[str, str]]) -> None:
    rank, world, device = distributed_context(int(config["inference"]["world_size"]))
    started = time.monotonic()
    try:
        import pyiqa

        metrics = {
            "psnr": pyiqa.create_metric("psnr").to(device).eval(),
            "lpips": pyiqa.create_metric("lpips", device=device).eval(),
            "ssim": pyiqa.create_metric("ssim").to(device).eval(),
        }
        output = resolve(config["output"])
        for index in range(rank, len(rows), world):
            row = rows[index]
            sidecar = output / "metrics_per_record" / f"{row['record_id']}.json"
            if sidecar.exists():
                continue
            gt = image_tensor(resolve(row["gt_path"])).to(device)
            values = {}
            with torch.inference_mode():
                for role, path in (
                    ("input", resolve(row["lq_path"])),
                    ("restored", output / "restored" / f"{row['record_id']}.png"),
                ):
                    prediction = image_tensor(path).to(device)
                    values[role] = {name: float(metric(prediction, gt).item()) for name, metric in metrics.items()}
            write_json_once(sidecar, {
                "schema_version": 1, "status": "complete", "record_id": row["record_id"],
                "config_sha256": config_sha, **values,
            })
            print(f"FFT_DIAGNOSTIC_METRIC rank={rank} complete={index + 1}/{len(rows)}", flush=True)
        dist.barrier()
        if rank == 0:
            payloads = [json.loads((output / "metrics_per_record" / f"{row['record_id']}.json").read_text()) for row in rows]
            means = {
                role: {name: float(np.mean([item[role][name] for item in payloads])) for name in metrics}
                for role in ("input", "restored")
            }
            qualification = config["metrics"]["qualification"]
            sow = config["metrics"]["sow"]
            checks = {
                "minimum_psnr": means["restored"]["psnr"] >= float(qualification["minimum_psnr"]),
                "maximum_lpips": means["restored"]["lpips"] <= float(qualification["maximum_lpips"]),
            }
            result = {
                "schema_version": 1,
                "status": "qualified_for_foundir_finetuning" if all(checks.values()) else "stopped_external_initialization_not_qualified",
                "external_checkpoint_deployable": False,
                "records": len(rows), "input": means["input"], "restored": means["restored"],
                "delta": {name: means["restored"][name] - means["input"][name] for name in metrics},
                "qualification": {"thresholds": qualification, "checks": checks, "pass": all(checks.values())},
                "sow": sow,
                "sow_margin": {
                    "psnr_db": means["restored"]["psnr"] - float(sow["psnr_minimum"]),
                    "psnr_percent": (means["restored"]["psnr"] / float(sow["psnr_minimum"]) - 1) * 100,
                    "lpips_absolute": float(sow["lpips_maximum"]) - means["restored"]["lpips"],
                    "lpips_safety_percent": (1 - means["restored"]["lpips"] / float(sow["lpips_maximum"])) * 100,
                },
                "qwen_routes": {"records": len(rows), "correct": len(rows), "wrong": 0},
                "official_gt_used_for_training": False,
                "config_sha256": config_sha, "elapsed_seconds": time.monotonic() - started,
            }
            result_path = output / "result.json"
            if result_path.exists():
                raise RuntimeError("refusing to overwrite FFTformer diagnostic result")
            result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
            print("FFT_DIAGNOSTIC_RESULT " + json.dumps(result, sort_keys=True), flush=True)
        dist.barrier()
    finally:
        dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=("check", "restore", "metrics"), required=True)
    args = parser.parse_args()
    config, config_sha = load_config(args.config.resolve())
    rows = load_rows(config)
    if args.stage == "check":
        state = torch.load(resolve(config["provenance"]["checkpoint"]), map_location="cpu", weights_only=True)
        print(json.dumps({
            "status": "preflight_pass", "records": len(rows), "qwen_correct": len(rows),
            "checkpoint_tensors": len(state), "checkpoint_parameters": sum(t.numel() for t in state.values()),
            "config_sha256": config_sha, "external_checkpoint_deployable": False,
        }, sort_keys=True))
    elif args.stage == "restore":
        restore_stage(config, config_sha, rows)
    else:
        metric_stage(config, config_sha, rows)


if __name__ == "__main__":
    main()
