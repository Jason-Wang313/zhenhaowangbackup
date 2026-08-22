#!/usr/bin/env python3
"""Cache the official FoundIR generalist and documented Lowlight cascade."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Mapping

import numpy as np
from PIL import Image

from lowlight_restoration.representation_ceiling import reconstruct_tiled_average


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/foundir_lowlight_gate2_teacher_v1b.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def workspace_path(raw: str) -> Path:
    path = (ROOT / raw).resolve()
    path.relative_to(ROOT)
    return path


def load_rgb(row: Mapping[str, str], key: str) -> np.ndarray:
    path = workspace_path(row[key])
    if path.suffix.lower() == ".npy":
        image = np.load(path, allow_pickle=False)
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise RuntimeError(f"invalid array: {path}")
        if row["stored_color_order"] == "BGR":
            image = image[..., ::-1]
        elif row["stored_color_order"] != "RGB":
            raise RuntimeError("unsupported stored color order")
        return np.ascontiguousarray(image)
    with Image.open(path) as opened:
        return np.asarray(opened.convert("RGB"), dtype=np.uint8)


def quantize(tensor) -> np.ndarray:
    array = tensor.detach().float().clamp(0, 1).mul(255).round().byte()
    return np.ascontiguousarray(array.permute(1, 2, 0).cpu().numpy())


def quantize_array(image: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.rint(np.clip(image, 0.0, 255.0)).astype(np.uint8))


def png_bytes(image: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(image, mode="RGB").save(buffer, format="PNG")
    return buffer.getvalue()


def save_once(path: Path, image: np.ndarray) -> str:
    content = png_bytes(image)
    if path.exists():
        if path.read_bytes() != content:
            raise RuntimeError(f"immutable teacher output differs: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as handle:
            handle.write(content)
    with Image.open(io.BytesIO(content)) as opened:
        reloaded = np.asarray(opened.convert("RGB"), dtype=np.uint8)
    if not np.array_equal(reloaded, image):
        raise RuntimeError("teacher PNG round trip changed pixels")
    return hashlib.sha256(content).hexdigest()


def read_config(path: Path) -> dict:
    config = json.loads(path.read_text())
    if config.get("schema_version") != 1 or config.get("protected_foundir_pixels_allowed") is not False:
        raise RuntimeError("teacher contract changed or permits protected pixels")
    if sha256(workspace_path(config["development_manifest"])) != config["development_manifest_sha256"]:
        raise RuntimeError("development manifest changed")
    for section in ("generalist", "specialist"):
        if sha256(workspace_path(config[section]["checkpoint"])) != config[section]["checkpoint_sha256"]:
            raise RuntimeError(f"{section} checkpoint changed")
    return config


def outer_rows(config: Mapping[str, object]) -> list[dict[str, str]]:
    with workspace_path(str(config["development_manifest"])).open(newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["role"] == config["role"]]
    if len(rows) != 245:
        raise RuntimeError("outer development count changed")
    return rows


def output_stem(index: int, row: Mapping[str, str]) -> str:
    digest = hashlib.sha256(row["record_id"].encode()).hexdigest()[:16]
    return f"{index:04d}_{row['family']}_{digest}.png"


def build_generalist(config: Mapping[str, object], device):
    import torch

    foundir = workspace_path(config["generalist"]["repository"])
    sys.path.insert(0, str(foundir))
    from src.model import ResidualDiffusion, UnetRes

    model = UnetRes(dim=64, dim_mults=(1, 2, 4, 8), num_unet=1, condition=True, objective="pred_res", test_res_or_noise="res")
    diffusion = ResidualDiffusion(
        model, image_size=1024, timesteps=1000, delta_end=float(config["generalist"]["delta_end"]),
        sampling_timesteps=int(config["generalist"]["sampling_steps"]), ddim_sampling_eta=float(config["generalist"]["ddim_eta"]),
        objective="pred_res", loss_type="l1", condition=True, sum_scale=float(config["generalist"]["sum_scale"]), test_res_or_noise="res",
    )
    checkpoint = torch.load(workspace_path(config["generalist"]["checkpoint"]), map_location="cpu", weights_only=False)
    state = {key[len("ema_model.") :]: value for key, value in checkpoint["ema"].items() if key.startswith("ema_model.")}
    diffusion.load_state_dict(state, strict=True)
    diffusion.init()
    return diffusion.to(device).eval().requires_grad_(False)


def pad_tensor(tensor, multiple: int):
    import torch.nn.functional as functional

    height, width = tensor.shape[-2:]
    bottom = (-height) % multiple
    right = (-width) % multiple
    if bottom or right:
        mode = "reflect" if height > bottom and width > right else "replicate"
        tensor = functional.pad(tensor, (0, right, 0, bottom), mode=mode)
    return tensor, height, width


def run_generalist(config: dict, rows: list[dict[str, str]], device) -> tuple[list[dict], dict]:
    import torch

    seed = int(config["runtime"]["seed"])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    model = build_generalist(config, device)
    output_root = workspace_path(config["runtime"]["output_directory"]) / "P1_FOUNDIR_GENERALIST"
    results = []
    total_seconds = 0.0
    peak = 0
    for index, row in enumerate(rows):
        source = load_rgb(row, "input_path")
        height, width = source.shape[:2]
        policy = config["generalist"]["large_image_policy"]

        def process_tile(tile: np.ndarray) -> np.ndarray:
            tensor = torch.from_numpy(tile.transpose(2, 0, 1).copy()).unsqueeze(0).to(device=device, dtype=torch.float32) / 255.0
            tensor, tile_height, tile_width = pad_tensor(tensor, int(config["generalist"]["padding_multiple"]))
            with torch.inference_mode():
                sequence = model.sample(tensor, batch_size=1, last=True, task=row["input_path"])
                prediction = sequence[-1][0, :, :tile_height, :tile_width]
            return quantize(prediction).astype(np.float32)

        torch.cuda.reset_peak_memory_stats(device); torch.cuda.synchronize(device)
        started = time.perf_counter()
        if height * width > int(policy["activation_max_pixels"]):
            reconstructed, tile_metadata = reconstruct_tiled_average(
                source, process_tile, tile_size=int(policy["tile_size"]), overlap=int(policy["overlap"])
            )
            image = quantize_array(reconstructed)
        else:
            image = quantize_array(process_tile(source))
            tile_metadata = {"tile_count": 1}
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        total_seconds += elapsed; peak = max(peak, int(torch.cuda.max_memory_allocated(device)))
        path = output_root / output_stem(index, row)
        digest = save_once(path, image)
        results.append({"candidate_id": "P1_FOUNDIR_GENERALIST", "family": row["family"], "group_id": row["group_id"], "record_id": row["record_id"], "target_path": row["target_path"], "output_path": str(path.relative_to(ROOT)), "output_sha256": digest, "width": width, "height": height, "tile_count": tile_metadata["tile_count"], "seconds": elapsed})
        print(f"FOUNDIR_GENERALIST_PROGRESS records={index + 1}/{len(rows)} tiles={tile_metadata['tile_count']} seconds={elapsed:.4f}", flush=True)
    del model; torch.cuda.empty_cache()
    return results, {"total_seconds": total_seconds, "mean_seconds": total_seconds / len(rows), "peak_memory_bytes": peak}


def build_specialist(config: Mapping[str, object], device):
    import torch

    repository = workspace_path(config["specialist"]["repository"])
    sys.path.insert(0, str(repository))
    from basicsr.archs.wavemamba_arch import WaveMamba

    model = WaveMamba(in_chn=3, wf=32, n_l_blocks=[1, 2, 4], n_h_blocks=[1, 1, 2], ffn_scale=2.0).to(device)
    checkpoint = torch.load(workspace_path(config["specialist"]["checkpoint"]), map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["params"], strict=True)
    return model.eval().requires_grad_(False)


def run_specialist(config: dict, rows: list[dict[str, str]], device) -> tuple[list[dict], dict]:
    import torch

    model = build_specialist(config, device)
    root = workspace_path(config["runtime"]["output_directory"])
    input_root = root / "P1_FOUNDIR_GENERALIST"
    output_root = root / "P1_FOUNDIR_CASCADE"
    results = []
    total_seconds = 0.0
    peak = 0
    for index, row in enumerate(rows):
        input_path = input_root / output_stem(index, row)
        if not input_path.is_file():
            raise RuntimeError(f"missing generalist proposal: {input_path}")
        with Image.open(input_path) as opened:
            source = np.asarray(opened.convert("RGB"), dtype=np.uint8)
        height, width = source.shape[:2]
        policy = config["specialist"]["large_image_policy"]

        def process_tile(tile: np.ndarray) -> np.ndarray:
            tensor = torch.from_numpy(tile.transpose(2, 0, 1).copy()).unsqueeze(0).to(device=device, dtype=torch.float32) / 255.0
            tensor, tile_height, tile_width = pad_tensor(tensor, int(config["specialist"]["padding_multiple"]))
            with torch.inference_mode():
                prediction = model.restoration_network(tensor)[:, :, :tile_height, :tile_width]
            return quantize(prediction[0]).astype(np.float32)

        torch.cuda.reset_peak_memory_stats(device); torch.cuda.synchronize(device)
        started = time.perf_counter()
        if height * width > int(policy["activation_max_pixels"]):
            reconstructed, tile_metadata = reconstruct_tiled_average(
                source, process_tile, tile_size=int(policy["tile_size"]), overlap=int(policy["overlap"])
            )
            image = quantize_array(reconstructed)
        else:
            image = quantize_array(process_tile(source))
            tile_metadata = {"tile_count": 1}
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        total_seconds += elapsed; peak = max(peak, int(torch.cuda.max_memory_allocated(device)))
        path = output_root / output_stem(index, row)
        digest = save_once(path, image)
        results.append({"candidate_id": "P1_FOUNDIR_CASCADE", "family": row["family"], "group_id": row["group_id"], "record_id": row["record_id"], "target_path": row["target_path"], "input_proposal_path": str(input_path.relative_to(ROOT)), "output_path": str(path.relative_to(ROOT)), "output_sha256": digest, "width": width, "height": height, "tile_count": tile_metadata["tile_count"], "seconds": elapsed})
        print(f"FOUNDIR_SPECIALIST_PROGRESS records={index + 1}/{len(rows)} tiles={tile_metadata['tile_count']} seconds={elapsed:.4f}", flush=True)
    del model; torch.cuda.empty_cache()
    return results, {"total_seconds": total_seconds, "mean_seconds": total_seconds / len(rows), "peak_memory_bytes": peak}


def write_manifest(path: Path, rows: list[dict]) -> str:
    fields = list(rows[0])
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n"); writer.writeheader(); writer.writerows(rows)
    content = buffer.getvalue().encode()
    if path.exists() and path.read_bytes() != content:
        raise RuntimeError(f"existing teacher manifest differs: {path}")
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as handle: handle.write(content)
    return hashlib.sha256(content).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--stage", choices=("both",), default="both")
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = read_config(config_path)
    extra = workspace_path(config["runtime"]["extra_packages"])
    sys.path.insert(0, str(extra))
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != int(config["runtime"]["device_count"]):
        raise SystemExit(f"TEACHER_REQUIRES_EXACTLY_ONE_VISIBLE_GPU actual={torch.cuda.device_count()}")
    device = torch.device("cuda:0")
    rows = outer_rows(config)
    result_path = workspace_path(config["outputs"]["result"])
    payload = {"schema_version": 1, "status": "in_progress", "protected_foundir_pixels_used": False, "config": str(config_path.relative_to(ROOT)), "config_sha256": sha256(config_path), "runner_sha256": sha256(Path(__file__)), "records": len(rows), "stages": {}}
    if args.stage in ("generalist", "both"):
        generalist_rows, runtime = run_generalist(config, rows, device)
        digest = write_manifest(workspace_path(config["outputs"]["generalist_manifest"]), generalist_rows)
        payload["stages"]["generalist"] = {"status": "pass", "manifest": config["outputs"]["generalist_manifest"], "manifest_sha256": digest, "runtime": runtime}
    if args.stage in ("specialist", "both"):
        specialist_rows, runtime = run_specialist(config, rows, device)
        digest = write_manifest(workspace_path(config["outputs"]["cascade_manifest"]), specialist_rows)
        payload["stages"]["cascade"] = {"status": "pass", "manifest": config["outputs"]["cascade_manifest"], "manifest_sha256": digest, "runtime": runtime}
    payload["status"] = "pass" if set(payload["stages"]) == {"generalist", "cascade"} else "partial_pass"
    content = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    if result_path.exists():
        if result_path.read_bytes() != content:
            raise RuntimeError("existing immutable teacher result differs")
    else:
        result_path.parent.mkdir(parents=True, exist_ok=True)
        with result_path.open("xb") as handle:
            handle.write(content)
    print(f"FOUNDIR_LOWLIGHT_GATE2_TEACHER_{payload['status'].upper()} stages={','.join(sorted(payload['stages']))}")


if __name__ == "__main__":
    main()
