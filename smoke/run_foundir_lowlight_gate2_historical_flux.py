#!/usr/bin/env python3
"""Cache the frozen historical synthetic-trained FLUX proposal on outer development."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party/flux2/src"))

from lowlight_restoration.lora import inject_lora, load_lora_checkpoint


DEFAULT_CONFIG = ROOT / "configs/foundir_lowlight_gate2_historical_flux_v1b.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def workspace_path(raw: str) -> Path:
    path = (ROOT / raw).resolve(); path.relative_to(ROOT); return path


def read_config(path: Path) -> dict:
    config = json.loads(path.read_text())
    if config.get("schema_version") != 1 or config.get("protected_foundir_pixels_allowed") is not False:
        raise RuntimeError("historical FLUX contract changed or permits protected pixels")
    if sha256(workspace_path(config["development_manifest"])) != config["development_manifest_sha256"]:
        raise RuntimeError("development manifest changed")
    for key in ("flow_weights", "autoencoder_weights", "lora_checkpoint", "prompt_context", "parent_config"):
        if sha256(workspace_path(config["model"][key])) != config["model"][f"{key}_sha256"]:
            raise RuntimeError(f"historical FLUX dependency changed: {key}")
    return config


def load_rgb(row: Mapping[str, str]) -> np.ndarray:
    path = workspace_path(row["input_path"])
    if path.suffix.lower() == ".npy":
        image = np.load(path, allow_pickle=False)
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise RuntimeError(f"invalid input array: {path}")
        if row["stored_color_order"] == "BGR": image = image[..., ::-1]
        return np.ascontiguousarray(image)
    with Image.open(path) as opened: return np.asarray(opened.convert("RGB"), dtype=np.uint8)


def resize_to_pixel_budget(image: np.ndarray, maximum_pixels: int) -> tuple[np.ndarray, dict[str, object]]:
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or maximum_pixels <= 0:
        raise ValueError("invalid bounded-resize input")
    height, width = image.shape[:2]
    if height * width <= maximum_pixels:
        return image, {"resized": False, "native_width": width, "native_height": height, "inference_width": width, "inference_height": height}
    scale = math.sqrt(maximum_pixels / float(height * width))
    inference_width = max(1, int(math.floor(width * scale)))
    inference_height = max(1, int(math.floor(height * scale)))
    while inference_width * inference_height > maximum_pixels:
        if inference_width >= inference_height: inference_width -= 1
        else: inference_height -= 1
    resized = Image.fromarray(image, mode="RGB").resize((inference_width, inference_height), Image.Resampling.LANCZOS)
    return np.asarray(resized, dtype=np.uint8), {
        "resized": True, "native_width": width, "native_height": height,
        "inference_width": inference_width, "inference_height": inference_height,
    }


def restore_native_dimensions(image: np.ndarray, metadata: Mapping[str, object]) -> np.ndarray:
    expected = (int(metadata["native_height"]), int(metadata["native_width"]), 3)
    if not bool(metadata["resized"]):
        if image.shape != expected: raise RuntimeError("non-resized FLUX output changed dimensions")
        return image
    restored = Image.fromarray(image, mode="RGB").resize(
        (int(metadata["native_width"]), int(metadata["native_height"])), Image.Resampling.LANCZOS
    )
    output = np.asarray(restored, dtype=np.uint8)
    if output.shape != expected: raise RuntimeError("bounded FLUX resize failed to restore native dimensions")
    return output


def pad_tensor(tensor: torch.Tensor, multiple: int) -> tuple[torch.Tensor, int, int]:
    import torch.nn.functional as functional
    height, width = tensor.shape[-2:]
    bottom, right = (-height) % multiple, (-width) % multiple
    if bottom or right:
        mode = "reflect" if height > bottom and width > right else "replicate"
        tensor = functional.pad(tensor, (0, right, 0, bottom), mode=mode)
    return tensor, height, width


def png_bytes(image: np.ndarray) -> bytes:
    buffer = io.BytesIO(); Image.fromarray(image, mode="RGB").save(buffer, format="PNG"); return buffer.getvalue()


def save_once(path: Path, image: np.ndarray) -> str:
    content = png_bytes(image)
    if path.exists():
        if path.read_bytes() != content: raise RuntimeError(f"immutable historical output differs: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as handle: handle.write(content)
    return hashlib.sha256(content).hexdigest()


def write_once(path: Path, content: bytes) -> None:
    if path.exists():
        if path.read_bytes() != content: raise RuntimeError(f"immutable result differs: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as handle: handle.write(content)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG); args = parser.parse_args()
    config_path = args.config.resolve(); config = read_config(config_path)
    if not torch.cuda.is_available() or torch.cuda.device_count() != int(config["runtime"]["device_count"]):
        raise SystemExit(f"HISTORICAL_FLUX_REQUIRES_EXACTLY_ONE_VISIBLE_GPU actual={torch.cuda.device_count()}")
    if config["runtime"]["offline"]:
        os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    os.environ.update(
        KLEIN_4B_BASE_MODEL_PATH=str(workspace_path(config["model"]["flow_weights"])),
        AE_MODEL_PATH=str(workspace_path(config["model"]["autoencoder_weights"])),
    )
    with workspace_path(config["development_manifest"]).open(newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["role"] == config["role"]]
    if len(rows) != 245: raise RuntimeError("outer development count changed")

    from flux2.sampling import batched_prc_img, denoise_cfg, get_schedule, scatter_ids
    from flux2.util import load_ae, load_flow_model

    device = torch.device("cuda:0"); model_name = config["model"]["name"]
    prompt = load_file(str(workspace_path(config["model"]["prompt_context"])), device="cpu")
    context = torch.cat([prompt["empty_context"], prompt["context"]], dim=0).to(device)
    context_ids = torch.cat([prompt["empty_context_ids"], prompt["context_ids"]], dim=0).to(device)
    autoencoder = load_ae(model_name, device=device).eval().requires_grad_(False)
    flow_model = load_flow_model(model_name, device=device).eval().requires_grad_(False)
    names = inject_lora(flow_model, config["lora"]["target_patterns"], rank=config["lora"]["rank"], alpha=config["lora"]["alpha"], dropout=config["lora"]["dropout"])
    if len(names) != config["lora"]["expected_module_count"]: raise RuntimeError("historical LoRA module count changed")
    load_lora_checkpoint(flow_model, workspace_path(config["model"]["lora_checkpoint"]))

    output_root = workspace_path(config["runtime"]["output_directory"])
    results = []; total_seconds = 0.0; peak = 0
    for index, row in enumerate(rows):
        native_source = load_rgb(row)
        source, resize_metadata = resize_to_pixel_budget(
            native_source, int(config["inference"]["native_whole_image_max_pixels"])
        )
        degraded = torch.from_numpy(source.transpose(2, 0, 1).copy()).unsqueeze(0).to(device=device, dtype=torch.float32).div(127.5).sub(1.0)
        degraded, inference_height, inference_width = pad_tensor(degraded, int(config["inference"]["padding_multiple"]))
        torch.cuda.reset_peak_memory_stats(device); torch.cuda.synchronize(device); started = time.perf_counter()
        with torch.inference_mode():
            reference_latent = autoencoder.encode(degraded).to(torch.bfloat16)
            reference_tokens, reference_ids = batched_prc_img(reference_latent)
            reference_ids = reference_ids.clone(); reference_ids[..., 0] = int(config["inference"]["reference_time_coordinate"])
            generator = torch.Generator(device=device).manual_seed(int(config["inference"]["seed"]) + index)
            noise = torch.randn(reference_latent.shape, generator=generator, device=device, dtype=torch.bfloat16)
            image_tokens, image_ids = batched_prc_img(noise)
            schedule = get_schedule(int(config["inference"]["num_steps"]), image_tokens.shape[1])
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                tokens = denoise_cfg(flow_model, image_tokens, image_ids, context, context_ids, timesteps=schedule, guidance=float(config["inference"]["guidance"]), img_cond_seq=reference_tokens, img_cond_seq_ids=reference_ids)
            latent = torch.cat(scatter_ids(tokens, image_ids)).squeeze(2)
            restored = autoencoder.decode(latent).float().clamp(-1, 1)[0, :, :inference_height, :inference_width]
        torch.cuda.synchronize(device); elapsed = time.perf_counter() - started
        total_seconds += elapsed; peak = max(peak, int(torch.cuda.max_memory_allocated(device)))
        inference_image = restored.add(1).mul(127.5).round().byte().permute(1, 2, 0).cpu().numpy()
        image = restore_native_dimensions(inference_image, resize_metadata)
        stem = f"{index:04d}_{row['family']}_{hashlib.sha256(row['record_id'].encode()).hexdigest()[:16]}.png"
        path = output_root / stem; digest = save_once(path, image)
        results.append({"candidate_id": "P0_HISTORICAL_FLUX", "family": row["family"], "group_id": row["group_id"], "record_id": row["record_id"], "target_path": row["target_path"], "output_path": str(path.relative_to(ROOT)), "output_sha256": digest, "width": resize_metadata["native_width"], "height": resize_metadata["native_height"], "inference_width": resize_metadata["inference_width"], "inference_height": resize_metadata["inference_height"], "resized": resize_metadata["resized"], "seconds": elapsed})
        print(f"HISTORICAL_FLUX_PROGRESS records={index + 1}/{len(rows)} resized={resize_metadata['resized']} inference={resize_metadata['inference_width']}x{resize_metadata['inference_height']} seconds={elapsed:.4f}", flush=True)
    fields = list(results[0]); buffer = io.StringIO(newline=""); writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n"); writer.writeheader(); writer.writerows(results)
    manifest_content = buffer.getvalue().encode(); manifest_path = workspace_path(config["outputs"]["manifest"]); write_once(manifest_path, manifest_content)
    payload = {"schema_version": 1, "status": "pass_historical_evidence_cached", "protected_foundir_pixels_used": False, "config": str(config_path.relative_to(ROOT)), "config_sha256": sha256(config_path), "runner_sha256": sha256(Path(__file__)), "manifest": str(manifest_path.relative_to(ROOT)), "manifest_sha256": hashlib.sha256(manifest_content).hexdigest(), "records": len(results), "runtime": {"total_seconds": total_seconds, "mean_seconds": total_seconds / len(results), "peak_memory_bytes": peak}, "interpretation": config["interpretation"]}
    write_once(workspace_path(config["outputs"]["result"]), (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode())
    print(f"FOUNDIR_LOWLIGHT_GATE2_HISTORICAL_FLUX_PASS records={len(results)} mean_seconds={payload['runtime']['mean_seconds']:.4f}")


if __name__ == "__main__": main()
