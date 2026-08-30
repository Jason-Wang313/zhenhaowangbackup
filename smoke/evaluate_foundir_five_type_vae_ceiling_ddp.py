#!/usr/bin/env python3
"""Measure the FLUX VAE ceiling on a balanced FoundIR-training-only pilot."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party/flux2/src"))

from lowlight_restoration.foundir_five_type import FIVE_TYPE_CATEGORIES  # noqa: E402
from lowlight_restoration.paired_development import (  # noqa: E402
    peak_signal_noise_ratio,
    rgb_structural_similarity,
)
from lowlight_restoration.representation_ceiling import (  # noqa: E402
    centered_crop,
    high_frequency_diagnostic,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_key(seed: int, row: Mapping[str, str]) -> str:
    return hashlib.sha256(f"{seed}:vae:{row['record_id']}".encode()).hexdigest()


def select_balanced_targets(
    rows: Sequence[Mapping[str, str]], *, per_category: int, seed: int
) -> list[dict[str, str]]:
    by_category: dict[str, list[dict[str, str]]] = defaultdict(list)
    seen_groups: set[str] = set()
    for original in sorted(rows, key=lambda row: stable_key(seed, row)):
        row = dict(original)
        if row["split"] != "train" or row["source_group"] in seen_groups:
            continue
        if len(by_category[row["category"]]) < per_category:
            by_category[row["category"]].append(row)
            seen_groups.add(row["source_group"])
    if set(by_category) != set(FIVE_TYPE_CATEGORIES) or any(
        len(values) != per_category for values in by_category.values()
    ):
        raise RuntimeError("VAE pilot cannot cover every category with disjoint GT groups")
    return [row for category in sorted(by_category) for row in by_category[category]]


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return rows


def reconstruct(autoencoder, image: np.ndarray, device) -> np.ndarray:
    import torch
    import torch.nn.functional as functional

    height, width = image.shape[:2]
    tensor = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).unsqueeze(0)
    tensor = tensor.to(device=device, dtype=torch.float32) * 2.0 - 1.0
    pad_height = (-height) % 16
    pad_width = (-width) % 16
    if pad_height or pad_width:
        tensor = functional.pad(tensor, (0, pad_width, 0, pad_height), mode="reflect")
    with torch.inference_mode():
        output = autoencoder.decode(autoencoder.encode(tensor)).float().clamp(-1.0, 1.0)
    output = (output[0, :, :height, :width] + 1.0) * 0.5
    return np.rint(output.permute(1, 2, 0).cpu().numpy().clip(0, 1) * 255).astype(np.uint8).astype(np.float32) / 255.0


def lpips_value(metric, reference: np.ndarray, prediction: np.ndarray, device) -> float:
    import torch

    tensors = []
    for image in (prediction, reference):
        tensors.append(
            torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))
            .unsqueeze(0)
            .to(device=device, dtype=torch.float32)
        )
    with torch.inference_mode():
        return float(metric(tensors[0], tensors[1]).item())


def evaluate_row(row: Mapping[str, str], config: Mapping[str, object], autoencoder, metric, device) -> dict[str, object]:
    path = ROOT / row["gt_path"]
    if sha256(path) != row["gt_sha256"]:
        raise RuntimeError(f"GT changed before VAE ceiling: {path}")
    with Image.open(path) as handle:
        image = np.asarray(handle.convert("RGB"), dtype=np.float32) / 255.0
    crop = centered_crop(image, int(config["vae_ceiling"]["center_crop_size"]))
    restored = reconstruct(autoencoder, crop, device)
    frequency = high_frequency_diagnostic(crop, restored)
    return {
        "record_id": row["record_id"],
        "category": row["category"],
        "source_group": row["source_group"],
        "gt_sha256": row["gt_sha256"],
        "width": int(crop.shape[1]),
        "height": int(crop.shape[0]),
        "psnr": peak_signal_noise_ratio(crop, restored),
        "ssim": rgb_structural_similarity(crop, restored),
        "lpips": lpips_value(metric, crop, restored, device),
        **frequency,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--contract-only", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    manifest = load_manifest(args.manifest)
    ceiling = config["vae_ceiling"]
    selected = select_balanced_targets(
        manifest,
        per_category=int(ceiling["records_per_category"]),
        seed=int(config["data"]["split_seed"]),
    )
    weights = ROOT / config["model"]["autoencoder_weights"]
    if weights.stat().st_size != int(ceiling["autoencoder_bytes"]) or sha256(weights) != config["model"]["autoencoder_weights_sha256"]:
        raise RuntimeError("FLUX VAE identity changed")
    revision = subprocess.run(
        ["git", "-C", str(ROOT / "third_party/flux2"), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    if revision != ceiling["flux2_revision"]:
        raise RuntimeError("FLUX source revision changed")
    for row in selected:
        if not (ROOT / row["gt_path"]).is_file():
            raise FileNotFoundError(row["gt_path"])
    if args.contract_only:
        print(f"FOUNDIR_FIVE_TYPE_VAE_CONTRACT_PASS records={len(selected)} categories={len(FIVE_TYPE_CATEGORIES)}")
        return

    import torch
    import torch.distributed as distributed
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

    distributed.init_process_group(backend="nccl")
    try:
        rank = distributed.get_rank()
        world = distributed.get_world_size()
        local_rank = int(os.environ["LOCAL_RANK"])
        if world != int(config["distributed"]["required_world_size"]):
            raise RuntimeError("VAE ceiling requires exactly eight GPUs")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        properties = torch.cuda.get_device_properties(device)
        if properties.total_memory < int(config["distributed"]["minimum_memory_gib"]) * 1024**3:
            raise RuntimeError("GPU memory is below the H100/H200 contract")
        if not any(name in properties.name for name in ("H100", "H200")):
            raise RuntimeError(f"unsupported GPU: {properties.name}")
        os.environ["AE_MODEL_PATH"] = str(weights)
        from flux2.util import load_ae

        autoencoder = load_ae(config["model"]["name"], device=device).eval().requires_grad_(False)
        metric = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(device).eval()
        local = [
            evaluate_row(row, config, autoencoder, metric, device)
            for index, row in enumerate(selected)
            if index % world == rank
        ]
        gathered: list[object | None] = [None] * world
        distributed.all_gather_object(gathered, local)
        if rank == 0:
            rows = [item for part in gathered if isinstance(part, list) for item in part]
            if len(rows) != len(selected):
                raise RuntimeError("distributed VAE result is incomplete")
            categories: dict[str, dict[str, float]] = {}
            for category in sorted(FIVE_TYPE_CATEGORIES):
                values = [row for row in rows if row["category"] == category]
                categories[category] = {
                    key: float(np.mean([float(row[key]) for row in values]))
                    for key in ("psnr", "ssim", "lpips", "high_frequency_correlation")
                }
            failures = []
            for category, values in categories.items():
                if values["psnr"] < float(ceiling["minimum_psnr_db"]): failures.append(f"{category}:psnr")
                if values["lpips"] > float(ceiling["maximum_lpips"]): failures.append(f"{category}:lpips")
                if values["high_frequency_correlation"] < float(ceiling["minimum_high_frequency_correlation"]): failures.append(f"{category}:hf")
            payload = {
                "schema_version": 1,
                "status": "pixel_refinement_required" if failures else "latent_ceiling_clear",
                "config_sha256": sha256(args.config),
                "manifest_sha256": sha256(args.manifest),
                "records": len(rows),
                "categories": categories,
                "failures": failures,
                "rows": sorted(rows, key=lambda row: row["record_id"]),
                "protected_foundir_test_accessed": False,
            }
            output = ROOT / config["runtime"]["output_dir"] / "vae_ceiling_result.json"
            output.parent.mkdir(parents=True, exist_ok=True)
            content = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
            if output.exists() and output.read_bytes() != content:
                raise RuntimeError("refusing to overwrite differing VAE result")
            if not output.exists():
                output.write_bytes(content)
            print("FOUNDIR_FIVE_TYPE_VAE_PASS " + json.dumps(payload, sort_keys=True), flush=True)
        distributed.barrier()
    finally:
        distributed.destroy_process_group()


if __name__ == "__main__":
    main()
