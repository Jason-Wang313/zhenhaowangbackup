#!/usr/bin/env python3
"""Training-only real-image output-collapse gate for CURE checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party/cure_official"))
from cure.inference_utils import load_runtime  # noqa: E402


PROMPTS = ("low_haze", "haze_rain", "low_rain")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_crop(path: Path, size: int = 256) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("RGB")
        if min(image.size) < size:
            raise RuntimeError(f"probe image too small: {path}")
        left = (image.width - size) // 2
        top = (image.height - size) // 2
        return np.asarray(image.crop((left, top, left + size, top + size)), dtype=np.uint8)


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else 10 * math.log10(255**2 / mse)


@torch.inference_mode()
def probe(checkpoint: Path, embedder: Path, output: Path, device: str) -> dict:
    if output.exists():
        raise RuntimeError(f"refusing to overwrite probe output: {output}")
    main = ROOT / "data/third_party/cure_two_track_v1/paired_train/main_data"
    names = sorted(path.name for path in (main / "clear").glob("*.png"))[:3]
    if len(names) != 3:
        raise RuntimeError("need three real paired training sources for collapse probe")
    restorer, encoder, torch_device = load_runtime(checkpoint, embedder, device)
    records = []
    output.mkdir(parents=True)
    for name, prompt in zip(names, PROMPTS, strict=True):
        source = main / prompt / name
        target = main / "clear" / name
        lq = read_crop(source)
        gt = read_crop(target)
        if lq.shape != gt.shape:
            raise RuntimeError(f"probe pair shape mismatch: {name}")
        tensor = torch.from_numpy(lq.copy()).permute(2, 0, 1).unsqueeze(0).float().div(255).to(torch_device)
        predicted = restorer(tensor, encoder([prompt]))
        if not torch.isfinite(predicted).all():
            raise RuntimeError(f"nonfinite probe output: {name}")
        pixels = predicted.squeeze(0).clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).cpu().numpy()
        destination = output / f"{Path(name).stem}_{prompt}.png"
        Image.fromarray(pixels).save(destination)
        records.append({
            "source": name,
            "prompt": prompt,
            "input_psnr": psnr(lq, gt),
            "output_psnr": psnr(pixels, gt),
            "input_mean": float(lq.mean()),
            "input_endpoint_fraction": float(((lq <= 1) | (lq >= 254)).mean()),
            "output_mean": float(pixels.mean()),
            "output_endpoint_fraction": float(((pixels <= 1) | (pixels >= 254)).mean()),
            "output": str(destination.relative_to(ROOT)),
            "output_sha256": sha256(destination),
        })
    collapsed = any(
        (row["output_mean"] < 3 and row["input_mean"] > 10)
        or (row["output_mean"] > 252 and row["input_mean"] < 245)
        or (row["output_endpoint_fraction"] >= 0.95 and row["input_endpoint_fraction"] < 0.8)
        for row in records
    ) or all(row["output_psnr"] < row["input_psnr"] - 5 for row in records)
    result = {
        "checkpoint": str(checkpoint.relative_to(ROOT)),
        "checkpoint_sha256": sha256(checkpoint),
        "embedder_sha256": sha256(embedder),
        "records": records,
        "output_collapse_gate_pass": not collapsed,
        "evaluation_split": "CCDD11_train_only_probe",
        "official_foundir_test_accessed": False,
    }
    result_path = output / "result.json"
    with result_path.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--embedder", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    result = probe(args.checkpoint.resolve(), args.embedder.resolve(), args.output.resolve(), args.device)
    print(json.dumps(result, indent=2))
    if not result["output_collapse_gate_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
