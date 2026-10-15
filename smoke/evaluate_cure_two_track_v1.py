#!/usr/bin/env python3
"""Frozen full-resolution CCDD-11 triple/selective evaluation for CURE tracks."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from skimage.metrics import structural_similarity

from build_cure_two_track_eval_manifest_v1 import TEST, build


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party/cure_official"))
from cure.inference_utils import load_runtime  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_rows(manifest: Path) -> list[dict[str, str]]:
    expected_content, expected_metadata = build(TEST)
    if manifest.read_text() != expected_content:
        raise RuntimeError("frozen evaluation manifest differs from exact CCDD-11 target mapping")
    metadata_path = manifest.with_suffix(".json")
    if not metadata_path.is_file() or json.loads(metadata_path.read_text()) != expected_metadata:
        raise RuntimeError("evaluation metadata missing or mismatched")
    with manifest.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    required = {"id", "kind", "source_prompt", "remove_prompt", "input_path", "target_path", "target_provenance"}
    if not rows or any(not required.issubset(row) for row in rows):
        raise RuntimeError("evaluation manifest missing required fields")
    if len({row["id"] for row in rows}) != len(rows):
        raise RuntimeError("duplicate evaluation IDs")
    for row in rows:
        if not row["id"] or not all(character.isalnum() or character in "_-" for character in row["id"]):
            raise RuntimeError(f"unsafe evaluation ID: {row['id']}")
        if row["kind"] not in {"full", "selective"}:
            raise RuntimeError(f"invalid task kind: {row['id']}")
        if row["source_prompt"] not in {"low_haze_rain", "low_haze_snow"}:
            raise RuntimeError(f"non-triple evaluation class: {row['id']}")
        if row["kind"] == "selective" and row["target_provenance"] != "verified_matched_intermediate":
            raise RuntimeError(f"unverified selective target: {row['id']}")
        for field in ("input_path", "target_path"):
            path = (ROOT / row[field]).resolve()
            path.relative_to(ROOT)
            if not path.is_file():
                raise FileNotFoundError(path)
    return rows


def read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def image_metrics(predicted: np.ndarray, target: np.ndarray) -> dict[str, float]:
    if predicted.shape != target.shape:
        raise RuntimeError("prediction/target shape mismatch")
    squared = (predicted.astype(np.float64) - target.astype(np.float64)) ** 2
    mse = float(np.mean(squared))
    psnr = float("inf") if mse == 0 else 10 * math.log10(255**2 / mse)
    ssim = float(structural_similarity(predicted, target, channel_axis=2, data_range=255))
    if not math.isfinite(psnr) or not math.isfinite(ssim):
        raise RuntimeError("nonfinite metrics")
    return {"psnr": psnr, "ssim": ssim}


def terminal_ready(track: str) -> tuple[Path, Path]:
    base = ROOT / "checkpoints/cure_two_track_v1"
    for name in ("published_style", "strict_holdout"):
        marker = base / name / "terminal_end.json"
        if not marker.is_file():
            raise RuntimeError(f"both frozen tracks required before test access: {marker}")
        payload = json.loads(marker.read_text())
        if payload.get("exit_status") != 0 or "cure" not in payload.get("checkpoints", {}):
            raise RuntimeError(f"track did not terminally complete: {marker}")
        probe = base / name / "cure_probe/result.json"
        if not probe.is_file() or json.loads(probe.read_text()).get("output_collapse_gate_pass") is not True:
            raise RuntimeError(f"track collapse gate did not pass: {probe}")
    return base / track / "cure/model_300.tar", base / track / "embedder/embedder_200.tar"


@torch.inference_mode()
def run(track: str, manifest: Path, output: Path) -> None:
    checkpoint, embedder = terminal_ready(track)
    rows = load_rows(manifest)
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required for frozen CCDD-11 evaluation")
    torch.cuda.set_device(local_rank)
    device = f"cuda:{local_rank}"
    restorer, encoder, torch_device = load_runtime(checkpoint, embedder, device)
    checkpoint_hash = sha256(checkpoint)
    embedder_hash = sha256(embedder)
    manifest_hash = sha256(manifest)
    output.mkdir(parents=True, exist_ok=True)
    for row in rows[rank::world]:
        source = ROOT / row["input_path"]
        target = ROOT / row["target_path"]
        lq = read_rgb(source)
        gt = read_rgb(target)
        if lq.shape != gt.shape:
            raise RuntimeError(f"unaligned evaluation pair: {row['id']}")
        tensor = torch.from_numpy(lq.copy()).permute(2, 0, 1).unsqueeze(0).float().div(255).to(torch_device)
        embedding = encoder([row["remove_prompt"]])
        prediction = restorer(tensor, embedding)
        if not torch.isfinite(prediction).all():
            raise RuntimeError(f"nonfinite output: {row['id']}")
        pixels = prediction.squeeze(0).clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).cpu().numpy()
        image_path = output / f"{row['id']}.png"
        sidecar = output / f"{row['id']}.json"
        if image_path.exists() or sidecar.exists():
            raise RuntimeError(f"refusing to overwrite evaluation artifact: {row['id']}")
        Image.fromarray(pixels).save(image_path)
        result = {
            "id": row["id"], "kind": row["kind"],
            "source_prompt": row["source_prompt"], "remove_prompt": row["remove_prompt"],
            "target_provenance": row["target_provenance"],
            "track": track, "metrics": image_metrics(pixels, gt),
            "input_metrics": image_metrics(lq, gt),
            "output_sha256": sha256(image_path),
            "input_sha256": sha256(source), "target_sha256": sha256(target),
            "checkpoint_sha256": checkpoint_hash, "embedder_sha256": embedder_hash,
            "manifest_sha256": manifest_hash, "official_foundir_test_accessed": False,
        }
        with sidecar.open("x") as stream:
            json.dump(result, stream, indent=2, sort_keys=True)
            stream.write("\n")
        print(f"CURE_EVAL rank={rank} record={row['id']} kind={row['kind']}", flush=True)
    if world > 1:
        torch.distributed.init_process_group(backend="nccl", device_id=torch_device)
        torch.distributed.barrier()
    if rank == 0:
        groups: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            sidecar = output / f"{row['id']}.json"
            if not sidecar.is_file():
                raise RuntimeError(f"missing evaluation sidecar: {sidecar}")
            result = json.loads(sidecar.read_text())
            image_path = output / f"{row['id']}.png"
            if result["output_sha256"] != sha256(image_path):
                raise RuntimeError(f"evaluation PNG hash mismatch: {row['id']}")
            groups[row["kind"]].append(result)
            groups[f"{row['kind']}:{row['source_prompt']}"].append(result)
        summary = {
            key: {
                "count": len(items),
                "psnr": float(np.mean([item["metrics"]["psnr"] for item in items])),
                "ssim": float(np.mean([item["metrics"]["ssim"] for item in items])),
            } for key, items in groups.items()
        }
        aggregate = {
            "track": track, "rows": len(rows), "summary": summary,
            "manifest_sha256": manifest_hash,
            "checkpoint_sha256": checkpoint_hash,
            "embedder_sha256": embedder_hash,
            "official_foundir_test_accessed": False,
        }
        with (output / "result.json").open("x") as stream:
            json.dump(aggregate, stream, indent=2, sort_keys=True)
            stream.write("\n")
        print(f"CURE_EVAL_END track={track} rows={len(rows)}", flush=True)
    if world > 1:
        torch.distributed.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track", choices=("published_style", "strict_holdout"), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.track, args.manifest.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
