#!/usr/bin/env python3
"""Evaluate frozen EBT arms on the exact existing 660-image whole-image protocol."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import smoke.train_foundir_compositional_ebt_scaled_learning_smoke_v1 as base  # noqa: E402
from lowlight_restoration.compositional_ebt import DEGRADATIONS  # noqa: E402


CONFIG_PATH = ROOT / "configs/foundir_compositional_ebt_tensor_shard_100k_v5.json"
base.CONFIG_PATH = CONFIG_PATH
ROUTE_TOKEN_ATOMS = {"blur_noise": ("blur", "noise")}


def routed_atoms(serialized_order: str) -> set[str]:
    atoms: set[str] = set()
    for token in map(str, json.loads(serialized_order)):
        atoms.update(ROUTE_TOKEN_ATOMS.get(token, (token,)))
    unknown = atoms - set(DEGRADATIONS)
    if unknown:
        raise RuntimeError(f"unknown routed degradation atoms: {unknown}")
    return atoms


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    data = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise RuntimeError(f"refusing to overwrite differing artifact: {path}")
        return
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def save_png(path: Path, pixels: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with Image.open(path) as handle:
            current = np.asarray(handle.convert("RGB"), dtype=np.uint8)
        if current.shape != pixels.shape or not np.array_equal(current, pixels):
            raise RuntimeError(f"existing EBT output changed: {path}")
        return
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    Image.fromarray(pixels, mode="RGB").save(temporary, format="PNG")
    os.replace(temporary, path)


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as handle:
        return np.asarray(handle.convert("RGB"), dtype=np.uint8).copy()


def full_psnr(reference: np.ndarray, prediction: np.ndarray) -> float:
    mse = float(np.mean((reference.astype(np.float64) - prediction.astype(np.float64)) ** 2))
    return float("inf") if mse == 0 else float(20 * np.log10(255.0) - 10 * np.log10(mse))


def tensor_m11(pixels: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(pixels.copy()).permute(2, 0, 1).float().div_(127.5).sub_(1)


def starts(length: int, tile: int, overlap: int) -> list[int]:
    if length <= tile:
        return [0]
    values = list(range(0, length - tile + 1, tile - overlap))
    if values[-1] != length - tile:
        values.append(length - tile)
    return values


def boxes(height: int, width: int, tile: int, overlap: int) -> list[tuple[int, int]]:
    return [(y, x) for y in starts(height, tile, overlap) for x in starts(width, tile, overlap)]


def restore_whole(
    model: torch.nn.Module,
    pixels: np.ndarray,
    q: torch.Tensor,
    config: dict[str, Any],
    device: torch.device,
) -> np.ndarray:
    settings = config["evaluation"]
    tile, overlap, batch_size = (int(settings[key]) for key in ("ebt_tile_size", "ebt_tile_overlap", "ebt_tile_batch_size"))
    height, width = pixels.shape[:2]
    pad_h, pad_w = max(0, tile - height), max(0, tile - width)
    source = tensor_m11(pixels).unsqueeze(0)
    if pad_h or pad_w:
        source = F.pad(source, (0, pad_w, 0, pad_h), mode="reflect")
    padded_h, padded_w = source.shape[-2:]
    locations = boxes(padded_h, padded_w, tile, overlap)
    accumulator = np.zeros((padded_h, padded_w, 3), dtype=np.float32)
    counts = np.zeros((padded_h, padded_w, 1), dtype=np.float32)
    for offset in range(0, len(locations), batch_size):
        batch_locations = locations[offset : offset + batch_size]
        batch = torch.cat([source[:, :, y:y+tile, x:x+tile] for y, x in batch_locations]).to(device, non_blocking=True)
        conditions = q.expand(batch.shape[0], -1).to(device, non_blocking=True)
        with torch.enable_grad():
            restored = base.restore(model, batch, conditions, config, create_graph=False)
        arrays = restored.detach().cpu().add(1).mul(127.5).permute(0, 2, 3, 1).numpy()
        for (y, x), array in zip(batch_locations, arrays, strict=True):
            accumulator[y:y+tile, x:x+tile] += array
            counts[y:y+tile, x:x+tile] += 1
    if np.any(counts == 0):
        raise RuntimeError("incomplete EBT tile coverage")
    return np.rint(accumulator[:height, :width] / counts[:height, :width]).clip(0, 255).astype(np.uint8)


def tiled_lpips(metric: Any, reference: np.ndarray, prediction: np.ndarray, device: torch.device) -> float:
    height, width = reference.shape[:2]
    weighted, area = 0.0, 0
    for y in range(0, height, 512):
        for x in range(0, width, 512):
            ref = tensor_m11(reference[y:min(y+512, height), x:min(x+512, width)]).unsqueeze(0)
            pred = tensor_m11(prediction[y:min(y+512, height), x:min(x+512, width)]).unsqueeze(0)
            tile_area = ref.shape[-2] * ref.shape[-1]
            pad_h, pad_w = max(0, 64-ref.shape[-2]), max(0, 64-ref.shape[-1])
            if pad_h or pad_w:
                ref = F.pad(ref, (0, pad_w, 0, pad_h), mode="replicate")
                pred = F.pad(pred, (0, pad_w, 0, pad_h), mode="replicate")
            with torch.inference_mode():
                value = float(metric((pred.to(device)+1)*0.5, (ref.to(device)+1)*0.5).item())
            metric.reset()
            weighted += value * tile_area
            area += tile_area
    return weighted / area


def load_contract(run_id: str) -> tuple[dict[str, Any], list[dict[str, str]], Path, str]:
    config = json.loads(CONFIG_PATH.read_text())
    config_sha = sha256(CONFIG_PATH)
    evaluation = config["evaluation"]
    manifest = ROOT / evaluation["manifest"]
    if sha256(manifest) != evaluation["manifest_sha256"]:
        raise RuntimeError("660-image manifest identity changed")
    comparison = ROOT / evaluation["existing_comparison_result"]
    if sha256(comparison) != evaluation["existing_comparison_result_sha256"]:
        raise RuntimeError("existing FLUX/Restormer comparison identity changed")
    with manifest.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != int(evaluation["records"]):
        raise RuntimeError("660-image manifest cardinality changed")
    run_root = ROOT / config["runtime"]["output_root"] / run_id
    return config, rows, run_root, config_sha


def summarize(values: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"records": len(values)}
    roles = ("ebt", "project_flux_restormer", "full_foundir_generalist_specialist")
    for role in roles:
        result[role] = {name: float(np.mean([item[role][name] for item in values])) for name in ("psnr", "lpips")}
    for baseline in roles[1:]:
        result[f"ebt_minus_{baseline}"] = {
            "psnr_db": result["ebt"]["psnr"] - result[baseline]["psnr"],
            "lpips": result["ebt"]["lpips"] - result[baseline]["lpips"],
        }
    return result


def run_arm(arm: str, run_id: str) -> None:
    config, rows, run_root, config_sha = load_contract(run_id)
    arm_result = json.loads((run_root / f"{arm}.json").read_text())
    checkpoint = ROOT / arm_result["checkpoint"]
    if arm_result["config_sha256"] != config_sha or sha256(checkpoint) != arm_result["checkpoint_sha256"]:
        raise RuntimeError("frozen checkpoint identity mismatch")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("each evaluation arm requires exactly one visible GPU")
    device = torch.device("cuda:0")
    model = base.build_model(config, arm, device)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if payload["arm"] != arm or payload["config_sha256"] != config_sha:
        raise RuntimeError("checkpoint payload identity mismatch")
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval().requires_grad_(False)
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    metric = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(device).eval()
    output = run_root / "whole_image_evaluation" / arm
    old_sidecars = ROOT / config["evaluation"]["existing_comparison_sidecars"]
    for index, row in enumerate(rows):
        destination = output / "per_record" / f"{row['record_id']}.json"
        if destination.exists():
            cached = json.loads(destination.read_text())
            if cached.get("config_sha256") != config_sha or cached.get("checkpoint_sha256") != arm_result["checkpoint_sha256"]:
                raise RuntimeError(f"stale EBT metric sidecar: {destination}")
            continue
        started = time.perf_counter()
        lq, gt = load_rgb(ROOT / row["lq_path"]), load_rgb(ROOT / row["gt_path"])
        if lq.shape != gt.shape or lq.shape[:2] != (int(row["height"]), int(row["width"])):
            raise RuntimeError(f"source geometry mismatch: {row['record_id']}")
        actual = routed_atoms(row["expert_order_json"])
        q = torch.tensor([[float(name in actual) for name in DEGRADATIONS]], dtype=torch.float32)
        # Preserve the frozen router's fail-closed contract: an abstained
        # record is passed through unchanged, exactly as in the existing
        # FLUX/Restormer whole-image comparison.
        prediction = restore_whole(model, lq, q, config, device) if actual else lq.copy()
        image_path = output / "images" / f"{row['record_id']}.png"
        save_png(image_path, prediction)
        old = json.loads((old_sidecars / f"{row['record_id']}.json").read_text())
        if old["record_id"] != row["record_id"]:
            raise RuntimeError("existing comparison sidecar mismatch")
        ebt = {"psnr": full_psnr(gt, prediction), "lpips": tiled_lpips(metric, gt, prediction, device)}
        write_json(destination, {
            "schema_version": 1,
            "status": "complete",
            "config_sha256": config_sha,
            "checkpoint_sha256": arm_result["checkpoint_sha256"],
            "arm": arm,
            "record_id": row["record_id"],
            "category": row["category"],
            "composition_size": int(row["composition_size"]),
            "routing_status": row["routing_status"],
            "routed_degradations": sorted(actual),
            "ebt": ebt,
            "project_flux_restormer": old["project"],
            "full_foundir_generalist_specialist": old["full_foundir_generalist_specialist"],
            "output_sha256": sha256(image_path),
            "seconds": time.perf_counter() - started,
            "official_foundir_test_accessed": False,
        })
        print(f"EBT_100K_WHOLE arm={arm} record={index+1}/{len(rows)} category={row['category']}", flush=True)
    payloads = [json.loads((output / "per_record" / f"{row['record_id']}.json").read_text()) for row in rows]
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_composition: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for item in payloads:
        by_category[item["category"]].append(item)
        by_composition[int(item["composition_size"])].append(item)
    result = {
        "schema_version": 1,
        "status": "foundir_compositional_ebt_100k_whole_image_arm_complete",
        "arm": arm,
        "records": len(payloads),
        "config_sha256": config_sha,
        "checkpoint_sha256": arm_result["checkpoint_sha256"],
        "overall_metrics": summarize(payloads),
        "composition_metrics": {str(key): summarize(value) for key, value in sorted(by_composition.items())},
        "category_metrics": {key: summarize(value) for key, value in sorted(by_category.items())},
        "metric_protocol": config["evaluation"],
        "triple_used_for_training_or_checkpoint_selection": False,
        "official_foundir_test_accessed": False,
        "sealed_internal_accessed": False,
    }
    write_json(output / "result.json", result)
    print("EBT_100K_WHOLE_ARM_COMPLETE " + json.dumps({"arm": arm, "composition_metrics": result["composition_metrics"]}, sort_keys=True), flush=True)


def finalize(run_id: str) -> None:
    config, _, run_root, config_sha = load_contract(run_id)
    results = {
        arm: json.loads((run_root / "whole_image_evaluation" / arm / "result.json").read_text())
        for arm in config["training"]["arms"]
    }
    if any(value["config_sha256"] != config_sha for value in results.values()):
        raise RuntimeError("evaluation arm config mismatch")
    result = {
        "schema_version": 1,
        "status": "foundir_compositional_ebt_100k_matched_660_complete",
        "run_id": run_id,
        "config_sha256": config_sha,
        "arms": results,
        "comparison_identity": config["evaluation"]["existing_comparison_result_sha256"],
        "triple_used_for_training_or_checkpoint_selection": False,
        "official_foundir_test_accessed": False,
        "sealed_internal_accessed": False,
    }
    write_json(run_root / "matched_660_result.json", result)
    print("EBT_100K_MATCHED_660_COMPLETE")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("plain", "compositional"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    if args.finalize:
        if args.arm:
            parser.error("--finalize does not accept --arm")
        finalize(args.run_id)
    else:
        if not args.arm:
            parser.error("arm evaluation requires --arm")
        run_arm(args.arm, args.run_id)


if __name__ == "__main__":
    main()
