#!/usr/bin/env python3
"""Bounded development audit of fixed FLUX/motion-deblur pixel blends."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lowlight_restoration.flux_prompt_cache import file_sha256  # noqa: E402
from lowlight_restoration.foundir_flux2_protocol import (  # noqa: E402
    evaluation_artifact_stem,
    resolve_workspace_path,
)
from lowlight_restoration.task_native_restormer import Restormer  # noqa: E402
from smoke.evaluate_foundir_five_type_flux2_ddp import (  # noqa: E402
    lpips_value,
    synchronize_rank_zero_finalization,
    torchrun_worker_identity,
    write_once,
    write_or_verify_png,
)
from smoke.train_foundir_five_type_flux2_overfit_ddp import psnr  # noqa: E402


EXPECTED_ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
EXPECTED_CHECKPOINT_SHA = "f223b06b73caf75adf9c936a9d5103e3924b8f5246419832b84eb179990d0cdc"
EXPECTED_TRAINING_CONFIG_SHA = "0a0799b08265b6184f01d0a8bd3c7efd010d0a5bb6cb33d64e60e6d8143019fa"


def rehearsal_digest(*parts: object) -> str:
    """Exact digest used by evaluate_foundir_mixed_restoration_pilot_ddp.py."""
    return hashlib.sha256("\0".join(map(str, parts)).encode()).hexdigest()


def alpha_key(alpha: float) -> str:
    return f"{alpha:.2f}"


def tensor_m11(pixels: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(pixels.copy()).permute(2, 0, 1).float().div_(127.5).sub_(1.0).unsqueeze(0)


def quantize_01(image: torch.Tensor) -> np.ndarray:
    return image[0].detach().float().cpu().permute(1, 2, 0).mul(255.0).round().clamp_(0, 255).byte().numpy()


def load_rehearsal_crop(
    row: Mapping[str, str], *, crop_size: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor, list[int], bool]:
    arrays = []
    for field, hash_field in (("lq_path", "lq_sha256"), ("gt_path", "gt_sha256")):
        path = resolve_workspace_path(ROOT, row[field])
        if file_sha256(path) != row[hash_field]:
            raise RuntimeError(f"rehearsal source image changed: {row['record_id']} {field}")
        with Image.open(path) as handle:
            arrays.append(np.asarray(handle.convert("RGB"), dtype=np.uint8).copy())
    if arrays[0].shape != arrays[1].shape or min(arrays[0].shape[:2]) < crop_size:
        raise RuntimeError(f"rehearsal pair is unaligned or undersized: {row['record_id']}")
    height, width = arrays[0].shape[:2]
    digest = bytes.fromhex(rehearsal_digest(seed, "mixed_restoration_crop", row["record_id"]))
    y = int.from_bytes(digest[:8], "big") % (height - crop_size + 1)
    x = int.from_bytes(digest[8:16], "big") % (width - crop_size + 1)

    def tensor(array: np.ndarray) -> torch.Tensor:
        crop = array[y : y + crop_size, x : x + crop_size]
        return torch.from_numpy(crop.copy()).permute(2, 0, 1).float().div_(127.5).sub_(1.0)

    return tensor(arrays[0]), tensor(arrays[1]), [int(x), int(y)], False


def load_config(path: Path) -> tuple[dict, list[dict[str, str]]]:
    config = json.loads(path.read_text())
    required = {
        "schema_version": 1,
        "status": "frozen_before_blur_motion_flux_fusion_development_audit",
        "category": "01Blur",
        "expected_records": 128,
        "expected_source_groups": 128,
        "expected_routing_status_counts": {"abstain": 8, "correct": 120},
        "crop_size": 512,
        "crop_seed": 2026092002,
        "alphas": EXPECTED_ALPHAS,
        "abstain_policy": "preserve_quantized_input",
        "sow": {"psnr_minimum": 25.9, "lpips_maximum": 0.18},
        "development_pixels_used_for_selection": True,
        "sealed_internal_test_accessed": False,
        "protected_foundir_test_accessed": False,
    }
    if any(config.get(key) != value for key, value in required.items()):
        raise RuntimeError("Blur fusion audit contract changed")
    manifest = resolve_workspace_path(ROOT, config["manifest"])
    checkpoint = resolve_workspace_path(ROOT, config["motion_checkpoint"])
    rehearsal = resolve_workspace_path(ROOT, config["rehearsal_result"])
    if file_sha256(manifest) != config["manifest_sha256"]:
        raise RuntimeError("frozen rehearsal manifest changed")
    if file_sha256(rehearsal) != config["rehearsal_result_sha256"]:
        raise RuntimeError("frozen rehearsal result changed")
    if config["motion_checkpoint_sha256"] != EXPECTED_CHECKPOINT_SHA or file_sha256(checkpoint) != EXPECTED_CHECKPOINT_SHA:
        raise RuntimeError("motion checkpoint changed")
    if config["motion_training_config_sha256"] != EXPECTED_TRAINING_CONFIG_SHA:
        raise RuntimeError("motion training identity changed")
    with manifest.open(newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle) if row["category"] == "01Blur"]
    if len(rows) != 128 or len({row["record_id"] for row in rows}) != 128:
        raise RuntimeError("Blur audit record coverage changed")
    if len({row["source_group"] for row in rows}) != 128:
        raise RuntimeError("Blur audit source groups changed")
    if Counter(row["routing_status"] for row in rows) != Counter({"correct": 120, "abstain": 8}):
        raise RuntimeError("Blur audit routing statuses changed")
    flux_cache = resolve_workspace_path(ROOT, config["flux_cache"])
    missing_flux = [
        row["record_id"]
        for row in rows
        if row["routing_status"] == "correct"
        and not (flux_cache / f"{row['record_id']}.png").is_file()
    ]
    if missing_flux:
        raise RuntimeError(f"missing frozen FLUX cache records: {missing_flux[:5]}")
    rehearsal_sidecars = resolve_workspace_path(
        ROOT, "checkpoints/foundir_final_eleven_all_in_one_development_rehearsal_v3/per_record"
    )
    for row in rows:
        old = json.loads((rehearsal_sidecars / f"{row['record_id']}.json").read_text())
        height, width = int(row["height"]), int(row["width"])
        digest = bytes.fromhex(rehearsal_digest(config["crop_seed"], "mixed_restoration_crop", row["record_id"]))
        y = int.from_bytes(digest[:8], "big") % (height - int(config["crop_size"]) + 1)
        x = int.from_bytes(digest[8:16], "big") % (width - int(config["crop_size"]) + 1)
        if old.get("crop_xy") != [x, y]:
            raise RuntimeError(f"rehearsal crop identity changed: {row['record_id']}")
    return config, rows


def metric_pair(metric, target: torch.Tensor, pixels: np.ndarray, device: torch.device) -> dict[str, float]:
    prediction = tensor_m11(pixels)
    result = {
        "psnr": psnr(target, prediction.to(device)),
        "lpips": lpips_value(metric, target, prediction, device),
    }
    metric.reset()
    return result


def verify_sidecar(path: Path, *, row: Mapping[str, str], config_sha: str, checkpoint_sha: str) -> dict:
    payload = json.loads(path.read_text())
    expected = {
        "status": "complete",
        "config_sha256": config_sha,
        "motion_checkpoint_sha256": checkpoint_sha,
        "record_id": row["record_id"],
        "source_group": row["source_group"],
        "routing_status": row["routing_status"],
        "sealed_internal_test_accessed": False,
        "protected_foundir_test_accessed": False,
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        raise RuntimeError(f"Blur fusion sidecar identity changed: {path}")
    candidates = payload.get("candidates", {})
    if set(candidates) != {alpha_key(value) for value in EXPECTED_ALPHAS}:
        raise RuntimeError(f"Blur fusion candidate set changed: {path}")
    for values in [payload.get("input", {}), payload.get("raw_motion", {}), *candidates.values()]:
        if not all(isinstance(values.get(name), (int, float)) and math.isfinite(float(values[name])) for name in ("psnr", "lpips")):
            raise RuntimeError(f"invalid Blur fusion metrics: {path}")
    return payload


def means(rows: Sequence[Mapping[str, object]], field: str) -> dict[str, float]:
    return {
        "psnr": float(np.mean([row[field]["psnr"] for row in rows])),
        "lpips": float(np.mean([row[field]["lpips"] for row in rows])),
    }


def candidate_means(rows: Sequence[Mapping[str, object]], key: str) -> dict[str, float]:
    return {
        "psnr": float(np.mean([row["candidates"][key]["psnr"] for row in rows])),
        "lpips": float(np.mean([row["candidates"][key]["lpips"] for row in rows])),
    }


def margins(values: Mapping[str, float], sow: Mapping[str, float]) -> dict[str, object]:
    psnr_delta = float(values["psnr"]) - float(sow["psnr_minimum"])
    lpips_headroom = float(sow["lpips_maximum"]) - float(values["lpips"])
    return {
        "psnr_db": psnr_delta,
        "psnr_percent_of_sow": 100.0 * psnr_delta / float(sow["psnr_minimum"]),
        "lpips_headroom": lpips_headroom,
        "lpips_headroom_percent_of_maximum": 100.0 * lpips_headroom / float(sow["lpips_maximum"]),
        "passes_both": psnr_delta >= 0.0 and lpips_headroom >= 0.0,
    }


def build_result(rows: Sequence[Mapping[str, object]], config: Mapping[str, object], config_sha: str) -> dict:
    fixed = {key: candidate_means(rows, key) for key in map(alpha_key, EXPECTED_ALPHAS)}
    fixed_with_margins = {key: {**value, "sow_margin": margins(value, config["sow"])} for key, value in fixed.items()}
    passing = [key for key, value in fixed_with_margins.items() if value["sow_margin"]["passes_both"]]
    best_key = max(fixed, key=lambda key: min(
        (fixed[key]["psnr"] - config["sow"]["psnr_minimum"]) / config["sow"]["psnr_minimum"],
        (config["sow"]["lpips_maximum"] - fixed[key]["lpips"]) / config["sow"]["lpips_maximum"],
    ))
    keys = list(map(alpha_key, EXPECTED_ALPHAS))
    psnr_choice = [max(keys, key=lambda key: row["candidates"][key]["psnr"]) for row in rows]
    lpips_choice = [min(keys, key=lambda key: row["candidates"][key]["lpips"]) for row in rows]

    def selected_mean(choices: Sequence[str]) -> dict[str, float]:
        return {
            name: float(np.mean([row["candidates"][choice][name] for row, choice in zip(rows, choices, strict=True)]))
            for name in ("psnr", "lpips")
        }

    independent = {
        "psnr": float(np.mean([max(row["candidates"][key]["psnr"] for key in keys) for row in rows])),
        "lpips": float(np.mean([min(row["candidates"][key]["lpips"] for key in keys) for row in rows])),
    }
    independent["sow_margin"] = margins(independent, config["sow"])
    oracle = {
        "warning": "GT-dependent ceilings only; not deployable",
        "independent_metric_ceiling": independent,
        "same_per_image_alpha_selected_for_psnr": {**selected_mean(psnr_choice), "alpha_counts": dict(Counter(psnr_choice))},
        "same_per_image_alpha_selected_for_lpips": {**selected_mean(lpips_choice), "alpha_counts": dict(Counter(lpips_choice))},
    }
    oracle["same_per_image_alpha_selected_for_psnr"]["sow_margin"] = margins(oracle["same_per_image_alpha_selected_for_psnr"], config["sow"])
    oracle["same_per_image_alpha_selected_for_lpips"]["sow_margin"] = margins(oracle["same_per_image_alpha_selected_for_lpips"], config["sow"])
    return {
        "schema_version": 1,
        "status": "blur_motion_flux_fusion_development_audit_complete",
        "config_sha256": config_sha,
        "records": len(rows),
        "source_groups": len({row["source_group"] for row in rows}),
        "routing_status_counts": dict(Counter(row["routing_status"] for row in rows)),
        "input": means(rows, "input"),
        "raw_motion_all_records_diagnostic": means(rows, "raw_motion"),
        "fixed_alpha_route_compatible": fixed_with_margins,
        "selected_fixed_alpha": best_key,
        "passing_fixed_alphas": passing,
        "fixed_alpha_gate_pass": bool(passing),
        "oracle": oracle,
        "decision": "fixed_alpha_candidate_pass" if passing else ("fixed_alpha_failed_oracle_has_capacity" if independent["sow_margin"]["passes_both"] else "fusion_route_stopped_no_oracle_capacity"),
        "development_pixels_used_for_selection": True,
        "sealed_internal_test_accessed": False,
        "protected_foundir_test_accessed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    config_path = resolve_workspace_path(ROOT, args.config)
    config, rows = load_config(config_path)
    config_sha = file_sha256(config_path)
    checkpoint_path = resolve_workspace_path(ROOT, config["motion_checkpoint"])
    if args.preflight:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        expected = {
            "status": "foundir_task_native_adaptation_checkpoint",
            "config_sha256": EXPECTED_TRAINING_CONFIG_SHA,
            "task": "blur_motion",
            "step": 1000,
            "sealed_internal_test_accessed": False,
            "protected_foundir_test_accessed": False,
        }
        if any(payload.get(key) != value for key, value in expected.items()):
            raise RuntimeError("motion checkpoint payload identity changed")
        print("FOUNDIR_BLUR_FUSION_PREFLIGHT_PASS " + json.dumps({"config_sha256": config_sha, "records": len(rows)}, sort_keys=True))
        return

    rank, world, local_rank = torchrun_worker_identity(8)
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if torch.cuda.device_count() != world or not any(name in torch.cuda.get_device_properties(device).name for name in ("H100", "H200")):
        raise RuntimeError("Blur fusion audit requires eight H100/H200 GPUs")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    expected = {
        "status": "foundir_task_native_adaptation_checkpoint",
        "config_sha256": EXPECTED_TRAINING_CONFIG_SHA,
        "task": "blur_motion",
        "step": 1000,
        "sealed_internal_test_accessed": False,
        "protected_foundir_test_accessed": False,
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        raise RuntimeError("motion checkpoint payload identity changed")
    model = Restormer(layer_norm_type="WithBias").to(device)
    model.load_state_dict(payload["model"], strict=True)
    del payload
    model.eval()
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    metric = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(device).eval()
    output = resolve_workspace_path(ROOT, config["output"])
    sidecars = output / "per_record"
    motion_cache = output / "motion_cache"
    sidecars.mkdir(parents=True, exist_ok=True)
    motion_cache.mkdir(parents=True, exist_ok=True)
    flux_cache = resolve_workspace_path(ROOT, config["flux_cache"])
    crop_size, crop_seed = int(config["crop_size"]), int(config["crop_seed"])
    checkpoint_sha = config["motion_checkpoint_sha256"]
    for index in range(rank, len(rows), world):
        row = rows[index]
        stem = evaluation_artifact_stem(row["record_id"])
        sidecar = sidecars / f"{stem}.json"
        if sidecar.exists():
            verify_sidecar(sidecar, row=row, config_sha=config_sha, checkpoint_sha=checkpoint_sha)
            continue
        lq_m11, gt_m11, crop_xy, resized = load_rehearsal_crop(row, crop_size=crop_size, seed=crop_seed)
        input_u8 = np.rint(((lq_m11.numpy().transpose(1, 2, 0) + 1.0) * 127.5).clip(0, 255)).astype(np.uint8)
        target = gt_m11.unsqueeze(0).to(device)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            motion = model(((lq_m11 + 1.0) * 0.5).unsqueeze(0).to(device)).clamp_(0, 1)
        motion_u8 = quantize_01(motion)
        motion_path = motion_cache / f"{stem}.png"
        write_or_verify_png(motion_path, motion_u8, label="motion")
        # The all-in-one rehearsal predates evaluation_artifact_stem and stores
        # stage images by the manifest's already-unique record_id.
        flux_path = flux_cache / f"{row['record_id']}.png"
        flux_sha = None
        if row["routing_status"] == "correct":
            with Image.open(flux_path) as handle:
                flux_u8 = np.asarray(handle.convert("RGB"), dtype=np.uint8).copy()
            if flux_u8.shape != (crop_size, crop_size, 3):
                raise RuntimeError(f"invalid FLUX cache dimensions: {flux_path}")
            flux_sha = file_sha256(flux_path)
        else:
            flux_u8 = input_u8
        candidates = {}
        for alpha in EXPECTED_ALPHAS:
            if row["routing_status"] == "abstain":
                blended = input_u8
            else:
                blended = np.rint((1.0 - alpha) * flux_u8.astype(np.float32) + alpha * motion_u8.astype(np.float32)).clip(0, 255).astype(np.uint8)
            candidates[alpha_key(alpha)] = metric_pair(metric, target, blended, device)
        result = {
            "schema_version": 1,
            "status": "complete",
            "config_sha256": config_sha,
            "motion_checkpoint_sha256": checkpoint_sha,
            "record_id": row["record_id"],
            "source_group": row["source_group"],
            "routing_status": row["routing_status"],
            "crop_xy": crop_xy,
            "undersized_pair_resized": resized,
            "input": metric_pair(metric, target, input_u8, device),
            "raw_motion": metric_pair(metric, target, motion_u8, device),
            "candidates": candidates,
            "motion_png_sha256": file_sha256(motion_path),
            "flux_png_sha256": flux_sha,
            "sealed_internal_test_accessed": False,
            "protected_foundir_test_accessed": False,
        }
        write_once(sidecar, (json.dumps(result, indent=2, sort_keys=True) + "\n").encode())
        print(f"FOUNDIR_BLUR_FUSION rank={rank} record={index + 1}/{len(rows)}", flush=True)

    def finalize() -> None:
        expected_paths = [sidecars / f"{evaluation_artifact_stem(row['record_id'])}.json" for row in rows]
        if set(sidecars.glob("*.json")) != set(expected_paths):
            raise RuntimeError("Blur fusion sidecar set is incomplete or contaminated")
        final_rows = [verify_sidecar(path, row=row, config_sha=config_sha, checkpoint_sha=checkpoint_sha) for row, path in zip(rows, expected_paths, strict=True)]
        result = build_result(final_rows, config, config_sha)
        write_once(output / "result.json", (json.dumps(result, indent=2, sort_keys=True) + "\n").encode())
        print("FOUNDIR_BLUR_FUSION_COMPLETE " + json.dumps(result, sort_keys=True), flush=True)

    synchronize_rank_zero_finalization(
        finalize,
        rank=rank,
        world=world,
        output=output,
        identity={"config_sha256": config_sha, "checkpoint_sha256": checkpoint_sha, "records": len(rows)},
        phase="Blur motion FLUX fusion development audit",
        timeout_seconds=2 * 60 * 60,
    )


if __name__ == "__main__":
    main()
