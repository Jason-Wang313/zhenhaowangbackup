#!/usr/bin/env python3
"""Measure development-only FLUX inference headroom on isolated SOW tasks."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party/flux2/src"))

from lowlight_restoration.flux_prompt_cache import PromptContextCache, file_sha256  # noqa: E402
from lowlight_restoration.foundir_flux2_protocol import (  # noqa: E402
    SOW_ISOLATED_CATEGORIES,
    aggregate_task_metrics,
    evaluation_artifact_stem,
    resolve_workspace_path,
    stable_digest,
)
from lowlight_restoration.lora import inject_lora, load_restoration_checkpoint  # noqa: E402
from lowlight_restoration.pixel_refinement import BoundedPixelRefiner, pixel_refiner_parameter_count  # noqa: E402
from smoke.evaluate_foundir_five_type_flux2_ddp import (  # noqa: E402
    load_aligned_crop,
    lpips_value,
    quantized_rgb,
    rgb_structural_similarity,
    selected_from_contract,
    synchronize_rank_zero_finalization,
    torchrun_worker_identity,
    write_once,
)
from smoke.train_foundir_five_type_flux2_full_ddp import load_rows  # noqa: E402
from smoke.train_foundir_five_type_flux2_overfit_ddp import psnr, restore  # noqa: E402


def validate_diagnostic(config: Mapping[str, object], path: Path) -> tuple[dict, Path, Path]:
    if (
        config.get("schema_version") != 1
        or config.get("status") != "development_only_inference_diagnostic"
        or config.get("split") != "validation"
        or config.get("sealed_internal_test_accessed") is not False
        or config.get("protected_foundir_test_accessed") is not False
    ):
        raise RuntimeError("inference diagnostic contract is invalid")
    parent_path = resolve_workspace_path(ROOT, str(config["parent_config"]))
    checkpoint = resolve_workspace_path(ROOT, str(config["checkpoint"]))
    if file_sha256(parent_path) != config.get("parent_config_sha256"):
        raise RuntimeError("parent FLUX config changed")
    if file_sha256(checkpoint) != config.get("checkpoint_sha256"):
        raise RuntimeError("diagnostic checkpoint changed")
    if set(config.get("categories", ())) != set(SOW_ISOLATED_CATEGORIES.values()):
        raise RuntimeError("diagnostic must cover exactly five isolated SOW categories")
    variants = config.get("variants")
    if not isinstance(variants, list) or not variants:
        raise RuntimeError("diagnostic variants are missing")
    names = []
    for variant in variants:
        if not isinstance(variant, dict):
            raise RuntimeError("diagnostic variant must be an object")
        name = str(variant.get("name", ""))
        steps = variant.get("steps")
        noise = variant.get("anchor_noise_scale")
        if (
            not name
            or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789_" for character in name)
            or isinstance(steps, bool)
            or not isinstance(steps, int)
            or steps < 1
            or isinstance(noise, bool)
            or not isinstance(noise, (int, float))
            or not 0.0 <= float(noise) <= 0.25
        ):
            raise RuntimeError(f"invalid diagnostic variant: {variant}")
        names.append(name)
    if len(names) != len(set(names)):
        raise RuntimeError("diagnostic variant names are not unique")
    return json.loads(parent_path.read_text()), parent_path, checkpoint


def quantized_tensor(value: torch.Tensor) -> tuple[np.ndarray, torch.Tensor]:
    pixels = quantized_rgb(value)
    tensor = (
        torch.from_numpy(pixels.copy())
        .permute(2, 0, 1)
        .float()
        .div_(127.5)
        .sub_(1.0)
        .unsqueeze(0)
    )
    return pixels, tensor


def path_metrics(metric, gt: torch.Tensor, prediction: torch.Tensor, device: torch.device) -> dict[str, float]:
    pixels, quantized = quantized_tensor(prediction)
    result = {
        "psnr": psnr(gt, quantized.to(device)),
        "lpips": lpips_value(metric, gt, quantized, device),
        "ssim": rgb_structural_similarity(
            ((gt[0].float().cpu().numpy().transpose(1, 2, 0) + 1.0) * 0.5).clip(0, 1),
            pixels.astype(np.float32) / 255.0,
        ),
    }
    if not all(math.isfinite(value) for value in result.values()):
        raise RuntimeError("diagnostic produced non-finite metrics")
    return result


def aggregate_path(rows: Sequence[Mapping[str, object]], path: str) -> dict[str, dict[str, float]]:
    return aggregate_task_metrics(
        [
            {
                "category": row["category"],
                "psnr": row[path]["psnr"],
                "lpips": row[path]["lpips"],
            }
            for row in rows
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    diagnostic_path = resolve_workspace_path(ROOT, args.config)
    diagnostic = json.loads(diagnostic_path.read_text())
    parent, parent_path, checkpoint = validate_diagnostic(diagnostic, diagnostic_path)
    diagnostic_sha = file_sha256(diagnostic_path)
    parent_sha = file_sha256(parent_path)
    checkpoint_sha = file_sha256(checkpoint)
    conditions_path = resolve_workspace_path(ROOT, parent["full_conditions"]["csv"])
    conditions_sha = file_sha256(conditions_path)
    rows = load_rows(conditions_path, int(parent["data"]["expected_five_type_records"]))
    selected = selected_from_contract(
        parent,
        rows,
        split="validation",
        config_sha=parent_sha,
        conditions_sha=conditions_sha,
    )
    selected = [row for row in selected if row["category"] in diagnostic["categories"]]
    counts = defaultdict(int)
    for row in selected:
        counts[row["category"]] += 1
    expected = int(diagnostic["records_per_category"])
    if set(counts) != set(diagnostic["categories"]) or any(value != expected for value in counts.values()):
        raise RuntimeError(f"isolated development coverage changed: {dict(counts)}")

    rank, world, local_rank = torchrun_worker_identity()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    output = resolve_workspace_path(ROOT, diagnostic["output"])
    sidecars = output / "per_record"
    sidecars.mkdir(parents=True, exist_ok=True)
    os.environ.update(
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        KLEIN_4B_BASE_MODEL_PATH=str(resolve_workspace_path(ROOT, parent["model"]["flow_weights"])),
        AE_MODEL_PATH=str(resolve_workspace_path(ROOT, parent["model"]["autoencoder_weights"])),
    )
    try:
        from flux2.sampling import batched_prc_img
        from flux2.util import load_ae, load_flow_model
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

        autoencoder = load_ae(parent["model"]["name"], device=device).eval().requires_grad_(False)
        model = load_flow_model(parent["model"]["name"], device=device).eval().requires_grad_(False)
        names = inject_lora(
            model,
            list(parent["lora"]["target_patterns"]),
            rank=int(parent["lora"]["rank"]),
            alpha=float(parent["lora"]["alpha"]),
            dropout=float(parent["lora"]["dropout"]),
        )
        if len(names) != int(parent["lora"]["expected_module_count"]):
            raise RuntimeError("diagnostic LoRA module count changed")
        refinement = parent["pixel_refinement"]
        pixel_refiner = BoundedPixelRefiner(
            width=int(refinement["width"]),
            blocks=int(refinement["blocks"]),
            maximum_residual_m11=float(refinement["maximum_residual_m11"]),
        ).to(device=device, dtype=torch.bfloat16).eval().requires_grad_(False)
        if pixel_refiner_parameter_count(pixel_refiner) != int(refinement["expected_trainable_parameters"]):
            raise RuntimeError("diagnostic refiner parameter count changed")
        load_restoration_checkpoint(model, pixel_refiner, checkpoint)
        cache = PromptContextCache(
            resolve_workspace_path(ROOT, parent["prompt_cache"]["index"]),
            maximum_resident=int(parent["prompt_cache"]["maximum_resident_contexts_per_rank"]),
        )
        metric = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(device).eval()
        for index in range(rank, len(selected), world):
            record = selected[index]
            lq, gt, crop_xy = load_aligned_crop(
                record,
                crop_size=int(parent["data"]["crop_size"]),
                seed=int(parent["data"]["split_seed"]),
            )
            lq_batch = lq.unsqueeze(0).to(device)
            gt_batch = gt.unsqueeze(0).to(device)
            context, context_ids = cache.get(record["qwen_text"], device)
            input_metrics = path_metrics(metric, gt_batch, lq_batch, device)
            for variant in diagnostic["variants"]:
                stem = evaluation_artifact_stem(record["record_id"])
                sidecar = sidecars / str(variant["name"]) / f"{stem}.json"
                identity = {
                    "schema_version": 1,
                    "status": "complete",
                    "diagnostic_config_sha256": diagnostic_sha,
                    "parent_config_sha256": parent_sha,
                    "checkpoint_sha256": checkpoint_sha,
                    "conditions_sha256": conditions_sha,
                    "record_id": record["record_id"],
                    "source_group": record["source_group"],
                    "category": record["category"],
                    "evaluation_fold": int(record["evaluation_fold"]),
                    "lq_sha256": record["lq_sha256"],
                    "gt_sha256": record["gt_sha256"],
                    "variant": dict(variant),
                    "crop_xy": crop_xy,
                    "prompt_sha256": hashlib.sha256(record["qwen_text"].encode()).hexdigest(),
                    "input": input_metrics,
                    "sealed_internal_test_accessed": False,
                    "protected_foundir_test_accessed": False,
                }
                if sidecar.exists():
                    payload = json.loads(sidecar.read_text())
                    if any(payload.get(key) != value for key, value in identity.items()):
                        raise RuntimeError(f"diagnostic sidecar identity changed: {sidecar}")
                    continue
                decoded, refined = restore(
                    model,
                    autoencoder,
                    lq_batch,
                    context,
                    context_ids,
                    batched_prc_img,
                    seed=int(stable_digest(int(parent["flow_training"]["seed"]), record["record_id"], "evaluation")[:16], 16),
                    steps=int(variant["steps"]),
                    noise_scale=float(variant["anchor_noise_scale"]),
                    reference_time=int(parent["flow_training"]["reference_time_coordinate"]),
                    pixel_refiner=pixel_refiner,
                    return_decoded=True,
                )
                payload = {
                    **identity,
                    "decoded": path_metrics(metric, gt_batch, decoded, device),
                    "refined": path_metrics(metric, gt_batch, refined, device),
                }
                write_once(sidecar, (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode())
                print(
                    f"FOUNDIR_INFERENCE_DIAGNOSTIC rank={rank} variant={variant['name']} "
                    f"record={index + 1}/{len(selected)} category={record['category']} "
                    f"decoded_psnr={payload['decoded']['psnr']:.4f} refined_psnr={payload['refined']['psnr']:.4f}",
                    flush=True,
                )

        def finalize() -> None:
            summaries = {}
            for variant in diagnostic["variants"]:
                result_rows = []
                for record in selected:
                    stem = evaluation_artifact_stem(record["record_id"])
                    path = sidecars / str(variant["name"]) / f"{stem}.json"
                    payload = json.loads(path.read_text())
                    if payload.get("diagnostic_config_sha256") != diagnostic_sha:
                        raise RuntimeError(f"invalid diagnostic sidecar: {path}")
                    result_rows.append(payload)
                summaries[str(variant["name"])] = {
                    "variant": dict(variant),
                    "records": len(result_rows),
                    "input_task_metrics": aggregate_path(result_rows, "input"),
                    "decoded_task_metrics": aggregate_path(result_rows, "decoded"),
                    "refined_task_metrics": aggregate_path(result_rows, "refined"),
                }
            result = {
                "schema_version": 1,
                "status": "development_only_inference_diagnostic_complete",
                "metric_contract": "sow_isolated_categories_v1",
                "diagnostic_config": diagnostic_path.relative_to(ROOT).as_posix(),
                "diagnostic_config_sha256": diagnostic_sha,
                "parent_config_sha256": parent_sha,
                "checkpoint": checkpoint.relative_to(ROOT).as_posix(),
                "checkpoint_sha256": checkpoint_sha,
                "split": "validation",
                "summaries": summaries,
                "checkpoint_selection_performed": False,
                "sealed_internal_test_accessed": False,
                "protected_foundir_test_accessed": False,
            }
            write_once(output / "result.json", (json.dumps(result, indent=2, sort_keys=True) + "\n").encode())
            print("FOUNDIR_INFERENCE_DIAGNOSTIC_PASS " + json.dumps(result, sort_keys=True), flush=True)

        synchronize_rank_zero_finalization(
            finalize,
            rank=rank,
            world=world,
            output=output,
            identity={
                "diagnostic_config_sha256": diagnostic_sha,
                "checkpoint_sha256": checkpoint_sha,
                "split": "validation",
            },
            phase="inference diagnostic finalization",
            timeout_seconds=float(parent["distributed"]["timeout_minutes"]) * 60,
        )
    finally:
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
