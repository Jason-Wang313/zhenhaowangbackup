#!/usr/bin/env python3
"""One-shot full-development confirmation of frozen alpha=.75 FLUX correction."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party/flux2/src"))

from lowlight_restoration.lora import inject_lora, load_lora_checkpoint  # noqa: E402
from lowlight_restoration.scalar_residual import quantized_scalar_residual  # noqa: E402
from smoke.evaluate_foundir_blur_only_flux2_full_development_ddp import (  # noqa: E402
    BlurDevelopmentDataset,
    RankSampler,
    collate,
    load_development_rows,
    load_json,
    restore_batch,
    sha256,
    stable_digest,
    workspace_path,
    write_once,
)
from smoke.train_foundir_blur_only_flux2_overfit_ddp import encode_prompt, lpips_value, psnr  # noqa: E402


EXPECTED_CONFIG_SHA = "b756c7b223d8aed8e2a7018eec8e2e395042f772e477584b79ba2a98c7a12da5"


def validate_config(path: Path, *, verify_images: bool) -> tuple[dict[str, object], dict[str, object], list[dict[str, str]]]:
    config = load_json(path)
    if sha256(path) != EXPECTED_CONFIG_SHA or config.get("status") != "user_authorized_frozen_alpha_full_development_confirmation":
        raise RuntimeError("frozen scalar full-development config changed")
    if sha256(workspace_path(config["authorization_evidence"]["feasibility_result"])) != config["authorization_evidence"]["feasibility_result_sha256"]:
        raise RuntimeError("feasibility evidence changed")
    scope, model, inference, gate = config["scope"], config["model"], config["inference"], config["gate"]
    forbidden = ("qwen_degradation_conditioning", "mixed_degradations", "model_weight_updates", "pixel_refiner", "decoder_tuning", "distillation", "development_gt_optimization", "sealed_internal_accessed", "official_foundir_test_accessed")
    if any(scope.get(key) is not False for key in forbidden):
        raise RuntimeError("protected scope changed")
    if float(inference["frozen_alpha"]) != 0.75 or inference["seed_namespace"] != "blur_full_development" or int(inference["seed"]) != 2026091113 or int(config["data"]["crop_seed"]) != 2026091111:
        raise RuntimeError("alpha/crop/inference identity changed")
    if gate["full_scale_training_automatic"] is not False or gate["require_visual_review"] is not True:
        raise RuntimeError("terminal stop or visual gate changed")
    parent_path, checkpoint = workspace_path(model["parent_config"]), workspace_path(model["checkpoint"])
    manifest = workspace_path(config["data"]["source_manifest"])
    for frozen, expected in ((parent_path, model["parent_config_sha256"]), (checkpoint, model["checkpoint_sha256"]), (manifest, config["data"]["source_manifest_sha256"])):
        if not frozen.is_file() or sha256(frozen) != expected:
            raise RuntimeError(f"frozen dependency changed: {frozen}")
    parent = load_json(parent_path)
    rows = load_development_rows(config)
    if len(rows) != 2595 or len({row["source_group"] for row in rows}) != 165:
        raise RuntimeError("full development cardinality changed")
    visual_ids = list(config["visual_record_ids"])
    if len(visual_ids) != 16 or len(set(visual_ids)) != 16 or not set(visual_ids) <= {row["record_id"] for row in rows}:
        raise RuntimeError("fixed visual IDs changed")
    if verify_images:
        for row in rows:
            for field, hash_field in (("lq_path", "lq_sha256"), ("gt_path", "gt_sha256")):
                image = workspace_path(row[field])
                if not image.is_file() or sha256(image) != row[hash_field]:
                    raise RuntimeError(f"development image changed: {row['record_id']} {field}")
    return config, parent, rows


def metrics_summary(rows: list[dict[str, object]], role: str) -> dict[str, float]:
    return {
        "psnr": float(np.mean([row[role]["psnr"] for row in rows])),
        "lpips": float(np.mean([row[role]["lpips"] for row in rows])),
    }


def save_triptych(path: Path, lq: torch.Tensor, corrected: torch.Tensor, gt: torch.Tensor) -> None:
    arrays = [value.detach().cpu().add(1).mul(127.5).round().clamp(0, 255).byte().permute(1, 2, 0).numpy() for value in (lq, corrected, gt)]
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.concatenate(arrays, axis=1), mode="RGB").save(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    config_path = workspace_path(args.config)
    config, parent, rows = validate_config(config_path, verify_images=args.preflight)
    if args.preflight:
        print(json.dumps({"status": "FLUX2_SCALAR_FULL_DEVELOPMENT_PREFLIGHT_PASS", "records": 2595, "source_groups": 165, "frozen_alpha": 0.75, "official_foundir_test_accessed": False}, sort_keys=True))
        return 0

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend=config["distributed"]["backend"], device_id=device)
    started = time.monotonic()
    try:
        rank, world = dist.get_rank(), dist.get_world_size()
        if world != 8 or torch.cuda.device_count() != 8:
            raise RuntimeError("full development requires exactly eight visible GPUs")
        props = torch.cuda.get_device_properties(device)
        profiles = [name for name in ("H100", "H200") if name in props.name]
        if len(profiles) != 1 or props.total_memory < int(config["distributed"]["minimum_memory_gib"]) * 1024**3:
            raise RuntimeError(f"unsupported GPU: {props.name}")
        gathered: list[object | None] = [None] * world
        dist.all_gather_object(gathered, profiles[0])
        if set(gathered) != {profiles[0]}:
            raise RuntimeError(f"mixed GPU profiles forbidden: {gathered}")
        profile = profiles[0]
        output = workspace_path(config["runtime"]["output_dir"])
        sidecar_dir, visual_dir = output / "per_record", output / "fixed_visuals"
        sidecar_dir.mkdir(parents=True, exist_ok=True)
        config_sha = sha256(config_path)
        os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TORCH_HOME=str(workspace_path(config["runtime"]["torch_home"])), KLEIN_4B_BASE_MODEL_PATH=str(workspace_path(parent["model"]["flow_weights"])), AE_MODEL_PATH=str(workspace_path(parent["model"]["autoencoder_weights"])), QWEN3_MODEL_PATH=str(workspace_path(parent["model"]["text_encoder"])))
        _, _, cfg_context, cfg_ids = encode_prompt(parent, device)
        from flux2.sampling import batched_prc_img
        from flux2.util import load_ae, load_flow_model
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

        autoencoder = load_ae(parent["model"]["name"], device=device).eval().requires_grad_(False)
        model = load_flow_model(parent["model"]["name"], device=device).eval().requires_grad_(False)
        names = inject_lora(model, list(parent["lora"]["target_patterns"]), rank=int(parent["lora"]["rank"]), alpha=float(parent["lora"]["alpha"]), dropout=float(parent["lora"]["dropout"]))
        if len(names) != int(parent["lora"]["expected_module_count"]):
            raise RuntimeError("LoRA target count changed")
        load_lora_checkpoint(model, workspace_path(config["model"]["checkpoint"]))
        metric = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(device).eval()
        dataset = BlurDevelopmentDataset(rows, crop_size=int(config["data"]["crop_size"]), seed=int(config["data"]["crop_seed"]))
        loader = DataLoader(dataset, batch_size=int(config["distributed"]["evaluation_batch_size_per_rank"][profile]), sampler=RankSampler(len(rows), rank, world), num_workers=int(config["distributed"]["data_loader_workers_per_rank"]), prefetch_factor=2, persistent_workers=True, pin_memory=True, collate_fn=collate)
        visual_ids = set(config["visual_record_ids"])
        for batch_index, batch in enumerate(loader):
            missing = [position for position, record_id in enumerate(batch["record_id"]) if not (sidecar_dir / f"{record_id}.json").exists()]
            if not missing:
                continue
            selection = torch.tensor(missing, dtype=torch.long)
            lq = batch["lq"].index_select(0, selection).to(device, non_blocking=True)
            gt = batch["gt"].index_select(0, selection).to(device, non_blocking=True)
            record_ids = [batch["record_id"][position] for position in missing]
            seeds = [int(stable_digest(config["inference"]["seed"], config["inference"]["seed_namespace"], record_id)[:16], 16) for record_id in record_ids]
            restored = restore_batch(model, autoencoder, lq, cfg_context, cfg_ids, batched_prc_img, seeds=seeds, steps=int(config["inference"]["num_steps"]), reference_time=int(config["inference"]["reference_time_coordinate"]))
            base_u8, base = quantized_scalar_residual(lq, restored, 1.0)
            candidate_u8, candidate = quantized_scalar_residual(lq, base, float(config["inference"]["frozen_alpha"]))
            for local, position in enumerate(missing):
                values = {}
                for role, image in (("input", lq), ("base", base), ("candidate", candidate)):
                    values[role] = {"psnr": psnr(gt[local:local+1], image[local:local+1]), "lpips": lpips_value(metric, gt[local:local+1], image[local:local+1])}
                    metric.reset()
                base_bytes = base_u8[local].permute(1, 2, 0).contiguous().cpu().numpy().tobytes()
                candidate_bytes = candidate_u8[local].permute(1, 2, 0).contiguous().cpu().numpy().tobytes()
                payload = {"schema_version": 1, "status": "complete", "config_sha256": config_sha, "checkpoint_sha256": config["model"]["checkpoint_sha256"], "record_id": record_ids[local], "source_group": batch["source_group"][position], "crop_xy": batch["crop_xy"][position], "inference_seed": seeds[local], **values, "base_uint8_sha256": hashlib.sha256(base_bytes).hexdigest(), "candidate_uint8_sha256": hashlib.sha256(candidate_bytes).hexdigest(), "frozen_alpha": 0.75, "development_gt_optimization": False, "sealed_internal_accessed": False, "official_foundir_test_accessed": False}
                write_once(sidecar_dir / f"{record_ids[local]}.json", payload)
                if record_ids[local] in visual_ids:
                    save_triptych(visual_dir / f"{record_ids[local]}.png", lq[local], candidate[local], gt[local])
            if rank == 0 and (batch_index == 0 or (batch_index + 1) % 10 == 0):
                print(f"FLUX2_SCALAR_FULL_DEVELOPMENT rank0_batch={batch_index+1}/{len(loader)}", flush=True)
        dist.barrier()
        if rank == 0:
            paths = [sidecar_dir / f"{row['record_id']}.json" for row in rows]
            if any(not path.is_file() for path in paths) or set(sidecar_dir.glob("*.json")) != set(paths) or len(list(visual_dir.glob("*.png"))) != 16:
                raise RuntimeError("full-development artifacts incomplete or contaminated")
            payloads = [load_json(path) for path in paths]
            base, candidate = metrics_summary(payloads, "base"), metrics_summary(payloads, "candidate")
            deltas = np.asarray([row["candidate"]["psnr"] - row["base"]["psnr"] for row in payloads])
            lpips_deltas = np.asarray([row["candidate"]["lpips"] - row["base"]["lpips"] for row in payloads])
            comparison = {"psnr_gain_over_base_db": float(deltas.mean()), "lpips_regression_vs_base": float(lpips_deltas.mean()), "psnr_win_fraction_vs_base": float((deltas > 0).mean()), "maximum_per_record_psnr_regression_vs_base_db": float((-deltas).max())}
            gate = config["gate"]
            checks = {"base_replay_psnr": abs(base["psnr"] - float(gate["base_replay_expected_psnr"])) <= float(gate["base_replay_psnr_tolerance"]), "base_replay_lpips": abs(base["lpips"] - float(gate["base_replay_expected_lpips"])) <= float(gate["base_replay_lpips_tolerance"]), "sow_psnr": candidate["psnr"] >= float(gate["sow_psnr_minimum"]), "sow_lpips": candidate["lpips"] <= float(gate["sow_lpips_maximum"]), "minimum_psnr_gain": comparison["psnr_gain_over_base_db"] >= float(gate["minimum_psnr_gain_over_base_db"]), "maximum_lpips_regression": comparison["lpips_regression_vs_base"] <= float(gate["maximum_lpips_regression_vs_base"]), "minimum_psnr_win_fraction": comparison["psnr_win_fraction_vs_base"] >= float(gate["minimum_psnr_win_fraction_vs_base"]), "maximum_per_record_psnr_regression": comparison["maximum_per_record_psnr_regression_vs_base_db"] <= float(gate["maximum_per_record_psnr_regression_vs_base_db"])}
            result = {"schema_version": 1, "status": "awaiting_human_visual_review" if all(checks.values()) else "failed", "automatic_metric_gate_pass": all(checks.values()), "human_visual_review_required": True, "human_visual_review_complete": False, "config": config_path.relative_to(ROOT).as_posix(), "config_sha256": config_sha, "checkpoint": config["model"]["checkpoint"], "checkpoint_sha256": config["model"]["checkpoint_sha256"], "model": "FLUX.2 Klein Base 4B isolated-Blur step-10000 plus frozen alpha=.75 scalar residual", "split": "all 2595 source-group-disjoint Blur development records", "records": 2595, "source_groups": 165, "frozen_alpha": 0.75, "base_metrics": base, "candidate_metrics": candidate, "comparison": comparison, "checks": checks, "visual_records": list(config["visual_record_ids"]), "full_scale_training_automatic": False, "qwen_degradation_conditioning": False, "mixed_degradations": False, "model_weight_updates": False, "decoder_tuning": False, "distillation": False, "development_gt_optimization": False, "sealed_internal_accessed": False, "official_foundir_test_accessed": False, "hardware_profile": profile, "gpu_name": props.name, "world_size": world, "wall_seconds": time.monotonic() - started}
            write_once(output / "result.json", result)
            print("FLUX2_SCALAR_FULL_DEVELOPMENT_RESULT " + json.dumps({"pass": result["automatic_metric_gate_pass"], "base": base, "candidate": candidate, "comparison": comparison, "checks": checks}, sort_keys=True), flush=True)
        dist.barrier()
        result = load_json(workspace_path(config["runtime"]["output_dir"]) / "result.json")
        return 0 if result["automatic_metric_gate_pass"] is True else 2
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
