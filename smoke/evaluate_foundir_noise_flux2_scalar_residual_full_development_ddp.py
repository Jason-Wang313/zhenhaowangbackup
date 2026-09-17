#!/usr/bin/env python3
"""One-shot full-development Noise evaluation with frozen scalar alpha=.30."""

from __future__ import annotations

import argparse
import csv
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
    load_json,
    restore_batch,
    sha256,
    stable_digest,
    workspace_path,
    write_once,
)
from smoke.train_foundir_blur_only_flux2_overfit_ddp import (  # noqa: E402
    encode_prompt,
    lpips_value,
    materialize_task_config,
    psnr,
)


EXPECTED_CONFIG_SHA = "5e7a356ea89d21f7b60b4b736f851ef26a0216d8555d73846f527b958b5b7dd6"


def load_noise_rows(config: dict[str, object]) -> list[dict[str, str]]:
    data = config["data"]
    manifest = workspace_path(data["source_manifest"])
    if sha256(manifest) != data["source_manifest_sha256"]:
        raise RuntimeError("FoundIR pair manifest changed")
    with manifest.open(newline="") as handle:
        all_rows = list(csv.DictReader(handle))
    isolated = [row for row in all_rows if row["category"] == "05Noise" and row["composition_size"] == "1"]
    rows = sorted((dict(row) for row in isolated if row["split"] == "validation"), key=lambda row: row["record_id"])
    train_groups = {row["source_group"] for row in isolated if row["split"] == "train"}
    groups = {row["source_group"] for row in rows}
    if (
        len(rows) != int(data["expected_records"])
        or len(groups) != int(data["expected_source_groups"])
        or len({row["record_id"] for row in rows}) != len(rows)
        or groups & train_groups
    ):
        raise RuntimeError("isolated Noise development cardinality or source isolation changed")
    for row in rows:
        if min(int(row["width"]), int(row["height"])) < int(data["crop_size"]):
            raise RuntimeError(f"undersized isolated Noise development record: {row['record_id']}")
        if "/" in row["record_id"] or ".." in row["record_id"]:
            raise RuntimeError(f"unsafe record ID: {row['record_id']}")
    return rows


def validate_config(path: Path, *, verify_images: bool) -> tuple[dict[str, object], dict[str, object], list[dict[str, str]]]:
    config = load_json(path)
    if sha256(path) != EXPECTED_CONFIG_SHA or config.get("status") != "user_authorized_noise_frozen_alpha_full_development":
        raise RuntimeError("frozen Noise full-development config changed")
    for key, hash_key in (("capacity_result", "capacity_result_sha256"), ("scalar_read_only_audit", "scalar_read_only_audit_sha256")):
        if sha256(workspace_path(config["evidence"][key])) != config["evidence"][hash_key]:
            raise RuntimeError(f"Noise selection evidence changed: {key}")
    scope = config["scope"]
    forbidden = ("qwen_degradation_conditioning", "mixed_degradations", "model_weight_updates", "pixel_refiner", "decoder_tuning", "distillation", "development_gt_optimization", "sealed_internal_accessed", "official_foundir_test_accessed")
    if scope.get("category") != "05Noise" or any(scope.get(key) is not False for key in forbidden):
        raise RuntimeError("protected Noise scope changed")
    inference, gate = config["inference"], config["gate"]
    if (
        float(inference["frozen_alpha"]) != 0.3
        or inference["formula"] != "lq_plus_alpha_times_flux_minus_lq"
        or gate["formal_checks"] != ["sow_psnr", "sow_lpips"]
        or gate["tail_statistics_descriptive_only"] is not True
        or gate["automatic_follow_on"] is not False
    ):
        raise RuntimeError("Noise alpha or SOW-only gate changed")
    model = config["model"]
    parent_path, checkpoint = workspace_path(model["parent_config"]), workspace_path(model["checkpoint"])
    for frozen, expected in ((parent_path, model["parent_config_sha256"]), (checkpoint, model["checkpoint_sha256"])):
        if not frozen.is_file() or sha256(frozen) != expected:
            raise RuntimeError(f"frozen dependency changed: {frozen}")
    parent = materialize_task_config(load_json(parent_path), "05Noise")
    if parent["scope"]["category"] != "05Noise" or parent["model"]["prompt"] != load_json(parent_path)["tasks"]["05Noise"]["prompt"]:
        raise RuntimeError("Noise parent materialization changed")
    rows = load_noise_rows(config)
    visual_ids = list(config["visual_record_ids"])
    if len(visual_ids) != 16 or len(set(visual_ids)) != 16 or not set(visual_ids) <= {row["record_id"] for row in rows}:
        raise RuntimeError("fixed Noise visual IDs changed")
    if verify_images:
        for row in rows:
            for field, hash_field in (("lq_path", "lq_sha256"), ("gt_path", "gt_sha256")):
                image = workspace_path(row[field])
                if not image.is_file() or sha256(image) != row[hash_field]:
                    raise RuntimeError(f"Noise development image changed: {row['record_id']} {field}")
    return config, parent, rows


def mean_metrics(rows: list[dict[str, object]], role: str) -> dict[str, float]:
    return {"psnr": float(np.mean([row[role]["psnr"] for row in rows])), "lpips": float(np.mean([row[role]["lpips"] for row in rows]))}


def save_triptych(path: Path, lq: torch.Tensor, candidate: torch.Tensor, gt: torch.Tensor) -> None:
    arrays = [value.detach().cpu().add(1).mul(127.5).round().clamp(0, 255).byte().permute(1, 2, 0).numpy() for value in (lq, candidate, gt)]
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
        print(json.dumps({"status": "NOISE_SCALAR_FULL_DEVELOPMENT_PREFLIGHT_PASS", "records": len(rows), "source_groups": len({row['source_group'] for row in rows}), "frozen_alpha": 0.3, "official_foundir_test_accessed": False}, sort_keys=True))
        return 0

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend=config["distributed"]["backend"], device_id=device)
    started = time.monotonic()
    try:
        rank, world = dist.get_rank(), dist.get_world_size()
        if world != 8 or torch.cuda.device_count() != 8:
            raise RuntimeError("Noise full development requires exactly eight visible GPUs")
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
                payload = {"schema_version": 1, "status": "complete", "config_sha256": config_sha, "checkpoint_sha256": config["model"]["checkpoint_sha256"], "record_id": record_ids[local], "source_group": batch["source_group"][position], "crop_xy": batch["crop_xy"][position], "inference_seed": seeds[local], **values, "base_uint8_sha256": hashlib.sha256(base_u8[local].permute(1, 2, 0).contiguous().cpu().numpy().tobytes()).hexdigest(), "candidate_uint8_sha256": hashlib.sha256(candidate_u8[local].permute(1, 2, 0).contiguous().cpu().numpy().tobytes()).hexdigest(), "frozen_alpha": 0.3, "development_gt_optimization": False, "sealed_internal_accessed": False, "official_foundir_test_accessed": False}
                write_once(sidecar_dir / f"{record_ids[local]}.json", payload)
                if record_ids[local] in visual_ids:
                    save_triptych(visual_dir / f"{record_ids[local]}.png", lq[local], candidate[local], gt[local])
            if rank == 0 and (batch_index == 0 or (batch_index + 1) % 10 == 0):
                print(f"NOISE_SCALAR_FULL_DEVELOPMENT rank0_batch={batch_index+1}/{len(loader)}", flush=True)
        dist.barrier()
        if rank == 0:
            paths = [sidecar_dir / f"{row['record_id']}.json" for row in rows]
            if any(not path.is_file() for path in paths) or set(sidecar_dir.glob("*.json")) != set(paths) or len(list(visual_dir.glob("*.png"))) != 16:
                raise RuntimeError("Noise full-development artifacts incomplete or contaminated")
            payloads = [load_json(path) for path in paths]
            input_metrics, base, candidate = (mean_metrics(payloads, role) for role in ("input", "base", "candidate"))
            vs_input = np.asarray([row["candidate"]["psnr"] - row["input"]["psnr"] for row in payloads])
            vs_base = np.asarray([row["candidate"]["psnr"] - row["base"]["psnr"] for row in payloads])
            tail = {"mean_psnr_delta_vs_input_db": float(vs_input.mean()), "psnr_win_fraction_vs_input": float((vs_input > 0).mean()), "records_regressing_more_than_1db_vs_input": int((vs_input < -1).sum()), "records_regressing_more_than_2db_vs_input": int((vs_input < -2).sum()), "worst_psnr_regression_vs_input_db": float((-vs_input).max()), "psnr_gain_over_base_db": float(vs_base.mean()), "psnr_win_fraction_vs_base": float((vs_base > 0).mean()), "worst_psnr_regression_vs_base_db": float((-vs_base).max())}
            formal_checks = {"sow_psnr": candidate["psnr"] >= float(config["gate"]["sow_psnr_minimum"]), "sow_lpips": candidate["lpips"] <= float(config["gate"]["sow_lpips_maximum"])}
            passed = all(formal_checks.values())
            result = {"schema_version": 1, "status": "awaiting_human_visual_review" if passed else "failed", "automatic_metric_gate_pass": passed, "formal_acceptance_checks": formal_checks, "tail_statistics_descriptive_only": True, "human_visual_review_required": True, "human_visual_review_complete": False, "config": config_path.relative_to(ROOT).as_posix(), "config_sha256": config_sha, "checkpoint": config["model"]["checkpoint"], "checkpoint_sha256": config["model"]["checkpoint_sha256"], "model": "FLUX.2 Klein Base 4B isolated-Noise step-400 plus frozen alpha=.30 scalar residual", "split": "all 1859 source-group-disjoint isolated-Noise development records", "records": len(rows), "source_groups": len({row["source_group"] for row in rows}), "frozen_alpha": 0.3, "input_metrics": input_metrics, "base_metrics": base, "candidate_metrics": candidate, "descriptive_tail": tail, "visual_records": list(config["visual_record_ids"]), "automatic_follow_on": False, "qwen_degradation_conditioning": False, "mixed_degradations": False, "model_weight_updates": False, "development_gt_optimization": False, "sealed_internal_accessed": False, "official_foundir_test_accessed": False, "hardware_profile": profile, "gpu_name": props.name, "world_size": world, "wall_seconds": time.monotonic() - started}
            write_once(output / "result.json", result)
            print("NOISE_SCALAR_FULL_DEVELOPMENT_RESULT " + json.dumps({"pass": passed, "input": input_metrics, "base": base, "candidate": candidate, "formal_checks": formal_checks, "descriptive_tail": tail}, sort_keys=True), flush=True)
        dist.barrier()
        result = load_json(workspace_path(config["runtime"]["output_dir"]) / "result.json")
        return 0 if result["automatic_metric_gate_pass"] is True else 2
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
