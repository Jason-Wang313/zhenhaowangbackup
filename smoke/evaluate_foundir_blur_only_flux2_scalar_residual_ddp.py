#!/usr/bin/env python3
"""Evaluate a frozen scalar LQ/FLUX residual blend without development access."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party/flux2/src"))

from lowlight_restoration.lora import inject_lora, load_lora_checkpoint  # noqa: E402
from lowlight_restoration.scalar_residual import (  # noqa: E402
    aggregate_alpha_metrics,
    alpha_key,
    compare_to_base,
    gate_comparison,
    quantized_scalar_residual,
    select_alpha,
)
from smoke.evaluate_foundir_blur_only_flux2_full_development_ddp import (  # noqa: E402
    RankSampler,
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
    psnr,
)


EXPECTED_CONFIG_SHA = "a2e49e2094a1f8f8911377729cd48297891dd3392435638d12faddb838844bac"


def validate_config(config_path: Path, *, verify_images: bool) -> tuple[dict[str, object], dict[str, object], list[dict[str, str]]]:
    config = load_json(config_path)
    if sha256(config_path) != EXPECTED_CONFIG_SHA:
        raise RuntimeError("scalar residual config identity changed")
    if config.get("status") != "user_authorized_scalar_residual_feasibility_design":
        raise RuntimeError("scalar residual authorization status changed")
    forbidden = (
        "qwen_degradation_conditioning", "mixed_degradations", "model_weight_updates",
        "decoder_tuning", "distillation", "development_gt_optimization",
        "development_accessed", "sealed_internal_accessed", "official_foundir_test_accessed",
    )
    if any(config["scope"].get(key) is not False for key in forbidden):
        raise RuntimeError("protected scalar residual scope changed")
    model, data, correction, gate = (
        config["model"], config["data"], config["correction"], config["gate"]
    )
    parent_path = workspace_path(model["parent_config"])
    checkpoint = workspace_path(model["checkpoint"])
    manifest = workspace_path(data["manifest"])
    metadata = workspace_path(data["manifest_metadata"])
    identities = (
        (parent_path, model["parent_config_sha256"]),
        (checkpoint, model["checkpoint_sha256"]),
        (manifest, data["manifest_sha256"]),
        (metadata, data["manifest_metadata_sha256"]),
    )
    for path, expected in identities:
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"frozen dependency changed: {path}")
    parent = load_json(parent_path)
    if (
        parent["scope"]["category"] != "01Blur"
        or parent["scope"]["qwen_degradation_conditioning"] is not False
        or parent["scope"]["mixed_degradations"] is not False
        or int(model["checkpoint_step"]) != 10000
    ):
        raise RuntimeError("parent model contract changed")
    if (
        correction["formula"] != "lq_plus_alpha_times_flux_minus_lq"
        or correction["selection_split"] != "calibration_only"
        or int(correction["trainable_parameters"]) != 0
        or [float(v) for v in correction["alpha_grid"]] != [0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]
        or gate["full_development_automatic"] is not False
        or gate["residual_network_automatic"] is not False
    ):
        raise RuntimeError("scalar residual selection/gate contract changed")
    with manifest.open(newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle) if row["role"] == "pilot_train"]
    rows.sort(key=lambda row: int(row["selection_rank"]))
    if (
        len(rows) != 512
        or [int(row["selection_rank"]) for row in rows] != list(range(512))
        or len({row["source_group"] for row in rows}) != 512
        or any(row["split"] != "train" or row["category"] != "01Blur" or row["composition_size"] != "1" for row in rows)
    ):
        raise RuntimeError("frozen 512-record training-only split changed")
    calibration = [row for row in rows if int(row["selection_rank"]) <= 383]
    validation = [row for row in rows if int(row["selection_rank"]) >= 384]
    if len(calibration) != 384 or len(validation) != 128:
        raise RuntimeError("calibration/validation partition changed")
    if verify_images:
        for row in rows:
            for field, hash_field in (("lq_path", "lq_sha256"), ("gt_path", "gt_sha256")):
                path = workspace_path(row[field])
                if not path.is_file() or sha256(path) != row[hash_field]:
                    raise RuntimeError(f"image identity changed: {row['record_id']} {field}")
    return config, parent, rows


class FixedCropDataset(Dataset[dict[str, object]]):
    def __init__(self, rows: Sequence[Mapping[str, str]], crop_size: int) -> None:
        self.rows, self.crop_size = rows, crop_size

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.rows[index]
        x, y = int(row["crop_x"]), int(row["crop_y"])
        arrays = []
        for field in ("lq_path", "gt_path"):
            with Image.open(workspace_path(row[field])) as handle:
                array = np.asarray(handle.convert("RGB"), dtype=np.uint8)
            arrays.append(array[y:y+self.crop_size, x:x+self.crop_size].copy())
        if any(array.shape != (self.crop_size, self.crop_size, 3) for array in arrays):
            raise RuntimeError(f"invalid frozen crop: {row['record_id']}")
        tensors = [torch.from_numpy(array).permute(2, 0, 1).float().div(127.5).sub(1) for array in arrays]
        return {
            "index": index, "record_id": row["record_id"], "source_group": row["source_group"],
            "selection_rank": int(row["selection_rank"]), "crop_xy": [x, y],
            "lq": tensors[0], "gt": tensors[1],
        }


def collate(items: Sequence[Mapping[str, object]]) -> dict[str, object]:
    return {
        key: ([item[key] for item in items] if key not in {"lq", "gt"} else torch.stack([item[key] for item in items]))
        for key in items[0]
    }


def write_base_png(path: Path, value: torch.Tensor) -> str:
    array = value.permute(1, 2, 0).contiguous().cpu().numpy()
    content_hash = hashlib.sha256(array.tobytes()).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with Image.open(path) as handle:
            existing = np.asarray(handle.convert("RGB"), dtype=np.uint8)
        if hashlib.sha256(existing.tobytes()).hexdigest() != content_hash:
            raise RuntimeError(f"refusing to overwrite differing base output: {path}")
    else:
        Image.fromarray(array, mode="RGB").save(path)
    return content_hash


def build_visuals(rows: Sequence[Mapping[str, str]], sidecars: Sequence[Mapping[str, object]], output: Path, alpha: float) -> None:
    visual_dir = output / "validation_visuals"
    selected = [(row, sidecar) for row, sidecar in zip(rows, sidecars, strict=True) if 384 <= int(row["selection_rank"]) < 400]
    for row, sidecar in selected:
        x, y = map(int, sidecar["crop_xy"])
        crops = []
        for field in ("lq_path", "gt_path"):
            with Image.open(workspace_path(row[field])) as handle:
                array = np.asarray(handle.convert("RGB"), dtype=np.uint8)
            crops.append(array[y:y+512, x:x+512].copy())
        with Image.open(output / "base_flux_png" / f"{row['record_id']}.png") as handle:
            base = np.asarray(handle.convert("RGB"), dtype=np.uint8).copy()
        corrected = np.rint(crops[0].astype(np.float32) + alpha * (base.astype(np.float32) - crops[0])).clip(0, 255).astype(np.uint8)
        triptych = np.concatenate([crops[0], corrected, crops[1]], axis=1)
        visual_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(triptych, mode="RGB").save(visual_dir / f"{int(row['selection_rank']):03d}_{row['record_id']}.png")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    config_path = workspace_path(args.config)
    config, parent, rows = validate_config(config_path, verify_images=args.preflight)
    if args.preflight:
        print(json.dumps({"status": "FLUX2_SCALAR_RESIDUAL_PREFLIGHT_PASS", "records": 512, "calibration": 384, "validation": 128, "development_accessed": False}, sort_keys=True))
        return 0

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend=config["distributed"]["backend"], device_id=device)
    started = time.monotonic()
    try:
        rank, world = dist.get_rank(), dist.get_world_size()
        if world != 8 or torch.cuda.device_count() != 8:
            raise RuntimeError("scalar residual feasibility requires exactly eight visible GPUs")
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
        sidecar_dir, png_dir = output / "per_record", output / "base_flux_png"
        sidecar_dir.mkdir(parents=True, exist_ok=True)
        config_sha = sha256(config_path)
        os.environ.update(
            HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
            TORCH_HOME=str(workspace_path(config["runtime"]["torch_home"])),
            KLEIN_4B_BASE_MODEL_PATH=str(workspace_path(parent["model"]["flow_weights"])),
            AE_MODEL_PATH=str(workspace_path(parent["model"]["autoencoder_weights"])),
            QWEN3_MODEL_PATH=str(workspace_path(parent["model"]["text_encoder"])),
        )
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
        dataset = FixedCropDataset(rows, int(config["data"]["crop_size"]))
        loader = DataLoader(dataset, batch_size=int(config["distributed"]["evaluation_batch_size_per_rank"][profile]), sampler=RankSampler(len(rows), rank, world), num_workers=2, prefetch_factor=2, persistent_workers=True, pin_memory=True, collate_fn=collate)
        alphas = [float(value) for value in config["correction"]["alpha_grid"]]
        for batch_index, batch in enumerate(loader):
            missing = [position for position, record_id in enumerate(batch["record_id"]) if not (sidecar_dir / f"{record_id}.json").exists()]
            if not missing:
                continue
            selection = torch.tensor(missing, dtype=torch.long)
            lq = batch["lq"].index_select(0, selection).to(device, non_blocking=True)
            gt = batch["gt"].index_select(0, selection).to(device, non_blocking=True)
            record_ids = [batch["record_id"][position] for position in missing]
            seeds = [int(stable_digest(config["inference"]["seed"], "scalar_residual", record_id)[:16], 16) for record_id in record_ids]
            restored = restore_batch(model, autoencoder, lq, cfg_context, cfg_ids, batched_prc_img, seeds=seeds, steps=int(config["inference"]["num_steps"]), reference_time=int(config["inference"]["reference_time_coordinate"]))
            restored_u8, restored_q = quantized_scalar_residual(lq, restored, 1.0)
            for local, position in enumerate(missing):
                alpha_metrics = {}
                for alpha in alphas:
                    _, corrected = quantized_scalar_residual(lq[local:local+1], restored_q[local:local+1], alpha)
                    alpha_metrics[alpha_key(alpha)] = {"psnr": psnr(gt[local:local+1], corrected), "lpips": lpips_value(metric, gt[local:local+1], corrected)}
                    metric.reset()
                selection_rank = int(batch["selection_rank"][position])
                png_hash = None
                if selection_rank >= 384:
                    png_hash = write_base_png(png_dir / f"{record_ids[local]}.png", restored_u8[local])
                payload = {
                    "schema_version": 1, "status": "complete", "config_sha256": config_sha,
                    "checkpoint_sha256": config["model"]["checkpoint_sha256"],
                    "record_id": record_ids[local], "source_group": batch["source_group"][position],
                    "selection_rank": selection_rank, "role": "calibration" if selection_rank <= 383 else "validation",
                    "crop_xy": batch["crop_xy"][position], "inference_seed": seeds[local],
                    "alpha_metrics": alpha_metrics, "base_flux_png_uint8_sha256": png_hash,
                    "development_accessed": False, "sealed_internal_accessed": False, "official_foundir_test_accessed": False,
                }
                write_once(sidecar_dir / f"{record_ids[local]}.json", payload)
            if rank == 0 and (batch_index == 0 or (batch_index + 1) % 10 == 0):
                print(f"FLUX2_SCALAR_RESIDUAL rank0_batch={batch_index+1}/{len(loader)}", flush=True)
        dist.barrier()
        if rank == 0:
            paths = [sidecar_dir / f"{row['record_id']}.json" for row in rows]
            if any(not path.is_file() for path in paths) or set(sidecar_dir.glob("*.json")) != set(paths):
                raise RuntimeError("scalar residual sidecars incomplete or contaminated")
            payloads = [load_json(path) for path in paths]
            if any(item.get("config_sha256") != config_sha or item.get("development_accessed") is not False for item in payloads):
                raise RuntimeError("scalar residual sidecar identity/protection changed")
            calibration = payloads[:384]
            validation = payloads[384:]
            calibration_metrics = aggregate_alpha_metrics(calibration, alphas)
            try:
                selected_alpha = select_alpha(calibration_metrics, maximum_lpips=0.17)
                selection_failure = None
                comparison = compare_to_base(validation, selected_alpha=selected_alpha)
                checks = gate_comparison(comparison, config["gate"])
                passed = all(checks.values())
            except RuntimeError as error:
                selected_alpha, comparison, checks, passed = None, None, {"calibration_lpips_eligible_alpha": False}, False
                selection_failure = str(error)
            if selected_alpha is not None:
                build_visuals(rows, payloads, output, selected_alpha)
            result = {
                "schema_version": 1, "status": "feasibility_pass_stop" if passed else "failed",
                "automatic_gate_pass": passed, "config": config_path.relative_to(ROOT).as_posix(), "config_sha256": config_sha,
                "checkpoint": config["model"]["checkpoint"], "checkpoint_sha256": config["model"]["checkpoint_sha256"],
                "model": "FLUX.2 Klein Base 4B isolated-Blur step-10000 plus non-trainable scalar residual correction",
                "split": "training-only 384 calibration plus frozen 128 validation",
                "records": 512, "source_groups": 512, "selected_alpha": selected_alpha,
                "selection_failure": selection_failure, "calibration_metrics": calibration_metrics,
                "validation_comparison": comparison, "checks": checks,
                "full_development_automatic": False, "residual_network_automatic": False,
                "qwen_degradation_conditioning": False, "mixed_degradations": False,
                "model_weight_updates": False, "decoder_tuning": False, "distillation": False,
                "development_gt_optimization": False, "development_accessed": False,
                "sealed_internal_accessed": False, "official_foundir_test_accessed": False,
                "hardware_profile": profile, "gpu_name": props.name, "world_size": world,
                "wall_seconds": time.monotonic() - started,
            }
            write_once(output / "result.json", result)
            print("FLUX2_SCALAR_RESIDUAL_RESULT " + json.dumps({"pass": passed, "selected_alpha": selected_alpha, "comparison": comparison, "checks": checks}, sort_keys=True), flush=True)
        dist.barrier()
        result = load_json(workspace_path(config["runtime"]["output_dir"]) / "result.json")
        return 0 if result["automatic_gate_pass"] is True else 2
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
