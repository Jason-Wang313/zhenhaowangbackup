#!/usr/bin/env python3
"""Replay immutable Blur step-10k outputs for the missing 384 calibration PNGs."""

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
    RankSampler,
    restore_batch,
    sha256,
    stable_digest,
    workspace_path,
    write_once,
)
from smoke.evaluate_foundir_blur_only_flux2_scalar_residual_ddp import (  # noqa: E402
    FixedCropDataset,
    collate,
    validate_config,
)
from smoke.train_foundir_blur_only_flux2_overfit_ddp import encode_prompt, psnr  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate-config", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    gate_path = workspace_path(args.gate_config)
    gate = json.loads(gate_path.read_text())
    if gate.get("status") != "user_authorized_one_shot_cached_low_frequency_anchor_gate":
        raise RuntimeError("calibration-cache authorization changed")
    source = gate["source"]
    cached_config_path = workspace_path(source["cached_config"])
    config, parent, all_rows = validate_config(cached_config_path, verify_images=args.preflight)
    rows = [row for row in all_rows if int(row["selection_rank"]) <= 383]
    if len(rows) != 384 or len({row["source_group"] for row in rows}) != 384:
        raise RuntimeError("calibration partition changed")
    if args.preflight:
        print(json.dumps({
            "status": "BLUR_CALIBRATION_CACHE_PREFLIGHT_PASS",
            "records": 384,
            "model_inference": True,
            "model_weight_updates": False,
            "development_accessed": False,
            "sealed_internal_accessed": False,
            "official_foundir_test_accessed": False,
        }, sort_keys=True))
        return 0

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend="nccl", device_id=device)
    started = time.monotonic()
    try:
        rank, world = dist.get_rank(), dist.get_world_size()
        if world != 8 or torch.cuda.device_count() != 8:
            raise RuntimeError("calibration replay requires exactly eight visible GPUs")
        props = torch.cuda.get_device_properties(device)
        if "H100" not in props.name and "H200" not in props.name:
            raise RuntimeError(f"unsupported GPU: {props.name}")
        output = workspace_path(source["calibration_export_dir"])
        png_dir, sidecar_dir = output / "base_flux_png", output / "per_record"
        png_dir.mkdir(parents=True, exist_ok=True)
        sidecar_dir.mkdir(parents=True, exist_ok=True)
        os.environ.update(
            HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
            TORCH_HOME=str(workspace_path(config["runtime"]["torch_home"])),
            KLEIN_4B_BASE_MODEL_PATH=str(workspace_path(parent["model"]["flow_weights"])),
            AE_MODEL_PATH=str(workspace_path(parent["model"]["autoencoder_weights"])),
            QWEN3_MODEL_PATH=str(workspace_path(parent["model"]["text_encoder"])),
        )
        _, _, cfg_context, cfg_ids = encode_prompt(parent, device)
        from flux2.sampling import batched_prc_img
        from flux2.util import load_ae, load_flow_model

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
            raise RuntimeError("LoRA target count changed")
        load_lora_checkpoint(model, workspace_path(config["model"]["checkpoint"]))
        dataset = FixedCropDataset(rows, int(config["data"]["crop_size"]))
        batch_size = 2 if "H100" in props.name else 4
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=RankSampler(len(rows), rank, world),
            num_workers=2,
            prefetch_factor=2,
            persistent_workers=True,
            pin_memory=True,
            collate_fn=collate,
        )
        old_sidecars = workspace_path(source["cached_output_dir"]) / "per_record"
        for batch_index, batch in enumerate(loader):
            missing = [
                position for position, record_id in enumerate(batch["record_id"])
                if not (sidecar_dir / f"{record_id}.json").exists()
            ]
            if not missing:
                continue
            selection = torch.tensor(missing, dtype=torch.long)
            lq = batch["lq"].index_select(0, selection).to(device, non_blocking=True)
            gt = batch["gt"].index_select(0, selection).to(device, non_blocking=True)
            record_ids = [batch["record_id"][position] for position in missing]
            seeds = [
                int(stable_digest(config["inference"]["seed"], "scalar_residual", record_id)[:16], 16)
                for record_id in record_ids
            ]
            restored = restore_batch(
                model, autoencoder, lq, cfg_context, cfg_ids, batched_prc_img,
                seeds=seeds,
                steps=int(config["inference"]["num_steps"]),
                reference_time=int(config["inference"]["reference_time_coordinate"]),
            )
            restored_u8, restored_q = quantized_scalar_residual(lq, restored, 1.0)
            for local, position in enumerate(missing):
                record_id = record_ids[local]
                old_sidecar_path = old_sidecars / f"{record_id}.json"
                old_sidecar = json.loads(old_sidecar_path.read_text())
                replay_psnr = psnr(gt[local:local + 1], restored_q[local:local + 1])
                expected_psnr = float(old_sidecar["alpha_metrics"]["1.000000"]["psnr"])
                if abs(replay_psnr - expected_psnr) > 1e-5:
                    raise RuntimeError(f"deterministic replay mismatch: {record_id}")
                array = restored_u8[local].permute(1, 2, 0).contiguous().cpu().numpy()
                content_hash = hashlib.sha256(array.tobytes()).hexdigest()
                png_path = png_dir / f"{record_id}.png"
                if png_path.exists():
                    with Image.open(png_path) as handle:
                        existing = np.asarray(handle.convert("RGB"), dtype=np.uint8)
                    if hashlib.sha256(existing.tobytes()).hexdigest() != content_hash:
                        raise RuntimeError(f"differing replay PNG exists: {record_id}")
                else:
                    Image.fromarray(array, mode="RGB").save(png_path)
                payload = {
                    "schema_version": 1,
                    "status": "complete",
                    "record_id": record_id,
                    "source_group": batch["source_group"][position],
                    "selection_rank": int(batch["selection_rank"][position]),
                    "inference_seed": seeds[local],
                    "checkpoint_sha256": source["checkpoint_sha256"],
                    "source_sidecar": old_sidecar_path.relative_to(ROOT).as_posix(),
                    "source_sidecar_sha256": sha256(old_sidecar_path),
                    "base_flux_png_uint8_sha256": content_hash,
                    "base_psnr_replay": replay_psnr,
                    "model_weight_updates": False,
                    "development_accessed": False,
                    "sealed_internal_accessed": False,
                    "official_foundir_test_accessed": False,
                }
                write_once(sidecar_dir / f"{record_id}.json", payload)
            if rank == 0 and (batch_index == 0 or (batch_index + 1) % 10 == 0):
                print(f"BLUR_CALIBRATION_CACHE rank0_batch={batch_index + 1}/{len(loader)}", flush=True)
        dist.barrier()
        if rank == 0:
            expected = {row["record_id"] for row in rows}
            completed = {path.stem for path in sidecar_dir.glob("*.json")}
            pngs = {path.stem for path in png_dir.glob("*.png")}
            if completed != expected or pngs != expected:
                raise RuntimeError("calibration cache incomplete or contaminated")
            result = {
                "schema_version": 1,
                "status": "complete",
                "records": 384,
                "source_groups": 384,
                "checkpoint": config["model"]["checkpoint"],
                "checkpoint_sha256": source["checkpoint_sha256"],
                "model_inference": True,
                "model_weight_updates": False,
                "development_accessed": False,
                "sealed_internal_accessed": False,
                "official_foundir_test_accessed": False,
                "world_size": 8,
                "gpu_name": props.name,
                "wall_seconds": time.monotonic() - started,
            }
            write_once(output / "result.json", result)
            print("BLUR_CALIBRATION_CACHE_COMPLETE " + json.dumps(result, sort_keys=True), flush=True)
        dist.barrier()
        return 0
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
