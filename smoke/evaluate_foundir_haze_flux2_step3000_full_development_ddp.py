#!/usr/bin/env python3
"""Evaluate immutable isolated-Haze FLUX step 3000 on all development records."""

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
from smoke.evaluate_foundir_noise_rain_full_development_ddp import (  # noqa: E402
    mean_metrics,
    source_group_bootstrap,
)
from smoke.train_foundir_blur_only_flux2_overfit_ddp import (  # noqa: E402
    encode_prompt,
    lpips_value,
    psnr,
)


EXPECTED_CONFIG_SHA = "a285c102fb63e1716aeb5ae7131b66037e188903c42aaa54caecc3bbae442b71"


def load_haze_rows(config: dict[str, object]) -> list[dict[str, str]]:
    data = config["data"]
    manifest = workspace_path(data["source_manifest"])
    if sha256(manifest) != data["source_manifest_sha256"]:
        raise RuntimeError("FoundIR pair manifest changed")
    with manifest.open(newline="") as handle:
        all_rows = list(csv.DictReader(handle))
    isolated = [
        row
        for row in all_rows
        if row["category"] == "08Haze" and row["composition_size"] == "1"
    ]
    rows = sorted(
        (dict(row) for row in isolated if row["split"] == "validation"),
        key=lambda row: row["record_id"],
    )
    train_rows = [row for row in isolated if row["split"] == "train"]
    groups = {row["source_group"] for row in rows}
    train_groups = {row["source_group"] for row in train_rows}
    if (
        len(rows) != int(data["expected_records"])
        or len(groups) != int(data["expected_source_groups"])
        or len(train_rows) != int(data["training_records"])
        or len(train_groups) != int(data["training_source_groups"])
        or len({row["record_id"] for row in rows}) != len(rows)
        or groups & train_groups
    ):
        raise RuntimeError("isolated Haze development cardinality or source isolation changed")
    for row in rows:
        if min(int(row["width"]), int(row["height"])) < int(data["crop_size"]):
            raise RuntimeError(f"undersized Haze development record: {row['record_id']}")
        if "/" in row["record_id"] or ".." in row["record_id"]:
            raise RuntimeError(f"unsafe record ID: {row['record_id']}")
    return rows


def validate_config(
    path: Path, *, verify_images: bool
) -> tuple[dict[str, object], dict[str, object], list[dict[str, str]]]:
    config = load_json(path)
    if (
        sha256(path) != EXPECTED_CONFIG_SHA
        or config.get("status") != "user_authorized_haze_step3000_full_development"
    ):
        raise RuntimeError("frozen Haze full-development config changed")
    scope = config["scope"]
    forbidden = (
        "qwen_degradation_conditioning",
        "mixed_degradations",
        "model_weight_updates",
        "pixel_refiner",
        "decoder_tuning",
        "distillation",
        "development_gt_optimization",
        "sealed_internal_accessed",
        "official_foundir_test_accessed",
    )
    if (
        scope.get("category") != "08Haze"
        or scope.get("split") != "validation"
        or any(scope.get(key) is not False for key in forbidden)
    ):
        raise RuntimeError("protected Haze scope changed")
    model = config["model"]
    parent_path = workspace_path(model["parent_config"])
    checkpoint = workspace_path(model["checkpoint"])
    evidence = config["evidence"]
    for frozen, expected in (
        (parent_path, model["parent_config_sha256"]),
        (checkpoint, model["checkpoint_sha256"]),
        (workspace_path(evidence["pilot_result"]), evidence["pilot_result_sha256"]),
    ):
        if not frozen.is_file() or sha256(frozen) != expected:
            raise RuntimeError(f"frozen dependency changed: {frozen}")
    parent = load_json(parent_path)
    inference, metrics, gate, data = (
        config["inference"], config["metrics"], config["gate"], config["data"]
    )
    if (
        parent.get("status") != "user_authorized_haze_source_diverse_continuation"
        or parent["scope"].get("category") != "08Haze"
        or parent["model"].get("name") != "flux.2-klein-base-4b"
        or inference.get("schedule") != "official_flux2_shifted_schedule"
        or int(inference.get("num_steps", -1)) != 50
        or float(inference.get("guidance", -1)) != 1.0
        or data.get("crop_namespace") != "full_development_crop"
        or gate.get("formal_checks") != ["sow_psnr", "sow_lpips"]
        or gate.get("tail_statistics_descriptive_only") is not True
        or gate.get("automatic_follow_on") is not False
        or metrics.get("bootstrap_unit") != "source_group_equal_weight"
        or int(metrics.get("bootstrap_resamples", -1)) != 5000
    ):
        raise RuntimeError("Haze evaluation contract changed")
    if verify_images:
        for key, hash_key in (
            ("flow_weights", "flow_weights_sha256"),
            ("autoencoder_weights", "autoencoder_weights_sha256"),
        ):
            frozen = workspace_path(parent["model"][key])
            if not frozen.is_file() or sha256(frozen) != parent["model"][hash_key]:
                raise RuntimeError(f"base model dependency changed: {frozen}")
    rows = load_haze_rows(config)
    visual_ids = list(config["visual_record_ids"])
    visual_rows = [row for row in rows if row["record_id"] in set(visual_ids)]
    if (
        len(visual_ids) != 16
        or len(set(visual_ids)) != 16
        or len(visual_rows) != 16
        or len({row["source_group"] for row in visual_rows}) != 16
    ):
        raise RuntimeError("fixed Haze visual IDs changed or lost source diversity")
    if verify_images:
        for row in rows:
            for field, hash_field in (("lq_path", "lq_sha256"), ("gt_path", "gt_sha256")):
                image = workspace_path(row[field])
                if not image.is_file() or sha256(image) != row[hash_field]:
                    raise RuntimeError(f"Haze image changed: {row['record_id']} {field}")
    return config, parent, rows


def save_triptych(
    path: Path, lq: torch.Tensor, restored: torch.Tensor, gt: torch.Tensor
) -> None:
    arrays = [
        value.detach().cpu().add(1).mul(127.5).round().clamp(0, 255).byte()
        .permute(1, 2, 0).numpy()
        for value in (lq, restored, gt)
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.concatenate(arrays, axis=1), mode="RGB").save(path)


def build_result(
    payloads: list[dict[str, object]],
    config: dict[str, object],
    *,
    config_path: Path,
    config_sha: str,
    profile: str,
    gpu_name: str,
    world: int,
    wall_seconds: float,
) -> dict[str, object]:
    aggregate = mean_metrics(payloads)
    metric_config = config["metrics"]
    bootstrap = source_group_bootstrap(
        payloads,
        resamples=int(metric_config["bootstrap_resamples"]),
        confidence=float(metric_config["bootstrap_confidence"]),
        seed=int(metric_config["bootstrap_seed"]),
    )
    psnr_deltas = np.asarray(
        [row["restored"]["psnr"] - row["input"]["psnr"] for row in payloads]
    )
    lpips_deltas = np.asarray(
        [row["restored"]["lpips"] - row["input"]["lpips"] for row in payloads]
    )
    gate = config["gate"]
    checks = {
        "sow_psnr": aggregate["psnr"] >= float(gate["sow_psnr_minimum"]),
        "sow_lpips": aggregate["lpips"] <= float(gate["sow_lpips_maximum"]),
    }
    sow_margins = {
        "psnr_absolute_db": aggregate["psnr"] - float(gate["sow_psnr_minimum"]),
        "psnr_relative_percent": 100.0
        * (aggregate["psnr"] - float(gate["sow_psnr_minimum"]))
        / float(gate["sow_psnr_minimum"]),
        "lpips_absolute_safety": float(gate["sow_lpips_maximum"]) - aggregate["lpips"],
        "lpips_relative_safety_percent": 100.0
        * (float(gate["sow_lpips_maximum"]) - aggregate["lpips"])
        / float(gate["sow_lpips_maximum"]),
    }
    passed = all(checks.values())
    return {
        "schema_version": 1,
        "status": "awaiting_human_visual_review" if passed else "failed",
        "automatic_metric_gate_pass": passed,
        "formal_acceptance_checks": checks,
        "sow_margins": sow_margins,
        "tail_statistics_descriptive_only": True,
        "descriptive_tail": {
            "psnr_win_fraction_vs_input": float((psnr_deltas > 0).mean()),
            "lpips_win_fraction_vs_input": float((lpips_deltas < 0).mean()),
            "records_regressing_more_than_1db_vs_input": int((psnr_deltas < -1).sum()),
            "records_regressing_more_than_2db_vs_input": int((psnr_deltas < -2).sum()),
            "worst_psnr_regression_vs_input_db": float((-psnr_deltas).max()),
        },
        "human_visual_review_required": True,
        "human_visual_review_complete": False,
        "config": config_path.relative_to(ROOT).as_posix(),
        "config_sha256": config_sha,
        "checkpoint": config["model"]["checkpoint"],
        "checkpoint_sha256": config["model"]["checkpoint_sha256"],
        "model": "FLUX.2 Klein Base 4B isolated-Haze LoRA step-3000",
        "split": "all 1580 source-group-disjoint isolated-Haze development records",
        "records": len(payloads),
        "source_groups": len({row["source_group"] for row in payloads}),
        "input_metrics": {"psnr": aggregate["input_psnr"], "lpips": aggregate["input_lpips"]},
        "output_metrics": {"psnr": aggregate["psnr"], "lpips": aggregate["lpips"]},
        "source_group_bootstrap": bootstrap,
        "pilot_metrics": {"records": 128, "psnr": 22.707913959058885, "lpips": 0.10345751804925385},
        "visual_records": list(config["visual_record_ids"]),
        "automatic_follow_on": False,
        "qwen_degradation_conditioning": False,
        "mixed_degradations": False,
        "model_weight_updates": False,
        "development_gt_optimization": False,
        "sealed_internal_accessed": False,
        "official_foundir_test_accessed": False,
        "hardware_profile": profile,
        "gpu_name": gpu_name,
        "world_size": world,
        "wall_seconds": wall_seconds,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    config_path = workspace_path(args.config)
    config, parent, rows = validate_config(config_path, verify_images=args.preflight)
    if args.preflight:
        print(json.dumps({
            "status": "HAZE_STEP3000_FULL_DEVELOPMENT_PREFLIGHT_PASS",
            "records": len(rows),
            "source_groups": len({row["source_group"] for row in rows}),
            "checkpoint_sha256": config["model"]["checkpoint_sha256"],
            "sealed_internal_accessed": False,
            "official_foundir_test_accessed": False,
        }, sort_keys=True))
        return 0

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend=config["distributed"]["backend"], device_id=device)
    started = time.monotonic()
    try:
        rank, world = dist.get_rank(), dist.get_world_size()
        if world != 8 or torch.cuda.device_count() != 8:
            raise RuntimeError("Haze full development requires exactly eight visible GPUs")
        props = torch.cuda.get_device_properties(device)
        profiles = [name for name in ("H100", "H200") if name in props.name]
        if len(profiles) != 1 or props.total_memory < 75 * 1024**3:
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
            raise RuntimeError("LoRA target count changed")
        load_lora_checkpoint(model, workspace_path(config["model"]["checkpoint"]))
        metric = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(device).eval()
        dataset = BlurDevelopmentDataset(
            rows,
            crop_size=int(config["data"]["crop_size"]),
            seed=int(config["data"]["crop_seed"]),
        )
        loader = DataLoader(
            dataset,
            batch_size=int(config["distributed"]["evaluation_batch_size_per_rank"][profile]),
            sampler=RankSampler(len(rows), rank, world),
            num_workers=int(config["distributed"]["data_loader_workers_per_rank"]),
            prefetch_factor=2,
            persistent_workers=True,
            pin_memory=True,
            collate_fn=collate,
        )
        visual_ids = set(config["visual_record_ids"])
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
                int(stable_digest(
                    config["inference"]["seed"],
                    config["inference"]["seed_namespace"],
                    record_id,
                )[:16], 16)
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
                input_values = {
                    "psnr": psnr(gt[local:local+1], lq[local:local+1]),
                    "lpips": lpips_value(metric, gt[local:local+1], lq[local:local+1]),
                }
                metric.reset()
                restored_values = {
                    "psnr": psnr(gt[local:local+1], restored_q[local:local+1]),
                    "lpips": lpips_value(metric, gt[local:local+1], restored_q[local:local+1]),
                }
                metric.reset()
                payload = {
                    "schema_version": 1,
                    "status": "complete",
                    "config_sha256": config_sha,
                    "checkpoint_sha256": config["model"]["checkpoint_sha256"],
                    "record_id": record_ids[local],
                    "source_group": batch["source_group"][position],
                    "category": "08Haze",
                    "split": "validation",
                    "crop_xy": batch["crop_xy"][position],
                    "inference_seed": seeds[local],
                    "input": input_values,
                    "restored": restored_values,
                    "restored_uint8_sha256": hashlib.sha256(
                        restored_u8[local].permute(1, 2, 0).contiguous().cpu().numpy().tobytes()
                    ).hexdigest(),
                    "development_gt_optimization": False,
                    "sealed_internal_accessed": False,
                    "official_foundir_test_accessed": False,
                }
                write_once(sidecar_dir / f"{record_ids[local]}.json", payload)
                if record_ids[local] in visual_ids:
                    save_triptych(visual_dir / f"{record_ids[local]}.png", lq[local], restored_q[local], gt[local])
            if rank == 0 and (batch_index == 0 or (batch_index + 1) % 10 == 0):
                print(f"HAZE_STEP3000_FULL_DEVELOPMENT rank0_batch={batch_index+1}/{len(loader)}", flush=True)
        dist.barrier()
        if rank == 0:
            paths = [sidecar_dir / f"{row['record_id']}.json" for row in rows]
            if (
                any(not path.is_file() for path in paths)
                or set(sidecar_dir.glob("*.json")) != set(paths)
                or len(list(visual_dir.glob("*.png"))) != 16
            ):
                raise RuntimeError("Haze full-development artifacts incomplete or contaminated")
            payloads = [load_json(path) for path in paths]
            result = build_result(
                payloads, config, config_path=config_path, config_sha=config_sha,
                profile=profile, gpu_name=props.name, world=world,
                wall_seconds=time.monotonic() - started,
            )
            write_once(output / "result.json", result)
            print("HAZE_STEP3000_FULL_DEVELOPMENT_RESULT " + json.dumps({
                "pass": result["automatic_metric_gate_pass"],
                "input": result["input_metrics"],
                "output": result["output_metrics"],
                "sow_margins": result["sow_margins"],
                "bootstrap": result["source_group_bootstrap"],
                "tail": result["descriptive_tail"],
            }, sort_keys=True), flush=True)
        dist.barrier()
        result = load_json(output / "result.json")
        return 0 if result["automatic_metric_gate_pass"] is True else 2
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
