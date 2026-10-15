#!/usr/bin/env python3
"""Run frozen singleton experts sequentially on a training-only mixed manifest."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party/flux2/src"))

from lowlight_restoration.lora import inject_lora, load_lora_checkpoint  # noqa: E402
from lowlight_restoration.scalar_residual import quantized_scalar_residual  # noqa: E402
from lowlight_restoration.task_native_restormer import load_task_native_restormer  # noqa: E402
from smoke.evaluate_foundir_blur_only_flux2_full_development_ddp import restore_batch  # noqa: E402
from smoke.train_foundir_blur_only_flux2_overfit_ddp import (  # noqa: E402
    encode_prompt,
    lpips_value,
    materialize_task_config,
    psnr,
)

CATEGORIES = {
    "05Noise": ["noise"],
    "02Blur_Noise": ["noise", "blur"],
    "09Lowlight_Haze": ["haze", "low_light"],
    "13Rain_Haze": ["rain", "haze"],
    "15Lowlight_Blur": ["low_light", "blur"],
    "16Lowlight_Noise": ["noise", "low_light"],
    "18Lowlight_Blur_Noise": ["noise", "low_light", "blur"],
}
STAGES = ["rain", "noise", "haze", "low_light", "blur"]
FINAL_INTEGRATION_STAGES = ["rain", "blur_noise", "noise", "haze", "low_light", "blur"]
TASK_CATEGORY = {"noise": "05Noise", "low_light": "14Lowlight"}


def workspace(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_digest(*parts: object) -> str:
    return hashlib.sha256("\0".join(map(str, parts)).encode()).hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def write_json_once(path: Path, payload: Mapping[str, object]) -> None:
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != encoded:
            raise RuntimeError(f"existing artifact identity changed: {path}")
        return
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_bytes(encoded)
    os.replace(temporary, path)


def save_png_once(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with Image.open(path) as handle:
            current = np.asarray(handle.convert("RGB"), dtype=np.uint8)
        if not np.array_equal(current, array):
            raise RuntimeError(f"existing stage PNG identity changed: {path}")
        return
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    Image.fromarray(array, mode="RGB").save(temporary, format="PNG")
    os.replace(temporary, path)


def load_rows(config: Mapping[str, object], *, verify_images: bool) -> list[dict[str, str]]:
    data = config["data"]
    manifest = workspace(data["manifest"])
    if sha256(manifest) != data["manifest_sha256"]:
        raise RuntimeError("mixed pilot manifest identity changed")
    if "routing_decisions" in data:
        decisions = workspace(data["routing_decisions"])
        if sha256(decisions) != data["routing_decisions_sha256"]:
            raise RuntimeError("frozen routing decision identity changed")
    with manifest.open(newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    orders = config["orders"]
    rehearsal = str(config.get("status", "")).startswith(
        "user_authorized_final_eleven_all_in_one_development_rehearsal_v"
    )
    expected_records = int(data["records"])
    if "records_by_category" in data:
        expected_counts = Counter({str(key): int(value) for key, value in data["records_by_category"].items()})
    else:
        per_category = int(data["records_per_category"])
        expected_counts = Counter({key: per_category for key in orders})
    if (
        len(rows) != expected_records
        or len({row["record_id"] for row in rows}) != expected_records
        or len({row["source_group"] for row in rows}) != int(data["source_groups"])
        or Counter(row["category"] for row in rows) != expected_counts
    ):
        raise RuntimeError("mixed pilot cardinality or source isolation changed")
    for row in rows:
        row_order = json.loads(row["expert_order_json"])
        if row["split"] != config["scope"]["split"] or (not rehearsal and row_order != orders[row["category"]]):
            raise RuntimeError(f"mixed pilot row contract changed: {row['record_id']}")
        if "routing_decisions" in data and not rehearsal and (
            row.get("routing_status") != "correct"
            or row.get("router_truth") != row.get("final_route")
        ):
            raise RuntimeError(f"fail-closed routing contract changed: {row['record_id']}")
        if rehearsal:
            expected = config["route_plans"].get(row.get("final_route"), [])
            if row.get("routing_status") not in {"correct", "wrong", "abstain", "fallback"} or row_order != expected:
                raise RuntimeError(f"all-in-one routing contract changed: {row['record_id']}")
        if verify_images:
            for field, hash_field in (("lq_path", "lq_sha256"), ("gt_path", "gt_sha256")):
                path = workspace(row[field])
                if not path.is_file() or sha256(path) != row[hash_field]:
                    raise RuntimeError(f"mixed pilot image changed: {row['record_id']} {field}")
    return rows


def validate_config(path: Path, *, verify_images: bool) -> tuple[dict, list[dict[str, str]]]:
    config = load_json(path)
    scope = config.get("scope", {})
    status = config.get("status")
    training_only_statuses = {
        "user_authorized_training_only_mixed_restoration_pilot_v1",
        "user_authorized_training_only_mixed_restoration_confirmation_v1",
    }
    development_status = "user_authorized_noise_family_fail_closed_integrated_development_v1"
    final_integration_status = "user_authorized_final_eleven_route_integration_v1"
    rehearsal_statuses = {
        "user_authorized_final_eleven_all_in_one_development_rehearsal_v1",
        "user_authorized_final_eleven_all_in_one_development_rehearsal_v2",
        "user_authorized_final_eleven_all_in_one_development_rehearsal_v3",
        "user_authorized_final_eleven_all_in_one_development_rehearsal_v4",
    }
    valid_scope = (
        status in training_only_statuses
        and scope.get("split") == "train"
        and scope.get("development_pixels_accessed") is False
    ) or (
        status == development_status
        and scope.get("split") == "validation"
        and scope.get("development_pixels_accessed") is True
    ) or (
        status == final_integration_status
        and scope.get("split") == "validation"
        and scope.get("development_pixels_accessed") is True
    ) or (
        status in rehearsal_statuses
        and scope.get("split") == "validation"
        and scope.get("development_pixels_accessed") is True
    )
    if (
        config.get("schema_version") != 1
        or not valid_scope
        or scope.get("model_weight_updates") is not False
        or scope.get("official_foundir_test_accessed") is not False
        or not isinstance(config.get("orders"), dict)
        or not config["orders"]
        or (status not in ({final_integration_status} | rehearsal_statuses) and any(CATEGORIES.get(category) != order for category, order in config["orders"].items()))
        or config.get("inference", {}).get("stage_order") != (FINAL_INTEGRATION_STAGES if status in ({final_integration_status} | rehearsal_statuses) else STAGES)
        or int(config["inference"].get("num_steps", -1)) != 50
        or config["inference"].get("quantization") != "round_clip_uint8_png_between_every_expert"
    ):
        raise RuntimeError("mixed restoration pilot contract changed")
    for task, expert in config["experts"].items():
        checkpoint = workspace(expert["checkpoint"])
        if not checkpoint.is_file() or sha256(checkpoint) != expert["checkpoint_sha256"]:
            raise RuntimeError(f"frozen expert checkpoint changed: {task}")
        if expert["kind"] == "flux2":
            parent = workspace(expert["parent_config"])
            if not parent.is_file() or sha256(parent) != expert["parent_config_sha256"]:
                raise RuntimeError(f"frozen expert parent changed: {task}")
    return config, load_rows(config, verify_images=verify_images)


def crop_pair(row: Mapping[str, str], config: Mapping[str, object]) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    arrays = []
    for field in ("lq_path", "gt_path"):
        with Image.open(workspace(row[field])) as handle:
            arrays.append(np.asarray(handle.convert("RGB"), dtype=np.uint8).copy())
    if arrays[0].shape != arrays[1].shape:
        raise RuntimeError(f"unaligned mixed pair: {row['record_id']}")
    size = int(config["data"]["crop_size"])
    height, width = arrays[0].shape[:2]
    if min(height, width) < size:
        raise RuntimeError(f"undersized mixed pair: {row['record_id']}")
    digest = bytes.fromhex(stable_digest(config["data"]["crop_seed"], "mixed_restoration_crop", row["record_id"]))
    y = int.from_bytes(digest[:8], "big") % (height - size + 1)
    x = int.from_bytes(digest[8:16], "big") % (width - size + 1)

    def tensor(array: np.ndarray) -> torch.Tensor:
        value = array[y : y + size, x : x + size]
        return torch.from_numpy(value.copy()).permute(2, 0, 1).float().div_(127.5).sub_(1)

    return tensor(arrays[0]), tensor(arrays[1]), [x, y]


def stage_path(output: Path, task: str, record_id: str) -> Path:
    return output / "stages" / task / f"{record_id}.png"


def expert_order(row: Mapping[str, str], config: Mapping[str, object]) -> list[str]:
    if str(config.get("status", "")).startswith(
        "user_authorized_final_eleven_all_in_one_development_rehearsal_v"
    ):
        return list(json.loads(row["expert_order_json"]))
    return list(config["orders"][row["category"]])


def input_for_stage(row: Mapping[str, str], task: str, config: Mapping[str, object], output: Path) -> torch.Tensor:
    order = expert_order(row, config)
    position = order.index(task)
    if position == 0:
        return crop_pair(row, config)[0]
    path = stage_path(output, order[position - 1], row["record_id"])
    if not path.is_file():
        raise RuntimeError(f"missing prior expert output: {path}")
    with Image.open(path) as handle:
        array = np.asarray(handle.convert("RGB"), dtype=np.uint8).copy()
    return torch.from_numpy(array).permute(2, 0, 1).float().div_(127.5).sub_(1)


def initialize_distributed(config: Mapping[str, object]) -> tuple[int, int, torch.device, str]:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    rank, world = dist.get_rank(), dist.get_world_size()
    if world != 8 or torch.cuda.device_count() != 8:
        raise RuntimeError("mixed restoration pilot requires exactly eight visible GPUs")
    props = torch.cuda.get_device_properties(device)
    profiles = [name for name in config["distributed"]["allowed_gpu_name_fragments"] if name in props.name]
    if len(profiles) != 1:
        raise RuntimeError(f"unsupported GPU: {props.name}")
    return rank, world, device, profiles[0]


def load_flux_expert(task: str, expert: Mapping[str, object], config: Mapping[str, object], device: torch.device):
    parent = load_json(workspace(expert["parent_config"]))
    if task in TASK_CATEGORY:
        parent = materialize_task_config(parent, TASK_CATEGORY[task])
    else:
        parent = copy.deepcopy(parent)
    parent["model"]["prompt"] = expert["prompt"]
    os.environ.update(
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        TORCH_HOME=str(workspace(config["runtime"]["torch_home"])),
        KLEIN_4B_BASE_MODEL_PATH=str(workspace(parent["model"]["flow_weights"])),
        AE_MODEL_PATH=str(workspace(parent["model"]["autoencoder_weights"])),
        QWEN3_MODEL_PATH=str(workspace(parent["model"]["text_encoder"])),
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
        raise RuntimeError(f"LoRA target count changed for {task}")
    load_lora_checkpoint(model, workspace(expert["checkpoint"]))
    return model, autoencoder, cfg_context, cfg_ids, batched_prc_img


def run_stage(task: str, config: Mapping[str, object], rows: Sequence[Mapping[str, str]], config_sha: str) -> None:
    rank, world, device, profile = initialize_distributed(config)
    started = time.monotonic()
    try:
        output = workspace(config["runtime"]["output_dir"])
        selected = [row for row in rows if task in expert_order(row, config)]
        expert = config["experts"][task]
        batch_size = int(config["distributed"]["batch_size_per_rank"][profile])
        if expert["kind"] == "flux2":
            model, autoencoder, cfg_context, cfg_ids, pack_image = load_flux_expert(task, expert, config, device)
        else:
            model, _ = load_task_native_restormer(expert["base_checkpoint_name"], workspace(expert["base_checkpoint_dir"]))
            payload = torch.load(workspace(expert["checkpoint"]), map_location="cpu", weights_only=False)
            expected_step = int(expert.get("checkpoint_step", 5000 if task == "rain" else -1))
            if payload.get("task") != task or int(payload.get("step", -1)) != expected_step:
                raise RuntimeError(f"{task} expert checkpoint metadata changed")
            model.load_state_dict(payload["model"], strict=True)
            model = model.to(device).eval().requires_grad_(False)
        for start in range(rank * batch_size, len(selected), world * batch_size):
            shard = selected[start : start + batch_size]
            pending = [row for row in shard if not stage_path(output, task, row["record_id"]).exists()]
            if not pending:
                continue
            source = torch.stack([input_for_stage(row, task, config, output) for row in pending]).to(device)
            if expert["kind"] == "flux2":
                seeds = [int(stable_digest(config["inference"]["seed"], task, row["record_id"])[:16], 16) for row in pending]
                restored = restore_batch(
                    model, autoencoder, source, cfg_context, cfg_ids, pack_image,
                    seeds=seeds,
                    steps=int(config["inference"]["num_steps"]),
                    reference_time=int(config["inference"]["reference_time_coordinate"]),
                )
            else:
                with torch.inference_mode():
                    restored = model((source + 1) * 0.5).float().clamp(0, 1).mul(2).sub(1)
            arrays, _ = quantized_scalar_residual(source, restored, float(expert["alpha"]))
            for row, array in zip(pending, arrays, strict=True):
                save_png_once(
                    stage_path(output, task, row["record_id"]),
                    array.permute(1, 2, 0).contiguous().cpu().numpy(),
                )
        dist.barrier()
        if rank == 0:
            paths = [stage_path(output, task, row["record_id"]) for row in selected]
            if any(not path.is_file() for path in paths):
                raise RuntimeError(f"{task} stage artifacts incomplete")
            write_json_once(output / "stages" / f"{task}.json", {
                "schema_version": 1,
                "status": "complete",
                "config_sha256": config_sha,
                "stage": task,
                "records": len(selected),
                "checkpoint_sha256": expert["checkpoint_sha256"],
                "wall_seconds": time.monotonic() - started,
                "model_weight_updates": False,
                "development_pixels_accessed": bool(config["scope"]["development_pixels_accessed"]),
                "official_foundir_test_accessed": False,
            })
            print(f"MIXED_RESTORATION_STAGE_COMPLETE {task} records={len(selected)}", flush=True)
        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def mean_metrics(rows: Sequence[Mapping[str, object]], role: str) -> dict[str, float]:
    return {
        "psnr": float(np.mean([row[role]["psnr"] for row in rows])),
        "lpips": float(np.mean([row[role]["lpips"] for row in rows])),
    }


def paired_source_group_bootstrap(
    rows: Sequence[Mapping[str, object]], *, resamples: int, confidence: float, seed: int
) -> dict[str, object]:
    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["source_group"])].append(row)
    groups = sorted(grouped)
    if len(groups) < 2:
        raise RuntimeError("mixed restoration bootstrap needs at least two source groups")
    values = np.asarray([
        [
            np.mean([item["restored"]["psnr"] - item["input"]["psnr"] for item in grouped[group]]),
            np.mean([item["restored"]["lpips"] - item["input"]["lpips"] for item in grouped[group]]),
        ]
        for group in groups
    ], dtype=np.float64)
    generator = np.random.default_rng(seed)
    draws = np.empty((resamples, 2), dtype=np.float64)
    for index in range(resamples):
        sample = generator.integers(0, len(groups), size=len(groups))
        draws[index] = values[sample].mean(axis=0)
    tail = (1.0 - confidence) / 2.0
    lower = np.quantile(draws, tail, axis=0)
    upper = np.quantile(draws, 1.0 - tail, axis=0)
    return {
        "unit": "source_group_equal_weight",
        "source_groups": len(groups),
        "resamples": resamples,
        "confidence": confidence,
        "group_balanced_mean": {"psnr_gain_db": float(values[:, 0].mean()), "lpips_delta": float(values[:, 1].mean())},
        "lower": {"psnr_gain_db": float(lower[0]), "lpips_delta": float(lower[1])},
        "upper": {"psnr_gain_db": float(upper[0]), "lpips_delta": float(upper[1])},
    }


def run_metrics(config: Mapping[str, object], rows: Sequence[Mapping[str, str]], config_sha: str) -> None:
    rank, world, device, profile = initialize_distributed(config)
    started = time.monotonic()
    try:
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

        metric = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(device).eval()
        output = workspace(config["runtime"]["output_dir"])
        sidecars = output / "per_record"
        visuals = output / "fixed_visuals"
        visual_ids = set(config["visual_record_ids"])
        for index in range(rank, len(rows), world):
            row = rows[index]
            sidecar = sidecars / f"{row['record_id']}.json"
            if sidecar.exists():
                continue
            lq, gt, crop_xy = crop_pair(row, config)
            order = expert_order(row, config)
            if order:
                with Image.open(stage_path(output, order[-1], row["record_id"])) as handle:
                    array = np.asarray(handle.convert("RGB"), dtype=np.uint8).copy()
                restored = torch.from_numpy(array).permute(2, 0, 1).float().div_(127.5).sub_(1)
            else:
                restored = lq.clone()
            target = gt.unsqueeze(0).to(device)
            values = {}
            for role, image in (("input", lq), ("restored", restored)):
                prediction = image.unsqueeze(0).to(device)
                values[role] = {"psnr": psnr(target, prediction), "lpips": lpips_value(metric, target, prediction)}
                metric.reset()
            write_json_once(sidecar, {
                "schema_version": 1,
                "status": "complete",
                "config_sha256": config_sha,
                "record_id": row["record_id"],
                "source_group": row["source_group"],
                "category": row["category"],
                "expert_order": order,
                "router_truth": row.get("router_truth"),
                "final_route": row.get("final_route") or None,
                "routing_status": row.get("routing_status"),
                "crop_xy": crop_xy,
                **values,
                "development_pixels_accessed": bool(config["scope"]["development_pixels_accessed"]),
                "official_foundir_test_accessed": False,
            })
            if row["record_id"] in visual_ids:
                triptych = np.concatenate([
                    value.add(1).mul(127.5).round().clamp(0, 255).byte().permute(1, 2, 0).numpy()
                    for value in (lq, restored, gt)
                ], axis=1)
                save_png_once(visuals / f"{row['record_id']}.png", triptych)
        dist.barrier()
        if rank == 0:
            paths = [sidecars / f"{row['record_id']}.json" for row in rows]
            if any(not path.is_file() for path in paths):
                raise RuntimeError("mixed restoration metric artifacts incomplete")
            payloads = [load_json(path) for path in paths]
            grouped: dict[str, list[dict]] = defaultdict(list)
            for payload in payloads:
                grouped[payload["category"]].append(payload)
            category_metrics = {}
            checks = {}
            bootstrap_required = bool(config["gate"].get("source_group_bootstrap_required", False))
            for category in config["orders"]:
                values = grouped[category]
                input_metrics, restored_metrics = mean_metrics(values, "input"), mean_metrics(values, "restored")
                bootstrap = paired_source_group_bootstrap(
                    values,
                    resamples=int(config["gate"].get("bootstrap_resamples", 5000)),
                    confidence=float(config["gate"].get("bootstrap_confidence", 0.95)),
                    seed=int(stable_digest(config["gate"].get("bootstrap_seed", 0), category)[:16], 16),
                )
                category_metrics[category] = {
                    "records": len(values),
                    "input": input_metrics,
                    "restored": restored_metrics,
                    "delta": {
                        "psnr_db": restored_metrics["psnr"] - input_metrics["psnr"],
                        "lpips": restored_metrics["lpips"] - input_metrics["lpips"],
                    },
                    "psnr_win_fraction": float(np.mean([row["restored"]["psnr"] > row["input"]["psnr"] for row in values])),
                    "lpips_win_fraction": float(np.mean([row["restored"]["lpips"] < row["input"]["lpips"] for row in values])),
                    "paired_source_group_bootstrap": bootstrap,
                }
                checks[category] = {
                    "mean_psnr_improves": restored_metrics["psnr"] > input_metrics["psnr"],
                    "mean_lpips_improves": restored_metrics["lpips"] < input_metrics["lpips"],
                }
                if bootstrap_required:
                    checks[category].update({
                        "bootstrap_psnr_gain_lower_positive": bootstrap["lower"]["psnr_gain_db"] > 0,
                        "bootstrap_lpips_delta_upper_negative": bootstrap["upper"]["lpips_delta"] < 0,
                    })
                thresholds = config["gate"].get("absolute_thresholds", {}).get(category)
                if thresholds:
                    checks[category].update({
                        "absolute_psnr": restored_metrics["psnr"] >= float(thresholds["psnr_minimum"]),
                        "absolute_lpips": restored_metrics["lpips"] <= float(thresholds["lpips_maximum"]),
                    })
            automatic_pass = all(all(item.values()) for item in checks.values())
            result = {
                "schema_version": 1,
                "status": "awaiting_fixed_visual_review" if automatic_pass else "failed",
                "automatic_metric_gate_pass": automatic_pass,
                "fixed_visual_review_required": True,
                "fixed_visual_review_complete": False,
                "config_sha256": config_sha,
                "split": config["scope"]["split"],
                "records": len(payloads),
                "source_groups": len({row["source_group"] for row in payloads}),
                "category_metrics": category_metrics,
                "checks": checks,
                "visual_record_ids": list(config["visual_record_ids"]),
                "hardware_profile": profile,
                "world_size": world,
                "wall_seconds": time.monotonic() - started,
                "model_weight_updates": False,
                "singleton_training_source_disjoint": bool(config["scope"].get("singleton_training_source_disjoint", False)),
                "pilot_source_disjoint": bool(config["scope"].get("pilot_source_disjoint", False)),
                "development_pixels_accessed": bool(config["scope"]["development_pixels_accessed"]),
                "official_foundir_test_accessed": False,
            }
            if "routing_decisions" in config["data"]:
                status_counts = Counter(row.get("routing_status", "") for row in rows)
                result["routing_integration"] = {
                    "accepted_records_executed": len(rows) - status_counts.get("abstain", 0),
                    "abstained_records_executed": status_counts.get("abstain", 0),
                    "correct_records": status_counts.get("correct", 0),
                    "wrong_records": status_counts.get("wrong", 0),
                    "abstention_output": "identity_input" if status_counts.get("abstain", 0) else None,
                    "expert_plan_counts": dict(sorted(Counter(
                        "+".join(json.loads(row["expert_order_json"])) or "abstain_identity" for row in rows
                    ).items())),
                    "routing_decisions_sha256": config["data"]["routing_decisions_sha256"],
                }
            write_json_once(output / "result.json", result)
            label = (
                "INTEGRATED_DEVELOPMENT"
                if "integrated_development" in str(config["status"])
                else "CONFIRMATION" if "confirmation" in str(config["status"])
                else "PILOT"
            )
            print(f"MIXED_RESTORATION_{label}_RESULT " + json.dumps({"pass": automatic_pass, "category_metrics": category_metrics}, sort_keys=True), flush=True)
        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=[*FINAL_INTEGRATION_STAGES, "metrics"])
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    config_path = workspace(args.config)
    config, rows = validate_config(config_path, verify_images=args.preflight)
    if args.preflight:
        label = (
            "INTEGRATED_DEVELOPMENT"
            if "integrated_development" in str(config["status"])
            else "CONFIRMATION" if "confirmation" in str(config["status"])
            else "PILOT"
        )
        print(json.dumps({
            "status": f"MIXED_RESTORATION_{label}_PREFLIGHT_PASS",
            "records": len(rows),
            "source_groups": len({row["source_group"] for row in rows}),
            "categories": dict(sorted(Counter(row["category"] for row in rows).items())),
            "development_pixels_accessed": bool(config["scope"]["development_pixels_accessed"]),
            "official_foundir_test_accessed": False,
        }, sort_keys=True))
        return 0
    if args.stage is None:
        raise RuntimeError("--stage is required outside preflight")
    config_sha = sha256(config_path)
    if args.stage == "metrics":
        run_metrics(config, rows, config_sha)
    else:
        run_stage(args.stage, config, rows, config_sha)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
