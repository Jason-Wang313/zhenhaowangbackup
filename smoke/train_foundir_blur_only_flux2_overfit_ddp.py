#!/usr/bin/env python3
"""FLUX.2 DiT-LoRA single-task capacity gate (legacy Blur or four-task suite)."""

from __future__ import annotations

import argparse
import copy
import csv
import gc
import hashlib
import io
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image, ImageDraw
from torch.nn.parallel import DistributedDataParallel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party/flux2/src"))

from lowlight_restoration.flow import build_reference_flow_batch  # noqa: E402
from lowlight_restoration.lora import inject_lora, load_lora_checkpoint, save_lora_checkpoint  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_once(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise RuntimeError(f"refusing to overwrite differing artifact: {path}")
        return
    path.write_bytes(content)


def stable_int(*values: object) -> int:
    return int.from_bytes(hashlib.sha256(":".join(map(str, values)).encode()).digest()[:8], "big")


def materialize_task_config(raw: Mapping[str, object], task: str | None) -> dict[str, object]:
    """Turn the frozen four-task suite into one independent task-arm config."""
    config = copy.deepcopy(dict(raw))
    if task is None:
        return config
    if raw.get("status") != "user_authorized_four_task_fail_fast_capacity_design":
        raise RuntimeError("--task is only valid for the frozen four-task capacity suite")
    tasks = raw.get("tasks")
    if not isinstance(tasks, dict) or task not in tasks:
        raise RuntimeError(f"unsupported suite task: {task}")
    task_config = tasks[task]
    scope = raw["scope"]
    if task not in scope["categories"]:
        raise RuntimeError("task is outside the frozen suite scope")
    config["status"] = "user_authorized_single_task_capacity_sanity"
    config["scope"] = {
        "category": task,
        "qwen_degradation_conditioning": scope["qwen_degradation_conditioning"],
        "mixed_degradations": scope["mixed_degradations"],
        "pixel_refiner": scope["pixel_refiner"],
        "official_foundir_test_accessed": scope["official_foundir_test_accessed"],
        "sealed_internal_accessed": scope["sealed_internal_accessed"],
    }
    config["model"]["prompt"] = task_config["prompt"]
    data = raw["data"]
    config["data"] = {
        "source_manifest": data["source_manifest"],
        "source_manifest_sha256": data["source_manifest_sha256"],
        "frozen_manifest": data["frozen_manifest"],
        "frozen_manifest_metadata": data["frozen_manifest_metadata"],
        "category": task,
        "train_role": "tiny_overfit_train",
        "validation_role": "descriptive_validation",
        "tiny_overfit_split": data["tiny_overfit_split"],
        "tiny_overfit_records": data["tiny_overfit_records_per_task"],
        "tiny_overfit_unique_source_groups": data["tiny_overfit_records_per_task"],
        "frozen_validation_split": data["frozen_validation_split"],
        "frozen_validation_records": data["frozen_validation_records_per_task"],
        "frozen_validation_unique_source_groups": data["frozen_validation_records_per_task"],
        "crop_size": data["crop_size"],
        "selection_seed": data["selection_seed"],
        "crop_seed": data["crop_seed"],
    }
    training = config["training"]
    training["global_batch_size"] = training.pop("global_batch_size_per_task")
    task_index = list(raw["tasks"]).index(task)
    training["seed"] = int(training["seed"]) + task_index * 10_000
    inference = config["inference"]
    inference["visual_records"] = inference["train_visual_records"]
    inference["seed"] = int(inference["seed"]) + task_index * 10_000
    config["gate"] = {
        **raw["gate"]["capacity_checks"],
        "require_human_visual_review": raw["gate"]["require_human_visual_review"],
    }
    distributed = config["distributed"]
    distributed["required_world_size"] = distributed["independent_task_world_size"]
    config["runtime"] = {
        "output_dir": f"{raw['runtime']['output_root']}/{task_config['slug']}",
        "torch_home": raw["runtime"]["torch_home"],
        "offline": raw["runtime"]["offline"],
    }
    config["_suite_task"] = task
    config["_sow_stage2"] = copy.deepcopy(task_config["sow_stage2"])
    return config


def validate_config(config: Mapping[str, object]) -> None:
    allowed_status = {
        "advisor_authorized_blur_only_tiny_overfit",
        "user_authorized_single_task_capacity_sanity",
        "user_authorized_haze_source_diverse_pilot",
        "user_authorized_haze_source_diverse_continuation",
        "user_authorized_rain_source_diverse_pilot",
        "user_authorized_rain_strength_identity_directional_gate",
        "user_authorized_rain_strength_identity_ratio_gate",
    }
    if config.get("status") not in allowed_status:
        raise RuntimeError("config is not authorized for a single-task tiny-overfit gate")
    scope = config["scope"]
    required_false = (
        "qwen_degradation_conditioning", "mixed_degradations", "pixel_refiner",
        "official_foundir_test_accessed", "sealed_internal_accessed",
    )
    allowed_categories = {"01Blur", "05Noise", "08Haze", "10Rain", "14Lowlight"}
    if scope.get("category") not in allowed_categories or any(scope.get(key) is not False for key in required_false):
        raise RuntimeError("single-task scope contract changed")
    if config["model"].get("name") != "flux.2-klein-base-4b":
        raise RuntimeError("model family changed")
    if config["training"].get("objective") != "pure_noise_to_clean_rectified_flow_with_native_reference":
        raise RuntimeError("tiny-overfit flow objective changed")
    inference = config["inference"]
    if inference.get("schedule") != "official_flux2_shifted_schedule" or int(inference["num_steps"]) != 50:
        raise RuntimeError("official Base-model inference schedule is required")
    if float(inference["guidance"]) != 1.0:
        raise RuntimeError("guidance 1.0 must match the conditional training branch")
    distributed = config["distributed"]
    training = config["training"]
    if int(distributed["required_world_size"]) not in {2, 8}:
        raise RuntimeError("this gate requires exactly two or eight workers")
    if int(training["global_batch_size"]) != (
        int(distributed["required_world_size"])
        * int(training["micro_batch_size_per_rank"])
        * int(training["gradient_accumulation_steps"])
    ):
        raise RuntimeError("global batch arithmetic changed")
    checkpoints = [int(value) for value in training["checkpoint_steps"]]
    evaluations = [int(value) for value in inference["evaluate_steps"]]
    if checkpoints != sorted(set(checkpoints)) or checkpoints[-1] != int(training["max_steps"]):
        raise RuntimeError("checkpoint schedule must be sorted and finish at max_steps")
    continuation = config.get("status") == "user_authorized_haze_source_diverse_continuation"
    rain_directional = config.get("status") == "user_authorized_rain_strength_identity_directional_gate"
    rain_ratio = config.get("status") == "user_authorized_rain_strength_identity_ratio_gate"
    expected_evaluations = (
        [int(config["initialization"]["start_step"]), *checkpoints]
        if rain_directional or rain_ratio else checkpoints if continuation else [0, *checkpoints]
    )
    if evaluations != expected_evaluations:
        raise RuntimeError("checkpoint evaluation schedule changed")
    if config.get("status") == "user_authorized_haze_source_diverse_pilot":
        data = config["data"]
        gate = config["gate"]
        if (
            scope.get("category") != "08Haze"
            or int(data["tiny_overfit_records"]) != 1580
            or int(data["tiny_overfit_unique_source_groups"]) != 1580
            or int(data["frozen_validation_records"]) != 128
            or int(data["frozen_validation_unique_source_groups"]) != 128
            or config["lora"].get("initialization") != "fresh_zero_lora"
            or int(training["max_steps"]) != 2000
            or int(training["global_batch_size"]) != 32
            or gate.get("protocol") != "final_frozen_source_disjoint_validation_sow_only"
            or float(gate["validation_psnr_minimum"]) != 22.1
            or float(gate["validation_lpips_maximum"]) != 0.19
            or gate.get("automatic_full_development_launch") is not False
        ):
            raise RuntimeError("Haze source-diverse pilot contract changed")
    if continuation:
        data = config["data"]
        gate = config["gate"]
        initialization = config.get("initialization", {})
        if (
            scope.get("category") != "08Haze"
            or int(data["tiny_overfit_records"]) != 1580
            or int(data["tiny_overfit_unique_source_groups"]) != 1580
            or int(data["frozen_validation_records"]) != 128
            or int(data["frozen_validation_unique_source_groups"]) != 128
            or config["lora"].get("initialization") != "resume_exact_checkpoint_and_optimizer"
            or int(initialization.get("start_step", -1)) != 2000
            or int(training["max_steps"]) != 3000
            or checkpoints != [3000]
            or int(training["global_batch_size"]) != 32
            or gate.get("protocol") != "final_frozen_source_disjoint_validation_sow_only"
            or float(gate["validation_psnr_minimum"]) != 22.1
            or float(gate["validation_lpips_maximum"]) != 0.19
            or gate.get("automatic_full_development_launch") is not False
        ):
            raise RuntimeError("Haze source-diverse continuation contract changed")
    if config.get("status") == "user_authorized_rain_source_diverse_pilot":
        data = config["data"]
        gate = config["gate"]
        if (
            scope.get("category") != "10Rain"
            or int(data["tiny_overfit_records"]) != 1580
            or int(data["tiny_overfit_unique_source_groups"]) != 1580
            or int(data["frozen_validation_records"]) != 100
            or int(data["frozen_validation_unique_source_groups"]) != 100
            or config["lora"].get("initialization") != "fresh_zero_lora"
            or int(training["max_steps"]) != 2000
            or int(training["global_batch_size"]) != 32
            or gate.get("protocol") != "final_frozen_source_disjoint_validation_sow_only"
            or float(gate["validation_psnr_minimum"]) != 34.2
            or float(gate["validation_lpips_maximum"]) != 0.08
            or gate.get("automatic_full_development_launch") is not False
        ):
            raise RuntimeError("Rain source-diverse pilot contract changed")
    if rain_directional:
        data = config["data"]
        gate = config["gate"]
        initialization = config.get("initialization", {})
        policy = training.get("strength_identity_policy", {})
        if (
            scope.get("category") != "10Rain"
            or int(data["tiny_overfit_records"]) != 1580
            or int(data["tiny_overfit_unique_source_groups"]) != 1580
            or int(data["frozen_validation_records"]) != 48
            or int(data["frozen_validation_unique_source_groups"]) != 48
            or int(data["development_records_per_strength_bin"]) != 16
            or int(initialization.get("start_step", -1)) != 2000
            or config["lora"].get("initialization") != "resume_exact_checkpoint_and_optimizer"
            or int(training["max_steps"]) != 2250
            or checkpoints != [2250]
            or int(training["global_batch_size"]) != 32
            or float(training["learning_rate"]) != 0.000025
            or policy.get("sampling") != "equal_three_bins_absolute_position"
            or list(policy.get("input_psnr_bins", [])) != [28.0, 34.0]
            or policy.get("near_clean_target") != "lq_identity"
            or policy.get("moderate_target") != "gt"
            or policy.get("heavy_target") != "gt"
            or gate.get("protocol") != "directional_strength_identity_gate_on_new_train_split_development"
            or float(gate["near_clean_minimum_psnr_gain_db"]) != 0.5
            or float(gate["moderate_maximum_psnr_regression_db"]) != 0.1
            or float(gate["heavy_maximum_psnr_regression_db"]) != 0.1
            or float(gate["overall_minimum_psnr_gain_db"]) != 0.2
            or float(gate["overall_maximum_lpips_regression"]) != 0.005
            or gate.get("automatic_continuation") is not False
        ):
            raise RuntimeError("Rain strength-identity directional contract changed")
    if rain_ratio:
        data = config["data"]
        gate = config["gate"]
        initialization = config.get("initialization", {})
        policy = training.get("strength_identity_policy", {})
        if (
            scope.get("category") != "10Rain"
            or int(data["tiny_overfit_records"]) != 1580
            or int(data["tiny_overfit_unique_source_groups"]) != 1580
            or int(data["frozen_validation_records"]) != 48
            or int(data["frozen_validation_unique_source_groups"]) != 48
            or int(data["development_records_per_strength_bin"]) != 16
            or int(initialization.get("start_step", -1)) != 2000
            or config["lora"].get("initialization") != "resume_exact_checkpoint_and_optimizer"
            or int(training["max_steps"]) != 2250
            or checkpoints != [2250]
            or int(training["global_batch_size"]) != 32
            or float(training["learning_rate"]) != 0.000025
            or policy.get("sampling") != "weighted_cycle_3_2_1_absolute_position"
            or list(policy.get("sampling_weights", [])) != [3, 2, 1]
            or list(policy.get("input_psnr_bins", [])) != [28.0, 34.0]
            or policy.get("near_clean_target") != "lq_identity"
            or policy.get("moderate_target") != "gt"
            or policy.get("heavy_target") != "gt"
            or gate.get("protocol") != "directional_strength_identity_gate_on_new_train_split_development"
            or float(gate["near_clean_minimum_psnr_gain_db"]) != 0.5
            or float(gate["moderate_maximum_psnr_regression_db"]) != 0.1
            or float(gate["heavy_maximum_psnr_regression_db"]) != 0.1
            or float(gate["overall_minimum_psnr_gain_db"]) != 0.2
            or float(gate["overall_maximum_lpips_regression"]) != 0.005
            or gate.get("automatic_continuation") is not False
        ):
            raise RuntimeError("Rain strength-identity ratio contract changed")


def validate_external_continuation(config: Mapping[str, object]) -> None:
    status = config.get("status")
    if status not in {
        "user_authorized_haze_source_diverse_continuation",
        "user_authorized_rain_strength_identity_directional_gate",
        "user_authorized_rain_strength_identity_ratio_gate",
    }:
        return
    initialization = config["initialization"]
    for path_key, hash_key in (
        ("parent_config", "parent_config_sha256"),
        ("parent_checkpoint", "parent_checkpoint_sha256"),
        ("parent_optimizer", "parent_optimizer_sha256"),
        ("parent_result", "parent_result_sha256"),
    ):
        path = ROOT / str(initialization[path_key])
        if not path.is_file() or sha256(path) != initialization[hash_key]:
            raise RuntimeError(f"external continuation parent identity changed: {path}")
    parent_config = json.loads((ROOT / str(initialization["parent_config"])).read_text())
    parent_result = json.loads((ROOT / str(initialization["parent_result"])).read_text())
    expected_parent_status = (
        "user_authorized_haze_source_diverse_pilot"
        if status == "user_authorized_haze_source_diverse_continuation"
        else "user_authorized_rain_source_diverse_pilot"
    )
    expected_records = 128 if status == "user_authorized_haze_source_diverse_continuation" else 100
    if (
        parent_config.get("status") != expected_parent_status
        or int(parent_config["training"]["max_steps"]) != 2000
        or parent_result.get("checkpoint_sha256") != initialization["parent_checkpoint_sha256"]
        or parent_result.get("automatic_gate_pass") is not False
        or parent_result.get("descriptive_validation_metrics", {}).get("records") != expected_records
    ):
        raise RuntimeError("external continuation parent result contract changed")


def load_manifest(config: Mapping[str, object], *, verify_images: bool) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    data = config["data"]
    manifest = ROOT / str(data["frozen_manifest"])
    metadata_path = ROOT / str(data["frozen_manifest_metadata"])
    metadata = json.loads(metadata_path.read_text())
    config_sha = sha256(Path(config["_config_path"]))
    expected_manifest_config_sha = config_sha
    if config.get("status") == "user_authorized_haze_source_diverse_continuation":
        expected_manifest_config_sha = str(config["initialization"]["parent_config_sha256"])
    if (
        metadata.get("config_sha256") != expected_manifest_config_sha
        or metadata.get("frozen_manifest_sha256") != sha256(manifest)
        or metadata.get("protected_foundir_test_accessed") is not False
        or metadata.get("sealed_internal_accessed") is not False
    ):
        raise RuntimeError("frozen Blur-only manifest identity changed")
    with manifest.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    category = str(data["category"])
    train_role = str(data.get("train_role", "tiny_overfit_train"))
    validation_role = str(data.get("validation_role", "future_blur_validation"))
    train = [row for row in rows if row["category"] == category and row["role"] == train_role]
    validation = [row for row in rows if row["category"] == category and row["role"] == validation_role]
    if len(train) != int(data["tiny_overfit_records"]) or len(validation) != int(data["frozen_validation_records"]):
        raise RuntimeError("frozen single-task manifest cardinality changed")
    for role, selected, split in (
        ("train", train, data["tiny_overfit_split"]),
        ("validation", validation, data["frozen_validation_split"]),
    ):
        if any(row["category"] != category or row["composition_size"] != "1" or row["split"] != split for row in selected):
            raise RuntimeError(f"{role} contains a record outside the isolated task")
        if len({row["source_group"] for row in selected}) != len(selected):
            raise RuntimeError(f"{role} source groups are not unique")
    if {row["source_group"] for row in train} & {row["source_group"] for row in validation}:
        raise RuntimeError("train/validation source-group leakage")
    if verify_images:
        for row in rows:
            for field, hash_field in (("lq_path", "lq_sha256"), ("gt_path", "gt_sha256")):
                path = ROOT / row[field]
                if not path.is_file() or sha256(path) != row[hash_field]:
                    raise RuntimeError(f"pair identity changed: {row['record_id']} {field}")
    return train, validation


def load_crop(row: Mapping[str, str], crop_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    x, y = int(row["crop_x"]), int(row["crop_y"])
    tensors = []
    for field in ("lq_path", "gt_path"):
        with Image.open(ROOT / row[field]) as image:
            rgb = image.convert("RGB")
            if rgb.size != (int(row["width"]), int(row["height"])):
                raise RuntimeError(f"pair dimensions changed: {row['record_id']}")
            crop = np.asarray(rgb.crop((x, y, x + crop_size, y + crop_size)), dtype=np.uint8).copy()
        tensors.append(torch.from_numpy(crop).permute(2, 0, 1).float().div_(127.5).sub_(1.0))
    return tensors[0], tensors[1]


def prepare_records(rows: Sequence[Mapping[str, str]], crop_size: int) -> list[dict[str, object]]:
    return [{**dict(row), "lq": lq, "gt": gt} for row in rows for lq, gt in [load_crop(row, crop_size)]]


def encode_prompt(config: Mapping[str, object], device: torch.device) -> tuple[torch.Tensor, ...]:
    from flux2.sampling import batched_prc_txt
    from flux2.util import load_text_encoder

    encoder = load_text_encoder(str(config["model"]["name"]), device=device).eval().requires_grad_(False)
    with torch.inference_mode():
        encoded = encoder(["", str(config["model"]["prompt"])]).to(torch.bfloat16)
        contexts, ids = batched_prc_txt(encoded)
    result = (contexts[1:2].contiguous(), ids[1:2].contiguous(), contexts.contiguous(), ids.contiguous())
    del encoder
    gc.collect()
    torch.cuda.empty_cache()
    return result


def unpack_tokens(tokens: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    batch, channels, height, width = shape
    if tokens.shape != (batch, height * width, channels):
        raise RuntimeError(f"unexpected packed token shape: {tokens.shape} for {shape}")
    return tokens.transpose(1, 2).reshape(shape)


def scheduled_index(position: int, length: int, seed: int) -> int:
    epoch, offset = divmod(position, length)
    generator = torch.Generator(device="cpu").manual_seed(seed + epoch)
    return int(torch.randperm(length, generator=generator)[offset])


def strength_bin(input_psnr: float, thresholds: Sequence[float] = (28.0, 34.0)) -> str:
    if input_psnr < float(thresholds[0]):
        return "heavy"
    if input_psnr < float(thresholds[1]):
        return "moderate"
    return "near_clean"


def build_strength_buckets(
    records: Sequence[Mapping[str, object]], thresholds: Sequence[float]
) -> dict[str, list[int]]:
    buckets = {name: [] for name in ("heavy", "moderate", "near_clean")}
    for index, record in enumerate(records):
        bucket = strength_bin(psnr(record["gt"], record["lq"]), thresholds)
        buckets[bucket].append(index)
    if any(not indexes for indexes in buckets.values()):
        raise RuntimeError(f"empty Rain strength bucket: { {key: len(value) for key, value in buckets.items()} }")
    return buckets


def strength_balanced_index(
    position: int, buckets: Mapping[str, Sequence[int]], seed: int
) -> tuple[int, str]:
    names = ("heavy", "moderate", "near_clean")
    bucket_name = names[position % len(names)]
    within_bucket_position = position // len(names)
    indexes = buckets[bucket_name]
    bucket_seed = seed + stable_int(bucket_name) % 1_000_000_007
    selected = scheduled_index(within_bucket_position, len(indexes), bucket_seed)
    return int(indexes[selected]), bucket_name


def strength_weighted_index(
    position: int, buckets: Mapping[str, Sequence[int]], seed: int,
    weights: Sequence[int],
) -> tuple[int, str]:
    names = ("heavy", "moderate", "near_clean")
    cycle = [name for name, weight in zip(names, weights) for _ in range(int(weight))]
    if len(weights) != len(names) or any(int(weight) <= 0 for weight in weights):
        raise RuntimeError("invalid Rain strength sampling weights")
    offset = position % len(cycle)
    bucket_name = cycle[offset]
    completed_cycles = position // len(cycle)
    within_cycle_occurrence = sum(name == bucket_name for name in cycle[:offset])
    within_bucket_position = completed_cycles * int(weights[names.index(bucket_name)]) + within_cycle_occurrence
    indexes = buckets[bucket_name]
    bucket_seed = seed + stable_int(bucket_name) % 1_000_000_007
    selected = scheduled_index(within_bucket_position, len(indexes), bucket_seed)
    return int(indexes[selected]), bucket_name


def aggregate_strength_metrics(
    metrics: Mapping[str, object], thresholds: Sequence[float]
) -> dict[str, dict[str, float | int]]:
    result: dict[str, dict[str, float | int]] = {}
    rows = metrics["rows"]
    for bucket_name in ("heavy", "moderate", "near_clean"):
        selected = [row for row in rows if strength_bin(float(row["input_psnr"]), thresholds) == bucket_name]
        if not selected:
            raise RuntimeError(f"development metrics have no {bucket_name} records")
        result[bucket_name] = {
            "records": len(selected),
            "mean_input_psnr": float(np.mean([row["input_psnr"] for row in selected])),
            "mean_output_psnr": float(np.mean([row["output_psnr"] for row in selected])),
            "mean_input_lpips": float(np.mean([row["input_lpips"] for row in selected])),
            "mean_output_lpips": float(np.mean([row["output_lpips"] for row in selected])),
            "psnr_improved_fraction": sum(row["psnr_gain_db"] > 0 for row in selected) / len(selected),
            "lpips_improved_fraction": sum(row["lpips_improvement"] > 0 for row in selected) / len(selected),
        }
    return result


def probe_loss(model, autoencoder, record: Mapping[str, object], context, context_ids, pack_image, *, seed: int, device: torch.device) -> float:
    lq = record["lq"].unsqueeze(0).to(device)
    gt = record["gt"].unsqueeze(0).to(device)
    with torch.inference_mode():
        clean, degraded = autoencoder.encode(torch.cat([gt, lq])).chunk(2)
        generator = torch.Generator(device=device).manual_seed(seed)
        noise = torch.randn(clean.shape, generator=generator, device=device, dtype=torch.bfloat16)
        flow = build_reference_flow_batch(
            clean.to(torch.bfloat16), degraded.to(torch.bfloat16), noise,
            torch.tensor([0.5], device=device, dtype=torch.bfloat16), pack_image,
            reference_time_coordinate=10,
        )
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            prediction = model(
                x=flow.model_tokens, x_ids=flow.model_ids, timesteps=flow.timesteps,
                ctx=context, ctx_ids=context_ids, guidance=None,
            )[:, : flow.target_token_count]
        loss = torch.nn.functional.mse_loss(prediction.float(), flow.target_velocity.float())
    value = torch.tensor([float(loss)], device=device)
    dist.all_reduce(value)
    return float(value.item() / dist.get_world_size())


def tensor_image(tensor: torch.Tensor) -> Image.Image:
    array = tensor.detach().float().clamp(-1, 1).add(1).mul(127.5).round().byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(array, mode="RGB")


def psnr(reference: torch.Tensor, prediction: torch.Tensor) -> float:
    mse = float(torch.mean(((reference.float() - prediction.float()) * 0.5) ** 2))
    return 100.0 if mse <= 1e-12 else -10.0 * math.log10(mse)


def lpips_value(metric, reference: torch.Tensor, prediction: torch.Tensor) -> float:
    with torch.inference_mode():
        return float(metric((prediction.float() + 1) * 0.5, (reference.float() + 1) * 0.5).item())


def save_triptych(path: Path, record_id: str, lq: torch.Tensor, output: torch.Tensor, gt: torch.Tensor) -> None:
    images = [tensor_image(value[0]) for value in (lq, output, gt)]
    labels = ("Input", "FLUX.2 Output", "Ground Truth")
    canvas = Image.new("RGB", (images[0].width * 3, images[0].height + 28), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (label, image) in enumerate(zip(labels, images)):
        x = index * image.width
        canvas.paste(image, (x, 28))
        draw.text((x + 6, 7), f"{record_id} | {label}", fill="black")
    buffer = io.BytesIO()
    canvas.save(buffer, format="PNG")
    write_once(path, buffer.getvalue())


def restore(model, autoencoder, lq, cfg_context, cfg_ids, pack_image, *, seed: int, steps: int, reference_time: int):
    from flux2.sampling import denoise_cfg, get_schedule, scatter_ids

    with torch.inference_mode():
        reference = autoencoder.encode(lq).to(torch.bfloat16)
        reference_tokens, reference_ids = pack_image(reference)
        reference_ids = reference_ids.clone()
        reference_ids[..., 0] = reference_time
        generator = torch.Generator(device=lq.device).manual_seed(seed)
        noise = torch.randn(reference.shape, generator=generator, device=lq.device, dtype=torch.bfloat16)
        tokens, ids = pack_image(noise)
        schedule = get_schedule(steps, tokens.shape[1])
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            restored = denoise_cfg(
                model, tokens, ids, cfg_context, cfg_ids, timesteps=schedule, guidance=1.0,
                img_cond_seq=reference_tokens, img_cond_seq_ids=reference_ids,
            )
        latent = torch.cat(scatter_ids(restored, ids)).squeeze(2)
        return autoencoder.decode(latent).float().clamp(-1, 1)


def evaluate_snapshot(model, autoencoder, records, cfg_context, cfg_ids, pack_image, metric, *, step: int, limit: int, config, output_dir: Path, rank: int, world: int, device: torch.device, role: str = "training") -> dict[str, object] | None:
    selected = records[:limit]
    local = []
    snapshot_dir = output_dir / f"{role}_triptychs" / f"step_{step:06d}"
    for index in range(rank, len(selected), world):
        record = selected[index]
        lq = record["lq"].unsqueeze(0).to(device)
        gt = record["gt"].unsqueeze(0).to(device)
        output = restore(
            model, autoencoder, lq, cfg_context, cfg_ids, pack_image,
            seed=int(config["inference"]["seed"]) + index,
            steps=int(config["inference"]["num_steps"]),
            reference_time=int(config["training"]["reference_time_coordinate"]),
        )
        # Gate exactly what is saved: quantize to PNG precision before metrics.
        output_q = torch.from_numpy(np.asarray(tensor_image(output[0]), dtype=np.uint8).copy()).permute(2, 0, 1).float().div_(127.5).sub_(1).unsqueeze(0).to(device)
        row = {
            "record_id": record["record_id"], "source_group": record["source_group"],
            "category": record["category"], "role": role, "step": step,
            "input_psnr": psnr(gt, lq), "output_psnr": psnr(gt, output_q),
            "input_lpips": lpips_value(metric, gt, lq), "output_lpips": lpips_value(metric, gt, output_q),
        }
        row["psnr_gain_db"] = row["output_psnr"] - row["input_psnr"]
        row["lpips_improvement"] = row["input_lpips"] - row["output_lpips"]
        local.append(row)
        save_triptych(snapshot_dir / f"{index:02d}_{record['record_id']}.png", str(record["record_id"]), lq, output_q, gt)
    gathered: list[object | None] = [None] * world
    dist.all_gather_object(gathered, local)
    dist.barrier()
    if rank != 0:
        return None
    rows = sorted([row for part in gathered if isinstance(part, list) for row in part], key=lambda row: row["record_id"])
    if len(rows) != limit:
        raise RuntimeError(f"snapshot row count mismatch: {len(rows)} != {limit}")
    aggregate = {
        "schema_version": 1, "status": "complete", "step": step, "records": limit,
        "mean_input_psnr": float(np.mean([row["input_psnr"] for row in rows])),
        "mean_output_psnr": float(np.mean([row["output_psnr"] for row in rows])),
        "mean_psnr_gain_db": float(np.mean([row["psnr_gain_db"] for row in rows])),
        "mean_input_lpips": float(np.mean([row["input_lpips"] for row in rows])),
        "mean_output_lpips": float(np.mean([row["output_lpips"] for row in rows])),
        "mean_lpips_improvement": float(np.mean([row["lpips_improvement"] for row in rows])),
        "psnr_improved_fraction": sum(row["psnr_gain_db"] > 0 for row in rows) / limit,
        "lpips_improved_fraction": sum(row["lpips_improvement"] > 0 for row in rows) / limit,
        "rows": rows, "config_sha256": sha256(Path(config["_config_path"])),
        "protected_foundir_test_accessed": False, "sealed_internal_accessed": False,
    }
    write_once(snapshot_dir / "metrics.json", (json.dumps(aggregate, indent=2, sort_keys=True) + "\n").encode())
    print("FOUNDIR_SINGLE_TASK_SNAPSHOT " + json.dumps({key: value for key, value in aggregate.items() if key != "rows"}, sort_keys=True), flush=True)
    return aggregate


def preflight(config_path: Path, task: str | None = None, *, verify_model_hashes: bool = True) -> None:
    config = materialize_task_config(json.loads(config_path.read_text()), task)
    config["_config_path"] = str(config_path.resolve())
    validate_config(config)
    validate_external_continuation(config)
    train, validation = load_manifest(config, verify_images=True)
    if verify_model_hashes:
        for key, hash_key in (("flow_weights", "flow_weights_sha256"), ("autoencoder_weights", "autoencoder_weights_sha256")):
            path = ROOT / config["model"][key]
            if sha256(path) != config["model"][hash_key]:
                raise RuntimeError(f"model artifact identity changed: {path}")
    crop = int(config["data"]["crop_size"])
    for row in [*train, *validation]:
        load_crop(row, crop)
    print(json.dumps({
        "status": "FOUNDIR_SINGLE_TASK_PREFLIGHT_PASS", "task": config["scope"]["category"],
        "config_sha256": sha256(config_path),
        "tiny_overfit_records": len(train), "future_validation_records": len(validation),
        "descriptive_validation_frozen": True, "model_hashes_verified": verify_model_hashes,
        "protected_foundir_test_accessed": False,
        "sealed_internal_accessed": False,
    }, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--task", choices=("05Noise", "08Haze", "10Rain", "14Lowlight"))
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--skip-model-hashes", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config_path = args.config.resolve()
    if args.preflight:
        preflight(config_path, args.task, verify_model_hashes=not args.skip_model_hashes)
        return 0
    config = materialize_task_config(json.loads(config_path.read_text()), args.task)
    config["_config_path"] = str(config_path)
    validate_config(config)
    validate_external_continuation(config)
    train_rows, validation_rows = load_manifest(config, verify_images=False)
    records = prepare_records(train_rows, int(config["data"]["crop_size"]))
    validation_records = prepare_records(validation_rows, int(config["data"]["crop_size"]))
    output_dir = ROOT / config["runtime"]["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "result.json").exists():
        raise RuntimeError("single-task sanity result already exists; refusing rerun")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend=str(config["distributed"]["backend"]), device_id=device)
    started = time.monotonic()
    try:
        rank, world = dist.get_rank(), dist.get_world_size()
        required_world = int(config["distributed"]["required_world_size"])
        if world != required_world or torch.cuda.device_count() != required_world:
            raise RuntimeError(f"single-task sanity gate requires exactly {required_world} visible GPUs")
        properties = torch.cuda.get_device_properties(device)
        if properties.total_memory < int(config["distributed"]["minimum_memory_gib"]) * 1024**3:
            raise RuntimeError("GPU memory is below contract")
        if not any(fragment in properties.name for fragment in config["distributed"]["allowed_gpu_name_fragments"]):
            raise RuntimeError(f"unsupported GPU: {properties.name}")
        seed = int(config["training"]["seed"])
        random.seed(seed + rank)
        np.random.seed(seed + rank)
        torch.manual_seed(seed + rank)
        torch.cuda.manual_seed_all(seed + rank)
        os.environ.update(
            HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TORCH_HOME=str(ROOT / config["runtime"]["torch_home"]),
            KLEIN_4B_BASE_MODEL_PATH=str(ROOT / config["model"]["flow_weights"]),
            AE_MODEL_PATH=str(ROOT / config["model"]["autoencoder_weights"]),
            QWEN3_MODEL_PATH=str(ROOT / config["model"]["text_encoder"]),
        )
        context, context_ids, cfg_context, cfg_ids = encode_prompt(config, device)
        from flux2.sampling import batched_prc_img
        from flux2.util import load_ae, load_flow_model
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

        autoencoder = load_ae(config["model"]["name"], device=device).eval().requires_grad_(False)
        model = load_flow_model(config["model"]["name"], device=device).eval().requires_grad_(False)
        names = inject_lora(
            model, list(config["lora"]["target_patterns"]), rank=int(config["lora"]["rank"]),
            alpha=float(config["lora"]["alpha"]), dropout=float(config["lora"]["dropout"]),
        )
        if len(names) != int(config["lora"]["expected_module_count"]):
            raise RuntimeError("FLUX LoRA target count changed")
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        wrapped = DistributedDataParallel(
            model, device_ids=[local_rank], broadcast_buffers=False,
            find_unused_parameters=False, gradient_as_bucket_view=True,
        )
        optimizer = torch.optim.AdamW(
            trainable, lr=float(config["training"]["learning_rate"]),
            weight_decay=float(config["training"]["weight_decay"]), fused=True,
        )
        start_step = 0
        external_continuation = config.get("status") in {
            "user_authorized_haze_source_diverse_continuation",
            "user_authorized_rain_strength_identity_directional_gate",
            "user_authorized_rain_strength_identity_ratio_gate",
        }
        rain_directional = config.get("status") in {
            "user_authorized_rain_strength_identity_directional_gate",
            "user_authorized_rain_strength_identity_ratio_gate",
        }
        rain_ratio = config.get("status") == "user_authorized_rain_strength_identity_ratio_gate"
        if args.resume:
            checkpoints = sorted(output_dir.glob("step_*.safetensors"))
            if checkpoints:
                latest = checkpoints[-1]
                start_step = int(latest.stem.split("_")[-1])
                load_lora_checkpoint(wrapped.module, latest)
                optimizer_payload = torch.load(latest.with_suffix(".optimizer.pt"), map_location="cpu", weights_only=False)
                if optimizer_payload.get("config_sha256") != sha256(config_path) or optimizer_payload.get("step") != start_step:
                    raise RuntimeError("resume optimizer identity changed")
                optimizer.load_state_dict(optimizer_payload["optimizer"])
        elif external_continuation:
            initialization = config["initialization"]
            start_step = int(initialization["start_step"])
            load_lora_checkpoint(wrapped.module, ROOT / str(initialization["parent_checkpoint"]))
            optimizer_payload = torch.load(
                ROOT / str(initialization["parent_optimizer"]), map_location="cpu", weights_only=False
            )
            if (
                optimizer_payload.get("config_sha256") != initialization["parent_config_sha256"]
                or optimizer_payload.get("step") != start_step
            ):
                raise RuntimeError("external continuation optimizer identity changed")
            optimizer.load_state_dict(optimizer_payload["optimizer"])
            # load_state_dict restores the parent's LR.  The Rain directional
            # gate intentionally uses the lower, config-bound continuation LR.
            for group in optimizer.param_groups:
                group["lr"] = float(config["training"]["learning_rate"])
        elif any(output_dir.iterdir()):
            raise RuntimeError("output directory is nonempty; use --resume only after inspecting a runtime interruption")

        metric = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(device).eval()
        baseline_path = output_dir / "baseline_probe.json"
        if baseline_path.exists():
            baseline = json.loads(baseline_path.read_text())
            initial_probe = float(baseline["probe_loss"])
        else:
            initial_probe = probe_loss(
                wrapped, autoencoder, records[rank % len(records)], context, context_ids, batched_prc_img,
                seed=seed + rank * 1009, device=device,
            )
            if rank == 0:
                write_once(baseline_path, (json.dumps({"config_sha256": sha256(config_path), "probe_loss": initial_probe}, indent=2, sort_keys=True) + "\n").encode())
        dist.barrier()
        parent_validation_metrics = None
        if start_step == 0:
            evaluate_snapshot(
                wrapped, autoencoder, records, cfg_context, cfg_ids, batched_prc_img, metric,
                step=0, limit=int(config["inference"]["visual_records"]), config=config,
                output_dir=output_dir, rank=rank, world=world, device=device,
            )
            evaluate_snapshot(
                wrapped, autoencoder, validation_records, cfg_context, cfg_ids, batched_prc_img, metric,
                step=0, limit=int(config["inference"].get("validation_visual_records", len(validation_records))),
                config=config, output_dir=output_dir, rank=rank, world=world, device=device,
                role="validation",
            )
        elif rain_directional:
            if start_step not in {int(value) for value in config["inference"]["evaluate_steps"]}:
                raise RuntimeError("Rain parent step is absent from the evaluation schedule")
            evaluate_snapshot(
                wrapped, autoencoder, records, cfg_context, cfg_ids, batched_prc_img, metric,
                step=start_step, limit=int(config["inference"]["visual_records"]), config=config,
                output_dir=output_dir, rank=rank, world=world, device=device,
            )
            parent_validation_metrics = evaluate_snapshot(
                wrapped, autoencoder, validation_records, cfg_context, cfg_ids, batched_prc_img, metric,
                step=start_step, limit=len(validation_records), config=config,
                output_dir=output_dir, rank=rank, world=world, device=device, role="validation",
            )
        training = config["training"]
        accumulation = int(training["gradient_accumulation_steps"])
        strength_buckets = None
        strength_thresholds = (28.0, 34.0)
        if rain_directional:
            strength_thresholds = tuple(float(value) for value in training["strength_identity_policy"]["input_psnr_bins"])
            strength_buckets = build_strength_buckets(records, strength_thresholds)
            counts = {key: len(value) for key, value in strength_buckets.items()}
            if rank == 0:
                print("FOUNDIR_RAIN_STRENGTH_BUCKETS " + json.dumps(counts, sort_keys=True), flush=True)
        optimizer.zero_grad(set_to_none=True)
        for step in range(start_step, int(training["max_steps"])):
            accumulated = 0.0
            for micro in range(accumulation):
                position = (step * accumulation + micro) * world + rank
                if rain_directional:
                    assert strength_buckets is not None
                    if rain_ratio:
                        record_index, bucket_name = strength_weighted_index(
                            position, strength_buckets, seed,
                            training["strength_identity_policy"]["sampling_weights"],
                        )
                    else:
                        record_index, bucket_name = strength_balanced_index(position, strength_buckets, seed)
                    record = records[record_index]
                else:
                    bucket_name = "gt"
                    record = records[scheduled_index(position, len(records), seed)]
                lq = record["lq"].unsqueeze(0).to(device)
                gt = record["gt"].unsqueeze(0).to(device)
                target = lq if rain_directional and bucket_name == "near_clean" else gt
                with torch.no_grad():
                    clean, degraded = autoencoder.encode(torch.cat([target, lq])).chunk(2)
                generator = torch.Generator(device=device).manual_seed(seed + position * 1_000_003)
                noise = torch.randn(clean.shape, generator=generator, device=device, dtype=torch.bfloat16)
                timesteps = torch.rand((1,), generator=generator, device=device, dtype=torch.bfloat16).clamp_(1e-4, 1 - 1e-4)
                flow = build_reference_flow_batch(
                    clean.to(torch.bfloat16), degraded.to(torch.bfloat16), noise, timesteps,
                    batched_prc_img, reference_time_coordinate=int(training["reference_time_coordinate"]),
                )
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    prediction = wrapped(
                        x=flow.model_tokens, x_ids=flow.model_ids, timesteps=flow.timesteps,
                        ctx=context, ctx_ids=context_ids, guidance=None,
                    )[:, : flow.target_token_count]
                    loss = torch.nn.functional.mse_loss(prediction.float(), flow.target_velocity.float())
                (loss / accumulation).backward()
                accumulated += float(loss.detach())
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, float(training["max_grad_norm"]))
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            value = torch.tensor([accumulated / accumulation], device=device)
            dist.all_reduce(value)
            mean_loss = float(value.item() / world)
            completed = step + 1
            if rank == 0 and (completed == 1 or completed % 10 == 0):
                print(f"FOUNDIR_SINGLE_TASK_TRAIN task={config['scope']['category']} step={completed}/{training['max_steps']} loss={mean_loss:.8f} grad_norm={float(grad_norm):.6f}", flush=True)
            if completed in set(int(value) for value in training["checkpoint_steps"]):
                dist.barrier()
                if rank == 0:
                    checkpoint = output_dir / f"step_{completed:06d}.safetensors"
                    if checkpoint.exists():
                        raise RuntimeError(f"refusing to overwrite checkpoint: {checkpoint}")
                    save_lora_checkpoint(wrapped.module, checkpoint, {
                        "step": completed, "config_sha256": sha256(config_path),
                        "scope": f"user_authorized_{config['scope']['category']}_dit_lora_capacity_only",
                    })
                    with checkpoint.with_suffix(".optimizer.pt").open("xb") as handle:
                        torch.save({"step": completed, "config_sha256": sha256(config_path), "optimizer": optimizer.state_dict()}, handle)
                dist.barrier()
                evaluate_snapshot(
                    wrapped, autoencoder, records, cfg_context, cfg_ids, batched_prc_img, metric,
                    step=completed, limit=int(config["inference"]["visual_records"]), config=config,
                    output_dir=output_dir, rank=rank, world=world, device=device,
                )
                evaluate_snapshot(
                    wrapped, autoencoder, validation_records, cfg_context, cfg_ids, batched_prc_img, metric,
                    step=completed, limit=int(config["inference"].get("validation_visual_records", len(validation_records))),
                    config=config, output_dir=output_dir, rank=rank, world=world, device=device,
                    role="validation",
                )
        final_probe = probe_loss(
            wrapped, autoencoder, records[rank % len(records)], context, context_ids, batched_prc_img,
            seed=seed + rank * 1009, device=device,
        )
        final_train_limit = int(config["inference"].get("final_train_metric_records", len(records)))
        final_validation_limit = int(config["inference"].get("final_validation_metric_records", len(validation_records)))
        final_metrics = evaluate_snapshot(
            wrapped, autoencoder, records, cfg_context, cfg_ids, batched_prc_img, metric,
            step=int(training["max_steps"]), limit=final_train_limit, config=config,
            output_dir=output_dir / "final_all_16", rank=rank, world=world, device=device,
        )
        final_validation_metrics = evaluate_snapshot(
            wrapped, autoencoder, validation_records, cfg_context, cfg_ids, batched_prc_img, metric,
            step=int(training["max_steps"]), limit=final_validation_limit, config=config,
            output_dir=output_dir / "final_all_16", rank=rank, world=world, device=device,
            role="validation",
        )
        if rank == 0:
            assert final_metrics is not None and final_validation_metrics is not None
            gate = config["gate"]
            loss_ratio = final_probe / initial_probe
            source_diverse_pilot = config.get("status") in {
                "user_authorized_haze_source_diverse_pilot",
                "user_authorized_haze_source_diverse_continuation",
                "user_authorized_rain_source_diverse_pilot",
            }
            if rain_directional:
                if parent_validation_metrics is None:
                    parent_metrics_path = output_dir / "validation_triptychs" / f"step_{start_step:06d}" / "metrics.json"
                    parent_validation_metrics = json.loads(parent_metrics_path.read_text())
                parent_strength = aggregate_strength_metrics(parent_validation_metrics, strength_thresholds)
                candidate_strength = aggregate_strength_metrics(final_validation_metrics, strength_thresholds)
                checks = {
                    "near_clean_psnr_gain_vs_parent": candidate_strength["near_clean"]["mean_output_psnr"]
                    >= parent_strength["near_clean"]["mean_output_psnr"] + float(gate["near_clean_minimum_psnr_gain_db"]),
                    "moderate_psnr_retention_vs_parent": candidate_strength["moderate"]["mean_output_psnr"]
                    >= parent_strength["moderate"]["mean_output_psnr"] - float(gate["moderate_maximum_psnr_regression_db"]),
                    "heavy_psnr_retention_vs_parent": candidate_strength["heavy"]["mean_output_psnr"]
                    >= parent_strength["heavy"]["mean_output_psnr"] - float(gate["heavy_maximum_psnr_regression_db"]),
                    "overall_psnr_gain_vs_parent": final_validation_metrics["mean_output_psnr"]
                    >= parent_validation_metrics["mean_output_psnr"] + float(gate["overall_minimum_psnr_gain_db"]),
                    "overall_lpips_retention_vs_parent": final_validation_metrics["mean_output_lpips"]
                    <= parent_validation_metrics["mean_output_lpips"] + float(gate["overall_maximum_lpips_regression"]),
                }
            elif source_diverse_pilot:
                checks = {
                    "validation_psnr_sow": final_validation_metrics["mean_output_psnr"] >= float(gate["validation_psnr_minimum"]),
                    "validation_lpips_sow": final_validation_metrics["mean_output_lpips"] <= float(gate["validation_lpips_maximum"]),
                }
            else:
                checks = {
                    "probe_loss_ratio": loss_ratio <= float(gate["maximum_probe_loss_ratio"]),
                    "mean_psnr_gain_db": final_metrics["mean_psnr_gain_db"] >= float(gate["minimum_mean_psnr_gain_db"]),
                    "mean_lpips_improvement": final_metrics["mean_lpips_improvement"] >= float(gate["minimum_mean_lpips_improvement"]),
                    "psnr_improved_fraction": final_metrics["psnr_improved_fraction"] >= float(gate["minimum_psnr_improved_fraction"]),
                    "lpips_improved_fraction": final_metrics["lpips_improved_fraction"] >= float(gate["minimum_lpips_improved_fraction"]),
                }
            automatic_pass = all(checks.values())
            sow = config.get("_sow_stage2")
            sow_proxy = None if not isinstance(sow, dict) else {
                "psnr": final_metrics["mean_output_psnr"] >= float(sow["psnr"]),
                "lpips": final_metrics["mean_output_lpips"] <= float(sow["lpips"]),
                "thresholds": sow,
            }
            checkpoint = output_dir / f"step_{int(training['max_steps']):06d}.safetensors"
            payload = {
                "schema_version": 1,
                "status": "awaiting_human_visual_review" if automatic_pass else "failed",
                "automatic_gate_pass": automatic_pass, "human_visual_review": "pending",
                "checks": checks, "initial_probe_loss": initial_probe, "final_probe_loss": final_probe,
                "task": config["scope"]["category"],
                "probe_loss_ratio": loss_ratio, "final_metrics": final_metrics,
                "descriptive_validation_metrics": final_validation_metrics,
                "validation_metrics_are_formal_gate": source_diverse_pilot or rain_directional,
                "automatic_full_development_launch": False,
                "train_sow_proxy": sow_proxy,
                "config": config_path.relative_to(ROOT).as_posix(), "config_sha256": sha256(config_path),
                "checkpoint": checkpoint.relative_to(ROOT).as_posix(), "checkpoint_sha256": sha256(checkpoint),
                "flow_objective": training["objective"], "inference_schedule": config["inference"]["schedule"],
                "inference_steps": int(config["inference"]["num_steps"]), "guidance": 1.0,
                "qwen_degradation_conditioning": False, "mixed_degradations": False,
                "pixel_refiner": False, "development_gt_optimization": False,
                "protected_foundir_test_accessed": False, "sealed_internal_accessed": False,
                "wall_seconds": time.monotonic() - started,
            }
            if rain_directional:
                payload.update({
                    "directional_gate_only": True,
                    "automatic_continuation": False,
                    "parent_development_metrics": parent_validation_metrics,
                    "parent_strength_metrics": parent_strength,
                    "candidate_strength_metrics": candidate_strength,
                    "strength_bucket_counts": {key: len(value) for key, value in strength_buckets.items()},
                    "near_clean_training_target": "lq_identity",
                    "heavy_moderate_training_target": "gt",
                })
            write_once(output_dir / "result.json", (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode())
            print("FOUNDIR_SINGLE_TASK_AUTOMATIC_PASS_AWAITING_VISUAL_REVIEW" if automatic_pass else "FOUNDIR_SINGLE_TASK_AUTOMATIC_GATE_FAILED", flush=True)
            # A completed scientific gate is a successful distributed workflow.
            # Keep pass/fail in result.json; a nonzero rank exit is normalized by
            # torchrun into a runtime failure and destroys that distinction.
            exit_value = 0
        else:
            exit_value = 0
        exit_tensor = torch.tensor([exit_value], device=device, dtype=torch.int32)
        dist.broadcast(exit_tensor, src=0)
        dist.barrier()
        return int(exit_tensor.item())
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
