#!/usr/bin/env python3
"""Train a one-step, RGB-aligned Rain FLUX LoRA with a frozen Restormer teacher."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party/flux2/src"))

from lowlight_restoration.lora import (  # noqa: E402
    inject_lora,
    load_lora_checkpoint,
    save_lora_checkpoint,
)
from lowlight_restoration.pixel_refinement import spatial_highpass  # noqa: E402
from lowlight_restoration.task_native_restormer import load_task_native_restormer  # noqa: E402
from smoke.train_foundir_blur_only_flux2_overfit_ddp import (  # noqa: E402
    encode_prompt,
    load_crop,
    lpips_value,
    psnr,
    save_triptych,
    sha256,
    tensor_image,
    unpack_tokens,
    write_once,
)


def validate_config(config: Mapping[str, object], config_path: Path) -> None:
    if config.get("status") != "user_authorized_rain_flux2_direct_endpoint_distillation":
        raise RuntimeError("direct-endpoint authorization changed")
    if config["scope"] != {
        "category": "10Rain",
        "official_foundir_test_accessed": False,
        "sealed_internal_accessed": False,
    }:
        raise RuntimeError("Rain-only safe scope changed")
    data = config["data"]
    training = config["training"]
    losses = training["losses"]
    inference = config["inference"]
    distributed = config["distributed"]
    qualification = bool(config["gate"]["qualification_only"])
    expected_steps = 25 if qualification else 3000
    expected_checkpoints = [25] if qualification else [250, 500, 1000, 2000, 3000]
    required_losses = {
        "flow_mse_weight": 0.1,
        "predicted_clean_latent_l1_weight": 0.25,
        "final_rgb_mse_weight": 10.0,
        "rgb_charbonnier_weight": 1.0,
        "highpass_l1_weight": 0.1,
        "lpips_weight": 0.05,
        "teacher_l1_weight": 0.5,
        "unchanged_identity_weight": 0.5,
        "rain_change_weight": 2.0,
        "rain_change_scale_m11": 0.08,
        "charbonnier_epsilon": 0.001,
    }
    checks = (
        config["model"]["name"] == "flux.2-klein-base-4b",
        data["sampling"] == "uniform_source_group_then_uniform_record",
        data["geometric_augmentation"] is True,
        int(data["train_records"]) == 38334,
        int(data["train_source_groups"]) == 5780,
        int(data["validation_records"]) == 100,
        int(data["validation_source_groups"]) == 100,
        int(data["train_crop_size"]) == 256,
        int(data["validation_crop_size"]) == 512,
        config["initialization"]["policy"] == "frozen_parent_lora_fresh_optimizer",
        config["teacher"]["frozen"] is True,
        config["teacher"]["inference_range"] == "zero_one",
        training["objective"] == "direct_endpoint_rgb_teacher_distillation",
        float(training["timestep"]) == 1.0,
        float(training["anchor_noise_scale"]) == 0.0,
        int(training["max_steps"]) == expected_steps,
        [int(value) for value in training["checkpoint_steps"]] == expected_checkpoints,
        int(training["global_batch_size"]) == 32,
        int(training["micro_batch_size_per_rank"]) == 1,
        training["gradient_accumulation_steps_by_world_size"] == {"4": 8, "8": 4},
        {key: float(losses[key]) for key in required_losses} == required_losses,
        inference["schedule"] == "direct_endpoint_one_step",
        int(inference["num_steps"]) == 1,
        [int(value) for value in inference["evaluate_steps"]] == expected_checkpoints,
        [int(value) for value in distributed["allowed_world_sizes"]] == [4, 8],
        config["gate"]["protocol"] == "frozen_100_source_disjoint_rain_development",
        float(config["gate"]["validation_psnr_minimum"]) == 34.2,
        float(config["gate"]["validation_lpips_maximum"]) == 0.08,
        config["gate"]["automatic_formal_launch"] is False,
    )
    if not all(checks):
        raise RuntimeError(f"direct-endpoint frozen contract changed: {config_path}")
    for world_text, accumulation in training["gradient_accumulation_steps_by_world_size"].items():
        if (
            int(world_text) not in {4, 8}
            or int(training["global_batch_size"])
            != int(world_text) * int(training["micro_batch_size_per_rank"]) * int(accumulation)
        ):
            raise RuntimeError("global batch arithmetic changed")


def verify_artifacts(config: Mapping[str, object]) -> None:
    artifacts = [
        (config["data"]["manifest"], config["data"]["manifest_sha256"]),
        (config["data"]["metadata"], config["data"]["metadata_sha256"]),
        (config["model"]["flow_weights"], config["model"]["flow_weights_sha256"]),
        (config["model"]["autoencoder_weights"], config["model"]["autoencoder_weights_sha256"]),
        (config["model"]["prompt_context_cache"], config["model"]["prompt_context_cache_sha256"]),
        (config["initialization"]["parent_lora"], config["initialization"]["parent_lora_sha256"]),
        (config["teacher"]["adapted_checkpoint"], config["teacher"]["adapted_checkpoint_sha256"]),
    ]
    for relative, expected in artifacts:
        path = ROOT / str(relative)
        if not path.is_file() or sha256(path) != str(expected):
            raise RuntimeError(f"artifact identity changed: {path}")


def load_rows(config: Mapping[str, object]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    data = config["data"]
    metadata = json.loads((ROOT / str(data["metadata"])).read_text())
    if (
        metadata.get("status") != "frozen_foundir_rain_direct_endpoint_distillation_v1"
        or metadata.get("frozen_manifest_sha256") != str(data["manifest_sha256"])
        or metadata.get("protected_foundir_test_accessed") is not False
        or metadata.get("sealed_internal_accessed") is not False
    ):
        raise RuntimeError("direct-endpoint manifest metadata changed")
    with (ROOT / str(data["manifest"])).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    train = [row for row in rows if row["role"] == "direct_endpoint_train"]
    validation = [row for row in rows if row["role"] == "direct_endpoint_validation"]
    if len(train) != int(data["train_records"]) or len(validation) != int(data["validation_records"]):
        raise RuntimeError("direct-endpoint manifest cardinality changed")
    train_groups = {row["source_group"] for row in train}
    validation_groups = {row["source_group"] for row in validation}
    if (
        len(train_groups) != int(data["train_source_groups"])
        or len(validation_groups) != int(data["validation_source_groups"])
        or train_groups & validation_groups
    ):
        raise RuntimeError("direct-endpoint source-group split changed or leaked")
    if any(row["category"] != "10Rain" or row["composition_size"] != "1" for row in rows):
        raise RuntimeError("manifest contains data outside singleton Rain")
    return train, validation


def group_index(rows: Sequence[Mapping[str, str]]) -> tuple[list[str], dict[str, list[int]]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[str(row["source_group"])].append(index)
    names = sorted(grouped)
    return names, dict(grouped)


def deterministic_choice(position: int, length: int, seed: int) -> int:
    epoch, offset = divmod(position, length)
    generator = torch.Generator(device="cpu").manual_seed(seed + epoch)
    return int(torch.randperm(length, generator=generator)[offset])


def sampled_row_index(
    position: int,
    group_names: Sequence[str],
    grouped: Mapping[str, Sequence[int]],
    seed: int,
) -> int:
    group_position = deterministic_choice(position, len(group_names), seed)
    group_name = group_names[group_position]
    candidates = grouped[group_name]
    occurrence = position // len(group_names)
    record_position = deterministic_choice(
        occurrence, len(candidates), seed + int(hashlib.sha256(group_name.encode()).hexdigest()[:8], 16)
    )
    return int(candidates[record_position])


def materialize(record: dict[str, object], crop_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    if "lq" not in record:
        record["lq"], record["gt"] = load_crop(record, crop_size)
    return record["lq"], record["gt"]


def augment_pair(lq: torch.Tensor, gt: torch.Tensor, code: int) -> tuple[torch.Tensor, torch.Tensor]:
    rotation = code % 4
    if rotation:
        lq, gt = torch.rot90(lq, rotation, (-2, -1)), torch.rot90(gt, rotation, (-2, -1))
    if code >= 4:
        lq, gt = torch.flip(lq, (-1,)), torch.flip(gt, (-1,))
    return lq.contiguous(), gt.contiguous()


def load_teacher(config: Mapping[str, object], device: torch.device) -> torch.nn.Module:
    teacher, _ = load_task_native_restormer(
        str(config["teacher"]["base_name"]), ROOT / str(config["teacher"]["base_directory"])
    )
    payload = torch.load(
        ROOT / str(config["teacher"]["adapted_checkpoint"]),
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    if (
        not isinstance(payload, dict)
        or payload.get("status") != "foundir_task_native_adaptation_checkpoint"
        or payload.get("task") != "rain"
        or int(payload.get("step", -1)) != 5000
        or not isinstance(payload.get("model"), dict)
    ):
        raise RuntimeError("adapted Rain Restormer checkpoint contract changed")
    teacher.load_state_dict(payload["model"], strict=True)
    return teacher.to(device).eval().requires_grad_(False)


def direct_endpoint(
    model: torch.nn.Module,
    autoencoder: torch.nn.Module,
    lq: torch.Tensor,
    context: torch.Tensor,
    context_ids: torch.Tensor,
    pack_image,
    reference_time: int,
) -> torch.Tensor:
    degraded = autoencoder.encode(lq).to(torch.bfloat16)
    tokens, ids = pack_image(degraded)
    reference_ids = ids.clone()
    reference_ids[..., 0] = reference_time
    timestep = torch.ones((lq.shape[0],), device=lq.device, dtype=torch.bfloat16)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        velocity = model(
            x=torch.cat([tokens, tokens], dim=1),
            x_ids=torch.cat([ids, reference_ids], dim=1),
            timesteps=timestep,
            ctx=context,
            ctx_ids=context_ids,
            guidance=None,
        )[:, : tokens.shape[1]]
        predicted_clean = degraded - unpack_tokens(velocity, degraded.shape)
        output = autoencoder.decode(predicted_clean)
    return output.float().clamp(-1, 1)


def evaluate(
    model,
    autoencoder,
    rows,
    context,
    context_ids,
    pack_image,
    metric,
    *,
    config,
    step: int,
    output_dir: Path,
    rank: int,
    world: int,
    device: torch.device,
) -> dict[str, object] | None:
    local: list[dict[str, object]] = []
    snapshot = output_dir / "validation_triptychs" / f"step_{step:06d}"
    crop = int(config["data"]["validation_crop_size"])
    model.eval()
    for index in range(rank, len(rows), world):
        record = rows[index]
        lq_cpu, gt_cpu = materialize(record, crop)
        lq, gt = lq_cpu.unsqueeze(0).to(device), gt_cpu.unsqueeze(0).to(device)
        with torch.inference_mode():
            output = direct_endpoint(
                model, autoencoder, lq, context, context_ids, pack_image,
                int(config["training"]["reference_time_coordinate"]),
            )
        output_q = (
            torch.from_numpy(np.asarray(tensor_image(output[0]), dtype=np.uint8).copy())
            .permute(2, 0, 1).float().div_(127.5).sub_(1).unsqueeze(0).to(device)
        )
        row = {
            "record_id": record["record_id"],
            "source_group": record["source_group"],
            "step": step,
            "input_psnr": psnr(gt, lq),
            "output_psnr": psnr(gt, output_q),
            "input_lpips": lpips_value(metric, gt, lq),
            "output_lpips": lpips_value(metric, gt, output_q),
        }
        row["psnr_gain_db"] = float(row["output_psnr"]) - float(row["input_psnr"])
        row["lpips_improvement"] = float(row["input_lpips"]) - float(row["output_lpips"])
        local.append(row)
        save_triptych(snapshot / f"{index:03d}_{record['record_id']}.png", str(record["record_id"]), lq, output_q, gt)
    gathered: list[object | None] = [None] * world
    dist.all_gather_object(gathered, local)
    dist.barrier()
    model.train()
    if rank != 0:
        return None
    combined = sorted(
        [row for shard in gathered if isinstance(shard, list) for row in shard],
        key=lambda row: str(row["record_id"]),
    )
    if len(combined) != len(rows):
        raise RuntimeError("validation gather lost records")
    result = {
        "schema_version": 1,
        "status": "complete",
        "step": step,
        "records": len(combined),
        "mean_input_psnr": float(np.mean([row["input_psnr"] for row in combined])),
        "mean_output_psnr": float(np.mean([row["output_psnr"] for row in combined])),
        "mean_input_lpips": float(np.mean([row["input_lpips"] for row in combined])),
        "mean_output_lpips": float(np.mean([row["output_lpips"] for row in combined])),
        "rows": combined,
        "official_foundir_test_accessed": False,
        "sealed_internal_accessed": False,
    }
    result["mean_psnr_gain_db"] = result["mean_output_psnr"] - result["mean_input_psnr"]
    result["mean_lpips_improvement"] = result["mean_input_lpips"] - result["mean_output_lpips"]
    write_once(snapshot / "metrics.json", (json.dumps(result, indent=2, sort_keys=True) + "\n").encode())
    print(
        "FOUNDIR_RAIN_DIRECT_ENDPOINT_VALIDATION "
        + json.dumps({key: value for key, value in result.items() if key != "rows"}, sort_keys=True),
        flush=True,
    )
    return result


def preflight(config_path: Path) -> None:
    config = json.loads(config_path.read_text())
    validate_config(config, config_path)
    verify_artifacts(config)
    train, validation = load_rows(config)
    # Geometry and crop checks are deterministic and do not decode all 38k images.
    for row, crop in (
        *[(row, int(config["data"]["train_crop_size"])) for row in train],
        *[(row, int(config["data"]["validation_crop_size"])) for row in validation],
    ):
        if (
            not (ROOT / row["lq_path"]).is_file()
            or not (ROOT / row["gt_path"]).is_file()
            or int(row["crop_x"]) < 0
            or int(row["crop_y"]) < 0
            or int(row["crop_x"]) + crop > int(row["width"])
            or int(row["crop_y"]) + crop > int(row["height"])
        ):
            raise RuntimeError(f"invalid crop/path contract: {row['record_id']}")
    print(
        "FOUNDIR_RAIN_DIRECT_ENDPOINT_PREFLIGHT_PASS "
        + json.dumps(
            {
                "config_sha256": sha256(config_path),
                "train_records": len(train),
                "train_source_groups": len({row["source_group"] for row in train}),
                "validation_records": len(validation),
                "official_foundir_test_accessed": False,
                "sealed_internal_accessed": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    config_path = args.config.resolve()
    if args.preflight:
        preflight(config_path)
        return 0
    config = json.loads(config_path.read_text())
    validate_config(config, config_path)
    train_rows, validation_rows = load_rows(config)
    records: list[dict[str, object]] = [dict(row) for row in train_rows]
    validation: list[dict[str, object]] = [dict(row) for row in validation_rows]
    group_names, grouped = group_index(train_rows)
    output_dir = ROOT / str(config["runtime"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise RuntimeError("direct-endpoint output is nonempty; qualification/formal must start independently")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend=str(config["distributed"]["backend"]), device_id=device)
    started = time.monotonic()
    try:
        rank, world = dist.get_rank(), dist.get_world_size()
        allowed_worlds = {int(value) for value in config["distributed"]["allowed_world_sizes"]}
        if world not in allowed_worlds or torch.cuda.device_count() != world:
            raise RuntimeError(
                f"direct-endpoint training requires 4 or 8 visible GPUs, got world={world} "
                f"visible={torch.cuda.device_count()}"
            )
        properties = torch.cuda.get_device_properties(device)
        if properties.total_memory < int(config["distributed"]["minimum_memory_gib"]) * 1024**3:
            raise RuntimeError("GPU memory is below the frozen contract")
        if not any(token in properties.name for token in config["distributed"]["allowed_gpu_name_fragments"]):
            raise RuntimeError(f"unsupported GPU: {properties.name}")
        artifact_error: list[object] = [None]
        if rank == 0:
            try:
                verify_artifacts(config)
            except Exception as error:  # broadcast the failure instead of hanging other ranks
                artifact_error[0] = f"{type(error).__name__}: {error}"
        dist.broadcast_object_list(artifact_error, src=0)
        if artifact_error[0] is not None:
            raise RuntimeError(str(artifact_error[0]))
        seed = int(config["training"]["seed"])
        random.seed(seed + rank)
        np.random.seed(seed + rank)
        torch.manual_seed(seed + rank)
        torch.cuda.manual_seed_all(seed + rank)
        os.environ.update(
            HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
            TORCH_HOME=str(ROOT / str(config["runtime"]["torch_home"])),
            KLEIN_4B_BASE_MODEL_PATH=str(ROOT / str(config["model"]["flow_weights"])),
            AE_MODEL_PATH=str(ROOT / str(config["model"]["autoencoder_weights"])),
            QWEN3_MODEL_PATH=str(ROOT / str(config["model"]["text_encoder"])),
        )
        context, context_ids, _, _ = encode_prompt(config, device)
        from flux2.sampling import batched_prc_img
        from flux2.util import load_ae, load_flow_model
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

        autoencoder = load_ae(str(config["model"]["name"]), device=device).eval().requires_grad_(False)
        model = load_flow_model(str(config["model"]["name"]), device=device).eval().requires_grad_(False)
        names = inject_lora(
            model,
            list(config["lora"]["target_patterns"]),
            rank=int(config["lora"]["rank"]),
            alpha=float(config["lora"]["alpha"]),
            dropout=float(config["lora"]["dropout"]),
        )
        if len(names) != int(config["lora"]["expected_module_count"]):
            raise RuntimeError("FLUX LoRA target count changed")
        load_lora_checkpoint(model, ROOT / str(config["initialization"]["parent_lora"]))
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        wrapped = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            broadcast_buffers=False,
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
        )
        teacher = load_teacher(config, device)
        metric = (
            LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True)
            .to(device).eval().requires_grad_(False)
        )
        optimizer = torch.optim.AdamW(
            trainable,
            lr=float(config["training"]["learning_rate"]),
            weight_decay=float(config["training"]["weight_decay"]),
            fused=True,
        )
        training = config["training"]
        losses = training["losses"]
        accumulation = int(training["gradient_accumulation_steps_by_world_size"][str(world)])
        crop = int(config["data"]["train_crop_size"])
        checkpoints = {int(value) for value in training["checkpoint_steps"]}
        latest_metrics = None
        optimizer.zero_grad(set_to_none=True)
        wrapped.train()
        for step in range(int(training["max_steps"])):
            accumulated = 0.0
            for micro in range(accumulation):
                position = (step * accumulation + micro) * world + rank
                index = sampled_row_index(position, group_names, grouped, seed)
                lq_cpu, gt_cpu = materialize(records[index], crop)
                if config["data"]["geometric_augmentation"]:
                    lq_cpu, gt_cpu = augment_pair(lq_cpu, gt_cpu, (seed + position) % 8)
                lq, gt = lq_cpu.unsqueeze(0).to(device), gt_cpu.unsqueeze(0).to(device)
                with torch.no_grad():
                    clean, degraded = autoencoder.encode(torch.cat([gt, lq])).chunk(2)
                    teacher_output_01 = teacher(((lq.float() + 1.0) * 0.5)).clamp(0, 1)
                path_tokens, path_ids = batched_prc_img(degraded.to(torch.bfloat16))
                reference_ids = path_ids.clone()
                reference_ids[..., 0] = int(training["reference_time_coordinate"])
                target_velocity = degraded - clean
                target_tokens, _ = batched_prc_img(target_velocity.to(torch.bfloat16))
                timestep = torch.ones((1,), device=device, dtype=torch.bfloat16)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    prediction = wrapped(
                        x=torch.cat([path_tokens, path_tokens], dim=1),
                        x_ids=torch.cat([path_ids, reference_ids], dim=1),
                        timesteps=timestep,
                        ctx=context,
                        ctx_ids=context_ids,
                        guidance=None,
                    )[:, : path_tokens.shape[1]]
                    predicted_clean = degraded - unpack_tokens(prediction, degraded.shape)
                    decoded = autoencoder.decode(predicted_clean).float().clamp(-1, 1)
                    flow_loss = F.mse_loss(prediction.float(), target_tokens.float())
                    latent_l1 = F.l1_loss(predicted_clean.float(), clean.float())
                    change = (gt.float() - lq.float()).abs().mean(dim=1, keepdim=True)
                    change_mask = (change / float(losses["rain_change_scale_m11"])).clamp(0, 1).detach()
                    change_weight = 1.0 + float(losses["rain_change_weight"]) * change_mask
                    rgb_mse = ((decoded - gt.float()).square() * change_weight).mean()
                    epsilon = float(losses["charbonnier_epsilon"])
                    charbonnier = (
                        torch.sqrt((decoded - gt.float()).square() + epsilon * epsilon) * change_weight
                    ).mean()
                    identity = (
                        torch.sqrt((decoded - lq.float()).square() + epsilon * epsilon)
                        * (1.0 - change_mask)
                    ).mean()
                    edge = F.l1_loss(spatial_highpass(decoded), spatial_highpass(gt.float()))
                    output_01 = (decoded + 1.0) * 0.5
                    perceptual = metric(output_01, (gt.float() + 1.0) * 0.5)
                    teacher_l1 = F.l1_loss(output_01, teacher_output_01)
                    loss = (
                        float(losses["flow_mse_weight"]) * flow_loss
                        + float(losses["predicted_clean_latent_l1_weight"]) * latent_l1
                        + float(losses["final_rgb_mse_weight"]) * rgb_mse
                        + float(losses["rgb_charbonnier_weight"]) * charbonnier
                        + float(losses["highpass_l1_weight"]) * edge
                        + float(losses["lpips_weight"]) * perceptual
                        + float(losses["teacher_l1_weight"]) * teacher_l1
                        + float(losses["unchanged_identity_weight"]) * identity
                    )
                (loss / accumulation).backward()
                accumulated += float(loss.detach())
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, float(training["max_grad_norm"]))
            if not math.isfinite(float(grad_norm)):
                raise RuntimeError("nonfinite direct-endpoint gradients")
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            value = torch.tensor([accumulated / accumulation], device=device)
            dist.all_reduce(value)
            completed = step + 1
            if rank == 0 and (completed == 1 or completed % 10 == 0):
                print(
                    f"FOUNDIR_RAIN_DIRECT_ENDPOINT_TRAIN step={completed}/{training['max_steps']} "
                    f"loss={float(value.item() / world):.8f} grad_norm={float(grad_norm):.6f}",
                    flush=True,
                )
            if completed in checkpoints:
                dist.barrier()
                if rank == 0:
                    checkpoint = output_dir / f"step_{completed:06d}.safetensors"
                    save_lora_checkpoint(
                        wrapped.module,
                        checkpoint,
                        {
                            "step": completed,
                            "config_sha256": sha256(config_path),
                            "objective": str(training["objective"]),
                            "parent_lora_sha256": str(config["initialization"]["parent_lora_sha256"]),
                            "teacher_checkpoint_sha256": str(config["teacher"]["adapted_checkpoint_sha256"]),
                        },
                    )
                    with checkpoint.with_suffix(".optimizer.pt").open("xb") as handle:
                        torch.save(
                            {
                                "step": completed,
                                "config_sha256": sha256(config_path),
                                "optimizer": optimizer.state_dict(),
                            },
                            handle,
                        )
                dist.barrier()
                latest_metrics = evaluate(
                    wrapped,
                    autoencoder,
                    validation,
                    context,
                    context_ids,
                    batched_prc_img,
                    metric,
                    config=config,
                    step=completed,
                    output_dir=output_dir,
                    rank=rank,
                    world=world,
                    device=device,
                )
        elapsed = time.monotonic() - started
        if rank == 0:
            assert latest_metrics is not None
            checks = {
                "validation_psnr": float(latest_metrics["mean_output_psnr"])
                >= float(config["gate"]["validation_psnr_minimum"]),
                "validation_lpips": float(latest_metrics["mean_output_lpips"])
                <= float(config["gate"]["validation_lpips_maximum"]),
            }
            payload = {
                "schema_version": 1,
                "status": "qualification_complete" if config["gate"]["qualification_only"] else "complete",
                "scientific_gate_pass": all(checks.values()),
                "qualification_is_engineering_only": bool(config["gate"]["qualification_only"]),
                "checks": checks,
                "validation_metrics": latest_metrics,
                "steps": int(training["max_steps"]),
                "wall_clock_seconds": elapsed,
                "wall_clock_hours": elapsed / 3600.0,
                "allocated_gpu_hours": elapsed * world / 3600.0,
                "gpu_count": world,
                "gpu_name": properties.name,
                "config_sha256": sha256(config_path),
                "official_foundir_test_accessed": False,
                "sealed_internal_accessed": False,
                "automatic_formal_launch": False,
            }
            write_once(output_dir / "result.json", (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode())
            print("FOUNDIR_RAIN_DIRECT_ENDPOINT_END " + json.dumps(payload, sort_keys=True), flush=True)
        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
