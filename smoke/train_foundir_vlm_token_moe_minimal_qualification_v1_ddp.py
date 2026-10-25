#!/usr/bin/env python3
"""Qualify matched image-only and VLM-assisted FLUX soft-MoE router arms."""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from safetensors.torch import load_file, save_file
from torch.nn.parallel import DistributedDataParallel


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party/flux2/src"))

from lowlight_restoration.flow import build_degraded_anchored_flow_batch  # noqa: E402
from lowlight_restoration.flux_prompt_cache import PromptContextCache, file_sha256  # noqa: E402
from lowlight_restoration.vlm_guided_moe import (  # noqa: E402
    ConditionedMoEMLP,
    conditioned_moe_modules,
    inject_vlm_guided_moe_into_flux_double_blocks,
    vlm_moe_conditioning,
)
from smoke.train_foundir_five_type_flux2_full_ddp import (  # noqa: E402
    flow_random_inputs,
    prepare_aligned_pair,
)
from smoke.train_foundir_five_type_flux2_overfit_ddp import unpack_tokens  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(path: Path) -> dict[str, object]:
    config = json.loads(path.read_text())
    if config.get("status") != "advisor_approved_minimal_vlm_token_moe_qualification":
        raise RuntimeError("minimal VLM-MoE authorization changed")
    data = config["data"]
    moe = config["moe"]
    training = config["training"]
    distributed = config["distributed"]
    checks = (
        config["scope"] == {
            "degradations": ["blur", "noise", "rain", "haze", "low_light"],
            "composition_sizes": [1],
            "official_foundir_test_accessed": False,
            "sealed_internal_accessed": False,
            "parent_test_accessed": False,
            "internal_test_accessed": False,
        },
        config["arms"] == ["image_only", "vlm_assisted"],
        int(data["fit_records"]) == 2250,
        int(data["calibration_records"]) == 450,
        int(data["holdout_records"]) == 450,
        int(data["crop_size"]) == 256,
        int(data["vlm_feature_dim"]) == 7168,
        moe["double_block_indices"] == [4],
        int(moe["num_experts"]) == 2,
        moe["routing"] == "dense_softmax_per_image",
        moe["top_k"] is None,
        int(moe["router_hidden_dim"]) == 256,
        moe["one_expert_bank_only"] is True,
        moe["lora_experts"] is False,
        training["qualification_only"] is True,
        int(training["optimizer_steps"]) == 25,
        int(training["global_batch_size"]) == 32,
        int(training["micro_batch_size_per_rank"]) == 1,
        int(training["gradient_accumulation_steps"]) == 4,
        training["qualification_checkpoint_may_seed_formal"] is False,
        int(distributed["world_size"]) == 8,
        config["gates"]["automatic_formal_launch"] is False,
    )
    if not all(checks):
        raise RuntimeError("minimal VLM-MoE frozen qualification contract changed")
    if int(training["global_batch_size"]) != (
        int(distributed["world_size"])
        * int(training["micro_batch_size_per_rank"])
        * int(training["gradient_accumulation_steps"])
    ):
        raise RuntimeError("global batch arithmetic changed")
    return config


def verify_artifacts(config: Mapping[str, object], *, verify_large_weights: bool) -> None:
    data = config["data"]
    model = config["model"]
    artifacts = [
        (data["manifest"], data["manifest_sha256"], True),
        (data["metadata"], data["metadata_sha256"], True),
        (data["feature_source_config"], data["feature_source_config_sha256"], True),
        (model["flow_weights"], model["flow_weights_sha256"], verify_large_weights),
        (model["autoencoder_weights"], model["autoencoder_weights_sha256"], verify_large_weights),
        (model["prompt_cache_index"], model["prompt_cache_index_sha256"], True),
    ]
    for relative, expected, verify_hash in artifacts:
        path = ROOT / str(relative)
        if not path.is_file():
            raise RuntimeError(f"missing frozen artifact: {path}")
        if verify_hash and sha256(path) != str(expected):
            raise RuntimeError(f"frozen artifact identity changed: {path}")


def load_rows(config: Mapping[str, object]) -> list[dict[str, str]]:
    data = config["data"]
    with (ROOT / str(data["manifest"])).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 3150 or len({row["record_id"] for row in rows}) != 3150:
        raise RuntimeError("pilot manifest cardinality changed")
    counts = {role: sum(row["role"] == role for row in rows) for role in ("fit", "calibration", "holdout")}
    if counts != {"fit": 2250, "calibration": 450, "holdout": 450}:
        raise RuntimeError(f"pilot split counts changed: {counts}")
    split_groups = {
        role: {row["source_group"] for row in rows if row["role"] == role}
        for role in counts
    }
    if any(
        split_groups[left] & split_groups[right]
        for left in split_groups
        for right in split_groups
        if left < right
    ):
        raise RuntimeError("pilot source groups leak across roles")
    return rows


def deterministic_fit_index(position: int, length: int, seed: int) -> int:
    epoch, offset = divmod(position, length)
    generator = torch.Generator(device="cpu").manual_seed(seed + epoch)
    return int(torch.randperm(length, generator=generator)[offset])


def load_training_item(
    row: Mapping[str, str],
    *,
    crop_size: int,
    presentation_index: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    with Image.open(ROOT / row["lq_path"]) as handle:
        lq = np.asarray(handle.convert("RGB"), dtype=np.uint8).copy()
    with Image.open(ROOT / row["gt_path"]) as handle:
        gt = np.asarray(handle.convert("RGB"), dtype=np.uint8).copy()
    lq, gt = prepare_aligned_pair(lq, gt, crop_size, row["record_id"])
    digest = hashlib.sha256(
        f"{seed}|{row['record_id']}|{presentation_index}|crop".encode()
    ).digest()
    height, width = gt.shape[:2]
    y = int.from_bytes(digest[:8], "big") % (height - crop_size + 1)
    x = int.from_bytes(digest[8:16], "big") % (width - crop_size + 1)
    lq = lq[y : y + crop_size, x : x + crop_size]
    gt = gt[y : y + crop_size, x : x + crop_size]

    def image_tensor(array: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(array.copy()).permute(2, 0, 1).float().div_(127.5).sub_(1.0)

    feature_payload = load_file(str(ROOT / row["vlm_feature_path"]), device="cpu")
    if set(feature_payload) != {"feature"} or tuple(feature_payload["feature"].shape) != (7168,):
        raise RuntimeError(f"frozen VLM feature schema changed: {row['record_id']}")
    return image_tensor(lq), image_tensor(gt), feature_payload["feature"].float()


def moe_parameter_groups(module: ConditionedMoEMLP, arm: str, config: Mapping[str, object]):
    training = config["training"]
    module.requires_grad_(True)
    if arm == "image_only":
        module.moe.vlm_projector.requires_grad_(False)
    groups = [
        {
            "params": list(module.moe.experts.parameters()),
            "lr": float(training["expert_learning_rate"]),
            "name": "experts",
        },
        {
            "params": list(module.moe.image_router.parameters()),
            "lr": float(training["router_learning_rate"]),
            "name": "image_router",
        },
    ]
    if arm == "vlm_assisted":
        groups.append(
            {
                "params": list(module.moe.vlm_projector.parameters()),
                "lr": float(training["router_learning_rate"]),
                "name": "vlm_projector",
            }
        )
    return groups


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--arm", choices=("image_only", "vlm_assisted"), required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    config = load_config(config_path)
    verify_artifacts(config, verify_large_weights=not args.preflight_only)
    rows = load_rows(config)
    fit_rows = sorted((row for row in rows if row["role"] == "fit"), key=lambda row: row["record_id"])
    prompts = {row["generic_text"] for row in rows}
    if len(prompts) != 1:
        raise RuntimeError("qualification requires one identical generic prompt in both arms")
    prompt = next(iter(prompts))

    if args.preflight_only:
        for row in (fit_rows[0], fit_rows[len(fit_rows) // 2], fit_rows[-1]):
            load_training_item(
                row,
                crop_size=int(config["data"]["crop_size"]),
                presentation_index=0,
                seed=int(config["training"]["seed"]),
            )
        cache = PromptContextCache(ROOT / str(config["model"]["prompt_cache_index"]), maximum_resident=1)
        context, context_ids = cache.get(prompt, torch.device("cpu"))
        payload = {
            "status": "minimal_vlm_token_moe_cpu_preflight_pass",
            "arm": args.arm,
            "records": len(rows),
            "fit_records": len(fit_rows),
            "generic_prompts": 1,
            "context_shape": list(context.shape),
            "context_ids_shape": list(context_ids.shape),
            "protected_data_accessed": False,
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend=str(config["distributed"]["backend"]), device_id=device)
    started = time.monotonic()
    try:
        rank = dist.get_rank()
        world = dist.get_world_size()
        if world != 8 or torch.cuda.device_count() != 8:
            raise RuntimeError("minimal VLM-MoE qualification requires exactly eight visible GPUs")
        properties = torch.cuda.get_device_properties(device)
        names: list[object | None] = [None] * world
        dist.all_gather_object(names, properties.name)
        if len(set(names)) != 1 or properties.name not in config["distributed"]["supported_gpu_names"]:
            raise RuntimeError(f"qualification requires homogeneous H100/H200 GPUs: {names}")
        if properties.total_memory < int(config["distributed"]["minimum_memory_gib"]) * 1024**3:
            raise RuntimeError("GPU memory is below the frozen qualification minimum")

        seed = int(config["training"]["seed"])
        random.seed(seed + rank)
        np.random.seed(seed + rank)
        torch.manual_seed(seed + rank)
        torch.cuda.manual_seed_all(seed + rank)
        os.environ.update(
            HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
            KLEIN_4B_BASE_MODEL_PATH=str(ROOT / config["model"]["flow_weights"]),
            AE_MODEL_PATH=str(ROOT / config["model"]["autoencoder_weights"]),
        )
        from flux2.sampling import batched_prc_img
        from flux2.util import load_ae, load_flow_model

        prompt_cache = PromptContextCache(
            ROOT / str(config["model"]["prompt_cache_index"]), maximum_resident=1
        )
        context, context_ids = prompt_cache.get(prompt, device)
        autoencoder = load_ae(config["model"]["name"], device=device).eval().requires_grad_(False)
        model = load_flow_model(config["model"]["name"], device=device).eval().requires_grad_(False)
        injected = inject_vlm_guided_moe_into_flux_double_blocks(
            model,
            block_indices=[int(value) for value in config["moe"]["double_block_indices"]],
            vlm_dim=int(config["data"]["vlm_feature_dim"]),
            num_experts=int(config["moe"]["num_experts"]),
            router_hidden_dim=int(config["moe"]["router_hidden_dim"]),
            top_k=config["moe"]["top_k"],
            vlm_scale=float(config["moe"]["vlm_scale"]),
            trainable=True,
        )
        if list(injected) != [str(config["moe"]["insertion"])]:
            raise RuntimeError(f"MoE insertion identity changed: {list(injected)}")
        moe_module = next(iter(injected.values()))
        parameter_groups = moe_parameter_groups(moe_module, args.arm, config)
        trainable = [parameter for group in parameter_groups for parameter in group["params"]]
        trainable_count = sum(parameter.numel() for parameter in trainable)
        wrapped = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            broadcast_buffers=False,
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
        )
        optimizer = torch.optim.AdamW(
            parameter_groups,
            weight_decay=float(config["training"]["weight_decay"]),
            fused=True,
        )
        initial_vlm_output = {
            key: value.detach().cpu().clone()
            for key, value in moe_module.moe.vlm_projector[-1].state_dict().items()
        }

        use_vlm = args.arm == "vlm_assisted"
        accumulation = int(config["training"]["gradient_accumulation_steps"])
        global_batch = int(config["training"]["global_batch_size"])
        micro_batch = int(config["training"]["micro_batch_size_per_rank"])
        if micro_batch != 1:
            raise RuntimeError("qualification implementation currently freezes microbatch at one")
        steps = int(config["training"]["optimizer_steps"])
        crop_size = int(config["data"]["crop_size"])
        recent_losses: list[float] = []
        route_weight_sums: list[float] = []
        route_entropies: list[float] = []
        maximum_vlm_adjustment = 0.0
        torch.cuda.reset_peak_memory_stats(device)
        for step in range(steps):
            optimizer.zero_grad(set_to_none=True)
            accumulated = 0.0
            for micro in range(accumulation):
                position = step * global_batch + micro * world + rank
                row = fit_rows[deterministic_fit_index(position, len(fit_rows), seed)]
                lq, gt, vlm_feature = load_training_item(
                    row,
                    crop_size=crop_size,
                    presentation_index=position,
                    seed=seed,
                )
                lq = lq.unsqueeze(0).to(device, non_blocking=True)
                gt = gt.unsqueeze(0).to(device, non_blocking=True)
                vlm_feature = vlm_feature.unsqueeze(0).to(device, dtype=torch.bfloat16, non_blocking=True)
                with torch.no_grad():
                    clean_latent, degraded_latent = autoencoder.encode(torch.cat([gt, lq])).chunk(2)
                noise, timesteps = flow_random_inputs(
                    clean_latent.shape,
                    [position],
                    seed=seed,
                    device=device,
                )
                flow_batch = build_degraded_anchored_flow_batch(
                    clean_latent.to(torch.bfloat16),
                    degraded_latent.to(torch.bfloat16),
                    noise,
                    timesteps,
                    batched_prc_img,
                    anchor_noise_scale=float(config["training"]["anchor_noise_scale"]),
                    reference_time_coordinate=int(config["training"]["reference_time_coordinate"]),
                )
                sync = contextlib.nullcontext() if micro + 1 == accumulation else wrapped.no_sync()
                with sync, vlm_moe_conditioning(wrapped.module, vlm_feature, use_vlm=use_vlm):
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        prediction = wrapped(
                            x=flow_batch.model_tokens,
                            x_ids=flow_batch.model_ids,
                            timesteps=flow_batch.timesteps,
                            ctx=context,
                            ctx_ids=context_ids,
                            guidance=None,
                        )[:, : flow_batch.target_token_count]
                        flow_loss = torch.nn.functional.mse_loss(
                            prediction.float(), flow_batch.target_velocity.float()
                        )
                        velocity_latent = unpack_tokens(prediction, clean_latent.shape)
                        source_latent = degraded_latent + float(config["training"]["anchor_noise_scale"]) * noise
                        predicted_clean = source_latent - velocity_latent
                        latent_loss = torch.nn.functional.l1_loss(
                            predicted_clean.float(), clean_latent.float()
                        )
                        loss = (
                            float(config["training"]["flow_mse_weight"]) * flow_loss
                            + float(config["training"]["predicted_clean_latent_l1_weight"]) * latent_loss
                        )
                    routing = moe_module.last_routing
                    if routing is None:
                        raise RuntimeError("MoE did not expose routing for the active forward")
                    weights = routing.weights.detach().float()
                    adjustment = routing.vlm_logit_adjustment.detach().float()
                    route_weight_sums.append(float(weights.sum(-1).mean()))
                    route_entropies.append(
                        float(-(weights.clamp_min(1e-12).log() * weights).sum(-1).mean())
                    )
                    maximum_vlm_adjustment = max(maximum_vlm_adjustment, float(adjustment.abs().max()))
                    (loss / accumulation).backward()
                accumulated += float(loss.detach())
            gradient_norm = float(torch.nn.utils.clip_grad_norm_(trainable, float(config["training"]["max_grad_norm"])))
            if not math.isfinite(accumulated) or not math.isfinite(gradient_norm):
                raise RuntimeError("qualification produced non-finite loss or gradients")
            optimizer.step()
            value = torch.tensor([accumulated / accumulation], device=device)
            dist.all_reduce(value)
            mean_loss = float(value.item() / world)
            recent_losses.append(mean_loss)
            if rank == 0:
                print(
                    f"VLM_TOKEN_MOE_QUAL arm={args.arm} step={step + 1}/{steps} "
                    f"loss={mean_loss:.8f} grad_norm={gradient_norm:.8f}",
                    flush=True,
                )

        elapsed = time.monotonic() - started
        final_vlm_output = moe_module.moe.vlm_projector[-1].state_dict()
        vlm_output_update = sum(
            float((final_vlm_output[key].detach().cpu() - initial_vlm_output[key]).abs().sum())
            for key in initial_vlm_output
        )
        if args.arm == "image_only" and maximum_vlm_adjustment != 0.0:
            raise RuntimeError("image-only arm used a nonzero VLM adjustment")
        if args.arm == "vlm_assisted" and vlm_output_update <= 0.0:
            raise RuntimeError("VLM-assisted arm did not update its small projector")
        tolerance = float(config["gates"]["expert_weight_sum_tolerance"])
        if max(abs(value - 1.0) for value in route_weight_sums) > tolerance:
            raise RuntimeError("expert weights no longer sum to one")

        dist.barrier()
        if rank == 0:
            output = ROOT / str(config["runtime"]["output_root"]) / args.arm
            output.mkdir(parents=True, exist_ok=False)
            state = {
                f"{name}.{key}": value.detach().cpu().contiguous()
                for name, module in conditioned_moe_modules(wrapped.module).items()
                for key, value in module.state_dict().items()
            }
            checkpoint = output / "step_000025.safetensors"
            save_file(state, str(checkpoint))
            wall_hours = elapsed / 3600.0
            result = {
                "schema_version": 1,
                "status": "minimal_vlm_token_moe_engineering_qualification_pass",
                "arm": args.arm,
                "config": config_path.relative_to(ROOT).as_posix(),
                "config_sha256": file_sha256(config_path),
                "steps": steps,
                "world_size": world,
                "gpu_name": properties.name,
                "global_batch_size": global_batch,
                "trainable_parameters": trainable_count,
                "mean_loss_last_five": sum(recent_losses[-5:]) / len(recent_losses[-5:]),
                "maximum_vlm_logit_adjustment": maximum_vlm_adjustment,
                "vlm_output_head_l1_update": vlm_output_update,
                "mean_route_entropy": sum(route_entropies) / len(route_entropies),
                "maximum_expert_weight_sum_error": max(abs(value - 1.0) for value in route_weight_sums),
                "peak_allocated_bytes_per_rank": int(torch.cuda.max_memory_allocated(device)),
                "wall_clock_hours": wall_hours,
                "allocated_gpu_hours": wall_hours * world,
                "checkpoint": checkpoint.relative_to(ROOT).as_posix(),
                "checkpoint_sha256": file_sha256(checkpoint),
                "qualification_checkpoint_may_seed_formal": False,
                "official_foundir_test_accessed": False,
                "sealed_internal_accessed": False,
                "parent_test_accessed": False,
                "internal_test_accessed": False,
                "scientific_result": False,
            }
            (output / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
            print("VLM_TOKEN_MOE_QUALIFICATION_PASS " + json.dumps(result, sort_keys=True), flush=True)
        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
