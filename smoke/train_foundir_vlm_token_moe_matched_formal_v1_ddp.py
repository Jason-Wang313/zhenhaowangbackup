#!/usr/bin/env python3
"""Train and evaluate the matched image-only/VLM-assisted FLUX MoE arms."""

from __future__ import annotations

import argparse
import contextlib
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
from smoke.train_foundir_five_type_flux2_full_ddp import flow_random_inputs  # noqa: E402
from smoke.train_foundir_five_type_flux2_overfit_ddp import psnr, restore, unpack_tokens  # noqa: E402
from smoke.train_foundir_vlm_token_moe_minimal_qualification_v1_ddp import (  # noqa: E402
    deterministic_fit_index,
    load_rows,
    load_training_item,
    moe_parameter_groups,
    sha256,
    verify_artifacts,
)


ARMS = ("image_only", "vlm_assisted")
LABELS = ("blur", "noise", "rain", "haze", "low_light")


def quantized_rgb(tensor: torch.Tensor) -> np.ndarray:
    image = ((tensor[0].detach().float().cpu() + 1.0) * 127.5).permute(1, 2, 0).numpy()
    return np.rint(image.clip(0, 255)).astype(np.uint8)


def lpips_value(
    metric: torch.nn.Module,
    reference: torch.Tensor,
    prediction: torch.Tensor,
    device: torch.device,
) -> float:
    reference_01 = (reference.float() + 1.0) * 0.5
    prediction_01 = (prediction.float() + 1.0) * 0.5
    with torch.inference_mode():
        return float(metric(prediction_01.to(device), reference_01.to(device)).item())


def write_once(path: Path, payload: Mapping[str, object]) -> None:
    content = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise RuntimeError(f"refusing to overwrite differing artifact: {path}")
        return
    with path.open("xb") as handle:
        handle.write(content)


def load_formal_config(path: Path) -> dict[str, object]:
    config = json.loads(path.read_text())
    data = config["data"]
    moe = config["moe"]
    training = config["training"]
    evaluation = config["evaluation"]
    distributed = config["distributed"]
    checks = (
        config.get("status") == "advisor_approved_minimal_vlm_token_moe_matched_formal",
        config["scope"] == {
            "degradations": list(LABELS),
            "composition_sizes": [1],
            "official_foundir_test_accessed": False,
            "sealed_internal_accessed": False,
            "parent_test_accessed": False,
            "internal_test_accessed": False,
        },
        config["arms"] == list(ARMS),
        int(data["fit_records"]) == 2250,
        int(data["calibration_records"]) == 450,
        int(data["holdout_records"]) == 450,
        int(data["train_crop_size"]) == 256,
        int(data["evaluation_crop_size"]) == 256,
        int(data["vlm_feature_dim"]) == 7168,
        moe["double_block_indices"] == [4],
        int(moe["num_experts"]) == 2,
        moe["routing"] == "dense_softmax_per_image",
        moe["top_k"] is None,
        int(moe["router_hidden_dim"]) == 256,
        moe["one_expert_bank_only"] is True,
        moe["lora_experts"] is False,
        training["formal_only"] is True,
        int(training["optimizer_steps"]) == 1000,
        training["checkpoint_steps"] == [500, 1000],
        int(training["global_batch_size"]) == 32,
        int(training["micro_batch_size_per_rank"]) == 1,
        int(training["gradient_accumulation_steps"]) == 4,
        training["fresh_pretrained_initialization"] is True,
        training["qualification_checkpoint_forbidden"] is True,
        evaluation["calibration_steps"] == [500, 1000],
        evaluation["holdout_steps"] == [1000],
        int(evaluation["inference_steps"]) == 6,
        evaluation["checkpoint_selection"] == "fixed_final_step_no_holdout_selection",
        int(distributed["world_size"]) == 8,
        config["gates"]["automatic_next_experiment_launch"] is False,
    )
    if not all(checks):
        raise RuntimeError("matched formal VLM-MoE contract changed")
    if int(training["global_batch_size"]) != (
        int(distributed["world_size"])
        * int(training["micro_batch_size_per_rank"])
        * int(training["gradient_accumulation_steps"])
    ):
        raise RuntimeError("formal global-batch arithmetic changed")
    if set(evaluation["metrics"]) != {"psnr", "lpips_alex"}:
        raise RuntimeError("formal metric contract changed")
    return config


def aggregate_rows(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    if not rows:
        raise ValueError("cannot aggregate zero evaluation rows")

    def metrics(values: Sequence[Mapping[str, object]]) -> dict[str, float | int]:
        return {
            "records": len(values),
            "input_psnr": float(np.mean([float(row["input_psnr"]) for row in values])),
            "output_psnr": float(np.mean([float(row["output_psnr"]) for row in values])),
            "input_lpips": float(np.mean([float(row["input_lpips"]) for row in values])),
            "output_lpips": float(np.mean([float(row["output_lpips"]) for row in values])),
            "mean_route_entropy": float(np.mean([float(row["route_entropy"]) for row in values])),
            "mean_expert_0_weight": float(np.mean([float(row["expert_weights"][0]) for row in values])),
            "mean_expert_1_weight": float(np.mean([float(row["expert_weights"][1]) for row in values])),
            "mean_vlm_adjustment_l1": float(
                np.mean([float(row["vlm_adjustment_l1"]) for row in values])
            ),
        }

    result = metrics(rows)
    result["psnr_gain_over_input"] = float(result["output_psnr"]) - float(result["input_psnr"])
    result["lpips_improvement_over_input"] = float(result["input_lpips"]) - float(result["output_lpips"])
    result["by_degradation"] = {
        label: metrics([row for row in rows if row["label"] == label]) for label in LABELS
    }
    return result


def deterministic_evaluation_seed(base_seed: int, record_id: str) -> int:
    digest = hashlib.sha256(f"{base_seed}|{record_id}|formal_eval".encode()).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def save_moe_checkpoint(model: torch.nn.Module, path: Path) -> None:
    state = {
        f"{name}.{key}": value.detach().cpu().contiguous()
        for name, module in conditioned_moe_modules(model).items()
        for key, value in module.state_dict().items()
    }
    if not state:
        raise RuntimeError("no conditioned MoE state was available to save")
    save_file(state, str(path))


def load_moe_checkpoint(model: torch.nn.Module, path: Path) -> None:
    payload = load_file(str(path), device="cpu")
    modules = conditioned_moe_modules(model)
    for name, module in modules.items():
        prefix = f"{name}."
        state = {key[len(prefix) :]: value for key, value in payload.items() if key.startswith(prefix)}
        if not state:
            raise RuntimeError(f"checkpoint has no state for {name}")
        module.load_state_dict(state, strict=True)
    consumed = {
        f"{name}.{key}" for name, module in modules.items() for key in module.state_dict()
    }
    if consumed != set(payload):
        raise RuntimeError("formal checkpoint contains unexpected MoE keys")


def discover_resume(output: Path, config_sha: str, arm: str) -> tuple[int, Path | None, Path | None]:
    candidates: list[tuple[int, Path, Path]] = []
    for checkpoint in output.glob("step_*.safetensors") if output.exists() else ():
        try:
            step = int(checkpoint.stem.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        optimizer = checkpoint.with_suffix(".optimizer.pt")
        metadata = checkpoint.with_suffix(".json")
        if not optimizer.is_file() or not metadata.is_file():
            raise RuntimeError(f"partial formal checkpoint: {checkpoint}")
        identity = json.loads(metadata.read_text())
        if (
            identity.get("config_sha256") != config_sha
            or identity.get("arm") != arm
            or int(identity.get("step", -1)) != step
            or identity.get("checkpoint_sha256") != file_sha256(checkpoint)
        ):
            raise RuntimeError(f"formal checkpoint identity changed: {checkpoint}")
        candidates.append((step, checkpoint, optimizer))
    if not candidates:
        return 0, None, None
    return max(candidates, key=lambda item: item[0])


def evaluate_split(
    model: torch.nn.Module,
    autoencoder: torch.nn.Module,
    rows: Sequence[Mapping[str, str]],
    context: torch.Tensor,
    context_ids: torch.Tensor,
    pack_image,
    metric: torch.nn.Module,
    moe_module: ConditionedMoEMLP,
    *,
    config: Mapping[str, object],
    arm: str,
    split: str,
    step: int,
    rank: int,
    world: int,
    device: torch.device,
) -> dict[str, object] | None:
    model.eval()
    evaluation = config["evaluation"]
    local: list[dict[str, object]] = []
    for index in range(rank, len(rows), world):
        row = rows[index]
        lq_cpu, gt_cpu, vlm_feature_cpu = load_training_item(
            row,
            crop_size=int(config["data"]["evaluation_crop_size"]),
            presentation_index=0,
            seed=int(evaluation["deterministic_seed"]),
        )
        lq = lq_cpu.unsqueeze(0).to(device)
        gt = gt_cpu.unsqueeze(0).to(device)
        vlm_feature = vlm_feature_cpu.unsqueeze(0).to(device, dtype=torch.bfloat16)
        with vlm_moe_conditioning(model, vlm_feature, use_vlm=arm == "vlm_assisted"):
            prediction = restore(
                model,
                autoencoder,
                lq,
                context,
                context_ids,
                pack_image,
                seed=deterministic_evaluation_seed(
                    int(evaluation["deterministic_seed"]), row["record_id"]
                ),
                steps=int(evaluation["inference_steps"]),
                noise_scale=float(config["training"]["anchor_noise_scale"]),
                reference_time=int(config["training"]["reference_time_coordinate"]),
            )
            routing = moe_module.last_routing
            if routing is None:
                raise RuntimeError("formal evaluation did not expose routing")
            weights = routing.weights[0].detach().float().cpu()
            adjustment = routing.vlm_logit_adjustment[0].detach().float().cpu()
        output_array = quantized_rgb(prediction)
        prediction_q = (
            torch.from_numpy(output_array.copy())
            .permute(2, 0, 1)
            .float()
            .div_(127.5)
            .sub_(1.0)
            .unsqueeze(0)
            .to(device)
        )
        route_entropy = float(-(weights.clamp_min(1e-12).log() * weights).sum())
        result_row = {
            "record_id": row["record_id"],
            "source_group": row["source_group"],
            "label": row["label"],
            "input_psnr": psnr(gt, lq),
            "output_psnr": psnr(gt, prediction_q),
            "input_lpips": lpips_value(metric, gt, lq, device),
            "output_lpips": lpips_value(metric, gt, prediction_q, device),
            "expert_weights": [float(value) for value in weights.tolist()],
            "route_entropy": route_entropy,
            "vlm_adjustment_l1": float(adjustment.abs().mean()),
        }
        if any(
            isinstance(value, float) and not math.isfinite(value)
            for value in result_row.values()
        ):
            raise RuntimeError("non-finite formal evaluation value")
        local.append(result_row)

    gathered: list[object | None] = [None] * world
    dist.all_gather_object(gathered, local)
    dist.barrier()
    model.train()
    if rank != 0:
        return None
    combined = sorted(
        [row for shard in gathered if isinstance(shard, list) for row in shard],
        key=lambda item: str(item["record_id"]),
    )
    if len(combined) != len(rows):
        raise RuntimeError("formal evaluation gather lost records")
    payload = {
        "schema_version": 1,
        "status": "matched_formal_evaluation_complete",
        "arm": arm,
        "split": split,
        "step": step,
        "metrics": aggregate_rows(combined),
        "rows": combined,
        "official_foundir_test_accessed": False,
        "sealed_internal_accessed": False,
        "parent_test_accessed": False,
        "internal_test_accessed": False,
    }
    print(
        "VLM_TOKEN_MOE_FORMAL_EVAL "
        + json.dumps({key: value for key, value in payload.items() if key != "rows"}, sort_keys=True),
        flush=True,
    )
    return payload


def preflight(config_path: Path, arm: str) -> None:
    config = load_formal_config(config_path)
    verify_artifacts(config, verify_large_weights=False)
    rows = load_rows(config)
    by_role = {
        role: sorted((row for row in rows if row["role"] == role), key=lambda item: item["record_id"])
        for role in ("fit", "calibration", "holdout")
    }
    for role, role_rows in by_role.items():
        for row in (role_rows[0], role_rows[len(role_rows) // 2], role_rows[-1]):
            load_training_item(
                row,
                crop_size=int(
                    config["data"]["train_crop_size" if role == "fit" else "evaluation_crop_size"]
                ),
                presentation_index=0,
                seed=int(config["training"]["seed"]),
            )
    prompts = {row["generic_text"] for row in rows}
    if len(prompts) != 1:
        raise RuntimeError("formal comparison requires one identical generic prompt")
    context, context_ids = PromptContextCache(
        ROOT / str(config["model"]["prompt_cache_index"]), maximum_resident=1
    ).get(next(iter(prompts)), torch.device("cpu"))
    print(
        json.dumps(
            {
                "status": "matched_formal_cpu_preflight_pass",
                "arm": arm,
                "records_by_role": {role: len(values) for role, values in by_role.items()},
                "context_shape": list(context.shape),
                "context_ids_shape": list(context_ids.shape),
                "optimizer_steps": int(config["training"]["optimizer_steps"]),
                "protected_data_accessed": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


def train(config_path: Path, arm: str) -> int:
    config = load_formal_config(config_path)
    verify_artifacts(config, verify_large_weights=True)
    rows = load_rows(config)
    fit_rows = sorted((row for row in rows if row["role"] == "fit"), key=lambda item: item["record_id"])
    calibration_rows = sorted(
        (row for row in rows if row["role"] == "calibration"), key=lambda item: item["record_id"]
    )
    holdout_rows = sorted(
        (row for row in rows if row["role"] == "holdout"), key=lambda item: item["record_id"]
    )
    prompts = {row["generic_text"] for row in rows}
    if len(prompts) != 1:
        raise RuntimeError("formal comparison requires one identical generic prompt")
    prompt = next(iter(prompts))

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend=str(config["distributed"]["backend"]), device_id=device)
    started = time.monotonic()
    try:
        rank = dist.get_rank()
        world = dist.get_world_size()
        if world != 8 or torch.cuda.device_count() != 8:
            raise RuntimeError("matched formal VLM-MoE requires exactly eight visible GPUs")
        properties = torch.cuda.get_device_properties(device)
        names: list[object | None] = [None] * world
        dist.all_gather_object(names, properties.name)
        if len(set(names)) != 1 or properties.name not in config["distributed"]["supported_gpu_names"]:
            raise RuntimeError(f"matched formal requires homogeneous H100/H200 GPUs: {names}")
        if properties.total_memory < int(config["distributed"]["minimum_memory_gib"]) * 1024**3:
            raise RuntimeError("GPU memory is below the frozen formal minimum")

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
        )
        from flux2.sampling import batched_prc_img
        from flux2.util import load_ae, load_flow_model
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

        context, context_ids = PromptContextCache(
            ROOT / str(config["model"]["prompt_cache_index"]), maximum_resident=1
        ).get(prompt, device)
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
            raise RuntimeError("formal MoE insertion identity changed")
        moe_module = next(iter(injected.values()))
        parameter_groups = moe_parameter_groups(moe_module, arm, config)
        trainable = [parameter for group in parameter_groups for parameter in group["params"]]
        optimizer = torch.optim.AdamW(
            parameter_groups,
            weight_decay=float(config["training"]["weight_decay"]),
            fused=True,
        )
        config_sha = sha256(config_path)
        output = ROOT / str(config["runtime"]["output_root"]) / arm
        start_step, checkpoint, optimizer_path = discover_resume(output, config_sha, arm)
        if (checkpoint is None) != (optimizer_path is None):
            raise RuntimeError("formal resume artifact set is incomplete")
        if checkpoint is not None:
            load_moe_checkpoint(model, checkpoint)
            optimizer_payload = torch.load(optimizer_path, map_location="cpu", weights_only=False)
            if (
                optimizer_payload.get("config_sha256") != config_sha
                or optimizer_payload.get("arm") != arm
                or int(optimizer_payload.get("step", -1)) != start_step
            ):
                raise RuntimeError("formal optimizer resume identity changed")
            optimizer.load_state_dict(optimizer_payload["optimizer"])
        elif output.exists():
            if any(output.iterdir()):
                raise RuntimeError("nonempty formal output exists without a resumable checkpoint")
        else:
            if rank == 0:
                output.mkdir(parents=True, exist_ok=False)
        dist.barrier()
        if start_step >= int(config["training"]["optimizer_steps"]):
            raise RuntimeError("formal endpoint already exists; refusing duplicate execution")

        wrapped = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            broadcast_buffers=False,
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
        )
        metric = (
            LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True)
            .to(device)
            .eval()
            .requires_grad_(False)
        )
        use_vlm = arm == "vlm_assisted"
        training = config["training"]
        accumulation = int(training["gradient_accumulation_steps"])
        global_batch = int(training["global_batch_size"])
        steps = int(training["optimizer_steps"])
        checkpoints = {int(value) for value in training["checkpoint_steps"]}
        recent_losses: list[float] = []
        maximum_vlm_adjustment = 0.0
        route_entropies: list[float] = []
        route_weight_sums: list[float] = []
        torch.cuda.reset_peak_memory_stats(device)
        wrapped.train()
        for step in range(start_step, steps):
            optimizer.zero_grad(set_to_none=True)
            accumulated = 0.0
            for micro in range(accumulation):
                position = step * global_batch + micro * world + rank
                row = fit_rows[deterministic_fit_index(position, len(fit_rows), seed)]
                lq_cpu, gt_cpu, vlm_feature_cpu = load_training_item(
                    row,
                    crop_size=int(config["data"]["train_crop_size"]),
                    presentation_index=position,
                    seed=seed,
                )
                lq = lq_cpu.unsqueeze(0).to(device, non_blocking=True)
                gt = gt_cpu.unsqueeze(0).to(device, non_blocking=True)
                vlm_feature = vlm_feature_cpu.unsqueeze(0).to(
                    device, dtype=torch.bfloat16, non_blocking=True
                )
                with torch.no_grad():
                    clean_latent, degraded_latent = autoencoder.encode(torch.cat([gt, lq])).chunk(2)
                noise, timesteps = flow_random_inputs(
                    clean_latent.shape, [position], seed=seed, device=device
                )
                flow_batch = build_degraded_anchored_flow_batch(
                    clean_latent.to(torch.bfloat16),
                    degraded_latent.to(torch.bfloat16),
                    noise,
                    timesteps,
                    batched_prc_img,
                    anchor_noise_scale=float(training["anchor_noise_scale"]),
                    reference_time_coordinate=int(training["reference_time_coordinate"]),
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
                        source_latent = degraded_latent + float(training["anchor_noise_scale"]) * noise
                        predicted_clean = source_latent - velocity_latent
                        latent_loss = torch.nn.functional.l1_loss(
                            predicted_clean.float(), clean_latent.float()
                        )
                        loss = (
                            float(training["flow_mse_weight"]) * flow_loss
                            + float(training["predicted_clean_latent_l1_weight"]) * latent_loss
                        )
                    routing = moe_module.last_routing
                    if routing is None:
                        raise RuntimeError("formal training did not expose routing")
                    weights = routing.weights.detach().float()
                    adjustment = routing.vlm_logit_adjustment.detach().float()
                    route_weight_sums.append(float(weights.sum(-1).mean()))
                    route_entropies.append(
                        float(-(weights.clamp_min(1e-12).log() * weights).sum(-1).mean())
                    )
                    maximum_vlm_adjustment = max(
                        maximum_vlm_adjustment, float(adjustment.abs().max())
                    )
                    (loss / accumulation).backward()
                accumulated += float(loss.detach())
            gradient_norm = float(
                torch.nn.utils.clip_grad_norm_(trainable, float(training["max_grad_norm"]))
            )
            if not math.isfinite(accumulated) or not math.isfinite(gradient_norm):
                raise RuntimeError("formal training produced non-finite loss or gradients")
            optimizer.step()
            value = torch.tensor([accumulated / accumulation], device=device)
            dist.all_reduce(value)
            mean_loss = float(value.item() / world)
            recent_losses.append(mean_loss)
            completed = step + 1
            if rank == 0 and (completed == 1 or completed % 10 == 0):
                print(
                    f"VLM_TOKEN_MOE_FORMAL_TRAIN arm={arm} step={completed}/{steps} "
                    f"loss={mean_loss:.8f} grad_norm={gradient_norm:.8f}",
                    flush=True,
                )

            if completed in checkpoints:
                dist.barrier()
                checkpoint_path = output / f"step_{completed:06d}.safetensors"
                optimizer_checkpoint = checkpoint_path.with_suffix(".optimizer.pt")
                metadata_path = checkpoint_path.with_suffix(".json")
                if rank == 0:
                    if checkpoint_path.exists() or optimizer_checkpoint.exists() or metadata_path.exists():
                        raise RuntimeError("refusing to overwrite formal checkpoint milestone")
                    save_moe_checkpoint(wrapped.module, checkpoint_path)
                    torch.save(
                        {
                            "step": completed,
                            "arm": arm,
                            "config_sha256": config_sha,
                            "optimizer": optimizer.state_dict(),
                        },
                        optimizer_checkpoint,
                    )
                    write_once(
                        metadata_path,
                        {
                            "schema_version": 1,
                            "status": "matched_formal_checkpoint",
                            "step": completed,
                            "arm": arm,
                            "config_sha256": config_sha,
                            "checkpoint_sha256": file_sha256(checkpoint_path),
                            "qualification_checkpoint_used": False,
                        },
                    )
                dist.barrier()
                if completed in {int(value) for value in config["evaluation"]["calibration_steps"]}:
                    calibration = evaluate_split(
                        wrapped.module,
                        autoencoder,
                        calibration_rows,
                        context,
                        context_ids,
                        batched_prc_img,
                        metric,
                        moe_module,
                        config=config,
                        arm=arm,
                        split="calibration",
                        step=completed,
                        rank=rank,
                        world=world,
                        device=device,
                    )
                    if rank == 0:
                        assert calibration is not None
                        write_once(output / f"calibration_step_{completed:06d}.json", calibration)
                if completed in {int(value) for value in config["evaluation"]["holdout_steps"]}:
                    holdout = evaluate_split(
                        wrapped.module,
                        autoencoder,
                        holdout_rows,
                        context,
                        context_ids,
                        batched_prc_img,
                        metric,
                        moe_module,
                        config=config,
                        arm=arm,
                        split="holdout",
                        step=completed,
                        rank=rank,
                        world=world,
                        device=device,
                    )
                    if rank == 0:
                        assert holdout is not None
                        write_once(output / f"holdout_step_{completed:06d}.json", holdout)
                dist.barrier()

        vlm_output_update = sum(
            float(value.detach().float().abs().sum().cpu())
            for value in moe_module.moe.vlm_projector[-1].state_dict().values()
        )
        if arm == "image_only" and maximum_vlm_adjustment != 0.0:
            raise RuntimeError("formal image-only arm used nonzero VLM adjustment")
        if arm == "vlm_assisted" and vlm_output_update <= 0.0:
            raise RuntimeError("formal VLM-assisted score head did not update")
        tolerance = float(config["gates"]["expert_weight_sum_tolerance"])
        if max(abs(value - 1.0) for value in route_weight_sums) > tolerance:
            raise RuntimeError("formal expert weights no longer sum to one")
        elapsed = time.monotonic() - started
        if rank == 0:
            holdout_path = output / f"holdout_step_{steps:06d}.json"
            calibration_path = output / f"calibration_step_{steps:06d}.json"
            if not holdout_path.is_file() or not calibration_path.is_file():
                raise RuntimeError("formal final evaluations are missing")
            result = {
                "schema_version": 1,
                "status": "matched_formal_arm_complete",
                "arm": arm,
                "steps": steps,
                "world_size": world,
                "gpu_name": properties.name,
                "global_batch_size": global_batch,
                "trainable_parameters": sum(parameter.numel() for parameter in trainable),
                "mean_loss_last_five": sum(recent_losses[-5:]) / len(recent_losses[-5:]),
                "maximum_vlm_logit_adjustment": maximum_vlm_adjustment,
                "vlm_output_head_l1": vlm_output_update,
                "mean_training_route_entropy": sum(route_entropies) / len(route_entropies),
                "maximum_expert_weight_sum_error": max(
                    abs(value - 1.0) for value in route_weight_sums
                ),
                "peak_allocated_bytes_per_rank": int(torch.cuda.max_memory_allocated(device)),
                "wall_clock_hours": elapsed / 3600.0,
                "allocated_gpu_hours": elapsed * world / 3600.0,
                "config": config_path.relative_to(ROOT).as_posix(),
                "config_sha256": config_sha,
                "final_checkpoint": f"step_{steps:06d}.safetensors",
                "final_checkpoint_sha256": file_sha256(output / f"step_{steps:06d}.safetensors"),
                "calibration_result": calibration_path.relative_to(ROOT).as_posix(),
                "holdout_result": holdout_path.relative_to(ROOT).as_posix(),
                "qualification_checkpoint_used": False,
                "official_foundir_test_accessed": False,
                "sealed_internal_accessed": False,
                "parent_test_accessed": False,
                "internal_test_accessed": False,
                "automatic_next_experiment_launch": False,
            }
            write_once(output / "result.json", result)
            print("VLM_TOKEN_MOE_FORMAL_END " + json.dumps(result, sort_keys=True), flush=True)
        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    if args.preflight_only:
        preflight(config_path, args.arm)
        return 0
    return train(config_path, args.arm)


if __name__ == "__main__":
    raise SystemExit(main())
