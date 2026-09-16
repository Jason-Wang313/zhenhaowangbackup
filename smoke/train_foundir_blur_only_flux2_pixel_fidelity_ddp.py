#!/usr/bin/env python3
"""Paired tiny gate for direct RGB fidelity on the best Blur-only FLUX.2 parent."""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file
from torch.nn.parallel import DistributedDataParallel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party/flux2/src"))

from lowlight_restoration.flow import build_reference_flow_batch  # noqa: E402
from lowlight_restoration.lora import (  # noqa: E402
    inject_lora,
    load_lora_checkpoint,
    lora_state_dict,
    save_lora_checkpoint,
)
from smoke.train_foundir_blur_only_flux2_overfit_ddp import (  # noqa: E402
    encode_prompt,
    evaluate_snapshot,
    lpips_value,
    prepare_records,
    psnr,
    scheduled_index,
    sha256,
    tensor_image,
    unpack_tokens,
    write_once,
)


STATUS = "user_authorized_blur_only_step10k_pixel_fidelity_ablation"
REQUIRED_FALSE = (
    "qwen_degradation_conditioning",
    "mixed_degradations",
    "pixel_refiner",
    "decoder_tuning",
    "development_gt_optimization",
    "official_foundir_test_accessed",
    "sealed_internal_accessed",
)


def validate_config(config: Mapping[str, object]) -> None:
    if config.get("status") != STATUS:
        raise RuntimeError("config is not authorized for the step-10k pixel-fidelity gate")
    scope = config["scope"]
    if scope.get("category") != "01Blur" or any(scope.get(key) is not False for key in REQUIRED_FALSE):
        raise RuntimeError("pixel-fidelity Blur-only scope contract changed")
    model = config["model"]
    if model.get("name") != "flux.2-klein-base-4b" or int(model.get("parent_step", -1)) != 10000:
        raise RuntimeError("model family or parent step changed")
    training = config["training"]
    if training.get("objective") != "pure_noise_to_clean_rectified_flow_with_native_reference":
        raise RuntimeError("pure flow objective changed")
    if training.get("predicted_clean_identity") != "noise_minus_predicted_velocity":
        raise RuntimeError("predicted-clean identity changed")
    if training.get("optimizer_initialization") != "fresh_adamw_per_arm":
        raise RuntimeError("each arm must use fresh AdamW state")
    arms = training.get("arms")
    expected_arms = [
        {"name": "flow_control", "flow_mse_weight": 1.0, "rgb_mse_weight": 0.0},
        {"name": "rgb_mse", "flow_mse_weight": 1.0, "rgb_mse_weight": 1.0},
    ]
    if arms != expected_arms:
        raise RuntimeError("paired loss arms changed")
    maximum = int(training["max_steps_per_arm"])
    checkpoints = [int(value) for value in training["checkpoint_steps"]]
    if checkpoints != [50, 100, 200, 400] or maximum != 400:
        raise RuntimeError("bounded checkpoint schedule changed")
    if config["inference"].get("schedule") != "official_flux2_shifted_schedule":
        raise RuntimeError("official FLUX.2 schedule is required")
    if int(config["inference"]["num_steps"]) != 50 or float(config["inference"]["guidance"]) != 1.0:
        raise RuntimeError("inference contract changed")
    if config["data"].get("development_used_for_selection") is not False:
        raise RuntimeError("development GT cannot select an arm, loss weight, or checkpoint")
    world = int(config["distributed"]["required_world_size"])
    global_batch = int(training["global_batch_size"])
    for family, profile in training["hardware_profiles"].items():
        if family not in {"H100", "H200"}:
            raise RuntimeError("unsupported hardware profile")
        actual = world * int(profile["micro_batch_size_per_rank"]) * int(profile["gradient_accumulation_steps"])
        if actual != global_batch:
            raise RuntimeError(f"{family} profile changes global batch: {actual} != {global_batch}")
    gate = config["gate"]
    if gate.get("candidate_arm") != "rgb_mse" or gate.get("control_arm") != "flow_control":
        raise RuntimeError("paired gate arm identities changed")
    if int(gate["predeclared_candidate_step"]) != maximum or gate.get("full_scale_automatic") is not False:
        raise RuntimeError("candidate selection or full-scale stop contract changed")


def load_frozen_manifest(
    config: Mapping[str, object], *, verify_images: bool
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    data = config["data"]
    manifest = ROOT / str(data["frozen_manifest"])
    metadata_path = ROOT / str(data["frozen_manifest_metadata"])
    if sha256(manifest) != data["frozen_manifest_sha256"]:
        raise RuntimeError("frozen tiny manifest hash changed")
    if sha256(metadata_path) != data["frozen_manifest_metadata_sha256"]:
        raise RuntimeError("frozen tiny manifest metadata hash changed")
    metadata = json.loads(metadata_path.read_text())
    if (
        metadata.get("frozen_manifest_sha256") != data["frozen_manifest_sha256"]
        or metadata.get("protected_foundir_test_accessed") is not False
        or metadata.get("sealed_internal_accessed") is not False
    ):
        raise RuntimeError("frozen manifest metadata contract changed")
    with manifest.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    train = [row for row in rows if row["role"] == "tiny_overfit_train"]
    development = [row for row in rows if row["role"] == "future_blur_validation"]
    if len(train) != int(data["tiny_overfit_records"]):
        raise RuntimeError("tiny training record count changed")
    if len(development) != int(data["descriptive_development_records"]):
        raise RuntimeError("descriptive development record count changed")
    for role, selected in (("train", train), ("development", development)):
        if any(row["category"] != "01Blur" or row["composition_size"] != "1" for row in selected):
            raise RuntimeError(f"{role} contains non-isolated Blur")
        if len({row["source_group"] for row in selected}) != len(selected):
            raise RuntimeError(f"{role} source groups are not unique")
    if {row["source_group"] for row in train} & {row["source_group"] for row in development}:
        raise RuntimeError("train/development source-group leakage")
    if verify_images:
        for row in rows:
            for field, hash_field in (("lq_path", "lq_sha256"), ("gt_path", "gt_sha256")):
                path = ROOT / row[field]
                if not path.is_file() or sha256(path) != row[hash_field]:
                    raise RuntimeError(f"pair identity changed: {row['record_id']} {field}")
    return train, development


def predicted_clean_from_pure_flow(
    prediction_tokens: torch.Tensor, noise: torch.Tensor, clean_shape: torch.Size
) -> torch.Tensor:
    """Recover x_0 from v for x_t=(1-t)x_0+t*noise and v=noise-x_0."""

    velocity = unpack_tokens(prediction_tokens, clean_shape)
    if velocity.shape != noise.shape:
        raise RuntimeError("prediction/noise latent shape mismatch")
    return noise.to(velocity) - velocity


def decoded_rgb_mse(
    prediction_tokens: torch.Tensor,
    noise: torch.Tensor,
    clean_shape: torch.Size,
    autoencoder,
    target_rgb: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    predicted_clean = predicted_clean_from_pure_flow(prediction_tokens, noise, clean_shape)
    decoded = autoencoder.decode(predicted_clean).float().clamp(-1, 1)
    return F.mse_loss(decoded, target_rgb.float()), decoded


def verify_parent(config: Mapping[str, object]) -> None:
    model = config["model"]
    for key, hash_key in (
        ("flow_weights", "flow_weights_sha256"),
        ("autoencoder_weights", "autoencoder_weights_sha256"),
        ("parent_config", "parent_config_sha256"),
        ("parent_checkpoint", "parent_checkpoint_sha256"),
    ):
        path = ROOT / str(model[key])
        if not path.is_file() or sha256(path) != model[hash_key]:
            raise RuntimeError(f"model/parent identity changed: {path}")
    with safe_open(str(ROOT / str(model["parent_checkpoint"])), framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
    if metadata.get("step") != "10000" or metadata.get("config_sha256") != model["parent_config_sha256"]:
        raise RuntimeError("parent checkpoint metadata changed")


def verify_loaded_parent(model: torch.nn.Module, checkpoint: Path) -> None:
    expected = load_file(str(checkpoint), device="cpu")
    actual = lora_state_dict(model)
    if set(expected) != set(actual):
        raise RuntimeError("loaded parent LoRA keys changed")
    for key in expected:
        if not torch.equal(expected[key], actual[key].to(expected[key])):
            raise RuntimeError(f"loaded parent tensor mismatch: {key}")


def preflight(config_path: Path) -> None:
    config = json.loads(config_path.read_text())
    config["_config_path"] = str(config_path.resolve())
    validate_config(config)
    verify_parent(config)
    train, development = load_frozen_manifest(config, verify_images=True)
    prior = ROOT / str(config["vae_ceiling"]["prior_evidence"])
    prior_payload = json.loads(prior.read_text())
    blur = prior_payload["categories"]["01Blur"]
    if (
        abs(float(blur["psnr"]) - float(config["vae_ceiling"]["prior_blur_psnr"])) > 1e-12
        or abs(float(blur["lpips"]) - float(config["vae_ceiling"]["prior_blur_lpips"])) > 1e-12
        or prior_payload.get("protected_foundir_test_accessed") is not False
    ):
        raise RuntimeError("prior VAE ceiling evidence changed")
    print(json.dumps({
        "status": "FOUNDIR_BLUR_PIXEL_FIDELITY_PREFLIGHT_PASS",
        "config_sha256": sha256(config_path),
        "parent_checkpoint_sha256": config["model"]["parent_checkpoint_sha256"],
        "train_records": len(train),
        "descriptive_development_records": len(development),
        "arms": [arm["name"] for arm in config["training"]["arms"]],
        "sealed_internal_accessed": False,
        "official_foundir_test_accessed": False,
    }, sort_keys=True))


def evaluate_vae_ceiling(
    autoencoder,
    records: Sequence[Mapping[str, object]],
    metric,
    *,
    role: str,
    output_dir: Path,
    rank: int,
    world: int,
    device: torch.device,
) -> dict[str, object] | None:
    local = []
    with torch.inference_mode():
        for index in range(rank, len(records), world):
            record = records[index]
            gt = record["gt"].unsqueeze(0).to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                reconstructed = autoencoder.decode(autoencoder.encode(gt)).float().clamp(-1, 1)
            quantized = torch.from_numpy(
                np.asarray(tensor_image(reconstructed[0]), dtype=np.uint8).copy()
            ).permute(2, 0, 1).float().div_(127.5).sub_(1).unsqueeze(0).to(device)
            local.append({
                "record_id": record["record_id"],
                "psnr": psnr(gt, quantized),
                "lpips": lpips_value(metric, gt, quantized),
            })
    gathered: list[object | None] = [None] * world
    dist.all_gather_object(gathered, local)
    dist.barrier()
    if rank != 0:
        return None
    rows = sorted(
        [row for part in gathered if isinstance(part, list) for row in part],
        key=lambda row: row["record_id"],
    )
    if len(rows) != len(records):
        raise RuntimeError("VAE ceiling row count mismatch")
    result = {
        "role": role,
        "records": len(rows),
        "mean_psnr": float(np.mean([row["psnr"] for row in rows])),
        "minimum_psnr": float(np.min([row["psnr"] for row in rows])),
        "mean_lpips": float(np.mean([row["lpips"] for row in rows])),
        "maximum_lpips": float(np.max([row["lpips"] for row in rows])),
        "rows": rows,
    }
    write_once(
        output_dir / f"vae_ceiling_{role}.json",
        (json.dumps(result, indent=2, sort_keys=True) + "\n").encode(),
    )
    return result


def load_metrics(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text())
    if payload.get("status") != "complete" or len(payload.get("rows", [])) != int(payload.get("records", -1)):
        raise RuntimeError(f"incomplete metrics artifact: {path}")
    return payload


def compare_final_arms(config: Mapping[str, object], arm_results: Mapping[str, Mapping[str, object]]) -> dict[str, object]:
    gate = config["gate"]
    control = arm_results[str(gate["control_arm"])]
    candidate = arm_results[str(gate["candidate_arm"])]
    control_final = control["final_train_metrics"]
    candidate_final = candidate["final_train_metrics"]
    candidate_parent = candidate["parent_train_metrics"]
    control_rows = {row["record_id"]: row for row in control_final["rows"]}
    candidate_rows = {row["record_id"]: row for row in candidate_final["rows"]}
    if set(control_rows) != set(candidate_rows):
        raise RuntimeError("paired final record identities changed")
    fraction = float(np.mean([
        candidate_rows[key]["output_psnr"] > control_rows[key]["output_psnr"]
        for key in sorted(control_rows)
    ]))
    psnr_over_control = float(candidate_final["mean_output_psnr"] - control_final["mean_output_psnr"])
    lpips_regression_control = float(candidate_final["mean_output_lpips"] - control_final["mean_output_lpips"])
    psnr_over_parent = float(candidate_final["mean_output_psnr"] - candidate_parent["mean_output_psnr"])
    lpips_regression_parent = float(candidate_final["mean_output_lpips"] - candidate_parent["mean_output_lpips"])
    checks = {
        "train_psnr_gain_over_control": psnr_over_control >= float(gate["minimum_train_psnr_gain_over_control_db"]),
        "train_lpips_not_worse_than_control": lpips_regression_control <= float(gate["maximum_train_lpips_regression_vs_control"]),
        "train_psnr_gain_over_parent": psnr_over_parent >= float(gate["minimum_train_psnr_gain_over_parent_db"]),
        "train_lpips_not_worse_than_parent": lpips_regression_parent <= float(gate["maximum_train_lpips_regression_vs_parent"]),
        "per_record_psnr_fraction_over_control": fraction >= float(gate["minimum_psnr_improved_fraction_vs_control"]),
    }
    return {
        "checks": checks,
        "train_psnr_gain_over_control_db": psnr_over_control,
        "train_lpips_regression_vs_control": lpips_regression_control,
        "train_psnr_gain_over_parent_db": psnr_over_parent,
        "train_lpips_regression_vs_parent": lpips_regression_parent,
        "per_record_psnr_improved_fraction_over_control": fraction,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config_path = args.config.resolve()
    if args.preflight:
        preflight(config_path)
        return 0
    config = json.loads(config_path.read_text())
    config["_config_path"] = str(config_path)
    validate_config(config)
    verify_parent(config)
    train_rows, development_rows = load_frozen_manifest(config, verify_images=False)
    crop_size = int(config["data"]["crop_size"])
    train_records = prepare_records(train_rows, crop_size)
    development_records = prepare_records(development_rows, crop_size)
    output_dir = ROOT / str(config["runtime"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "result.json").exists():
        raise RuntimeError("pixel-fidelity result already exists; refusing rerun")
    if any(output_dir.iterdir()) and not args.resume:
        raise RuntimeError("pixel-fidelity output is nonempty; inspect and use --resume")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend=str(config["distributed"]["backend"]), device_id=device)
    started = time.monotonic()
    try:
        rank, world = dist.get_rank(), dist.get_world_size()
        if world != int(config["distributed"]["required_world_size"]) or torch.cuda.device_count() != world:
            raise RuntimeError("pixel-fidelity gate requires exactly eight visible GPUs")
        properties = torch.cuda.get_device_properties(device)
        if properties.total_memory < int(config["distributed"]["minimum_memory_gib"]) * 1024**3:
            raise RuntimeError("GPU memory is below contract")
        families = [name for name in ("H100", "H200") if name in properties.name]
        if len(families) != 1:
            raise RuntimeError(f"unsupported GPU: {properties.name}")
        family = families[0]
        names: list[str | None] = [None] * world
        dist.all_gather_object(names, properties.name)
        if any(family not in str(name) for name in names):
            raise RuntimeError(f"mixed GPU families are prohibited: {names}")
        profile = config["training"]["hardware_profiles"][family]
        if int(profile["micro_batch_size_per_rank"]) != 1:
            raise RuntimeError("this tiny gate requires microbatch one")
        accumulation = int(profile["gradient_accumulation_steps"])
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
        context, context_ids, cfg_context, cfg_ids = encode_prompt(config, device)
        from flux2.sampling import batched_prc_img
        from flux2.util import load_ae, load_flow_model
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

        autoencoder = load_ae(config["model"]["name"], device=device).eval().requires_grad_(False)
        model = load_flow_model(config["model"]["name"], device=device).eval().requires_grad_(False)
        injected = inject_lora(
            model,
            list(config["lora"]["target_patterns"]),
            rank=int(config["lora"]["rank"]),
            alpha=float(config["lora"]["alpha"]),
            dropout=float(config["lora"]["dropout"]),
        )
        if len(injected) != int(config["lora"]["expected_module_count"]):
            raise RuntimeError("FLUX LoRA target count changed")
        parent_checkpoint = ROOT / str(config["model"]["parent_checkpoint"])
        load_lora_checkpoint(model, parent_checkpoint)
        verify_loaded_parent(model, parent_checkpoint)
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        wrapped = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            broadcast_buffers=False,
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
        )
        metric = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(device).eval()

        ceiling_train_path = output_dir / "vae_ceiling_train.json"
        ceiling_development_path = output_dir / "vae_ceiling_descriptive_development.json"
        if ceiling_train_path.exists() and ceiling_development_path.exists():
            ceiling_train = json.loads(ceiling_train_path.read_text()) if rank == 0 else None
            ceiling_development = json.loads(ceiling_development_path.read_text()) if rank == 0 else None
            dist.barrier()
        elif ceiling_train_path.exists() or ceiling_development_path.exists():
            raise RuntimeError("partial VAE ceiling evidence requires inspection before resume")
        else:
            ceiling_train = evaluate_vae_ceiling(
                autoencoder, train_records, metric, role="train", output_dir=output_dir,
                rank=rank, world=world, device=device,
            )
            ceiling_development = evaluate_vae_ceiling(
                autoencoder, development_records, metric, role="descriptive_development", output_dir=output_dir,
                rank=rank, world=world, device=device,
            )
        if rank == 0:
            assert ceiling_train is not None and ceiling_development is not None
            ceiling_checks = {
                "train_mean_psnr": ceiling_train["mean_psnr"] >= float(config["vae_ceiling"]["minimum_mean_psnr"]),
                "train_mean_lpips": ceiling_train["mean_lpips"] <= float(config["vae_ceiling"]["maximum_mean_lpips"]),
                "development_mean_psnr": ceiling_development["mean_psnr"] >= float(config["vae_ceiling"]["minimum_mean_psnr"]),
                "development_mean_lpips": ceiling_development["mean_lpips"] <= float(config["vae_ceiling"]["maximum_mean_lpips"]),
            }
            ceiling_pass = all(ceiling_checks.values())
        else:
            ceiling_checks = {}
            ceiling_pass = False
        ceiling_tensor = torch.tensor([int(ceiling_pass)], dtype=torch.int32, device=device)
        dist.broadcast(ceiling_tensor, src=0)
        if not bool(ceiling_tensor.item()):
            raise RuntimeError("frozen VAE ceiling does not support this gate")

        arm_results: dict[str, dict[str, object]] = {}
        maximum = int(config["training"]["max_steps_per_arm"])
        checkpoint_steps = {int(value) for value in config["training"]["checkpoint_steps"]}
        for arm_index, arm in enumerate(config["training"]["arms"]):
            arm_name = str(arm["name"])
            arm_dir = output_dir / "arms" / arm_name
            arm_dir.mkdir(parents=True, exist_ok=True)
            arm_result_path = arm_dir / "result.json"
            if arm_result_path.exists():
                arm_results[arm_name] = json.loads(arm_result_path.read_text())
                dist.barrier()
                continue
            load_lora_checkpoint(wrapped.module, parent_checkpoint)
            verify_loaded_parent(wrapped.module, parent_checkpoint)
            optimizer = torch.optim.AdamW(
                trainable,
                lr=float(config["training"]["learning_rate"]),
                weight_decay=float(config["training"]["weight_decay"]),
                fused=True,
            )
            start_step = 0
            checkpoints = sorted(arm_dir.glob("step_*.safetensors")) if args.resume else []
            if checkpoints:
                latest = checkpoints[-1]
                start_step = int(latest.stem.split("_")[-1])
                load_lora_checkpoint(wrapped.module, latest)
                optimizer_path = latest.with_suffix(".optimizer.pt")
                optimizer_payload = torch.load(optimizer_path, map_location="cpu", weights_only=False)
                if (
                    optimizer_payload.get("config_sha256") != sha256(config_path)
                    or optimizer_payload.get("arm") != arm_name
                    or int(optimizer_payload.get("step", -1)) != start_step
                    or optimizer_payload.get("parent_checkpoint_sha256") != config["model"]["parent_checkpoint_sha256"]
                ):
                    raise RuntimeError("resume optimizer identity changed")
                optimizer.load_state_dict(optimizer_payload["optimizer"])
            elif any(arm_dir.iterdir()):
                allowed = {
                    arm_dir / "training_triptychs",
                    arm_dir / "descriptive_development",
                }
                if any(path not in allowed for path in arm_dir.iterdir()):
                    raise RuntimeError(f"unexpected partial arm artifacts: {arm_dir}")

            if start_step == 0:
                parent_train = evaluate_snapshot(
                    wrapped, autoencoder, train_records, cfg_context, cfg_ids, batched_prc_img, metric,
                    step=0, limit=len(train_records), config=config, output_dir=arm_dir,
                    rank=rank, world=world, device=device,
                )
                evaluate_snapshot(
                    wrapped, autoencoder, development_records, cfg_context, cfg_ids, batched_prc_img, metric,
                    step=0, limit=len(development_records), config=config,
                    output_dir=arm_dir / "descriptive_development", rank=rank, world=world, device=device,
                )
            else:
                parent_train = load_metrics(arm_dir / "training_triptychs/step_000000/metrics.json") if rank == 0 else None

            optimizer.zero_grad(set_to_none=True)
            final_train = None
            for step in range(start_step, maximum):
                flow_sum = 0.0
                rgb_sum = 0.0
                total_sum = 0.0
                for micro in range(accumulation):
                    position = (step * accumulation + micro) * world + rank
                    record = train_records[scheduled_index(position, len(train_records), seed)]
                    lq = record["lq"].unsqueeze(0).to(device)
                    gt = record["gt"].unsqueeze(0).to(device)
                    with torch.no_grad():
                        clean, degraded = autoencoder.encode(torch.cat([gt, lq])).chunk(2)
                    generator = torch.Generator(device=device).manual_seed(seed + position * 1_000_003)
                    noise = torch.randn(clean.shape, generator=generator, device=device, dtype=torch.bfloat16)
                    timesteps = torch.rand(
                        (1,), generator=generator, device=device, dtype=torch.bfloat16
                    ).clamp_(1e-4, 1 - 1e-4)
                    flow = build_reference_flow_batch(
                        clean.to(torch.bfloat16), degraded.to(torch.bfloat16), noise, timesteps,
                        batched_prc_img,
                        reference_time_coordinate=int(config["training"]["reference_time_coordinate"]),
                    )
                    sync = wrapped.no_sync() if micro + 1 < accumulation else contextlib.nullcontext()
                    with sync:
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            prediction = wrapped(
                                x=flow.model_tokens,
                                x_ids=flow.model_ids,
                                timesteps=flow.timesteps,
                                ctx=context,
                                ctx_ids=context_ids,
                                guidance=None,
                            )[:, : flow.target_token_count]
                            flow_loss = F.mse_loss(prediction.float(), flow.target_velocity.float())
                            if float(arm["rgb_mse_weight"]) > 0:
                                rgb_loss, _ = decoded_rgb_mse(
                                    prediction, noise, clean.shape, autoencoder, gt
                                )
                            else:
                                rgb_loss = flow_loss.new_zeros(())
                            loss = (
                                float(arm["flow_mse_weight"]) * flow_loss
                                + float(arm["rgb_mse_weight"]) * rgb_loss
                            )
                        (loss / accumulation).backward()
                    flow_sum += float(flow_loss.detach())
                    rgb_sum += float(rgb_loss.detach())
                    total_sum += float(loss.detach())
                if any(parameter.grad is not None for parameter in autoencoder.parameters()):
                    raise RuntimeError("frozen VAE unexpectedly accumulated gradients")
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable, float(config["training"]["max_grad_norm"]))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                values = torch.tensor(
                    [flow_sum / accumulation, rgb_sum / accumulation, total_sum / accumulation],
                    device=device,
                )
                dist.all_reduce(values)
                values = values / world
                completed = step + 1
                if rank == 0 and (completed == 1 or completed % 10 == 0):
                    print(
                        f"FOUNDIR_BLUR_PIXEL_FIDELITY arm={arm_name} step={completed}/{maximum} "
                        f"flow={float(values[0]):.8f} rgb={float(values[1]):.8f} "
                        f"total={float(values[2]):.8f} grad_norm={float(grad_norm):.6f}",
                        flush=True,
                    )
                if completed in checkpoint_steps:
                    dist.barrier()
                    if rank == 0:
                        checkpoint = arm_dir / f"step_{completed:06d}.safetensors"
                        if checkpoint.exists():
                            raise RuntimeError(f"refusing to overwrite checkpoint: {checkpoint}")
                        metadata = {
                            "step": completed,
                            "arm": arm_name,
                            "config_sha256": sha256(config_path),
                            "parent_checkpoint_sha256": config["model"]["parent_checkpoint_sha256"],
                            "optimizer_initialization": "fresh_adamw",
                        }
                        save_lora_checkpoint(wrapped.module, checkpoint, metadata)
                        with checkpoint.with_suffix(".optimizer.pt").open("xb") as handle:
                            torch.save({**metadata, "optimizer": optimizer.state_dict()}, handle)
                    dist.barrier()
                    final_train = evaluate_snapshot(
                        wrapped, autoencoder, train_records, cfg_context, cfg_ids, batched_prc_img, metric,
                        step=completed, limit=len(train_records), config=config, output_dir=arm_dir,
                        rank=rank, world=world, device=device,
                    )

            if rank == 0 and final_train is None:
                final_train = load_metrics(arm_dir / f"training_triptychs/step_{maximum:06d}/metrics.json")
            final_development = evaluate_snapshot(
                wrapped, autoencoder, development_records, cfg_context, cfg_ids, batched_prc_img, metric,
                step=maximum, limit=len(development_records), config=config,
                output_dir=arm_dir / "descriptive_development", rank=rank, world=world, device=device,
            )
            if rank == 0:
                assert parent_train is not None and final_train is not None and final_development is not None
                checkpoint = arm_dir / f"step_{maximum:06d}.safetensors"
                arm_result = {
                    "schema_version": 1,
                    "status": "complete",
                    "arm": arm_name,
                    "loss_weights": arm,
                    "parent_train_metrics": parent_train,
                    "final_train_metrics": final_train,
                    "final_descriptive_development_metrics": final_development,
                    "development_used_for_selection": False,
                    "checkpoint": checkpoint.relative_to(ROOT).as_posix(),
                    "checkpoint_sha256": sha256(checkpoint),
                    "config_sha256": sha256(config_path),
                    "parent_checkpoint_sha256": config["model"]["parent_checkpoint_sha256"],
                    "optimizer_initialization": "fresh_adamw",
                    "qwen_degradation_conditioning": False,
                    "mixed_degradations": False,
                    "pixel_refiner": False,
                    "decoder_tuning": False,
                    "development_gt_optimization": False,
                    "sealed_internal_accessed": False,
                    "official_foundir_test_accessed": False,
                }
                write_once(
                    arm_result_path,
                    (json.dumps(arm_result, indent=2, sort_keys=True) + "\n").encode(),
                )
                arm_results[arm_name] = arm_result
            dist.barrier()
            if rank != 0:
                arm_results[arm_name] = json.loads(arm_result_path.read_text())

        if rank == 0:
            comparison = compare_final_arms(config, arm_results)
            all_checks = {
                **comparison["checks"],
                **{f"vae_ceiling_{key}": value for key, value in ceiling_checks.items()},
            }
            metric_pass = all(all_checks.values())
            candidate = arm_results[str(config["gate"]["candidate_arm"])]
            result = {
                "schema_version": 1,
                "status": "awaiting_human_visual_review" if metric_pass else "failed",
                "automatic_metric_gate_pass": metric_pass,
                "human_visual_review_required": True,
                "human_visual_review_complete": False,
                "checks": all_checks,
                "comparison": {key: value for key, value in comparison.items() if key != "checks"},
                "arms": arm_results,
                "vae_ceiling": {"train": ceiling_train, "descriptive_development": ceiling_development},
                "selected_checkpoint": candidate["checkpoint"],
                "selected_checkpoint_sha256": candidate["checkpoint_sha256"],
                "config": config_path.relative_to(ROOT).as_posix(),
                "config_sha256": sha256(config_path),
                "parent_checkpoint": config["model"]["parent_checkpoint"],
                "parent_checkpoint_sha256": config["model"]["parent_checkpoint_sha256"],
                "development_used_for_selection": False,
                "full_scale_automatic": False,
                "qwen_degradation_conditioning": False,
                "mixed_degradations": False,
                "pixel_refiner": False,
                "decoder_tuning": False,
                "development_gt_optimization": False,
                "sealed_internal_accessed": False,
                "official_foundir_test_accessed": False,
                "gpu_family": family,
                "world_size": world,
                "global_batch_size": int(config["training"]["global_batch_size"]),
                "wall_seconds": time.monotonic() - started,
            }
            write_once(output_dir / "result.json", (json.dumps(result, indent=2, sort_keys=True) + "\n").encode())
            print(
                "FOUNDIR_BLUR_PIXEL_FIDELITY_METRIC_PASS_STOP_FOR_VISUAL_REVIEW"
                if metric_pass
                else "FOUNDIR_BLUR_PIXEL_FIDELITY_SCIENTIFIC_FAILURE",
                flush=True,
            )
            exit_value = 0 if metric_pass else 2
        else:
            exit_value = 0
        exit_tensor = torch.tensor([exit_value], dtype=torch.int32, device=device)
        dist.broadcast(exit_tensor, src=0)
        dist.barrier()
        return int(exit_tensor.item())
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
