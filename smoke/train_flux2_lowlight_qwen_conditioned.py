#!/usr/bin/env python3
"""Train the fair frozen-Qwen-conditioned FLUX.2 Lowlight LoRA."""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party/flux2/src"))

from lowlight_restoration.condition_adapter import LowLightLevelContext
from lowlight_restoration.conditioned_authorization import (
    QWEN_CHECKPOINT_SHA256,
    validate_conditioned_authorization,
    workspace_path,
)
from lowlight_restoration.conditioned_checkpoint import (
    conditioned_state_dict,
    load_conditioned_checkpoint,
    save_conditioned_checkpoint,
    trainable_parameters,
)
from lowlight_restoration.config import TrainConfig, load_train_config
from lowlight_restoration.data import LowLightPairDataset, load_pair_manifest
from lowlight_restoration.flow import build_reference_flow_batch
from lowlight_restoration.lora import inject_lora
from lowlight_restoration.qwen_predictions import load_prediction_manifest
from train_flux2_lowlight import (
    append_metric_once,
    atomic_torch_save,
    atomic_write_text,
    checkpoint_paths,
    collate,
    deterministic_index,
    load_checkpoint_record,
    load_or_create_baseline_probe,
    prepare_output_dir,
    publish_once,
    recover_checkpoint_metric,
    require_finite_numbers,
    sha256,
    write_training_summary,
)


def stage_frozen_prompt_context(
    config: TrainConfig, output_dir: Path, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    condition = config.section("condition")
    source = workspace_path(
        condition["baseline_prompt_context"], label="condition.baseline_prompt_context"
    )
    source_metadata = workspace_path(
        condition["baseline_prompt_metadata"], label="condition.baseline_prompt_metadata"
    )
    destination = output_dir / "prompt_context.safetensors"
    destination_metadata = destination.with_suffix(destination.suffix + ".json")
    for origin, target, hash_key in (
        (source, destination, "baseline_prompt_context_sha256"),
        (source_metadata, destination_metadata, "baseline_prompt_metadata_sha256"),
    ):
        expected = str(condition[hash_key])
        if target.exists():
            if not target.is_file() or sha256(target) != expected:
                raise RuntimeError(f"staged prompt artifact differs: {target}")
            continue
        temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
        shutil.copyfile(origin, temporary)
        if sha256(temporary) != expected:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(f"copied prompt artifact hash mismatch: {origin}")
        publish_once(temporary, target)
    tensors = load_file(str(destination), device="cpu")
    required = {"context", "context_ids", "empty_context", "empty_context_ids"}
    if set(tensors) != required:
        raise RuntimeError("frozen prompt context tensor set changed")
    context = tensors["context"]
    context_ids = tensors["context_ids"]
    if context.shape != (1, 512, 7680) or context_ids.shape != (1, 512, 4):
        raise RuntimeError("frozen prompt context shape changed")
    return context.to(device), context_ids.to(device)


def conditioned_collate(
    dataset: LowLightPairDataset,
    positions: list[int],
    prediction_levels: Mapping[str, int],
) -> dict[str, object]:
    batch = collate(dataset, positions)
    levels = []
    for pair_id in batch["pair_id"]:
        try:
            level = prediction_levels[str(pair_id)]
        except KeyError as exc:
            raise RuntimeError(f"missing frozen condition for pair: {pair_id}") from exc
        if level not in range(1, 6):
            raise RuntimeError(f"invalid frozen condition for pair: {pair_id}")
        levels.append(level)
    batch["predicted_level"] = torch.tensor(levels, dtype=torch.int64)
    return batch


def apply_condition(
    adapter: LowLightLevelContext,
    context: torch.Tensor,
    context_ids: torch.Tensor,
    levels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size = int(levels.shape[0])
    return adapter(
        context.expand(batch_size, -1, -1),
        context_ids.expand(batch_size, -1, -1),
        levels,
    )


def deterministic_conditioned_probe(
    flow_model: torch.nn.Module,
    autoencoder: torch.nn.Module,
    adapter: LowLightLevelContext,
    dataset: LowLightPairDataset,
    prediction_levels: Mapping[str, int],
    context: torch.Tensor,
    context_ids: torch.Tensor,
    pack_image,
    *,
    batch_size: int,
    seed: int,
    reference_time: int,
    device: torch.device,
) -> dict[str, object]:
    probe_size = min(batch_size, len(dataset))
    positions = [
        deterministic_index(index, len(dataset), seed + 71_003)
        for index in range(probe_size)
    ]
    batch = conditioned_collate(dataset, positions, prediction_levels)
    clean = batch["clean"].to(device=device, dtype=torch.float32)
    degraded = batch["degraded"].to(device=device, dtype=torch.float32)
    levels = batch["predicted_level"].to(device=device)
    model_was_training = flow_model.training
    adapter_was_training = adapter.training
    flow_model.eval()
    adapter.eval()
    try:
        with torch.no_grad():
            latents = autoencoder.encode(torch.cat([clean, degraded], dim=0))
            clean_latents, degraded_latents = latents.chunk(2)
            clean_latents = clean_latents.to(torch.bfloat16)
            degraded_latents = degraded_latents.to(torch.bfloat16)
            noise_seed = seed + 91_009
            generator = torch.Generator(device=device).manual_seed(noise_seed)
            noise = torch.randn(
                clean_latents.shape,
                generator=generator,
                device=device,
                dtype=torch.bfloat16,
            )
            timesteps = torch.full(
                (probe_size,), 0.5, device=device, dtype=torch.bfloat16
            )
            flow_batch = build_reference_flow_batch(
                clean_latents,
                degraded_latents,
                noise,
                timesteps,
                pack_image,
                reference_time_coordinate=reference_time,
            )
            batch_context, batch_context_ids = apply_condition(
                adapter, context, context_ids, levels
            )
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                prediction = flow_model(
                    x=flow_batch.model_tokens,
                    x_ids=flow_batch.model_ids,
                    timesteps=flow_batch.timesteps,
                    ctx=batch_context,
                    ctx_ids=batch_context_ids,
                    guidance=None,
                )[:, : flow_batch.target_token_count]
                loss = torch.nn.functional.mse_loss(
                    prediction.float(), flow_batch.target_velocity.float()
                )
    finally:
        flow_model.train(model_was_training)
        adapter.train(adapter_was_training)
    result = {
        "probe_loss": float(loss),
        "probe_pairs": list(batch["pair_id"]),
        "probe_predicted_levels": levels.cpu().tolist(),
        "probe_positions": positions,
        "probe_noise_seed": noise_seed,
        "probe_timestep": 0.5,
    }
    require_finite_numbers(result)
    return result


def save_training_checkpoint(
    flow_model: torch.nn.Module,
    adapter: LowLightLevelContext,
    optimizer: torch.optim.Optimizer,
    output_dir: Path,
    step: int,
    config: TrainConfig,
    metric: dict[str, object],
    prediction_manifest_sha256: str,
) -> Path:
    path, optimizer_path, metadata_path = checkpoint_paths(output_dir, step)
    if any(item.exists() for item in (path, optimizer_path, metadata_path)):
        raise RuntimeError(f"refusing to overwrite checkpoint artifacts for step {step}")
    metadata = {
        "step": step,
        "config_sha256": config.sha256,
        "base_model": config.section("model")["name"],
        "kind": "flux2_lowlight_qwen_conditioned_lora_and_context_adapter",
        "prediction_manifest_sha256": prediction_manifest_sha256,
        "qwen_checkpoint_sha256": QWEN_CHECKPOINT_SHA256,
        "metric": metric,
    }
    temporary_dir = Path(
        tempfile.mkdtemp(dir=output_dir, prefix=f".step_{step:06d}.")
    )
    temporary_checkpoint = temporary_dir / path.name
    temporary_optimizer = temporary_dir / optimizer_path.name
    temporary_metadata = temporary_checkpoint.with_suffix(
        temporary_checkpoint.suffix + ".json"
    )
    try:
        save_conditioned_checkpoint(
            flow_model, adapter, temporary_checkpoint, metadata
        )
        atomic_torch_save(
            {
                "step": step,
                "optimizer": optimizer.state_dict(),
                "config_sha256": config.sha256,
                "prediction_manifest_sha256": prediction_manifest_sha256,
            },
            temporary_optimizer,
        )
        metadata["checkpoint_sha256"] = sha256(temporary_checkpoint)
        metadata["optimizer_sha256"] = sha256(temporary_optimizer)
        atomic_write_text(
            temporary_metadata, json.dumps(metadata, indent=2, sort_keys=True) + "\n"
        )
        publish_once(temporary_optimizer, optimizer_path)
        publish_once(temporary_checkpoint, path)
        publish_once(temporary_metadata, metadata_path)
    finally:
        shutil.rmtree(temporary_dir, ignore_errors=True)
    record = load_checkpoint_record(path, config)
    if (
        record.get("prediction_manifest_sha256") != prediction_manifest_sha256
        or record.get("qwen_checkpoint_sha256") != QWEN_CHECKPOINT_SHA256
    ):
        raise RuntimeError("published conditioned checkpoint provenance changed")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = load_train_config(args.config)
    try:
        scope = validate_conditioned_authorization(config)
    except RuntimeError as exc:
        raise SystemExit(f"RUN_NOT_AUTHORIZED {exc}") from exc
    if scope != "full_qwen_conditioned":
        raise SystemExit("QWEN_CONDITIONED_SCOPE_REQUIRED")
    print(f"RUN_AUTHORIZATION_PASS scope={scope}", flush=True)

    os.environ.update(
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        KLEIN_4B_BASE_MODEL_PATH=str(config.workspace_path("model", "flow_weights")),
        AE_MODEL_PATH=str(config.workspace_path("model", "autoencoder_weights")),
        QWEN3_MODEL_PATH=str(config.workspace_path("model", "text_encoder")),
    )
    if not torch.cuda.is_available():
        raise SystemExit("CUDA_NOT_AVAILABLE")
    device = torch.device("cuda")
    training = config.section("training")
    data_config = config.section("data")
    lora_config = config.section("lora")
    condition = config.section("condition")
    seed = int(training["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    records = load_pair_manifest(config.workspace_path("data", "manifest"))
    prediction_levels, _, prediction_manifest_sha = load_prediction_manifest(
        config, records, verify_sidecars=False
    )
    dataset = LowLightPairDataset(
        records,
        split=str(data_config["split"]),
        crop_size=int(data_config["crop_size"]),
        seed=seed,
    )
    if len(dataset) != int(data_config["expected_records"]):
        raise SystemExit(f"BAD_TRAIN_RECORD_COUNT {len(dataset)}")

    output_dir, start_step, resume_path = prepare_output_dir(config, args.resume)
    context, context_ids = stage_frozen_prompt_context(config, output_dir, device)

    from flux2.sampling import batched_prc_img
    from flux2.util import load_ae, load_flow_model

    model_name = str(config.section("model")["name"])
    autoencoder = load_ae(model_name, device=device).eval().requires_grad_(False)
    flow_model = load_flow_model(model_name, device=device).eval().requires_grad_(False)
    target_names = inject_lora(
        flow_model,
        list(lora_config["target_patterns"]),
        rank=int(lora_config["rank"]),
        alpha=float(lora_config["alpha"]),
        dropout=float(lora_config["dropout"]),
    )
    if len(target_names) != int(lora_config["expected_module_count"]):
        raise SystemExit(f"BAD_LORA_MODULE_COUNT {len(target_names)}")
    adapter = LowLightLevelContext(
        context_dim=int(condition["context_dim"]),
        levels=int(condition["levels"]),
        init_std=float(condition["init_std"]),
    ).to(device)
    adapter_count = sum(parameter.numel() for parameter in adapter.parameters())
    if adapter_count != int(condition["expected_trainable_parameters"]):
        raise SystemExit(f"BAD_CONDITION_ADAPTER_PARAMETER_COUNT {adapter_count}")
    trainable = trainable_parameters(flow_model, adapter)
    trainable_count = sum(parameter.numel() for parameter in trainable)
    if trainable_count != int(condition["expected_total_trainable_parameters"]):
        raise SystemExit(f"BAD_TRAINABLE_PARAMETER_COUNT {trainable_count}")

    batch_size = int(training["batch_size"])
    accumulation = int(training["gradient_accumulation_steps"])
    reference_time = int(training["reference_time_coordinate"])
    checkpoint_every = int(training["checkpoint_every"])
    log_every = int(training["log_every"])
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    baseline_probe_path = output_dir / "probe_baseline.json"
    new_probe = None
    if not baseline_probe_path.exists() and start_step == 0:
        new_probe = deterministic_conditioned_probe(
            flow_model,
            autoencoder,
            adapter,
            dataset,
            prediction_levels,
            context,
            context_ids,
            batched_prc_img,
            batch_size=batch_size,
            seed=seed,
            reference_time=reference_time,
            device=device,
        )
    baseline_probe = load_or_create_baseline_probe(
        output_dir, config, start_step=start_step, probe=new_probe
    )
    print("BASELINE_PROBE " + json.dumps(baseline_probe, sort_keys=True), flush=True)
    if resume_path is not None:
        load_conditioned_checkpoint(flow_model, adapter, resume_path)
        optimizer_payload = torch.load(
            resume_path.with_suffix(".optimizer.pt"),
            map_location="cpu",
            weights_only=True,
        )
        if (
            optimizer_payload.get("config_sha256") != config.sha256
            or optimizer_payload.get("prediction_manifest_sha256")
            != prediction_manifest_sha
        ):
            raise RuntimeError("resume optimizer belongs to a different conditioned run")
        optimizer.load_state_dict(optimizer_payload["optimizer"])
        recovered = recover_checkpoint_metric(output_dir, resume_path, config)
        print(f"RESUMED step={start_step} checkpoint={resume_path}", flush=True)
        if recovered:
            print(f"RECOVERED_CHECKPOINT_METRIC step={start_step}", flush=True)

    max_steps = int(training["max_steps"])
    if start_step >= max_steps:
        summary = write_training_summary(
            output_dir, config, trainable_count, max_steps
        )
        print("TRAINING_ALREADY_COMPLETE " + json.dumps(summary, sort_keys=True))
        return
    metrics_path = output_dir / "metrics.jsonl"
    optimizer.zero_grad(set_to_none=True)

    for step in range(start_step, max_steps):
        accumulated_loss = 0.0
        step_pairs: list[str] = []
        step_levels: list[int] = []
        for micro in range(accumulation):
            base_position = (step * accumulation + micro) * batch_size
            positions = [
                deterministic_index(base_position + offset, len(dataset), seed)
                for offset in range(batch_size)
            ]
            batch = conditioned_collate(dataset, positions, prediction_levels)
            clean = batch["clean"].to(device=device, dtype=torch.float32)
            degraded = batch["degraded"].to(device=device, dtype=torch.float32)
            levels = batch["predicted_level"].to(device=device)
            step_pairs.extend(batch["pair_id"])
            step_levels.extend(levels.cpu().tolist())
            with torch.no_grad():
                latents = autoencoder.encode(torch.cat([clean, degraded], dim=0))
                clean_latents, degraded_latents = latents.chunk(2)
                clean_latents = clean_latents.to(torch.bfloat16)
                degraded_latents = degraded_latents.to(torch.bfloat16)
            generator = torch.Generator(device=device).manual_seed(
                seed + (step * accumulation + micro) * 1_000_003
            )
            noise = torch.randn(
                clean_latents.shape,
                generator=generator,
                device=device,
                dtype=torch.bfloat16,
            )
            timesteps = torch.rand(
                (batch_size,),
                generator=generator,
                device=device,
                dtype=torch.bfloat16,
            ).clamp_(0.0001, 0.9999)
            flow_batch = build_reference_flow_batch(
                clean_latents,
                degraded_latents,
                noise,
                timesteps,
                batched_prc_img,
                reference_time_coordinate=reference_time,
            )
            batch_context, batch_context_ids = apply_condition(
                adapter, context, context_ids, levels
            )
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                prediction = flow_model(
                    x=flow_batch.model_tokens,
                    x_ids=flow_batch.model_ids,
                    timesteps=flow_batch.timesteps,
                    ctx=batch_context,
                    ctx_ids=batch_context_ids,
                    guidance=None,
                )[:, : flow_batch.target_token_count]
                loss = torch.nn.functional.mse_loss(
                    prediction.float(), flow_batch.target_velocity.float()
                )
                scaled_loss = loss / accumulation
            scaled_loss.backward()
            accumulated_loss += float(loss.detach())

        grad_norm = torch.nn.utils.clip_grad_norm_(
            trainable, float(training["max_grad_norm"])
        )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        completed_step = step + 1
        metric = {
            "step": completed_step,
            "loss": accumulated_loss / accumulation,
            "grad_norm": float(grad_norm),
            "pairs": step_pairs,
            "predicted_levels": step_levels,
            "max_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
        }
        if completed_step % log_every == 0:
            print("TRAIN " + json.dumps(metric, sort_keys=True), flush=True)
        if completed_step % checkpoint_every == 0 or completed_step == max_steps:
            metric.update(
                deterministic_conditioned_probe(
                    flow_model,
                    autoencoder,
                    adapter,
                    dataset,
                    prediction_levels,
                    context,
                    context_ids,
                    batched_prc_img,
                    batch_size=batch_size,
                    seed=seed,
                    reference_time=reference_time,
                    device=device,
                )
            )
            checkpoint = save_training_checkpoint(
                flow_model,
                adapter,
                optimizer,
                output_dir,
                completed_step,
                config,
                metric,
                prediction_manifest_sha,
            )
            before = {
                key: value.clone()
                for key, value in conditioned_state_dict(flow_model, adapter).items()
            }
            load_conditioned_checkpoint(flow_model, adapter, checkpoint)
            after = conditioned_state_dict(flow_model, adapter)
            if any(not torch.equal(before[key], after[key]) for key in before):
                raise RuntimeError("conditioned checkpoint round-trip verification failed")
            append_metric_once(metrics_path, metric)
            print(f"CHECKPOINT_ROUNDTRIP_PASS {checkpoint}", flush=True)

    summary = write_training_summary(output_dir, config, trainable_count, max_steps)
    print(
        "LOWLIGHT_QWEN_CONDITIONED_TRAINING_COMPLETE "
        + json.dumps(summary, sort_keys=True),
        flush=True,
    )
    del autoencoder, flow_model, adapter
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
