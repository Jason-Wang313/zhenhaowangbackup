#!/usr/bin/env python3
"""Eight-GPU FLUX.2 native-reference overfit gate for five-type FoundIR."""

from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import hashlib
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
from PIL import Image
from torch.nn.parallel import DistributedDataParallel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party/flux2/src"))

from lowlight_restoration.flow import build_degraded_anchored_flow_batch  # noqa: E402
from lowlight_restoration.foundir_five_type import FIVE_TYPE_CATEGORIES  # noqa: E402
from lowlight_restoration.lora import (  # noqa: E402
    inject_lora,
    load_lora_checkpoint,
    save_lora_checkpoint,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_int(*values: object) -> int:
    return int.from_bytes(hashlib.sha256(":".join(map(str, values)).encode()).digest()[:8], "big")


def write_once(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise RuntimeError(f"refusing to overwrite differing result: {path}")
        return
    with path.open("xb") as handle:
        handle.write(content)


def resume_checkpoint(model, optimizer, output_dir: Path, config_sha: str) -> tuple[int, list[float]]:
    candidates: list[tuple[int, Path]] = []
    for checkpoint in output_dir.glob("step_*.safetensors"):
        try:
            step = int(checkpoint.stem.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        optimizer_path = checkpoint.with_suffix(".optimizer.pt")
        metadata_path = checkpoint.with_suffix(checkpoint.suffix + ".json")
        if not optimizer_path.is_file() or not metadata_path.is_file():
            raise RuntimeError(f"partial overfit checkpoint: {checkpoint}")
        metadata = json.loads(metadata_path.read_text())
        if int(metadata.get("step", -1)) != step or metadata.get("config_sha256") != config_sha:
            raise RuntimeError(f"overfit checkpoint contract changed: {checkpoint}")
        candidates.append((step, checkpoint))
    if not candidates:
        return 0, []
    step, checkpoint = max(candidates)
    load_lora_checkpoint(model, checkpoint)
    payload = torch.load(checkpoint.with_suffix(".optimizer.pt"), map_location="cpu", weights_only=False)
    if payload.get("config_sha256") != config_sha or int(payload.get("step", -1)) != step:
        raise RuntimeError("overfit optimizer checkpoint contract changed")
    optimizer.load_state_dict(payload["optimizer"])
    losses = [float(value) for value in payload.get("training_losses", [])]
    if len(losses) != step:
        raise RuntimeError("overfit checkpoint loss history is incomplete")
    return step, losses


def load_conditions(path: Path, expected: int) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    required = {
        "record_id", "category", "composition_size", "lq_path", "gt_path",
        "lq_sha256", "gt_sha256", "qwen_text", "type_oracle_text", "generic_text",
    }
    if reader.fieldnames is None or not required.issubset(reader.fieldnames):
        raise RuntimeError("FoundIR condition cache schema changed")
    if len(rows) != expected or len({row["record_id"] for row in rows}) != expected:
        raise RuntimeError("FoundIR overfit condition cardinality changed")
    if set(row["category"] for row in rows) != set(FIVE_TYPE_CATEGORIES):
        raise RuntimeError("FoundIR overfit conditions do not cover every category")
    if set(int(row["composition_size"]) for row in rows) != {1, 2, 3}:
        raise RuntimeError("FoundIR overfit conditions do not cover single/double/triple")
    return rows


def image_tensor(path: Path) -> torch.Tensor:
    with Image.open(path) as handle:
        array = np.asarray(handle.convert("RGB"), dtype=np.uint8).copy()
    return torch.from_numpy(array).permute(2, 0, 1).float().div_(127.5).sub_(1.0)


def prepare_crops(rows: Sequence[Mapping[str, str]], *, crop_size: int, seed: int) -> list[dict[str, object]]:
    prepared: list[dict[str, object]] = []
    for row in rows:
        lq_path = ROOT / row["lq_path"]
        gt_path = ROOT / row["gt_path"]
        if sha256(lq_path) != row["lq_sha256"] or sha256(gt_path) != row["gt_sha256"]:
            raise RuntimeError(f"FoundIR pair changed: {row['record_id']}")
        lq = image_tensor(lq_path)
        gt = image_tensor(gt_path)
        if lq.shape != gt.shape or lq.shape[1] < crop_size or lq.shape[2] < crop_size:
            raise RuntimeError(f"FoundIR pair is unaligned or too small: {row['record_id']}")
        y = stable_int(seed, row["record_id"], "y") % (lq.shape[1] - crop_size + 1)
        x = stable_int(seed, row["record_id"], "x") % (lq.shape[2] - crop_size + 1)
        prepared.append(
            {
                **dict(row),
                "lq": lq[:, y : y + crop_size, x : x + crop_size].contiguous(),
                "gt": gt[:, y : y + crop_size, x : x + crop_size].contiguous(),
                "crop_xy": [int(x), int(y)],
            }
        )
    return prepared


def encode_prompts(config: Mapping[str, object], prompts: Sequence[str], device: torch.device) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    from flux2.sampling import batched_prc_txt
    from flux2.util import load_text_encoder

    model_name = str(config["model"]["name"])
    encoder = load_text_encoder(model_name, device=device).eval().requires_grad_(False)
    result: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    unique = sorted(set(prompts))
    with torch.inference_mode():
        for offset in range(0, len(unique), 4):
            chunk = unique[offset : offset + 4]
            contexts = encoder(chunk).to(torch.bfloat16)
            contexts, ids = batched_prc_txt(contexts)
            for index, prompt in enumerate(chunk):
                result[prompt] = (
                    contexts[index].cpu().contiguous(),
                    ids[index].cpu().contiguous(),
                )
    del encoder
    gc.collect()
    torch.cuda.empty_cache()
    return result


def batch_context(cache: Mapping[str, tuple[torch.Tensor, torch.Tensor]], prompt: str, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    context, ids = cache[prompt]
    return context.unsqueeze(0).to(device, non_blocking=True), ids.unsqueeze(0).to(device, non_blocking=True)


def scheduled_index(position: int, length: int, seed: int) -> int:
    epoch, offset = divmod(position, length)
    generator = torch.Generator(device="cpu").manual_seed(seed + epoch)
    return int(torch.randperm(length, generator=generator)[offset])


def fixed_probe(
    model,
    autoencoder,
    records: Sequence[Mapping[str, object]],
    contexts,
    pack_image,
    *,
    rank: int,
    world: int,
    seed: int,
    noise_scale: float,
    reference_time: int,
    device: torch.device,
) -> float:
    index = rank % len(records)
    record = records[index]
    gt = record["gt"].unsqueeze(0).to(device)
    lq = record["lq"].unsqueeze(0).to(device)
    context, context_ids = batch_context(contexts, str(record["qwen_text"]), device)
    with torch.inference_mode():
        clean_latent, degraded_latent = autoencoder.encode(torch.cat([gt, lq])).chunk(2)
        generator = torch.Generator(device=device).manual_seed(seed + index * 1009)
        noise = torch.randn(clean_latent.shape, device=device, dtype=torch.bfloat16, generator=generator)
        flow = build_degraded_anchored_flow_batch(
            clean_latent.to(torch.bfloat16), degraded_latent.to(torch.bfloat16), noise,
            torch.tensor([0.5], device=device, dtype=torch.bfloat16), pack_image,
            anchor_noise_scale=noise_scale, reference_time_coordinate=reference_time,
        )
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            prediction = model(
                x=flow.model_tokens, x_ids=flow.model_ids, timesteps=flow.timesteps,
                ctx=context, ctx_ids=context_ids, guidance=None,
            )[:, : flow.target_token_count]
            loss = torch.nn.functional.mse_loss(prediction.float(), flow.target_velocity.float())
    value = torch.tensor([float(loss)], device=device)
    dist.all_reduce(value)
    return float(value.item() / world)


def unpack_tokens(tokens: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    batch, channels, height, width = shape
    if tokens.shape != (batch, height * width, channels):
        raise RuntimeError(f"unexpected FLUX token shape: {tokens.shape} for {shape}")
    return tokens.transpose(1, 2).reshape(shape)


def restore(
    model,
    autoencoder,
    lq: torch.Tensor,
    context: torch.Tensor,
    context_ids: torch.Tensor,
    pack_image,
    *,
    seed: int,
    steps: int,
    noise_scale: float,
    reference_time: int,
) -> torch.Tensor:
    device = lq.device
    with torch.inference_mode():
        degraded = autoencoder.encode(lq)
        generator = torch.Generator(device=device).manual_seed(seed)
        noise = torch.randn(degraded.shape, device=device, dtype=torch.bfloat16, generator=generator)
        current = degraded.to(torch.bfloat16) + noise_scale * noise
        reference_tokens, reference_ids = pack_image(degraded.to(torch.bfloat16))
        reference_ids = reference_ids.clone()
        reference_ids[..., 0] = reference_time
        schedule = torch.linspace(1.0, 0.0, steps + 1, device=device)
        for index in range(steps):
            current_tokens, current_ids = pack_image(current)
            timestep = torch.full((1,), float(schedule[index]), device=device, dtype=torch.bfloat16)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                velocity = model(
                    x=torch.cat([current_tokens, reference_tokens], dim=1),
                    x_ids=torch.cat([current_ids, reference_ids], dim=1),
                    timesteps=timestep, ctx=context, ctx_ids=context_ids, guidance=None,
                )[:, : current_tokens.shape[1]]
            delta = float(schedule[index + 1] - schedule[index])
            current = current + delta * unpack_tokens(velocity, current.shape)
        return autoencoder.decode(current).float().clamp(-1.0, 1.0)


def psnr(reference: torch.Tensor, prediction: torch.Tensor) -> float:
    mse = float(torch.mean(((reference.float() - prediction.float()) * 0.5) ** 2))
    return 100.0 if mse <= 1e-12 else -10.0 * math.log10(mse)


def evaluate_arms(
    model,
    autoencoder,
    records: Sequence[Mapping[str, object]],
    contexts,
    pack_image,
    config: Mapping[str, object],
    *,
    rank: int,
    world: int,
    device: torch.device,
) -> list[dict[str, object]]:
    count = int(config["overfit"]["evaluation_records"])
    if count % world:
        raise RuntimeError("overfit evaluation count must divide world size")
    selected = list(records[:count])
    shift = max(1, len(records) // 2)
    flow = config["flow_training"]
    configured_arms = tuple(config["evaluation"].get("condition_arms", ("qwen",)))
    allowed_arms = {"qwen", "generic", "type_oracle", "shuffled_wrong"}
    if not configured_arms or set(configured_arms) - allowed_arms:
        raise RuntimeError(f"invalid overfit condition arms: {configured_arms}")
    results: list[dict[str, object]] = []
    for index in range(rank, count, world):
        record = selected[index]
        gt = record["gt"].unsqueeze(0).to(device)
        lq = record["lq"].unsqueeze(0).to(device)
        input_psnr = psnr(gt, lq)
        prompts = {
            "qwen": str(record["qwen_text"]),
            "generic": str(record["generic_text"]),
            "type_oracle": str(record["type_oracle_text"]),
            "shuffled_wrong": str(records[(index + shift) % len(records)]["qwen_text"]),
        }
        for arm in configured_arms:
            prompt = prompts[arm]
            context, ids = batch_context(contexts, prompt, device)
            prediction = restore(
                model, autoencoder, lq, context, ids, pack_image,
                seed=int(flow["seed"]) + index * 10007,
                steps=int(flow["inference_steps"]),
                noise_scale=float(flow["anchor_noise_scale"]),
                reference_time=int(flow["reference_time_coordinate"]),
            )
            results.append(
                {
                    "record_id": record["record_id"],
                    "category": record["category"],
                    "arm": arm,
                    "input_psnr": input_psnr,
                    "restored_psnr": psnr(gt, prediction),
                }
            )
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    distributed = config["distributed"]
    overfit = config["overfit"]
    flow = config["flow_training"]
    output_dir = ROOT / config["runtime"]["output_dir"] / "native_reference_overfit"
    conditions_path = ROOT / overfit["conditions_csv"]
    conditions_summary = ROOT / overfit["conditions_json"]
    summary = json.loads(conditions_summary.read_text())
    if summary["conditions_csv_sha256"] != sha256(conditions_path):
        raise RuntimeError("Qwen condition cache identity changed")
    rows = load_conditions(conditions_path, int(overfit["records"]))

    dist.init_process_group(backend=str(distributed["backend"]))
    started = time.monotonic()
    try:
        rank = dist.get_rank()
        world = dist.get_world_size()
        local_rank = int(os.environ["LOCAL_RANK"])
        if world != int(distributed["required_world_size"]) or torch.cuda.device_count() != world:
            raise RuntimeError("FLUX overfit requires exactly eight visible GPUs")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        properties = torch.cuda.get_device_properties(device)
        if properties.total_memory < int(distributed["minimum_memory_gib"]) * 1024**3:
            raise RuntimeError("GPU memory is below the H100/H200 contract")
        if not any(name in properties.name for name in ("H100", "H200")):
            raise RuntimeError(f"unsupported GPU: {properties.name}")
        seed = int(flow["seed"])
        random.seed(seed + rank)
        np.random.seed(seed + rank)
        torch.manual_seed(seed + rank)
        torch.cuda.manual_seed_all(seed + rank)
        os.environ.update(
            HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
            KLEIN_4B_BASE_MODEL_PATH=str(ROOT / config["model"]["flow_weights"]),
            AE_MODEL_PATH=str(ROOT / config["model"]["autoencoder_weights"]),
            QWEN3_MODEL_PATH=str(ROOT / config["model"]["text_encoder"]),
        )
        configured_arms = tuple(config["evaluation"].get("condition_arms", ("qwen",)))
        prompt_keys = {
            "qwen": "qwen_text",
            "generic": "generic_text",
            "type_oracle": "type_oracle_text",
            "shuffled_wrong": "qwen_text",
        }
        prompts = [
            str(row[prompt_keys[arm]])
            for row in rows
            for arm in configured_arms
        ]
        contexts = encode_prompts(config, prompts, device)
        records = prepare_crops(rows, crop_size=int(config["data"]["crop_size"]), seed=seed)

        from flux2.sampling import batched_prc_img
        from flux2.util import load_ae, load_flow_model

        autoencoder = load_ae(config["model"]["name"], device=device).eval().requires_grad_(False)
        model = load_flow_model(config["model"]["name"], device=device).eval().requires_grad_(False)
        names = inject_lora(
            model, list(config["lora"]["target_patterns"]),
            rank=int(config["lora"]["rank"]), alpha=float(config["lora"]["alpha"]),
            dropout=float(config["lora"]["dropout"]),
        )
        if len(names) != int(config["lora"]["expected_module_count"]):
            raise RuntimeError("FLUX LoRA target count changed")
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        trainable_count = sum(parameter.numel() for parameter in trainable)
        wrapped = DistributedDataParallel(
            model, device_ids=[local_rank], broadcast_buffers=False,
            find_unused_parameters=False, gradient_as_bucket_view=True,
        )
        optimizer = torch.optim.AdamW(
            trainable, lr=float(flow["learning_rate"]), weight_decay=float(flow["weight_decay"]), fused=True,
        )
        config_sha = sha256(args.config)
        start_step, losses = resume_checkpoint(
            wrapped.module, optimizer, output_dir, config_sha
        )
        baseline_path = output_dir / "baseline_probe.json"
        if baseline_path.is_file():
            baseline = json.loads(baseline_path.read_text())
            if baseline.get("config_sha256") != config_sha:
                raise RuntimeError("baseline probe belongs to another config")
            initial_probe = float(baseline["initial_probe_loss"])
        else:
            if start_step:
                raise RuntimeError("resume checkpoint has no baseline probe")
            initial_probe = fixed_probe(
                wrapped, autoencoder, records, contexts, batched_prc_img,
                rank=rank, world=world, seed=seed, noise_scale=float(flow["anchor_noise_scale"]),
                reference_time=int(flow["reference_time_coordinate"]), device=device,
            )
            if rank == 0:
                write_once(
                    baseline_path,
                    (json.dumps({"config_sha256": config_sha, "initial_probe_loss": initial_probe}, indent=2, sort_keys=True) + "\n").encode(),
                )
            dist.barrier()
        steps = int(overfit["steps"])
        accumulation = int(distributed["gradient_accumulation_steps"])
        optimizer.zero_grad(set_to_none=True)
        if rank == 0 and start_step:
            print(f"FOUNDIR_FLUX_OVERFIT_RESUME step={start_step}", flush=True)
        for step in range(start_step, steps):
            step_loss = 0.0
            for micro in range(accumulation):
                global_position = (step * accumulation + micro) * world + rank
                record = records[scheduled_index(global_position, len(records), seed)]
                gt = record["gt"].unsqueeze(0).to(device, non_blocking=True)
                lq = record["lq"].unsqueeze(0).to(device, non_blocking=True)
                context, context_ids = batch_context(contexts, str(record["qwen_text"]), device)
                with torch.no_grad():
                    clean_latent, degraded_latent = autoencoder.encode(torch.cat([gt, lq])).chunk(2)
                generator = torch.Generator(device=device).manual_seed(seed + global_position * 1000003)
                noise = torch.randn(clean_latent.shape, device=device, dtype=torch.bfloat16, generator=generator)
                timestep = torch.rand((1,), device=device, dtype=torch.bfloat16, generator=generator).clamp_(1e-4, 1 - 1e-4)
                batch = build_degraded_anchored_flow_batch(
                    clean_latent.to(torch.bfloat16), degraded_latent.to(torch.bfloat16), noise, timestep,
                    batched_prc_img, anchor_noise_scale=float(flow["anchor_noise_scale"]),
                    reference_time_coordinate=int(flow["reference_time_coordinate"]),
                )
                sync = contextlib.nullcontext() if micro + 1 == accumulation else wrapped.no_sync()
                with sync:
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        prediction = wrapped(
                            x=batch.model_tokens, x_ids=batch.model_ids, timesteps=batch.timesteps,
                            ctx=context, ctx_ids=context_ids, guidance=None,
                        )[:, : batch.target_token_count]
                        loss = torch.nn.functional.mse_loss(prediction.float(), batch.target_velocity.float())
                    (loss / accumulation).backward()
                step_loss += float(loss.detach())
            torch.nn.utils.clip_grad_norm_(trainable, float(flow["max_grad_norm"]))
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            value = torch.tensor([step_loss / accumulation], device=device)
            dist.all_reduce(value)
            mean_loss = float(value.item() / world)
            losses.append(mean_loss)
            if rank == 0 and ((step + 1) % 10 == 0 or step == 0):
                print(f"FOUNDIR_FLUX_OVERFIT step={step+1}/{steps} loss={mean_loss:.8f}", flush=True)
            if (step + 1) in {50, 100, steps}:
                dist.barrier()
                if rank == 0:
                    checkpoint = output_dir / f"step_{step+1:06d}.safetensors"
                    if checkpoint.exists():
                        raise RuntimeError(f"refusing to overwrite checkpoint: {checkpoint}")
                    save_lora_checkpoint(
                        wrapped.module, checkpoint,
                        {"step": step + 1, "config_sha256": config_sha, "trainable_parameters": trainable_count},
                    )
                    optimizer_path = checkpoint.with_suffix(".optimizer.pt")
                    with optimizer_path.open("xb") as handle:
                        torch.save(
                            {
                                "step": step + 1,
                                "config_sha256": config_sha,
                                "optimizer": optimizer.state_dict(),
                                "training_losses": losses,
                            },
                            handle,
                        )
                dist.barrier()
        final_probe = fixed_probe(
            wrapped, autoencoder, records, contexts, batched_prc_img,
            rank=rank, world=world, seed=seed, noise_scale=float(flow["anchor_noise_scale"]),
            reference_time=int(flow["reference_time_coordinate"]), device=device,
        )
        local_results = evaluate_arms(
            wrapped, autoencoder, records, contexts, batched_prc_img, config,
            rank=rank, world=world, device=device,
        )
        gathered: list[object | None] = [None] * world
        dist.all_gather_object(gathered, local_results)
        if rank == 0:
            result_rows = [row for part in gathered if isinstance(part, list) for row in part]
            arm_metrics = {}
            for arm in configured_arms:
                values = [row for row in result_rows if row["arm"] == arm]
                arm_metrics[arm] = {
                    "records": len(values),
                    "input_psnr": float(np.mean([row["input_psnr"] for row in values])),
                    "restored_psnr": float(np.mean([row["restored_psnr"] for row in values])),
                    "psnr_gain": float(np.mean([row["restored_psnr"] - row["input_psnr"] for row in values])),
                }
            ratio = final_probe / initial_probe
            checks = {
                "loss_ratio": ratio <= float(overfit["require_loss_ratio_at_most"]),
                "qwen_restoration_gain": arm_metrics["qwen"]["psnr_gain"] >= float(overfit["minimum_restored_psnr_gain_db"]),
            }
            payload = {
                "schema_version": 1,
                "status": "pass" if all(checks.values()) else "failed",
                "automatic_gate_pass": all(checks.values()),
                "checks": checks,
                "config_sha256": config_sha,
                "conditions_sha256": sha256(conditions_path),
                "world_size": world,
                "gpu_family": properties.name,
                "trainable_parameters": trainable_count,
                "initial_probe_loss": initial_probe,
                "final_probe_loss": final_probe,
                "probe_loss_ratio": ratio,
                "arm_metrics": arm_metrics,
                "evaluation_rows": result_rows,
                "training_losses": losses,
                "wall_seconds": time.monotonic() - started,
                "protected_foundir_test_accessed": False,
            }
            write_once(output_dir / "result.json", (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode())
            print("FOUNDIR_FIVE_TYPE_FLUX_OVERFIT_COMPLETE " + json.dumps(payload, sort_keys=True), flush=True)
            gate_value = 1 if payload["automatic_gate_pass"] else 0
        else:
            gate_value = 0
        gate_tensor = torch.tensor([gate_value], device=device, dtype=torch.int32)
        dist.broadcast(gate_tensor, src=0)
        dist.barrier()
        return 0 if int(gate_tensor.item()) == 1 else 2
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
