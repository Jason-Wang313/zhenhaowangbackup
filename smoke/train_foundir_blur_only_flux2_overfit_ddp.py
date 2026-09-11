#!/usr/bin/env python3
"""Advisor-authorized eight-GPU FLUX.2 DiT-LoRA-only Blur overfit gate."""

from __future__ import annotations

import argparse
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


def validate_config(config: Mapping[str, object]) -> None:
    if config.get("status") != "advisor_authorized_blur_only_tiny_overfit":
        raise RuntimeError("config is not authorized for the Blur-only tiny overfit")
    scope = config["scope"]
    required_false = (
        "qwen_degradation_conditioning", "mixed_degradations", "pixel_refiner",
        "official_foundir_test_accessed", "sealed_internal_accessed",
    )
    if scope.get("category") != "01Blur" or any(scope.get(key) is not False for key in required_false):
        raise RuntimeError("Blur-only scope contract changed")
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
    if int(distributed["required_world_size"]) != 8:
        raise RuntimeError("this gate requires exactly eight workers")
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
    if evaluations != [0, *checkpoints]:
        raise RuntimeError("every candidate checkpoint plus initialization must be visualized")


def load_manifest(config: Mapping[str, object], *, verify_images: bool) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    data = config["data"]
    manifest = ROOT / str(data["frozen_manifest"])
    metadata_path = ROOT / str(data["frozen_manifest_metadata"])
    metadata = json.loads(metadata_path.read_text())
    config_sha = sha256(Path(config["_config_path"]))
    if (
        metadata.get("config_sha256") != config_sha
        or metadata.get("frozen_manifest_sha256") != sha256(manifest)
        or metadata.get("protected_foundir_test_accessed") is not False
        or metadata.get("sealed_internal_accessed") is not False
    ):
        raise RuntimeError("frozen Blur-only manifest identity changed")
    with manifest.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    train = [row for row in rows if row["role"] == "tiny_overfit_train"]
    validation = [row for row in rows if row["role"] == "future_blur_validation"]
    if len(train) != int(data["tiny_overfit_records"]) or len(validation) != int(data["frozen_validation_records"]):
        raise RuntimeError("frozen Blur-only manifest cardinality changed")
    for role, selected, split in (
        ("train", train, data["tiny_overfit_split"]),
        ("validation", validation, data["frozen_validation_split"]),
    ):
        if any(row["category"] != "01Blur" or row["composition_size"] != "1" or row["split"] != split for row in selected):
            raise RuntimeError(f"{role} contains a non-isolated-Blur record")
        if len({row["source_group"] for row in selected}) != len(selected):
            raise RuntimeError(f"{role} source groups are not unique")
    if {row["source_group"] for row in train} & {row["source_group"] for row in validation}:
        raise RuntimeError("Blur train/validation source-group leakage")
    if verify_images:
        for row in rows:
            for field, hash_field in (("lq_path", "lq_sha256"), ("gt_path", "gt_sha256")):
                path = ROOT / row[field]
                if not path.is_file() or sha256(path) != row[hash_field]:
                    raise RuntimeError(f"Blur pair identity changed: {row['record_id']} {field}")
    return train, validation


def load_crop(row: Mapping[str, str], crop_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    x, y = int(row["crop_x"]), int(row["crop_y"])
    tensors = []
    for field in ("lq_path", "gt_path"):
        with Image.open(ROOT / row[field]) as image:
            rgb = image.convert("RGB")
            if rgb.size != (int(row["width"]), int(row["height"])):
                raise RuntimeError(f"Blur pair dimensions changed: {row['record_id']}")
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


def evaluate_snapshot(model, autoencoder, records, cfg_context, cfg_ids, pack_image, metric, *, step: int, limit: int, config, output_dir: Path, rank: int, world: int, device: torch.device) -> dict[str, object] | None:
    selected = records[:limit]
    local = []
    snapshot_dir = output_dir / "training_triptychs" / f"step_{step:06d}"
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
            "record_id": record["record_id"], "step": step,
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
    print("FOUNDIR_BLUR_ONLY_SNAPSHOT " + json.dumps({key: value for key, value in aggregate.items() if key != "rows"}, sort_keys=True), flush=True)
    return aggregate


def preflight(config_path: Path) -> None:
    config = json.loads(config_path.read_text())
    config["_config_path"] = str(config_path.resolve())
    validate_config(config)
    train, validation = load_manifest(config, verify_images=True)
    for key, hash_key in (("flow_weights", "flow_weights_sha256"), ("autoencoder_weights", "autoencoder_weights_sha256")):
        path = ROOT / config["model"][key]
        if sha256(path) != config["model"][hash_key]:
            raise RuntimeError(f"model artifact identity changed: {path}")
    crop = int(config["data"]["crop_size"])
    for row in [*train, *validation]:
        load_crop(row, crop)
    print(json.dumps({
        "status": "FOUNDIR_BLUR_ONLY_PREFLIGHT_PASS", "config_sha256": sha256(config_path),
        "tiny_overfit_records": len(train), "future_validation_records": len(validation),
        "future_validation_not_inferred": True, "protected_foundir_test_accessed": False,
        "sealed_internal_accessed": False,
    }, sort_keys=True))


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
    train_rows, _ = load_manifest(config, verify_images=False)
    records = prepare_records(train_rows, int(config["data"]["crop_size"]))
    output_dir = ROOT / config["runtime"]["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "result.json").exists():
        raise RuntimeError("Blur-only sanity result already exists; refusing rerun")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend=str(config["distributed"]["backend"]), device_id=device)
    started = time.monotonic()
    try:
        rank, world = dist.get_rank(), dist.get_world_size()
        if world != 8 or torch.cuda.device_count() != 8:
            raise RuntimeError("Blur-only sanity gate requires exactly eight visible GPUs")
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
        if start_step == 0:
            evaluate_snapshot(
                wrapped, autoencoder, records, cfg_context, cfg_ids, batched_prc_img, metric,
                step=0, limit=int(config["inference"]["visual_records"]), config=config,
                output_dir=output_dir, rank=rank, world=world, device=device,
            )
        training = config["training"]
        accumulation = int(training["gradient_accumulation_steps"])
        optimizer.zero_grad(set_to_none=True)
        for step in range(start_step, int(training["max_steps"])):
            accumulated = 0.0
            for micro in range(accumulation):
                position = (step * accumulation + micro) * world + rank
                record = records[scheduled_index(position, len(records), seed)]
                lq = record["lq"].unsqueeze(0).to(device)
                gt = record["gt"].unsqueeze(0).to(device)
                with torch.no_grad():
                    clean, degraded = autoencoder.encode(torch.cat([gt, lq])).chunk(2)
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
                print(f"FOUNDIR_BLUR_ONLY_TRAIN step={completed}/{training['max_steps']} loss={mean_loss:.8f} grad_norm={float(grad_norm):.6f}", flush=True)
            if completed in set(int(value) for value in training["checkpoint_steps"]):
                dist.barrier()
                if rank == 0:
                    checkpoint = output_dir / f"step_{completed:06d}.safetensors"
                    if checkpoint.exists():
                        raise RuntimeError(f"refusing to overwrite checkpoint: {checkpoint}")
                    save_lora_checkpoint(wrapped.module, checkpoint, {
                        "step": completed, "config_sha256": sha256(config_path),
                        "scope": "advisor_authorized_blur_only_dit_lora_only",
                    })
                    with checkpoint.with_suffix(".optimizer.pt").open("xb") as handle:
                        torch.save({"step": completed, "config_sha256": sha256(config_path), "optimizer": optimizer.state_dict()}, handle)
                dist.barrier()
                evaluate_snapshot(
                    wrapped, autoencoder, records, cfg_context, cfg_ids, batched_prc_img, metric,
                    step=completed, limit=int(config["inference"]["visual_records"]), config=config,
                    output_dir=output_dir, rank=rank, world=world, device=device,
                )
        final_probe = probe_loss(
            wrapped, autoencoder, records[rank % len(records)], context, context_ids, batched_prc_img,
            seed=seed + rank * 1009, device=device,
        )
        final_metrics = evaluate_snapshot(
            wrapped, autoencoder, records, cfg_context, cfg_ids, batched_prc_img, metric,
            step=int(training["max_steps"]), limit=len(records), config=config,
            output_dir=output_dir / "final_all_16", rank=rank, world=world, device=device,
        )
        if rank == 0:
            assert final_metrics is not None
            gate = config["gate"]
            loss_ratio = final_probe / initial_probe
            checks = {
                "probe_loss_ratio": loss_ratio <= float(gate["maximum_probe_loss_ratio"]),
                "mean_psnr_gain_db": final_metrics["mean_psnr_gain_db"] >= float(gate["minimum_mean_psnr_gain_db"]),
                "mean_lpips_improvement": final_metrics["mean_lpips_improvement"] >= float(gate["minimum_mean_lpips_improvement"]),
                "psnr_improved_fraction": final_metrics["psnr_improved_fraction"] >= float(gate["minimum_psnr_improved_fraction"]),
                "lpips_improved_fraction": final_metrics["lpips_improved_fraction"] >= float(gate["minimum_lpips_improved_fraction"]),
            }
            automatic_pass = all(checks.values())
            checkpoint = output_dir / f"step_{int(training['max_steps']):06d}.safetensors"
            payload = {
                "schema_version": 1,
                "status": "awaiting_human_visual_review" if automatic_pass else "failed",
                "automatic_gate_pass": automatic_pass, "human_visual_review": "pending",
                "checks": checks, "initial_probe_loss": initial_probe, "final_probe_loss": final_probe,
                "probe_loss_ratio": loss_ratio, "final_metrics": final_metrics,
                "config": config_path.relative_to(ROOT).as_posix(), "config_sha256": sha256(config_path),
                "checkpoint": checkpoint.relative_to(ROOT).as_posix(), "checkpoint_sha256": sha256(checkpoint),
                "flow_objective": training["objective"], "inference_schedule": config["inference"]["schedule"],
                "inference_steps": int(config["inference"]["num_steps"]), "guidance": 1.0,
                "qwen_degradation_conditioning": False, "mixed_degradations": False,
                "pixel_refiner": False, "future_validation_not_inferred": True,
                "protected_foundir_test_accessed": False, "sealed_internal_accessed": False,
                "wall_seconds": time.monotonic() - started,
            }
            write_once(output_dir / "result.json", (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode())
            print("FOUNDIR_BLUR_ONLY_AUTOMATIC_PASS_AWAITING_VISUAL_REVIEW" if automatic_pass else "FOUNDIR_BLUR_ONLY_AUTOMATIC_GATE_FAILED", flush=True)
            exit_value = 0 if automatic_pass else 2
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
