#!/usr/bin/env python3
"""Evaluate the trained 16-pair LoRA and build an identity-review contact sheet."""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import re
import sys
import hashlib
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from safetensors.torch import load_file
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party/flux2/src"))

from lowlight_restoration.config import load_train_config
from lowlight_restoration.data import LowLightPairDataset, load_pair_manifest
from lowlight_restoration.lora import inject_lora, load_lora_checkpoint


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    array = tensor.detach().float().clamp(-1, 1).add(1).mul(127.5).byte()
    return Image.fromarray(array.permute(1, 2, 0).cpu().numpy(), mode="RGB")


def metrics(prediction: Image.Image, target: Image.Image) -> tuple[float, float]:
    pred = np.asarray(prediction, dtype=np.float32) / 255.0
    clean = np.asarray(target, dtype=np.float32) / 255.0
    psnr = float(peak_signal_noise_ratio(clean, pred, data_range=1.0))
    if not math.isfinite(psnr):
        psnr = 100.0
    return psnr, float(structural_similarity(clean, pred, channel_axis=2, data_range=1.0))


def save_image_once_or_verify(path: Path, image: Image.Image, *, quality: int | None = None) -> None:
    buffer = io.BytesIO()
    if quality is None:
        image.save(buffer, format="PNG")
    else:
        image.save(buffer, format="JPEG", quality=quality)
    encoded = buffer.getvalue()
    if path.exists():
        if path.read_bytes() != encoded:
            raise RuntimeError(f"existing evaluation image differs; refusing overwrite: {path}")
        return
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(encoded)
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_finite_numbers(value: object, location: str = "root") -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise RuntimeError(f"non-finite metric at {location}: {value}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            require_finite_numbers(item, f"{location}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            require_finite_numbers(item, f"{location}.{key}")


def validate_reusable_evaluation(
    existing: dict[str, object],
    *,
    checkpoint: Path,
    config_sha256: str,
    samples: int,
    guidance: float,
    evaluation_name: str,
) -> None:
    expected_checkpoint = str(checkpoint.relative_to(ROOT))
    if (
        existing.get("checkpoint") != expected_checkpoint
        or existing.get("checkpoint_sha256") != sha256(checkpoint)
        or existing.get("config_sha256") != config_sha256
        or existing.get("samples") != samples
        or existing.get("guidance") != guidance
        or existing.get("evaluation_name") != evaluation_name
        or "automatic_gate_pass" not in existing
    ):
        raise RuntimeError("existing evaluation metadata does not match this run")
    require_finite_numbers(existing)
    artifact_paths = [
        str(item["restored_path"])
        for item in existing.get("per_image", [])
        if isinstance(item, dict) and "restored_path" in item
    ]
    artifact_paths.extend(str(path) for path in existing.get("contact_sheets", []))
    if len(existing.get("per_image", [])) != samples or not artifact_paths:
        raise RuntimeError("existing evaluation has incomplete per-image evidence")
    for relative in artifact_paths:
        path = (ROOT / relative).resolve()
        try:
            path.relative_to(ROOT.resolve())
        except ValueError as exc:
            raise RuntimeError(f"evaluation artifact escapes workspace: {relative}") from exc
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"missing evaluation artifact: {relative}")
    if existing["automatic_gate_pass"] is not True:
        raise RuntimeError("existing evaluation records a failed automatic gate")


def labeled_thumbnail(image: Image.Image, label: str, size: int = 256) -> Image.Image:
    thumb = image.copy()
    thumb.thumbnail((size, size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size + 24), "white")
    canvas.paste(thumb, ((size - thumb.width) // 2, 24 + (size - thumb.height) // 2))
    ImageDraw.Draw(canvas).text((4, 4), label, fill="black")
    return canvas


def make_contact_sheets(rows: list[tuple[str, Image.Image, Image.Image, Image.Image]], output_dir: Path) -> list[str]:
    paths: list[str] = []
    rows_per_sheet = 8
    cell = 256
    for sheet_index in range(math.ceil(len(rows) / rows_per_sheet)):
        subset = rows[sheet_index * rows_per_sheet : (sheet_index + 1) * rows_per_sheet]
        sheet = Image.new("RGB", (cell * 3, (cell + 24) * len(subset)), "white")
        for row_index, (pair_id, degraded, restored, clean) in enumerate(subset):
            for column, (name, image) in enumerate(
                (("degraded", degraded), ("restored", restored), ("clean", clean))
            ):
                tile = labeled_thumbnail(image, f"{pair_id} | {name}", size=cell)
                sheet.paste(tile, (column * cell, row_index * (cell + 24)))
        path = output_dir / f"contact_sheet_{sheet_index + 1}.jpg"
        save_image_once_or_verify(path, sheet, quality=92)
        paths.append(str(path.relative_to(ROOT)))
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--guidance",
        type=float,
        help="Diagnostic CFG override; 1.0 is the conditional-only path used in training.",
    )
    parser.add_argument(
        "--evaluation-name",
        default="evaluation",
        help="New output subdirectory name; existing differing artifacts are never overwritten.",
    )
    args = parser.parse_args()
    config = load_train_config(args.config)
    if config.raw.get("status") != "approved_for_16_pair_gpu_smoke_only":
        raise SystemExit("ONLY_THE_16_PAIR_SMOKE_IS_CURRENTLY_AUTHORIZED")
    os.environ.update(
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        KLEIN_4B_BASE_MODEL_PATH=str(config.workspace_path("model", "flow_weights")),
        AE_MODEL_PATH=str(config.workspace_path("model", "autoencoder_weights")),
        QWEN3_MODEL_PATH=str(config.workspace_path("model", "text_encoder")),
    )
    if not torch.cuda.is_available():
        raise SystemExit("CUDA_NOT_AVAILABLE")
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.evaluation_name) is None:
        raise SystemExit("BAD_EVALUATION_NAME")
    device = torch.device("cuda")
    inference = config.section("inference")
    guidance = float(inference["guidance"] if args.guidance is None else args.guidance)
    if not math.isfinite(guidance) or guidance <= 0:
        raise SystemExit("BAD_GUIDANCE")
    sample_count = int(inference["num_samples"])
    output_dir = config.workspace_path("runtime", "output_dir")
    training_summary_path = output_dir / "training_summary.json"
    if not training_summary_path.is_file():
        raise SystemExit("MISSING_TRAINING_SUMMARY")
    training_summary = json.loads(training_summary_path.read_text())
    checkpoint = ROOT / training_summary["checkpoint"]
    if not checkpoint.is_file():
        raise SystemExit("MISSING_TRAINING_CHECKPOINT")
    expected_checkpoint_sha = training_summary.get("checkpoint_sha256")
    if expected_checkpoint_sha != sha256(checkpoint):
        raise SystemExit("TRAINING_CHECKPOINT_SHA256_MISMATCH")
    eval_dir = output_dir / args.evaluation_name
    metrics_path = eval_dir / "metrics.json"
    if metrics_path.exists():
        existing = json.loads(metrics_path.read_text())
        try:
            validate_reusable_evaluation(
                existing,
                checkpoint=checkpoint,
                config_sha256=config.sha256,
                samples=sample_count,
                guidance=guidance,
                evaluation_name=args.evaluation_name,
            )
        except RuntimeError as exc:
            raise SystemExit(f"EXISTING_EVALUATION_INVALID {exc}") from exc
        print("LOWLIGHT_OVERFIT_EVALUATION_REUSED " + json.dumps(existing, sort_keys=True))
        return
    eval_dir.mkdir(parents=True, exist_ok=True)

    records = load_pair_manifest(config.workspace_path("data", "manifest"))
    data = config.section("data")
    dataset = LowLightPairDataset(
        records,
        split=str(data["split"]),
        crop_size=int(data["crop_size"]),
        seed=int(config.section("training")["seed"]),
    )
    if sample_count != len(dataset):
        raise SystemExit(f"OVERFIT_EVAL_MUST_COVER_ALL_RECORDS {sample_count} != {len(dataset)}")

    from flux2.sampling import batched_prc_img, denoise_cfg, get_schedule, scatter_ids
    from flux2.util import load_ae, load_flow_model

    prompt_tensors = load_file(str(output_dir / "prompt_context.safetensors"), device="cpu")
    context = torch.cat(
        [prompt_tensors["empty_context"], prompt_tensors["context"]], dim=0
    ).to(device)
    context_ids = torch.cat(
        [prompt_tensors["empty_context_ids"], prompt_tensors["context_ids"]], dim=0
    ).to(device)
    model_name = str(config.section("model")["name"])
    autoencoder = load_ae(model_name, device=device).eval().requires_grad_(False)
    flow_model = load_flow_model(model_name, device=device).eval().requires_grad_(False)
    lora = config.section("lora")
    names = inject_lora(
        flow_model,
        list(lora["target_patterns"]),
        rank=int(lora["rank"]),
        alpha=float(lora["alpha"]),
        dropout=float(lora["dropout"]),
    )
    if len(names) != int(lora["expected_module_count"]):
        raise SystemExit("BAD_LORA_MODULE_COUNT")
    load_lora_checkpoint(flow_model, checkpoint)

    per_image: list[dict[str, object]] = []
    contact_rows: list[tuple[str, Image.Image, Image.Image, Image.Image]] = []
    reference_time = int(config.section("training")["reference_time_coordinate"])
    for index in range(sample_count):
        item = dataset[index]
        clean_tensor = item["clean"].unsqueeze(0).to(device=device, dtype=torch.float32)
        degraded_tensor = item["degraded"].unsqueeze(0).to(device=device, dtype=torch.float32)
        with torch.no_grad():
            reference_latent = autoencoder.encode(degraded_tensor).to(torch.bfloat16)
            reference_tokens, reference_ids = batched_prc_img(reference_latent)
            reference_ids = reference_ids.clone()
            reference_ids[..., 0] = reference_time
            generator = torch.Generator(device=device).manual_seed(int(inference["seed"]) + index)
            noise = torch.randn(
                reference_latent.shape,
                generator=generator,
                device=device,
                dtype=torch.bfloat16,
            )
            image_tokens, image_ids = batched_prc_img(noise)
            schedule = get_schedule(int(inference["num_steps"]), image_tokens.shape[1])
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                restored_tokens = denoise_cfg(
                    flow_model,
                    image_tokens,
                    image_ids,
                    context,
                    context_ids,
                    timesteps=schedule,
                    guidance=guidance,
                    img_cond_seq=reference_tokens,
                    img_cond_seq_ids=reference_ids,
                )
            restored_latent = torch.cat(scatter_ids(restored_tokens, image_ids)).squeeze(2)
            restored_tensor = autoencoder.decode(restored_latent).float().clamp(-1, 1)

        clean_image = tensor_to_image(clean_tensor[0])
        degraded_image = tensor_to_image(degraded_tensor[0])
        restored_image = tensor_to_image(restored_tensor[0])
        pair_id = str(item["pair_id"])
        restored_path = eval_dir / f"{pair_id}_restored.png"
        save_image_once_or_verify(restored_path, restored_image)
        degraded_psnr, degraded_ssim = metrics(degraded_image, clean_image)
        restored_psnr, restored_ssim = metrics(restored_image, clean_image)
        result = {
            "pair_id": pair_id,
            "project_level": item["project_level"],
            "crop_xy": item["crop_xy"].tolist(),
            "degraded_psnr": degraded_psnr,
            "restored_psnr": restored_psnr,
            "degraded_ssim": degraded_ssim,
            "restored_ssim": restored_ssim,
            "restored_path": str(restored_path.relative_to(ROOT)),
        }
        per_image.append(result)
        contact_rows.append((pair_id, degraded_image, restored_image, clean_image))
        print("EVAL " + json.dumps(result, sort_keys=True), flush=True)

    mean = lambda items, key: sum(float(item[key]) for item in items) / len(items)
    degraded_only = [item for item in per_image if item["project_level"] != "L1"]
    clean_identity = [item for item in per_image if item["project_level"] == "L1"]
    if not degraded_only or not clean_identity:
        raise RuntimeError("overfit evaluation requires both L1 and L2-L5 records")
    aggregate = {
        "schema_version": 1,
        "checkpoint": str(checkpoint.relative_to(ROOT)),
        "checkpoint_sha256": expected_checkpoint_sha,
        "config_sha256": config.sha256,
        "samples": len(per_image),
        "guidance": guidance,
        "evaluation_name": args.evaluation_name,
        "guidance_override": args.guidance is not None,
        "degraded_l2_l5_mean_psnr": mean(degraded_only, "degraded_psnr"),
        "restored_l2_l5_mean_psnr": mean(degraded_only, "restored_psnr"),
        "degraded_l2_l5_mean_ssim": mean(degraded_only, "degraded_ssim"),
        "restored_l2_l5_mean_ssim": mean(degraded_only, "restored_ssim"),
        "restored_l1_mean_psnr": mean(clean_identity, "restored_psnr"),
        "restored_l1_mean_ssim": mean(clean_identity, "restored_ssim"),
        "loss_decreased": bool(training_summary["loss_decreased"]),
        "psnr_improved": mean(degraded_only, "restored_psnr") > mean(degraded_only, "degraded_psnr"),
        "ssim_improved": mean(degraded_only, "restored_ssim") > mean(degraded_only, "degraded_ssim"),
        "l1_identity_floor_pass": mean(clean_identity, "restored_ssim") >= 0.70,
        "per_image": per_image,
    }
    aggregate["automatic_gate_pass"] = bool(
        aggregate["loss_decreased"]
        and aggregate["psnr_improved"]
        and aggregate["ssim_improved"]
        and aggregate["l1_identity_floor_pass"]
    )
    aggregate["contact_sheets"] = make_contact_sheets(contact_rows, eval_dir)
    require_finite_numbers(aggregate)
    temporary = metrics_path.with_name(f".{metrics_path.name}.tmp")
    temporary.write_text(json.dumps(aggregate, indent=2) + "\n")
    os.replace(temporary, metrics_path)
    print("LOWLIGHT_OVERFIT_AUTOMATIC_GATE " + json.dumps(aggregate, sort_keys=True), flush=True)
    if not aggregate["automatic_gate_pass"]:
        raise SystemExit("LOWLIGHT_OVERFIT_AUTOMATIC_GATE_FAILED")
    print("LOWLIGHT_OVERFIT_AWAITING_HUMAN_IDENTITY_REVIEW")


if __name__ == "__main__":
    main()
