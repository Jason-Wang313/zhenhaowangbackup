#!/usr/bin/env python3
"""Restart-safe validation/test evaluation for the full FLUX.2 Lowlight baseline."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch
from PIL import Image
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party/flux2/src"))

from lowlight_restoration.authorization import sha256, validate_run_authorization
from lowlight_restoration.config import load_train_config
from lowlight_restoration.data import LowLightPairDataset, load_pair_manifest
from lowlight_restoration.lora import inject_lora, load_lora_checkpoint
from smoke.evaluate_lowlight_overfit import (
    make_contact_sheets,
    metrics,
    require_finite_numbers,
    save_image_once_or_verify,
    tensor_to_image,
)


def atomic_write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, indent=2, sort_keys=True) + "\n"
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(encoded)
    os.replace(temporary, path)


def workspace_artifact(relative: object, *, label: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise RuntimeError(f"{label} must be a non-empty path")
    path = (ROOT / relative).resolve()
    try:
        path.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise RuntimeError(f"{label} escapes workspace: {relative}") from exc
    return path


def validate_reusable_record(
    record: dict[str, object],
    *,
    pair_id: str,
    project_level: str,
    index: int,
    split: str,
    checkpoint_sha256: str,
    config_sha256: str,
    guidance: float,
) -> None:
    expected = {
        "schema_version": 1,
        "pair_id": pair_id,
        "project_level": project_level,
        "index": index,
        "split": split,
        "checkpoint_sha256": checkpoint_sha256,
        "config_sha256": config_sha256,
        "guidance": guidance,
    }
    for key, value in expected.items():
        if record.get(key) != value:
            raise RuntimeError(f"reusable record differs at {key}: {pair_id}")
    require_finite_numbers(record)
    restored_path = workspace_artifact(record.get("restored_path"), label="restored_path")
    if not restored_path.is_file() or sha256(restored_path) != record.get("restored_sha256"):
        raise RuntimeError(f"reusable restored image hash mismatch: {pair_id}")


def lpips_value(metric, prediction: torch.Tensor, target: torch.Tensor) -> float:
    with torch.no_grad():
        value = float(metric(prediction.add(1).div(2), target.add(1).div(2)))
    metric.reset()
    if not math.isfinite(value):
        raise RuntimeError(f"non-finite LPIPS value: {value}")
    return value


def mean(items: list[dict[str, object]], key: str) -> float:
    return sum(float(item[key]) for item in items) / len(items)


def aggregate_metrics(items: list[dict[str, object]]) -> dict[str, float]:
    return {
        "input_psnr": mean(items, "degraded_psnr"),
        "restored_psnr": mean(items, "restored_psnr"),
        "input_ssim": mean(items, "degraded_ssim"),
        "restored_ssim": mean(items, "restored_ssim"),
        "input_lpips": mean(items, "degraded_lpips"),
        "restored_lpips": mean(items, "restored_lpips"),
    }


def contact_rows(
    dataset: LowLightPairDataset,
    per_image: list[dict[str, object]],
    indices: list[int],
) -> list[tuple[str, Image.Image, Image.Image, Image.Image]]:
    rows: list[tuple[str, Image.Image, Image.Image, Image.Image]] = []
    by_index = {int(result["index"]): result for result in per_image}
    if len(by_index) != len(per_image):
        raise RuntimeError("per-image evidence contains duplicate indices")
    for index in indices:
        item = dataset[index]
        try:
            result = by_index[index]
        except KeyError as exc:
            raise RuntimeError(f"missing per-image evidence for index {index}") from exc
        restored_path = workspace_artifact(result["restored_path"], label="restored_path")
        with Image.open(restored_path) as image:
            restored = image.convert("RGB").copy()
        rows.append(
            (
                str(item["pair_id"]),
                tensor_to_image(item["degraded"]),
                restored,
                tensor_to_image(item["clean"]),
            )
        )
    return rows


def validate_reusable_aggregate(
    aggregate: dict[str, object],
    *,
    split: str,
    evaluation_name: str,
    checkpoint_sha256: str,
    config_sha256: str,
    guidance: float,
    samples: int,
    pair_ids: list[str],
    project_levels: list[str],
    result_dir: Path,
) -> None:
    expected = {
        "schema_version": 1,
        "split": split,
        "evaluation_name": evaluation_name,
        "checkpoint_sha256": checkpoint_sha256,
        "config_sha256": config_sha256,
        "guidance": guidance,
        "samples": samples,
        "automatic_sanity_pass": True,
    }
    for key, value in expected.items():
        if aggregate.get(key) != value:
            raise RuntimeError(f"existing aggregate differs at {key}")
    require_finite_numbers(aggregate)
    records = aggregate.get("per_image")
    if not isinstance(records, list) or len(records) != samples:
        raise RuntimeError("existing aggregate has incomplete per-image evidence")
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise RuntimeError(f"existing aggregate record {index} is not an object")
        validate_reusable_record(
            record,
            pair_id=pair_ids[index],
            project_level=project_levels[index],
            index=index,
            split=split,
            checkpoint_sha256=checkpoint_sha256,
            config_sha256=config_sha256,
            guidance=guidance,
        )
        sidecar = result_dir / f"{index:04d}_{pair_ids[index]}.json"
        if not sidecar.is_file() or json.loads(sidecar.read_text()) != record:
            raise RuntimeError(f"reusable sidecar missing or differs: {pair_ids[index]}")
    contact_sheets = aggregate.get("contact_sheets")
    contact_hashes = aggregate.get("contact_sheet_sha256")
    if (
        not isinstance(contact_sheets, list)
        or not contact_sheets
        or not all(isinstance(relative, str) for relative in contact_sheets)
    ):
        raise RuntimeError("existing aggregate has no contact sheets")
    if not isinstance(contact_hashes, dict) or set(contact_hashes) != set(contact_sheets):
        raise RuntimeError("existing aggregate has incomplete contact-sheet hashes")
    for relative in contact_sheets:
        path = workspace_artifact(relative, label="contact_sheet")
        expected_hash = contact_hashes[relative]
        if (
            not isinstance(expected_hash, str)
            or re.fullmatch(r"[0-9a-f]{64}", expected_hash) is None
            or not path.is_file()
            or sha256(path) != expected_hash
        ):
            raise RuntimeError(f"missing or changed contact sheet: {relative}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--evaluation-name", required=True)
    args = parser.parse_args()

    config = load_train_config(args.config)
    try:
        scope = validate_run_authorization(config)
    except RuntimeError as exc:
        raise SystemExit(f"RUN_NOT_AUTHORIZED {exc}") from exc
    if scope != "full_baseline":
        raise SystemExit("FULL_BASELINE_CONFIG_REQUIRED")
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.evaluation_name) is None:
        raise SystemExit("BAD_EVALUATION_NAME")
    if not args.evaluation_name.startswith(args.split):
        raise SystemExit("EVALUATION_NAME_MUST_START_WITH_SPLIT")

    os.environ.update(
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        TORCH_HOME=str(ROOT / "models/torch"),
        KLEIN_4B_BASE_MODEL_PATH=str(config.workspace_path("model", "flow_weights")),
        AE_MODEL_PATH=str(config.workspace_path("model", "autoencoder_weights")),
        QWEN3_MODEL_PATH=str(config.workspace_path("model", "text_encoder")),
    )
    if not torch.cuda.is_available():
        raise SystemExit("CUDA_NOT_AVAILABLE")
    device = torch.device("cuda")
    inference = config.section("inference")
    guidance = float(inference["guidance"])
    sample_count = int(inference["num_samples"])
    if guidance != 1.0 or sample_count != 1000:
        raise SystemExit("FULL_EVALUATION_REQUIRES_GUIDANCE1_AND_1000_SAMPLES")

    output_dir = config.workspace_path("runtime", "output_dir")
    summary_path = output_dir / "training_summary.json"
    if not summary_path.is_file():
        raise SystemExit("MISSING_FULL_TRAINING_SUMMARY")
    summary = json.loads(summary_path.read_text())
    if (
        summary.get("config_sha256") != config.sha256
        or summary.get("completed_steps") != int(config.section("training")["max_steps"])
        or summary.get("loss_decreased") is not True
    ):
        raise SystemExit("FULL_TRAINING_SUMMARY_INVALID")
    checkpoint = workspace_artifact(summary.get("checkpoint"), label="checkpoint")
    checkpoint_sha = summary.get("checkpoint_sha256")
    if not checkpoint.is_file() or not isinstance(checkpoint_sha, str) or sha256(checkpoint) != checkpoint_sha:
        raise SystemExit("FULL_CHECKPOINT_SHA256_MISMATCH")

    records = load_pair_manifest(config.workspace_path("data", "manifest"))
    split_records = [record for record in records if record.split == args.split]
    if len(split_records) != sample_count or len({record.source_id for record in split_records}) != 200:
        raise SystemExit("BAD_FIXED_EVALUATION_SPLIT")
    levels = Counter(record.project_level for record in split_records)
    if levels != Counter({f"L{level}": 200 for level in range(1, 6)}):
        raise SystemExit(f"BAD_EVALUATION_LEVEL_BALANCE {levels}")
    dataset = LowLightPairDataset(
        records,
        split=args.split,
        crop_size=int(config.section("data")["crop_size"]),
        seed=int(config.section("training")["seed"]),
    )

    eval_dir = output_dir / args.evaluation_name
    result_dir = eval_dir / "records"
    metrics_path = eval_dir / "metrics.json"
    if metrics_path.exists():
        aggregate = json.loads(metrics_path.read_text())
        validate_reusable_aggregate(
            aggregate,
            split=args.split,
            evaluation_name=args.evaluation_name,
            checkpoint_sha256=checkpoint_sha,
            config_sha256=config.sha256,
            guidance=guidance,
            samples=sample_count,
            pair_ids=[record.pair_id for record in dataset.records],
            project_levels=[record.project_level for record in dataset.records],
            result_dir=result_dir,
        )
        print("LOWLIGHT_SPLIT_EVALUATION_REUSED " + json.dumps(aggregate, sort_keys=True))
        return
    result_dir.mkdir(parents=True, exist_ok=True)

    from flux2.sampling import batched_prc_img, denoise_cfg, get_schedule, scatter_ids
    from flux2.util import load_ae, load_flow_model
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

    prompt_tensors = load_file(str(output_dir / "prompt_context.safetensors"), device="cpu")
    context = torch.cat([prompt_tensors["empty_context"], prompt_tensors["context"]], dim=0).to(device)
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
    lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(device).eval()

    per_image: list[dict[str, object]] = []
    reference_time = int(config.section("training")["reference_time_coordinate"])
    for index in range(sample_count):
        item = dataset[index]
        pair_id = str(item["pair_id"])
        result_path = result_dir / f"{index:04d}_{pair_id}.json"
        if result_path.exists():
            result = json.loads(result_path.read_text())
            validate_reusable_record(
                result,
                pair_id=pair_id,
                project_level=str(item["project_level"]),
                index=index,
                split=args.split,
                checkpoint_sha256=checkpoint_sha,
                config_sha256=config.sha256,
                guidance=guidance,
            )
            per_image.append(result)
            print(f"EVAL_REUSED index={index} pair_id={pair_id}", flush=True)
            continue

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
        restored_path = eval_dir / f"{pair_id}_restored.png"
        save_image_once_or_verify(restored_path, restored_image)
        degraded_psnr, degraded_ssim = metrics(degraded_image, clean_image)
        restored_psnr, restored_ssim = metrics(restored_image, clean_image)
        result = {
            "schema_version": 1,
            "index": index,
            "pair_id": pair_id,
            "project_level": item["project_level"],
            "split": args.split,
            "crop_xy": item["crop_xy"].tolist(),
            "checkpoint_sha256": checkpoint_sha,
            "config_sha256": config.sha256,
            "guidance": guidance,
            "seed": int(inference["seed"]) + index,
            "degraded_psnr": degraded_psnr,
            "restored_psnr": restored_psnr,
            "degraded_ssim": degraded_ssim,
            "restored_ssim": restored_ssim,
            "degraded_lpips": lpips_value(lpips, degraded_tensor, clean_tensor),
            "restored_lpips": lpips_value(lpips, restored_tensor, clean_tensor),
            "restored_path": str(restored_path.relative_to(ROOT)),
            "restored_sha256": sha256(restored_path),
        }
        require_finite_numbers(result)
        atomic_write_json(result_path, result)
        per_image.append(result)
        print("EVAL " + json.dumps(result, sort_keys=True), flush=True)

    by_level: dict[str, list[dict[str, object]]] = defaultdict(list)
    for result in per_image:
        by_level[str(result["project_level"])].append(result)
    degraded_only = [result for result in per_image if result["project_level"] != "L1"]
    clean_identity = by_level["L1"]
    overall = aggregate_metrics(per_image)
    degraded_aggregate = aggregate_metrics(degraded_only)
    l1_aggregate = aggregate_metrics(clean_identity)
    level_aggregates = {level: aggregate_metrics(items) for level, items in sorted(by_level.items())}

    fixed_indices = [
        int(next(item for item in per_image if item["project_level"] == f"L{level}")["index"])
        for level in range(1, 6)
    ]
    worst_psnr = [int(item["index"]) for item in sorted(degraded_only, key=lambda item: float(item["restored_psnr"]))[:8]]
    worst_l1 = [int(item["index"]) for item in sorted(clean_identity, key=lambda item: float(item["restored_ssim"]))[:8]]
    contact_sheets: list[str] = []
    for prefix, indices in (
        ("contact_stratified", fixed_indices),
        ("contact_worst_psnr", worst_psnr),
        ("contact_worst_l1_ssim", worst_l1),
    ):
        contact_sheets.extend(
            make_contact_sheets(contact_rows(dataset, per_image, indices), eval_dir, prefix=prefix)
        )

    aggregate: dict[str, object] = {
        "schema_version": 1,
        "split": args.split,
        "evaluation_name": args.evaluation_name,
        "checkpoint": str(checkpoint.relative_to(ROOT)),
        "checkpoint_sha256": checkpoint_sha,
        "config_sha256": config.sha256,
        "git_revision": summary.get("git_revision"),
        "flux_revision": summary.get("flux_revision"),
        "guidance": guidance,
        "samples": sample_count,
        "sources": 200,
        "level_counts": dict(levels),
        "overall": overall,
        "degraded_l2_l5": degraded_aggregate,
        "l1_identity": l1_aggregate,
        "by_level": level_aggregates,
        "loss_decreased": True,
        "psnr_improved": degraded_aggregate["restored_psnr"] > degraded_aggregate["input_psnr"],
        "ssim_improved": degraded_aggregate["restored_ssim"] > degraded_aggregate["input_ssim"],
        "lpips_improved": degraded_aggregate["restored_lpips"] < degraded_aggregate["input_lpips"],
        "l1_identity_floor_pass": l1_aggregate["restored_ssim"] >= 0.70,
        "contact_sheets": contact_sheets,
        "contact_sheet_sha256": {
            relative: sha256(workspace_artifact(relative, label="contact_sheet"))
            for relative in contact_sheets
        },
        "per_image": per_image,
    }
    aggregate["automatic_sanity_pass"] = bool(
        aggregate["loss_decreased"]
        and aggregate["psnr_improved"]
        and aggregate["ssim_improved"]
        and aggregate["lpips_improved"]
        and aggregate["l1_identity_floor_pass"]
    )
    require_finite_numbers(aggregate)
    atomic_write_json(metrics_path, aggregate)
    print("LOWLIGHT_SPLIT_AUTOMATIC_SANITY " + json.dumps(aggregate, sort_keys=True), flush=True)
    if not aggregate["automatic_sanity_pass"]:
        raise SystemExit("LOWLIGHT_SPLIT_AUTOMATIC_SANITY_FAILED")


if __name__ == "__main__":
    main()
