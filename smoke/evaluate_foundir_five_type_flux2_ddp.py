#!/usr/bin/env python3
"""Resume-safe five-type FLUX development or sealed internal evaluation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party/flux2/src"))

from lowlight_restoration.flux_prompt_cache import PromptContextCache, file_sha256  # noqa: E402
from lowlight_restoration.foundir_five_type import FIVE_TYPE_CATEGORIES  # noqa: E402
from lowlight_restoration.foundir_flux2_protocol import (  # noqa: E402
    aggregate_task_metrics,
    evaluation_artifact_stem,
    resolve_workspace_path,
    stable_digest,
    worst_normalized_margin,
)
from lowlight_restoration.lora import inject_lora, load_restoration_checkpoint  # noqa: E402
from lowlight_restoration.pixel_refinement import (  # noqa: E402
    BoundedPixelRefiner,
    pixel_refiner_parameter_count,
)
from lowlight_restoration.paired_development import rgb_structural_similarity  # noqa: E402
from smoke.train_foundir_five_type_flux2_overfit_ddp import psnr, restore  # noqa: E402
from smoke.train_foundir_five_type_flux2_full_ddp import contexts_for, load_rows  # noqa: E402


ARMS = ("qwen", "generic", "shuffled_wrong", "type_oracle")


def write_once(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise RuntimeError(f"refusing to overwrite differing evaluation artifact: {path}")
        return
    with path.open("xb") as handle:
        handle.write(content)


def selected_from_contract(
    config: Mapping[str, object],
    rows: Sequence[Mapping[str, str]],
    *,
    split: str,
    config_sha: str,
    conditions_sha: str,
) -> list[dict[str, str]]:
    evaluation = config["evaluation"]
    contract_csv = ROOT / evaluation["contract_csv"]
    contract_json = ROOT / evaluation["contract_json"]
    payload = json.loads(contract_json.read_text())
    if (
        payload.get("status") != "frozen_before_flux_full_training"
        or payload.get("config_sha256") != config_sha
        or payload.get("conditions_sha256") != conditions_sha
        or payload.get("contract_csv_sha256") != file_sha256(contract_csv)
        or payload.get("sealed_internal_test_accessed") is not False
        or payload.get("protected_foundir_test_accessed") is not False
    ):
        raise RuntimeError("frozen evaluation contract identity changed")
    role = "development" if split == "validation" else "sealed_internal_test"
    with contract_csv.open(newline="") as handle:
        contract = [row for row in csv.DictReader(handle) if row["evaluation_role"] == role]
    by_id = {row["record_id"]: dict(row) for row in rows}
    selected = []
    for item in contract:
        record = by_id.get(item["record_id"])
        if record is None or any(
            record[key] != item[key]
            for key in ("source_group", "category", "composition_size", "lq_sha256", "gt_sha256")
        ):
            raise RuntimeError("evaluation contract record changed")
        selected.append({**record, "evaluation_fold": item["evaluation_fold"]})
    expected = (
        int(evaluation["folds"])
        * int(evaluation["records_per_category_per_fold"])
        * len(FIVE_TYPE_CATEGORIES)
    )
    if len(selected) != expected:
        raise RuntimeError("evaluation contract role cardinality changed")
    return selected


def prompt_for(
    record: Mapping[str, str],
    *,
    arm: str,
    selected: Sequence[Mapping[str, str]],
    index: int,
    seed: int,
) -> str:
    if arm == "qwen":
        return record["qwen_text"]
    if arm == "generic":
        return record["generic_text"]
    if arm == "type_oracle":
        return record["type_oracle_text"]
    if arm != "shuffled_wrong":
        raise ValueError(f"unknown condition arm: {arm}")
    candidates = [
        row
        for row in selected
        if row["source_group"] != record["source_group"] and row["category"] != record["category"]
    ]
    if not candidates:
        raise RuntimeError("no cross-category shuffled condition candidate")
    candidate = candidates[
        int(stable_digest(seed, "shuffle", record["record_id"], index)[:16], 16)
        % len(candidates)
    ]
    return candidate["qwen_text"]


def load_aligned_crop(record: Mapping[str, str], *, crop_size: int, seed: int) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    lq_path = ROOT / record["lq_path"]
    gt_path = ROOT / record["gt_path"]
    if file_sha256(lq_path) != record["lq_sha256"] or file_sha256(gt_path) != record["gt_sha256"]:
        raise RuntimeError(f"evaluation pair changed: {record['record_id']}")
    with Image.open(lq_path) as handle:
        lq = np.asarray(handle.convert("RGB"), dtype=np.uint8).copy()
    with Image.open(gt_path) as handle:
        gt = np.asarray(handle.convert("RGB"), dtype=np.uint8).copy()
    if lq.shape != gt.shape or min(gt.shape[:2]) < crop_size:
        raise RuntimeError(f"evaluation pair is unaligned or undersized: {record['record_id']}")
    digest = bytes.fromhex(stable_digest(seed, "evaluation_crop", record["record_id"]))
    y = int.from_bytes(digest[:8], "big") % (gt.shape[0] - crop_size + 1)
    x = int.from_bytes(digest[8:16], "big") % (gt.shape[1] - crop_size + 1)

    def tensor(image: np.ndarray) -> torch.Tensor:
        crop = image[y : y + crop_size, x : x + crop_size]
        return torch.from_numpy(crop.copy()).permute(2, 0, 1).float().div_(127.5).sub_(1.0)

    return tensor(lq), tensor(gt), [int(x), int(y)]


def quantized_rgb(tensor: torch.Tensor) -> np.ndarray:
    image = ((tensor[0].detach().float().cpu() + 1.0) * 127.5).permute(1, 2, 0).numpy()
    return np.rint(image.clip(0, 255)).astype(np.uint8)


def lpips_value(metric, reference: torch.Tensor, prediction: torch.Tensor, device: torch.device) -> float:
    reference = (reference + 1.0) * 0.5
    prediction = (prediction + 1.0) * 0.5
    with torch.inference_mode():
        return float(metric(prediction.to(device), reference.to(device)).item())


def sidecar_identity(
    *, config_sha: str, checkpoint_sha: str, conditions_sha: str, split: str, arm: str, record: Mapping[str, str]
) -> dict[str, object]:
    return {
        "config_sha256": config_sha,
        "checkpoint_sha256": checkpoint_sha,
        "conditions_sha256": conditions_sha,
        "split": split,
        "arm": arm,
        "record_id": record["record_id"],
        "lq_sha256": record["lq_sha256"],
        "gt_sha256": record["gt_sha256"],
    }


def verify_sidecar(path: Path, identity: Mapping[str, object]) -> dict[str, object]:
    payload = json.loads(path.read_text())
    if payload.get("status") != "complete" or any(payload.get(key) != value for key, value in identity.items()):
        raise RuntimeError(f"evaluation sidecar identity changed: {path}")
    output = resolve_workspace_path(ROOT, str(payload["output_path"]))
    if file_sha256(output) != payload["output_sha256"]:
        raise RuntimeError(f"evaluation output changed: {output}")
    return payload


def aggregate(
    rows: Sequence[Mapping[str, object]],
    *,
    config: Mapping[str, object],
    include_fold_metrics: bool = True,
) -> dict[str, object]:
    tasks = aggregate_task_metrics(rows)
    categories = {}
    folds = {}
    for category in sorted(FIVE_TYPE_CATEGORIES):
        values = [row for row in rows if row["category"] == category]
        categories[category] = {
            "records": len(values),
            "psnr": float(np.mean([row["psnr"] for row in values])),
            "lpips": float(np.mean([row["lpips"] for row in values])),
            "ssim": float(np.mean([row["ssim"] for row in values])),
        }
    if include_fold_metrics:
        for fold in range(int(config["evaluation"]["folds"])):
            values = [row for row in rows if int(row["evaluation_fold"]) == fold]
            folds[str(fold)] = {
                "records": len(values),
                "task_metrics": aggregate_task_metrics(values),
            }
    summary = {
        "records": len(rows),
        "task_metrics": tasks,
        "category_metrics": categories,
        "worst_normalized_margin": worst_normalized_margin(tasks),
    }
    if include_fold_metrics:
        summary["fold_metrics"] = folds
    return summary


def synchronize_rank_zero_finalization(
    finalize: Callable[[], None],
    *,
    rank: int,
    device: torch.device,
    phase: str,
) -> None:
    """Run rank-zero finalization and propagate Python failures without a barrier hang."""
    error: Exception | None = None
    if rank == 0:
        try:
            finalize()
        except Exception as exc:
            error = exc
    failed = torch.tensor([error is not None], dtype=torch.uint8, device=device)
    dist.broadcast(failed, src=0)
    if failed.item():
        if error is not None:
            raise error
        raise RuntimeError(f"rank-zero {phase} failed; inspect the rank-zero traceback")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "internal_test"), required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selection", type=Path)
    args = parser.parse_args()
    args.config = resolve_workspace_path(ROOT, args.config)
    args.checkpoint = resolve_workspace_path(ROOT, args.checkpoint)
    args.output = resolve_workspace_path(ROOT, args.output)
    if args.selection is not None:
        args.selection = resolve_workspace_path(ROOT, args.selection)
    config = json.loads(args.config.read_text())
    admitted_arms = tuple(config["evaluation"].get("condition_arms", ARMS))
    if args.arm not in admitted_arms:
        raise RuntimeError(
            f"condition arm {args.arm!r} is disabled by this experiment contract"
        )
    config_sha = file_sha256(args.config)
    checkpoint_sha = file_sha256(args.checkpoint)
    conditions_path = ROOT / config["full_conditions"]["csv"]
    conditions_sha = file_sha256(conditions_path)
    rows = load_rows(conditions_path, int(config["data"]["expected_five_type_records"]))
    evaluation = config["evaluation"]
    selected = selected_from_contract(
        config,
        rows,
        split=args.split,
        config_sha=config_sha,
        conditions_sha=conditions_sha,
    )
    if args.split == "internal_test":
        if args.arm != "qwen" or args.selection is None:
            raise RuntimeError("sealed internal test permits only the selected Qwen system")
        selection = json.loads(args.selection.read_text())
        if (
            selection.get("status") != "frozen_development_selected_checkpoint"
            or selection.get("checkpoint_sha256") != checkpoint_sha
            or selection.get("config_sha256") != config_sha
            or selection.get("sealed_internal_test_accessed") is not False
        ):
            raise RuntimeError("sealed internal-test checkpoint was not frozen from development")
        canonical_consumption = resolve_workspace_path(
            ROOT,
            Path(config["runtime"]["output_dir"]) / "sealed_internal_consumption.json",
        )
        intended_consumption = {
            "schema_version": 1,
            "status": "sealed_internal_test_consumed",
            "config_sha256": config_sha,
            "checkpoint_sha256": checkpoint_sha,
            "selection_sha256": file_sha256(args.selection),
            "output": args.output.relative_to(ROOT).as_posix(),
            "protected_foundir_test_accessed": False,
        }
        if canonical_consumption.exists():
            if json.loads(canonical_consumption.read_text()) != intended_consumption:
                raise RuntimeError("sealed internal test was already consumed by another invocation")
            existing_result = args.output / "result.json"
            if not existing_result.is_file():
                raise RuntimeError("sealed consumption record exists without its immutable result")
            print("FOUNDIR_SEALED_INTERNAL_ALREADY_COMPLETE", flush=True)
            return

    dist.init_process_group(backend="nccl")
    try:
        rank = dist.get_rank()
        world = dist.get_world_size()
        local_rank = int(os.environ["LOCAL_RANK"])
        if world != 8 or torch.cuda.device_count() != 8:
            raise RuntimeError("five-type evaluation requires exactly eight GPUs")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        properties = torch.cuda.get_device_properties(device)
        if not any(name in properties.name for name in ("H100", "H200")):
            raise RuntimeError(f"unsupported GPU: {properties.name}")
        os.environ.update(
            HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
            KLEIN_4B_BASE_MODEL_PATH=str(ROOT / config["model"]["flow_weights"]),
            AE_MODEL_PATH=str(ROOT / config["model"]["autoencoder_weights"]),
        )
        prompt_cache = PromptContextCache(
            ROOT / config["prompt_cache"]["index"],
            maximum_resident=int(config["prompt_cache"]["maximum_resident_contexts_per_rank"]),
        )
        from flux2.sampling import batched_prc_img
        from flux2.util import load_ae, load_flow_model
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

        autoencoder = load_ae(config["model"]["name"], device=device).eval().requires_grad_(False)
        model = load_flow_model(config["model"]["name"], device=device).eval().requires_grad_(False)
        names = inject_lora(
            model,
            list(config["lora"]["target_patterns"]),
            rank=int(config["lora"]["rank"]),
            alpha=float(config["lora"]["alpha"]),
            dropout=float(config["lora"]["dropout"]),
        )
        if len(names) != int(config["lora"]["expected_module_count"]):
            raise RuntimeError("evaluation LoRA module count changed")
        refinement = config["pixel_refinement"]
        if refinement.get("enabled") is not True:
            raise RuntimeError("evaluation requires the admitted pixel-refinement path")
        pixel_refiner = BoundedPixelRefiner(
            width=int(refinement["width"]),
            blocks=int(refinement["blocks"]),
            maximum_residual_m11=float(refinement["maximum_residual_m11"]),
        ).to(device=device, dtype=torch.bfloat16).eval().requires_grad_(False)
        if pixel_refiner_parameter_count(pixel_refiner) != int(
            refinement["expected_trainable_parameters"]
        ):
            raise RuntimeError("evaluation pixel-refiner parameter count changed")
        load_restoration_checkpoint(model, pixel_refiner, args.checkpoint)
        metric = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(device).eval()
        sidecars = args.output / "per_record"
        images = args.output / "images"
        sidecars.mkdir(parents=True, exist_ok=True)
        images.mkdir(parents=True, exist_ok=True)
        for index in range(rank, len(selected), world):
            record = selected[index]
            identity = sidecar_identity(
                config_sha=config_sha,
                checkpoint_sha=checkpoint_sha,
                conditions_sha=conditions_sha,
                split=args.split,
                arm=args.arm,
                record=record,
            )
            stem = evaluation_artifact_stem(record["record_id"])
            sidecar = sidecars / f"{stem}.json"
            if sidecar.exists():
                verify_sidecar(sidecar, identity)
                continue
            lq, gt, crop_xy = load_aligned_crop(
                record,
                crop_size=int(config["data"]["crop_size"]),
                seed=int(config["data"]["split_seed"]),
            )
            prompt = prompt_for(
                record,
                arm=args.arm,
                selected=selected,
                index=index,
                seed=int(config["data"]["split_seed"]),
            )
            context, context_ids = prompt_cache.get(prompt, device)
            lq_batch = lq.unsqueeze(0).to(device)
            gt_batch = gt.unsqueeze(0).to(device)
            prediction = restore(
                model,
                autoencoder,
                lq_batch,
                context,
                context_ids,
                batched_prc_img,
                seed=int(stable_digest(int(config["flow_training"]["seed"]), record["record_id"], "evaluation")[:16], 16),
                steps=int(evaluation["inference_steps"]),
                noise_scale=float(config["flow_training"]["anchor_noise_scale"]),
                reference_time=int(config["flow_training"]["reference_time_coordinate"]),
                pixel_refiner=pixel_refiner,
            )
            output_array = quantized_rgb(prediction)
            output_path = images / f"{stem}.png"
            if output_path.exists():
                raise RuntimeError(f"orphan evaluation output exists: {output_path}")
            Image.fromarray(output_array).save(output_path)
            with Image.open(output_path) as handle:
                reloaded = np.asarray(handle.convert("RGB"), dtype=np.uint8)
            if not np.array_equal(reloaded, output_array):
                raise RuntimeError("evaluation PNG round trip changed pixels")
            prediction_q = torch.from_numpy(reloaded.copy()).permute(2, 0, 1).float().div_(127.5).sub_(1.0).unsqueeze(0)
            row = {
                "status": "complete",
                **identity,
                "category": record["category"],
                "composition_size": int(record["composition_size"]),
                "evaluation_fold": int(record["evaluation_fold"]),
                "source_group": record["source_group"],
                "crop_xy": crop_xy,
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "input_psnr": psnr(gt_batch, lq_batch),
                "psnr": psnr(gt_batch, prediction_q.to(device)),
                "ssim": rgb_structural_similarity(
                    ((gt.numpy().transpose(1, 2, 0) + 1) * 0.5).clip(0, 1),
                    reloaded.astype(np.float32) / 255.0,
                ),
                "lpips": lpips_value(metric, gt_batch, prediction_q, device),
                "output_path": output_path.relative_to(ROOT).as_posix(),
                "output_sha256": file_sha256(output_path),
            }
            if any(isinstance(value, float) and not math.isfinite(value) for value in row.values()):
                raise RuntimeError("non-finite evaluation metric")
            write_once(sidecar, (json.dumps(row, indent=2, sort_keys=True) + "\n").encode())
            print(
                f"FOUNDIR_FLUX_EVAL rank={rank} split={args.split} arm={args.arm} "
                f"record={index+1}/{len(selected)} psnr={row['psnr']:.4f} lpips={row['lpips']:.6f}",
                flush=True,
            )
        dist.barrier()
        def finalize_result() -> None:
            result_rows = []
            for record in selected:
                identity = sidecar_identity(
                    config_sha=config_sha,
                    checkpoint_sha=checkpoint_sha,
                    conditions_sha=conditions_sha,
                    split=args.split,
                    arm=args.arm,
                    record=record,
                )
                stem = evaluation_artifact_stem(record["record_id"])
                result_rows.append(verify_sidecar(sidecars / f"{stem}.json", identity))
            summary = aggregate(result_rows, config=config)
            payload = {
                "schema_version": 1,
                "status": "sealed_internal_test_complete" if args.split == "internal_test" else "development_evaluation_complete",
                "config_sha256": config_sha,
                "checkpoint": args.checkpoint.relative_to(ROOT).as_posix(),
                "checkpoint_sha256": checkpoint_sha,
                "conditions_sha256": conditions_sha,
                "split": args.split,
                "arm": args.arm,
                "summary": summary,
                "protected_foundir_test_accessed": False,
            }
            write_once(args.output / "result.json", (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode())
            if args.split == "internal_test":
                write_once(
                    canonical_consumption,
                    (json.dumps(intended_consumption, indent=2, sort_keys=True) + "\n").encode(),
                )
            print("FOUNDIR_FLUX_EVALUATION_PASS " + json.dumps(payload, sort_keys=True), flush=True)

        synchronize_rank_zero_finalization(
            finalize_result,
            rank=rank,
            device=device,
            phase="evaluation finalization",
        )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
