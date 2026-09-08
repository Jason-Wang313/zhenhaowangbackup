#!/usr/bin/env python3
"""Evaluate released FoundIR generalist and valid specialists on the frozen contract."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lowlight_restoration.flux_prompt_cache import file_sha256  # noqa: E402
from lowlight_restoration.foundir_five_type import FIVE_TYPE_CATEGORIES  # noqa: E402
from lowlight_restoration.foundir_flux2_protocol import (  # noqa: E402
    aggregate_task_metrics,
    evaluation_artifact_stem,
    resolve_workspace_path,
    stable_digest,
)
from lowlight_restoration.paired_development import rgb_structural_similarity  # noqa: E402
from smoke.evaluate_foundir_five_type_flux2_ddp import (  # noqa: E402
    load_aligned_crop,
    lpips_value,
    selected_from_contract,
    synchronize_rank_zero_finalization,
    write_once,
)
from smoke.run_foundir_lowlight_gate2_teachers import (  # noqa: E402
    SPECIALIST_DEPENDENCY_DIRECTORIES,
    build_generalist,
    build_specialist,
    pad_tensor,
)
from smoke.train_foundir_five_type_flux2_full_ddp import load_rows  # noqa: E402
from smoke.train_foundir_five_type_flux2_overfit_ddp import psnr  # noqa: E402


def released_config(config: dict[str, object], checkpoint: Path | None = None) -> dict[str, object]:
    baseline = config["released_baseline"]
    return {
        "generalist": {
            "repository": "third_party/foundir",
            "checkpoint": baseline["generalist"],
            "delta_end": 0.0014,
            "sampling_steps": 4,
            "ddim_eta": 0.0,
            "sum_scale": 0.01,
        },
        "specialist": {
            "repository": "third_party/foundir/specialist_model",
            "checkpoint": str(checkpoint.relative_to(ROOT)) if checkpoint else baseline["lowlight_specialist"],
        },
    }


def run_generalist(model, source: torch.Tensor, device: torch.device, task: str) -> torch.Tensor:
    tensor, height, width = pad_tensor(source.to(device), 8)
    with torch.inference_mode():
        sequence = model.sample(tensor, batch_size=1, last=True, task=task)
    return sequence[-1][:, :, :height, :width].float().clamp(0, 1)


def run_specialist(model, source: torch.Tensor) -> torch.Tensor:
    tensor, height, width = pad_tensor(source, 128)
    with torch.inference_mode():
        output = model.restoration_network(tensor)
    return output[:, :, :height, :width].float().clamp(0, 1)


def verify_sidecar(path: Path, *, row: dict[str, str], config_sha: str, conditions_sha: str) -> dict[str, object]:
    payload = json.loads(path.read_text())
    if (
        payload.get("status") != "complete"
        or payload.get("record_id") != row["record_id"]
        or payload.get("config_sha256") != config_sha
        or payload.get("conditions_sha256") != conditions_sha
        or payload.get("protected_foundir_test_accessed") is not False
    ):
        raise RuntimeError(f"released baseline sidecar identity changed: {path}")
    for candidate in payload.get("candidates", {}).values():
        output = resolve_workspace_path(ROOT, candidate["output_path"])
        if file_sha256(output) != candidate["output_sha256"]:
            raise RuntimeError(f"released baseline output changed: {output}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "internal_test"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.config = resolve_workspace_path(ROOT, args.config)
    args.output = resolve_workspace_path(ROOT, args.output)
    config = json.loads(args.config.read_text())
    config_sha = file_sha256(args.config)
    baseline = config["released_baseline"]
    for path_key, hash_key in (
        ("generalist", "generalist_sha256"),
        ("lowlight_specialist", "lowlight_specialist_sha256"),
        ("weather_specialist", "weather_specialist_sha256"),
    ):
        path = resolve_workspace_path(ROOT, baseline[path_key])
        if file_sha256(path) != baseline[hash_key]:
            raise RuntimeError(f"released baseline changed: {path}")
    if args.split == "internal_test":
        consumption = resolve_workspace_path(
            ROOT, Path(config["runtime"]["output_dir"]) / "sealed_internal_consumption.json"
        )
        if not consumption.is_file():
            raise RuntimeError("released sealed baseline may run only after the selected system consumes sealed data")
    conditions = resolve_workspace_path(ROOT, config["full_conditions"]["csv"])
    conditions_sha = file_sha256(conditions)
    rows = load_rows(conditions, int(config["data"]["expected_five_type_records"]))
    selected = selected_from_contract(
        config, rows, split=args.split, config_sha=config_sha, conditions_sha=conditions_sha
    )
    extra = ROOT / ".tools/foundir-runtime"
    if not extra.is_dir():
        raise RuntimeError(f"missing released FoundIR runtime: {extra}")
    sys.path.insert(0, str(extra))
    for dependency in reversed(SPECIALIST_DEPENDENCY_DIRECTORIES):
        if not dependency.is_dir():
            raise RuntimeError(f"missing released specialist dependency: {dependency}")
        sys.path.insert(0, str(dependency))

    dist.init_process_group(backend="nccl")
    try:
        rank, world, local_rank = dist.get_rank(), dist.get_world_size(), int(os.environ["LOCAL_RANK"])
        if world != 8 or torch.cuda.device_count() != 8:
            raise RuntimeError("released baseline comparison requires exactly eight GPUs")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        if not any(name in torch.cuda.get_device_properties(device).name for name in ("H100", "H200")):
            raise RuntimeError("released baseline comparison requires H100 or H200")
        seed = int(config["flow_training"]["seed"])
        random.seed(seed + rank)
        np.random.seed(seed + rank)
        torch.manual_seed(seed + rank)
        torch.cuda.manual_seed_all(seed + rank)
        generalist = build_generalist(released_config(config), device)
        lowlight = build_specialist(released_config(config), device)
        weather_path = resolve_workspace_path(ROOT, baseline["weather_specialist"])
        weather = build_specialist(released_config(config, weather_path), device)
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

        metric = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(device).eval()
        sidecars = args.output / "per_record"
        images = args.output / "images"
        sidecars.mkdir(parents=True, exist_ok=True)
        images.mkdir(parents=True, exist_ok=True)
        for index in range(rank, len(selected), world):
            row = selected[index]
            stem = evaluation_artifact_stem(row["record_id"])
            sidecar = sidecars / f"{stem}.json"
            if sidecar.exists():
                verify_sidecar(sidecar, row=row, config_sha=config_sha, conditions_sha=conditions_sha)
                continue
            lq_m11, gt_m11, crop_xy = load_aligned_crop(
                row,
                crop_size=int(config["data"]["crop_size"]),
                seed=int(config["data"]["split_seed"]),
            )
            source = ((lq_m11 + 1) * 0.5).unsqueeze(0)
            target = gt_m11.unsqueeze(0)
            record_seed = int(stable_digest(seed, "released_baseline", row["record_id"])[:16], 16) % (2**63 - 1)
            random.seed(record_seed)
            np.random.seed(record_seed % (2**32 - 1))
            torch.manual_seed(record_seed)
            torch.cuda.manual_seed_all(record_seed)
            candidates = {"generalist": run_generalist(generalist, source, device, row["record_id"])}
            types = set(FIVE_TYPE_CATEGORIES[row["category"]])
            if "low_light" in types:
                candidates["generalist_lowlight_specialist"] = run_specialist(lowlight, candidates["generalist"])
            if types & {"rain", "haze"}:
                candidates["generalist_weather_specialist"] = run_specialist(weather, candidates["generalist"])
            candidate_rows = {}
            for name, output_01 in candidates.items():
                output_m11 = output_01 * 2 - 1
                array = output_01[0].detach().cpu().permute(1, 2, 0).mul(255).round().byte().numpy()
                output_path = images / f"{stem}_{name}.png"
                if output_path.exists():
                    raise RuntimeError(f"orphan released baseline output exists: {output_path}")
                Image.fromarray(array).save(output_path)
                candidate_rows[name] = {
                    "category": row["category"],
                    "psnr": psnr(target.to(device), output_m11),
                    "lpips": lpips_value(metric, target, output_m11.cpu(), device),
                    "ssim": rgb_structural_similarity(
                        ((gt_m11.numpy().transpose(1, 2, 0) + 1) * 0.5).clip(0, 1),
                        array.astype(np.float32) / 255,
                    ),
                    "output_path": output_path.relative_to(ROOT).as_posix(),
                    "output_sha256": file_sha256(output_path),
                }
            result = {
                "status": "complete",
                "record_id": row["record_id"],
                "source_group": row["source_group"],
                "category": row["category"],
                "composition_size": int(row["composition_size"]),
                "evaluation_fold": int(row["evaluation_fold"]),
                "crop_xy": crop_xy,
                "candidates": candidate_rows,
                "config_sha256": config_sha,
                "conditions_sha256": conditions_sha,
                "protected_foundir_test_accessed": False,
            }
            write_once(sidecar, (json.dumps(result, indent=2, sort_keys=True) + "\n").encode())
            print(f"FOUNDIR_RELEASED_BASELINE rank={rank} record={index+1}/{len(selected)}", flush=True)
        dist.barrier()
        def finalize_result() -> None:
            result_rows = [
                verify_sidecar(
                    sidecars / f"{evaluation_artifact_stem(row['record_id'])}.json",
                    row=row,
                    config_sha=config_sha,
                    conditions_sha=conditions_sha,
                )
                for row in selected
            ]
            by_candidate: dict[str, list[dict[str, object]]] = defaultdict(list)
            for row in result_rows:
                for name, metrics in row["candidates"].items():
                    by_candidate[name].append({"category": row["category"], **metrics})
            summaries = {}
            for name, candidate_rows in sorted(by_candidate.items()):
                summaries[name] = {
                    "records": len(candidate_rows),
                    "mean_psnr": float(np.mean([row["psnr"] for row in candidate_rows])),
                    "mean_lpips": float(np.mean([row["lpips"] for row in candidate_rows])),
                }
                if name == "generalist":
                    summaries[name]["task_metrics"] = aggregate_task_metrics(candidate_rows)
            payload = {
                "schema_version": 1,
                "status": "released_foundir_baseline_comparison_complete",
                "config_sha256": config_sha,
                "split": args.split,
                "records": len(result_rows),
                "candidate_summaries": summaries,
                "specialist_scope": "lowlight only for low_light categories; weather only for rain/haze categories",
                "protected_foundir_test_accessed": False,
            }
            write_once(args.output / "result.json", (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode())
            print("FOUNDIR_RELEASED_BASELINE_PASS " + json.dumps(payload, sort_keys=True), flush=True)

        synchronize_rank_zero_finalization(
            finalize_result,
            rank=rank,
            device=device,
            phase="released-baseline finalization",
        )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
