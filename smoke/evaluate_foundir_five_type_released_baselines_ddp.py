#!/usr/bin/env python3
"""Evaluate released FoundIR generalist and valid specialists on the frozen contract."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.distributed as dist

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
    write_or_verify_png,
)
from smoke.run_foundir_lowlight_gate2_teachers import (  # noqa: E402
    SPECIALIST_DEPENDENCY_DIRECTORIES,
    build_generalist,
    build_specialist,
    build_weather_specialist,
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


def run_weather_specialist(model, source: torch.Tensor) -> torch.Tensor:
    # The released weather inference pads to and processes 512-pixel patches.
    tensor, height, width = pad_tensor(source, 512)
    with torch.inference_mode():
        output = model(tensor)
    return output[:, :, :height, :width].float().clamp(0, 1)


def expected_candidate_names(row: Mapping[str, str]) -> set[str]:
    names = {"generalist"}
    types = set(FIVE_TYPE_CATEGORIES[row["category"]])
    if "low_light" in types:
        names.add("generalist_lowlight_specialist")
    if types & {"rain", "haze"}:
        names.add("generalist_weather_specialist")
    return names


def finite_metric(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(float(value))


def verify_sidecar(
    path: Path,
    *,
    row: Mapping[str, str],
    split: str,
    config_sha: str,
    conditions_sha: str,
) -> dict[str, object]:
    payload = json.loads(path.read_text())
    if (
        payload.get("status") != "complete"
        or payload.get("record_id") != row["record_id"]
        or payload.get("source_group") != row["source_group"]
        or payload.get("category") != row["category"]
        or payload.get("composition_size") != int(row["composition_size"])
        or payload.get("evaluation_fold") != int(row["evaluation_fold"])
        or payload.get("split") != split
        or payload.get("config_sha256") != config_sha
        or payload.get("conditions_sha256") != conditions_sha
        or payload.get("protected_foundir_test_accessed") is not False
    ):
        raise RuntimeError(f"released baseline sidecar identity changed: {path}")
    candidates = payload.get("candidates")
    if not isinstance(candidates, dict) or set(candidates) != expected_candidate_names(row):
        raise RuntimeError(f"released baseline candidate set changed: {path}")
    for candidate in candidates.values():
        if candidate.get("category") != row["category"] or not all(
            finite_metric(candidate.get(key)) for key in ("psnr", "lpips", "ssim")
        ):
            raise RuntimeError(f"released baseline metrics are invalid: {path}")
        output = resolve_workspace_path(ROOT, candidate["output_path"])
        if file_sha256(output) != candidate["output_sha256"]:
            raise RuntimeError(f"released baseline output changed: {output}")
    return payload


def build_result_payload(
    result_rows: Sequence[Mapping[str, object]],
    *,
    config_sha: str,
    conditions_sha: str,
    split: str,
    sealed_consumption_sha: str | None = None,
) -> dict[str, object]:
    if not result_rows:
        raise RuntimeError("released baseline result has no records")
    by_candidate: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in result_rows:
        for name, metrics in row["candidates"].items():
            by_candidate[name].append({"category": row["category"], **metrics})
    if len(by_candidate.get("generalist", [])) != len(result_rows):
        raise RuntimeError("released generalist does not cover every evaluation record")
    summaries: dict[str, dict[str, object]] = {}
    for name, candidate_rows in sorted(by_candidate.items()):
        summary: dict[str, object] = {
            "records": len(candidate_rows),
            "mean_psnr": float(np.mean([row["psnr"] for row in candidate_rows])),
            "mean_lpips": float(np.mean([row["lpips"] for row in candidate_rows])),
        }
        if not finite_metric(summary["mean_psnr"]) or not finite_metric(summary["mean_lpips"]):
            raise RuntimeError(f"released baseline {name} aggregate is non-finite")
        if name == "generalist":
            task_metrics = aggregate_task_metrics(candidate_rows)
            if set(task_metrics) != {"blur", "noise", "rain", "haze", "low_light"}:
                raise RuntimeError("released generalist task coverage changed")
            if not all(
                finite_metric(metrics.get(metric))
                for metrics in task_metrics.values()
                for metric in ("psnr", "lpips")
            ):
                raise RuntimeError("released generalist task metrics are non-finite")
            summary["task_metrics"] = task_metrics
        summaries[name] = summary
    payload: dict[str, object] = {
        "schema_version": 1,
        "status": "released_foundir_baseline_comparison_complete",
        "config_sha256": config_sha,
        "conditions_sha256": conditions_sha,
        "split": split,
        "records": len(result_rows),
        # ``summary`` is the deployable released generalist contract consumed
        # by every final-report freezer.  Specialist summaries remain separate
        # evidence because they cover only applicable categories.
        "summary": summaries["generalist"],
        "candidate_summaries": summaries,
        "specialist_scope": "lowlight only for low_light categories; weather only for rain/haze categories",
        "protected_foundir_test_accessed": False,
    }
    if sealed_consumption_sha is not None:
        payload["sealed_internal_consumption_sha256"] = sealed_consumption_sha
    return payload


def verify_sealed_release_order(
    config: Mapping[str, object], config_sha: str, conditions_sha: str
) -> str:
    output_root = resolve_workspace_path(ROOT, config["runtime"]["output_dir"])
    consumption_path = output_root / "sealed_internal_consumption.json"
    selection_path = output_root / "checkpoint_selection.json"
    sealed_path = output_root / "sealed_internal/qwen/result.json"
    if not all(path.is_file() for path in (consumption_path, selection_path, sealed_path)):
        raise RuntimeError("released sealed baseline may run only after the selected system consumes sealed data")
    consumption = json.loads(consumption_path.read_text())
    selection = json.loads(selection_path.read_text())
    sealed = json.loads(sealed_path.read_text())
    expected_output = (output_root / "sealed_internal/qwen").relative_to(ROOT).as_posix()
    if (
        consumption.get("status") != "sealed_internal_test_consumed"
        or consumption.get("config_sha256") != config_sha
        or consumption.get("selection_sha256") != file_sha256(selection_path)
        or consumption.get("checkpoint_sha256") != selection.get("checkpoint_sha256")
        or consumption.get("output") != expected_output
        or consumption.get("protected_foundir_test_accessed") is not False
        or selection.get("status") != "frozen_development_selected_checkpoint"
        or selection.get("config_sha256") != config_sha
        or selection.get("sealed_internal_test_accessed") is not False
        or selection.get("protected_foundir_test_accessed") is not False
        or sealed.get("status") != "sealed_internal_test_complete"
        or sealed.get("config_sha256") != config_sha
        or sealed.get("checkpoint_sha256") != selection.get("checkpoint_sha256")
        or sealed.get("checkpoint") != selection.get("checkpoint")
        or sealed.get("conditions_sha256") != conditions_sha
        or sealed.get("split") != "internal_test"
        or sealed.get("arm") != "qwen"
        or sealed.get("protected_foundir_test_accessed") is not False
    ):
        raise RuntimeError("sealed internal consumption identity changed before released baseline")
    return file_sha256(consumption_path)


def finalize_existing_sidecars(
    *,
    output: Path,
    selected: Sequence[Mapping[str, str]],
    split: str,
    config_sha: str,
    conditions_sha: str,
    sealed_consumption_sha: str | None,
) -> bool:
    """Publish an aggregate only when every immutable per-record result exists."""
    sidecars = output / "per_record"
    paths = [
        sidecars / f"{evaluation_artifact_stem(row['record_id'])}.json"
        for row in selected
    ]
    present = sum(path.is_file() for path in paths)
    if present != len(paths):
        print(
            f"FOUNDIR_RELEASED_BASELINE_RESUME_PENDING "
            f"present={present} expected={len(paths)}",
            flush=True,
        )
        return False
    result_rows = [
        verify_sidecar(
            path,
            row=row,
            split=split,
            config_sha=config_sha,
            conditions_sha=conditions_sha,
        )
        for path, row in zip(paths, selected, strict=True)
    ]
    payload = build_result_payload(
        result_rows,
        config_sha=config_sha,
        conditions_sha=conditions_sha,
        split=split,
        sealed_consumption_sha=sealed_consumption_sha,
    )
    write_once(
        output / "result.json",
        (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode(),
    )
    print("FOUNDIR_RELEASED_BASELINE_PASS " + json.dumps(payload, sort_keys=True), flush=True)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "internal_test"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--finalize-existing-if-complete", action="store_true")
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
    conditions = resolve_workspace_path(ROOT, config["full_conditions"]["csv"])
    conditions_sha = file_sha256(conditions)
    sealed_consumption_sha = None
    if args.split == "internal_test":
        sealed_consumption_sha = verify_sealed_release_order(
            config, config_sha, conditions_sha
        )
    rows = load_rows(conditions, int(config["data"]["expected_five_type_records"]))
    selected = selected_from_contract(
        config, rows, split=args.split, config_sha=config_sha, conditions_sha=conditions_sha
    )
    if args.finalize_existing_if_complete:
        finalize_existing_sidecars(
            output=args.output,
            selected=selected,
            split=args.split,
            config_sha=config_sha,
            conditions_sha=conditions_sha,
            sealed_consumption_sha=sealed_consumption_sha,
        )
        return
    extra = ROOT / ".tools/foundir-runtime"
    if not extra.is_dir():
        raise RuntimeError(f"missing released FoundIR runtime: {extra}")
    sys.path.insert(0, str(extra))
    for dependency in reversed(SPECIALIST_DEPENDENCY_DIRECTORIES):
        if not dependency.is_dir():
            raise RuntimeError(f"missing released specialist dependency: {dependency}")
        sys.path.insert(0, str(dependency))

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend="nccl", device_id=device)
    try:
        rank, world = dist.get_rank(), dist.get_world_size()
        if world != 8 or torch.cuda.device_count() != 8:
            raise RuntimeError("released baseline comparison requires exactly eight GPUs")
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
        weather = build_weather_specialist(released_config(config, weather_path), device)
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
                verify_sidecar(
                    sidecar,
                    row=row,
                    split=args.split,
                    config_sha=config_sha,
                    conditions_sha=conditions_sha,
                )
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
                candidates["generalist_weather_specialist"] = run_weather_specialist(
                    weather, candidates["generalist"]
                )
            candidate_rows = {}
            for name, output_01 in candidates.items():
                array = output_01[0].detach().cpu().permute(1, 2, 0).mul(255).round().byte().numpy()
                output_q_m11 = (
                    torch.from_numpy(array.copy())
                    .permute(2, 0, 1)
                    .float()
                    .div_(127.5)
                    .sub_(1.0)
                    .unsqueeze(0)
                )
                output_path = images / f"{stem}_{name}.png"
                candidate = {
                    "category": row["category"],
                    "psnr": psnr(target.to(device), output_q_m11.to(device)),
                    "lpips": lpips_value(metric, target, output_q_m11, device),
                    "ssim": rgb_structural_similarity(
                        ((gt_m11.numpy().transpose(1, 2, 0) + 1) * 0.5).clip(0, 1),
                        array.astype(np.float32) / 255,
                    ),
                }
                if not all(finite_metric(candidate[key]) for key in ("psnr", "lpips", "ssim")):
                    raise RuntimeError(f"released baseline produced non-finite metrics for {row['record_id']}")
                write_or_verify_png(output_path, array, label="released baseline")
                candidate.update(
                    output_path=output_path.relative_to(ROOT).as_posix(),
                    output_sha256=file_sha256(output_path),
                )
                candidate_rows[name] = candidate
            result = {
                "status": "complete",
                "record_id": row["record_id"],
                "source_group": row["source_group"],
                "category": row["category"],
                "composition_size": int(row["composition_size"]),
                "evaluation_fold": int(row["evaluation_fold"]),
                "split": args.split,
                "crop_xy": crop_xy,
                "candidates": candidate_rows,
                "config_sha256": config_sha,
                "conditions_sha256": conditions_sha,
                "protected_foundir_test_accessed": False,
            }
            write_once(sidecar, (json.dumps(result, indent=2, sort_keys=True) + "\n").encode())
            print(f"FOUNDIR_RELEASED_BASELINE rank={rank} record={index+1}/{len(selected)}", flush=True)
        dist.barrier(device_ids=[local_rank])
        def finalize_result() -> None:
            if not finalize_existing_sidecars(
                output=args.output,
                selected=selected,
                split=args.split,
                config_sha=config_sha,
                conditions_sha=conditions_sha,
                sealed_consumption_sha=sealed_consumption_sha,
            ):
                raise RuntimeError(
                    "released-baseline finalization reached with incomplete sidecars"
                )

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
