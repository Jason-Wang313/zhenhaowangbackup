#!/usr/bin/env python3
"""One-shot confirmation of the frozen Blur+Noise Restormer on its sealed training-only holdout."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lowlight_restoration.flux_prompt_cache import file_sha256  # noqa: E402
from lowlight_restoration.foundir_flux2_protocol import (  # noqa: E402
    evaluation_artifact_stem,
    resolve_workspace_path,
    stable_digest,
)
from lowlight_restoration.task_native_restormer import load_task_native_restormer  # noqa: E402
from smoke.evaluate_foundir_five_type_flux2_ddp import (  # noqa: E402
    lpips_value,
    synchronize_rank_zero_finalization,
    torchrun_worker_identity,
    write_once,
)
from smoke.evaluate_foundir_noise_rain_full_development_ddp import (  # noqa: E402
    finite_metric,
    load_audit_crop,
    source_group_bootstrap,
    strong_gate_checks,
)
from smoke.train_foundir_five_type_flux2_overfit_ddp import psnr  # noqa: E402
from smoke.train_foundir_task_native_restormer_adaptation_ddp import (  # noqa: E402
    load_task_rows,
    validate_config,
)

CONFIG = ROOT / "configs/foundir_blur_noise_restormer_pilot_v1.json"
SELECTION = ROOT / "checkpoints/foundir_blur_noise_restormer_pilot_v1/development_selection.json"
OUTPUT = ROOT / "checkpoints/foundir_blur_noise_restormer_pilot_v1/internal_holdout_v1"
EXPECTED_STEP = 1000
EXPECTED_RECORDS = 128
EXPECTED_GROUPS = 83


def contract() -> tuple[dict, str, dict, Path]:
    config = validate_config(CONFIG)
    config_sha = file_sha256(CONFIG)
    selection = json.loads(SELECTION.read_text())
    checkpoint = resolve_workspace_path(ROOT, selection.get("checkpoint", ""))
    if (
        config.get("task") != "blur_noise"
        or config.get("protected_foundir_test_accessed") is not False
        or selection.get("status") != "foundir_task_native_adaptation_development_selection"
        or selection.get("config_sha256") != config_sha
        or selection.get("selected_step") != EXPECTED_STEP
        or selection.get("strong_gate_pass") is not True
        or selection.get("sealed_internal_test_accessed") is not False
        or selection.get("protected_foundir_test_accessed") is not False
        or not checkpoint.is_file()
        or file_sha256(checkpoint) != selection.get("checkpoint_sha256")
    ):
        raise RuntimeError("Blur+Noise holdout parent contract changed")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if (
        payload.get("status") != "foundir_task_native_adaptation_checkpoint"
        or payload.get("config_sha256") != config_sha
        or payload.get("task") != "blur_noise"
        or payload.get("step") != EXPECTED_STEP
        or payload.get("sealed_internal_test_accessed") is not False
        or payload.get("protected_foundir_test_accessed") is not False
        or not isinstance(payload.get("model"), Mapping)
    ):
        raise RuntimeError("Blur+Noise holdout checkpoint contract changed")
    return config, config_sha, selection, checkpoint


def identity(row: Mapping[str, str], config_sha: str, checkpoint_sha: str) -> dict[str, object]:
    return {
        "config_sha256": config_sha,
        "checkpoint_sha256": checkpoint_sha,
        "record_id": row["record_id"],
        "source_group": row["source_group"],
        "category": row["category"],
        "lq_sha256": row["lq_sha256"],
        "gt_sha256": row["gt_sha256"],
    }


def verify_sidecar(path: Path, expected: Mapping[str, object]) -> dict[str, object]:
    payload = json.loads(path.read_text())
    if (
        payload.get("schema_version") != 1
        or payload.get("status") != "complete"
        or payload.get("split") != "internal_test"
        or payload.get("identity") != dict(expected)
        or payload.get("sealed_internal_test_accessed") is not True
        or payload.get("protected_foundir_test_accessed") is not False
    ):
        raise RuntimeError(f"Blur+Noise holdout sidecar changed: {path}")
    for arm in ("input", "restored"):
        metrics = payload.get(arm, {})
        if not all(finite_metric(metrics.get(name)) for name in ("psnr", "lpips")):
            raise RuntimeError(f"Blur+Noise holdout metrics invalid: {path}")
    return payload


def aggregate(
    rows: Sequence[Mapping[str, object]], config: Mapping[str, object], config_sha: str,
    checkpoint: Path, selection: Mapping[str, object]
) -> dict[str, object]:
    input_psnr = float(np.mean([row["input"]["psnr"] for row in rows]))
    input_lpips = float(np.mean([row["input"]["lpips"] for row in rows]))
    restored_psnr = float(np.mean([row["restored"]["psnr"] for row in rows]))
    restored_lpips = float(np.mean([row["restored"]["lpips"] for row in rows]))
    metrics = {
        "records": len(rows),
        "source_groups": len({str(row["identity"]["source_group"]) for row in rows}),
        "input_psnr": input_psnr,
        "input_lpips": input_lpips,
        "psnr": restored_psnr,
        "lpips": restored_lpips,
        "psnr_gain_db": restored_psnr - input_psnr,
        "psnr_relative_improvement_percent": 100.0 * (restored_psnr - input_psnr) / input_psnr,
        "lpips_reduction": input_lpips - restored_lpips,
        "lpips_relative_improvement_percent": 100.0 * (input_lpips - restored_lpips) / input_lpips,
        "psnr_win_rate_percent": 100.0 * np.mean([
            row["restored"]["psnr"] > row["input"]["psnr"] for row in rows
        ]),
        "lpips_win_rate_percent": 100.0 * np.mean([
            row["restored"]["lpips"] < row["input"]["lpips"] for row in rows
        ]),
    }
    shaped_restored = [
        {"source_group": row["identity"]["source_group"], "restored": row["restored"]}
        for row in rows
    ]
    shaped_input = [
        {"source_group": row["identity"]["source_group"], "restored": row["input"]}
        for row in rows
    ]
    gate = config["strong_gate"]
    restored_bootstrap = source_group_bootstrap(
        shaped_restored,
        resamples=int(gate["bootstrap_resamples"]),
        confidence=float(gate["bootstrap_confidence"]),
        seed=int(stable_digest(int(gate["bootstrap_seed"]), "blur_noise", "internal_test", "restored")[:16], 16),
    )
    input_bootstrap = source_group_bootstrap(
        shaped_input,
        resamples=int(gate["bootstrap_resamples"]),
        confidence=float(gate["bootstrap_confidence"]),
        seed=int(stable_digest(int(gate["bootstrap_seed"]), "blur_noise", "internal_test", "input")[:16], 16),
    )
    checks = strong_gate_checks(metrics, restored_bootstrap, gate)
    checks.update({
        "paired_mean_psnr_improves_input": restored_psnr > input_psnr,
        "paired_mean_lpips_improves_input": restored_lpips < input_lpips,
    })
    if len(rows) != EXPECTED_RECORDS or metrics["source_groups"] != EXPECTED_GROUPS:
        raise RuntimeError("Blur+Noise holdout aggregate coverage changed")
    return {
        "schema_version": 1,
        "status": "foundir_blur_noise_restormer_internal_holdout_complete",
        "split": "internal_test",
        "scope": "foundir_training_only_source_disjoint_holdout",
        "config_sha256": config_sha,
        "selection_sha256": file_sha256(SELECTION),
        "checkpoint": checkpoint.relative_to(ROOT).as_posix(),
        "checkpoint_sha256": selection["checkpoint_sha256"],
        "selected_step": EXPECTED_STEP,
        "metrics": metrics,
        "input_source_group_bootstrap": input_bootstrap,
        "restored_source_group_bootstrap": restored_bootstrap,
        "confirmation_checks": checks,
        "confirmation_pass": all(checks.values()),
        "selection_was_frozen_before_holdout": True,
        "sealed_internal_test_accessed": True,
        "protected_foundir_test_accessed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    config, config_sha, selection, checkpoint = contract()
    rows = load_task_rows(config)["internal_test"]
    if len(rows) != EXPECTED_RECORDS or len({row["source_group"] for row in rows}) != EXPECTED_GROUPS:
        raise RuntimeError("Blur+Noise internal holdout coverage changed")
    if args.preflight:
        print("FOUNDIR_BLUR_NOISE_INTERNAL_HOLDOUT_PREFLIGHT_PASS " + json.dumps({
            "records": len(rows), "source_groups": EXPECTED_GROUPS,
            "selected_step": EXPECTED_STEP, "checkpoint_sha256": selection["checkpoint_sha256"],
            "protected_foundir_test_accessed": False,
        }, sort_keys=True), flush=True)
        return

    rank, world, local_rank = torchrun_worker_identity(8)
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if torch.cuda.device_count() != world or not any(
        name in torch.cuda.get_device_properties(device).name for name in ("H100", "H200")
    ):
        raise RuntimeError("Blur+Noise internal holdout requires eight H100/H200 GPUs")
    model, _ = load_task_native_restormer(
        config["initial_checkpoint_name"], resolve_workspace_path(ROOT, Path(config["initial_checkpoint"]).parent)
    )
    checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint_payload["model"], strict=True)
    model = model.to(device).eval()
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    metric = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(device).eval()
    sidecars = OUTPUT / "per_record"
    sidecars.mkdir(parents=True, exist_ok=True)
    checkpoint_sha = selection["checkpoint_sha256"]
    for index in range(rank, len(rows), world):
        row = rows[index]
        expected = identity(row, config_sha, checkpoint_sha)
        path = sidecars / f"{evaluation_artifact_stem(row['record_id'])}.json"
        if path.exists():
            verify_sidecar(path, expected)
            continue
        lq_m11, gt_m11, crop_xy, resized = load_audit_crop(
            row, crop_size=int(config["data"]["validation_crop_size"]),
            seed=int(config["data"]["validation_crop_seed"]),
        )
        source = ((lq_m11 + 1.0) * 0.5).unsqueeze(0).to(device)
        target = gt_m11.unsqueeze(0).to(device)
        with torch.inference_mode():
            restored = model(source).float().clamp(0, 1)
        restored_u8 = restored[0].cpu().permute(1, 2, 0).mul(255).round().byte().numpy()
        restored_m11 = torch.from_numpy(restored_u8.copy()).permute(2, 0, 1).float().div_(127.5).sub_(1).unsqueeze(0)
        lq_batch = lq_m11.unsqueeze(0).to(device)
        payload = {
            "schema_version": 1, "status": "complete", "split": "internal_test",
            "identity": expected, "crop_xy": crop_xy, "undersized_pair_resized": resized,
            "input": {"psnr": psnr(target, lq_batch), "lpips": lpips_value(metric, target, lq_batch, device)},
            "restored": {"psnr": psnr(target, restored_m11.to(device)), "lpips": lpips_value(metric, target, restored_m11, device)},
            "restored_uint8_sha256": hashlib.sha256(restored_u8.tobytes()).hexdigest(),
            "sealed_internal_test_accessed": True, "protected_foundir_test_accessed": False,
        }
        metric.reset()
        write_once(path, (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode())
        print(f"FOUNDIR_BLUR_NOISE_INTERNAL_HOLDOUT_PROGRESS rank={rank} record={index + 1}/{len(rows)}", flush=True)

    def finalize() -> None:
        expected_paths = [sidecars / f"{evaluation_artifact_stem(row['record_id'])}.json" for row in rows]
        if set(sidecars.glob("*.json")) != set(expected_paths):
            raise RuntimeError("Blur+Noise holdout sidecar set incomplete or contaminated")
        complete = [
            verify_sidecar(path, identity(row, config_sha, checkpoint_sha))
            for row, path in zip(rows, expected_paths, strict=True)
        ]
        result = aggregate(complete, config, config_sha, checkpoint, selection)
        write_once(OUTPUT / "result.json", (json.dumps(result, indent=2, sort_keys=True) + "\n").encode())
        print("FOUNDIR_BLUR_NOISE_INTERNAL_HOLDOUT_COMPLETE " + json.dumps(result, sort_keys=True), flush=True)

    synchronize_rank_zero_finalization(
        finalize, rank=rank, world=world, output=OUTPUT,
        identity={"config_sha256": config_sha, "checkpoint_sha256": checkpoint_sha, "records": EXPECTED_RECORDS},
        phase="Blur+Noise Restormer training-only internal holdout", timeout_seconds=600,
    )


if __name__ == "__main__":
    main()
