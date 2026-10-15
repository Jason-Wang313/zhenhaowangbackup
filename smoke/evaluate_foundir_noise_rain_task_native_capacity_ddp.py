#!/usr/bin/env python3
"""Measure task-native Restormer capacity and actual-Qwen hybrid diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
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
from lowlight_restoration.task_native_restormer import (  # noqa: E402
    TASK_NATIVE_RESTORMER_SPECS,
    load_task_native_restormer,
)
from smoke.evaluate_foundir_five_type_flux2_ddp import (  # noqa: E402
    lpips_value,
    synchronize_rank_zero_finalization,
    torchrun_worker_identity,
    write_once,
)
from smoke.evaluate_foundir_five_type_qwen_routed_integrated_ddp import (  # noqa: E402
    route_decision,
)
from smoke.evaluate_foundir_noise_rain_full_development_ddp import (  # noqa: E402
    finite_metric,
    load_audit_crop,
    mean_metrics,
    source_group_bootstrap,
    strong_gate_checks,
    verify_sidecar as verify_baseline_sidecar,
)
from smoke.train_foundir_five_type_flux2_full_ddp import load_rows  # noqa: E402
from smoke.train_foundir_five_type_flux2_overfit_ddp import psnr  # noqa: E402


EXPECTED_CATEGORIES = {"05Noise": 1859, "10Rain": 835}
EXPECTED_CANDIDATES = {
    "noise": ["real_denoising", "gaussian_color_denoising_blind"],
    "rain": ["deraining"],
}


def validate_config(path: Path) -> tuple[dict, dict, dict, Path, Path]:
    config = json.loads(path.read_text())
    if (
        config.get("schema_version") != 1
        or config.get("status")
        != "frozen_after_full_development_baseline_failure_slice_audit"
        or config.get("selection", {}).get("categories") != EXPECTED_CATEGORIES
        or int(config["selection"].get("manifest_expected_records", -1)) != 685495
        or int(config["selection"].get("expected_records", -1)) != 2694
        or int(config["selection"].get("crop_size", -1)) != 512
        or int(config["selection"].get("crop_seed", -1)) != 2026083001
        or config.get("candidates", {}).get("noise") != EXPECTED_CANDIDATES["noise"]
        or config.get("candidates", {}).get("rain") != EXPECTED_CANDIDATES["rain"]
        or config.get("candidates", {}).get(
            "category_selects_candidate_family_for_capacity_measurement_only"
        )
        is not True
        or config.get("candidates", {}).get("category_must_not_control_final_deployable_route")
        is not True
        or config.get("routing_diagnostics", {}).get("thresholds") != [2, 3, 4, 5]
        or config.get("routing_diagnostics", {}).get("not_a_frozen_deployable_selection")
        is not True
        or config.get("sealed_internal_test_accessed") is not False
        or config.get("protected_foundir_test_accessed") is not False
    ):
        raise RuntimeError("task-native capacity audit contract changed")
    baseline_config_path = resolve_workspace_path(ROOT, config["baseline_config"])
    baseline_result_path = resolve_workspace_path(ROOT, config["baseline_result"])
    conditions = resolve_workspace_path(ROOT, config["conditions"])
    for candidate, spec in TASK_NATIVE_RESTORMER_SPECS.items():
        if candidate not in {name for values in EXPECTED_CANDIDATES.values() for name in values}:
            continue
        checkpoint = resolve_workspace_path(
            ROOT, Path(config["candidates"]["checkpoint_directory"]) / spec.filename
        )
        if (
            not checkpoint.is_file()
            or checkpoint.stat().st_size != spec.size_bytes
            or file_sha256(checkpoint) != spec.sha256
        ):
            raise RuntimeError(f"task-native candidate checkpoint changed: {candidate}")
    identities = (
        (baseline_config_path, config["baseline_config_sha256"]),
        (baseline_result_path, config["baseline_result_sha256"]),
        (conditions, config["conditions_sha256"]),
    )
    if any(not p.is_file() or file_sha256(p) != expected for p, expected in identities):
        raise RuntimeError("task-native capacity parent identity changed")
    baseline_config = json.loads(baseline_config_path.read_text())
    baseline_result = json.loads(baseline_result_path.read_text())
    if (
        baseline_result.get("status") != "noise_rain_full_development_audit_complete"
        or baseline_result.get("config_sha256") != config["baseline_config_sha256"]
        or baseline_result.get("records") != 7175
        or baseline_result.get("sealed_internal_test_accessed") is not False
        or baseline_result.get("protected_foundir_test_accessed") is not False
    ):
        raise RuntimeError("task-native capacity requires completed development baseline")
    router_path = resolve_workspace_path(ROOT, baseline_config["router_config"])
    router = json.loads(router_path.read_text())
    return config, baseline_config, router, conditions, baseline_result_path


def select_rows(rows: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    selected = [
        dict(row)
        for row in rows
        if row["split"] == "validation" and row["category"] in EXPECTED_CATEGORIES
    ]
    if Counter(row["category"] for row in selected) != Counter(EXPECTED_CATEGORIES):
        raise RuntimeError("task-native isolated capacity coverage changed")
    if len({row["record_id"] for row in selected}) != len(selected):
        raise RuntimeError("task-native isolated capacity record IDs are not unique")
    return selected


def candidate_names(row: Mapping[str, str]) -> list[str]:
    return EXPECTED_CANDIDATES["noise" if row["category"] == "05Noise" else "rain"]


def baseline_sidecar(
    row: Mapping[str, str], baseline_config: Mapping[str, object], router: Mapping[str, object]
) -> dict[str, object]:
    output = resolve_workspace_path(ROOT, baseline_config["output"])
    path = output / "per_record" / f"{evaluation_artifact_stem(row['record_id'])}.json"
    levels = json.loads(row["qwen_json"])
    return verify_baseline_sidecar(
        path,
        row=row,
        decision=route_decision(levels, router["routing"]),
        config_sha=file_sha256(resolve_workspace_path(ROOT, "configs/foundir_noise_rain_full_development_audit_v1.json")),
        router_sha=file_sha256(resolve_workspace_path(ROOT, baseline_config["router_config"])),
    )


def verify_sidecar(
    path: Path, *, row: Mapping[str, str], config_sha: str, baseline_result_sha: str
) -> dict[str, object]:
    payload = json.loads(path.read_text())
    if (
        payload.get("schema_version") != 1
        or payload.get("status") != "complete"
        or payload.get("config_sha256") != config_sha
        or payload.get("baseline_result_sha256") != baseline_result_sha
        or payload.get("record_id") != row["record_id"]
        or payload.get("source_group") != row["source_group"]
        or payload.get("category") != row["category"]
        or payload.get("qwen_levels") != json.loads(row["qwen_json"])
        or set(payload.get("candidates", {})) != set(candidate_names(row))
        or payload.get("sealed_internal_test_accessed") is not False
        or payload.get("protected_foundir_test_accessed") is not False
    ):
        raise RuntimeError(f"task-native capacity sidecar identity changed: {path}")
    for metrics in payload["candidates"].values():
        if (
            not all(finite_metric(metrics.get(name)) for name in ("psnr", "lpips", "residual_l1", "residual_rms"))
            or not isinstance(metrics.get("restored_uint8_sha256"), str)
            or len(metrics["restored_uint8_sha256"]) != 64
        ):
            raise RuntimeError(f"task-native capacity sidecar metrics invalid: {path}")
    return payload


def metrics_for(rows: Sequence[Mapping[str, object]], selector) -> dict[str, object]:
    shaped = []
    for row in rows:
        restored = selector(row)
        shaped.append(
            {
                "source_group": row["source_group"],
                "input": row["baseline"]["input"],
                "restored": restored,
            }
        )
    return mean_metrics(shaped)


def bootstrap_for(
    rows: Sequence[Mapping[str, object]], selector, config: Mapping[str, object], label: str
) -> dict[str, object]:
    shaped = [
        {"source_group": row["source_group"], "restored": selector(row)} for row in rows
    ]
    metrics = config["metrics"]
    return source_group_bootstrap(
        shaped,
        resamples=int(metrics["bootstrap_resamples"]),
        confidence=float(metrics["bootstrap_confidence"]),
        seed=int(stable_digest(int(metrics["bootstrap_seed"]), label)[:16], 16),
    )


def build_result(
    rows: Sequence[Mapping[str, object]], config: Mapping[str, object], config_sha: str
) -> dict[str, object]:
    result_candidates = {}
    routing = {}
    for task, category in (("noise", "05Noise"), ("rain", "10Rain")):
        task_rows = [row for row in rows if row["category"] == category]
        gate = config["strong_gates"][task]
        for candidate in EXPECTED_CANDIDATES[task]:
            key = f"{task}:{candidate}"
            selector = lambda row, candidate=candidate: row["candidates"][candidate]
            metrics = metrics_for(task_rows, selector)
            bootstrap = bootstrap_for(task_rows, selector, config, key)
            checks = strong_gate_checks(metrics, bootstrap, gate)
            result_candidates[key] = {
                "scope": "category_selected_capacity_diagnostic_not_deployable_routing",
                "metrics": metrics,
                "source_group_bootstrap": bootstrap,
                "strong_gate_checks": checks,
                "strong_gate_pass": all(checks.values()),
            }
            for threshold in config["routing_diagnostics"]["thresholds"]:
                route_key = f"{key}:qwen_l{threshold}_baseline_fallback"
                hybrid = lambda row, candidate=candidate, task=task, threshold=threshold: (
                    row["candidates"][candidate]
                    if int(row["qwen_levels"][task]) >= int(threshold)
                    else row["baseline"]["restored"]
                )
                hybrid_metrics = metrics_for(task_rows, hybrid)
                hybrid_bootstrap = bootstrap_for(task_rows, hybrid, config, route_key)
                hybrid_checks = strong_gate_checks(hybrid_metrics, hybrid_bootstrap, gate)
                routing[route_key] = {
                    "triggered_records": sum(
                        int(row["qwen_levels"][task]) >= int(threshold) for row in task_rows
                    ),
                    "metrics": hybrid_metrics,
                    "source_group_bootstrap": hybrid_bootstrap,
                    "strong_gate_checks": hybrid_checks,
                    "strong_gate_pass": all(hybrid_checks.values()),
                }
    return {
        "schema_version": 1,
        "status": "task_native_capacity_audit_complete",
        "config_sha256": config_sha,
        "records": len(rows),
        "candidate_capacity": result_candidates,
        "actual_qwen_hybrid_diagnostics": routing,
        "deployable_selection_frozen": False,
        "sealed_internal_test_accessed": False,
        "protected_foundir_test_accessed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    config_path = resolve_workspace_path(ROOT, args.config)
    config, baseline_config, router, conditions, baseline_result_path = validate_config(config_path)
    config_sha = file_sha256(config_path)
    baseline_result_sha = file_sha256(baseline_result_path)
    rows = load_rows(conditions, int(config["selection"]["manifest_expected_records"]))
    selected = select_rows(rows)
    if args.preflight:
        print(
            "FOUNDIR_NOISE_RAIN_TASK_NATIVE_CAPACITY_PREFLIGHT_PASS "
            + json.dumps(
                {
                    "config_sha256": config_sha,
                    "records": len(selected),
                    "categories": dict(Counter(row["category"] for row in selected)),
                    "baseline_result_sha256": baseline_result_sha,
                    "sealed_internal_test_accessed": False,
                    "protected_foundir_test_accessed": False,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return

    rank, world, local_rank = torchrun_worker_identity(8)
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if torch.cuda.device_count() != world or not any(
        name in torch.cuda.get_device_properties(device).name for name in ("H100", "H200")
    ):
        raise RuntimeError("task-native capacity audit requires eight H100/H200 GPUs")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    models = {}
    checkpoint_dir = resolve_workspace_path(ROOT, config["candidates"]["checkpoint_directory"])
    for name in {value for values in EXPECTED_CANDIDATES.values() for value in values}:
        model, _ = load_task_native_restormer(name, checkpoint_dir)
        models[name] = model.to(device).eval()
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

    metric = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(device).eval()
    output = resolve_workspace_path(ROOT, config["output"])
    sidecars = output / "per_record"
    sidecars.mkdir(parents=True, exist_ok=True)
    for index in range(rank, len(selected), world):
        row = selected[index]
        path = sidecars / f"{evaluation_artifact_stem(row['record_id'])}.json"
        if path.exists():
            verify_sidecar(
                path, row=row, config_sha=config_sha, baseline_result_sha=baseline_result_sha
            )
            continue
        baseline = baseline_sidecar(row, baseline_config, router)
        lq_m11, gt_m11, crop_xy, resized = load_audit_crop(
            row,
            crop_size=int(config["selection"]["crop_size"]),
            seed=int(config["selection"]["crop_seed"]),
        )
        source = ((lq_m11 + 1.0) * 0.5).unsqueeze(0).to(device)
        target = gt_m11.unsqueeze(0).to(device)
        candidates = {}
        for name in candidate_names(row):
            with torch.inference_mode():
                restored = models[name](source).float().clamp(0, 1)
            restored_u8 = (
                restored[0].detach().cpu().permute(1, 2, 0).mul(255).round().byte().numpy()
            )
            restored_m11 = (
                torch.from_numpy(restored_u8.copy())
                .permute(2, 0, 1)
                .float()
                .div_(127.5)
                .sub_(1.0)
                .unsqueeze(0)
            )
            residual = restored - source
            candidates[name] = {
                "psnr": psnr(target, restored_m11.to(device)),
                "lpips": lpips_value(metric, target, restored_m11, device),
                "residual_l1": float(residual.abs().mean().item()),
                "residual_rms": float(residual.square().mean().sqrt().item()),
                "restored_uint8_sha256": hashlib.sha256(restored_u8.tobytes()).hexdigest(),
            }
            metric.reset()
        payload = {
            "schema_version": 1,
            "status": "complete",
            "config_sha256": config_sha,
            "baseline_result_sha256": baseline_result_sha,
            "record_id": row["record_id"],
            "source_group": row["source_group"],
            "category": row["category"],
            "qwen_levels": json.loads(row["qwen_json"]),
            "crop_xy": crop_xy,
            "undersized_pair_resized": resized,
            "baseline": {"input": baseline["input"], "restored": baseline["restored"]},
            "candidates": candidates,
            "sealed_internal_test_accessed": False,
            "protected_foundir_test_accessed": False,
        }
        write_once(path, (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode())
        if (index + 1) % 160 == 0:
            print(f"FOUNDIR_TASK_NATIVE_CAPACITY rank={rank} record={index+1}/{len(selected)}", flush=True)

    def finalize() -> None:
        expected = [
            sidecars / f"{evaluation_artifact_stem(row['record_id'])}.json" for row in selected
        ]
        if set(sidecars.glob("*.json")) != set(expected):
            raise RuntimeError("task-native capacity sidecar set incomplete or contaminated")
        result_rows = [
            verify_sidecar(
                path, row=row, config_sha=config_sha, baseline_result_sha=baseline_result_sha
            )
            for row, path in zip(selected, expected, strict=True)
        ]
        result = build_result(result_rows, config, config_sha)
        write_once(output / "result.json", (json.dumps(result, indent=2, sort_keys=True) + "\n").encode())
        print("FOUNDIR_TASK_NATIVE_CAPACITY_COMPLETE " + json.dumps(result, sort_keys=True), flush=True)

    synchronize_rank_zero_finalization(
        finalize,
        rank=rank,
        world=world,
        output=output,
        identity={
            "config_sha256": config_sha,
            "baseline_result_sha256": baseline_result_sha,
            "records": len(selected),
        },
        phase="Noise/Rain task-native capacity audit",
        timeout_seconds=3 * 60 * 60,
    )


if __name__ == "__main__":
    main()
