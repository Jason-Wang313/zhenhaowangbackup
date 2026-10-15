#!/usr/bin/env python3
"""Audit all Blur/Haze/Low-light-related records in frozen development."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
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
from smoke.evaluate_foundir_five_type_flux2_ddp import (  # noqa: E402
    lpips_value,
    synchronize_rank_zero_finalization,
    torchrun_worker_identity,
    write_once,
)
from smoke.evaluate_foundir_five_type_qwen_routed_integrated_ddp import (  # noqa: E402
    route_decision,
    validate_router_config,
)
from smoke.evaluate_foundir_five_type_released_baselines_ddp import (  # noqa: E402
    released_config,
    run_generalist,
    run_specialist,
    run_weather_specialist,
)
from smoke.evaluate_foundir_noise_rain_full_development_ddp import (  # noqa: E402
    finite_metric,
    load_audit_crop,
    mean_metrics,
    source_group_bootstrap,
    strong_gate_checks,
    verify_sidecar,
)
from smoke.run_foundir_lowlight_gate2_teachers import (  # noqa: E402
    SPECIALIST_DEPENDENCY_DIRECTORIES,
    build_specialist,
    build_weather_specialist,
)
from smoke.train_foundir_blur_generalist_specialization_ddp import build_diffusion  # noqa: E402
from smoke.train_foundir_five_type_flux2_full_ddp import load_rows  # noqa: E402
from smoke.train_foundir_five_type_flux2_overfit_ddp import psnr  # noqa: E402


EXPECTED_CATEGORIES = {
    "01Blur": 2595,
    "02Blur_Noise": 579,
    "08Haze": 1580,
    "09Lowlight_Haze": 1580,
    "13Rain_Haze": 1580,
    "14Lowlight": 1400,
    "15Lowlight_Blur": 2131,
    "16Lowlight_Noise": 1743,
    "18Lowlight_Blur_Noise": 579,
}
EXPECTED_STRONG_GATES = {
    "blur": {
        "category": "01Blur",
        "sow_psnr_minimum": 25.9,
        "sow_lpips_maximum": 0.18,
        "mean_psnr_minimum": 26.9,
        "mean_lpips_maximum": 0.16,
        "group_bootstrap_psnr_lower_minimum": 26.4,
        "group_bootstrap_lpips_upper_maximum": 0.17,
    },
    "haze": {
        "category": "08Haze",
        "sow_psnr_minimum": 22.1,
        "sow_lpips_maximum": 0.19,
        "mean_psnr_minimum": 23.1,
        "mean_lpips_maximum": 0.17,
        "group_bootstrap_psnr_lower_minimum": 22.6,
        "group_bootstrap_lpips_upper_maximum": 0.18,
    },
    "low_light": {
        "category": "14Lowlight",
        "sow_psnr_minimum": 18.9,
        "sow_lpips_maximum": 0.28,
        "mean_psnr_minimum": 19.9,
        "mean_lpips_maximum": 0.26,
        "group_bootstrap_psnr_lower_minimum": 19.4,
        "group_bootstrap_lpips_upper_maximum": 0.27,
    },
}


def validate_audit_config(path: Path) -> tuple[dict, dict, dict, Path]:
    config = json.loads(path.read_text())
    selection = config.get("selection", {})
    metrics = config.get("metrics", {})
    if (
        config.get("schema_version") != 1
        or config.get("status")
        != "frozen_before_blur_haze_lowlight_full_development_audit"
        or config.get("checkpoint_selection_uses_this_development_audit_only") is not True
        or config.get("sealed_internal_test_accessed") is not False
        or config.get("protected_foundir_test_accessed") is not False
        or config.get("strong_gates") != EXPECTED_STRONG_GATES
        or selection.get("split") != "validation"
        or selection.get("policy")
        != "all_blur_or_haze_or_lowlight_related_records_in_frozen_validation_split"
        or selection.get("categories") != EXPECTED_CATEGORIES
        or int(selection.get("expected_records", -1)) != sum(EXPECTED_CATEGORIES.values())
        or int(selection.get("crop_size", -1)) != 512
        or selection.get("undersized_pair_policy")
        != "bicubic_short_side_to_512_then_aligned_crop_v1"
        or metrics.get("names") != ["psnr", "lpips"]
        or metrics.get("quantization")
        != "round_clip_uint8_png_equivalent_before_metrics"
        or metrics.get("bootstrap_unit") != "source_group_equal_weight"
        or int(metrics.get("bootstrap_resamples", -1)) != 5000
        or float(metrics.get("bootstrap_confidence", -1)) != 0.95
        or int(metrics.get("bootstrap_seed", -1)) != 2026091007
        or int(metrics.get("worst_records_per_category", -1)) != 20
    ):
        raise RuntimeError("Blur/Haze/Low-light full-development audit contract changed")
    router_path = resolve_workspace_path(ROOT, config["router_config"])
    conditions = resolve_workspace_path(ROOT, config["conditions"])
    if file_sha256(router_path) != config["router_config_sha256"]:
        raise RuntimeError("Blur/Haze/Low-light audit router identity changed")
    if file_sha256(conditions) != config["conditions_sha256"]:
        raise RuntimeError("Blur/Haze/Low-light audit condition identity changed")
    router, parent, specialization, checkpoint = validate_router_config(router_path)
    if router["conditions"] != config["conditions"]:
        raise RuntimeError("Blur/Haze/Low-light conditions differ from frozen router")
    return config, parent, specialization, checkpoint


def select_audit_rows(
    rows: Sequence[Mapping[str, str]], config: Mapping[str, object]
) -> list[dict[str, str]]:
    selected = [
        dict(row)
        for row in rows
        if row["split"] == "validation" and row["category"] in EXPECTED_CATEGORIES
    ]
    counts = Counter(row["category"] for row in selected)
    if counts != Counter(EXPECTED_CATEGORIES) or len(selected) != int(
        config["selection"]["expected_records"]
    ):
        raise RuntimeError(f"Blur/Haze/Low-light full-development coverage changed: {counts}")
    if len({row["record_id"] for row in selected}) != len(selected):
        raise RuntimeError("Blur/Haze/Low-light development record IDs are not unique")
    return selected


def build_result(
    rows: Sequence[Mapping[str, object]], config: Mapping[str, object], *, config_sha: str
) -> dict[str, object]:
    by_category: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    by_route: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        by_category[str(row["category"])].append(row)
        by_route[str(row["decision"]["route"])].append(row)
    category_metrics = {
        category: mean_metrics(values) for category, values in sorted(by_category.items())
    }
    route_metrics = {route: mean_metrics(values) for route, values in sorted(by_route.items())}
    bootstrap = {}
    checks = {}
    metric_config = config["metrics"]
    for task, gate in config["strong_gates"].items():
        values = by_category[gate["category"]]
        bootstrap[task] = source_group_bootstrap(
            values,
            resamples=int(metric_config["bootstrap_resamples"]),
            confidence=float(metric_config["bootstrap_confidence"]),
            seed=int(stable_digest(int(metric_config["bootstrap_seed"]), task)[:16], 16),
        )
        checks[task] = strong_gate_checks(
            category_metrics[gate["category"]], bootstrap[task], gate
        )
    worst_count = int(metric_config["worst_records_per_category"])
    worst = {
        category: {
            "psnr": [
                row["record_id"]
                for row in sorted(values, key=lambda item: item["restored"]["psnr"])[
                    :worst_count
                ]
            ],
            "lpips": [
                row["record_id"]
                for row in sorted(values, key=lambda item: -item["restored"]["lpips"])[
                    :worst_count
                ]
            ],
        }
        for category, values in sorted(by_category.items())
    }
    result = {
        "schema_version": 1,
        "status": "blur_haze_lowlight_full_development_audit_complete",
        "config_sha256": config_sha,
        "split": "validation",
        "records": len(rows),
        "category_metrics": category_metrics,
        "route_metrics": route_metrics,
        "route_counts": dict(sorted(Counter(row["decision"]["route"] for row in rows).items())),
        "source_group_bootstrap": bootstrap,
        "strong_gate_checks": checks,
        "strong_gate_pass": {task: all(values.values()) for task, values in checks.items()},
        "worst_record_ids": worst,
        "sealed_internal_test_accessed": False,
        "protected_foundir_test_accessed": False,
    }
    if not all(
        finite_metric(value)
        for metrics in category_metrics.values()
        for key, value in metrics.items()
        if key not in ("records", "source_groups")
    ):
        raise RuntimeError("Blur/Haze/Low-light aggregate contains non-finite metrics")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    config_path = resolve_workspace_path(ROOT, args.config)
    config, parent, specialization, blur_checkpoint = validate_audit_config(config_path)
    config_sha = file_sha256(config_path)
    router_path = resolve_workspace_path(ROOT, config["router_config"])
    router = json.loads(router_path.read_text())
    rows = load_rows(
        resolve_workspace_path(ROOT, config["conditions"]),
        int(parent["data"]["expected_five_type_records"]),
    )
    selected = select_audit_rows(rows, config)
    if args.preflight:
        print(
            "FOUNDIR_BLUR_HAZE_LOWLIGHT_FULL_DEVELOPMENT_PREFLIGHT_PASS "
            + json.dumps(
                {
                    "config_sha256": config_sha,
                    "records": len(selected),
                    "categories": dict(sorted(Counter(row["category"] for row in selected).items())),
                    "sealed_internal_test_accessed": False,
                    "protected_foundir_test_accessed": False,
                },
                sort_keys=True,
            ),
            flush=True,
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
    rank, world, local_rank = torchrun_worker_identity(8)
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if torch.cuda.device_count() != world or not any(
        name in torch.cuda.get_device_properties(device).name for name in ("H100", "H200")
    ):
        raise RuntimeError("Blur/Haze/Low-light audit requires eight H100/H200 GPUs")
    pretrained = resolve_workspace_path(ROOT, specialization["pretrained_checkpoint"])
    generalist = build_diffusion(specialization, pretrained, device).eval()
    blur_model = build_diffusion(specialization, pretrained, device)
    checkpoint_payload = torch.load(blur_checkpoint, map_location="cpu", weights_only=False)
    blur_model.model.load_state_dict(checkpoint_payload["model"], strict=True)
    del checkpoint_payload
    blur_model.eval()
    lowlight = build_specialist(released_config(parent), device).eval()
    weather_path = resolve_workspace_path(ROOT, parent["released_baseline"]["weather_specialist"])
    weather = build_weather_specialist(released_config(parent, weather_path), device).eval()
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

    lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(device).eval()
    output = resolve_workspace_path(ROOT, config["output"])
    sidecars = output / "per_record"
    sidecars.mkdir(parents=True, exist_ok=True)
    router_sha = file_sha256(router_path)
    crop_size = int(config["selection"]["crop_size"])
    crop_seed = int(parent["data"]["split_seed"])
    inference_seed = int(router["evaluation"]["seed"])
    for index in range(rank, len(selected), world):
        row = selected[index]
        levels = json.loads(row["qwen_json"])
        decision = route_decision(levels, router["routing"])
        sidecar = sidecars / f"{evaluation_artifact_stem(row['record_id'])}.json"
        if sidecar.exists():
            verify_sidecar(
                sidecar,
                row=row,
                decision=decision,
                config_sha=config_sha,
                router_sha=router_sha,
            )
            continue
        lq_m11, gt_m11, crop_xy, resized = load_audit_crop(
            row, crop_size=crop_size, seed=crop_seed
        )
        source = ((lq_m11 + 1.0) * 0.5).unsqueeze(0).to(device)
        target = gt_m11.unsqueeze(0).to(device)
        record_seed = int(
            stable_digest(inference_seed, "qwen-routed-integrated", row["record_id"])[:16], 16
        )
        random.seed(record_seed)
        np.random.seed(record_seed % (2**32 - 1))
        torch.manual_seed(record_seed)
        torch.cuda.manual_seed_all(record_seed)
        base = blur_model if decision["base"] == "blur_specialization" else generalist
        restored = run_generalist(base, source, device, row["record_id"])
        if decision["refinement"] == "low_light":
            restored = run_specialist(lowlight, restored)
        elif decision["refinement"] == "weather":
            restored = run_weather_specialist(weather, restored)
        output_u8 = restored[0].detach().cpu().permute(1, 2, 0).mul(255).round().byte().numpy()
        output_m11 = (
            torch.from_numpy(output_u8.copy())
            .permute(2, 0, 1)
            .float()
            .div_(127.5)
            .sub_(1.0)
            .unsqueeze(0)
        )
        input_metrics = {
            "psnr": psnr(target, lq_m11.unsqueeze(0).to(device)),
            "lpips": lpips_value(lpips_metric, target, lq_m11.unsqueeze(0), device),
        }
        lpips_metric.reset()
        restored_metrics = {
            "psnr": psnr(target, output_m11.to(device)),
            "lpips": lpips_value(lpips_metric, target, output_m11, device),
        }
        lpips_metric.reset()
        payload = {
            "schema_version": 1,
            "status": "complete",
            "config_sha256": config_sha,
            "router_config_sha256": router_sha,
            "split": "validation",
            "record_id": row["record_id"],
            "source_group": row["source_group"],
            "category": row["category"],
            "composition_size": int(row["composition_size"]),
            "qwen_levels": levels,
            "decision": decision,
            "crop_xy": crop_xy,
            "undersized_pair_resized": resized,
            "input": input_metrics,
            "restored": restored_metrics,
            "restored_uint8_sha256": hashlib.sha256(output_u8.tobytes()).hexdigest(),
            "sealed_internal_test_accessed": False,
            "protected_foundir_test_accessed": False,
        }
        write_once(sidecar, (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode())
        if (index + 1) % 80 == 0:
            print(
                f"FOUNDIR_BLUR_HAZE_LOWLIGHT_FULL_DEVELOPMENT rank={rank} "
                f"record={index+1}/{len(selected)}",
                flush=True,
            )

    def finalize() -> None:
        expected_paths = [
            sidecars / f"{evaluation_artifact_stem(row['record_id'])}.json" for row in selected
        ]
        if set(sidecars.glob("*.json")) != set(expected_paths):
            raise RuntimeError("Blur/Haze/Low-light sidecar set is incomplete or contaminated")
        result_rows = [
            verify_sidecar(
                path,
                row=row,
                decision=route_decision(json.loads(row["qwen_json"]), router["routing"]),
                config_sha=config_sha,
                router_sha=router_sha,
            )
            for row, path in zip(selected, expected_paths, strict=True)
        ]
        result = build_result(result_rows, config, config_sha=config_sha)
        write_once(
            output / "result.json", (json.dumps(result, indent=2, sort_keys=True) + "\n").encode()
        )
        print(
            "FOUNDIR_BLUR_HAZE_LOWLIGHT_FULL_DEVELOPMENT_COMPLETE "
            + json.dumps(result, sort_keys=True),
            flush=True,
        )

    synchronize_rank_zero_finalization(
        finalize,
        rank=rank,
        world=world,
        output=output,
        identity={
            "config_sha256": config_sha,
            "router_config_sha256": router_sha,
            "conditions_sha256": config["conditions_sha256"],
            "records": len(selected),
        },
        phase="Blur/Haze/Low-light full development audit",
        timeout_seconds=6 * 60 * 60,
    )


if __name__ == "__main__":
    main()
