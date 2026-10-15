#!/usr/bin/env python3
"""Evaluate the frozen five-type system using actual Qwen routing decisions."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lowlight_restoration.flux_prompt_cache import file_sha256  # noqa: E402
from lowlight_restoration.foundir_flux2_protocol import (  # noqa: E402
    SOW_STAGE2_THRESHOLDS,
    aggregate_task_metrics,
    evaluation_artifact_stem,
    resolve_workspace_path,
    stable_digest,
    worst_normalized_margin,
)
from smoke.evaluate_foundir_five_type_flux2_ddp import (  # noqa: E402
    load_aligned_crop,
    lpips_value,
    selected_from_contract,
    synchronize_rank_zero_finalization,
    torchrun_worker_identity,
    write_once,
    write_or_verify_png,
)
from smoke.evaluate_foundir_five_type_released_baselines_ddp import (  # noqa: E402
    released_config,
    run_generalist,
    run_specialist,
    run_weather_specialist,
)
from smoke.run_foundir_lowlight_gate2_teachers import (  # noqa: E402
    SPECIALIST_DEPENDENCY_DIRECTORIES,
    build_specialist,
    build_weather_specialist,
)
from smoke.train_foundir_blur_generalist_specialization_ddp import build_diffusion  # noqa: E402
from smoke.train_foundir_five_type_flux2_full_ddp import load_rows  # noqa: E402
from smoke.train_foundir_five_type_flux2_overfit_ddp import psnr  # noqa: E402


QWEN_KEYS = ("blur", "noise", "rain", "haze", "low_light")
EXPECTED_PRECEDENCE = ["low_light", "weather", "none"]
LEGACY_POLICY = "qwen_active_l2_cascade_v1"
CONFIDENCE_GUARDED_POLICY = "development_selected_confidence_guarded_v2"


def finite_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def route_decision(
    levels: Mapping[str, object],
    routing: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Return a category-blind route derived only from actual Qwen levels."""

    if set(levels) != set(QWEN_KEYS) or any(
        isinstance(levels[key], bool)
        or not isinstance(levels[key], int)
        or not 1 <= int(levels[key]) <= 5
        for key in QWEN_KEYS
    ):
        raise RuntimeError("invalid frozen Qwen five-type levels")
    active = {key for key in QWEN_KEYS if int(levels[key]) >= 2}
    policy = LEGACY_POLICY if routing is None else routing.get("policy", LEGACY_POLICY)
    if policy == LEGACY_POLICY:
        base = "blur_specialization" if "blur" in active else "generalist"
        if "low_light" in active:
            refinement = "low_light"
        elif active & {"rain", "haze"}:
            refinement = "weather"
        else:
            refinement = "none"
    elif policy == CONFIDENCE_GUARDED_POLICY:
        blur_active = int(levels["blur"]) >= int(routing["blur_level_minimum"])
        low_light_active = int(levels["low_light"]) >= int(
            routing["low_light_level_minimum"]
        )
        haze_active = int(levels["haze"]) >= int(
            routing["weather_haze_level_minimum"]
        )
        base = "blur_specialization" if blur_active else "generalist"
        if low_light_active:
            refinement = "low_light"
        elif haze_active and base != "blur_specialization":
            refinement = "weather"
        else:
            refinement = "none"
    else:
        raise RuntimeError(f"unsupported integrated Qwen routing policy: {policy}")
    decision = {
        "active_types": sorted(active),
        "base": base,
        "refinement": refinement,
        "route": f"{base}+{refinement}",
    }
    if policy != LEGACY_POLICY:
        decision["policy"] = policy
    return decision


def validate_router_config(path: Path) -> tuple[dict, dict, dict, Path]:
    router = json.loads(path.read_text())
    if (
        router.get("schema_version") != 1
        or router.get("status") != "frozen_development_selected_qwen_routed_integrated_protocol"
        or router.get("checkpoint_and_routing_selected_on_development_only") is not True
        or router.get("sealed_internal_test_accessed_for_this_protocol") is not False
        or router.get("protected_foundir_test_accessed") is not False
    ):
        raise RuntimeError("integrated Qwen routing contract is invalid")
    routing = router.get("routing", {})
    common_routing_valid = (
        routing.get("condition_source") != "frozen_actual_physical_v8_qwen_json"
        or routing.get("refinement_precedence") != EXPECTED_PRECEDENCE
        or routing.get("noise_uses_base_without_a_separate_specialist") is not True
        or routing.get("specialist_input") != "unquantized_selected_base_output"
        or routing.get("ground_truth_category_must_not_control_route") is not True
    )
    policy = routing.get("policy", LEGACY_POLICY)
    if policy == LEGACY_POLICY:
        policy_valid = (
            routing.get("active_level_minimum") == 2
            and routing.get("base")
            == "selected_blur_specialization_if_qwen_blur_active_else_released_generalist"
            and routing.get("low_light_trigger") == "qwen_low_light_active"
            and routing.get("weather_trigger") == "qwen_rain_or_haze_active"
            and routing.get("weather_is_skipped_when_low_light_active") is True
        )
    elif policy == CONFIDENCE_GUARDED_POLICY:
        policy_valid = (
            routing.get("active_level_minimum") == 2
            and routing.get("blur_level_minimum") == 4
            and routing.get("low_light_level_minimum") == 3
            and routing.get("weather_haze_level_minimum") == 2
            and routing.get("base")
            == "selected_blur_specialization_if_qwen_blur_l4plus_else_released_generalist"
            and routing.get("low_light_trigger") == "qwen_low_light_l3plus"
            and routing.get("weather_trigger")
            == "qwen_haze_l2plus_without_blur_specialization"
            and routing.get("rain_alone_uses_generalist") is True
            and routing.get("weather_is_skipped_when_blur_base") is True
            and routing.get("weather_is_skipped_when_low_light_active") is True
        )
    else:
        policy_valid = False
    if common_routing_valid or not policy_valid:
        raise RuntimeError("integrated Qwen route policy changed")
    identities = (
        ("parent_config", "parent_config_sha256"),
        ("conditions", "conditions_sha256"),
        ("evaluation_contract_csv", "evaluation_contract_csv_sha256"),
        ("evaluation_contract_json", "evaluation_contract_json_sha256"),
        ("blur_selection", "blur_selection_sha256"),
    )
    resolved = {}
    for path_key, hash_key in identities:
        resolved[path_key] = resolve_workspace_path(ROOT, router[path_key])
        if file_sha256(resolved[path_key]) != router[hash_key]:
            raise RuntimeError(f"integrated routing identity changed: {path_key}")
    if policy == CONFIDENCE_GUARDED_POLICY:
        evidence = router.get("development_selection_evidence", {})
        evidence_identities = (
            ("v1_router_config", "v1_router_config_sha256"),
            ("v1_development_result", "v1_development_result_sha256"),
            ("v1_development_freeze", "v1_development_freeze_sha256"),
            ("released_development_result", "released_development_result_sha256"),
        )
        if (
            evidence.get("selection_split") != "development_only"
            or evidence.get("sealed_internal_test_accessed_for_selection") is not False
            or evidence.get("protected_foundir_test_accessed") is not False
        ):
            raise RuntimeError("confidence-guarded route selection evidence changed")
        for path_key, hash_key in evidence_identities:
            evidence_path = resolve_workspace_path(ROOT, evidence[path_key])
            if file_sha256(evidence_path) != evidence[hash_key]:
                raise RuntimeError(
                    f"confidence-guarded route evidence identity changed: {path_key}"
                )
    parent = json.loads(resolved["parent_config"].read_text())
    if (
        parent["full_conditions"]["csv"] != router["conditions"]
        or parent["evaluation"]["contract_csv"] != router["evaluation_contract_csv"]
        or parent["evaluation"]["contract_json"] != router["evaluation_contract_json"]
        or int(router["evaluation"]["world_size"]) != 8
        or int(router["evaluation"]["crop_size"]) != int(parent["data"]["crop_size"])
        or router["evaluation"]["metric_contract"] != "sow_isolated_categories_v1"
        or router["evaluation"]["metrics"] != ["psnr", "lpips"]
    ):
        raise RuntimeError("integrated routing parent/evaluation contract changed")
    selection = json.loads(resolved["blur_selection"].read_text())
    if (
        selection.get("status") != "frozen_blur_specialization_development_selection"
        or selection.get("selected_step") != 3000
        or selection.get("sealed_internal_test_accessed") is not False
        or selection.get("protected_foundir_test_accessed") is not False
    ):
        raise RuntimeError("Blur development selection changed")
    specialization_config_path = resolve_workspace_path(ROOT, selection["config"])
    if file_sha256(specialization_config_path) != selection["config_sha256"]:
        raise RuntimeError("Blur specialization config changed")
    specialization_config = json.loads(specialization_config_path.read_text())
    checkpoint = resolve_workspace_path(ROOT, selection["checkpoint"])
    if file_sha256(checkpoint) != selection["checkpoint_sha256"]:
        raise RuntimeError("selected Blur specialization checkpoint changed")
    for path_key, hash_key in (
        ("generalist", "generalist_sha256"),
        ("lowlight_specialist", "lowlight_specialist_sha256"),
        ("weather_specialist", "weather_specialist_sha256"),
    ):
        released = resolve_workspace_path(ROOT, parent["released_baseline"][path_key])
        if file_sha256(released) != parent["released_baseline"][hash_key]:
            raise RuntimeError(f"released integrated component changed: {path_key}")
    return router, parent, specialization_config, checkpoint


def verify_development_freeze(router_path: Path, router: Mapping[str, object]) -> str:
    freeze_path = resolve_workspace_path(ROOT, router["outputs"]["development_freeze"])
    payload = json.loads(freeze_path.read_text())
    development = resolve_workspace_path(ROOT, router["outputs"]["development"]) / "result.json"
    if (
        payload.get("schema_version") != 1
        or payload.get("status") != "frozen_qwen_routed_integrated_development_result"
        or payload.get("router_config_sha256") != file_sha256(router_path)
        or payload.get("development_result") != development.relative_to(ROOT).as_posix()
        or payload.get("development_result_sha256") != file_sha256(development)
        or payload.get("sealed_internal_test_accessed") is not False
        or payload.get("protected_foundir_test_accessed") is not False
    ):
        raise RuntimeError("integrated development freeze changed before internal evaluation")
    return file_sha256(freeze_path)


def sidecar_identity(
    *,
    router_sha: str,
    checkpoint_sha: str,
    conditions_sha: str,
    split: str,
    row: Mapping[str, str],
    decision: Mapping[str, object],
) -> dict[str, object]:
    return {
        "router_config_sha256": router_sha,
        "blur_checkpoint_sha256": checkpoint_sha,
        "conditions_sha256": conditions_sha,
        "split": split,
        "record_id": row["record_id"],
        "source_group": row["source_group"],
        "category": row["category"],
        "composition_size": int(row["composition_size"]),
        "evaluation_fold": int(row["evaluation_fold"]),
        "decision": dict(decision),
        "protected_foundir_test_accessed": False,
    }


def verify_sidecar(
    path: Path,
    *,
    identity: Mapping[str, object],
) -> dict[str, object]:
    payload = json.loads(path.read_text())
    if payload.get("status") != "complete" or any(
        payload.get(key) != value for key, value in identity.items()
    ):
        raise RuntimeError(f"integrated routing sidecar identity changed: {path}")
    for name in ("input", "restored"):
        metrics = payload.get(name, {})
        if not all(finite_number(metrics.get(metric)) for metric in ("psnr", "lpips")):
            raise RuntimeError(f"integrated routing metrics are invalid: {path}")
    output = resolve_workspace_path(ROOT, payload["output_path"])
    if file_sha256(output) != payload.get("output_sha256"):
        raise RuntimeError(f"integrated routing output changed: {output}")
    return payload


def build_result(
    rows: Sequence[Mapping[str, object]],
    *,
    router_path: Path,
    router: Mapping[str, object],
    checkpoint: Path,
    split: str,
    development_freeze_sha: str | None,
) -> dict[str, object]:
    expected = int(
        router["evaluation"][
            "development_records" if split == "validation" else "sealed_internal_records"
        ]
    )
    if len(rows) != expected or len({row["record_id"] for row in rows}) != expected:
        raise RuntimeError("integrated routing result coverage changed")
    restored_rows = [
        {"category": row["category"], **row["restored"]}
        for row in rows
    ]
    input_rows = [
        {"category": row["category"], **row["input"]}
        for row in rows
    ]
    task_metrics = aggregate_task_metrics(restored_rows)
    input_task_metrics = aggregate_task_metrics(input_rows)
    checks = {
        task: {
            "psnr": float(metrics["psnr"]) >= SOW_STAGE2_THRESHOLDS[task]["psnr"],
            "lpips": float(metrics["lpips"]) <= SOW_STAGE2_THRESHOLDS[task]["lpips"],
        }
        for task, metrics in task_metrics.items()
    }
    route_counts = dict(sorted(Counter(row["decision"]["route"] for row in rows).items()))
    if split == "validation" and router["routing"].get("policy") == CONFIDENCE_GUARDED_POLICY:
        projected = router["development_selection_evidence"].get(
            "projected_development_route_counts"
        )
        if route_counts != projected:
            raise RuntimeError(
                "confidence-guarded development route counts changed from selection evidence"
            )
    result: dict[str, object] = {
        "schema_version": 1,
        "status": (
            "qwen_routed_integrated_development_complete"
            if split == "validation"
            else "qwen_routed_integrated_internal_complete"
        ),
        "router_config": router_path.relative_to(ROOT).as_posix(),
        "router_config_sha256": file_sha256(router_path),
        "blur_checkpoint": checkpoint.relative_to(ROOT).as_posix(),
        "blur_checkpoint_sha256": file_sha256(checkpoint),
        "split": split,
        "records": len(rows),
        "routing": router["routing"],
        "route_counts": route_counts,
        "input_task_metrics": input_task_metrics,
        "task_metrics": task_metrics,
        "checks": checks,
        "gate_pass": all(value for task in checks.values() for value in task.values()),
        "worst_normalized_margin": worst_normalized_margin(task_metrics),
        "sealed_internal_reuse_disclosure": split == "internal_test",
        "protected_foundir_test_accessed": False,
    }
    if development_freeze_sha is not None:
        result["development_freeze_sha256"] = development_freeze_sha
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--router-config", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "internal_test"), required=True)
    args = parser.parse_args()
    router_path = resolve_workspace_path(ROOT, args.router_config)
    router, parent, specialization_config, checkpoint = validate_router_config(router_path)
    router_sha = file_sha256(router_path)
    checkpoint_sha = file_sha256(checkpoint)
    conditions = resolve_workspace_path(ROOT, router["conditions"])
    conditions_sha = file_sha256(conditions)
    development_freeze_sha = None
    if args.split == "internal_test":
        development_freeze_sha = verify_development_freeze(router_path, router)
    rows = load_rows(conditions, int(parent["data"]["expected_five_type_records"]))
    selected = selected_from_contract(
        parent,
        rows,
        split=args.split,
        config_sha=file_sha256(resolve_workspace_path(ROOT, router["parent_config"])),
        conditions_sha=conditions_sha,
    )
    expected = int(
        router["evaluation"][
            "development_records" if args.split == "validation" else "sealed_internal_records"
        ]
    )
    if len(selected) != expected:
        raise RuntimeError("integrated routing selected-record count changed")
    output = resolve_workspace_path(ROOT, router["outputs"][
        "development" if args.split == "validation" else "sealed_internal"
    ])
    extra = ROOT / ".tools/foundir-runtime"
    if not extra.is_dir():
        raise RuntimeError(f"missing released FoundIR runtime: {extra}")
    sys.path.insert(0, str(extra))
    for dependency in reversed(SPECIALIST_DEPENDENCY_DIRECTORIES):
        if not dependency.is_dir():
            raise RuntimeError(f"missing released specialist dependency: {dependency}")
        sys.path.insert(0, str(dependency))

    rank, world, local_rank = torchrun_worker_identity(int(router["evaluation"]["world_size"]))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if torch.cuda.device_count() != world or not any(
        name in torch.cuda.get_device_properties(device).name for name in ("H100", "H200")
    ):
        raise RuntimeError("integrated routing evaluation requires eight H100/H200 GPUs")

    pretrained = resolve_workspace_path(ROOT, specialization_config["pretrained_checkpoint"])
    generalist = build_diffusion(specialization_config, pretrained, device).eval()
    blur_model = build_diffusion(specialization_config, pretrained, device)
    checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    blur_model.model.load_state_dict(checkpoint_payload["model"], strict=True)
    del checkpoint_payload
    blur_model.eval()
    lowlight = build_specialist(released_config(parent), device).eval()
    weather_path = resolve_workspace_path(ROOT, parent["released_baseline"]["weather_specialist"])
    weather = build_weather_specialist(released_config(parent, weather_path), device).eval()
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

    metric = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(device).eval()
    sidecars = output / "per_record"
    images = output / "images"
    sidecars.mkdir(parents=True, exist_ok=True)
    images.mkdir(parents=True, exist_ok=True)
    seed = int(router["evaluation"]["seed"])
    for index in range(rank, len(selected), world):
        row = selected[index]
        levels = json.loads(row["qwen_json"])
        decision = route_decision(levels, router["routing"])
        identity = sidecar_identity(
            router_sha=router_sha,
            checkpoint_sha=checkpoint_sha,
            conditions_sha=conditions_sha,
            split=args.split,
            row=row,
            decision=decision,
        )
        stem = evaluation_artifact_stem(row["record_id"])
        sidecar = sidecars / f"{stem}.json"
        if sidecar.exists():
            verify_sidecar(sidecar, identity=identity)
            continue
        lq_m11, gt_m11, crop_xy = load_aligned_crop(
            row,
            crop_size=int(router["evaluation"]["crop_size"]),
            seed=int(parent["data"]["split_seed"]),
        )
        source = ((lq_m11 + 1.0) * 0.5).unsqueeze(0).to(device)
        target = gt_m11.unsqueeze(0).to(device)
        record_seed = int(stable_digest(seed, "qwen-routed-integrated", row["record_id"])[:16], 16)
        random.seed(record_seed)
        np.random.seed(record_seed % (2**32 - 1))
        torch.manual_seed(record_seed)
        torch.cuda.manual_seed_all(record_seed)
        base_model = blur_model if decision["base"] == "blur_specialization" else generalist
        restored = run_generalist(base_model, source, device, row["record_id"])
        if decision["refinement"] == "low_light":
            restored = run_specialist(lowlight, restored)
        elif decision["refinement"] == "weather":
            restored = run_weather_specialist(weather, restored)
        array = restored[0].detach().cpu().permute(1, 2, 0).mul(255).round().byte().numpy()
        output_q_m11 = (
            torch.from_numpy(array.copy()).permute(2, 0, 1).float().div_(127.5).sub_(1.0).unsqueeze(0).to(device)
        )
        input_metrics = {
            "psnr": psnr(target, lq_m11.unsqueeze(0).to(device)),
            "lpips": lpips_value(metric, target, lq_m11.unsqueeze(0), device),
        }
        metric.reset()
        restored_metrics = {
            "psnr": psnr(target, output_q_m11),
            "lpips": lpips_value(metric, target, output_q_m11.cpu(), device),
        }
        metric.reset()
        output_path = images / f"{stem}.png"
        write_or_verify_png(output_path, array, label="Qwen-routed integrated")
        payload = {
            "schema_version": 1,
            "status": "complete",
            **identity,
            "qwen_levels": levels,
            "crop_xy": crop_xy,
            "input": input_metrics,
            "restored": restored_metrics,
            "output_path": output_path.relative_to(ROOT).as_posix(),
            "output_sha256": file_sha256(output_path),
        }
        write_once(sidecar, (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode())
        print(
            f"FOUNDIR_QWEN_ROUTED_INTEGRATED rank={rank} record={index+1}/{len(selected)} "
            f"route={decision['route']}",
            flush=True,
        )

    def finalize() -> None:
        expected_paths = [
            sidecars / f"{evaluation_artifact_stem(row['record_id'])}.json"
            for row in selected
        ]
        if set(sidecars.glob("*.json")) != set(expected_paths):
            raise RuntimeError("integrated routing sidecar set is incomplete or contaminated")
        result_rows = []
        for row, path in zip(selected, expected_paths, strict=True):
            decision = route_decision(json.loads(row["qwen_json"]), router["routing"])
            identity = sidecar_identity(
                router_sha=router_sha,
                checkpoint_sha=checkpoint_sha,
                conditions_sha=conditions_sha,
                split=args.split,
                row=row,
                decision=decision,
            )
            result_rows.append(verify_sidecar(path, identity=identity))
        result = build_result(
            result_rows,
            router_path=router_path,
            router=router,
            checkpoint=checkpoint,
            split=args.split,
            development_freeze_sha=development_freeze_sha,
        )
        write_once(output / "result.json", (json.dumps(result, indent=2, sort_keys=True) + "\n").encode())
        print("FOUNDIR_QWEN_ROUTED_INTEGRATED_PASS " + json.dumps(result, sort_keys=True), flush=True)

    synchronize_rank_zero_finalization(
        finalize,
        rank=rank,
        world=world,
        output=output,
        identity={
            "router_config_sha256": router_sha,
            "blur_checkpoint_sha256": checkpoint_sha,
            "conditions_sha256": conditions_sha,
            "split": args.split,
            "development_freeze_sha256": development_freeze_sha,
        },
        phase=f"Qwen-routed integrated {args.split}",
        timeout_seconds=float(parent["distributed"]["timeout_minutes"]) * 60,
    )


if __name__ == "__main__":
    main()
