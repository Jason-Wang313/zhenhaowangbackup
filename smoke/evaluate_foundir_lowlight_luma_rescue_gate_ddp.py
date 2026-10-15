#!/usr/bin/env python3
"""Evaluate category-blind luma rescue for ambiguous Qwen low-light routes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from PIL import Image

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
)
from smoke.evaluate_foundir_noise_rain_full_development_ddp import (  # noqa: E402
    finite_metric,
    load_audit_crop,
    mean_metrics,
    source_group_bootstrap,
    strong_gate_checks,
)
from smoke.run_foundir_lowlight_gate2_teachers import (  # noqa: E402
    SPECIALIST_DEPENDENCY_DIRECTORIES,
    build_specialist,
)
from smoke.train_foundir_blur_generalist_specialization_ddp import build_diffusion  # noqa: E402
from smoke.train_foundir_five_type_flux2_full_ddp import load_rows  # noqa: E402
from smoke.train_foundir_five_type_flux2_overfit_ddp import psnr  # noqa: E402


EXPECTED_CATEGORIES = {
    "01Blur": 2595,
    "02Blur_Noise": 579,
    "05Noise": 1859,
    "08Haze": 1580,
    "09Lowlight_Haze": 1580,
    "10Rain": 835,
    "13Rain_Haze": 1580,
    "14Lowlight": 1400,
    "15Lowlight_Blur": 2131,
    "16Lowlight_Noise": 1743,
    "18Lowlight_Blur_Noise": 579,
}
ISOLATED_TASKS = {
    "blur": "01Blur",
    "noise": "05Noise",
    "rain": "10Rain",
    "haze": "08Haze",
    "low_light": "14Lowlight",
}
EXPECTED_CANDIDATES = {
    "current_qwen_l3": (None, None),
    "l2_dark_0175": (0.175, None),
    "l2_dark_0200": (0.2, None),
    "l2_dark_0225": (0.225, None),
    "l2_dark_0200_l1_extreme_0075": (0.2, 0.075),
    "l2_dark_0200_l1_extreme_0100": (0.2, 0.1),
}


def candidate_triggers(
    qwen_level: int, mean_luma: float, candidate: Mapping[str, object]
) -> bool:
    if isinstance(qwen_level, bool) or not isinstance(qwen_level, int) or not 1 <= qwen_level <= 5:
        raise RuntimeError("invalid Qwen low-light level")
    if not finite_metric(mean_luma) or not 0.0 <= float(mean_luma) <= 1.0:
        raise RuntimeError("invalid input mean luma")
    if qwen_level >= int(candidate["qwen_level_minimum"]):
        return True
    threshold = (
        candidate["l2_mean_luma_maximum"]
        if qwen_level == 2
        else candidate["l1_mean_luma_maximum"]
    )
    return threshold is not None and float(mean_luma) <= float(threshold)


def crop_mean_luma(lq_m11: torch.Tensor, weights: Sequence[float]) -> float:
    if tuple(lq_m11.shape) != (3, 512, 512):
        raise RuntimeError(f"unexpected frozen crop shape: {tuple(lq_m11.shape)}")
    values = (lq_m11.float() + 1.0).mul(0.5)
    weight = torch.tensor(weights, dtype=values.dtype).view(3, 1, 1)
    result = float((values * weight).sum(dim=0).mean().item())
    if not 0.0 <= result <= 1.0:
        raise RuntimeError("computed input mean luma is out of range")
    return result


def validate_config(path: Path) -> tuple[dict, dict, dict, dict, Path]:
    config = json.loads(path.read_text())
    if (
        config.get("schema_version") != 1
        or config.get("status") != "frozen_before_lowlight_luma_rescue_development_gate"
        or config.get("checkpoint_and_router_selection_uses_development_only") is not True
        or config.get("sealed_internal_test_accessed") is not False
        or config.get("protected_foundir_test_accessed") is not False
        or config.get("selection", {}).get("split") != "validation"
        or int(config.get("selection", {}).get("baseline_records", -1)) != sum(EXPECTED_CATEGORIES.values())
        or int(config.get("selection", {}).get("counterfactual_records", -1)) != 1076
        or int(config.get("selection", {}).get("crop_size", -1)) != 512
        or config.get("luma", {}).get("weights") != [0.2126, 0.7152, 0.0722]
        or set(config.get("strong_gates", {})) != set(ISOLATED_TASKS)
        or set(config.get("candidates", {})) != set(EXPECTED_CANDIDATES)
    ):
        raise RuntimeError("low-light luma rescue gate contract changed")
    for name, (l2, l1) in EXPECTED_CANDIDATES.items():
        candidate = config["candidates"][name]
        if (
            candidate.get("qwen_level_minimum") != 3
            or candidate.get("l2_mean_luma_maximum") != l2
            or candidate.get("l1_mean_luma_maximum") != l1
        ):
            raise RuntimeError(f"low-light rescue candidate changed: {name}")
    router_path = resolve_workspace_path(ROOT, config["router_config"])
    conditions = resolve_workspace_path(ROOT, config["conditions"])
    if file_sha256(router_path) != config["router_config_sha256"]:
        raise RuntimeError("low-light rescue router identity changed")
    if file_sha256(conditions) != config["conditions_sha256"]:
        raise RuntimeError("low-light rescue conditions identity changed")
    for audit in config["baseline_audits"]:
        result = resolve_workspace_path(ROOT, audit["result"])
        sidecars = resolve_workspace_path(ROOT, audit["sidecars"])
        if file_sha256(result) != audit["result_sha256"] or not sidecars.is_dir():
            raise RuntimeError(f"low-light rescue baseline audit changed: {audit['name']}")
        payload = json.loads(result.read_text())
        if (
            payload.get("split") != "validation"
            or payload.get("sealed_internal_test_accessed") is not False
            or payload.get("protected_foundir_test_accessed") is not False
        ):
            raise RuntimeError(f"invalid low-light rescue baseline audit: {audit['name']}")
    router, parent, specialization, checkpoint = validate_router_config(router_path)
    return config, router, parent, specialization, checkpoint


def baseline_sidecars(config: Mapping[str, object]) -> dict[str, dict]:
    merged: dict[str, dict] = {}
    comparable = (
        "record_id",
        "source_group",
        "category",
        "composition_size",
        "qwen_levels",
        "decision",
        "crop_xy",
        "undersized_pair_resized",
        "input",
        "restored",
        "restored_uint8_sha256",
    )
    for audit in config["baseline_audits"]:
        directory = resolve_workspace_path(ROOT, audit["sidecars"])
        for path in sorted(directory.glob("*.json")):
            row = json.loads(path.read_text())
            if (
                row.get("status") != "complete"
                or row.get("split") != "validation"
                or row.get("sealed_internal_test_accessed") is not False
                or row.get("protected_foundir_test_accessed") is not False
            ):
                raise RuntimeError(f"invalid baseline audit sidecar: {path}")
            record_id = str(row["record_id"])
            if record_id in merged and any(merged[record_id].get(key) != row.get(key) for key in comparable):
                raise RuntimeError(f"overlapping baseline sidecars disagree: {record_id}")
            merged[record_id] = row
    counts = Counter(row["category"] for row in merged.values())
    if counts != Counter(EXPECTED_CATEGORIES) or len(merged) != sum(EXPECTED_CATEGORIES.values()):
        raise RuntimeError(f"baseline audit union coverage changed: {counts}")
    return merged


def candidate_superset(
    rows: Sequence[Mapping[str, str]],
    config: Mapping[str, object],
    parent: Mapping[str, object],
    baseline: Mapping[str, Mapping[str, object]],
) -> list[dict[str, object]]:
    selected = []
    crop_size = int(config["selection"]["crop_size"])
    crop_seed = int(parent["data"]["split_seed"])
    weights = config["luma"]["weights"]
    for row in rows:
        if row["split"] != "validation" or row["category"] not in EXPECTED_CATEGORIES:
            continue
        levels = json.loads(row["qwen_json"])
        level = int(levels["low_light"])
        if level >= 3:
            continue
        frozen = baseline[row["record_id"]]
        crop_xy = frozen["crop_xy"]
        resized = bool(frozen["undersized_pair_resized"])
        if resized:
            # The one undersized development panorama follows the exact paired
            # resize path. Normal records use the already verified frozen crop
            # coordinates so preselection does not redundantly hash both full
            # images. Every selected counterfactual is reloaded with full hash
            # verification immediately before GPU inference below.
            lq_m11, _, verified_xy, verified_resized = load_audit_crop(
                row, crop_size=crop_size, seed=crop_seed
            )
            if verified_xy != crop_xy or verified_resized is not True:
                raise RuntimeError(f"undersized baseline crop changed: {row['record_id']}")
        else:
            x, y = map(int, crop_xy)
            lq_path = resolve_workspace_path(ROOT, row["lq_path"])
            with Image.open(lq_path) as handle:
                array = np.asarray(
                    handle.convert("RGB").crop((x, y, x + crop_size, y + crop_size)),
                    dtype=np.uint8,
                ).copy()
            if array.shape != (crop_size, crop_size, 3):
                raise RuntimeError(f"frozen baseline crop is out of bounds: {row['record_id']}")
            lq_m11 = torch.from_numpy(array).permute(2, 0, 1).float().div_(127.5).sub_(1.0)
        mean_luma = crop_mean_luma(lq_m11, weights)
        if any(
            candidate_triggers(level, mean_luma, candidate)
            for name, candidate in config["candidates"].items()
            if name != "current_qwen_l3"
        ):
            selected.append(
                {"row": dict(row), "mean_luma": mean_luma, "crop_xy": crop_xy, "resized": resized}
            )
    if len(selected) != int(config["selection"]["counterfactual_records"]):
        raise RuntimeError(f"low-light rescue counterfactual coverage changed: {len(selected)}")
    return selected


def verify_counterfactual(
    path: Path, *, item: Mapping[str, object], config_sha: str, router_sha: str
) -> dict:
    payload = json.loads(path.read_text())
    row = item["row"]
    if (
        payload.get("status") != "complete"
        or payload.get("config_sha256") != config_sha
        or payload.get("router_config_sha256") != router_sha
        or payload.get("record_id") != row["record_id"]
        or payload.get("source_group") != row["source_group"]
        or payload.get("category") != row["category"]
        or payload.get("qwen_levels") != json.loads(row["qwen_json"])
        or payload.get("crop_xy") != item["crop_xy"]
        or payload.get("undersized_pair_resized") != item["resized"]
        or not math.isclose(float(payload.get("mean_luma", -1)), float(item["mean_luma"]), abs_tol=1e-8)
        or payload.get("sealed_internal_test_accessed") is not False
        or payload.get("protected_foundir_test_accessed") is not False
        or not all(finite_metric(payload.get("rescued", {}).get(key)) for key in ("psnr", "lpips"))
    ):
        raise RuntimeError(f"low-light rescue sidecar identity changed: {path}")
    return payload


def candidate_rows(
    baseline: Mapping[str, dict], rescue: Mapping[str, dict], candidate: Mapping[str, object]
) -> tuple[list[dict], int]:
    rows = []
    changes = 0
    for record_id, original in baseline.items():
        row = dict(original)
        counterfactual = rescue.get(record_id)
        if counterfactual is not None and candidate_triggers(
            int(row["qwen_levels"]["low_light"]),
            float(counterfactual["mean_luma"]),
            candidate,
        ):
            row["restored"] = counterfactual["rescued"]
            row["decision"] = counterfactual["rescued_decision"]
            changes += 1
        rows.append(row)
    return rows, changes


def summarize_candidate(
    rows: Sequence[Mapping[str, object]], config: Mapping[str, object], changes: int
) -> dict[str, object]:
    by_category: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        by_category[str(row["category"])].append(row)
    category_metrics = {
        category: mean_metrics(values) for category, values in sorted(by_category.items())
    }
    bootstrap = {}
    checks = {}
    metrics = config["metrics"]
    for task, category in ISOLATED_TASKS.items():
        values = by_category[category]
        bootstrap[task] = source_group_bootstrap(
            values,
            resamples=int(metrics["bootstrap_resamples"]),
            confidence=float(metrics["bootstrap_confidence"]),
            seed=int(stable_digest(int(metrics["bootstrap_seed"]), task)[:16], 16),
        )
        checks[task] = strong_gate_checks(category_metrics[category], bootstrap[task], config["strong_gates"][task])
    return {
        "route_changes": changes,
        "category_metrics": category_metrics,
        "source_group_bootstrap": bootstrap,
        "strong_gate_checks": checks,
        "strong_gate_pass": {task: all(values.values()) for task, values in checks.items()},
    }


def select_candidate(summaries: Mapping[str, Mapping[str, object]], config: Mapping[str, object]) -> dict:
    current = summaries["current_qwen_l3"]
    feasible = []
    for name, summary in summaries.items():
        safe = True
        for task, category in ISOLATED_TASKS.items():
            if task == "low_light":
                continue
            before = current["category_metrics"][category]
            after = summary["category_metrics"][category]
            if float(after["psnr"]) < float(before["psnr"]) - 0.10 or float(after["lpips"]) > float(before["lpips"]) + 0.002:
                safe = False
        if not safe:
            continue
        lowlight = summary["source_group_bootstrap"]["low_light"]
        gate = config["strong_gates"]["low_light"]
        margin = min(
            float(lowlight["lower"]["psnr"]) / float(gate["group_bootstrap_psnr_lower_minimum"]) - 1.0,
            1.0 - float(lowlight["upper"]["lpips"]) / float(gate["group_bootstrap_lpips_upper_maximum"]),
        )
        feasible.append((margin, -int(summary["route_changes"]), name))
    if not feasible:
        raise RuntimeError("no low-light luma rescue candidate satisfies non-lowlight safety constraints")
    margin, _, name = max(feasible)
    return {"candidate": name, "low_light_worst_normalized_strong_gate_margin": margin}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    config_path = resolve_workspace_path(ROOT, args.config)
    config, router, parent, specialization, blur_checkpoint = validate_config(config_path)
    config_sha = file_sha256(config_path)
    router_path = resolve_workspace_path(ROOT, config["router_config"])
    router_sha = file_sha256(router_path)
    rows = load_rows(resolve_workspace_path(ROOT, config["conditions"]), int(parent["data"]["expected_five_type_records"]))
    baseline = baseline_sidecars(config)
    if set(baseline) != {
        row["record_id"] for row in rows if row["split"] == "validation" and row["category"] in EXPECTED_CATEGORIES
    }:
        raise RuntimeError("baseline sidecars do not match frozen validation rows")
    selected = candidate_superset(rows, config, parent, baseline)
    if args.preflight:
        print(
            "FOUNDIR_LOWLIGHT_LUMA_RESCUE_PREFLIGHT_PASS "
            + json.dumps(
                {
                    "config_sha256": config_sha,
                    "baseline_records": len(baseline),
                    "counterfactual_records": len(selected),
                    "counterfactual_categories": dict(sorted(Counter(item["row"]["category"] for item in selected).items())),
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
        raise RuntimeError("low-light luma rescue gate requires eight H100/H200 GPUs")
    pretrained = resolve_workspace_path(ROOT, specialization["pretrained_checkpoint"])
    generalist = build_diffusion(specialization, pretrained, device).eval()
    blur_model = build_diffusion(specialization, pretrained, device)
    checkpoint_payload = torch.load(blur_checkpoint, map_location="cpu", weights_only=False)
    blur_model.model.load_state_dict(checkpoint_payload["model"], strict=True)
    del checkpoint_payload
    blur_model.eval()
    lowlight = build_specialist(released_config(parent), device).eval()
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

    lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(device).eval()
    output = resolve_workspace_path(ROOT, config["output"])
    sidecars = output / "per_record"
    sidecars.mkdir(parents=True, exist_ok=True)
    crop_size = int(config["selection"]["crop_size"])
    crop_seed = int(parent["data"]["split_seed"])
    inference_seed = int(router["evaluation"]["seed"])
    weights = config["luma"]["weights"]
    for index in range(rank, len(selected), world):
        item = selected[index]
        row = item["row"]
        sidecar = sidecars / f"{evaluation_artifact_stem(row['record_id'])}.json"
        if sidecar.exists():
            verify_counterfactual(sidecar, item=item, config_sha=config_sha, router_sha=router_sha)
            continue
        lq_m11, gt_m11, crop_xy, resized = load_audit_crop(row, crop_size=crop_size, seed=crop_seed)
        mean_luma = crop_mean_luma(lq_m11, weights)
        if crop_xy != item["crop_xy"] or resized != item["resized"] or not math.isclose(mean_luma, float(item["mean_luma"]), abs_tol=1e-8):
            raise RuntimeError(f"low-light rescue selection changed during inference: {row['record_id']}")
        levels = json.loads(row["qwen_json"])
        decision = route_decision(levels, router["routing"])
        source = ((lq_m11 + 1.0) * 0.5).unsqueeze(0).to(device)
        target = gt_m11.unsqueeze(0).to(device)
        record_seed = int(stable_digest(inference_seed, "qwen-routed-integrated", row["record_id"])[:16], 16)
        random.seed(record_seed)
        np.random.seed(record_seed % (2**32 - 1))
        torch.manual_seed(record_seed)
        torch.cuda.manual_seed_all(record_seed)
        base = blur_model if decision["base"] == "blur_specialization" else generalist
        restored = run_specialist(lowlight, run_generalist(base, source, device, row["record_id"]))
        output_u8 = restored[0].detach().cpu().permute(1, 2, 0).mul(255).round().byte().numpy()
        output_m11 = torch.from_numpy(output_u8.copy()).permute(2, 0, 1).float().div_(127.5).sub_(1.0).unsqueeze(0)
        rescued = {
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
            "original_decision": decision,
            "rescued_decision": {
                **decision,
                "refinement": "low_light",
                "route": f"{decision['base']}+low_light",
                "low_light_rescue": "qwen_ambiguous_input_luma_gate_v1",
            },
            "mean_luma": mean_luma,
            "crop_xy": crop_xy,
            "undersized_pair_resized": resized,
            "rescued": rescued,
            "rescued_uint8_sha256": hashlib.sha256(output_u8.tobytes()).hexdigest(),
            "sealed_internal_test_accessed": False,
            "protected_foundir_test_accessed": False,
        }
        write_once(sidecar, (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode())
        if (index + 1) % 80 == 0:
            print(f"FOUNDIR_LOWLIGHT_LUMA_RESCUE rank={rank} record={index+1}/{len(selected)}", flush=True)

    def finalize() -> None:
        expected_paths = [sidecars / f"{evaluation_artifact_stem(item['row']['record_id'])}.json" for item in selected]
        if set(sidecars.glob("*.json")) != set(expected_paths):
            raise RuntimeError("low-light rescue sidecar set is incomplete or contaminated")
        rescue = {
            payload["record_id"]: payload
            for item, path in zip(selected, expected_paths, strict=True)
            for payload in [verify_counterfactual(path, item=item, config_sha=config_sha, router_sha=router_sha)]
        }
        summaries = {}
        for name, candidate in config["candidates"].items():
            values, changes = candidate_rows(baseline, rescue, candidate)
            summaries[name] = summarize_candidate(values, config, changes)
        result = {
            "schema_version": 1,
            "status": "lowlight_luma_rescue_development_gate_complete",
            "config_sha256": config_sha,
            "router_config_sha256": router_sha,
            "split": "validation",
            "baseline_records": len(baseline),
            "counterfactual_records": len(rescue),
            "candidate_summaries": summaries,
            "selection": select_candidate(summaries, config),
            "sealed_internal_test_accessed": False,
            "protected_foundir_test_accessed": False,
        }
        write_once(output / "result.json", (json.dumps(result, indent=2, sort_keys=True) + "\n").encode())
        print("FOUNDIR_LOWLIGHT_LUMA_RESCUE_COMPLETE " + json.dumps(result, sort_keys=True), flush=True)

    synchronize_rank_zero_finalization(
        finalize,
        rank=rank,
        world=world,
        output=output,
        identity={"config_sha256": config_sha, "router_config_sha256": router_sha, "records": len(selected)},
        phase="low-light luma rescue development gate",
        timeout_seconds=60 * 60,
    )


if __name__ == "__main__":
    main()
