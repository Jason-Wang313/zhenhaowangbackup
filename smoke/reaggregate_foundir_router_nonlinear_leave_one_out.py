#!/usr/bin/env python3
"""Factorial CPU ablation of the two frozen Noise/Low-light nonlinear heads.

This script reads only frozen Qwen feature tensors and frozen router heads.  It
does not read images, train a model, run Qwen, run a restoration model, or
access an official/sealed split.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors import safe_open


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lowlight_restoration.qwen_compositional_router import (  # noqa: E402
    COMPONENTS,
    compositional_route,
)
from smoke.run_foundir_qwen_compositional_router_v1 import (  # noqa: E402
    load_contract,
    load_feature,
    predict_probabilities,
)
from smoke.run_foundir_qwen_compositional_router_v3 import apply_gate  # noqa: E402
from smoke.run_foundir_qwen_lowlight_noise_dual_nonlinear_v2 import (  # noqa: E402
    model_from,
    tensors,
)
from smoke.run_foundir_qwen_five_class_gate_ddp import (  # noqa: E402
    sha256,
    workspace_path,
)


CONFIG_PATH = ROOT / "configs/foundir_qwen_lowlight_noise_dual_nonlinear_v2.json"
DUAL_RESULT_PATH = ROOT / "checkpoints/foundir_qwen_lowlight_noise_dual_nonlinear_v2/result.json"
OUT_JSON = ROOT / "reports/foundir_router_nonlinear_leave_one_out_2026-09-28.json"
OUT_CSV = ROOT / "reports/foundir_router_nonlinear_leave_one_out_2026-09-28.csv"
OUT_MD = ROOT / "reports/foundir_router_nonlinear_leave_one_out_2026-09-28.md"
FOCUS_CLASSES = {"noise", "low_light", "noise+low_light"}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(
    record_ids: list[str], truths: list[str], routes: list[str | None]
) -> dict[str, Any]:
    per_class: dict[str, dict[str, int]] = defaultdict(
        lambda: {"records": 0, "correct": 0, "wrong": 0, "abstain": 0}
    )
    correct = wrong = abstain = 0
    for _record_id, truth, route in zip(record_ids, truths, routes, strict=True):
        per_class[truth]["records"] += 1
        if route is None:
            abstain += 1
            per_class[truth]["abstain"] += 1
        elif route == truth:
            correct += 1
            per_class[truth]["correct"] += 1
        else:
            wrong += 1
            per_class[truth]["wrong"] += 1
    accepted = correct + wrong
    return {
        "records": len(routes),
        "accepted": accepted,
        "correct": correct,
        "wrong": wrong,
        "abstain": abstain,
        "coverage": accepted / len(routes),
        "accepted_error_rate": wrong / accepted if accepted else None,
        "per_class": dict(sorted(per_class.items())),
    }


def effective_confidence(values: np.ndarray) -> np.ndarray:
    """Minimum binary-decision confidence across the five effective factors."""
    return np.min(np.abs(values - 0.5) * 2.0, axis=1)


def matched_routes(
    record_ids: list[str],
    routes: list[str | None],
    confidence: np.ndarray,
    accepted_count: int,
) -> list[str | None]:
    candidates = [
        (float(confidence[index]), record_ids[index], index)
        for index, route in enumerate(routes)
        if route is not None
    ]
    if len(candidates) < accepted_count:
        raise RuntimeError("variant cannot reach matched accepted count")
    candidates.sort(key=lambda row: (-row[0], row[1]))
    keep = {index for _score, _record_id, index in candidates[:accepted_count]}
    return [route if index in keep else None for index, route in enumerate(routes)]


def apply_lowlight_nonlinear(
    values: np.ndarray,
    base_probabilities: np.ndarray,
    lowlight_probabilities: np.ndarray,
    config: dict[str, Any],
) -> None:
    indexes = {name: COMPONENTS.index(name) for name in COMPONENTS}
    route_margin = float(config["base"]["route_margin"])
    absent = lambda name: base_probabilities[:, indexes[name]] <= 1.0 - route_margin
    target = (
        absent("blur")
        & absent("rain")
        & absent("haze")
        & (base_probabilities[:, indexes["low_light"]] >= route_margin)
    )
    margin = float(config["lowlight_side"]["margin"])
    values[target & (lowlight_probabilities >= margin), indexes["noise"]] = 1.0
    values[target & (lowlight_probabilities <= 1.0 - margin), indexes["noise"]] = 0.0
    uncertain = target & (lowlight_probabilities > 1.0 - margin) & (lowlight_probabilities < margin)
    values[uncertain, indexes["noise"]] = 0.5


def apply_noise_nonlinear(
    values: np.ndarray,
    base_probabilities: np.ndarray,
    noise_probabilities: np.ndarray,
    config: dict[str, Any],
    margin: float,
) -> None:
    indexes = {name: COMPONENTS.index(name) for name in COMPONENTS}
    route_margin = float(config["base"]["route_margin"])
    absent = lambda name: base_probabilities[:, indexes[name]] <= 1.0 - route_margin
    target = (
        absent("blur")
        & absent("rain")
        & absent("haze")
        & (base_probabilities[:, indexes["noise"]] >= route_margin)
        & absent("low_light")
    )
    values[target & (noise_probabilities >= margin), indexes["low_light"]] = 1.0
    values[target & (noise_probabilities <= 1.0 - margin), indexes["low_light"]] = 0.0
    uncertain = target & (noise_probabilities > 1.0 - margin) & (noise_probabilities < margin)
    values[uncertain, indexes["low_light"]] = 0.5


def focus_summary(stats: dict[str, Any]) -> dict[str, Any]:
    rows = [stats["per_class"][name] for name in sorted(FOCUS_CLASSES)]
    records = sum(row["records"] for row in rows)
    correct = sum(row["correct"] for row in rows)
    wrong = sum(row["wrong"] for row in rows)
    abstain = sum(row["abstain"] for row in rows)
    accepted = correct + wrong
    return {
        "records": records,
        "accepted": accepted,
        "correct": correct,
        "wrong": wrong,
        "abstain": abstain,
        "coverage": accepted / records,
        "accepted_error_rate": wrong / accepted if accepted else None,
    }


def main() -> None:
    if any(path.exists() for path in (OUT_JSON, OUT_CSV, OUT_MD)):
        raise FileExistsError("refusing to overwrite nonlinear leave-one-out artifacts")

    config, config_sha, _fit_rows, singleton_rows, mixed_rows = load_contract(CONFIG_PATH)
    dual_result = json.loads(DUAL_RESULT_PATH.read_text())
    base_path = workspace_path(config["base"]["head"])
    lowlight_path = workspace_path(config["lowlight_side"]["head"])
    noise_path = workspace_path(config["outputs"]["head"])
    for path, expected in (
        (base_path, config["base"]["head_sha256"]),
        (lowlight_path, config["lowlight_side"]["head_sha256"]),
        (noise_path, dual_result["head_sha256"]),
    ):
        if sha256(path) != expected:
            raise RuntimeError(f"frozen head changed: {path}")

    rows = singleton_rows + mixed_rows
    record_ids = [row["record_id"] for row in rows]
    truths = [row["label"] for row in rows]
    singleton_dir = workspace_path(config["features"]["singleton_development_dir"])
    mixed_dir = workspace_path(config["features"]["mixed_development_dir"])
    features = np.stack(
        [load_feature(singleton_dir, row, config) for row in singleton_rows]
        + [load_feature(mixed_dir, row, config) for row in mixed_rows]
    )

    base_tensors = {name: value.numpy() for name, value in tensors(base_path).items()}
    standardized = (
        (features - base_tensors["feature_mean"]) / base_tensors["feature_scale"]
    ).astype(np.float32)
    base_probabilities = predict_probabilities(
        standardized, base_tensors["coefficient"], base_tensors["intercept"]
    )
    linear_conditional = predict_probabilities(
        standardized,
        base_tensors["conditional_noise_coefficient"][None, :],
        base_tensors["conditional_noise_intercept"],
    )[:, 0]
    lowlight_model = model_from(lowlight_path, standardized.shape[1])
    noise_model = model_from(noise_path, standardized.shape[1])
    torch.set_num_threads(max(1, min(16, torch.get_num_threads())))
    with torch.inference_mode():
        feature_tensor = torch.from_numpy(standardized)
        lowlight_probabilities = torch.sigmoid(lowlight_model(feature_tensor)).numpy()
        noise_probabilities = torch.sigmoid(noise_model(feature_tensor)).numpy()

    linear_values = apply_gate(
        base_probabilities,
        linear_conditional,
        route_margin=float(config["base"]["route_margin"]),
        positive_margin=float(config["base"]["conditional_positive_margin"]),
        negative_margin=float(config["base"]["conditional_negative_margin"]),
    )
    variants: dict[str, np.ndarray] = {}
    variants["atomic_plus_linear"] = linear_values.copy()
    variants["plus_lowlight_nonlinear"] = linear_values.copy()
    apply_lowlight_nonlinear(
        variants["plus_lowlight_nonlinear"], base_probabilities, lowlight_probabilities, config
    )
    variants["plus_noise_nonlinear"] = linear_values.copy()
    apply_noise_nonlinear(
        variants["plus_noise_nonlinear"],
        base_probabilities,
        noise_probabilities,
        config,
        float(dual_result["margin_selection"]["margin"]),
    )
    variants["plus_both_nonlinear"] = variants["plus_lowlight_nonlinear"].copy()
    apply_noise_nonlinear(
        variants["plus_both_nonlinear"],
        base_probabilities,
        noise_probabilities,
        config,
        float(dual_result["margin_selection"]["margin"]),
    )

    route_margin = float(config["base"]["route_margin"])
    routes_by_variant = {
        name: [compositional_route(row, margin=route_margin) for row in values]
        for name, values in variants.items()
    }
    raw = {
        name: summarize(record_ids, truths, routes)
        for name, routes in routes_by_variant.items()
    }
    matched_count = min(item["accepted"] for item in raw.values())
    matched = {}
    for name, routes in routes_by_variant.items():
        selected = matched_routes(
            record_ids, routes, effective_confidence(variants[name]), matched_count
        )
        matched[name] = summarize(record_ids, truths, selected)

    payload = {
        "schema_version": 1,
        "status": "frozen_nonlinear_factorial_ablation_complete",
        "date": "2026-09-28",
        "scope": {
            "records": len(rows),
            "gpu_used": False,
            "model_training": False,
            "qwen_inference": False,
            "restoration_inference": False,
            "image_pixels_accessed": False,
            "official_or_sealed_images_accessed": False,
            "development_reused": True,
        },
        "variant_definitions": {
            "atomic_plus_linear": "Atomic five-factor head plus the frozen linear Low-light/Noise conditional correction.",
            "plus_lowlight_nonlinear": "Replace the eligible Low-light-entry linear decision with the frozen Low-light-side nonlinear head.",
            "plus_noise_nonlinear": "Retain the linear Low-light-entry decision and add the frozen Noise-side nonlinear head.",
            "plus_both_nonlinear": "Use both frozen ambiguity-specific nonlinear heads.",
        },
        "raw_threshold": raw,
        "matched_coverage": {
            "accepted_count": matched_count,
            "coverage": matched_count / len(rows),
            "ranking": "top accepted routes by minimum effective five-factor binary confidence; record_id breaks ties",
            "variants": matched,
        },
        "focus_classes": {
            "classes": sorted(FOCUS_CLASSES),
            "raw_threshold": {name: focus_summary(stats) for name, stats in raw.items()},
            "matched_coverage": {name: focus_summary(stats) for name, stats in matched.items()},
        },
        "provenance": {
            "config": str(CONFIG_PATH.relative_to(ROOT)),
            "config_sha256": config_sha,
            "dual_result": str(DUAL_RESULT_PATH.relative_to(ROOT)),
            "dual_result_sha256": digest(DUAL_RESULT_PATH),
            "base_head": str(base_path.relative_to(ROOT)),
            "base_head_sha256": sha256(base_path),
            "lowlight_head": str(lowlight_path.relative_to(ROOT)),
            "lowlight_head_sha256": sha256(lowlight_path),
            "noise_head": str(noise_path.relative_to(ROOT)),
            "noise_head_sha256": sha256(noise_path),
        },
        "limitations": [
            "This is a frozen development-set reaggregation, not an independent lockbox result.",
            "Atomic+linear is the historical linear conditional baseline; it is not an unconditioned atomic-only head.",
            "Matched coverage uses post-hoc ranking of already accepted routes and does not recalibrate or retrain any head.",
            "No downstream restoration inference was run; results measure routing only.",
        ],
    }

    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    with OUT_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "protocol",
                "variant",
                "records",
                "accepted",
                "correct",
                "wrong",
                "abstain",
                "coverage",
                "accepted_error_rate",
            ),
        )
        writer.writeheader()
        for protocol, table in (("raw_threshold", raw), ("matched_coverage", matched)):
            for name, stats in table.items():
                writer.writerow(
                    {
                        "protocol": protocol,
                        "variant": name,
                        **{
                            key: stats[key]
                            for key in writer.fieldnames
                            if key not in {"protocol", "variant"}
                        },
                    }
                )

    lines = [
        "# FoundIR router nonlinear-head leave-one-out ablation",
        "",
        "CPU-only factorial reaggregation of 16,461 frozen development features and frozen tiny-head checkpoints. No image, Qwen, restoration, training, GPU, or official/sealed-image access.",
        "",
        "## Frozen thresholds",
        "",
        "| Variant | Coverage | Correct | Wrong | Abstain | Accepted error |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, stats in raw.items():
        lines.append(
            f"| {name} | {stats['coverage']:.4%} | {stats['correct']:,} | {stats['wrong']:,} | {stats['abstain']:,} | {stats['accepted_error_rate']:.4%} |"
        )
    lines.extend(
        [
            "",
            f"## Matched coverage ({matched_count:,}/{len(rows):,}, {matched_count/len(rows):.4%})",
            "",
            "| Variant | Correct | Wrong | Abstain | Accepted error |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name, stats in matched.items():
        lines.append(
            f"| {name} | {stats['correct']:,} | {stats['wrong']:,} | {stats['abstain']:,} | {stats['accepted_error_rate']:.4%} |"
        )
    lines.extend(
        [
            "",
            "## Noise / Low-light focus at frozen thresholds",
            "",
            "| Variant | Coverage | Correct | Wrong | Abstain | Accepted error |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for name, stats in payload["focus_classes"]["raw_threshold"].items():
        lines.append(
            f"| {name} | {stats['coverage']:.4%} | {stats['correct']:,} | {stats['wrong']:,} | {stats['abstain']:,} | {stats['accepted_error_rate']:.4%} |"
        )
    lines.extend(["", "## Limitations", ""])
    lines.extend(f"- {item}" for item in payload["limitations"])
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(
        json.dumps(
            {
                "status": payload["status"],
                "raw": {
                    name: {
                        "coverage": stats["coverage"],
                        "wrong": stats["wrong"],
                        "accepted_error_rate": stats["accepted_error_rate"],
                    }
                    for name, stats in raw.items()
                },
                "matched_accepted_count": matched_count,
                "matched": {
                    name: {
                        "wrong": stats["wrong"],
                        "accepted_error_rate": stats["accepted_error_rate"],
                    }
                    for name, stats in matched.items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
