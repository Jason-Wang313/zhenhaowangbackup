#!/usr/bin/env python3
"""Compare identical expert transitions on double versus triple compositions."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from diagnose_foundir_triple_sequential_stages import WORKSPACE, image_stats, load_rgb


MANIFEST = WORKSPACE / "manifests/foundir_composition_whole_image_v1.csv"
ARTIFACT_ROOT = WORKSPACE / "checkpoints/foundir_composition_whole_image_v1"
TRIPLE_JSON = WORKSPACE / "reports/foundir_triple_sequential_stage_failure_diagnosis_2026-09-27.json"
JSON_OUTPUT = WORKSPACE / "reports/foundir_expert_interaction_transfer_diagnosis_2026-09-27.json"
REPORT_OUTPUT = WORKSPACE / "reports/foundir_expert_interaction_transfer_diagnosis_2026-09-27.md"


CASES = {
    "noise_to_low_light_double": {
        "category": "16Lowlight_Noise",
        "route": "noise+low_light",
        "before": "noise",
        "after": "low_light",
    },
    "low_light_to_blur_double": {
        "category": "15Lowlight_Blur",
        "route": "blur+low_light",
        "before": "low_light",
        "after": "blur",
    },
}


def transition_summary(records: list[dict[str, Any]], before: str, after: str) -> dict[str, Any]:
    psnr_delta = [record[after]["psnr"] - record[before]["psnr"] for record in records]
    return {
        "records": len(records),
        "mean_before_psnr": float(np.mean([record[before]["psnr"] for record in records])),
        "mean_after_psnr": float(np.mean([record[after]["psnr"] for record in records])),
        "mean_psnr_delta": float(np.mean(psnr_delta)),
        "median_psnr_delta": float(np.median(psnr_delta)),
        "improved_records": sum(value > 0.0 for value in psnr_delta),
        "worsened_records": sum(value < 0.0 for value in psnr_delta),
        "mean_luma_bias_before": float(np.mean([record[before]["luma_bias_to_gt"] for record in records])),
        "mean_luma_bias_after": float(np.mean([record[after]["luma_bias_to_gt"] for record in records])),
        "mean_luma_mae_delta": float(np.mean([record[after]["luma_mae_to_gt"] - record[before]["luma_mae_to_gt"] for record in records])),
        "mean_gradient_error_delta": float(np.mean([record[after]["gradient_mae_to_gt"] - record[before]["gradient_mae_to_gt"] for record in records])),
        "mean_laplacian_error_delta": float(np.mean([record[after]["laplacian_mae_to_gt"] - record[before]["laplacian_mae_to_gt"] for record in records])),
    }


def main() -> None:
    with MANIFEST.open(newline="", encoding="utf-8") as handle:
        manifest = list(csv.DictReader(handle))

    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "complete",
        "scope": "development_validation_only",
        "official_foundir_test_accessed": False,
        "model_inference_run": False,
        "model_weight_updates": False,
        "double_cases": {},
    }
    for name, case in CASES.items():
        rows = [
            row for row in manifest
            if row["category"] == case["category"]
            and row["routing_status"] == "correct"
            and row["final_route"] == case["route"]
        ]
        records = []
        for index, row in enumerate(rows, start=1):
            record_id = row["record_id"]
            gt = load_rgb(WORKSPACE / row["gt_path"])
            before_image = load_rgb(ARTIFACT_ROOT / "ours" / "stages" / case["before"] / f"{record_id}.png")
            after_image = load_rgb(ARTIFACT_ROOT / "ours" / "stages" / case["after"] / f"{record_id}.png")
            if before_image.shape != gt.shape or after_image.shape != gt.shape:
                raise RuntimeError(f"shape mismatch: {record_id}")
            records.append(
                {
                    "record_id": record_id,
                    case["before"]: image_stats(before_image, gt),
                    case["after"]: image_stats(after_image, gt),
                }
            )
            print(f"INTERACTION_TRANSFER case={name} record={index}/{len(rows)}", flush=True)
        result["double_cases"][name] = {
            "category": case["category"],
            "transition": [case["before"], case["after"]],
            "summary": transition_summary(records, case["before"], case["after"]),
            "per_record": records,
        }

    triple = json.loads(TRIPLE_JSON.read_text())
    triple_by_key = {(item["before"], item["after"]): item for item in triple["transitions"]}
    result["triple_cases"] = {
        "noise_to_low_light": triple_by_key[("after_noise", "after_low_light")],
        "low_light_to_blur": triple_by_key[("after_low_light", "after_blur")],
    }

    comparisons = {}
    for short_name, double_name, triple_name in (
        ("noise_to_low_light", "noise_to_low_light_double", "noise_to_low_light"),
        ("low_light_to_blur", "low_light_to_blur_double", "low_light_to_blur"),
    ):
        double = result["double_cases"][double_name]["summary"]
        triple_item = result["triple_cases"][triple_name]
        comparisons[short_name] = {
            "double_mean_psnr_delta": double["mean_psnr_delta"],
            "triple_mean_psnr_delta": triple_item["mean_psnr_delta"],
            "triple_minus_double_delta": triple_item["mean_psnr_delta"] - double["mean_psnr_delta"],
            "double_improved_fraction": double["improved_records"] / double["records"],
            "triple_improved_fraction": triple_item["improved_records"] / 53.0,
        }
    result["comparisons"] = comparisons
    JSON_OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def row(label: str, values: dict[str, Any]) -> str:
        return (
            f"| {label} | {values['records']} | {values['mean_before_psnr']:.6f} | "
            f"{values['mean_after_psnr']:.6f} | {values['mean_psnr_delta']:+.6f} | "
            f"{values['improved_records']} / {values['worsened_records']} | "
            f"{values['mean_luma_bias_before']:+.3f} → {values['mean_luma_bias_after']:+.3f} | "
            f"{values['mean_gradient_error_delta']:+.3f} | {values['mean_laplacian_error_delta']:+.3f} |"
        )

    noise_double = result["double_cases"]["noise_to_low_light_double"]["summary"]
    blur_double = result["double_cases"]["low_light_to_blur_double"]["summary"]
    noise_triple = result["triple_cases"]["noise_to_low_light"]
    blur_triple = result["triple_cases"]["low_light_to_blur"]
    report = [
        "# FoundIR expert-interaction transfer diagnosis — 2026-09-27",
        "",
        "**DIAGNOSTIC ONLY / 非主线。** Existing full-resolution development-validation artifacts only; no inference, training, or official-test access.",
        "",
        "## Same transition on double compositions",
        "",
        "| Transition/domain | N | Before PSNR | After PSNR | Mean ΔPSNR | Improved / worsened | Luma bias before → after | Δ gradient error | Δ Laplacian error |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        row("Noise → Low-light on Low-light+Noise", noise_double),
        row("Low-light → Blur on Low-light+Blur", blur_double),
        "",
        "## Double-to-triple transfer",
        "",
        "| Transition | Double mean ΔPSNR | Triple mean ΔPSNR | Triple minus double | Double improved | Triple improved |",
        "|---|---:|---:|---:|---:|---:|",
        f"| Noise → Low-light | {noise_double['mean_psnr_delta']:+.6f} | {noise_triple['mean_psnr_delta']:+.6f} | {comparisons['noise_to_low_light']['triple_minus_double_delta']:+.6f} | {noise_double['improved_records']}/{noise_double['records']} | {noise_triple['improved_records']}/53 |",
        f"| Low-light → Blur | {blur_double['mean_psnr_delta']:+.6f} | {blur_triple['mean_psnr_delta']:+.6f} | {comparisons['low_light_to_blur']['triple_minus_double_delta']:+.6f} | {blur_double['improved_records']}/{blur_double['records']} | {blur_triple['improved_records']}/53 |",
        "",
        "## Interpretation rule",
        "",
        "A large double-to-triple deterioration supports composition-domain interaction failure. Similar behavior in both domains instead means the expert transition itself is weak and should not be framed as a triple-specific generalization failure. This comparison is observational because the image sets differ; it is not a paired causal estimate.",
        "",
        f"Machine-readable evidence: `{JSON_OUTPUT.relative_to(WORKSPACE)}`.",
    ]
    REPORT_OUTPUT.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "comparisons": comparisons}, sort_keys=True))


if __name__ == "__main__":
    main()
