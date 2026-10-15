#!/usr/bin/env python3
"""Bound how much of the triple gap can be recovered by scalar exposure alone."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from diagnose_foundir_triple_sequential_stages import WORKSPACE, load_rgb, luma, psnr


MANIFEST = WORKSPACE / "manifests/foundir_composition_whole_image_v1.csv"
ROOT = WORKSPACE / "checkpoints/foundir_composition_whole_image_v1"
JSON_OUTPUT = WORKSPACE / "reports/foundir_triple_exposure_oracle_diagnosis_2026-09-27.json"
REPORT_OUTPUT = WORKSPACE / "reports/foundir_triple_exposure_oracle_diagnosis_2026-09-27.md"


def corrected(image: np.ndarray, alpha: float) -> np.ndarray:
    return np.rint(image.astype(np.float32) * alpha).clip(0, 255).astype(np.uint8)


def summarize(values: list[dict[str, float]]) -> dict[str, float]:
    keys = values[0]
    return {key: float(np.mean([item[key] for item in values])) for key in keys}


def main() -> None:
    with MANIFEST.open(newline="", encoding="utf-8") as handle:
        rows = [
            row for row in csv.DictReader(handle)
            if row["category"] == "18Lowlight_Blur_Noise"
            and row["routing_status"] == "correct"
            and row["final_route"] == "blur+noise+low_light"
        ]
    if len(rows) != 53:
        raise RuntimeError(f"expected 53 records, found {len(rows)}")

    by_stage: dict[str, list[dict[str, float]]] = {"after_low_light": [], "after_blur": []}
    for index, row in enumerate(rows, start=1):
        gt = load_rgb(WORKSPACE / row["gt_path"])
        gt_float = gt.astype(np.float64)
        gt_luma_mean = float(np.mean(luma(gt)))
        for stage_name, directory in (("after_low_light", "low_light"), ("after_blur", "blur")):
            image = load_rgb(ROOT / "ours" / "stages" / directory / f"{row['record_id']}.png")
            image_float = image.astype(np.float64)
            luma_alpha = gt_luma_mean / max(float(np.mean(luma(image))), 1e-12)
            least_squares_alpha = float(np.sum(image_float * gt_float) / max(np.sum(image_float * image_float), 1e-12))
            original_psnr = psnr(gt, image)
            luma_psnr = psnr(gt, corrected(image, luma_alpha))
            ls_psnr = psnr(gt, corrected(image, least_squares_alpha))
            by_stage[stage_name].append(
                {
                    "original_psnr": original_psnr,
                    "luma_match_alpha": luma_alpha,
                    "luma_match_psnr": luma_psnr,
                    "luma_match_gain": luma_psnr - original_psnr,
                    "least_squares_alpha": least_squares_alpha,
                    "least_squares_psnr": ls_psnr,
                    "least_squares_gain": ls_psnr - original_psnr,
                }
            )
        print(f"TRIPLE_EXPOSURE_ORACLE record={index}/{len(rows)}", flush=True)

    aggregate = {stage: summarize(values) for stage, values in by_stage.items()}
    payload = {
        "schema_version": 1,
        "status": "complete",
        "scope": "oracle_diagnostic_only",
        "records": len(rows),
        "official_foundir_test_accessed": False,
        "model_inference_run": False,
        "model_weight_updates": False,
        "warning": "GT-derived scalar corrections are non-deployable diagnostic upper bounds, not method results.",
        "aggregate": aggregate,
        "per_record": [
            {
                "record_id": row["record_id"],
                **{stage: by_stage[stage][index] for stage in by_stage},
            }
            for index, row in enumerate(rows)
        ],
    }
    JSON_OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# FoundIR triple scalar-exposure oracle diagnosis — 2026-09-27",
        "",
        "**DIAGNOSTIC ONLY / 非主线。** The scalar uses GT and is therefore non-deployable. It is an upper-bound diagnosis, not a restoration result. Existing full-resolution development-validation artifacts only; no inference, training, or official-test access.",
        "",
        "| Stage | Original PSNR | Mean luma-match α | Luma-match PSNR / gain | Mean least-squares α | Least-squares PSNR / gain |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for stage in ("after_low_light", "after_blur"):
        item = aggregate[stage]
        lines.append(
            f"| {stage} | {item['original_psnr']:.6f} | {item['luma_match_alpha']:.6f} | "
            f"{item['luma_match_psnr']:.6f} / {item['luma_match_gain']:+.6f} | "
            f"{item['least_squares_alpha']:.6f} | {item['least_squares_psnr']:.6f} / {item['least_squares_gain']:+.6f} |"
        )
    lines.extend(
        [
            "",
            "Interpretation: a large oracle gain means exposure overshoot is an important correctable component. The remaining gap after the least-squares scalar is a lower bound on non-scalar color/structure/detail error under this restricted correction family; it must not be called an exact causal decomposition.",
            "",
            f"Machine-readable evidence: `{JSON_OUTPUT.relative_to(WORKSPACE)}`.",
        ]
    )
    REPORT_OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "aggregate": aggregate}, sort_keys=True))


if __name__ == "__main__":
    main()
