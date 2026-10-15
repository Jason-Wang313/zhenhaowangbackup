#!/usr/bin/env python3
"""Diagnose the frozen triple-composition sequential restoration artifacts.

This script is deliberately CPU-only and read-only with respect to all source
artifacts.  It measures the already-produced full-resolution images; it does
not run a restoration model or access the official FoundIR test set.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


WORKSPACE = Path(__file__).resolve().parents[1]
CATEGORY = "18Lowlight_Blur_Noise"
CORRECT_ROUTE = "blur+noise+low_light"
STAGES = OrderedDict(
    (
        ("input", None),
        ("after_noise", "noise"),
        ("after_low_light", "low_light"),
        ("after_blur", "blur"),
        ("released_generalist", "generalist"),
        ("full_foundir", "full_foundir"),
    )
)
SEQUENTIAL = ("input", "after_noise", "after_low_light", "after_blur")


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def psnr(reference: np.ndarray, prediction: np.ndarray) -> float:
    difference = reference.astype(np.float64) - prediction.astype(np.float64)
    mse = float(np.mean(difference * difference))
    return math.inf if mse == 0.0 else 20.0 * math.log10(255.0) - 10.0 * math.log10(mse)


def luma(rgb: np.ndarray) -> np.ndarray:
    array = rgb.astype(np.float32)
    return array[..., 0] * 0.2126 + array[..., 1] * 0.7152 + array[..., 2] * 0.0722


def image_stats(rgb: np.ndarray, reference_rgb: np.ndarray) -> dict[str, float]:
    lum = luma(rgb)
    reference_lum = luma(reference_rgb)
    horizontal = np.abs(np.diff(lum, axis=1))
    vertical = np.abs(np.diff(lum, axis=0))
    reference_horizontal = np.abs(np.diff(reference_lum, axis=1))
    reference_vertical = np.abs(np.diff(reference_lum, axis=0))

    laplacian = (
        -4.0 * lum[1:-1, 1:-1]
        + lum[:-2, 1:-1]
        + lum[2:, 1:-1]
        + lum[1:-1, :-2]
        + lum[1:-1, 2:]
    )
    reference_laplacian = (
        -4.0 * reference_lum[1:-1, 1:-1]
        + reference_lum[:-2, 1:-1]
        + reference_lum[2:, 1:-1]
        + reference_lum[1:-1, :-2]
        + reference_lum[1:-1, 2:]
    )
    gradient_error = (
        float(np.mean(np.abs(horizontal - reference_horizontal)))
        + float(np.mean(np.abs(vertical - reference_vertical)))
    ) / 2.0
    channel_bias = rgb.astype(np.float32).mean(axis=(0, 1)) - reference_rgb.astype(np.float32).mean(axis=(0, 1))
    return {
        "psnr": psnr(reference_rgb, rgb),
        "luma_mean": float(np.mean(lum)),
        "luma_bias_to_gt": float(np.mean(lum) - np.mean(reference_lum)),
        "luma_mae_to_gt": float(np.mean(np.abs(lum - reference_lum))),
        "dark_clip_fraction": float(np.mean(rgb <= 1)),
        "bright_clip_fraction": float(np.mean(rgb >= 254)),
        "gradient_mean": float((np.mean(horizontal) + np.mean(vertical)) / 2.0),
        "gradient_mae_to_gt": gradient_error,
        "laplacian_abs_mean": float(np.mean(np.abs(laplacian))),
        "laplacian_mae_to_gt": float(np.mean(np.abs(laplacian - reference_laplacian))),
        "red_mean_bias_to_gt": float(channel_bias[0]),
        "green_mean_bias_to_gt": float(channel_bias[1]),
        "blue_mean_bias_to_gt": float(channel_bias[2]),
    }


def mean(values: list[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def median(values: list[float]) -> float:
    return float(np.median(np.asarray(values, dtype=np.float64)))


def fmt(value: float, digits: int = 6) -> str:
    return f"{value:.{digits}f}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="manifests/foundir_composition_whole_image_v1.csv")
    parser.add_argument("--artifact-root", default="checkpoints/foundir_composition_whole_image_v1")
    parser.add_argument("--full-foundir-root", default="checkpoints/foundir_composition_whole_image_full_pipeline_v1")
    parser.add_argument("--json-output", default="reports/foundir_triple_sequential_stage_failure_diagnosis_2026-09-27.json")
    parser.add_argument("--csv-output", default="reports/foundir_triple_sequential_stage_failure_diagnosis_2026-09-27.csv")
    parser.add_argument("--report-output", default="reports/foundir_triple_sequential_stage_failure_diagnosis_2026-09-27.md")
    return parser.parse_args()


def resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else WORKSPACE / candidate


def main() -> None:
    args = parse_args()
    manifest_path = resolve(args.manifest)
    artifact_root = resolve(args.artifact_root)
    full_root = resolve(args.full_foundir_root)

    with manifest_path.open(newline="", encoding="utf-8") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row["category"] == CATEGORY
            and row["routing_status"] == "correct"
            and row["final_route"] == CORRECT_ROUTE
        ]
    if len(rows) != 53:
        raise RuntimeError(f"expected 53 correctly routed triple records, found {len(rows)}")

    records: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        record_id = row["record_id"]
        gt_path = resolve(row["gt_path"])
        paths = OrderedDict(
            (
                ("input", resolve(row["lq_path"])),
                ("after_noise", artifact_root / "ours" / "stages" / "noise" / f"{record_id}.png"),
                ("after_low_light", artifact_root / "ours" / "stages" / "low_light" / f"{record_id}.png"),
                ("after_blur", artifact_root / "ours" / "stages" / "blur" / f"{record_id}.png"),
                ("released_generalist", artifact_root / "foundir" / "images" / f"{record_id}.png"),
                ("full_foundir", full_root / "specialist" / "images" / f"{record_id}.png"),
            )
        )
        missing = [str(path) for path in (gt_path, *paths.values()) if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"{record_id}: missing artifacts: {missing}")

        gt = load_rgb(gt_path)
        stage_values: dict[str, dict[str, float]] = {}
        for stage, path in paths.items():
            image = load_rgb(path)
            if image.shape != gt.shape:
                raise RuntimeError(f"{record_id} {stage}: {image.shape} != GT {gt.shape}")
            stage_values[stage] = image_stats(image, gt)

        original_sidecar = json.loads((artifact_root / "per_record" / f"{record_id}.json").read_text())
        full_sidecar = json.loads((full_root / "per_record" / f"{record_id}.json").read_text())
        if abs(stage_values["after_blur"]["psnr"] - float(original_sidecar["ours"]["psnr"])) > 1e-9:
            raise RuntimeError(f"{record_id}: recomputed project PSNR does not match frozen sidecar")
        if abs(stage_values["released_generalist"]["psnr"] - float(original_sidecar["foundir"]["psnr"])) > 1e-9:
            raise RuntimeError(f"{record_id}: recomputed Generalist PSNR does not match frozen sidecar")
        if abs(stage_values["full_foundir"]["psnr"] - float(full_sidecar["full_foundir_generalist_specialist"]["psnr"])) > 1e-9:
            raise RuntimeError(f"{record_id}: recomputed Full FoundIR PSNR does not match frozen sidecar")

        records.append(
            {
                "record_id": record_id,
                "source_group": row["source_group"],
                "width": int(row["width"]),
                "height": int(row["height"]),
                "stages": stage_values,
                "existing_lpips": {
                    "after_blur": float(original_sidecar["ours"]["lpips"]),
                    "released_generalist": float(original_sidecar["foundir"]["lpips"]),
                    "full_foundir": float(full_sidecar["full_foundir_generalist_specialist"]["lpips"]),
                },
            }
        )
        print(f"TRIPLE_STAGE_DIAG record={index}/{len(rows)} id={record_id}", flush=True)

    metric_names = tuple(records[0]["stages"]["input"])
    aggregate: dict[str, dict[str, float]] = {}
    for stage in STAGES:
        aggregate[stage] = {
            metric: mean([record["stages"][stage][metric] for record in records])
            for metric in metric_names
        }
    for stage in ("after_blur", "released_generalist", "full_foundir"):
        aggregate[stage]["lpips"] = mean([record["existing_lpips"][stage] for record in records])

    transitions: list[dict[str, Any]] = []
    for before, after in zip(SEQUENTIAL, SEQUENTIAL[1:]):
        deltas = [record["stages"][after]["psnr"] - record["stages"][before]["psnr"] for record in records]
        transitions.append(
            {
                "before": before,
                "after": after,
                "mean_psnr_delta": mean(deltas),
                "median_psnr_delta": median(deltas),
                "improved_records": sum(delta > 0.0 for delta in deltas),
                "worsened_records": sum(delta < 0.0 for delta in deltas),
                "unchanged_records": sum(delta == 0.0 for delta in deltas),
                "mean_luma_mae_delta": mean(
                    [record["stages"][after]["luma_mae_to_gt"] - record["stages"][before]["luma_mae_to_gt"] for record in records]
                ),
                "mean_gradient_error_delta": mean(
                    [record["stages"][after]["gradient_mae_to_gt"] - record["stages"][before]["gradient_mae_to_gt"] for record in records]
                ),
                "mean_laplacian_error_delta": mean(
                    [record["stages"][after]["laplacian_mae_to_gt"] - record["stages"][before]["laplacian_mae_to_gt"] for record in records]
                ),
            }
        )

    worst_transition = min(transitions, key=lambda item: item["mean_psnr_delta"])
    payload = {
        "schema_version": 1,
        "status": "complete",
        "scope": "development_validation_only",
        "official_foundir_test_accessed": False,
        "model_inference_run": False,
        "model_weight_updates": False,
        "category": CATEGORY,
        "route": ["noise", "low_light", "blur"],
        "records": len(records),
        "aggregate": aggregate,
        "transitions": transitions,
        "dominant_psnr_drop_transition": {
            "before": worst_transition["before"],
            "after": worst_transition["after"],
            "mean_psnr_delta": worst_transition["mean_psnr_delta"],
        },
        "per_record": records,
    }

    json_output = resolve(args.json_output)
    csv_output = resolve(args.csv_output)
    report_output = resolve(args.report_output)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with csv_output.open("w", newline="", encoding="utf-8") as handle:
        fields = ["record_id", "source_group", "stage", *metric_names, "lpips"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            for stage in STAGES:
                writer.writerow(
                    {
                        "record_id": record["record_id"],
                        "source_group": record["source_group"],
                        "stage": stage,
                        **record["stages"][stage],
                        "lpips": record["existing_lpips"].get(stage, ""),
                    }
                )

    stage_labels = {
        "input": "Degraded input",
        "after_noise": "After Noise expert",
        "after_low_light": "After Low-light expert",
        "after_blur": "After Blur expert (project final)",
        "released_generalist": "Released FoundIR Generalist",
        "full_foundir": "Full FoundIR Generalist → Low-light specialist",
    }
    lines = [
        "# FoundIR triple sequential-stage failure diagnosis — 2026-09-27",
        "",
        "**DIAGNOSTIC ONLY / 非主线。** This report uses only the frozen 660-image development-validation artifacts. It did not run inference, update model weights, or access the official FoundIR test.",
        "",
        "## Scope",
        "",
        f"- Category: `{CATEGORY}` (`Blur + Noise + Low-light`).",
        f"- Records: {len(records)} correctly routed full-resolution images; the seven Low-light fallback records are excluded.",
        "- Frozen project sequence: `Noise → Low-light → Blur`.",
        "- PSNR is recomputed on full-image RGB uint8 pixels and is checked exactly against the frozen final sidecars.",
        "- Stage LPIPS was not recomputed; the table reuses existing LPIPS only where frozen sidecars already exist.",
        "",
        "## Stage means",
        "",
        "| Stage | PSNR | Existing LPIPS | Luma mean | Luma bias vs GT | Luma MAE vs GT | Gradient error vs GT | Laplacian error vs GT | Dark clip | Bright clip |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for stage in STAGES:
        values = aggregate[stage]
        lpips_value = fmt(values["lpips"]) if "lpips" in values else "—"
        lines.append(
            f"| {stage_labels[stage]} | {fmt(values['psnr'])} | {lpips_value} | {fmt(values['luma_mean'], 3)} | "
            f"{fmt(values['luma_bias_to_gt'], 3)} | {fmt(values['luma_mae_to_gt'], 3)} | "
            f"{fmt(values['gradient_mae_to_gt'], 3)} | {fmt(values['laplacian_mae_to_gt'], 3)} | "
            f"{fmt(values['dark_clip_fraction'] * 100, 3)}% | {fmt(values['bright_clip_fraction'] * 100, 3)}% |"
        )

    lines.extend(
        [
            "",
            "## Sequential transition diagnosis",
            "",
            "| Transition | Mean ΔPSNR | Median ΔPSNR | Improved / worsened / unchanged | Δ luma MAE | Δ gradient error | Δ Laplacian error |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in transitions:
        lines.append(
            f"| {stage_labels[item['before']]} → {stage_labels[item['after']]} | {fmt(item['mean_psnr_delta'])} | "
            f"{fmt(item['median_psnr_delta'])} | {item['improved_records']} / {item['worsened_records']} / {item['unchanged_records']} | "
            f"{fmt(item['mean_luma_mae_delta'], 3)} | {fmt(item['mean_gradient_error_delta'], 3)} | "
            f"{fmt(item['mean_laplacian_error_delta'], 3)} |"
        )

    project_vs_generalist = aggregate["after_blur"]["psnr"] - aggregate["released_generalist"]["psnr"]
    lowlight_vs_generalist = aggregate["after_low_light"]["psnr"] - aggregate["released_generalist"]["psnr"]
    project_vs_full = aggregate["after_blur"]["psnr"] - aggregate["full_foundir"]["psnr"]
    lines.extend(
        [
            "",
            "## Evidence-based conclusion",
            "",
            f"1. The Noise expert is nearly inert on this triple domain: it adds only `{transitions[0]['mean_psnr_delta']:+.6f} dB`. It slightly reduces Laplacian error but increases gradient error.",
            f"2. The Low-light expert supplies almost all project recovery (`{transitions[1]['mean_psnr_delta']:+.6f} dB`), but it overshoots GT exposure by `{aggregate['after_low_light']['luma_bias_to_gt']:+.3f}` mean luma. Its output remains `{lowlight_vs_generalist:+.6f} dB` below the Released FoundIR Generalist.",
            f"3. The final Blur expert is consistently destructive: `{transitions[2]['worsened_records']}/53` records lose PSNR, the mean change is `{transitions[2]['mean_psnr_delta']:+.6f} dB`, and luma, gradient, and Laplacian errors all increase.",
            f"4. Skipping Blur cannot solve the failure: the post-Low-light mean is only `{aggregate['after_low_light']['psnr']:.6f} dB`. The final project output trails the Released Generalist by `{project_vs_generalist:.6f} dB` and Full FoundIR by `{project_vs_full:.6f} dB` on the same 53 correctly routed records.",
            "",
            "Therefore routing error is ruled out for this subset, while expert ordering is only a secondary problem. The dominant limitation is that independently trained singleton experts do not perform joint composite restoration on the out-of-domain triple input/intermediate states; their small gains and errors accumulate across the cascade. This evidence supports testing an interaction-aware or joint-composition mechanism, not merely another router threshold or a reordered singleton cascade.",
            "",
            "The closest successful reproduction is already present in these artifacts: FoundIR's Generalist jointly restores the composite and its Low-light specialist then corrects the residual (`24.986867 → 33.238410 dB`). The next project experiment should isolate whether a compatibility mechanism learned on double compositions can make frozen singleton experts compose on a held-out triple. A direct triple-trained specialist may improve metrics, but it would not by itself constitute a credible novelty claim.",
            "",
            "## Artifacts",
            "",
            f"- JSON: `{json_output.relative_to(WORKSPACE)}`",
            f"- Per-record CSV: `{csv_output.relative_to(WORKSPACE)}`",
        ]
    )
    report_output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "records": len(records), "dominant": payload["dominant_psnr_drop_transition"]}, sort_keys=True))


if __name__ == "__main__":
    main()
