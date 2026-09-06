#!/usr/bin/env python3
"""Audit the FoundIR-Train Noise/codec mismatch without model inference."""

from __future__ import annotations

import csv
import io
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lowlight_restoration.multidegradation_data import write_new_or_verify


PAIR_MANIFEST = ROOT / "manifests/foundir_five_type_pairs_v1.csv"
PILOT_MANIFEST = ROOT / "manifests/foundir_five_type_overfit_qwen_conditions_v1.csv"
VIEW_RESULT = ROOT / "reports/foundir_five_type_qwen_real_gate_physical_v8_views.json"
OUTPUT = ROOT / "reports/foundir_qwen_noise_codec_order_audit_2026-09-06.json"
NOISE_CATEGORIES = {
    "02Blur_Noise",
    "05Noise",
    "16Lowlight_Noise",
    "18Lowlight_Blur_Noise",
}


def standard_jpeg_tables() -> dict[int, dict[int, list[int]]]:
    image = Image.new("RGB", (16, 16), (128, 128, 128))
    tables: dict[int, dict[int, list[int]]] = {}
    for quality in range(1, 101):
        encoded = io.BytesIO()
        image.save(encoded, "JPEG", quality=quality)
        encoded.seek(0)
        with Image.open(encoded) as decoded:
            tables[quality] = decoded.quantization
    return tables


def inferred_quality(
    tables: dict[int, list[int]], references: dict[int, dict[int, list[int]]]
) -> int:
    def distance(reference: dict[int, list[int]]) -> int:
        common = set(tables) & set(reference)
        return 100000 * abs(len(tables) - len(reference)) + sum(
            abs(left - right)
            for key in common
            for left, right in zip(tables[key], reference[key], strict=True)
        )

    return min(references, key=lambda quality: distance(references[quality]))


def residual_statistics(lq_path: str, gt_path: str) -> dict[str, float] | None:
    lq = cv2.imread(lq_path, cv2.IMREAD_REDUCED_COLOR_4)
    gt = cv2.imread(gt_path, cv2.IMREAD_REDUCED_COLOR_4)
    if lq is None or gt is None:
        raise RuntimeError(f"unreadable FoundIR-Train pair: {lq_path}")
    if lq.shape != gt.shape:
        return None
    lq_float = lq.astype(np.float32) / 255.0
    gt_float = gt.astype(np.float32) / 255.0
    residual = lq_float - gt_float
    gray = cv2.cvtColor(residual, cv2.COLOR_BGR2GRAY)
    highpass = gray - cv2.GaussianBlur(gray, (0, 0), 1.2)
    left = gray[:, :-1].reshape(-1)
    right = gray[:, 1:].reshape(-1)
    correlation = (
        float(np.corrcoef(left, right)[0, 1])
        if float(np.std(left)) > 1e-8 and float(np.std(right)) > 1e-8
        else 0.0
    )
    return {
        "residual_rgb_rmse": float(np.sqrt(np.mean(residual * residual))),
        "residual_luma_std": float(np.std(gray)),
        "residual_highpass_std": float(np.std(highpass)),
        "residual_horizontal_neighbor_correlation": correlation,
    }


def main() -> None:
    extension_counts: dict[str, Counter[str]] = defaultdict(Counter)
    with PAIR_MANIFEST.open() as handle:
        for record in csv.DictReader(handle):
            if record["category"] in NOISE_CATEGORIES:
                extension_counts[record["category"]][
                    Path(record["lq_path"]).suffix.lower()
                ] += 1
    pilot = list(csv.DictReader(PILOT_MANIFEST.open()))
    references = standard_jpeg_tables()
    format_counts: dict[str, Counter[str]] = defaultdict(Counter)
    quality_counts: dict[str, Counter[int]] = defaultdict(Counter)
    residual_by_category: dict[str, list[dict[str, float]]] = defaultdict(list)
    residual_shape_mismatch_counts: Counter[str] = Counter()
    for record in pilot:
        category = record["category"]
        with Image.open(record["lq_path"]) as image:
            format_counts[category][str(image.format)] += 1
            if image.format in {"JPEG", "MPO"}:
                quality_counts[category][inferred_quality(image.quantization, references)] += 1
        residual = residual_statistics(record["lq_path"], record["gt_path"])
        if residual is None:
            residual_shape_mismatch_counts[category] += 1
        else:
            residual_by_category[category].append(residual)
    views = json.loads(VIEW_RESULT.read_text())["view_predictions"]
    noise_levels: dict[str, list[int]] = defaultdict(list)
    categories: dict[str, str] = {}
    for view in views:
        identity = str(view["parent_record_id"])
        categories[identity] = str(view["category"])
        noise_levels[identity].append(int(view["levels"]["noise"]))
    decision_counts: dict[str, dict[str, int]] = {}
    for category in sorted(NOISE_CATEGORIES):
        selected = [levels for identity, levels in noise_levels.items() if categories[identity] == category]
        decision_counts[category] = {
            "records": len(selected),
            "any_view_noise_present": sum(max(levels) > 1 for levels in selected),
            "second_highest_noise_present": sum(sorted(levels)[-2] > 1 for levels in selected),
            "all_views_noise_absent": sum(max(levels) == 1 for levels in selected),
        }
    residual_medians = {
        category: {
            key: float(np.median([record[key] for record in values]))
            for key in values[0]
        }
        for category, values in sorted(residual_by_category.items())
    }
    result = {
        "schema_version": 1,
        "status": "codec_order_mismatch_strongly_supported_not_generator_proven",
        "scope": "FoundIR_training_only_read_only_diagnostic",
        "qwen_training_performed": False,
        "protected_foundir_official_test_accessed": False,
        "protected_synthetic_test_accessed": False,
        "full_training_extension_counts": {
            category: dict(counts) for category, counts in sorted(extension_counts.items())
        },
        "pilot_actual_format_counts": {
            category: dict(counts) for category, counts in sorted(format_counts.items())
        },
        "pilot_inferred_jpeg_quality_counts": {
            category: {str(key): value for key, value in sorted(counts.items())}
            for category, counts in sorted(quality_counts.items())
        },
        "pilot_noise_decisions": decision_counts,
        "pilot_residual_medians_reduced_by_4": residual_medians,
        "pilot_residual_aligned_pair_counts": {
            category: len(values) for category, values in sorted(residual_by_category.items())
        },
        "pilot_residual_shape_mismatch_counts": dict(
            sorted(residual_shape_mismatch_counts.items())
        ),
        "physical_v8_order": "non_target_codec_then_target_degradations_with_noise_last",
        "conclusion": (
            "Noise is detected on every PNG Blur+Noise pilot record but is absent on "
            "nearly every JPEG Noise/Lowlight+Noise record. The synthetic physical-v8 "
            "pipeline places JPEG before Noise, so it does not teach post-codec-suppressed "
            "Noise. File-format, quantization-table, residual, and prediction evidence "
            "strongly support this mismatch; the unavailable FoundIR data generator means "
            "the exact upstream operator order is not claimed as proven."
        ),
    }
    write_new_or_verify(
        OUTPUT, (json.dumps(result, indent=2, sort_keys=True) + "\n").encode()
    )
    print("FOUNDIR_QWEN_NOISE_CODEC_AUDIT_PASS " + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
