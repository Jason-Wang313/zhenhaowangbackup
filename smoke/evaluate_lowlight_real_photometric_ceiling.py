#!/usr/bin/env python3
"""Measure input and oracle-global-photometric ceilings on paired real data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import subprocess
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Mapping

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lowlight_restoration.paired_development import (  # noqa: E402
    clustered_metric_summary,
    fit_oracle_gamma_color,
    gradient_alignment_diagnostic,
    peak_signal_noise_ratio,
    rgb_structural_similarity,
)


DEFAULT_CONFIG = ROOT / "configs/lowlight_real_photometric_ceiling_v1.json"
REQUIRED_MANIFEST_FIELDS = {
    "development_id",
    "dataset",
    "scene_id",
    "target_path",
    "input_path",
    "target_sha256",
    "input_sha256",
    "target_kind",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def workspace_path(raw_path: str, *, label: str) -> Path:
    path = (ROOT / raw_path).resolve()
    try:
        path.relative_to(ROOT)
    except ValueError as exc:
        raise RuntimeError(f"{label} escapes workspace: {path}") from exc
    return path


def write_once_or_verify(path: Path, content: bytes) -> None:
    if path.exists():
        if path.read_bytes() != content:
            raise RuntimeError(f"existing immutable output differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(content)


def load_config(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise RuntimeError(f"missing photometric-ceiling config: {path}")
    payload = json.loads(path.read_text())
    required = {
        "schema_version",
        "input_manifest",
        "input_manifest_sha256",
        "output_csv",
        "output_json",
        "oracle",
        "alignment",
        "bootstrap",
        "metrics",
    }
    if set(payload) != required or payload["schema_version"] != 1:
        raise RuntimeError("photometric-ceiling config schema changed")
    oracle = payload["oracle"]
    if not isinstance(oracle, dict):
        raise RuntimeError("oracle config must be an object")
    candidates = oracle.get("gamma_candidates")
    if (
        not isinstance(candidates, list)
        or not candidates
        or len(set(candidates)) != len(candidates)
        or any(not isinstance(value, (int, float)) or value <= 0 for value in candidates)
    ):
        raise RuntimeError("invalid oracle gamma candidates")
    return payload


def load_manifest(config: Mapping[str, object]) -> list[dict[str, str]]:
    path = workspace_path(str(config["input_manifest"]), label="input manifest")
    if sha256(path) != str(config["input_manifest_sha256"]):
        raise RuntimeError("paired-development manifest identity changed")
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not REQUIRED_MANIFEST_FIELDS.issubset(
            reader.fieldnames
        ):
            raise RuntimeError("paired-development manifest schema changed")
        rows = list(reader)
    if not rows or len({row["development_id"] for row in rows}) != len(rows):
        raise RuntimeError("paired-development manifest is empty or duplicated")
    return rows


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def mean_luma(image: np.ndarray) -> float:
    return float(
        np.mean(
            0.2126 * image[..., 0]
            + 0.7152 * image[..., 1]
            + 0.0722 * image[..., 2]
        )
    )


def evaluate_record(arguments: tuple[dict[str, str], dict[str, object]]) -> dict[str, object]:
    row, config = arguments
    cv2.setNumThreads(1)
    target_path = workspace_path(row["target_path"], label="target image")
    input_path = workspace_path(row["input_path"], label="input image")
    if sha256(target_path) != row["target_sha256"]:
        raise RuntimeError(f"target identity changed: {row['development_id']}")
    if sha256(input_path) != row["input_sha256"]:
        raise RuntimeError(f"input identity changed: {row['development_id']}")
    target = load_rgb(target_path)
    source = load_rgb(input_path)
    if target.shape != source.shape:
        raise RuntimeError(f"paired dimensions changed: {row['development_id']}")

    oracle_config = config["oracle"]
    assert isinstance(oracle_config, dict)
    fit = fit_oracle_gamma_color(
        source,
        target,
        gamma_candidates=tuple(float(value) for value in oracle_config["gamma_candidates"]),
        maximum_fit_pixels=int(oracle_config["maximum_fit_pixels"]),
        ridge=float(oracle_config["ridge"]),
    )
    continuous_prediction = np.asarray(fit["prediction"], dtype=np.float32)
    oracle_prediction = np.rint(continuous_prediction * 255.0) / 255.0
    alignment_config = config["alignment"]
    assert isinstance(alignment_config, dict)
    alignment = gradient_alignment_diagnostic(
        source,
        target,
        maximum_dimension=int(alignment_config["maximum_dimension"]),
    )
    alignment_flag = (
        alignment["estimated_shift_magnitude_pixels"]
        > float(alignment_config["flag_maximum_shift_pixels"])
        or alignment["gradient_correlation_shifted"]
        < float(alignment_config["flag_minimum_shifted_gradient_correlation"])
    )
    input_psnr = peak_signal_noise_ratio(target, source)
    oracle_psnr = peak_signal_noise_ratio(target, oracle_prediction)
    input_ssim = rgb_structural_similarity(target, source)
    oracle_ssim = rgb_structural_similarity(target, oracle_prediction)
    coefficients = np.asarray(fit["coefficients"], dtype=np.float64)
    result: dict[str, object] = {
        "development_id": row["development_id"],
        "dataset": row["dataset"],
        "scene_id": row["scene_id"],
        "target_kind": row["target_kind"],
        "width": int(target.shape[1]),
        "height": int(target.shape[0]),
        "input_mean_luma": mean_luma(source),
        "target_mean_luma": mean_luma(target),
        "input_shadow_fraction_luma_le_4": float(
            np.mean(
                (
                    0.2126 * source[..., 0]
                    + 0.7152 * source[..., 1]
                    + 0.0722 * source[..., 2]
                )
                <= (4.0 / 255.0)
            )
        ),
        "input_psnr": input_psnr,
        "oracle_psnr": oracle_psnr,
        "psnr_gain": oracle_psnr - input_psnr,
        "input_ssim": input_ssim,
        "oracle_ssim": oracle_ssim,
        "ssim_gain": oracle_ssim - input_ssim,
        "oracle_gamma": float(fit["gamma"]),
        "oracle_fit_mse": float(fit["fit_mse"]),
        "oracle_clipped_pixel_fraction": float(
            np.mean(np.any((continuous_prediction <= 0.0) | (continuous_prediction >= 1.0), axis=2))
        ),
        "oracle_coefficients_json": json.dumps(
            coefficients.tolist(), separators=(",", ":")
        ),
        **alignment,
        "alignment_diagnostic_flag": str(bool(alignment_flag)).lower(),
    }
    for key, value in result.items():
        if isinstance(value, float) and not math.isfinite(value):
            raise RuntimeError(f"non-finite {key}: {row['development_id']}")
    return result


def git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    args = parser.parse_args()
    if args.workers <= 0:
        raise RuntimeError("workers must be positive")
    config_path = args.config.resolve()
    try:
        config_path.relative_to(ROOT)
    except ValueError as exc:
        raise RuntimeError("config must be inside the workspace") from exc
    config = load_config(config_path)
    rows = load_manifest(config)
    arguments = [(row, config) for row in rows]
    if args.workers == 1:
        results = [evaluate_record(item) for item in arguments]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            results = list(executor.map(evaluate_record, arguments, chunksize=1))

    fields = list(results[0])
    csv_buffer = io.StringIO(newline="")
    writer = csv.DictWriter(csv_buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(results)
    csv_content = csv_buffer.getvalue().encode("utf-8")
    bootstrap = config["bootstrap"]
    alignment = config["alignment"]
    assert isinstance(bootstrap, dict) and isinstance(alignment, dict)
    metric_fields = (
        "input_psnr",
        "oracle_psnr",
        "psnr_gain",
        "input_ssim",
        "oracle_ssim",
        "ssim_gain",
    )
    summary = clustered_metric_summary(
        results,
        fields=metric_fields,
        bootstrap_seed=int(bootstrap["seed"]),
        bootstrap_resamples=int(bootstrap["resamples"]),
        confidence=float(bootstrap["confidence"]),
    )
    output_csv = workspace_path(str(config["output_csv"]), label="output CSV")
    output_json = workspace_path(str(config["output_json"]), label="output JSON")
    payload = {
        "schema_version": 1,
        "status": "completed_oracle_diagnostic_not_deployable_model",
        "official_foundir_inference_performed": False,
        "official_foundir_images_opened": False,
        "oracle_uses_each_development_target": True,
        "oracle_must_never_be_used_as_inference": True,
        "config": str(config_path.relative_to(ROOT)),
        "config_sha256": sha256(config_path),
        "input_manifest": str(config["input_manifest"]),
        "input_manifest_sha256": str(config["input_manifest_sha256"]),
        "output_csv": str(output_csv.relative_to(ROOT)),
        "output_csv_sha256": hashlib.sha256(csv_content).hexdigest(),
        "code": {
            "git_head": git_head(),
            "evaluator_sha256": sha256(Path(__file__).resolve()),
            "paired_development_module_sha256": sha256(
                ROOT / "lowlight_restoration/paired_development.py"
            ),
        },
        "metrics": config["metrics"],
        "oracle": config["oracle"],
        "alignment": {
            **alignment,
            "diagnostic_only_no_metric_warping": True,
            "flagged_records": sum(
                row["alignment_diagnostic_flag"] == "true" for row in results
            ),
            "flagged_by_dataset": dict(
                sorted(
                    Counter(
                        str(row["dataset"])
                        for row in results
                        if row["alignment_diagnostic_flag"] == "true"
                    ).items()
                )
            ),
        },
        "gamma_selection_counts": dict(
            sorted(Counter(str(row["oracle_gamma"]) for row in results).items())
        ),
        "summary": summary,
    }
    json_content = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    write_once_or_verify(output_csv, csv_content)
    write_once_or_verify(output_json, json_content)
    point = summary["point_estimates"]["scene_balanced"]
    print(
        "LOWLIGHT_REAL_PHOTOMETRIC_CEILING_PASS "
        f"records={len(results)} scenes={summary['scenes']} "
        f"input_psnr={point['input_psnr']:.4f} "
        f"oracle_psnr={point['oracle_psnr']:.4f} "
        f"gain={point['psnr_gain']:.4f} "
        f"alignment_flags={payload['alignment']['flagged_records']}"
    )


if __name__ == "__main__":
    main()
