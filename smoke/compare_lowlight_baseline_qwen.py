#!/usr/bin/env python3
"""Paired internal comparison of baseline and Qwen-conditioned restoration."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lowlight_restoration.authorization import sha256
from lowlight_restoration.conditioned_authorization import (
    QWEN_CHECKPOINT_SHA256,
    validate_conditioned_authorization,
    workspace_path,
)
from lowlight_restoration.config import load_train_config
from lowlight_restoration.data import LowLightPairDataset, load_pair_manifest
from lowlight_restoration.qwen_predictions import load_prediction_manifest
from predict_qwen_lowlight import atomic_write_new
from smoke.evaluate_lowlight_overfit import (
    labeled_thumbnail,
    require_finite_numbers,
    save_image_once_or_verify,
    tensor_to_image,
)


def git_revision() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def mean(values: list[float]) -> float:
    if not values:
        raise RuntimeError("cannot average an empty metric list")
    return sum(values) / len(values)


def load_bound_json(path: Path, expected_sha: str | None = None) -> dict[str, object]:
    if not path.is_file():
        raise RuntimeError(f"missing comparison artifact: {path}")
    if expected_sha is not None and sha256(path) != expected_sha:
        raise RuntimeError(f"comparison artifact hash mismatch: {path}")
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise RuntimeError(f"comparison artifact is not an object: {path}")
    return payload


def paired_sheet(
    rows: list[tuple[str, Image.Image, Image.Image, Image.Image, Image.Image]],
    output: Path,
) -> str:
    cell = 256
    sheet = Image.new("RGB", (cell * 4, (cell + 24) * len(rows)), "white")
    for row_index, (label, degraded, baseline, conditioned, clean) in enumerate(rows):
        for column, (name, image) in enumerate(
            (
                ("degraded", degraded),
                ("baseline", baseline),
                ("qwen", conditioned),
                ("clean", clean),
            )
        ):
            tile = labeled_thumbnail(image, f"{label} | {name}", size=cell)
            sheet.paste(tile, (column * cell, row_index * (cell + 24)))
    save_image_once_or_verify(output, sheet, quality=92)
    return str(output.relative_to(ROOT))


def summarize_pairs(pairs: list[dict[str, object]]) -> dict[str, object]:
    psnr = [float(item["psnr_delta"]) for item in pairs]
    ssim = [float(item["ssim_delta"]) for item in pairs]
    lpips = [float(item["lpips_delta"]) for item in pairs]
    return {
        "samples": len(pairs),
        "mean_psnr_delta": mean(psnr),
        "median_psnr_delta": median(psnr),
        "psnr_win_rate": sum(value > 0 for value in psnr) / len(psnr),
        "mean_ssim_delta": mean(ssim),
        "median_ssim_delta": median(ssim),
        "ssim_win_rate": sum(value > 0 for value in ssim) / len(ssim),
        "mean_lpips_delta": mean(lpips),
        "median_lpips_delta": median(lpips),
        "lpips_win_rate": sum(value < 0 for value in lpips) / len(lpips),
        "baseline_restored_psnr": mean(
            [float(item["baseline_restored_psnr"]) for item in pairs]
        ),
        "conditioned_restored_psnr": mean(
            [float(item["conditioned_restored_psnr"]) for item in pairs]
        ),
        "baseline_restored_ssim": mean(
            [float(item["baseline_restored_ssim"]) for item in pairs]
        ),
        "conditioned_restored_ssim": mean(
            [float(item["conditioned_restored_ssim"]) for item in pairs]
        ),
        "baseline_restored_lpips": mean(
            [float(item["baseline_restored_lpips"]) for item in pairs]
        ),
        "conditioned_restored_lpips": mean(
            [float(item["conditioned_restored_lpips"]) for item in pairs]
        ),
    }


def compare_split(
    baseline: dict[str, object], conditioned: dict[str, object], split: str
) -> tuple[dict[str, object], list[dict[str, object]]]:
    if (
        baseline.get("split") != split
        or conditioned.get("split") != split
        or baseline.get("samples") != 1000
        or conditioned.get("samples") != 1000
        or baseline.get("guidance") != conditioned.get("guidance")
    ):
        raise RuntimeError(f"{split} aggregate contract differs")
    baseline_records = baseline.get("per_image")
    conditioned_records = conditioned.get("per_image")
    if not isinstance(baseline_records, list) or not isinstance(conditioned_records, list):
        raise RuntimeError(f"{split} per-image evidence is missing")
    pairs: list[dict[str, object]] = []
    for index, (base, qwen) in enumerate(
        zip(baseline_records, conditioned_records, strict=True)
    ):
        if not isinstance(base, dict) or not isinstance(qwen, dict):
            raise RuntimeError(f"{split} record {index} is invalid")
        for key in ("index", "pair_id", "project_level", "split", "crop_xy", "seed"):
            if base.get(key) != qwen.get(key):
                raise RuntimeError(f"{split} pair differs at {key}: index {index}")
        for key in ("degraded_psnr", "degraded_ssim", "degraded_lpips"):
            tolerance = 1e-7 if key == "degraded_lpips" else 1e-12
            if not math.isclose(
                float(base[key]), float(qwen[key]), rel_tol=0.0, abs_tol=tolerance
            ):
                raise RuntimeError(f"{split} input metric differs at {key}: index {index}")
        pair = {
            "index": index,
            "pair_id": base["pair_id"],
            "project_level": base["project_level"],
            "predicted_level": qwen["predicted_level"],
            "condition_correct": qwen["condition_correct"],
            "baseline_restored_path": base["restored_path"],
            "baseline_restored_sha256": base["restored_sha256"],
            "conditioned_restored_path": qwen["restored_path"],
            "conditioned_restored_sha256": qwen["restored_sha256"],
            "baseline_restored_psnr": base["restored_psnr"],
            "conditioned_restored_psnr": qwen["restored_psnr"],
            "psnr_delta": float(qwen["restored_psnr"]) - float(base["restored_psnr"]),
            "baseline_restored_ssim": base["restored_ssim"],
            "conditioned_restored_ssim": qwen["restored_ssim"],
            "ssim_delta": float(qwen["restored_ssim"]) - float(base["restored_ssim"]),
            "baseline_restored_lpips": base["restored_lpips"],
            "conditioned_restored_lpips": qwen["restored_lpips"],
            "lpips_delta": float(qwen["restored_lpips"]) - float(base["restored_lpips"]),
        }
        pairs.append(pair)
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for pair in pairs:
        grouped[str(pair["project_level"])].append(pair)
    degraded = [pair for pair in pairs if pair["project_level"] != "L1"]
    summary = {
        "all": summarize_pairs(pairs),
        "l2_l5": summarize_pairs(degraded),
        "l1_identity": summarize_pairs(grouped["L1"]),
        "by_level": {
            level: summarize_pairs(items) for level, items in sorted(grouped.items())
        },
        "condition_accuracy": sum(bool(pair["condition_correct"]) for pair in pairs)
        / len(pairs),
    }
    return summary, pairs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = load_train_config(args.config)
    try:
        scope = validate_conditioned_authorization(config)
    except RuntimeError as exc:
        raise SystemExit(f"RUN_NOT_AUTHORIZED {exc}") from exc
    if scope != "full_qwen_conditioned":
        raise SystemExit("QWEN_CONDITIONED_SCOPE_REQUIRED")

    comparison = config.section("comparison")
    output_dir = config.workspace_path("runtime", "output_dir")
    baseline_paths = {
        "validation": workspace_path(
            comparison["baseline_validation_metrics"],
            label="comparison.baseline_validation_metrics",
        ),
        "test": workspace_path(
            comparison["baseline_test_metrics"],
            label="comparison.baseline_test_metrics",
        ),
    }
    conditioned_paths = {
        "validation": output_dir / "validation_full/metrics.json",
        "test": output_dir / "test_full/metrics.json",
    }
    baseline_hashes = {
        "validation": str(comparison["baseline_validation_metrics_sha256"]),
        "test": str(comparison["baseline_test_metrics_sha256"]),
    }
    summaries: dict[str, dict[str, object]] = {}
    paired: dict[str, list[dict[str, object]]] = {}
    artifact_hashes: dict[str, dict[str, str]] = {}
    for split in ("validation", "test"):
        baseline = load_bound_json(baseline_paths[split], baseline_hashes[split])
        conditioned = load_bound_json(conditioned_paths[split])
        if conditioned.get("automatic_sanity_pass") is not True:
            raise RuntimeError(f"conditioned {split} automatic sanity did not pass")
        summaries[split], paired[split] = compare_split(baseline, conditioned, split)
        artifact_hashes[split] = {
            "baseline": sha256(baseline_paths[split]),
            "conditioned": sha256(conditioned_paths[split]),
        }

    test_primary = summaries["test"]["l2_l5"]
    test_l1 = summaries["test"]["l1_identity"]
    assert isinstance(test_primary, dict) and isinstance(test_l1, dict)
    gates = {
        "test_l2_l5_psnr_improved": float(test_primary["mean_psnr_delta"]) > 0,
        "test_l2_l5_ssim_improved": float(test_primary["mean_ssim_delta"]) > 0,
        "test_l2_l5_lpips_improved": float(test_primary["mean_lpips_delta"]) < 0,
        "test_l1_ssim_noninferior": float(test_l1["mean_ssim_delta"])
        >= -float(comparison["maximum_l1_ssim_regression"]),
    }
    automatic_pass = all(gates.values())

    records = load_pair_manifest(config.workspace_path("data", "manifest"))
    prediction_levels, _, prediction_manifest_sha = load_prediction_manifest(
        config, records, verify_sidecars=False
    )
    test_dataset = LowLightPairDataset(
        records,
        split="test",
        crop_size=int(config.section("data")["crop_size"]),
        seed=int(config.section("training")["seed"]),
    )
    test_pairs = paired["test"]
    selections = {
        "stratified": [
            next(
                int(pair["index"])
                for pair in test_pairs
                if pair["project_level"] == f"L{level}"
            )
            for level in range(1, 6)
        ],
        "largest_regressions": [
            int(pair["index"])
            for pair in sorted(test_pairs, key=lambda item: float(item["psnr_delta"]))[:8]
        ],
        "largest_gains": [
            int(pair["index"])
            for pair in sorted(
                test_pairs, key=lambda item: float(item["psnr_delta"]), reverse=True
            )[:8]
        ],
    }
    by_index = {int(pair["index"]): pair for pair in test_pairs}
    contact_sheets: list[str] = []
    for name, indices in selections.items():
        rows = []
        for index in indices:
            pair = by_index[index]
            item = test_dataset[index]
            pair_id = str(pair["pair_id"])
            if prediction_levels[pair_id] != pair["predicted_level"]:
                raise RuntimeError(f"comparison prediction changed: {pair_id}")
            baseline_path = workspace_path(
                pair["baseline_restored_path"], label="baseline restored path"
            )
            conditioned_path = workspace_path(
                pair["conditioned_restored_path"], label="conditioned restored path"
            )
            if (
                sha256(baseline_path) != pair["baseline_restored_sha256"]
                or sha256(conditioned_path) != pair["conditioned_restored_sha256"]
            ):
                raise RuntimeError(f"comparison restored image changed: {pair_id}")
            with Image.open(baseline_path) as source:
                baseline_image = source.convert("RGB").copy()
            with Image.open(conditioned_path) as source:
                conditioned_image = source.convert("RGB").copy()
            label = (
                f"{pair_id} pred=L{pair['predicted_level']} "
                f"dPSNR={float(pair['psnr_delta']):+.2f}"
            )
            rows.append(
                (
                    label,
                    tensor_to_image(item["degraded"]),
                    baseline_image,
                    conditioned_image,
                    tensor_to_image(item["clean"]),
                )
            )
        contact_sheets.append(
            paired_sheet(rows, output_dir / f"comparison_contact_{name}.jpg")
        )

    result = {
        "schema_version": 1,
        "status": (
            "quantitative_pass_visual_review_required"
            if automatic_pass
            else "quantitative_failed"
        ),
        "scope": "synthetic_internal_lowlight_baseline_vs_qwen_conditioned",
        "config_sha256": config.sha256,
        "git_revision": git_revision(),
        "prediction_manifest_sha256": prediction_manifest_sha,
        "qwen_checkpoint_sha256": QWEN_CHECKPOINT_SHA256,
        "artifact_sha256": artifact_hashes,
        "comparison_protocol": comparison,
        "splits": summaries,
        "gates": gates,
        "automatic_quantitative_improvement_pass": automatic_pass,
        "visual_identity_review_required": True,
        "visual_identity_review_pass": None,
        "contact_sheets": contact_sheets,
        "contact_sheet_sha256": {
            relative: sha256(workspace_path(relative, label="comparison contact sheet"))
            for relative in contact_sheets
        },
        "internal_test_used": True,
        "official_foundir_used": False,
        "paired_test_records": test_pairs,
    }
    require_finite_numbers(result)
    atomic_write_new(output_dir / "comparison.json", result)
    marker = (
        "LOWLIGHT_QWEN_CONDITIONING_QUANTITATIVE_PASS_VISUAL_REVIEW_REQUIRED"
        if automatic_pass
        else "LOWLIGHT_QWEN_CONDITIONING_QUANTITATIVE_FAILED"
    )
    print(
        marker
        + " "
        + json.dumps(
            {
                "gates": gates,
                "test_l2_l5": test_primary,
                "test_l1_identity": test_l1,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if not automatic_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
