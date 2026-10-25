#!/usr/bin/env python3
"""Compare the two fixed-final-step VLM-token MoE holdout results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
LABELS = ("blur", "noise", "rain", "haze", "low_light")


def load_rows(path: Path, expected_arm: str) -> list[dict[str, object]]:
    payload = json.loads(path.read_text())
    if (
        payload.get("status") != "matched_formal_evaluation_complete"
        or payload.get("arm") != expected_arm
        or payload.get("split") != "holdout"
        or int(payload.get("step", -1)) != 1000
        or payload.get("official_foundir_test_accessed") is not False
        or payload.get("sealed_internal_accessed") is not False
        or payload.get("parent_test_accessed") is not False
        or payload.get("internal_test_accessed") is not False
    ):
        raise RuntimeError(f"holdout result contract changed: {path}")
    rows = payload.get("rows")
    if not isinstance(rows, list) or len(rows) != 450:
        raise RuntimeError(f"holdout row count changed: {path}")
    if len({str(row["record_id"]) for row in rows}) != 450:
        raise RuntimeError(f"duplicate holdout records: {path}")
    return rows


def percentile_interval(values: np.ndarray, seed: int, draws: int = 10000) -> list[float]:
    generator = np.random.default_rng(seed)
    means = np.empty(draws, dtype=np.float64)
    for offset in range(0, draws, 500):
        size = min(500, draws - offset)
        indices = generator.integers(0, len(values), size=(size, len(values)))
        means[offset : offset + size] = values[indices].mean(axis=1)
    return [float(value) for value in np.percentile(means, [2.5, 97.5])]


def paired_summary(
    image_rows: Sequence[Mapping[str, object]],
    vlm_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    image = {str(row["record_id"]): row for row in image_rows}
    vlm = {str(row["record_id"]): row for row in vlm_rows}
    if set(image) != set(vlm):
        raise RuntimeError("matched holdout identities differ between arms")
    ids = sorted(image)
    psnr_delta = np.asarray(
        [float(vlm[key]["output_psnr"]) - float(image[key]["output_psnr"]) for key in ids]
    )
    lpips_delta = np.asarray(
        [float(vlm[key]["output_lpips"]) - float(image[key]["output_lpips"]) for key in ids]
    )
    labels = np.asarray([str(image[key]["label"]) for key in ids])
    if any(str(image[key]["label"]) != str(vlm[key]["label"]) for key in ids):
        raise RuntimeError("matched holdout labels differ between arms")

    def one(indices: np.ndarray, seed: int) -> dict[str, object]:
        p = psnr_delta[indices]
        l = lpips_delta[indices]
        return {
            "records": int(len(indices)),
            "mean_psnr_delta_db_vlm_minus_image": float(p.mean()),
            "mean_lpips_delta_vlm_minus_image": float(l.mean()),
            "psnr_delta_95pct_paired_bootstrap": percentile_interval(p, seed),
            "lpips_delta_95pct_paired_bootstrap": percentile_interval(l, seed + 1),
            "psnr_record_win_rate": float(np.mean(p > 0)),
            "lpips_record_win_rate": float(np.mean(l < 0)),
            "joint_record_win_rate": float(np.mean((p > 0) & (l < 0))),
        }

    overall = one(np.arange(len(ids)), 2026102503)
    by_degradation = {
        label: one(np.flatnonzero(labels == label), 2026102600 + index * 10)
        for index, label in enumerate(LABELS)
    }
    psnr_delta_mean = float(overall["mean_psnr_delta_db_vlm_minus_image"])
    lpips_delta_mean = float(overall["mean_lpips_delta_vlm_minus_image"])
    return {
        "schema_version": 1,
        "status": "matched_formal_comparison_complete",
        "primary_hypothesis": "vlm_assisted_improves_both_mean_holdout_psnr_and_mean_holdout_lpips_over_image_only",
        "primary_hypothesis_pass": psnr_delta_mean > 0 and lpips_delta_mean < 0,
        "overall": overall,
        "by_degradation": by_degradation,
        "official_foundir_test_accessed": False,
        "sealed_internal_accessed": False,
        "parent_test_accessed": False,
        "internal_test_accessed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT / "checkpoints/foundir_vlm_token_moe_matched_formal_v1",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports/foundir_vlm_token_moe_matched_formal_v1_comparison.json",
    )
    args = parser.parse_args()
    image_path = args.root / "image_only/holdout_step_001000.json"
    vlm_path = args.root / "vlm_assisted/holdout_step_001000.json"
    result = paired_summary(
        load_rows(image_path, "image_only"),
        load_rows(vlm_path, "vlm_assisted"),
    )
    content = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() and args.output.read_text() != content:
        raise RuntimeError(f"refusing to overwrite differing comparison: {args.output}")
    args.output.write_text(content)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
