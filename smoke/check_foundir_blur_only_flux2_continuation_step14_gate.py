#!/usr/bin/env python3
"""Prove that step 12k warrants the conditionally authorized 12k→14k stage."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping

ROOT = Path(__file__).resolve().parents[1]
BASELINE_PSNR = 25.734236
SOW_LPIPS_MAXIMUM = 0.18


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_gate(result: Mapping[str, object], *, expected_config_sha: str) -> dict[str, float]:
    curve = result.get("checkpoint_curve")
    if (
        result.get("status") != "blur_only_full_development_evaluation_complete"
        or result.get("config_sha256") != expected_config_sha
        or result.get("records") != 2595
        or result.get("source_groups") != 165
        or result.get("sealed_internal_accessed") is not False
        or result.get("protected_foundir_test_accessed") is not False
        or not isinstance(curve, list)
        or len(curve) != 1
        or int(curve[0].get("checkpoint_step", -1)) != 12000
    ):
        raise RuntimeError("step-12k full-development result identity changed")
    metrics = curve[0].get("metrics", {})
    psnr, lpips = metrics.get("psnr"), metrics.get("lpips")
    if not all(isinstance(value, (int, float)) and math.isfinite(float(value)) for value in (psnr, lpips)):
        raise RuntimeError("step-12k metrics are invalid")
    if float(psnr) <= BASELINE_PSNR:
        raise RuntimeError(
            f"step-14k gate closed: step-12k PSNR {float(psnr):.6f} did not improve "
            f"over step-10k {BASELINE_PSNR:.6f}"
        )
    if float(lpips) > SOW_LPIPS_MAXIMUM:
        raise RuntimeError(
            f"step-14k gate closed: step-12k LPIPS {float(lpips):.6f} lost the <=0.18 pass"
        )
    return {"step12_psnr": float(psnr), "step12_lpips": float(lpips)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result",
        type=Path,
        default=ROOT / "checkpoints/foundir_blur_only_flux2_continuation_v1/full_development_step_012000/result.json",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/foundir_blur_only_flux2_continuation_development_v1.json",
    )
    args = parser.parse_args()
    result_path, config_path = args.result.resolve(), args.config.resolve()
    metrics = validate_gate(json.loads(result_path.read_text()), expected_config_sha=sha256(config_path))
    checkpoint = ROOT / "checkpoints/foundir_blur_only_flux2_continuation_v1/step_012000.safetensors"
    optimizer = checkpoint.with_suffix(".optimizer.pt")
    visual = ROOT / "checkpoints/foundir_blur_only_flux2_continuation_v1/fixed_validation/training_triptychs/step_012000"
    if not checkpoint.is_file() or not optimizer.is_file():
        raise RuntimeError("step-12k checkpoint or optimizer state is missing")
    if not (visual / "metrics.json").is_file() or len(list(visual.glob("*.png"))) != 16:
        raise RuntimeError("step-12k fixed visual snapshot is incomplete")
    print("FOUNDIR_BLUR_FLUX2_CONTINUATION_STEP14_GATE_PASS " + json.dumps({
        **metrics,
        "baseline_step10_psnr": BASELINE_PSNR,
        "lpips_maximum": SOW_LPIPS_MAXIMUM,
        "checkpoint_sha256": sha256(checkpoint),
        "fixed_visual_records": 16,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
