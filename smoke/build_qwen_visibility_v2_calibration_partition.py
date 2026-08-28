#!/usr/bin/env python3
"""Freeze a source-disjoint training-only Qwen calibration partition."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lowlight_restoration.multidegradation_data import read_jsonl, sha256, write_new_or_verify
from lowlight_restoration.qwen_type_calibration import build_training_source_partition


def main() -> None:
    manifest = ROOT / "manifests/qwen_multidegradation_pilot_v2.jsonl"
    output = ROOT / "manifests/qwen_multidegradation_visibility_v2_calibration.json"
    manifest_sha = sha256(manifest)
    payload = build_training_source_partition(
        read_jsonl(manifest),
        manifest_sha256=manifest_sha,
        seed=2026082817,
        calibration_fraction_numerator=1,
        calibration_fraction_denominator=8,
        training_image_batch_multiple=40,
        minimum_calibration_type_level_records=50,
    )
    payload["manifest"] = manifest.relative_to(ROOT).as_posix()
    digest = write_new_or_verify(
        output, (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    )
    print(
        "QWEN_VISIBILITY_V2_CALIBRATION_PARTITION_PASS "
        + json.dumps(
            {
                "output": output.relative_to(ROOT).as_posix(),
                "sha256": digest,
                "calibration_records": payload["calibration_records"],
                "gradient_eligible_records": payload["gradient_eligible_records"],
                "gradient_records_per_epoch": payload["gradient_records_per_epoch"],
                "minimum_type_level_records": min(
                    payload["calibration_type_level_counts"].values()
                ),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
