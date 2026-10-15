#!/usr/bin/env python3
"""Audit training-only Blur severity against disclosed official aggregate range."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "manifests/foundir_blur_heavy_training_v1.csv"
OUTPUT = ROOT / "reports/foundir_blur_training_severity_audit_v1.json"
PER_GROUP = ROOT / "reports/foundir_blur_training_severity_per_group_v1.csv"
WORKERS = 16
SEED = 2026092115
OFFICIAL_INPUT_RANGE = (18.646469116210938, 26.396728515625)
OFFICIAL_INPUT_MEAN = 21.863902956644694


def digest(*parts: object) -> str:
    return hashlib.sha256("\0".join(map(str, parts)).encode()).hexdigest()


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def measure(item: tuple[str, list[dict[str, str]]]) -> dict[str, object]:
    group, rows = item
    row = min(rows, key=lambda value: digest(SEED, group, value["record_id"]))
    if "foundir_official" in row["lq_path"] or "foundir_official" in row["gt_path"]:
        raise RuntimeError("official path escaped into training-only severity audit")
    arrays = []
    for key in ("lq_path", "gt_path"):
        with Image.open(ROOT / row[key]) as handle:
            arrays.append(np.asarray(handle.convert("RGB"), dtype=np.uint8))
    lq, gt = arrays
    if lq.shape != gt.shape or lq.shape[:2] != (1080, 1920):
        raise RuntimeError(f"aligned 1920x1080 contract changed: {row['record_id']}")
    difference = lq.astype(np.float32) - gt.astype(np.float32)
    mse = float(np.mean(difference * difference, dtype=np.float64))
    psnr = 100.0 if mse <= 0 else 10.0 * math.log10((255.0 * 255.0) / mse)
    return {
        "source_group": group,
        "representative_record_id": row["record_id"],
        "representative_psnr": psnr,
        "group_records": len(rows),
    }


def quantile(values: list[float], fraction: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=np.float64), fraction))


def main() -> None:
    if OUTPUT.exists() or PER_GROUP.exists():
        raise RuntimeError("severity audit output already exists")
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    with MANIFEST.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["split"] == "train":
                grouped[row["source_group"]].append(dict(row))
    if len(grouped) != 2782:
        raise RuntimeError("training-only source-group count changed")
    started = time.monotonic()
    results = []
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for index, result in enumerate(pool.map(measure, sorted(grouped.items())), 1):
            results.append(result)
            if index % 200 == 0 or index == len(grouped):
                print(
                    f"FOUNDIR_BLUR_SEVERITY_PROGRESS groups={index}/{len(grouped)} "
                    f"elapsed_seconds={time.monotonic() - started:.1f}",
                    flush=True,
                )
    results.sort(key=lambda row: (float(row["representative_psnr"]), str(row["source_group"])))
    with PER_GROUP.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    values = [float(row["representative_psnr"]) for row in results]
    thresholds = [20.0, OFFICIAL_INPUT_MEAN, 22.5, 23.5, 25.0]
    payload = {
        "schema_version": 1,
        "status": "foundir_blur_training_severity_audit_complete",
        "scope": "training_only_1920x1080_one_deterministic_representative_per_source_group",
        "official_test_pixels_accessed": False,
        "official_reference": {
            "input_psnr_mean": OFFICIAL_INPUT_MEAN,
            "input_psnr_range": list(OFFICIAL_INPUT_RANGE),
            "role": "previously_disclosed_aggregate_only",
        },
        "training": {
            "source_groups": len(results),
            "records_represented": sum(int(row["group_records"]) for row in results),
            "representative_psnr_mean": float(np.mean(values)),
            "representative_psnr_quantiles": {
                str(fraction): quantile(values, fraction)
                for fraction in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
            },
            "groups_at_or_below_threshold": {
                str(threshold): sum(value <= threshold for value in values)
                for threshold in thresholds
            },
            "groups_inside_official_range": sum(
                OFFICIAL_INPUT_RANGE[0] <= value <= OFFICIAL_INPUT_RANGE[1] for value in values
            ),
        },
        "per_group_csv": PER_GROUP.relative_to(ROOT).as_posix(),
        "per_group_csv_sha256": sha256(PER_GROUP),
        "manifest": MANIFEST.relative_to(ROOT).as_posix(),
        "manifest_sha256": sha256(MANIFEST),
        "workers": WORKERS,
        "elapsed_seconds": time.monotonic() - started,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print("FOUNDIR_BLUR_SEVERITY_RESULT " + json.dumps(payload, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
