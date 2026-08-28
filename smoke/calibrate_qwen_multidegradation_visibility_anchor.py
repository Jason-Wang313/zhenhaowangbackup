#!/usr/bin/env python3
"""Calibrate visibility magnitude from training-only isolated degradations."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lowlight_restoration.multidegradation_data import (
    generate_record,
    read_jsonl,
    sha256,
    write_new_or_verify,
)
from lowlight_restoration.multidegradation_visibility import image_delta_metrics


def _measure(record: dict[str, object]) -> dict[str, object]:
    full, full_sha = generate_record(record)
    output_path = ROOT / str(record["output_path"])
    if not output_path.is_file() or sha256(output_path) != full_sha:
        raise RuntimeError(f"isolated anchor does not reproduce: {record['record_id']}")
    clean, clean_sha = generate_record(record, application_order=())
    name = str(record["degradations"][0])
    return {
        "record_id": str(record["record_id"]),
        "degradation": name,
        "level": int(record["levels"][name]),
        "full_sha256": full_sha,
        "clean_sha256": clean_sha,
        "metrics": image_delta_metrics(full, clean),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/qwen_multidegradation_visibility_anchor_v1.json",
    )
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    args = parser.parse_args()
    if args.workers <= 0:
        raise SystemExit("workers must be positive")
    config_path = args.config.resolve()
    config_path.relative_to(ROOT.resolve())
    config = json.loads(config_path.read_text())
    manifest_path = (ROOT / str(config["manifest"])).resolve()
    manifest_path.relative_to(ROOT.resolve())
    if sha256(manifest_path) != config["manifest_sha256"]:
        raise RuntimeError("visibility anchor manifest changed")
    cohort = config["cohort"]
    records = [
        record
        for record in read_jsonl(manifest_path)
        if record["split"] == cohort["split"]
        and int(record["composition_size"]) == int(cohort["composition_size"])
    ]
    if len(records) != int(cohort["expected_records"]):
        raise RuntimeError("visibility anchor cohort cardinality changed")
    rows: list[dict[str, object]] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        for index, row in enumerate(executor.map(_measure, records), start=1):
            rows.append(row)
            if index % 100 == 0 or index == len(records):
                print(f"QWEN_VISIBILITY_ANCHOR_PROGRESS {index}/{len(records)}", flush=True)
    grouped: dict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["degradation"]), int(row["level"]))].append(row)
    expected = int(cohort["expected_per_type_level"])
    quantiles = [float(value) for value in config["quantiles"]]
    metric_names = tuple(rows[0]["metrics"].keys())
    anchors: dict[str, object] = {}
    for (name, level), group in sorted(grouped.items()):
        if len(group) != expected:
            raise RuntimeError(f"unbalanced visibility anchor: {name} L{level}")
        metric_summary: dict[str, object] = {}
        for metric in metric_names:
            values = np.asarray(
                [float(row["metrics"][metric]) for row in group], dtype=np.float64
            )
            metric_summary[metric] = {
                f"q{int(round(quantile * 100)):02d}": float(np.quantile(values, quantile))
                for quantile in quantiles
            }
            metric_summary[metric]["mean"] = float(values.mean())
        anchors[f"{name}:L{level}"] = {
            "records": len(group),
            "metrics": metric_summary,
        }
    if len(anchors) != 20:
        raise RuntimeError("visibility anchor must cover five types by four levels")
    output = {
        "schema_version": 1,
        "status": "pass",
        "config": config_path.relative_to(ROOT).as_posix(),
        "config_sha256": sha256(config_path),
        "manifest_sha256": config["manifest_sha256"],
        "records": len(rows),
        "anchors": anchors,
    }
    output_path = (ROOT / str(config["output"])).resolve()
    output_path.relative_to(ROOT.resolve())
    write_new_or_verify(
        output_path,
        (json.dumps(output, indent=2, sort_keys=True) + "\n").encode(),
    )
    print(
        "QWEN_VISIBILITY_ANCHOR_PASS "
        + json.dumps(
            {
                "records": len(rows),
                "anchors": len(anchors),
                "output": output_path.relative_to(ROOT).as_posix(),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
