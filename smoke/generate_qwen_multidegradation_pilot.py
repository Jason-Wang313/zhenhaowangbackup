#!/usr/bin/env python3
"""Generate or resume all images in the frozen Qwen pilot manifest."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lowlight_restoration.multidegradation_data import (
    read_jsonl,
    sha256,
    write_generated_png,
    write_new_or_verify,
)


def generate_or_verify(record: dict[str, object]) -> dict[str, object]:
    evidence_path = (
        ROOT
        / "data/qwen_multidegradation_pilot_v1/.generation"
        / str(record["split"])
        / f"{record['record_id']}.json"
    )
    output_path = ROOT / str(record["output_path"])
    if evidence_path.is_file():
        evidence = json.loads(evidence_path.read_text())
        if evidence.get("record_id") != record["record_id"]:
            raise RuntimeError(f"evidence record mismatch: {evidence_path}")
        if not output_path.is_file() or sha256(output_path) != evidence.get("output_sha256"):
            raise RuntimeError(f"generated output changed: {output_path}")
        return {**evidence, "reused": True}
    try:
        result = write_generated_png(record)
    except Exception as exc:
        raise RuntimeError(f"generation failed for {record['record_id']}: {exc}") from exc
    result["manifest_record_sha256"] = __import__("hashlib").sha256(
        json.dumps(record, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()
    write_new_or_verify(
        evidence_path,
        (json.dumps(result, indent=2, sort_keys=True) + "\n").encode(),
    )
    return {**result, "reused": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=min(24, os.cpu_count() or 1))
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.workers <= 0 or args.limit is not None and args.limit <= 0:
        raise SystemExit("workers and limit must be positive")
    manifest = ROOT / "manifests/qwen_multidegradation_pilot_v1.jsonl"
    records = read_jsonl(manifest)
    if args.limit is not None:
        records = records[: args.limit]
    started = time.monotonic()
    results: list[dict[str, object]] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(generate_or_verify, record) for record in records]
        for index, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            results.append(future.result())
            if index % 100 == 0 or index == len(futures):
                print(
                    f"QWEN_MULTIDEGRADATION_GENERATION_PROGRESS {index}/{len(futures)}",
                    flush=True,
                )
    elapsed = time.monotonic() - started
    reused = sum(bool(result["reused"]) for result in results)
    summary = {
        "schema_version": 1,
        "status": "pass" if len(records) == 12000 else "bounded_partial_pass",
        "manifest": manifest.relative_to(ROOT).as_posix(),
        "manifest_sha256": sha256(manifest),
        "requested_records": len(records),
        "completed_records": len(results),
        "reused_records": reused,
        "generated_records": len(results) - reused,
        "elapsed_seconds": elapsed,
        "workers": args.workers,
        "split_counts": dict(Counter(result["output_path"].split("/")[2] for result in results)),
    }
    if len(records) == 12000:
        result_path = ROOT / "reports/qwen_multidegradation_generation_result_v1.json"
        write_new_or_verify(
            result_path,
            (json.dumps(summary, indent=2, sort_keys=True) + "\n").encode(),
        )
    print("QWEN_MULTIDEGRADATION_GENERATION_PASS " + json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
