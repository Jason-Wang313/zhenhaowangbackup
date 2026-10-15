#!/usr/bin/env python3
"""Freeze a zero-training Blur cross-compatibility audit of the paired specialist."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "manifests/foundir_final_eleven_all_in_one_development_rehearsal_v3.csv"
SOURCE_SHA256 = "ff05a36c9e9a101e9227be626dc97f0d867c17ac05cafdfc8b94b8d5c26e1386"
OUTPUT = ROOT / "manifests/foundir_final_eleven_blur_via_blur_noise_audit_v4.csv"


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    if sha256(SOURCE) != SOURCE_SHA256:
        raise RuntimeError("frozen v3 manifest identity changed")
    with SOURCE.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]
    changed = 0
    for row in rows:
        row["blur_specialist_audit"] = ""
        if row["category"] == "01Blur" and row["routing_status"] == "correct":
            row["final_route"] = "blur_via_blur_noise_audit"
            row["expert_order_json"] = '["blur_noise"]'
            row["blur_specialist_audit"] = "paired_blur_noise_restormer_zero_training"
            changed += 1
    if changed != 120:
        raise RuntimeError(f"Blur audit cardinality changed: {changed}")
    temporary = OUTPUT.with_suffix(".csv.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields + ["blur_specialist_audit"])
        writer.writeheader(); writer.writerows(rows)
    temporary.replace(OUTPUT)
    payload = {
        "schema_version": 1,
        "status": "frozen_blur_via_blur_noise_cross_compatibility_audit_v4",
        "source_manifest_sha256": SOURCE_SHA256,
        "manifest_sha256": sha256(OUTPUT),
        "records": len(rows),
        "blur_records_using_paired_specialist": changed,
        "blur_abstentions_retaining_identity": 8,
        "model_weight_updates": False,
        "official_or_sealed_test_accessed": False,
        "not_an_independent_lockbox": True,
    }
    OUTPUT.with_suffix(".json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
