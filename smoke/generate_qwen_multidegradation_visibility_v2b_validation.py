#!/usr/bin/env python3
"""Generate and hash-verify every frozen source-disjoint v2b validation PNG."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Mapping

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import cv2

cv2.setNumThreads(1)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lowlight_restoration.multidegradation_data import (
    generate_record,
    read_jsonl,
    sha256,
    write_new_or_verify,
)
from smoke.build_qwen_multidegradation_visibility_v2b_validation import (
    DEFAULT_CONFIG,
    load_contract,
    load_fresh_sources,
    validate_records,
    workspace_path,
)

REPORT = ROOT / "reports/qwen_multidegradation_visibility_v2b_generation.json"


def expected_visibility_hash(record: Mapping[str, object]) -> str | None:
    evidence = record.get("visibility_evidence")
    if not isinstance(evidence, list) or len(evidence) != int(record["composition_size"]):
        raise RuntimeError("v2b record visibility evidence is incomplete")
    if not evidence:
        return None
    hashes = {str(component["full_sha256"]) for component in evidence}
    if len(hashes) != 1:
        raise RuntimeError("v2b visibility hashes disagree")
    return hashes.pop()


def generate_or_verify(
    payload: tuple[dict[str, object], str, str],
) -> dict[str, object]:
    record, manifest_sha, evidence_root = payload
    evidence_path = (
        workspace_path(evidence_root)
        / "validation"
        / f"{record['record_id']}.json"
    )
    output_path = workspace_path(record["output_path"])
    record_sha = hashlib.sha256(
        json.dumps(record, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()
    visibility_sha = expected_visibility_hash(record)
    if evidence_path.is_file():
        evidence = json.loads(evidence_path.read_text())
        if (
            evidence.get("record_id") != record["record_id"]
            or evidence.get("manifest_sha256") != manifest_sha
            or evidence.get("manifest_record_sha256") != record_sha
            or evidence.get("visibility_full_sha256") != visibility_sha
            or not output_path.is_file()
            or sha256(output_path) != evidence.get("output_sha256")
        ):
            raise RuntimeError(f"v2b existing generation evidence changed: {record['record_id']}")
        return {**evidence, "reused": True}
    image, rendered_sha = generate_record(record)
    if visibility_sha is not None and rendered_sha != visibility_sha:
        raise RuntimeError(f"v2b render differs from screening: {record['record_id']}")
    success, encoded = cv2.imencode(".png", image)
    if not success:
        raise RuntimeError(f"v2b PNG encoding failed: {record['record_id']}")
    png = encoded.tobytes()
    if hashlib.sha256(png).hexdigest() != rendered_sha:
        raise RuntimeError("v2b deterministic PNG hash changed")
    write_new_or_verify(output_path, png)
    evidence = {
        "schema_version": 1,
        "record_id": record["record_id"],
        "manifest_sha256": manifest_sha,
        "manifest_record_sha256": record_sha,
        "output_path": record["output_path"],
        "output_sha256": rendered_sha,
        "visibility_full_sha256": visibility_sha,
        "width": int(image.shape[1]),
        "height": int(image.shape[0]),
    }
    write_new_or_verify(
        evidence_path,
        (json.dumps(evidence, indent=2, sort_keys=True) + "\n").encode(),
    )
    return {**evidence, "reused": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--workers", type=int, default=min(48, os.cpu_count() or 1))
    args = parser.parse_args()
    if args.workers <= 0:
        raise SystemExit("workers must be positive")
    config_path = args.config.resolve()
    config = load_contract(config_path)
    metadata_path = workspace_path(config["metadata"])
    metadata = json.loads(metadata_path.read_text())
    manifest_path = workspace_path(config["manifest"])
    manifest_sha = sha256(manifest_path)
    if (
        metadata.get("status")
        != "frozen_validation_manifest_before_model_correction_or_inference"
        or metadata.get("config_sha256") != sha256(config_path)
        or metadata.get("manifest_sha256") != manifest_sha
        or metadata.get("model_checkpoint_accessed") is not False
        or metadata.get("protected_test_inference_performed") is not False
    ):
        raise RuntimeError("v2b validation manifest was not frozen before inference")
    records = read_jsonl(manifest_path)
    sources = load_fresh_sources(config)
    validation = validate_records(records, config=config, fresh_sources=sources)
    started = time.monotonic()
    inputs = [
        (record, manifest_sha, str(config["generation_evidence_root"]))
        for record in records
    ]
    results: list[dict[str, object]] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        for index, result in enumerate(
            executor.map(generate_or_verify, inputs, chunksize=4),
            start=1,
        ):
            results.append(result)
            if index % 120 == 0 or index == len(inputs):
                print(f"QWEN_VISIBILITY_V2B_GENERATION_PROGRESS {index}/1200", flush=True)
    summary = {
        "schema_version": 1,
        "status": "pass",
        "config": config_path.relative_to(ROOT).as_posix(),
        "config_sha256": sha256(config_path),
        "manifest": manifest_path.relative_to(ROOT).as_posix(),
        "manifest_sha256": manifest_sha,
        "metadata": metadata_path.relative_to(ROOT).as_posix(),
        "metadata_sha256": sha256(metadata_path),
        "completed_records": len(results),
        "generated_records": sum(not bool(row["reused"]) for row in results),
        "reused_records": sum(bool(row["reused"]) for row in results),
        "visibility_hash_verified_records": sum(
            row["visibility_full_sha256"] is not None for row in results
        ),
        "elapsed_seconds": time.monotonic() - started,
        "workers": args.workers,
        "validation": validation,
        "model_checkpoint_accessed": False,
        "protected_test_inference_performed": False,
    }
    report_sha = write_new_or_verify(
        REPORT,
        (json.dumps(summary, indent=2, sort_keys=True) + "\n").encode(),
    )
    print(
        "QWEN_VISIBILITY_V2B_GENERATION_PASS "
        + json.dumps(
            {
                "records": len(results),
                "manifest_sha256": manifest_sha,
                "report_sha256": report_sha,
                "elapsed_seconds": summary["elapsed_seconds"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
