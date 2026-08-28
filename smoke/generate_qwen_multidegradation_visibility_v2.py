#!/usr/bin/env python3
"""Generate/resume v2 images and verify every render against visibility evidence."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Mapping

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lowlight_restoration.multidegradation_data import (
    read_jsonl,
    generate_record,
    sha256,
    write_new_or_verify,
)
from smoke.build_qwen_multidegradation_visibility_v2 import (
    load_production_contract,
    validate_visibility_records,
)


def expected_visibility_hash(record: Mapping[str, object]) -> str | None:
    composition_size = int(record["composition_size"])
    evidence = record.get("visibility_evidence")
    if not isinstance(evidence, list) or len(evidence) != composition_size:
        raise RuntimeError("v2 record visibility evidence is incomplete")
    if composition_size == 0:
        return None
    hashes = {str(component["full_sha256"]) for component in evidence}
    if len(hashes) != 1:
        raise RuntimeError("v2 visibility evidence has inconsistent full hashes")
    expected = hashes.pop()
    if len(expected) != 64:
        raise RuntimeError("v2 visibility full hash is malformed")
    return expected


def render_verified_png(record: Mapping[str, object]) -> tuple[bytes, dict[str, object]]:
    image, rendered_sha = generate_record(record)
    expected_sha = expected_visibility_hash(record)
    if expected_sha is not None and rendered_sha != expected_sha:
        raise RuntimeError(
            f"final render differs from visibility evidence: {record['record_id']}"
        )
    success, encoded = cv2.imencode(".png", image)
    if not success:
        raise RuntimeError(f"failed to encode {record['record_id']}")
    payload = encoded.tobytes()
    if hashlib.sha256(payload).hexdigest() != rendered_sha:
        raise RuntimeError("PNG digest changed between deterministic encodes")
    return payload, {
        "record_id": record["record_id"],
        "output_path": record["output_path"],
        "output_sha256": rendered_sha,
        "visibility_full_sha256": expected_sha,
        "width": int(image.shape[1]),
        "height": int(image.shape[0]),
    }


def generate_or_verify(
    payload: tuple[dict[str, object], str],
) -> dict[str, object]:
    record, manifest_sha = payload
    evidence_path = (
        ROOT
        / "data/qwen_multidegradation_pilot_v2/.generation"
        / str(record["split"])
        / f"{record['record_id']}.json"
    )
    output_path = ROOT / str(record["output_path"])
    record_sha = hashlib.sha256(
        json.dumps(record, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()
    expected_sha = expected_visibility_hash(record)
    if evidence_path.is_file():
        evidence = json.loads(evidence_path.read_text())
        if (
            evidence.get("record_id") != record["record_id"]
            or evidence.get("manifest_sha256") != manifest_sha
            or evidence.get("manifest_record_sha256") != record_sha
            or evidence.get("visibility_full_sha256") != expected_sha
        ):
            raise RuntimeError(f"generation evidence contract mismatch: {evidence_path}")
        if (
            not output_path.is_file()
            or sha256(output_path) != evidence.get("output_sha256")
        ):
            raise RuntimeError(f"generated output changed: {output_path}")
        return {**evidence, "reused": True}
    try:
        png, result = render_verified_png(record)
        write_new_or_verify(output_path, png)
    except Exception as exc:
        raise RuntimeError(f"generation failed for {record['record_id']}: {exc}") from exc
    result["manifest_sha256"] = manifest_sha
    result["manifest_record_sha256"] = record_sha
    write_new_or_verify(
        evidence_path,
        (json.dumps(result, indent=2, sort_keys=True) + "\n").encode(),
    )
    return {**result, "reused": False}


def load_frozen_manifest(config_path: Path) -> tuple[list[dict[str, object]], str]:
    config = load_production_contract(config_path)
    manifest = (ROOT / str(config["manifest"])).resolve()
    metadata_path = (ROOT / str(config["metadata"])).resolve()
    manifest.relative_to(ROOT.resolve())
    metadata_path.relative_to(ROOT.resolve())
    metadata = json.loads(metadata_path.read_text())
    manifest_sha = sha256(manifest)
    if (
        metadata.get("status") != "frozen_manifest_only_images_not_implied_generated"
        or metadata.get("config_sha256") != sha256(config_path.resolve())
        or metadata.get("manifest_sha256") != manifest_sha
    ):
        raise RuntimeError("v2 manifest metadata contract changed")
    records = read_jsonl(manifest)
    validate_visibility_records(records, policy=config["visibility_policy"])
    return records, manifest_sha


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT
        / "configs/qwen_multidegradation_visibility_v2_production_correction1.json",
    )
    parser.add_argument("--workers", type=int, default=min(64, os.cpu_count() or 1))
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.workers <= 0 or args.limit is not None and args.limit <= 0:
        raise SystemExit("workers and limit must be positive")
    records, manifest_sha = load_frozen_manifest(args.config.resolve())
    if args.limit is not None:
        records = records[: args.limit]
    started = time.monotonic()
    results: list[dict[str, object]] = []
    inputs = [(record, manifest_sha) for record in records]
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        for index, result in enumerate(
            executor.map(generate_or_verify, inputs, chunksize=4),
            start=1,
        ):
            results.append(result)
            if index % 100 == 0 or index == len(inputs):
                print(
                    f"QWEN_VISIBILITY_V2_GENERATION_PROGRESS {index}/{len(inputs)}",
                    flush=True,
                )
    elapsed = time.monotonic() - started
    reused = sum(bool(result["reused"]) for result in results)
    summary = {
        "schema_version": 1,
        "status": "pass" if len(records) == 12000 else "bounded_partial_pass",
        "config": args.config.resolve().relative_to(ROOT).as_posix(),
        "config_sha256": sha256(args.config.resolve()),
        "manifest": "manifests/qwen_multidegradation_pilot_v2.jsonl",
        "manifest_sha256": manifest_sha,
        "requested_records": len(records),
        "completed_records": len(results),
        "reused_records": reused,
        "generated_records": len(results) - reused,
        "visibility_hash_verified_records": sum(
            result["visibility_full_sha256"] is not None for result in results
        ),
        "elapsed_seconds": elapsed,
        "workers": args.workers,
        "split_counts": dict(
            Counter(str(result["output_path"]).split("/")[2] for result in results)
        ),
    }
    if len(records) == 12000:
        result_path = ROOT / "reports/qwen_multidegradation_generation_result_v2.json"
        write_new_or_verify(
            result_path,
            (json.dumps(summary, indent=2, sort_keys=True) + "\n").encode(),
        )
    print(
        "QWEN_VISIBILITY_V2_GENERATION_PASS " + json.dumps(summary, sort_keys=True),
        flush=True,
    )


if __name__ == "__main__":
    main()
