#!/usr/bin/env python3
"""Generate/resume the immutable 120k-image scaled Qwen dataset."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import sys
import time
from collections import Counter
from collections import defaultdict
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lowlight_restoration.multidegradation_data import (
    read_jsonl,
    sha256,
    write_generated_png,
    write_new_or_verify,
)


def workspace_path(value: object) -> Path:
    path = (ROOT / str(value)).resolve()
    path.relative_to(ROOT.resolve())
    return path


def load_contract(path: Path) -> tuple[dict[str, object], str]:
    raw = path.read_bytes()
    config = json.loads(raw)
    if config.get("status") not in {
        "human_approved_extreme_blur_v6_100k_contract",
        "advisor_directed_mild_blur_noise_v7_100k_contract",
    }:
        raise ValueError("scaled data generation contract is not approved")
    return config, hashlib.sha256(raw).hexdigest()


def generate_or_verify(
    record: dict[str, object], evidence_root_text: str
) -> dict[str, object]:
    evidence_root = workspace_path(evidence_root_text)
    evidence_path = (
        evidence_root / str(record["split"]) / f"{record['record_id']}.json"
    )
    output_path = workspace_path(record["output_path"])
    if evidence_path.is_file():
        evidence = json.loads(evidence_path.read_text())
        if (
            evidence.get("record_id") != record["record_id"]
            or not output_path.is_file()
            or sha256(output_path) != evidence.get("output_sha256")
        ):
            raise RuntimeError(f"scaled generation evidence changed: {evidence_path}")
        return {**evidence, "reused": True}
    result = write_generated_png(record)
    result["manifest_record_sha256"] = hashlib.sha256(
        json.dumps(record, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()
    write_new_or_verify(
        evidence_path,
        (json.dumps(result, indent=2, sort_keys=True) + "\n").encode(),
    )
    return {**result, "reused": False}


def generate_chunk(
    payload: tuple[list[dict[str, object]], str]
) -> list[dict[str, object]]:
    records, evidence_root = payload
    return [generate_or_verify(record, evidence_root) for record in records]


def chunks(
    records: Sequence[dict[str, object]], size: int
) -> list[list[dict[str, object]]]:
    return [list(records[offset : offset + size]) for offset in range(0, len(records), size)]


def source_grouped_chunks(
    records: Sequence[dict[str, object]], size: int
) -> list[list[dict[str, object]]]:
    """Keep records from one source together so workers reuse decoded inputs."""

    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for record in records:
        grouped[(str(record["source_path"]), str(record["source_sha256"]))].append(
            record
        )
    work: list[list[dict[str, object]]] = []
    for key in sorted(grouped):
        work.extend(chunks(grouped[key], size))
    if sum(map(len, work)) != len(records):
        raise RuntimeError("source-grouped generation lost records")
    return work


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/qwen_multidegradation_100k_data_v1.json",
    )
    parser.add_argument("--workers", type=int, default=min(48, os.cpu_count() or 1))
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument(
        "--group-by-source",
        action="store_true",
        help="group records by immutable source to avoid repeated decode/hash work",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="generate one example per composition/combination/level pattern",
    )
    args = parser.parse_args()
    if args.workers <= 0 or args.chunk_size <= 0:
        raise SystemExit("workers and chunk size must be positive")
    config, config_sha = load_contract(args.config.resolve())
    manifest = workspace_path(config["manifest"])
    metadata = json.loads(workspace_path(config["metadata"]).read_text())
    if (
        metadata.get("data_contract_sha256") != config_sha
        or metadata.get("manifest_sha256") != sha256(manifest)
    ):
        raise RuntimeError("scaled manifest is not bound to its data contract")
    records = read_jsonl(manifest)
    if len(records) != 120000:
        raise RuntimeError("scaled manifest must contain exactly 120,000 images")
    if args.limit is not None and args.smoke:
        raise SystemExit("limit and smoke are mutually exclusive")
    if args.smoke:
        selected: dict[tuple[object, ...], dict[str, object]] = {}
        for record in records:
            if record["split"] != "train":
                continue
            names = tuple(record["degradations"])
            level_signature = tuple(record["levels"].values())
            key = (record["composition_size"], names, level_signature)
            selected.setdefault(key, record)
        records = list(selected.values())
    elif args.limit is not None:
        if args.limit <= 0:
            raise SystemExit("limit must be positive")
        records = records[: args.limit]
    record_chunks = (
        source_grouped_chunks(records, args.chunk_size)
        if args.group_by_source
        else chunks(records, args.chunk_size)
    )
    work = [
        (chunk, str(config["generation_evidence_root"]))
        for chunk in record_chunks
    ]
    started = time.monotonic()
    results: list[dict[str, object]] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        completed = 0
        for chunk_results in executor.map(generate_chunk, work):
            results.extend(chunk_results)
            completed += len(chunk_results)
            if completed % 1000 == 0 or completed == len(records):
                print(
                    f"QWEN_100K_GENERATION_PROGRESS {completed}/{len(records)}",
                    flush=True,
                )
    summary = {
        "schema_version": 1,
        "status": "pass" if len(records) == 120000 else "bounded_partial_pass",
        "data_contract": args.config.resolve().relative_to(ROOT).as_posix(),
        "data_contract_sha256": config_sha,
        "manifest": config["manifest"],
        "manifest_sha256": sha256(manifest),
        "requested_records": len(records),
        "completed_records": len(results),
        "reused_records": sum(bool(result["reused"]) for result in results),
        "generated_records": sum(not bool(result["reused"]) for result in results),
        "split_counts": dict(
            Counter(str(record["split"]) for record in records)
        ),
        "workers": args.workers,
        "chunk_size": args.chunk_size,
        "source_grouped": args.group_by_source,
        "elapsed_seconds": time.monotonic() - started,
        "protected_test_inference_performed": False,
        "foundir_used": False,
    }
    if len(records) == 120000:
        write_new_or_verify(
            workspace_path(config["generation_result"]),
            (json.dumps(summary, indent=2, sort_keys=True) + "\n").encode(),
        )
    print("QWEN_100K_GENERATION_PASS " + json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
