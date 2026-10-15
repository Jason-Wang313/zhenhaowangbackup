#!/usr/bin/env python3
"""Freeze/verify the 60k-train, 6k-validation lower-order EBT data contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/foundir_compositional_ebt_scaled_data_v1.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_line(payload: object) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()


def write_or_verify(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise RuntimeError(f"refusing to overwrite differing artifact: {path}")
        return
    path.write_bytes(payload)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def selected_rows(config: dict[str, Any]) -> list[dict[str, Any]]:
    parent = ROOT / config["parent_manifest"]
    if sha256(parent) != config["parent_manifest_sha256"]:
        raise RuntimeError("parent manifest hash mismatch")
    rows = []
    for record in load_jsonl(parent):
        split = str(record["split"])
        size = int(record["composition_size"])
        if split not in {"train", "validation"} or size not in {1, 2}:
            continue
        sidecar = (
            ROOT
            / config["rendered_root"]
            / ".generation"
            / split
            / f"{record['record_id']}.json"
        )
        evidence = json.loads(sidecar.read_text())
        if evidence["record_id"] != record["record_id"]:
            raise RuntimeError(f"sidecar identity mismatch: {sidecar}")
        degraded = ROOT / str(record["output_path"])
        if evidence["output_path"] != record["output_path"]:
            raise RuntimeError(f"sidecar output mismatch: {sidecar}")
        rows.append(
            {
                **record,
                "degraded_sha256": evidence["output_sha256"],
                "target_application_order": [],
                "target_rule": config["target_rule"],
            }
        )
    return sorted(rows, key=lambda row: (str(row["split"]), str(row["record_id"])))


def validate(config: dict[str, Any], rows: list[dict[str, Any]], *, full_hash: bool) -> dict[str, Any]:
    degradations = tuple(config["required_degradations"])
    required_pairs = {tuple(sorted(pair)) for pair in combinations(degradations, 2)}
    counts: Counter[tuple[str, int, tuple[str, ...]]] = Counter()
    sources: dict[str, set[str]] = defaultdict(set)
    unique_sources: dict[tuple[str, str], tuple[Path, str]] = {}
    for row in rows:
        split = str(row["split"])
        size = int(row["composition_size"])
        names = tuple(sorted(map(str, row["degradations"])))
        if size != len(names) or size not in {1, 2}:
            raise RuntimeError("non-lower-order row entered scaled contract")
        counts[(split, size, names)] += 1
        source_id = str(row["source_id"])
        sources[split].add(source_id)
        unique_sources[(split, source_id)] = (
            ROOT / str(row["source_path"]),
            str(row["source_sha256"]),
        )
        degraded = ROOT / str(row["output_path"])
        if not degraded.is_file():
            raise RuntimeError(f"missing degraded image: {degraded}")
        if full_hash and sha256(degraded) != row["degraded_sha256"]:
            raise RuntimeError(f"degraded image hash mismatch: {degraded}")

    if sources["train"] & sources["validation"]:
        raise RuntimeError("train/validation source leakage")
    expected_totals = {"train": 60000, "validation": 6000}
    actual_totals = Counter(str(row["split"]) for row in rows)
    if dict(actual_totals) != expected_totals:
        raise RuntimeError(f"record totals changed: {actual_totals}")
    if {split: len(ids) for split, ids in sources.items()} != {"train": 1600, "validation": 200}:
        raise RuntimeError("source cardinality changed")
    for split, single_count, pair_count in (("train", 6000, 3000), ("validation", 600, 300)):
        singles = {names for s, size, names in counts if s == split and size == 1}
        pairs = {names for s, size, names in counts if s == split and size == 2}
        if singles != {(name,) for name in degradations} or pairs != required_pairs:
            raise RuntimeError(f"composition coverage changed: {split}")
        if any(counts[(split, 1, names)] != single_count for names in singles):
            raise RuntimeError(f"singleton balance changed: {split}")
        if any(counts[(split, 2, names)] != pair_count for names in pairs):
            raise RuntimeError(f"pair balance changed: {split}")
    if full_hash:
        for path, expected in unique_sources.values():
            if sha256(path) != expected:
                raise RuntimeError(f"source hash mismatch: {path}")
    return {
        "records": len(rows),
        "split_records": dict(actual_totals),
        "split_sources": {split: len(ids) for split, ids in sources.items()},
        "pair_coverage": {split: 10 for split in ("train", "validation")},
        "full_hash_verification": full_hash,
        "triple_records": 0,
        "parent_test_records": 0,
    }


def build() -> dict[str, Any]:
    config = json.loads(CONFIG_PATH.read_text())
    rows = selected_rows(config)
    checks = validate(config, rows, full_hash=True)
    manifest_payload = b"".join(canonical_line(row) for row in rows)
    manifest_path = ROOT / config["output_manifest"]
    write_or_verify(manifest_path, manifest_payload)
    metadata = {
        "schema_version": 1,
        "status": "foundir_compositional_ebt_scaled_data_v1_ready",
        "config_path": config["output_manifest"].replace("manifests/", "configs/").replace(".jsonl", ".json"),
        "config_sha256": sha256(CONFIG_PATH),
        "parent_manifest": config["parent_manifest"],
        "parent_manifest_sha256": config["parent_manifest_sha256"],
        "output_manifest": config["output_manifest"],
        "output_manifest_sha256": hashlib.sha256(manifest_payload).hexdigest(),
        "checks": checks,
        "gpu_training_launched": False,
        "gpu_hours": 0.0,
        "heldout_triple_images_accessed": False,
        "foundir_60_image_triple_accessed": False,
        "official_foundir_test_accessed": False,
        "sealed_internal_accessed": False
    }
    write_or_verify(
        ROOT / config["output_metadata"],
        (json.dumps(metadata, indent=2, sort_keys=True) + "\n").encode(),
    )
    return metadata


def check() -> dict[str, Any]:
    config = json.loads(CONFIG_PATH.read_text())
    metadata_path = ROOT / config["output_metadata"]
    metadata = json.loads(metadata_path.read_text())
    manifest = ROOT / metadata["output_manifest"]
    if sha256(CONFIG_PATH) != metadata["config_sha256"]:
        raise RuntimeError("config identity changed")
    if sha256(manifest) != metadata["output_manifest_sha256"]:
        raise RuntimeError("output manifest identity changed")
    checks = validate(config, load_jsonl(manifest), full_hash=False)
    if {key: value for key, value in checks.items() if key != "full_hash_verification"} != {
        key: value for key, value in metadata["checks"].items() if key != "full_hash_verification"
    }:
        raise RuntimeError("metadata checks changed")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    result = check() if args.check else build()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
