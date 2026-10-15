#!/usr/bin/env python3
"""Freeze/verify the full-order development contract without test access."""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/foundir_vlm_energy_full_order_data_v1.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def canonical(payload: object) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()


def write_once(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise RuntimeError(f"refusing to overwrite differing artifact: {path}")
        return
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def select(config: dict[str, Any]) -> list[dict[str, Any]]:
    parent = ROOT / config["parent_manifest"]
    if sha256(parent) != config["parent_manifest_sha256"]:
        raise RuntimeError("parent manifest identity changed")
    allowed_splits = set(config["allowed_splits"])
    allowed_sizes = set(map(int, config["allowed_composition_sizes"]))
    rows = []
    for row in load_jsonl(parent):
        split, size = str(row["split"]), int(row["composition_size"])
        if split not in allowed_splits or size not in allowed_sizes:
            continue
        sidecar = ROOT / config["rendered_root"] / ".generation" / split / f"{row['record_id']}.json"
        evidence = json.loads(sidecar.read_text())
        if evidence["record_id"] != row["record_id"] or evidence["output_path"] != row["output_path"]:
            raise RuntimeError(f"generation sidecar mismatch: {sidecar}")
        rows.append({
            **row,
            "degraded_sha256": evidence["output_sha256"],
            "target_application_order": [],
            "target_rule": config["target_rule"],
        })
    return sorted(rows, key=lambda row: (str(row["split"]), str(row["record_id"])))


def validate(config: dict[str, Any], rows: list[dict[str, Any]], *, verify_images: bool) -> dict[str, Any]:
    counts = Counter((str(row["split"]), int(row["composition_size"])) for row in rows)
    expected_counts = {
        ("train", 1): 30000, ("train", 2): 30000, ("train", 3): 30000,
        ("validation", 1): 3000, ("validation", 2): 3000, ("validation", 3): 3000,
    }
    if counts != expected_counts:
        raise RuntimeError(f"full-order record counts changed: {counts}")
    sources: dict[str, set[str]] = defaultdict(set)
    combo_counts: Counter[tuple[str, int, tuple[str, ...]]] = Counter()
    degradations = tuple(map(str, config["required_degradations"]))
    for row in rows:
        split = str(row["split"])
        size = int(row["composition_size"])
        names = tuple(sorted(map(str, row["degradations"])))
        if len(names) != size or not set(names).issubset(degradations):
            raise RuntimeError(f"bad degradation descriptor: {row['record_id']}")
        sources[split].add(str(row["source_id"]))
        combo_counts[(split, size, names)] += 1
        image = ROOT / str(row["output_path"])
        if not image.is_file():
            raise RuntimeError(f"missing rendered image: {image}")
        if verify_images and sha256(image) != row["degraded_sha256"]:
            raise RuntimeError(f"rendered image hash mismatch: {image}")
    if sources["train"] & sources["validation"]:
        raise RuntimeError("train/validation source leakage")
    expected_sources = {
        "train": int(config["expected"]["train_sources"]),
        "validation": int(config["expected"]["validation_sources"]),
    }
    if {key: len(value) for key, value in sources.items()} != expected_sources:
        raise RuntimeError("source cardinality changed")
    for split, per_single, per_higher in (("train", 6000, 3000), ("validation", 600, 300)):
        for size, per_combo in ((1, per_single), (2, per_higher), (3, per_higher)):
            expected_combos = {tuple(sorted(value)) for value in combinations(degradations, size)}
            actual = {names for s, z, names in combo_counts if s == split and z == size}
            if actual != expected_combos or any(combo_counts[(split, size, names)] != per_combo for names in actual):
                raise RuntimeError(f"combination balance changed for {split} order {size}")
    return {
        "records": len(rows),
        "split_order_records": {f"{split}_{size}": count for (split, size), count in sorted(counts.items())},
        "split_sources": expected_sources,
        "combination_counts": {"single": 5, "double": 10, "triple": 10},
        "image_hashes_verified": verify_images,
        "parent_test_records": 0,
        "official_foundir_test_accessed": False,
        "sealed_internal_accessed": False,
    }


def main() -> None:
    config = json.loads(CONFIG_PATH.read_text())
    rows = select(config)
    checks = validate(config, rows, verify_images=False)
    payload = b"".join(canonical(row) for row in rows)
    write_once(ROOT / config["output_manifest"], payload)
    result = {
        "schema_version": 1,
        "status": "foundir_vlm_energy_full_order_data_v1_ready",
        "config_sha256": sha256(CONFIG_PATH),
        "parent_manifest_sha256": config["parent_manifest_sha256"],
        "output_manifest": config["output_manifest"],
        "output_manifest_sha256": hashlib.sha256(payload).hexdigest(),
        "checks": checks,
        "training_launched": False,
        "official_foundir_test_accessed": False,
        "sealed_internal_accessed": False,
    }
    write_once(ROOT / config["output_metadata"], (json.dumps(result, indent=2, sort_keys=True) + "\n").encode())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
