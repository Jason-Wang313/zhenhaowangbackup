#!/usr/bin/env python3
"""Freeze 200 unused, source-disjoint LSDIR sources for v2b validation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lowlight_restoration.multidegradation_data import sha256, stable_digest, write_new_or_verify
from smoke.check_low_light_source_pool import hamming, perceptual_hash


OLD_SOURCE_MANIFEST = ROOT / "manifests/low_light_source_pool_2000.json"
OLD_SOURCE_MANIFEST_SHA256 = "031a6e43cf46e8bf2363be7ca6c67c533ffcf7a44241a5f605a542daac1729d5"
OLD_V2_MANIFEST = ROOT / "manifests/qwen_multidegradation_pilot_v2.jsonl"
OLD_V2_MANIFEST_SHA256 = "0ec9b367d3b795b01f8a2185b0ff6bfda8a8d21b29926929371bac4bf013ce7e"
DEPTH_REPORT = ROOT / "reports/qwen_multidegradation_visibility_v2b_depth.json"
DEPTH_REPORT_SHA256 = "19321433ea07dfbca936b93dd7a1f69ede94319693c58edbe416dd5b5acf546f"
SOURCE_ROOT = ROOT / "data/clean_pool/lsdir"
OUTPUT = ROOT / "manifests/qwen_multidegradation_visibility_v2b_sources.json"
SEED = 2026082833
EXPECTED_SOURCES = 200
EXPECTED_DEPTH_SOURCES = 20
MINIMUM_WIDTH = 512
MINIMUM_HEIGHT = 512
MAXIMUM_PHASH_DISTANCE_FOR_DUPLICATE = 4


def old_records() -> list[dict[str, object]]:
    if sha256(OLD_SOURCE_MANIFEST) != OLD_SOURCE_MANIFEST_SHA256:
        raise RuntimeError("old source manifest changed")
    payload = json.loads(OLD_SOURCE_MANIFEST.read_text())
    records = payload.get("records")
    if not isinstance(records, list) or len(records) != 2000:
        raise RuntimeError("old source manifest cardinality changed")
    return records


def structural_old_split_counts() -> dict[str, int]:
    if sha256(OLD_V2_MANIFEST) != OLD_V2_MANIFEST_SHA256:
        raise RuntimeError("old v2 manifest changed")
    roles: dict[str, str] = {}
    with OLD_V2_MANIFEST.open() as handle:
        for line in handle:
            record = json.loads(line)
            source_id = str(record["source_id"])
            split = str(record["split"])
            previous = roles.setdefault(source_id, split)
            if previous != split:
                raise RuntimeError("old v2 manifest has source leakage")
    return {
        split: sum(role == split for role in roles.values())
        for split in ("train", "validation", "test")
    }


def depth_lookup() -> dict[str, dict[str, object]]:
    if sha256(DEPTH_REPORT) != DEPTH_REPORT_SHA256:
        raise RuntimeError("v2b depth report changed")
    payload = json.loads(DEPTH_REPORT.read_text())
    outputs = payload.get("outputs")
    if (
        payload.get("status") != "pass"
        or not isinstance(outputs, list)
        or len(outputs) != EXPECTED_DEPTH_SOURCES
    ):
        raise RuntimeError("v2b depth report is incomplete")
    result: dict[str, dict[str, object]] = {}
    for row in outputs:
        path = ROOT / str(row["depth_path"])
        if not path.is_file() or sha256(path) != row["depth_sha256"]:
            raise RuntimeError("v2b depth output failed its frozen hash")
        depth = loadmat(path).get("data_obj")
        if depth is None or depth.ndim != 2 or float(depth.max()) <= 0:
            raise RuntimeError("v2b depth output is malformed")
        result[str(row["source_path"])] = dict(row)
    return result


def main() -> None:
    old = old_records()
    split_counts = structural_old_split_counts()
    depth = depth_lookup()
    used_paths = {(ROOT / str(record["path"])).resolve() for record in old}
    old_sha = {str(record["sha256"]) for record in old}
    old_phash = [int(str(record["average_hash"]), 16) for record in old]
    candidates = [
        path.resolve()
        for path in sorted(SOURCE_ROOT.iterdir())
        if path.is_file() and path.resolve() not in used_paths
    ]
    if len(candidates) != EXPECTED_SOURCES:
        raise RuntimeError(
            f"expected {EXPECTED_SOURCES} unused sources, got {len(candidates)}"
        )
    observed_sha: set[str] = set()
    observed_phash: list[int] = []
    records: list[dict[str, object]] = []
    minimum_old_distance = 65
    minimum_internal_distance = 65
    for path in candidates:
        with Image.open(path) as image:
            image.load()
            width, height = image.size
            mode = image.mode
            if mode != "RGB" or width < MINIMUM_WIDTH or height < MINIMUM_HEIGHT:
                raise RuntimeError(f"invalid fresh source: {path}")
            phash = perceptual_hash(image)
        digest = sha256(path)
        if digest in old_sha or digest in observed_sha:
            raise RuntimeError(f"exact source overlap: {path}")
        old_distance = min(hamming(phash, value) for value in old_phash)
        internal_distance = min(
            (hamming(phash, value) for value in observed_phash), default=65
        )
        if min(old_distance, internal_distance) <= MAXIMUM_PHASH_DISTANCE_FOR_DUPLICATE:
            raise RuntimeError(f"perceptual source overlap: {path}")
        minimum_old_distance = min(minimum_old_distance, old_distance)
        minimum_internal_distance = min(minimum_internal_distance, internal_distance)
        relative = path.relative_to(ROOT).as_posix()
        record: dict[str, object] = {
            "source_id": f"lsdir_{path.stem.lower()}",
            "dataset": "lsdir",
            "path": relative,
            "sha256": digest,
            "bytes": path.stat().st_size,
            "width": width,
            "height": height,
            "mode": mode,
            "average_hash": f"{phash:016x}",
            "source": "LSDIR official training shard-00",
            "license": "LSDIR dataset terms",
            "license_status": "inherited_official_training_source",
        }
        if relative in depth:
            row = depth[relative]
            record.update(
                {
                    "depth_path": row["depth_path"],
                    "depth_sha256": row["depth_sha256"],
                    "depth_model_revision": "5426e4f0f36572d16453bbda7a8389317b1bef99",
                    "depth_orientation": "matched_to_official_mio_depth",
                }
            )
        records.append(record)
        observed_sha.add(digest)
        observed_phash.append(phash)
    if sum("depth_path" in record for record in records) != EXPECTED_DEPTH_SOURCES:
        raise RuntimeError("fresh source pool has the wrong depth-qualified count")
    records.sort(
        key=lambda record: stable_digest(SEED, "v2b_source", record["source_id"])
    )
    payload = {
        "schema_version": 1,
        "status": "frozen_before_model_correction_or_inference",
        "selection_seed": SEED,
        "old_source_manifest": OLD_SOURCE_MANIFEST.relative_to(ROOT).as_posix(),
        "old_source_manifest_sha256": OLD_SOURCE_MANIFEST_SHA256,
        "old_v2_manifest": OLD_V2_MANIFEST.relative_to(ROOT).as_posix(),
        "old_v2_manifest_sha256": OLD_V2_MANIFEST_SHA256,
        "depth_report": DEPTH_REPORT.relative_to(ROOT).as_posix(),
        "depth_report_sha256": DEPTH_REPORT_SHA256,
        "source_count": len(records),
        "depth_qualified_source_count": sum(
            "depth_path" in record for record in records
        ),
        "excluded_old_source_count": len(old),
        "excluded_old_v2_source_roles": split_counts,
        "exact_source_overlap_count": 0,
        "perceptual_source_overlap_count_at_hamming_le_4": 0,
        "minimum_old_source_phash_distance": minimum_old_distance,
        "minimum_internal_phash_distance": minimum_internal_distance,
        "records": records,
    }
    digest = write_new_or_verify(
        OUTPUT,
        (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode(),
    )
    print(
        "QWEN_VISIBILITY_V2B_SOURCES_PASS "
        + json.dumps(
            {
                "sources": len(records),
                "depth_sources": payload["depth_qualified_source_count"],
                "minimum_old_phash_distance": minimum_old_distance,
                "minimum_internal_phash_distance": minimum_internal_distance,
                "manifest_sha256": digest,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
