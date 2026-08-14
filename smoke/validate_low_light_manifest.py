#!/usr/bin/env python3
"""Independent fail-closed quality gate for a low-light paired manifest."""
from __future__ import annotations

import argparse
import csv
import hashlib
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
INTERVALS = {
    "L1": (0.95, 1.00),
    "L2": (0.48, 0.52),
    "L3": (0.36, 0.40),
    "L4": (0.28, 0.32),
    "L5": (0.12, 0.16),
}
EXPECTED_LEVELS = set(INTERVALS)
WEIGHTS = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
REQUIRED_FIELDS = {
    "pair_id",
    "source_id",
    "source_dataset",
    "source_original_path",
    "source_license",
    "source_license_status",
    "clean_path",
    "degraded_path",
    "degradation_type",
    "project_level",
    "seed",
    "split",
    "clean_sha256",
    "degraded_sha256",
}


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def image_info(path: Path) -> tuple[tuple[int, int], float]:
    cv2.setNumThreads(1)
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"UNREADABLE_IMAGE {path}")
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
    value = float(np.tensordot(rgb, WEIGHTS, axes=([-1], [0])).mean())
    return image.shape[:2], value


def analyze_file(path: Path) -> tuple[Path, str, tuple[int, int], float]:
    shape, value = image_info(path)
    return path, digest(path), shape, value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--expected-pairs", type=int, required=True)
    parser.add_argument("--expected-sources", type=int, required=True)
    parser.add_argument(
        "--expected-dataset-count",
        action="append",
        default=[],
        metavar="DATASET=COUNT",
    )
    parser.add_argument("--allow-smoke-license", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    if args.workers <= 0:
        raise SystemExit("BAD_WORKER_COUNT")

    with args.manifest.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise SystemExit("EMPTY_MANIFEST")
        missing_columns = sorted(REQUIRED_FIELDS - set(reader.fieldnames))
        if missing_columns:
            raise SystemExit(f"MISSING_COLUMNS {','.join(missing_columns)}")
        rows = list(reader)

    if len(rows) != args.expected_pairs:
        raise SystemExit(f"BAD_PAIR_COUNT {len(rows)} != {args.expected_pairs}")

    expected_dataset_counts: dict[str, int] = {}
    for item in args.expected_dataset_count:
        try:
            name, count_text = item.split("=", 1)
            expected_dataset_counts[name] = int(count_text)
        except ValueError as exc:
            raise SystemExit(f"BAD_DATASET_COUNT {item}") from exc

    by_source: dict[str, list[dict[str, str]]] = defaultdict(list)
    seen_pairs: set[str] = set()
    file_hash_cache: dict[Path, str] = {}
    image_info_cache: dict[Path, tuple[tuple[int, int], float]] = {}

    unique_paths: list[Path] = []
    seen_paths: set[Path] = set()
    for row in rows:
        for field in ("clean_path", "degraded_path"):
            path = resolve_path(row[field])
            if path in seen_paths:
                continue
            if not path.is_file():
                raise SystemExit(f"MISSING_FILE {path}")
            seen_paths.add(path)
            unique_paths.append(path)

    iterator = map(analyze_file, unique_paths)
    executor: ProcessPoolExecutor | None = None
    if args.workers > 1:
        executor = ProcessPoolExecutor(max_workers=args.workers)
        iterator = executor.map(analyze_file, unique_paths, chunksize=1)
    try:
        for completed, (path, file_digest, shape, value) in enumerate(iterator, start=1):
            file_hash_cache[path] = file_digest
            image_info_cache[path] = (shape, value)
            if completed % 250 == 0 or completed == len(unique_paths):
                print(
                    f"LOW_LIGHT_VALIDATION_PROGRESS {completed}/{len(unique_paths)} files",
                    flush=True,
                )
    finally:
        if executor is not None:
            executor.shutdown()

    for row in rows:
        pair = row["pair_id"]
        if pair in seen_pairs:
            raise SystemExit(f"DUPLICATE_PAIR {pair}")
        seen_pairs.add(pair)
        source_id = row["source_id"]
        by_source[source_id].append(row)
        level = row["project_level"]
        if level not in EXPECTED_LEVELS:
            raise SystemExit(f"BAD_LEVEL {level}")
        if row["degradation_type"] != "low_light":
            raise SystemExit(f"BAD_DEGRADATION_TYPE {pair}")
        for field in REQUIRED_FIELDS:
            if not row[field]:
                raise SystemExit(f"EMPTY_FIELD {pair} {field}")
        clean_path = resolve_path(row["clean_path"])
        degraded_path = resolve_path(row["degraded_path"])
        if file_hash_cache[clean_path] != row["clean_sha256"]:
            raise SystemExit(f"BAD_CLEAN_HASH {pair}")
        if file_hash_cache[degraded_path] != row["degraded_sha256"]:
            raise SystemExit(f"BAD_DEGRADED_HASH {pair}")

        clean_shape, clean_luma = image_info_cache[clean_path]
        degraded_shape, degraded_luma = image_info_cache[degraded_path]
        if clean_shape != degraded_shape:
            raise SystemExit(f"SHAPE_MISMATCH {pair} {clean_shape} != {degraded_shape}")
        if clean_luma <= 0:
            raise SystemExit(f"ZERO_CLEAN_LUMINANCE {pair}")
        recorded_ratio = degraded_luma / clean_luma
        if not INTERVALS[level][0] <= recorded_ratio <= INTERVALS[level][1]:
            raise SystemExit(f"BAD_RATIO {pair} {recorded_ratio}")
        if level == "L1" and (
            clean_path != degraded_path
            or row["clean_sha256"] != row["degraded_sha256"]
        ):
            raise SystemExit(f"L1_NOT_BYTE_IDENTICAL {pair}")

    if len(by_source) != args.expected_sources:
        raise SystemExit(f"BAD_SOURCE_COUNT {len(by_source)} != {args.expected_sources}")

    source_hash_owner: dict[str, str] = {}
    source_split_by_hash: dict[str, str] = {}
    dataset_counts: Counter[str] = Counter()
    for source_id, entries in by_source.items():
        levels = {entry["project_level"] for entry in entries}
        if len(entries) != 5 or levels != EXPECTED_LEVELS:
            raise SystemExit(f"BAD_LEVEL_SET {source_id} {sorted(levels)}")
        constant_fields = (
            "source_dataset",
            "source_original_path",
            "source_license",
            "source_license_status",
            "clean_path",
            "clean_sha256",
            "split",
        )
        for field in constant_fields:
            values = {entry[field] for entry in entries}
            if len(values) != 1:
                raise SystemExit(f"INCONSISTENT_SOURCE_FIELD {source_id} {field}")
        dataset_counts[entries[0]["source_dataset"]] += 1
        clean_hash = entries[0]["clean_sha256"]
        split = entries[0]["split"]
        if clean_hash in source_hash_owner and source_hash_owner[clean_hash] != source_id:
            raise SystemExit(f"DUPLICATE_CLEAN_SOURCE {source_id} {source_hash_owner[clean_hash]}")
        if clean_hash in source_split_by_hash and source_split_by_hash[clean_hash] != split:
            raise SystemExit(f"SOURCE_SPLIT_LEAK_BY_HASH {source_id}")
        source_hash_owner[clean_hash] = source_id
        source_split_by_hash[clean_hash] = split

    if expected_dataset_counts and dict(dataset_counts) != expected_dataset_counts:
        raise SystemExit(f"BAD_DATASET_COUNTS {dict(dataset_counts)} != {expected_dataset_counts}")

    print(f"LOW_LIGHT MANIFEST PASS: {len(rows)} pairs, {len(by_source)} sources")
    print("levels", dict(Counter(row["project_level"] for row in rows)))
    print("source_splits", dict(Counter(entries[0]["split"] for entries in by_source.values())))
    print("source_datasets", dict(dataset_counts))


if __name__ == "__main__":
    main()
