#!/usr/bin/env python3
"""Build a fail-closed, deduplicated source manifest for the 10k pilot."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

ROOT = Path(__file__).resolve().parents[1]
EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}
DEFAULT_QUOTAS = {"mio100": 100, "div2k": 800, "lsdir": 1100}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def parse_quotas(values: list[str]) -> dict[str, int]:
    quotas: dict[str, int] = {}
    for value in values:
        try:
            name, count_text = value.split("=", 1)
            count = int(count_text)
        except ValueError as exc:
            raise SystemExit(f"BAD_QUOTA {value!r}; expected DATASET=COUNT") from exc
        name = name.strip().lower()
        if not name or count < 0 or name in quotas:
            raise SystemExit(f"BAD_QUOTA {value!r}")
        quotas[name] = count
    return quotas


def source_id(dataset: str, relative_path: Path) -> str:
    raw = "_".join(relative_path.with_suffix("").parts)
    normalized = re.sub(r"[^A-Za-z0-9_-]+", "_", raw).strip("_").lower()
    if not normalized:
        raise SystemExit(f"EMPTY_SOURCE_ID {relative_path}")
    return f"{dataset}_{normalized}"


def dct_matrix(size: int) -> np.ndarray:
    coordinates = np.arange(size, dtype=np.float64)
    frequencies = coordinates[:, None]
    scale = np.full(size, np.sqrt(2.0 / size), dtype=np.float64)
    scale[0] = np.sqrt(1.0 / size)
    return scale[:, None] * np.cos(np.pi * (2.0 * coordinates + 1.0) * frequencies / (2.0 * size))


DCT32 = dct_matrix(32)


def perceptual_hash(image: Image.Image) -> int:
    gray = image.convert("L").resize((32, 32), Image.Resampling.LANCZOS)
    pixels = np.asarray(gray, dtype=np.float64)
    low_frequencies = (DCT32 @ pixels @ DCT32.T)[:8, :8]
    median = np.median(low_frequencies.reshape(-1)[1:])
    bits = (low_frequencies >= median).reshape(-1)
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def hamming(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def load_dataset_metadata(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None or not path.is_file():
        return {}
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise SystemExit("BAD_DATASET_METADATA expected a JSON object")
    return payload


def read_blocked_hashes(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    if not path.is_file():
        raise SystemExit(f"MISSING_OVERLAP_INDEX {path}")
    values = {line.strip().lower() for line in path.read_text().splitlines() if line.strip()}
    invalid = [value for value in values if not re.fullmatch(r"[0-9a-f]{64}", value)]
    if invalid:
        raise SystemExit(f"BAD_OVERLAP_HASH {invalid[0]}")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=ROOT / "data/clean_pool")
    parser.add_argument("--required", type=int, default=2000)
    parser.add_argument(
        "--quota",
        action="append",
        default=None,
        metavar="DATASET=COUNT",
        help="Repeat for each dataset; defaults to mio100=100, div2k=800, lsdir=1100.",
    )
    parser.add_argument("--flat-dataset", help="Dataset name for a legacy flat source directory.")
    parser.add_argument("--min-width", type=int, default=512)
    parser.add_argument("--min-height", type=int, default=512)
    parser.add_argument("--near-duplicate-distance", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026080600)
    parser.add_argument("--dataset-metadata", type=Path)
    parser.add_argument("--allow-missing-license-metadata", action="store_true")
    parser.add_argument("--foundir-sha256-index", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "manifests/low_light_source_pool.json")
    args = parser.parse_args()

    if not args.source_root.is_dir():
        raise SystemExit(f"MISSING_SOURCE_ROOT {args.source_root}")
    if min(args.required, args.min_width, args.min_height) <= 0:
        raise SystemExit("BAD_POSITIVE_ARGUMENT")
    if not 0 <= args.near_duplicate_distance <= 64:
        raise SystemExit("BAD_NEAR_DUPLICATE_DISTANCE")

    quota_values = args.quota or [f"{name}={count}" for name, count in DEFAULT_QUOTAS.items()]
    quotas = parse_quotas(quota_values)
    if sum(quotas.values()) != args.required:
        raise SystemExit(f"QUOTA_TOTAL_MISMATCH {sum(quotas.values())} != {args.required}")

    metadata = load_dataset_metadata(args.dataset_metadata)
    if not args.allow_missing_license_metadata:
        for dataset in quotas:
            entry = metadata.get(dataset, {})
            if not entry.get("source") or not entry.get("license") or entry.get("license_status") != "approved":
                raise SystemExit(
                    f"UNAPPROVED_DATASET_METADATA {dataset}; provide source, license, and license_status=approved"
                )
    blocked_hashes = read_blocked_hashes(args.foundir_sha256_index)

    candidates: dict[str, list[tuple[Path, Path]]] = defaultdict(list)
    for path in args.source_root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
            continue
        relative = path.relative_to(args.source_root)
        if args.flat_dataset:
            dataset = args.flat_dataset.lower()
            within_dataset = relative
        elif len(relative.parts) >= 2:
            dataset = relative.parts[0].lower()
            within_dataset = Path(*relative.parts[1:])
        else:
            raise SystemExit(f"FLAT_FILE_NEEDS_DATASET {path}; use --flat-dataset")
        if dataset in quotas:
            candidates[dataset].append((path, within_dataset))

    records: list[dict[str, object]] = []
    rejected: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    seen_sha: dict[str, str] = {}
    seen_hashes: list[tuple[int, str]] = []
    realized = {dataset: 0 for dataset in quotas}

    for dataset, quota in quotas.items():
        dataset_candidates = sorted(candidates.get(dataset, []), key=lambda item: str(item[1]))
        random.Random(f"{args.seed}:{dataset}").shuffle(dataset_candidates)
        for path, relative in dataset_candidates:
            if realized[dataset] >= quota:
                break
            identifier = source_id(dataset, relative)
            if identifier in seen_ids:
                raise SystemExit(f"SOURCE_ID_COLLISION {identifier}")
            try:
                with Image.open(path) as image:
                    image.load()
                    width, height = image.size
                    mode = image.mode
                    if mode != "RGB":
                        rejected.append({"path": display_path(path), "reason": f"NOT_RGB:{mode}"})
                        continue
                    if width < args.min_width or height < args.min_height:
                        rejected.append({"path": display_path(path), "reason": f"TOO_SMALL:{width}x{height}"})
                        continue
                    perceptual = perceptual_hash(image)
            except (OSError, ValueError, UnidentifiedImageError) as exc:
                rejected.append({"path": display_path(path), "reason": f"UNREADABLE:{type(exc).__name__}"})
                continue

            digest = sha256(path)
            if digest in seen_sha:
                rejected.append({"path": display_path(path), "reason": f"EXACT_DUPLICATE:{seen_sha[digest]}"})
                continue
            if blocked_hashes is not None and digest in blocked_hashes:
                rejected.append({"path": display_path(path), "reason": "FOUNDIR_OVERLAP"})
                continue
            near_match = next(
                (other_id for other_hash, other_id in seen_hashes if hamming(perceptual, other_hash) <= args.near_duplicate_distance),
                None,
            )
            if near_match is not None:
                rejected.append({"path": display_path(path), "reason": f"NEAR_DUPLICATE:{near_match}"})
                continue

            dataset_info = metadata.get(dataset, {})
            record = {
                "source_id": identifier,
                "dataset": dataset,
                "path": display_path(path),
                "sha256": digest,
                "bytes": path.stat().st_size,
                "width": width,
                "height": height,
                "mode": mode,
                "average_hash": f"{perceptual:016x}",
                "source": dataset_info.get("source", "UNKNOWN"),
                "license": dataset_info.get("license", "UNKNOWN"),
                "license_status": dataset_info.get("license_status", "unknown"),
            }
            records.append(record)
            seen_ids.add(identifier)
            seen_sha[digest] = identifier
            seen_hashes.append((perceptual, identifier))
            realized[dataset] += 1

    ready = len(records) == args.required and realized == quotas
    report = {
        "schema_version": 2,
        "source_root": display_path(args.source_root),
        "seed": args.seed,
        "minimum_resolution": [args.min_width, args.min_height],
        "near_duplicate_method": "64-bit DCT perceptual hash",
        "near_duplicate_max_hamming_distance": args.near_duplicate_distance,
        "required_unique_sources": args.required,
        "verified_unique_sources": len(records),
        "missing_unique_sources": max(0, args.required - len(records)),
        "requested_quotas": quotas,
        "realized_quotas": realized,
        "pilot_pairs_if_complete": len(records) * 5,
        "foundir_overlap_check": "completed" if blocked_hashes is not None else "not_run_no_index",
        "license_metadata_enforced": not args.allow_missing_license_metadata,
        "ready_for_10000_pair_pilot": ready,
        "records": records,
        "rejected": rejected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key not in {"records", "rejected"}}, indent=2))
    print(f"rejected_candidates: {len(rejected)}")
    if not ready:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
