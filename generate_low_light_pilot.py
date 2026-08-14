#!/usr/bin/env python3
"""Generate the reproducible AgenticIR low-light 10,000-pair pilot.

Production input is a verified 2,000-record source-pool manifest. A direct
source directory is retained only for explicitly smaller smoke tests.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import random
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
AGENTICIR_REVISION = "9640a291480dee3ba8f2974125d4ee9e3440f3d6"
AGENTICIR_DEPS = ROOT / "third_party/AgenticIR/installation/tune_depictqa"
EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}
PRODUCTION_QUOTAS = {"mio100": 100, "div2k": 800, "lsdir": 1100}

TARGET_RATIOS = {"L1": 1.00, "L2": 0.50, "L3": 0.38, "L4": 0.30, "L5": 0.14}
INTERVALS = {
    "L1": (0.95, 1.00),
    "L2": (0.48, 0.52),
    "L3": (0.36, 0.40),
    "L4": (0.28, 0.32),
    "L5": (0.12, 0.16),
}
WEIGHTS = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
FIELDS = [
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
    "official_agenticir_level",
    "darkening_method",
    "seed",
    "raw_luminance",
    "final_luminance",
    "target_ratio",
    "calibration_factor",
    "agenticir_revision",
    "generator_revision",
    "split",
    "clean_sha256",
    "degraded_sha256",
]


@dataclass(frozen=True)
class Source:
    source_id: str
    dataset: str
    path: Path
    expected_sha256: str
    license_name: str
    license_status: str


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def luma(image: np.ndarray) -> float:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
    return float(np.tensordot(rgb, WEIGHTS, axes=([-1], [0])).mean())


def atomic_write(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
    if not cv2.imwrite(str(temporary), image):
        raise RuntimeError(f"failed to write {temporary}")
    temporary.replace(path)


def write_text_safely(path: Path, content: str, resume: bool) -> str:
    """Write generated metadata without replacing a differing resume target."""
    encoded = content.encode()
    if path.exists() and resume:
        if path.read_bytes() != encoded:
            raise RuntimeError(f"resume target differs; refusing to overwrite: {path}")
        return "reused"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(encoded)
    temporary.replace(path)
    return "written"


def manifest_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def resolve_manifest_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def split_sources(source_ids: list[str], seed: int) -> dict[str, str]:
    ordered = list(source_ids)
    random.Random(seed).shuffle(ordered)
    count = len(ordered)
    train_end, valid_end = int(count * 0.8), int(count * 0.9)
    return {
        source_id: "train" if index < train_end else "validation" if index < valid_end else "test"
        for index, source_id in enumerate(ordered)
    }


def load_manifest_sources(path: Path, production: bool) -> list[Source]:
    if not path.is_file():
        raise SystemExit(f"MISSING_SOURCE_MANIFEST {path}")
    payload = json.loads(path.read_text())
    if production and payload.get("ready_for_10000_pair_pilot") is not True:
        raise SystemExit("SOURCE_MANIFEST_NOT_READY_FOR_10000_PAIR_PILOT")
    records = payload.get("records")
    if not isinstance(records, list):
        raise SystemExit("BAD_SOURCE_MANIFEST records must be a list")
    sources: list[Source] = []
    seen_ids: set[str] = set()
    seen_hashes: set[str] = set()
    for record in records:
        required = {"source_id", "dataset", "path", "sha256"}
        missing = sorted(required - record.keys())
        if missing:
            raise SystemExit(f"BAD_SOURCE_RECORD missing {','.join(missing)}")
        source_id = str(record["source_id"])
        dataset = str(record["dataset"]).lower()
        if not source_id.startswith(f"{dataset}_"):
            raise SystemExit(f"SOURCE_ID_MISSING_DATASET_PREFIX {source_id}")
        expected_hash = str(record["sha256"]).lower()
        if source_id in seen_ids:
            raise SystemExit(f"DUPLICATE_SOURCE_ID {source_id}")
        if expected_hash in seen_hashes:
            raise SystemExit(f"DUPLICATE_SOURCE_SHA256 {source_id}")
        source_path = resolve_manifest_path(str(record["path"]))
        if not source_path.is_file():
            raise SystemExit(f"MISSING_SOURCE_FILE {source_path}")
        actual_hash = sha256(source_path)
        if actual_hash != expected_hash:
            raise SystemExit(f"SOURCE_HASH_CHANGED {source_id}")
        sources.append(
            Source(
                source_id=source_id,
                dataset=dataset,
                path=source_path,
                expected_sha256=expected_hash,
                license_name=str(record.get("license", "UNKNOWN")),
                license_status=str(record.get("license_status", "not_provided")),
            )
        )
        seen_ids.add(source_id)
        seen_hashes.add(expected_hash)
    if production:
        counts = {dataset: sum(1 for source in sources if source.dataset == dataset) for dataset in PRODUCTION_QUOTAS}
        if counts != PRODUCTION_QUOTAS or len(sources) != sum(PRODUCTION_QUOTAS.values()):
            raise SystemExit(f"BAD_PRODUCTION_SOURCE_COMPOSITION {counts}")
    return sources


def load_smoke_sources(root: Path) -> list[Source]:
    if not root.is_dir():
        raise SystemExit(f"MISSING_SOURCE_ROOT {root}")
    paths = sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in EXTENSIONS)
    sources: list[Source] = []
    seen_ids: set[str] = set()
    seen_hashes: set[str] = set()
    for path in paths:
        relative = path.relative_to(root)
        normalized = "_".join(relative.with_suffix("").parts).replace(" ", "_").lower()
        source_id = f"smoke_{normalized}"
        digest = sha256(path)
        if source_id in seen_ids:
            raise SystemExit(f"DUPLICATE_SOURCE_ID {source_id}")
        if digest in seen_hashes:
            raise SystemExit(f"DUPLICATE_SOURCE_SHA256 {source_id}")
        sources.append(Source(source_id, "smoke", path, digest, "SMOKE_ONLY", "smoke_only"))
        seen_ids.add(source_id)
        seen_hashes.add(digest)
    return sources


def process_source(
    task: tuple[int, Source, int, str, Path, bool],
) -> tuple[list[dict[str, object]], int, int]:
    index, source, seed, split, output_root, resume = task
    cv2.setNumThreads(1)
    sys.path.insert(0, str(AGENTICIR_DEPS))
    from add_single_degradation import darken

    source_seed = seed + index * 10
    clean = cv2.imread(str(source.path), cv2.IMREAD_COLOR)
    if clean is None:
        raise RuntimeError(f"cannot read source image: {source.path}")
    clean_luma = luma(clean)
    if clean_luma <= 0:
        raise RuntimeError(f"zero-luminance source image: {source.path}")
    np.random.seed(source_seed)

    suffix = source.path.suffix.lower()
    clean_rel = Path("clean") / f"{source.source_id}{suffix}"
    clean_out = output_root / clean_rel
    clean_out.parent.mkdir(parents=True, exist_ok=True)
    reused = 0
    written = 0
    if resume and clean_out.exists():
        clean_hash = sha256(clean_out)
        if clean_hash != source.expected_sha256:
            raise RuntimeError(f"resume clean file differs; refusing to overwrite: {source.source_id}")
        reused += 1
    else:
        shutil.copy2(source.path, clean_out)
        clean_hash = sha256(clean_out)
        if clean_hash != source.expected_sha256:
            raise RuntimeError(f"byte-preserving copy failed: {source.source_id}")
        written += 1

    rows: list[dict[str, object]] = []
    for level_index, level in enumerate(TARGET_RATIOS):
        level_seed = source_seed
        if level == "L1":
            result, raw_luma, factor = clean, clean_luma, 1.0
            official_level = "clean"
            degraded_out = clean_out
        else:
            result = darken(clean, level=level_index, darken_type="linear mapping")
            raw_luma = luma(result)
            if raw_luma <= 0:
                raise RuntimeError(f"zero-luminance degradation: {source.source_id} {level}")
            target_luma = clean_luma * TARGET_RATIOS[level]
            factor = target_luma / raw_luma
            result = np.rint(result.astype(np.float32) * factor).clip(0, 255).astype(np.uint8)
            official_level = str(level_index)
            degraded_out = output_root / "pairs" / source.source_id / f"{level}.png"
            if resume and degraded_out.exists():
                existing = cv2.imread(str(degraded_out), cv2.IMREAD_COLOR)
                if existing is None or existing.shape != result.shape or not np.array_equal(existing, result):
                    raise RuntimeError(
                        f"resume degraded file differs; refusing to overwrite: {source.source_id} {level}"
                    )
                reused += 1
            else:
                atomic_write(degraded_out, result)
                written += 1

        final_luma = luma(result)
        ratio = final_luma / clean_luma
        low, high = INTERVALS[level]
        if not low <= ratio <= high:
            raise RuntimeError(f"{source.source_id} {level} ratio {ratio:.5f} outside [{low}, {high}]")
        rows.append(
            {
                "pair_id": f"{source.source_id}_{level}",
                "source_id": source.source_id,
                "source_dataset": source.dataset,
                "source_original_path": manifest_path(source.path),
                "source_license": source.license_name,
                "source_license_status": source.license_status,
                "clean_path": manifest_path(clean_out),
                "degraded_path": manifest_path(degraded_out),
                "degradation_type": "low_light",
                "project_level": level,
                "official_agenticir_level": official_level,
                "darkening_method": "linear mapping",
                "seed": level_seed,
                "raw_luminance": f"{raw_luma:.6f}",
                "final_luminance": f"{final_luma:.6f}",
                "target_ratio": TARGET_RATIOS[level],
                "calibration_factor": f"{factor:.9f}",
                "agenticir_revision": AGENTICIR_REVISION,
                "generator_revision": "generate_low_light_pilot.py:v3-resume",
                "split": split,
                "clean_sha256": clean_hash,
                "degraded_sha256": sha256(degraded_out),
            }
        )
    return rows, reused, written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--source-manifest", type=Path)
    inputs.add_argument("--source-root", type=Path)
    parser.add_argument("--output-root", type=Path, default=ROOT / "data/low_light_pilot")
    parser.add_argument("--manifest", type=Path, default=ROOT / "manifests/low_light_pilot_10000.csv")
    parser.add_argument("--metadata", type=Path, default=ROOT / "manifests/low_light_pilot_10000.json")
    parser.add_argument("--num-sources", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2026080600)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Verify and reuse matching outputs; never overwrite a differing existing file.",
    )
    parser.add_argument(
        "--allow-fewer-sources",
        action="store_true",
        help="Permit a smaller direct-directory smoke run; never use for the 10,000-pair pilot.",
    )
    args = parser.parse_args()

    if args.num_sources <= 0 or args.workers <= 0:
        raise SystemExit("BAD_NUM_SOURCES")
    if not args.allow_fewer_sources and args.num_sources != sum(PRODUCTION_QUOTAS.values()):
        raise SystemExit("PRODUCTION_NUM_SOURCES_MUST_BE_2000")
    if args.source_root is not None and not args.allow_fewer_sources:
        raise SystemExit("PRODUCTION_REQUIRES_SOURCE_MANIFEST; direct --source-root is smoke-only")
    source_manifest = args.source_manifest or ROOT / "manifests/low_light_source_pool.json"
    sources = (
        load_smoke_sources(args.source_root)
        if args.source_root is not None
        else load_manifest_sources(source_manifest, production=not args.allow_fewer_sources)
    )
    if len(sources) < args.num_sources and not args.allow_fewer_sources:
        raise SystemExit(f"NEED_{args.num_sources}_VERIFIED_SOURCES_FOUND_{len(sources)}")
    sources = sources[: min(args.num_sources, len(sources))]
    if not sources:
        raise SystemExit("NO_VALID_SOURCES")

    splits = split_sources([source.source_id for source in sources], args.seed)
    tasks = [
        (index, source, args.seed, splits[source.source_id], args.output_root, args.resume)
        for index, source in enumerate(sources)
    ]
    rows: list[dict[str, object]] = []
    reused = 0
    written = 0
    iterator = map(process_source, tasks)
    if args.workers > 1:
        executor = ProcessPoolExecutor(max_workers=args.workers)
        iterator = executor.map(process_source, tasks, chunksize=1)
    try:
        for completed, (source_rows, source_reused, source_written) in enumerate(iterator, start=1):
            rows.extend(source_rows)
            reused += source_reused
            written += source_written
            if completed % 50 == 0 or completed == len(tasks):
                print(
                    f"LOW_LIGHT_PROGRESS {completed}/{len(tasks)} sources "
                    f"reused={reused} written={written}",
                    flush=True,
                )
    finally:
        if args.workers > 1:
            executor.shutdown()

    csv_buffer = io.StringIO(newline="")
    writer = csv.DictWriter(csv_buffer, fieldnames=FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    write_text_safely(args.manifest, csv_buffer.getvalue(), args.resume)
    summary = {
        "schema_version": 2,
        "pairs": len(rows),
        "sources": len(sources),
        "levels": list(TARGET_RATIOS),
        "source_datasets": {
            dataset: sum(1 for source in sources if source.dataset == dataset)
            for dataset in sorted({source.dataset for source in sources})
        },
        "splits": {
            split: sum(1 for source_split in splits.values() if source_split == split)
            for split in ("train", "validation", "test")
        },
        "seed": args.seed,
        "agenticir_revision": AGENTICIR_REVISION,
        "generator_revision": "generate_low_light_pilot.py:v3-resume",
        "resume": args.resume,
        "manifest": manifest_path(args.manifest),
        "output_root": manifest_path(args.output_root),
    }
    write_text_safely(args.metadata, json.dumps(summary, indent=2) + "\n", args.resume)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
