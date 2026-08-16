#!/usr/bin/env python3
"""Fail-closed validation and manifesting for the official FoundIR test set."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import subprocess
from collections import Counter
from pathlib import Path

from PIL import Image, UnidentifiedImageError

ROOT = Path(__file__).resolve().parents[1]
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}
EXPECTED_FOUNDIR_REVISION = "6d19958499622d90ca71fe971d83b1e428c17cd0"
EXPECTED_META_SHA256 = "15467efd89a55d70a381f813d29f84b4c63ae5a958ef67ffa9bcf1883deb3761"
EXPECTED_METRIC_SHA256 = "bf5bbff7c73b3a8a9623180764ff6ded6cdf4e03960d12825a5b68eaacaf9b32"
LOWLIGHT_CATEGORY = "14Lowlight"
LOWLIGHT_IDS = {f"{value:04d}" for value in range(1101, 1151)}
EXPECTED_CATEGORY_COUNTS = {
    "01Blur": 150,
    "02Blur_Noise": 50,
    "03Blur_JPEG": 50,
    "04Blur_Noise_JPEG": 50,
    "05Noise": 100,
    "06JPEG": 50,
    "07Noise_JPEG": 50,
    "08Haze": 200,
    "09Lowlight_Haze": 100,
    "10Rain": 100,
    "11Raindrop": 100,
    "12NightRain": 50,
    "13Rain_Haze": 50,
    "14Lowlight": 50,
    "15Lowlight_Blur": 100,
    "16Lowlight_Noise": 50,
    "17Lowlight_JPEG": 50,
    "18Lowlight_Blur_Noise": 50,
    "19Lowlight_Blur_JPEG": 50,
    "20Lowlight_Noise_JPEG": 50,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def workspace_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        raise SystemExit(f"PATH_OUTSIDE_WORKSPACE {resolved}") from None


def repo_revision(repo: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        text=True,
        capture_output=True,
    )
    return result.stdout.strip()


def parse_meta(meta_path: Path) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    seen_keys: set[tuple[str, str]] = set()
    with meta_path.open(newline="") as handle:
        reader = csv.reader(handle, skipinitialspace=True)
        for line_number, row in enumerate(reader, start=1):
            if len(row) != 2:
                raise SystemExit(f"BAD_META_ROW line={line_number} fields={len(row)}")
            gt_rel, lq_rel = (value.strip().replace("\\", "/") for value in row)
            gt_parts = Path(gt_rel).parts
            lq_parts = Path(lq_rel).parts
            if len(gt_parts) != 3 or len(lq_parts) != 3:
                raise SystemExit(f"BAD_META_PATH line={line_number}")
            category, gt_side, gt_name = gt_parts
            lq_category, lq_side, lq_name = lq_parts
            if category != lq_category or gt_side != "GT" or lq_side != "LQ":
                raise SystemExit(f"BAD_META_PAIR line={line_number}")
            if Path(gt_name).stem != Path(lq_name).stem:
                raise SystemExit(f"BAD_META_ID line={line_number}")
            pair_id = Path(gt_name).stem
            if not re.fullmatch(r"\d{4}", pair_id):
                raise SystemExit(f"BAD_PAIR_ID line={line_number} id={pair_id}")
            key = (category, pair_id)
            if key in seen_keys:
                raise SystemExit(f"DUPLICATE_META_PAIR {category}/{pair_id}")
            seen_keys.add(key)
            records.append(
                {
                    "category": category,
                    "pair_id": pair_id,
                    "gt_name": gt_name,
                    "lq_name": lq_name,
                }
            )
    counts = Counter(record["category"] for record in records)
    if len(records) != 1500 or dict(counts) != EXPECTED_CATEGORY_COUNTS:
        raise SystemExit(f"BAD_META_COMPOSITION total={len(records)} counts={dict(counts)}")
    isolated_ids = {
        record["pair_id"] for record in records if record["category"] == LOWLIGHT_CATEGORY
    }
    if isolated_ids != LOWLIGHT_IDS:
        raise SystemExit("BAD_ISOLATED_LOWLIGHT_IDS")
    return records


def index_images(root: Path, expected_categories: set[str]) -> dict[tuple[str, str], Path]:
    index: dict[tuple[str, str], Path] = {}
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        categories = [part for part in path.relative_to(root).parts if part in expected_categories]
        if len(categories) != 1:
            raise SystemExit(f"UNRESOLVED_CATEGORY {path}")
        key = (categories[0], path.stem)
        if key in index:
            raise SystemExit(f"DUPLICATE_DATA_IMAGE {key} {index[key]} {path}")
        index[key] = path
    return index


def inspect_image(path: Path) -> tuple[int, int, str]:
    try:
        with Image.open(path) as image:
            image.load()
            width, height = image.size
            mode = image.mode
    except (OSError, ValueError, UnidentifiedImageError) as exc:
        raise SystemExit(f"UNREADABLE_IMAGE {path}: {type(exc).__name__}") from exc
    if width <= 0 or height <= 0:
        raise SystemExit(f"BAD_IMAGE_SIZE {path} {width}x{height}")
    return width, height, mode


def load_source_hashes(path: Path) -> dict[str, str]:
    payload = json.loads(path.read_text())
    records = payload.get("records")
    if not isinstance(records, list) or len(records) != 2000:
        raise SystemExit("BAD_SOURCE_MANIFEST")
    result: dict[str, str] = {}
    for record in records:
        digest = record.get("sha256")
        source_id = record.get("source_id")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise SystemExit("BAD_SOURCE_SHA256")
        if not isinstance(source_id, str) or not source_id:
            raise SystemExit("BAD_SOURCE_ID")
        result[digest] = source_id
    return result


def write_or_verify_bytes(path: Path, content: bytes) -> None:
    if path.exists():
        if not path.is_file() or path.read_bytes() != content:
            raise SystemExit(f"EXISTING_OUTPUT_MISMATCH {path}")
        print(f"VERIFIED_EXISTING_OUTPUT {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(content)


def write_csv_or_verify(path: Path, records: list[dict[str, object]]) -> None:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(records[0]))
    writer.writeheader()
    writer.writerows(records)
    write_or_verify_bytes(path, buffer.getvalue().encode())


def write_json_or_verify(path: Path, payload: object) -> None:
    content = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    write_or_verify_bytes(path, content)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--foundir-repo", type=Path, default=ROOT / "third_party/foundir")
    parser.add_argument("--gt-root", type=Path, required=True)
    parser.add_argument("--lq-root", type=Path, required=True)
    parser.add_argument("--gt-archive", type=Path, required=True)
    parser.add_argument("--lq-archive", type=Path, required=True)
    parser.add_argument(
        "--source-manifest", type=Path, default=ROOT / "manifests/low_light_source_pool_2000.json"
    )
    parser.add_argument(
        "--output-csv", type=Path, default=ROOT / "manifests/foundir_lowlight_official.csv"
    )
    parser.add_argument(
        "--output-json", type=Path, default=ROOT / "manifests/foundir_lowlight_official.json"
    )
    parser.add_argument(
        "--hash-index", type=Path, default=ROOT / "manifests/foundir_test_gt_sha256.json"
    )
    args = parser.parse_args()

    for path in (args.gt_root, args.lq_root):
        if not path.is_dir():
            raise SystemExit(f"MISSING_DATA_ROOT {path}")
    for path in (args.gt_archive, args.lq_archive, args.source_manifest):
        if not path.is_file():
            raise SystemExit(f"MISSING_REQUIRED_FILE {path}")

    foundir_repo = args.foundir_repo.resolve()
    meta_path = foundir_repo / "Testset_meta_info.txt"
    metric_path = foundir_repo / "cal_metrics.py"
    revision = repo_revision(foundir_repo)
    if revision != EXPECTED_FOUNDIR_REVISION:
        raise SystemExit(f"BAD_FOUNDIR_REVISION {revision}")
    if sha256(meta_path) != EXPECTED_META_SHA256:
        raise SystemExit("BAD_FOUNDIR_META_HASH")
    if sha256(metric_path) != EXPECTED_METRIC_SHA256:
        raise SystemExit("BAD_FOUNDIR_METRIC_HASH")

    meta_records = parse_meta(meta_path)
    categories = set(EXPECTED_CATEGORY_COUNTS)
    gt_index = index_images(args.gt_root, categories)
    lq_index = index_images(args.lq_root, categories)
    expected_keys = {(record["category"], record["pair_id"]) for record in meta_records}
    if set(gt_index) != expected_keys:
        raise SystemExit(
            f"GT_INDEX_MISMATCH missing={len(expected_keys - set(gt_index))} "
            f"extra={len(set(gt_index) - expected_keys)}"
        )
    if set(lq_index) != expected_keys:
        raise SystemExit(
            f"LQ_INDEX_MISMATCH missing={len(expected_keys - set(lq_index))} "
            f"extra={len(set(lq_index) - expected_keys)}"
        )

    source_hashes = load_source_hashes(args.source_manifest)
    all_records: list[dict[str, object]] = []
    lowlight_records: list[dict[str, object]] = []
    full_test_overlap: list[dict[str, str]] = []
    for index, record in enumerate(meta_records, start=1):
        key = (record["category"], record["pair_id"])
        gt_path = gt_index[key]
        lq_path = lq_index[key]
        gt_width, gt_height, gt_mode = inspect_image(gt_path)
        lq_width, lq_height, lq_mode = inspect_image(lq_path)
        if (gt_width, gt_height) != (lq_width, lq_height):
            raise SystemExit(
                f"PAIR_DIMENSION_MISMATCH {key} GT={gt_width}x{gt_height} LQ={lq_width}x{lq_height}"
            )
        gt_digest = sha256(gt_path)
        lq_digest = sha256(lq_path)
        if gt_digest in source_hashes:
            full_test_overlap.append(
                {
                    "pair_id": record["pair_id"],
                    "category": record["category"],
                    "source_id": source_hashes[gt_digest],
                    "sha256": gt_digest,
                }
            )
        common = {
            "pair_id": record["pair_id"],
            "category": record["category"],
            "gt_path": workspace_path(gt_path),
            "lq_path": workspace_path(lq_path),
            "gt_sha256": gt_digest,
            "lq_sha256": lq_digest,
            "gt_bytes": gt_path.stat().st_size,
            "lq_bytes": lq_path.stat().st_size,
            "width": gt_width,
            "height": gt_height,
            "gt_mode": gt_mode,
            "lq_mode": lq_mode,
        }
        all_records.append(common)
        if record["category"] == LOWLIGHT_CATEGORY:
            lowlight_records.append(common)
        if index % 100 == 0:
            print(f"VALIDATED {index}/1500", flush=True)

    if len(lowlight_records) != 50:
        raise SystemExit(f"BAD_LOWLIGHT_COUNT {len(lowlight_records)}")
    lowlight_overlap = [
        record for record in full_test_overlap if record["category"] == LOWLIGHT_CATEGORY
    ]
    if lowlight_overlap:
        raise SystemExit(
            f"FOUNDIR_LOWLIGHT_TRAIN_SOURCE_OVERLAP {json.dumps(lowlight_overlap, sort_keys=True)}"
        )

    gt_archive_sha = sha256(args.gt_archive)
    lq_archive_sha = sha256(args.lq_archive)
    index_payload = {
        "schema_version": 1,
        "foundir_revision": revision,
        "meta_sha256": EXPECTED_META_SHA256,
        "records": [
            {
                "pair_id": record["pair_id"],
                "category": record["category"],
                "gt_sha256": record["gt_sha256"],
            }
            for record in all_records
        ],
    }
    summary = {
        "schema_version": 1,
        "status": "official_foundir_isolated_lowlight_validated",
        "benchmark": "FoundIR isolated Lowlight",
        "category": LOWLIGHT_CATEGORY,
        "pair_id_range": ["1101", "1150"],
        "pairs": 50,
        "full_test_pairs_validated": 1500,
        "category_counts": EXPECTED_CATEGORY_COUNTS,
        "foundir_revision": revision,
        "meta_path": workspace_path(meta_path),
        "meta_sha256": EXPECTED_META_SHA256,
        "metric_path": workspace_path(metric_path),
        "metric_sha256": EXPECTED_METRIC_SHA256,
        "gt_archive": workspace_path(args.gt_archive),
        "gt_archive_bytes": args.gt_archive.stat().st_size,
        "gt_archive_sha256": gt_archive_sha,
        "lq_archive": workspace_path(args.lq_archive),
        "lq_archive_bytes": args.lq_archive.stat().st_size,
        "lq_archive_sha256": lq_archive_sha,
        "source_manifest": workspace_path(args.source_manifest),
        "source_overlap_method": "exact SHA-256 against all 1,500 official GT images",
        "full_test_source_overlap_count": len(full_test_overlap),
        "full_test_source_overlaps": full_test_overlap,
        "isolated_lowlight_source_overlap_count": 0,
        "protocol": {
            "selection": "14Lowlight only; IDs 1101-1150",
            "input": "official paired LQ image",
            "target": "official paired GT image",
            "metrics": ["PSNR", "SSIM", "LPIPS"],
            "tensor": "PIL RGB -> float32 NCHW / 255.0",
            "metric_resize_or_crop": "none",
            "aggregation": "unweighted arithmetic mean over 50 pairs",
            "phase1_thresholds": {"psnr_min_db": 15.1, "lpips_max": 0.34},
        },
        "records": lowlight_records,
    }
    write_csv_or_verify(args.output_csv, lowlight_records)
    write_json_or_verify(args.hash_index, index_payload)
    write_json_or_verify(args.output_json, summary)
    print(
        "FOUNDIR_OFFICIAL_VALIDATION_PASS",
        json.dumps(
            {
                "full_test_pairs": 1500,
                "isolated_lowlight_pairs": 50,
                "full_test_source_overlap_count": len(full_test_overlap),
                "isolated_lowlight_source_overlap_count": 0,
                "gt_archive_sha256": gt_archive_sha,
                "lq_archive_sha256": lq_archive_sha,
            },
            sort_keys=True,
        ),
    )


if __name__ == "__main__":
    main()
