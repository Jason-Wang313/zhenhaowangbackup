#!/usr/bin/env python3
"""Build the immutable source-disjoint five-type FoundIR training manifest."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import os
import sys
import tempfile
from collections import Counter, defaultdict
from functools import partial
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lowlight_restoration.foundir_five_type import (
    category_counts,
    load_five_type_metadata,
    perceptual_components,
    stable_group_split,
)
from smoke.check_low_light_source_pool import perceptual_hash

PROGRESS_INTERVAL = 10_000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_pair(item: tuple[int, object], training_root: Path) -> dict[str, object]:
    index, record = item
    gt_path = training_root / record.gt_relative
    lq_path = training_root / record.lq_relative
    if not gt_path.is_file() or not lq_path.is_file():
        raise FileNotFoundError(f"missing FoundIR pair: {gt_path}, {lq_path}")
    with Image.open(gt_path) as handle:
        gt = handle.convert("RGB")
        gt_size = gt.size
        gt_phash = perceptual_hash(gt)
    with Image.open(lq_path) as handle:
        lq = handle.convert("RGB")
        lq_size = lq.size
    if gt_size != lq_size:
        raise RuntimeError(f"unaligned FoundIR dimensions: {record.gt_relative}, {record.lq_relative}")
    if min(gt_size) < 64:
        raise RuntimeError(f"unexpectedly small FoundIR pair: {record.gt_relative} {gt_size}")
    return {
        "index": index,
        "category": record.category,
        "degradations": list(record.degradations),
        "composition_size": record.composition_size,
        "gt_relative": record.gt_relative,
        "lq_relative": record.lq_relative,
        "gt_sha256": sha256(gt_path),
        "lq_sha256": sha256(lq_path),
        "gt_phash": f"{gt_phash:016x}",
        "width": gt_size[0],
        "height": gt_size[1],
    }


def publish_once(temporary: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.read_bytes() == temporary.read_bytes():
            temporary.unlink()
            return
        raise RuntimeError(f"refusing to overwrite differing artifact: {destination}")
    os.link(temporary, destination)
    temporary.unlink()


def verify_existing(config_path: Path, output_csv: Path, output_json: Path) -> dict[str, object]:
    if not output_csv.is_file() or not output_json.is_file():
        raise FileNotFoundError("existing FoundIR manifest artifact set is incomplete")
    summary = json.loads(output_json.read_text())
    config = json.loads(config_path.read_text())
    expected = int(config["data"]["expected_five_type_records"])
    if (
        summary.get("status") != "frozen_five_type_foundir_training_manifest"
        or summary.get("config_sha256") != sha256(config_path)
        or summary.get("manifest_sha256") != sha256(output_csv)
        or int(summary.get("records", -1)) != expected
        or summary.get("official_test_accessed") is not False
    ):
        raise RuntimeError("existing FoundIR manifest identity changed")
    with output_csv.open(newline="") as handle:
        count = sum(1 for _ in csv.DictReader(handle))
    if count != expected:
        raise RuntimeError("existing FoundIR manifest row count changed")
    print("FOUNDIR_FIVE_TYPE_MANIFEST_VERIFY_PASS " + json.dumps(summary, sort_keys=True), flush=True)
    return summary


def build(config_path: Path, output_csv: Path, output_json: Path, *, workers: int) -> dict[str, object]:
    config = json.loads(config_path.read_text())
    data = config["data"]
    metadata = ROOT / data["official_metadata"]
    records, _ = load_five_type_metadata(metadata)
    if len(records) != data["expected_five_type_records"]:
        raise RuntimeError("five-type metadata count changed")
    training_root = (ROOT / data["training_root"]).resolve()
    if not training_root.is_dir():
        raise FileNotFoundError(
            f"FoundIR official training root is missing: {training_root}. "
            "Do not point this builder at downloads/foundir_official; that is the protected test."
        )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    total_records = len(records)
    indexed = enumerate(records)
    inspected: list[dict[str, object]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        for count, result in enumerate(
            executor.map(partial(inspect_pair, training_root=training_root), indexed), start=1
        ):
            inspected.append(result)
            if count % PROGRESS_INTERVAL == 0:
                print(f"FOUNDIR_FIVE_TYPE_INSPECTED {count}/{total_records}", flush=True)
    inspected.sort(key=lambda item: int(item["index"]))

    exact_first: dict[str, int] = {}
    components = perceptual_components(
        [int(str(item["gt_phash"]), 16) for item in inspected], maximum_distance=4
    )
    # Exact duplicates must share a component even if a future pHash
    # implementation changes while retaining the frozen exact bytes.
    from lowlight_restoration.foundir_five_type import DisjointSet

    union = DisjointSet(len(inspected))
    for index, component in enumerate(components):
        union.union(index, component)
    for index, item in enumerate(inspected):
        digest = str(item["gt_sha256"])
        if digest in exact_first:
            union.union(index, exact_first[digest])
        else:
            exact_first[digest] = index
    # Normalize every representative after both pHash and exact-byte unions.
    # A representative returned by the first union pass can itself become a
    # child during the exact-byte pass, so using stale roots here would split a
    # single connected component across multiple data splits.
    member_groups: dict[int, list[int]] = defaultdict(list)
    for index in range(len(inspected)):
        member_groups[union.find(index)].append(index)

    group_identity: dict[int, str] = {}
    group_split: dict[int, str] = {}
    for root, members in member_groups.items():
        identity = hashlib.sha256(
            "\n".join(sorted(str(inspected[index]["gt_sha256"]) for index in members)).encode()
        ).hexdigest()
        group_identity[root] = identity
        group_split[root] = stable_group_split(identity, seed=int(data["split_seed"]))

    split_counts: Counter[str] = Counter()
    split_categories: dict[str, Counter[str]] = defaultdict(Counter)
    fieldnames = [
        "record_id", "split", "source_group", "category", "composition_size",
        "degradations_json", "lq_path", "gt_path", "lq_sha256", "gt_sha256",
        "gt_phash", "width", "height",
    ]
    with tempfile.NamedTemporaryFile(
        mode="w", newline="", dir=output_csv.parent, prefix=f".{output_csv.name}.", delete=False
    ) as handle:
        temporary_csv = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, item in enumerate(inspected):
            root = union.find(index)
            split = group_split[root]
            split_counts[split] += 1
            split_categories[split][str(item["category"])] += 1
            writer.writerow(
                {
                    "record_id": f"foundir_train_{index:07d}",
                    "split": split,
                    "source_group": group_identity[root],
                    "category": item["category"],
                    "composition_size": item["composition_size"],
                    "degradations_json": json.dumps(item["degradations"], separators=(",", ":")),
                    "lq_path": str(Path(data["training_root"]) / str(item["lq_relative"])),
                    "gt_path": str(Path(data["training_root"]) / str(item["gt_relative"])),
                    "lq_sha256": item["lq_sha256"],
                    "gt_sha256": item["gt_sha256"],
                    "gt_phash": item["gt_phash"],
                    "width": item["width"],
                    "height": item["height"],
                }
            )
        handle.flush()
        os.fsync(handle.fileno())

    summary = {
        "schema_version": 1,
        "status": "frozen_five_type_foundir_training_manifest",
        "config": str(config_path.relative_to(ROOT)),
        "config_sha256": sha256(config_path),
        "official_test_accessed": False,
        "records": len(inspected),
        "source_components": len(member_groups),
        "category_counts": category_counts(records),
        "split_counts": dict(sorted(split_counts.items())),
        "split_category_counts": {
            split: dict(sorted(counts.items())) for split, counts in sorted(split_categories.items())
        },
        "grouping": "GT_SHA256_union_64bit_DCT_pHash_hamming_le_4",
        "manifest": str(output_csv.relative_to(ROOT)),
        "manifest_sha256": sha256(temporary_csv),
    }
    with tempfile.NamedTemporaryFile(
        mode="w", dir=output_json.parent, prefix=f".{output_json.name}.", delete=False
    ) as handle:
        temporary_json = Path(handle.name)
        handle.write(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    publish_once(temporary_csv, output_csv)
    publish_once(temporary_json, output_json)
    print("FOUNDIR_FIVE_TYPE_MANIFEST_PASS " + json.dumps(summary, sort_keys=True), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/foundir_five_type_flux2_dit_v1.json")
    parser.add_argument("--output-csv", type=Path, default=ROOT / "manifests/foundir_five_type_pairs_v1.csv")
    parser.add_argument("--output-json", type=Path, default=ROOT / "manifests/foundir_five_type_pairs_v1.json")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--verify-existing", action="store_true")
    args = parser.parse_args()
    if args.workers < 1 or args.workers > 128:
        raise ValueError("workers must be in [1, 128]")
    if args.verify_existing:
        verify_existing(args.config.resolve(), args.output_csv.resolve(), args.output_json.resolve())
    else:
        build(args.config.resolve(), args.output_csv.resolve(), args.output_json.resolve(), workers=args.workers)


if __name__ == "__main__":
    main()
