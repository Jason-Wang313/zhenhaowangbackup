#!/usr/bin/env python3
"""Fail fast on FoundIR pair geometry before loading eight FLUX replicas."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from smoke.train_foundir_five_type_flux2_full_ddp import (  # noqa: E402
    TRAINING_UNDERSIZED_PAIR_POLICY,
    resized_pair_dimensions,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def geometry_summary(
    rows: Sequence[Mapping[str, str]], crop_size: int
) -> dict[str, object]:
    undersized: list[Mapping[str, str]] = []
    categories: Counter[str] = Counter()
    splits: Counter[str] = Counter()
    minimum_width: int | None = None
    minimum_height: int | None = None
    seen: set[str] = set()
    for row in rows:
        record_id = str(row.get("record_id", ""))
        if not record_id or record_id in seen:
            raise RuntimeError(f"missing or duplicate training record ID: {record_id!r}")
        seen.add(record_id)
        try:
            width, height = int(row["width"]), int(row["height"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"invalid manifest dimensions for {record_id}") from exc
        if width < 1 or height < 1:
            raise RuntimeError(f"non-positive manifest dimensions for {record_id}: {width}x{height}")
        minimum_width = width if minimum_width is None else min(minimum_width, width)
        minimum_height = height if minimum_height is None else min(minimum_height, height)
        if min(width, height) < crop_size:
            resized_width, resized_height = resized_pair_dimensions(width, height, crop_size)
            if min(resized_width, resized_height) < crop_size:
                raise RuntimeError(f"resize policy leaves an undersized pair: {record_id}")
            undersized.append(row)
            categories[str(row["category"])] += 1
            splits[str(row["split"])] += 1
    ids = sorted(str(row["record_id"]) for row in undersized)
    return {
        "records": len(rows),
        "crop_size": crop_size,
        "minimum_width": minimum_width,
        "minimum_height": minimum_height,
        "undersized_records": len(undersized),
        "undersized_record_ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
        "undersized_categories": dict(sorted(categories.items())),
        "undersized_splits": dict(sorted(splits.items())),
        "training_undersized_pair_policy": TRAINING_UNDERSIZED_PAIR_POLICY,
    }


def verify_undersized_pairs(
    rows: Sequence[Mapping[str, str]], crop_size: int
) -> int:
    verified = 0
    for row in rows:
        width, height = int(row["width"]), int(row["height"])
        if min(width, height) >= crop_size:
            continue
        observed = []
        for key in ("lq_path", "gt_path"):
            path = (ROOT / str(row[key])).resolve()
            path.relative_to(ROOT.resolve())
            with Image.open(path) as image:
                image.load()
                observed.append(image.size)
        if observed != [(width, height), (width, height)]:
            raise RuntimeError(
                f"undersized pair geometry changed for {row['record_id']}: "
                f"manifest={(width, height)} actual={observed}"
            )
        verified += 1
    return verified


def run(config_path: Path) -> dict[str, object]:
    config_path = config_path.resolve()
    config_path.relative_to(ROOT.resolve())
    config = json.loads(config_path.read_text())
    manifest_path = (ROOT / config["runtime"]["manifest"]).resolve()
    manifest_path.relative_to(ROOT.resolve())
    manifest_summary_path = manifest_path.with_suffix(".json")
    manifest_summary = json.loads(manifest_summary_path.read_text())
    expected = int(config["data"]["expected_five_type_records"])
    manifest_sha = sha256(manifest_path)
    if (
        manifest_summary.get("manifest_sha256") != manifest_sha
        or int(manifest_summary.get("records", -1)) != expected
        or manifest_summary.get("official_test_accessed") is not False
    ):
        raise RuntimeError("frozen FoundIR training manifest identity changed")
    with manifest_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    required = {
        "record_id", "split", "category", "width", "height", "lq_path", "gt_path"
    }
    if reader.fieldnames is None or not required.issubset(reader.fieldnames):
        raise RuntimeError("training manifest geometry schema changed")
    if len(rows) != expected:
        raise RuntimeError(f"training manifest cardinality changed: {len(rows)} != {expected}")
    payload = geometry_summary(rows, int(config["data"]["crop_size"]))
    payload.update(
        schema_version=1,
        status="pass",
        manifest=manifest_path.relative_to(ROOT).as_posix(),
        manifest_sha256=manifest_sha,
        actual_undersized_pairs_verified=verify_undersized_pairs(
            rows, int(config["data"]["crop_size"])
        ),
        protected_foundir_test_accessed=False,
    )
    print("FOUNDIR_FLUX_TRAINING_GEOMETRY_PREFLIGHT_PASS " + json.dumps(payload, sort_keys=True))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
