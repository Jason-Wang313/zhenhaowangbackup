#!/usr/bin/env python3
"""Freeze the source-diverse isolated-Blur pilot without reading protected data."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "manifests/foundir_five_type_pairs_v1.csv"
SANITY = ROOT / "manifests/foundir_blur_only_sanity_v1.csv"
OUTPUT = ROOT / "manifests/foundir_blur_only_flux2_source_diverse_pilot_v1.csv"
METADATA = OUTPUT.with_suffix(".json")
SOURCE_SHA256 = "e8c018879029c0bcc2da8d074cc4999541158ca97d07f48aadbcdddcdccc3ca7"
SANITY_SHA256 = "7114dab36b3978f28657f366e9a961462c3dbfb06927c9262754cced0d8c8e6f"
GROUP_SEED = "flux2-blur-source-diverse-pilot-v1-groups-20260916"
RECORD_SEED = "flux2-blur-source-diverse-pilot-v1-records-20260916"
CROP_SEED = "flux2-blur-source-diverse-pilot-v1-crops-20260916"
COUNT = 512
CROP = 512
FIELDS = [
    "record_id", "role", "split", "source_group", "category", "composition_size",
    "lq_path", "gt_path", "lq_sha256", "gt_sha256", "width", "height",
    "crop_x", "crop_y", "selection_rank", "visual_train",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_digest(seed: str, *parts: str) -> bytes:
    return hashlib.sha256("\0".join((seed, *parts)).encode()).digest()


def main() -> int:
    if sha256(SOURCE) != SOURCE_SHA256 or sha256(SANITY) != SANITY_SHA256:
        raise RuntimeError("source manifest identity changed")
    with SANITY.open(newline="") as handle:
        sanity = list(csv.DictReader(handle))
    excluded = {row["source_group"] for row in sanity}
    old_train = {row["source_group"] for row in sanity if row["role"] == "tiny_overfit_train"}
    development = [row for row in sanity if row["role"] == "future_blur_validation"]
    development_groups = {row["source_group"] for row in development}
    if len(old_train) != 16 or len(development_groups) != 16 or len(excluded) != 32:
        raise RuntimeError("frozen sanity exclusion contract changed")

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    with SOURCE.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if (
                row["split"] == "train"
                and row["category"] == "01Blur"
                and row["composition_size"] == "1"
                and row["source_group"] not in excluded
            ):
                grouped[row["source_group"]].append(row)
    ordered_groups = sorted(grouped, key=lambda group: stable_digest(GROUP_SEED, group))[:COUNT]
    if len(ordered_groups) != COUNT:
        raise RuntimeError("not enough eligible source groups")

    rows: list[dict[str, object]] = []
    for rank, group in enumerate(ordered_groups):
        chosen = min(grouped[group], key=lambda row: stable_digest(RECORD_SEED, group, row["record_id"]))
        width, height = int(chosen["width"]), int(chosen["height"])
        if width < CROP or height < CROP:
            raise RuntimeError(f"record is smaller than crop: {chosen['record_id']}")
        crop_hash = stable_digest(CROP_SEED, group, chosen["record_id"])
        x = int.from_bytes(crop_hash[:8], "big") % (width - CROP + 1)
        y = int.from_bytes(crop_hash[8:16], "big") % (height - CROP + 1)
        rows.append({
            **{key: chosen[key] for key in (
                "record_id", "split", "source_group", "category", "composition_size",
                "lq_path", "gt_path", "lq_sha256", "gt_sha256", "width", "height",
            )},
            "role": "pilot_train",
            "crop_x": x,
            "crop_y": y,
            "selection_rank": rank,
            "visual_train": str(rank < 16).lower(),
        })
    for rank, source in enumerate(development):
        rows.append({
            **{key: source[key] for key in (
                "record_id", "split", "source_group", "category", "composition_size",
                "lq_path", "gt_path", "lq_sha256", "gt_sha256", "width", "height",
                "crop_x", "crop_y",
            )},
            "role": "fixed_development",
            "selection_rank": rank,
            "visual_train": "false",
        })
    if {row["source_group"] for row in rows[:COUNT]} & excluded:
        raise RuntimeError("pilot source exclusion failed")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    output_hash = sha256(OUTPUT)
    metadata = {
        "schema_version": 1,
        "status": "frozen_flux2_blur_source_diverse_pilot_manifest",
        "source_manifest": SOURCE.relative_to(ROOT).as_posix(),
        "source_manifest_sha256": SOURCE_SHA256,
        "excluded_sanity_manifest": SANITY.relative_to(ROOT).as_posix(),
        "excluded_sanity_manifest_sha256": SANITY_SHA256,
        "frozen_manifest": OUTPUT.relative_to(ROOT).as_posix(),
        "frozen_manifest_sha256": output_hash,
        "selection": {
            "method": "ascending_sha256_seeded_hash",
            "group_seed": GROUP_SEED,
            "record_seed": RECORD_SEED,
            "crop_seed": CROP_SEED,
            "one_record_per_source_group": True,
            "crop_size": CROP,
        },
        "pilot_train_records": COUNT,
        "pilot_train_unique_source_groups": COUNT,
        "visual_train_records": 16,
        "fixed_development_records": len(development),
        "fixed_development_unique_source_groups": len(development_groups),
        "excluded_old_tiny_train_groups": len(old_train),
        "excluded_fixed_development_groups": len(development_groups),
        "train_development_source_group_overlap": 0,
        "sealed_internal_accessed": False,
        "official_foundir_test_accessed": False,
    }
    METADATA.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metadata, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
