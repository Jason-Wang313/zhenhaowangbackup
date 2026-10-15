#!/usr/bin/env python3
"""Freeze source-disjoint double-only stage trajectories for the tiny gate."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ENDPOINT = ROOT / "manifests/foundir_composition_whole_image_v1.csv"
NOISE_MANIFEST = ROOT / "manifests/foundir_noise_family_integrated_development_v1.csv"
NOISE_ROOT = ROOT / "checkpoints/foundir_noise_family_integrated_development_v1"
BLUR_MANIFEST = ROOT / "manifests/foundir_final_eleven_all_in_one_development_rehearsal_v3.csv"
BLUR_ROOT = ROOT / "checkpoints/foundir_final_eleven_all_in_one_development_rehearsal_v3"
OUTPUT = ROOT / "manifests/foundir_stage_trajectory_compatibility_v1.csv"
METADATA = OUTPUT.with_suffix(".json")
SEED = 2026092701


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def order_key(*parts: object) -> str:
    return hashlib.sha256("\0".join(map(str, parts)).encode()).hexdigest()


def collect(source_manifest: Path, artifact_root: Path, category: str, stages: tuple[str, ...], excluded: set[str]) -> list[dict[str, str]]:
    selected = []
    with source_manifest.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["category"] != category or row["source_group"] in excluded:
                continue
            stage_paths = {stage: artifact_root / "stages" / stage / f"{row['record_id']}.png" for stage in stages}
            sidecar = artifact_root / "per_record" / f"{row['record_id']}.json"
            if not sidecar.is_file() or not all(path.is_file() for path in stage_paths.values()):
                continue
            evidence = json.loads(sidecar.read_text())
            crop = evidence.get("crop_xy")
            if not isinstance(crop, list) or len(crop) != 2:
                raise RuntimeError(f"missing crop coordinates: {row['record_id']}")
            item = {
                "record_id": row["record_id"],
                "source_group": row["source_group"],
                "category": category,
                "lq_path": row["lq_path"],
                "gt_path": row["gt_path"],
                "lq_sha256": row["lq_sha256"],
                "gt_sha256": row["gt_sha256"],
                "crop_x": str(int(crop[0])),
                "crop_y": str(int(crop[1])),
                "after_noise_path": "",
                "after_low_light_path": "",
                "after_blur_path": "",
                "presence_noise": "1" if "noise" in stages else "0",
                "presence_low_light": "1",
                "presence_blur": "1" if "blur" in stages else "0",
                "artifact_root": artifact_root.relative_to(ROOT).as_posix(),
            }
            for stage, path in stage_paths.items():
                item[f"after_{stage}_path"] = path.relative_to(ROOT).as_posix()
            selected.append(item)
    return selected


def main() -> None:
    with ENDPOINT.open(newline="", encoding="utf-8") as handle:
        endpoint_rows = list(csv.DictReader(handle))
    endpoint_groups = {row["source_group"] for row in endpoint_rows}
    if len(endpoint_rows) != 660 or len(endpoint_groups) != 221:
        raise RuntimeError("frozen full-resolution endpoint identity changed")

    records = collect(NOISE_MANIFEST, NOISE_ROOT, "16Lowlight_Noise", ("noise", "low_light"), endpoint_groups)
    records += collect(BLUR_MANIFEST, BLUR_ROOT, "15Lowlight_Blur", ("low_light", "blur"), endpoint_groups)
    expected = {"16Lowlight_Noise": 46, "15Lowlight_Blur": 62}
    observed = {category: sum(row["category"] == category for row in records) for category in expected}
    if observed != expected:
        raise RuntimeError(f"eligible trajectory cardinality changed: {observed}")

    groups_by_category = {
        category: {row["source_group"] for row in records if row["category"] == category}
        for category in expected
    }
    validation_groups = set()
    for category, groups in groups_by_category.items():
        count = 2 if category == "16Lowlight_Noise" else 11
        validation_groups.update(sorted(groups, key=lambda group: order_key(SEED, category, group))[:count])

    for row in records:
        row["split"] = "validation" if row["source_group"] in validation_groups else "train"
    train_groups = {row["source_group"] for row in records if row["split"] == "train"}
    if train_groups & validation_groups or (train_groups | validation_groups) & endpoint_groups:
        raise RuntimeError("trajectory source-group isolation failed")
    counts = {
        split: {category: sum(row["split"] == split and row["category"] == category for row in records) for category in expected}
        for split in ("train", "validation")
    }
    if any(counts[split][category] == 0 for split in counts for category in expected):
        raise RuntimeError("each split must contain both double categories")

    fields = [
        "record_id", "split", "source_group", "category", "lq_path", "gt_path", "lq_sha256", "gt_sha256",
        "crop_x", "crop_y", "after_noise_path", "after_low_light_path", "after_blur_path",
        "presence_noise", "presence_low_light", "presence_blur", "artifact_root",
    ]
    if OUTPUT.exists() or METADATA.exists():
        raise FileExistsError("refusing to overwrite frozen trajectory manifest")
    with OUTPUT.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted(records, key=lambda row: (row["split"], row["category"], row["record_id"])))
    payload = {
        "schema_version": 1,
        "status": "frozen_double_only_stage_trajectory_compatibility_v1",
        "seed": SEED,
        "records": len(records),
        "split_category_records": counts,
        "split_source_groups": {"train": len(train_groups), "validation": len(validation_groups)},
        "endpoint_records": len(endpoint_rows),
        "excluded_endpoint_source_groups": len(endpoint_groups),
        "triple_records_in_gradients_or_selection": 0,
        "source_disjoint_train_validation_endpoint": True,
        "manifest": OUTPUT.relative_to(ROOT).as_posix(),
        "manifest_sha256": sha256(OUTPUT),
        "source_manifest_sha256": {"noise": sha256(NOISE_MANIFEST), "blur": sha256(BLUR_MANIFEST)},
        "artifact_roots": {"noise": NOISE_ROOT.relative_to(ROOT).as_posix(), "blur": BLUR_ROOT.relative_to(ROOT).as_posix()},
        "development_pixels_accessed": True,
        "official_foundir_test_accessed": False,
    }
    METADATA.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
