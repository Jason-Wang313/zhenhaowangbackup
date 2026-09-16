#!/usr/bin/env python3
"""Freeze deterministic source-disjoint records/crops for four capacity arms."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Mapping


ROOT = Path(__file__).resolve().parents[1]
FIELDS = (
    "record_id", "role", "split", "source_group", "category", "composition_size",
    "lq_path", "gt_path", "lq_sha256", "gt_sha256", "width", "height", "crop_x", "crop_y",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_int(*values: object) -> int:
    return int.from_bytes(hashlib.sha256("\0".join(map(str, values)).encode()).digest()[:8], "big")


def select_rows(config: Mapping[str, object]) -> list[dict[str, str]]:
    data = config["data"]
    source = ROOT / str(data["source_manifest"])
    if sha256(source) != data["source_manifest_sha256"]:
        raise RuntimeError("source manifest identity changed")
    with source.open(newline="") as handle:
        source_rows = list(csv.DictReader(handle))
    result: list[dict[str, str]] = []
    specs = (
        ("tiny_overfit_train", str(data["tiny_overfit_split"]), int(data["tiny_overfit_records_per_task"])),
        ("descriptive_validation", str(data["frozen_validation_split"]), int(data["frozen_validation_records_per_task"])),
    )
    for task_index, category in enumerate(config["scope"]["categories"]):
        for role_index, (role, split, count) in enumerate(specs):
            candidates = [
                row for row in source_rows
                if row["category"] == category and row["composition_size"] == "1" and row["split"] == split
            ]
            groups: dict[str, list[dict[str, str]]] = defaultdict(list)
            for row in candidates:
                if int(row["width"]) >= int(data["crop_size"]) and int(row["height"]) >= int(data["crop_size"]):
                    groups[row["source_group"]].append(row)
            ordered_groups = sorted(
                groups,
                key=lambda group: stable_int(data["selection_seed"], task_index, role_index, group),
            )
            if len(ordered_groups) < count:
                raise RuntimeError(f"insufficient unique groups for {category} {role}")
            for group in ordered_groups[:count]:
                row = min(
                    groups[group],
                    key=lambda item: stable_int(data["selection_seed"], category, role, item["record_id"]),
                )
                width, height, crop = int(row["width"]), int(row["height"]), int(data["crop_size"])
                x = stable_int(data["crop_seed"], category, role, row["record_id"], "x") % (width - crop + 1)
                y = stable_int(data["crop_seed"], category, role, row["record_id"], "y") % (height - crop + 1)
                result.append({
                    key: str(value) for key, value in {
                        **{field: row[field] for field in FIELDS if field in row},
                        "role": role,
                        "crop_x": x,
                        "crop_y": y,
                    }.items()
                })
    return result


def render_csv(rows: list[dict[str, str]]) -> bytes:
    import io
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode()


def write_once(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != content:
        raise RuntimeError(f"refusing to overwrite differing frozen artifact: {path}")
    if not path.exists():
        path.write_bytes(content)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())
    rows = select_rows(config)
    manifest = ROOT / config["data"]["frozen_manifest"]
    metadata_path = ROOT / config["data"]["frozen_manifest_metadata"]
    content = render_csv(rows)
    write_once(manifest, content)
    tasks = {}
    for category in config["scope"]["categories"]:
        train = [row for row in rows if row["category"] == category and row["role"] == "tiny_overfit_train"]
        validation = [row for row in rows if row["category"] == category and row["role"] == "descriptive_validation"]
        tasks[category] = {
            "train_records": len(train),
            "train_source_groups": len({row["source_group"] for row in train}),
            "validation_records": len(validation),
            "validation_source_groups": len({row["source_group"] for row in validation}),
            "train_validation_source_group_overlap": len(
                {row["source_group"] for row in train} & {row["source_group"] for row in validation}
            ),
        }
    metadata = {
        "schema_version": 1,
        "status": "frozen_four_task_capacity_sanity_manifest",
        "config": config_path.relative_to(ROOT).as_posix(),
        "config_sha256": sha256(config_path),
        "source_manifest": config["data"]["source_manifest"],
        "source_manifest_sha256": config["data"]["source_manifest_sha256"],
        "frozen_manifest": manifest.relative_to(ROOT).as_posix(),
        "frozen_manifest_sha256": hashlib.sha256(content).hexdigest(),
        "records": len(rows),
        "tasks": tasks,
        "qwen_degradation_conditioning": False,
        "mixed_degradations": False,
        "sealed_internal_accessed": False,
        "official_foundir_test_accessed": False,
        "protected_foundir_test_accessed": False,
    }
    write_once(metadata_path, (json.dumps(metadata, indent=2, sort_keys=True) + "\n").encode())
    print("FOUNDIR_FOUR_TASK_CAPACITY_MANIFEST_PASS " + json.dumps({"records": len(rows), "tasks": tasks}, sort_keys=True))


if __name__ == "__main__":
    main()
