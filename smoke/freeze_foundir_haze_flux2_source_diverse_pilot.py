#!/usr/bin/env python3
"""Freeze the user-authorized source-diverse Haze train/validation pilot."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from collections import defaultdict
from pathlib import Path


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


def select_rows(config: dict) -> tuple[list[dict[str, str]], dict[str, int]]:
    data = config["data"]
    source_path = ROOT / data["source_manifest"]
    if sha256(source_path) != data["source_manifest_sha256"]:
        raise RuntimeError("source manifest identity changed")
    with source_path.open(newline="") as handle:
        source_rows = list(csv.DictReader(handle))
    excluded_groups: dict[str, set[str]] = defaultdict(set)
    excluded_path = ROOT / data["excluded_manifest"]
    with excluded_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["category"] == data["category"]:
                excluded_groups[row["split"]].add(row["source_group"])

    selected: list[dict[str, str]] = []
    available = {}
    specs = (
        (data["train_role"], data["tiny_overfit_split"], int(data["tiny_overfit_records"])),
        (data["validation_role"], data["frozen_validation_split"], int(data["frozen_validation_records"])),
    )
    for role, split, count in specs:
        groups: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in source_rows:
            if (
                row["category"] == data["category"]
                and row["composition_size"] == "1"
                and row["split"] == split
                and row["source_group"] not in excluded_groups[split]
                and int(row["width"]) >= int(data["crop_size"])
                and int(row["height"]) >= int(data["crop_size"])
            ):
                groups[row["source_group"]].append(row)
        available[role] = len(groups)
        ordered_groups = sorted(groups, key=lambda group: stable_int(data["selection_seed"], role, group))
        if len(ordered_groups) < count:
            raise RuntimeError(f"insufficient source groups for {role}: {len(ordered_groups)} < {count}")
        for group in ordered_groups[:count]:
            row = min(groups[group], key=lambda value: stable_int(data["selection_seed"], role, value["record_id"]))
            width, height, crop = int(row["width"]), int(row["height"]), int(data["crop_size"])
            x = stable_int(data["crop_seed"], role, row["record_id"], "x") % (width - crop + 1)
            y = stable_int(data["crop_seed"], role, row["record_id"], "y") % (height - crop + 1)
            selected.append({
                field: str({**row, "role": role, "crop_x": x, "crop_y": y}[field])
                for field in FIELDS
            })
    return selected, available


def render_csv(rows: list[dict[str, str]]) -> bytes:
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
    if config.get("status") != "user_authorized_haze_source_diverse_pilot":
        raise RuntimeError("wrong pilot authorization status")
    rows, available = select_rows(config)
    data = config["data"]
    manifest = ROOT / data["frozen_manifest"]
    content = render_csv(rows)
    write_once(manifest, content)
    train = [row for row in rows if row["role"] == data["train_role"]]
    validation = [row for row in rows if row["role"] == data["validation_role"]]
    overlap = {row["source_group"] for row in train} & {row["source_group"] for row in validation}
    metadata = {
        "schema_version": 1,
        "status": "frozen_haze_source_diverse_pilot_manifest",
        "config": config_path.relative_to(ROOT).as_posix(),
        "config_sha256": sha256(config_path),
        "source_manifest": data["source_manifest"],
        "source_manifest_sha256": data["source_manifest_sha256"],
        "frozen_manifest": data["frozen_manifest"],
        "frozen_manifest_sha256": hashlib.sha256(content).hexdigest(),
        "records": len(rows),
        "train_records": len(train),
        "train_source_groups": len({row["source_group"] for row in train}),
        "validation_records": len(validation),
        "validation_source_groups": len({row["source_group"] for row in validation}),
        "train_validation_source_group_overlap": len(overlap),
        "available_source_groups_after_exclusion": available,
        "excluded_prior_capacity_groups": True,
        "qwen_degradation_conditioning": False,
        "mixed_degradations": False,
        "sealed_internal_accessed": False,
        "official_foundir_test_accessed": False,
        "protected_foundir_test_accessed": False,
    }
    write_once(ROOT / data["frozen_manifest_metadata"], (json.dumps(metadata, indent=2, sort_keys=True) + "\n").encode())
    print("FOUNDIR_HAZE_SOURCE_DIVERSE_MANIFEST_PASS " + json.dumps(metadata, sort_keys=True))


if __name__ == "__main__":
    main()
