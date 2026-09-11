#!/usr/bin/env python3
"""Freeze source-group-distinct Blur-only tiny-train and future-validation rows."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
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
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable(seed: int, *values: object) -> int:
    text = ":".join([str(seed), *(str(value) for value in values)])
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")


def select(rows: list[dict[str, str]], *, split: str, count: int, seed: int) -> list[dict[str, str]]:
    candidates = [row for row in rows if row["split"] == split]
    candidates.sort(key=lambda row: (stable(seed, split, row["source_group"], row["record_id"]), row["record_id"]))
    selected: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in candidates:
        if row["source_group"] in seen:
            continue
        seen.add(row["source_group"])
        selected.append(row)
        if len(selected) == count:
            break
    if len(selected) != count:
        raise RuntimeError(f"insufficient distinct Blur source groups in {split}: {len(selected)} != {count}")
    return selected


def frozen_row(row: Mapping[str, str], *, role: str, crop_size: int, crop_seed: int) -> dict[str, object]:
    width, height = int(row["width"]), int(row["height"])
    if width < crop_size or height < crop_size:
        raise RuntimeError(f"selected pair is smaller than crop: {row['record_id']} {width}x{height}")
    x = stable(crop_seed, row["record_id"], "x") % (width - crop_size + 1)
    y = stable(crop_seed, row["record_id"], "y") % (height - crop_size + 1)
    return {
        "record_id": row["record_id"], "role": role, "split": row["split"],
        "source_group": row["source_group"], "category": row["category"],
        "composition_size": int(row["composition_size"]), "lq_path": row["lq_path"],
        "gt_path": row["gt_path"], "lq_sha256": row["lq_sha256"], "gt_sha256": row["gt_sha256"],
        "width": width, "height": height, "crop_x": x, "crop_y": y,
    }


def encoded_csv(rows: list[dict[str, object]]) -> bytes:
    import io
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode()


def write_or_verify(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise RuntimeError(f"refusing to overwrite differing frozen artifact: {path}")
        return
    path.write_bytes(content)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/foundir_blur_only_flux2_sanity_v1.json")
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())
    data = config["data"]
    source = ROOT / data["source_manifest"]
    if sha256(source) != data["source_manifest_sha256"]:
        raise RuntimeError("FoundIR development manifest identity changed")
    with source.open(newline="") as handle:
        all_rows = list(csv.DictReader(handle))
    blur = [row for row in all_rows if row["category"] == data["category"]]
    if any(row["composition_size"] != "1" for row in blur):
        raise RuntimeError("01Blur unexpectedly contains mixed degradation")
    train = select(blur, split=data["tiny_overfit_split"], count=int(data["tiny_overfit_records"]), seed=int(data["selection_seed"]))
    validation = select(blur, split=data["frozen_validation_split"], count=int(data["frozen_validation_records"]), seed=int(data["selection_seed"]))
    rows = [frozen_row(row, role="tiny_overfit_train", crop_size=int(data["crop_size"]), crop_seed=int(data["crop_seed"])) for row in train]
    rows += [frozen_row(row, role="future_blur_validation", crop_size=int(data["crop_size"]), crop_seed=int(data["crop_seed"])) for row in validation]
    train_groups = {row["source_group"] for row in rows if row["role"] == "tiny_overfit_train"}
    validation_groups = {row["source_group"] for row in rows if row["role"] == "future_blur_validation"}
    if train_groups & validation_groups:
        raise RuntimeError("train/validation source-group leakage")
    csv_bytes = encoded_csv(rows)
    csv_path = ROOT / data["frozen_manifest"]
    write_or_verify(csv_path, csv_bytes)
    metadata = {
        "schema_version": 1, "status": "frozen_blur_only_sanity_manifest",
        "config": config_path.relative_to(ROOT).as_posix(), "config_sha256": sha256(config_path),
        "source_manifest": data["source_manifest"], "source_manifest_sha256": sha256(source),
        "frozen_manifest": data["frozen_manifest"], "frozen_manifest_sha256": hashlib.sha256(csv_bytes).hexdigest(),
        "category": data["category"], "tiny_overfit_records": len(train),
        "tiny_overfit_unique_source_groups": len(train_groups), "future_validation_records": len(validation),
        "future_validation_unique_source_groups": len(validation_groups),
        "train_validation_source_group_overlap": 0, "protected_foundir_test_accessed": False,
        "sealed_internal_accessed": False,
    }
    json_bytes = (json.dumps(metadata, indent=2, sort_keys=True) + "\n").encode()
    write_or_verify(ROOT / data["frozen_manifest_metadata"], json_bytes)
    print("FOUNDIR_BLUR_ONLY_SANITY_MANIFEST_PASS " + json.dumps(metadata, sort_keys=True))


if __name__ == "__main__":
    main()
