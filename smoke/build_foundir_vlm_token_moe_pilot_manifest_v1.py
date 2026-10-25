#!/usr/bin/env python3
"""Join the frozen training-only VLM features to paired FoundIR records."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HEAD = ROOT / "manifests/foundir_qwen_five_class_train_only_v1.csv"
CONDITIONS = ROOT / "manifests/foundir_five_type_all_qwen_conditions_v3.csv"
FEATURES = ROOT / "checkpoints/foundir_qwen_five_class_gate_v1/features"
OUTPUT = ROOT / "manifests/foundir_vlm_token_moe_pilot_v1.csv"
METADATA = OUTPUT.with_suffix(".json")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_once(path: Path, data: bytes) -> None:
    if path.exists():
        if path.read_bytes() != data:
            raise RuntimeError(f"refusing to overwrite differing artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(data)


def main() -> None:
    expected = {
        HEAD: "af8798fab6ea340ba4589684ac02997b0cb01fe88a56a78bafcd92e75f36fc6c",
        CONDITIONS: "e07e16234d4d1eb674b5d667732600628448b215e9fab67e6e6732f99c1ea2f7",
    }
    for path, digest in expected.items():
        if not path.is_file() or sha256(path) != digest:
            raise RuntimeError(f"frozen input changed: {path}")

    with HEAD.open(newline="") as handle:
        head_rows = list(csv.DictReader(handle))
    if len(head_rows) != 3150 or len({row["record_id"] for row in head_rows}) != 3150:
        raise RuntimeError("training-only head manifest cardinality changed")
    wanted = {row["record_id"] for row in head_rows}

    condition_rows: dict[str, dict[str, str]] = {}
    with CONDITIONS.open(newline="") as handle:
        for row in csv.DictReader(handle):
            record_id = row["record_id"]
            if record_id in wanted:
                if record_id in condition_rows:
                    raise RuntimeError(f"duplicate condition record: {record_id}")
                condition_rows[record_id] = row
    if set(condition_rows) != wanted:
        raise RuntimeError("not every training-only VLM record has a paired FoundIR condition row")

    rows: list[dict[str, str]] = []
    groups_by_split: dict[str, set[str]] = defaultdict(set)
    for head in sorted(head_rows, key=lambda row: row["record_id"]):
        condition = condition_rows[head["record_id"]]
        if condition["split"] != "train" or int(condition["composition_size"]) != 1:
            raise RuntimeError("pilot input escaped the training-only singleton scope")
        if condition["source_group"] != head["source_group"]:
            raise RuntimeError("source-group identity changed during manifest join")
        feature = FEATURES / f"{head['record_id']}.safetensors"
        if not feature.is_file():
            raise RuntimeError(f"missing frozen VLM feature: {feature}")
        groups_by_split[head["head_split"]].add(head["source_group"])
        rows.append(
            {
                "record_id": head["record_id"],
                "role": head["head_split"],
                "label": head["label"],
                "source_group": head["source_group"],
                "lq_path": condition["lq_path"],
                "lq_sha256": condition["lq_sha256"],
                "gt_path": condition["gt_path"],
                "gt_sha256": condition["gt_sha256"],
                "vlm_feature_path": feature.relative_to(ROOT).as_posix(),
                "generic_text": condition["generic_text"],
            }
        )

    roles = {row["role"] for row in rows}
    if roles != {"fit", "calibration", "holdout"}:
        raise RuntimeError(f"pilot roles changed: {roles}")
    for left in roles:
        for right in roles:
            if left < right and groups_by_split[left] & groups_by_split[right]:
                raise RuntimeError(f"source-group leakage between {left} and {right}")
    counts = Counter((row["role"], row["label"]) for row in rows)
    expected_counts = {
        (role, label): count
        for role, count in (("fit", 450), ("calibration", 90), ("holdout", 90))
        for label in ("blur", "noise", "haze", "rain", "low_light")
    }
    if counts != expected_counts:
        raise RuntimeError(f"pilot class/split counts changed: {counts}")

    fieldnames = list(rows[0])
    from io import StringIO

    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    content = buffer.getvalue().encode()
    write_once(OUTPUT, content)
    metadata = {
        "schema_version": 1,
        "status": "foundir_vlm_token_moe_training_only_pilot_manifest",
        "records": len(rows),
        "role_counts": dict(Counter(row["role"] for row in rows)),
        "label_counts": dict(Counter(row["label"] for row in rows)),
        "source_groups_by_role": {key: len(value) for key, value in groups_by_split.items()},
        "manifest_sha256": hashlib.sha256(content).hexdigest(),
        "head_manifest_sha256": expected[HEAD],
        "conditions_manifest_sha256": expected[CONDITIONS],
        "vlm_feature_dim": 7168,
        "official_foundir_test_accessed": False,
        "sealed_internal_accessed": False,
        "parent_test_accessed": False,
        "internal_test_accessed": False,
    }
    write_once(METADATA, (json.dumps(metadata, indent=2, sort_keys=True) + "\n").encode())
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
