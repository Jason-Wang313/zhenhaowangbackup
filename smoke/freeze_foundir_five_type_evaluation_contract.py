#!/usr/bin/env python3
"""Freeze exact development folds and sealed internal records before training."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lowlight_restoration.flux_prompt_cache import file_sha256  # noqa: E402
from lowlight_restoration.foundir_flux2_protocol import (  # noqa: E402
    balanced_evaluation_subset,
    resolve_workspace_path,
)
from smoke.train_foundir_five_type_flux2_full_ddp import load_rows  # noqa: E402


FIELDS = (
    "evaluation_role",
    "evaluation_fold",
    "record_id",
    "source_group",
    "category",
    "composition_size",
    "lq_path",
    "gt_path",
    "lq_sha256",
    "gt_sha256",
)


def contract_rows(config: Mapping[str, object], rows: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    evaluation = config["evaluation"]
    selected = []
    for split, role in (("validation", "development"), ("internal_test", "sealed_internal_test")):
        values = balanced_evaluation_subset(
            rows,
            split=split,
            records_per_category_per_fold=int(evaluation["records_per_category_per_fold"]),
            folds=int(evaluation["folds"]),
            seed=int(config["data"]["split_seed"]),
        )
        selected.extend(
            {
                "evaluation_role": role,
                **{field: row[field] for field in FIELDS if field != "evaluation_role"},
            }
            for row in values
        )
    development_groups = {
        row["source_group"] for row in selected if row["evaluation_role"] == "development"
    }
    sealed_groups = {
        row["source_group"] for row in selected if row["evaluation_role"] == "sealed_internal_test"
    }
    if development_groups & sealed_groups:
        raise RuntimeError("evaluation contract leaks source groups across development/internal")
    return selected


def encode_csv(rows: Sequence[Mapping[str, str]]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(FIELDS), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode()


def publish_once(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise RuntimeError(f"refusing to overwrite differing evaluation contract: {path}")
        return
    with path.open("xb") as handle:
        handle.write(content)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--conditions", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    args.config = resolve_workspace_path(ROOT, args.config)
    args.conditions = resolve_workspace_path(ROOT, args.conditions)
    args.output_csv = resolve_workspace_path(ROOT, args.output_csv)
    args.output_json = resolve_workspace_path(ROOT, args.output_json)
    config = json.loads(args.config.read_text())
    rows = load_rows(args.conditions, int(config["data"]["expected_five_type_records"]))
    selected = contract_rows(config, rows)
    content = encode_csv(selected)
    roles = Counter(row["evaluation_role"] for row in selected)
    role_category_fold_counts = Counter(
        (row["evaluation_role"], row["category"], row["evaluation_fold"])
        for row in selected
    )
    payload = {
        "schema_version": 1,
        "status": "frozen_before_flux_full_training",
        "config_sha256": file_sha256(args.config),
        "conditions_sha256": file_sha256(args.conditions),
        "records": len(selected),
        "role_counts": dict(sorted(roles.items())),
        "folds": int(config["evaluation"]["folds"]),
        "records_per_category_per_fold": int(config["evaluation"]["records_per_category_per_fold"]),
        "contract_csv": args.output_csv.relative_to(ROOT).as_posix(),
        "contract_csv_sha256": hashlib.sha256(content).hexdigest(),
        "role_category_fold_counts": {
            "|".join(key): value for key, value in sorted(role_category_fold_counts.items())
        },
        "sealed_internal_test_accessed": False,
        "protected_foundir_test_accessed": False,
    }
    publish_once(args.output_csv, content)
    publish_once(args.output_json, (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode())
    print("FOUNDIR_FIVE_TYPE_EVALUATION_CONTRACT_PASS " + json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
