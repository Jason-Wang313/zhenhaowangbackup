#!/usr/bin/env python3
"""Export frozen Joint-Qwen conditions for FoundIR training images only."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lowlight_restoration.foundir_five_type import (  # noqa: E402
    DEGRADATION_ORDER,
    FIVE_TYPE_CATEGORIES,
    canonical_restoration_text,
)
from lowlight_restoration.qwen_lora import load_qwen_adapter  # noqa: E402
from lowlight_restoration.qwen_multidegradation import (  # noqa: E402
    FIXED_FIVE_SLOT_SCHEMA,
    canonical_fixed_five_slot_target,
    parse_output,
)
from smoke.train_qwen_multidegradation_lora_ddp import (  # noqa: E402
    generate_outputs,
    load_model,
    seed_everything,
    setup_distributed,
    validate_runtime,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_key(seed: int, *values: str) -> str:
    return hashlib.sha256((str(seed) + ":" + ":".join(values)).encode()).hexdigest()


def select_balanced_pilot(
    rows: Sequence[Mapping[str, str]], *, count: int, seed: int
) -> list[dict[str, str]]:
    """Select a deterministic category-covered training pilot."""

    candidates = [dict(row) for row in rows if row["split"] == "train"]
    if count < len(FIVE_TYPE_CATEGORIES) or count > len(candidates):
        raise ValueError("pilot count cannot cover every FoundIR category")
    by_category: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in candidates:
        by_category[row["category"]].append(row)
    if set(by_category) != set(FIVE_TYPE_CATEGORIES):
        raise RuntimeError("training manifest does not cover all five-type categories")
    for category, values in by_category.items():
        values.sort(key=lambda row: stable_key(seed, category, row["record_id"]))
    selected: list[dict[str, str]] = []
    depth = 0
    while len(selected) < count:
        progressed = False
        for category in sorted(by_category):
            values = by_category[category]
            if depth < len(values):
                selected.append(values[depth])
                progressed = True
                if len(selected) == count:
                    break
        if not progressed:
            raise RuntimeError("pilot selection exhausted before requested count")
        depth += 1
    return selected


def condition_row(record: Mapping[str, str], raw_output: str) -> dict[str, str]:
    label = parse_output(raw_output, FIXED_FIVE_SLOT_SCHEMA)
    predicted = {name: label.level_map.get(name, 1) for name in DEGRADATION_ORDER}
    oracle = {
        name: (
            max(2, predicted[name])
            if name in FIVE_TYPE_CATEGORIES[str(record["category"])]
            else 1
        )
        for name in DEGRADATION_ORDER
    }
    generic = {name: 1 for name in DEGRADATION_ORDER}
    return {
        "record_id": str(record["record_id"]),
        "split": str(record["split"]),
        "source_group": str(record["source_group"]),
        "category": str(record["category"]),
        "composition_size": str(record["composition_size"]),
        "lq_path": str(record["lq_path"]),
        "gt_path": str(record["gt_path"]),
        "lq_sha256": str(record["lq_sha256"]),
        "gt_sha256": str(record["gt_sha256"]),
        "qwen_raw_output": raw_output,
        "qwen_json": canonical_fixed_five_slot_target(label),
        "qwen_levels_json": json.dumps(predicted, separators=(",", ":")),
        "qwen_text": canonical_restoration_text(predicted),
        "type_oracle_levels_json": json.dumps(oracle, separators=(",", ":")),
        "type_oracle_text": canonical_restoration_text(oracle),
        "generic_text": canonical_restoration_text(generic),
    }


def encode_csv(rows: Sequence[Mapping[str, str]]) -> bytes:
    if not rows:
        raise ValueError("condition cache cannot be empty")
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode()


def write_once(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise RuntimeError(f"refusing to overwrite differing condition artifact: {path}")
        return
    with path.open("xb") as handle:
        handle.write(content)


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    required = {
        "record_id", "split", "source_group", "category", "composition_size",
        "lq_path", "gt_path", "lq_sha256", "gt_sha256",
    }
    if reader.fieldnames is None or not required.issubset(reader.fieldnames):
        raise RuntimeError("FoundIR five-type manifest schema changed")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--qwen-config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--records", type=int, default=128)
    parser.add_argument("--all-train", action="store_true")
    parser.add_argument("--verify-existing", action="store_true")
    args = parser.parse_args()
    restoration_config = json.loads(args.config.read_text())
    qwen_config = json.loads(args.qwen_config.read_text())
    manifest_rows = load_manifest(args.manifest)
    seed = int(restoration_config["data"]["split_seed"])
    selected = (
        [row for row in manifest_rows if row["split"] == "train"]
        if args.all_train
        else select_balanced_pilot(manifest_rows, count=args.records, seed=seed)
    )
    if sha256(args.qwen_config) != restoration_config["qwen_condition"]["training_config_sha256"]:
        raise RuntimeError("frozen Joint-Qwen config changed")
    adapter = ROOT / restoration_config["qwen_condition"]["adapter"]
    if sha256(adapter) != restoration_config["qwen_condition"]["adapter_sha256"]:
        raise RuntimeError("frozen Joint-Qwen adapter changed")
    if qwen_config["prompt"]["target_schema"] != FIXED_FIVE_SLOT_SCHEMA:
        raise RuntimeError("Joint-Qwen output schema changed")
    if args.verify_existing:
        payload = json.loads(args.output_json.read_text())
        expected_scope = "all_train" if args.all_train else "balanced_overfit_pilot"
        if (
            payload.get("status") != "frozen_joint_qwen_foundir_training_conditions"
            or payload.get("scope") != expected_scope
            or int(payload.get("records", -1)) != len(selected)
            or payload.get("manifest_sha256") != sha256(args.manifest)
            or payload.get("qwen_config_sha256") != sha256(args.qwen_config)
            or payload.get("qwen_adapter_sha256") != sha256(adapter)
            or payload.get("conditions_csv_sha256") != sha256(args.output_csv)
            or payload.get("protected_foundir_test_accessed") is not False
        ):
            raise RuntimeError("existing Qwen condition cache identity changed")
        print("FOUNDIR_FIVE_TYPE_QWEN_CONDITIONS_VERIFY_PASS " + json.dumps(payload, sort_keys=True))
        return

    rank = -1
    initialized = False
    try:
        rank, world, _, device = setup_distributed(qwen_config)
        initialized = True
        if world != int(restoration_config["distributed"]["required_world_size"]):
            raise RuntimeError("condition export requires exactly eight GPUs")
        validate_runtime(qwen_config, device.index or 0)
        os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        seed_everything(seed)
        model, processor, _ = load_model(qwen_config, variant="merger_trainable", device=device)
        load_qwen_adapter(model, adapter)
        generation_records = [{**row, "output_path": row["lq_path"]} for row in selected]
        raw_outputs = generate_outputs(
            model,
            processor,
            generation_records,
            qwen_config,
            rank=rank,
            world=world,
            device=device,
        )
        if rank == 0:
            rows = [condition_row(record, raw) for record, raw in zip(selected, raw_outputs, strict=True)]
            csv_content = encode_csv(rows)
            write_once(args.output_csv, csv_content)
            counts: dict[str, int] = defaultdict(int)
            for row in rows:
                counts[row["category"]] += 1
            payload = {
                "schema_version": 1,
                "status": "frozen_joint_qwen_foundir_training_conditions",
                "scope": "all_train" if args.all_train else "balanced_overfit_pilot",
                "records": len(rows),
                "category_counts": dict(sorted(counts.items())),
                "manifest": args.manifest.relative_to(ROOT).as_posix(),
                "manifest_sha256": sha256(args.manifest),
                "qwen_config": args.qwen_config.relative_to(ROOT).as_posix(),
                "qwen_config_sha256": sha256(args.qwen_config),
                "qwen_adapter": adapter.relative_to(ROOT).as_posix(),
                "qwen_adapter_sha256": sha256(adapter),
                "conditions_csv": args.output_csv.relative_to(ROOT).as_posix(),
                "conditions_csv_sha256": hashlib.sha256(csv_content).hexdigest(),
                "protected_foundir_test_accessed": False,
            }
            write_once(args.output_json, (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode())
            print("FOUNDIR_FIVE_TYPE_QWEN_CONDITIONS_PASS " + json.dumps(payload, sort_keys=True), flush=True)
        dist.barrier()
    finally:
        if initialized and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
