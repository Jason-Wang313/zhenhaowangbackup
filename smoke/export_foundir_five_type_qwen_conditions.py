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
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import torch
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
    prompt_text,
)
from smoke.train_qwen_multidegradation_lora_ddp import (  # noqa: E402
    broadcast_rank0,
    load_model,
    open_images,
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


def raw_shard_identity(
    records: Sequence[Mapping[str, str]],
    *,
    manifest_sha256: str,
    qwen_config_sha256: str,
    adapter_sha256: str,
) -> str:
    """Bind resume shards to every input and frozen model artifact."""

    payload = {
        "manifest_sha256": manifest_sha256,
        "qwen_config_sha256": qwen_config_sha256,
        "adapter_sha256": adapter_sha256,
        "records": [
            [str(record["record_id"]), str(record["lq_sha256"])]
            for record in records
        ],
    }
    return hashlib.sha256(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()


def raw_shard_path(output_csv: Path, rank: int) -> Path:
    return output_csv.parent / f".{output_csv.name}.raw_shards" / f"rank_{rank:02d}.json"


def encode_raw_shard(
    rows: Sequence[tuple[int, Mapping[str, str], str]],
    *,
    identity: str,
    rank: int,
    world: int,
) -> bytes:
    payload = {
        "schema_version": 1,
        "identity": identity,
        "rank": rank,
        "world_size": world,
        "rows": [
            {
                "index": index,
                "record_id": str(record["record_id"]),
                "lq_sha256": str(record["lq_sha256"]),
                "raw_output": raw,
            }
            for index, record, raw in rows
        ],
    }
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()


def write_raw_shard_once(path: Path, content: bytes) -> None:
    """Publish a complete rank shard atomically and never replace it."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        if path.exists():
            if path.read_bytes() != content:
                raise RuntimeError(f"refusing to overwrite differing raw Qwen shard: {path}")
            return
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != content:
                raise RuntimeError(f"raw Qwen shard raced with differing content: {path}")
    finally:
        temporary.unlink(missing_ok=True)


def load_raw_shard(
    path: Path,
    assigned: Sequence[tuple[int, Mapping[str, str]]],
    *,
    identity: str,
    rank: int,
    world: int,
) -> list[tuple[int, str]]:
    payload = json.loads(path.read_text())
    if (
        payload.get("schema_version") != 1
        or payload.get("identity") != identity
        or payload.get("rank") != rank
        or payload.get("world_size") != world
    ):
        raise RuntimeError(f"raw Qwen shard identity changed: {path}")
    rows = payload.get("rows")
    if not isinstance(rows, list) or len(rows) != len(assigned):
        raise RuntimeError(f"raw Qwen shard cardinality changed: {path}")
    loaded: list[tuple[int, str]] = []
    for value, (expected_index, expected_record) in zip(rows, assigned, strict=True):
        if not isinstance(value, dict):
            raise RuntimeError(f"raw Qwen shard row is invalid: {path}")
        raw = value.get("raw_output")
        if (
            value.get("index") != expected_index
            or value.get("record_id") != expected_record["record_id"]
            or value.get("lq_sha256") != expected_record["lq_sha256"]
            or not isinstance(raw, str)
        ):
            raise RuntimeError(f"raw Qwen shard input mapping changed: {path}")
        loaded.append((expected_index, raw))
    return loaded


def load_all_raw_shards(
    output_csv: Path,
    records: Sequence[Mapping[str, str]],
    *,
    identity: str,
    world: int,
) -> list[str]:
    indexed: list[tuple[int, str]] = []
    for rank in range(world):
        assigned = [
            (index, record)
            for index, record in enumerate(records)
            if index % world == rank
        ]
        indexed.extend(
            load_raw_shard(
                raw_shard_path(output_csv, rank),
                assigned,
                identity=identity,
                rank=rank,
                world=world,
            )
        )
    if sorted(index for index, _ in indexed) != list(range(len(records))):
        raise RuntimeError("raw Qwen shards do not cover the selected records exactly once")
    return [raw for _, raw in sorted(indexed)]


def condition_rows_from_raw_outputs(
    records: Sequence[Mapping[str, str]], raw_outputs: Sequence[str]
) -> list[dict[str, str]]:
    if len(records) != len(raw_outputs):
        raise RuntimeError("raw Qwen output cardinality changed")
    rows: list[dict[str, str]] = []
    for record, raw in zip(records, raw_outputs, strict=True):
        try:
            rows.append(condition_row(record, raw))
        except Exception as exc:
            raise ValueError(
                "invalid frozen-Qwen output for "
                f"record_id={record['record_id']}: {raw!r}"
            ) from exc
    return rows


def generate_rank_raw_outputs(
    model: torch.nn.Module,
    processor: object,
    records: Sequence[Mapping[str, str]],
    config: Mapping[str, object],
    *,
    rank: int,
    world: int,
    device: torch.device,
) -> list[tuple[int, Mapping[str, str], str]]:
    """Generate only this rank's rows; cross-rank transfer uses durable shards."""

    generation = config["generation"]
    prompt_config = config["prompt"]
    assert isinstance(generation, Mapping)
    assert isinstance(prompt_config, Mapping)
    model.eval()
    model.gradient_checkpointing_disable()
    model.config.use_cache = True
    prompt = prompt_text(processor, str(prompt_config["target_schema"]))
    indexed = [
        (index, record)
        for index, record in enumerate(records)
        if index % world == rank
    ]
    local: list[tuple[int, Mapping[str, str], str]] = []
    batch_size = int(generation["batch_size_per_rank"])
    for offset in range(0, len(indexed), batch_size):
        chunk = indexed[offset : offset + batch_size]
        images = open_images([record for _, record in chunk])
        encoded = processor(
            text=[prompt] * len(chunk), images=images, padding=True, return_tensors="pt"
        )
        inputs = {
            key: value.to(device)
            for key, value in encoded.items()
            if isinstance(value, torch.Tensor)
        }
        input_length = int(inputs["input_ids"].shape[1])
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=int(generation["max_new_tokens"]),
                use_cache=True,
            )
        for row, (index, record) in enumerate(chunk):
            raw = processor.tokenizer.decode(
                generated[row, input_length:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            ).strip()
            local.append((index, record, raw))
        if (offset // batch_size + 1) % 25 == 0 or offset + batch_size >= len(indexed):
            print(
                f"QWEN_MULTIDEGRADATION_GENERATE rank={rank} "
                f"completed={min(offset+batch_size,len(indexed))}/{len(indexed)}",
                flush=True,
            )
    return local


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
        generation_records = [{**row, "output_path": row["lq_path"]} for row in selected]
        manifest_sha = sha256(args.manifest)
        qwen_config_sha = sha256(args.qwen_config)
        adapter_sha = sha256(adapter)
        identity = raw_shard_identity(
            selected,
            manifest_sha256=manifest_sha,
            qwen_config_sha256=qwen_config_sha,
            adapter_sha256=adapter_sha,
        )
        assigned = [
            (index, record)
            for index, record in enumerate(selected)
            if index % world == rank
        ]
        shard = raw_shard_path(args.output_csv, rank)
        if shard.exists():
            load_raw_shard(
                shard, assigned, identity=identity, rank=rank, world=world
            )
            print(
                f"FOUNDIR_FIVE_TYPE_QWEN_RAW_SHARD_VERIFY_PASS rank={rank} "
                f"records={len(assigned)}",
                flush=True,
            )
        else:
            model, processor, _ = load_model(
                qwen_config, variant="merger_trainable", device=device
            )
            load_qwen_adapter(model, adapter)
            generated = generate_rank_raw_outputs(
                model,
                processor,
                generation_records,
                qwen_config,
                rank=rank,
                world=world,
                device=device,
            )
            content = encode_raw_shard(
                generated, identity=identity, rank=rank, world=world
            )
            write_raw_shard_once(shard, content)
            load_raw_shard(
                shard, assigned, identity=identity, rank=rank, world=world
            )
            print(
                f"FOUNDIR_FIVE_TYPE_QWEN_RAW_SHARD_PASS rank={rank} "
                f"records={len(assigned)}",
                flush=True,
            )

        # All ranks finish durable publication before rank zero parses.  The
        # rank-zero helper catches and broadcasts parser/publication errors,
        # so nonzero ranks cannot wait for an unmatched final barrier.
        dist.barrier()

        def publish_conditions() -> dict[str, object]:
            raw_outputs = load_all_raw_shards(
                args.output_csv, selected, identity=identity, world=world
            )
            rows = condition_rows_from_raw_outputs(selected, raw_outputs)
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
                "manifest_sha256": manifest_sha,
                "qwen_config": args.qwen_config.relative_to(ROOT).as_posix(),
                "qwen_config_sha256": qwen_config_sha,
                "qwen_adapter": adapter.relative_to(ROOT).as_posix(),
                "qwen_adapter_sha256": adapter_sha,
                "conditions_csv": args.output_csv.relative_to(ROOT).as_posix(),
                "conditions_csv_sha256": hashlib.sha256(csv_content).hexdigest(),
                "protected_foundir_test_accessed": False,
            }
            write_once(args.output_json, (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode())
            return payload

        payload = broadcast_rank0(rank, publish_conditions)
        if rank == 0:
            print(
                "FOUNDIR_FIVE_TYPE_QWEN_CONDITIONS_PASS "
                + json.dumps(payload, sort_keys=True),
                flush=True,
            )
    finally:
        if initialized and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
