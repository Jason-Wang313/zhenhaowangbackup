#!/usr/bin/env python3
"""Resume-safe full FoundIR-training condition export with physical-v8 Qwen."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence

import torch
import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lowlight_restoration.qwen_lora import load_qwen_adapter  # noqa: E402
from lowlight_restoration.qwen_multidegradation import (  # noqa: E402
    DEGRADATIONS,
    PER_TYPE_DIALOGUE_SCHEMA,
    aggregate_per_type_responses,
    per_type_prompt_text,
)
from smoke.export_foundir_five_type_qwen_conditions import (  # noqa: E402
    condition_row,
    load_manifest,
    sha256,
)
from smoke.train_qwen_multidegradation_lora_ddp import (  # noqa: E402
    broadcast_rank0,
    generate_outputs,
    left_padded_generation,
    load_model,
    open_images,
    seed_everything,
    setup_distributed,
    validate_runtime,
)


OPTIMIZED_EXPORT_ENGINE = "record_sharded_per_type_batch_v1"
UNDERSIZED_INPUT_POLICY = "bicubic_short_side_to_448_then_center_crop_v1"


def workspace_path(value: str | Path) -> Path:
    path = (ROOT / value).resolve()
    path.relative_to(ROOT.resolve())
    return path


def resized_dimensions(width: int, height: int, crop_size: int) -> tuple[int, int]:
    """Upscale the short side to ``crop_size`` without changing aspect ratio."""

    if width < 1 or height < 1 or crop_size < 1:
        raise ValueError("image and crop dimensions must be positive")
    short_side = min(width, height)
    if short_side >= crop_size:
        return width, height
    # Integer ceiling ensures that rounding can never leave the long side below
    # the requested crop size.  PIL bicubic is the Qwen processor's native
    # resampling family and is deterministic for a fixed Pillow version.
    return (
        max(crop_size, (width * crop_size + short_side - 1) // short_side),
        max(crop_size, (height * crop_size + short_side - 1) // short_side),
    )


def generation_geometry_summary(
    rows: Sequence[Mapping[str, str]], crop_size: int
) -> dict[str, object]:
    """Validate every manifest dimension before model loading."""

    undersized: list[str] = []
    minimum_width: int | None = None
    minimum_height: int | None = None
    for row in rows:
        try:
            width, height = int(row["width"]), int(row["height"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                f"invalid manifest dimensions for {row.get('record_id', '<unknown>')}"
            ) from exc
        if width < 1 or height < 1:
            raise RuntimeError(
                f"non-positive manifest dimensions for {row.get('record_id', '<unknown>')}: "
                f"{width}x{height}"
            )
        minimum_width = width if minimum_width is None else min(minimum_width, width)
        minimum_height = height if minimum_height is None else min(minimum_height, height)
        if min(width, height) < crop_size:
            undersized.append(row["record_id"])
    return {
        "records": len(rows),
        "crop_size": crop_size,
        "minimum_width": minimum_width,
        "minimum_height": minimum_height,
        "undersized_records": len(undersized),
        "undersized_input_policy": UNDERSIZED_INPUT_POLICY,
        "status": "pass",
    }


def center_generation_records(rows: Sequence[Mapping[str, str]], crop_size: int) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for row in rows:
        width, height = int(row["width"]), int(row["height"])
        resized_width, resized_height = resized_dimensions(width, height, crop_size)
        left, top = (resized_width - crop_size) // 2, (resized_height - crop_size) // 2
        record: dict[str, object] = {
            **row,
            "output_path": row["lq_path"],
            "crop_box": [left, top, left + crop_size, top + crop_size],
        }
        if (resized_width, resized_height) != (width, height):
            record["resize_size"] = [resized_width, resized_height]
            record["undersized_input_policy"] = UNDERSIZED_INPUT_POLICY
        output.append(record)
    return output


def optimized_batch_size(device: torch.device, local_records: int) -> int:
    """Use the largest useful batch while remaining safe on 80 GiB H100s.

    A 512-record export chunk assigns at most 64 images to each of eight
    ranks. H200 uses all 64 in one call; 80 GiB H100 conservatively uses 32 in
    two calls. This avoids speculative OOM recovery inside a distributed
    collective while still eliminating most legacy launch and image-I/O
    overhead.
    """

    if local_records < 1:
        return 1
    properties = torch.cuda.get_device_properties(device)
    minimum_h100_bytes = 75 * 1024**3
    if properties.total_memory < minimum_h100_bytes:
        raise RuntimeError("optimized Qwen export requires at least 75 GiB per GPU")
    cap = 64 if properties.total_memory >= 100 * 1024**3 else 32
    return min(cap, local_records)


def record_indices_for_rank(total: int, rank: int, world: int) -> list[int]:
    if total < 0 or world < 1 or rank < 0 or rank >= world:
        raise ValueError("invalid record-sharding dimensions")
    return list(range(rank, total, world))


def chunk_belongs_to_partition(chunk_index: int, partitions: int, partition_index: int) -> bool:
    if chunk_index < 0 or partitions < 1 or partition_index < 0 or partition_index >= partitions:
        raise ValueError("invalid chunk-partition dimensions")
    return chunk_index % partitions == partition_index


def generate_outputs_record_sharded(
    model: torch.nn.Module,
    processor: object,
    records: Sequence[Mapping[str, object]],
    config: Mapping[str, object],
    *,
    rank: int,
    world: int,
    device: torch.device,
) -> list[str]:
    """Run the same five dialogues while decoding every source image once.

    The legacy implementation shards the flattened image/question Cartesian
    product.  Consequently each image is opened and decoded five times and a
    512-image chunk needs 20 small generation calls per rank.  This version
    shards images first, opens each local image once, and evaluates one large
    batch for each of the same five prompts.  Model weights, crops, prompts,
    greedy decoding, and aggregation are unchanged.
    """

    generation = config["generation"]
    if not isinstance(generation, Mapping):
        raise ValueError("Qwen generation contract is malformed")
    model.eval()
    model.gradient_checkpointing_disable()
    model.config.use_cache = True
    indexed = [(index, records[index]) for index in record_indices_for_rank(len(records), rank, world)]
    batch_size = optimized_batch_size(device, len(indexed))
    local: list[tuple[int, str, str]] = []
    images = open_images([record for _, record in indexed]) if indexed else []
    try:
        with left_padded_generation(processor), torch.inference_mode():
            for degradation in DEGRADATIONS:
                prompt = per_type_prompt_text(processor, degradation)
                for offset in range(0, len(indexed), batch_size):
                    chunk = indexed[offset : offset + batch_size]
                    chunk_images = images[offset : offset + len(chunk)]
                    encoded = processor(
                        text=[prompt] * len(chunk),
                        images=chunk_images,
                        padding=True,
                        return_tensors="pt",
                    )
                    inputs = {
                        key: value.to(device)
                        for key, value in encoded.items()
                        if isinstance(value, torch.Tensor)
                    }
                    input_length = int(inputs["input_ids"].shape[1])
                    generated = model.generate(
                        **inputs,
                        do_sample=False,
                        max_new_tokens=int(generation["max_new_tokens"]),
                        use_cache=True,
                    )
                    decoded = processor.tokenizer.batch_decode(
                        generated[:, input_length:],
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )
                    local.extend(
                        (record_index, degradation, raw.strip())
                        for (record_index, _), raw in zip(chunk, decoded, strict=True)
                    )
                print(
                    f"FOUNDIR_PHYSICAL_V8_OPTIMIZED_GENERATE rank={rank} "
                    f"type={degradation} completed={len(indexed)}/{len(indexed)} "
                    f"batch_size={batch_size}",
                    flush=True,
                )
    finally:
        for image in images:
            image.close()

    gathered: list[object | None] = [None] * world
    dist.all_gather_object(gathered, local)
    flattened = [
        item
        for rank_items in gathered
        if isinstance(rank_items, list)
        for item in rank_items
    ]
    expected = {(index, degradation) for index in range(len(records)) for degradation in DEGRADATIONS}
    identities = {(index, degradation) for index, degradation, _ in flattened}
    if len(flattened) != len(expected) or identities != expected:
        raise RuntimeError("optimized distributed Qwen generation is incomplete")
    lookup = {(index, degradation): raw for index, degradation, raw in flattened}
    outputs: list[str] = []
    for index in range(len(records)):
        raw_by_type = {degradation: lookup[(index, degradation)] for degradation in DEGRADATIONS}
        serialized, _ = aggregate_per_type_responses(raw_by_type)
        outputs.append(serialized if serialized is not None else "INVALID_PER_TYPE_DIALOGUE")
    return outputs


def validate_optimized_engine(
    *,
    model: torch.nn.Module,
    processor: object,
    rows: Sequence[Mapping[str, str]],
    generation_records: Sequence[Mapping[str, object]],
    existing_conditions: Sequence[Mapping[str, str]],
    config: Mapping[str, object],
    rank: int,
    world: int,
    device: torch.device,
) -> dict[str, object]:
    """Prove exact condition equivalence against an immutable existing chunk."""

    started = time.monotonic()
    outputs = generate_outputs_record_sharded(
        model,
        processor,
        generation_records,
        config,
        rank=rank,
        world=world,
        device=device,
    )
    elapsed = time.monotonic() - started

    def verify() -> dict[str, object]:
        candidate = [condition_row(row, raw) for row, raw in zip(rows, outputs, strict=True)]
        exact = candidate == list(existing_conditions)
        payload = {
            "schema_version": 1,
            "status": "pass" if exact else "failed_exact_condition_equivalence",
            "engine": OPTIMIZED_EXPORT_ENGINE,
            "records": len(rows),
            "queries": len(rows) * len(DEGRADATIONS),
            "exact_condition_equivalence": exact,
            "elapsed_seconds": elapsed,
            "images_per_second": len(rows) / elapsed,
            "protected_foundir_test_accessed": False,
        }
        if not exact:
            mismatch = next(
                index
                for index, (left, right) in enumerate(zip(candidate, existing_conditions, strict=True))
                if left != right
            )
            payload["first_mismatch_record_id"] = rows[mismatch]["record_id"]
        return payload

    return broadcast_rank0(rank, verify)


def chunk_identity(
    rows: Sequence[Mapping[str, str]], *, config_sha: str, qwen_config_sha: str, adapter_sha: str
) -> dict[str, object]:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(
            f"{row['record_id']}\0{row['lq_sha256']}\0{row['gt_sha256']}\n".encode()
        )
    identity = {
        "schema_version": 1,
        "status": "complete_physical_v8_qwen_condition_chunk",
        "config_sha256": config_sha,
        "qwen_config_sha256": qwen_config_sha,
        "qwen_adapter_sha256": adapter_sha,
        "first_record_id": rows[0]["record_id"],
        "last_record_id": rows[-1]["record_id"],
        "records": len(rows),
        "input_digest": digest.hexdigest(),
        "qwen_method": "per_type_dialogue_center_crop_448",
        "query_count_per_image": 5,
        "jpeg_included": False,
        "protected_foundir_test_accessed": False,
    }
    undersized = [
        row["record_id"]
        for row in rows
        if "width" in row
        and "height" in row
        and min(int(row["width"]), int(row["height"])) < 448
    ]
    if undersized:
        identity["undersized_input_policy"] = UNDERSIZED_INPUT_POLICY
        identity["undersized_record_ids"] = undersized
    return identity


def verify_chunk(path: Path, identity: Mapping[str, object], rows: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    payload = json.loads(path.read_text())
    for key, value in identity.items():
        if payload.get(key) != value:
            raise RuntimeError(f"Qwen condition chunk identity changed: {path}: {key}")
    conditions = payload.get("conditions")
    if not isinstance(conditions, list) or len(conditions) != len(rows):
        raise RuntimeError(f"Qwen condition chunk cardinality changed: {path}")
    for condition, row in zip(conditions, rows, strict=True):
        if (
            not isinstance(condition, dict)
            or condition.get("record_id") != row["record_id"]
            or condition.get("lq_sha256") != row["lq_sha256"]
            or condition.get("gt_sha256") != row["gt_sha256"]
        ):
            raise RuntimeError(f"Qwen condition chunk record mapping changed: {path}")
    return conditions


def write_chunk_once(path: Path, payload: Mapping[str, object]) -> None:
    content = (json.dumps(dict(payload), separators=(",", ":"), sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise RuntimeError(f"refusing to overwrite differing Qwen condition chunk: {path}")
        return
    with path.open("xb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


def publish_csv(
    rows: Sequence[Mapping[str, str]],
    *,
    chunk_dir: Path,
    chunk_records: int,
    output_csv: Path,
    output_json: Path,
    config_path: Path,
    qwen_config_path: Path,
    adapter: Path,
    manifest_path: Path,
) -> dict[str, object]:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    config_sha, qwen_sha, adapter_sha = sha256(config_path), sha256(qwen_config_path), sha256(adapter)
    with tempfile.NamedTemporaryFile(
        mode="w", newline="", dir=output_csv.parent, prefix=f".{output_csv.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
        writer = None
        for start in range(0, len(rows), chunk_records):
            selected = rows[start : start + chunk_records]
            path = chunk_dir / f"chunk_{start:07d}_{start+len(selected):07d}.json"
            conditions = verify_chunk(
                path,
                chunk_identity(selected, config_sha=config_sha, qwen_config_sha=qwen_sha, adapter_sha=adapter_sha),
                selected,
            )
            if writer is None:
                writer = csv.DictWriter(handle, fieldnames=list(conditions[0]), lineterminator="\n")
                writer.writeheader()
            writer.writerows(conditions)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        csv_sha = sha256(temporary)
        if output_csv.exists():
            if sha256(output_csv) != csv_sha:
                raise RuntimeError("refusing to overwrite differing full Qwen condition CSV")
        else:
            os.link(temporary, output_csv)
    finally:
        temporary.unlink(missing_ok=True)
    payload = {
        "schema_version": 1,
        "status": "frozen_physical_v8_qwen_foundir_all_development_conditions",
        "records": len(rows),
        "split_counts": dict(sorted(Counter(row["split"] for row in rows).items())),
        "category_counts": dict(sorted(Counter(row["category"] for row in rows).items())),
        "manifest": manifest_path.relative_to(ROOT).as_posix(),
        "manifest_sha256": sha256(manifest_path),
        "qwen_config": qwen_config_path.relative_to(ROOT).as_posix(),
        "qwen_config_sha256": qwen_sha,
        "qwen_adapter": adapter.relative_to(ROOT).as_posix(),
        "qwen_adapter_sha256": adapter_sha,
        "restoration_config": config_path.relative_to(ROOT).as_posix(),
        "restoration_config_sha256": config_sha,
        "conditions_csv": output_csv.relative_to(ROOT).as_posix(),
        "conditions_csv_sha256": sha256(output_csv),
        "qwen_method": "per_type_dialogue_center_crop_448",
        "query_count_per_image": 5,
        "undersized_input_policy": UNDERSIZED_INPUT_POLICY,
        "undersized_records": sum(
            min(int(row["width"]), int(row["height"])) < 448 for row in rows
        ),
        "condition_arms": ["qwen"],
        "jpeg_included": False,
        "protected_foundir_test_accessed": False,
    }
    content = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    if output_json.exists() and output_json.read_bytes() != content:
        raise RuntimeError("refusing to overwrite differing full Qwen condition summary")
    if not output_json.exists():
        with output_json.open("xb") as handle:
            handle.write(content)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--chunk-records", type=int, default=512)
    parser.add_argument("--verify-existing", action="store_true")
    parser.add_argument("--benchmark-only", action="store_true")
    parser.add_argument("--partitions", type=int, default=1)
    parser.add_argument("--partition-index", type=int, default=0)
    args = parser.parse_args()
    if args.verify_existing and args.benchmark_only:
        parser.error("--verify-existing and --benchmark-only are mutually exclusive")
    if args.partitions < 1 or args.partition_index < 0 or args.partition_index >= args.partitions:
        parser.error("--partition-index must be in [0, --partitions)")
    if (args.verify_existing or args.benchmark_only) and args.partitions != 1:
        parser.error("verification and benchmark modes do not support partitions")
    config_path, manifest_path = workspace_path(args.config), workspace_path(args.manifest)
    config = json.loads(config_path.read_text())
    qwen = config["qwen_condition"]
    qwen_config_path = workspace_path(qwen["training_config"])
    adapter = workspace_path(qwen["adapter"])
    output_csv = workspace_path(config["full_conditions"]["csv"])
    output_json = workspace_path(config["full_conditions"]["summary"])
    chunk_dir = workspace_path(config["full_conditions"]["shard_directory"])
    if (
        qwen.get("method") != "per_type_dialogue_center_crop_448"
        or config["scope"].get("exclude_jpeg") is not True
        or config["evaluation"].get("condition_arms") != ["qwen"]
        or sha256(qwen_config_path) != qwen["training_config_sha256"]
        or sha256(adapter) != qwen["adapter_sha256"]
    ):
        raise RuntimeError("physical-v8 no-JPEG/no-ablation export contract changed")
    qwen_config = json.loads(qwen_config_path.read_text())
    if qwen_config["prompt"]["target_schema"] != PER_TYPE_DIALOGUE_SCHEMA:
        raise RuntimeError("physical-v8 Qwen is not a per-type dialogue model")
    rows = load_manifest(manifest_path)
    if len(rows) != int(config["data"]["expected_five_type_records"]):
        raise RuntimeError("full FoundIR five-type manifest cardinality changed")
    geometry = generation_geometry_summary(rows, int(qwen["crop_size"]))
    if int(os.environ.get("LOCAL_RANK", "0")) == 0:
        print(
            "FOUNDIR_PHYSICAL_V8_GEOMETRY_PREFLIGHT_PASS "
            + json.dumps(geometry, sort_keys=True),
            flush=True,
        )

    if args.verify_existing:
        payload = publish_csv(
            rows,
            chunk_dir=chunk_dir,
            chunk_records=args.chunk_records,
            output_csv=output_csv,
            output_json=output_json,
            config_path=config_path,
            qwen_config_path=qwen_config_path,
            adapter=adapter,
            manifest_path=manifest_path,
        )
        print("FOUNDIR_PHYSICAL_V8_FULL_EXPORT_VERIFY_PASS " + json.dumps(payload, sort_keys=True))
        return

    initialized = False
    try:
        rank, world, local_rank, device = setup_distributed(qwen_config)
        initialized = True
        if world != int(config["distributed"]["required_world_size"]):
            raise RuntimeError("physical-v8 full condition export requires exactly eight GPUs")
        validate_runtime(qwen_config, local_rank)
        config_sha, qwen_sha, adapter_sha = sha256(config_path), sha256(qwen_config_path), sha256(adapter)
        missing = []
        complete = []
        assigned_chunks = 0
        for chunk_index, start in enumerate(range(0, len(rows), args.chunk_records)):
            selected = rows[start : start + args.chunk_records]
            path = chunk_dir / f"chunk_{start:07d}_{start+len(selected):07d}.json"
            if path.exists():
                verify_chunk(path, chunk_identity(selected, config_sha=config_sha, qwen_config_sha=qwen_sha, adapter_sha=adapter_sha), selected)
                complete.append(start)
            elif chunk_belongs_to_partition(chunk_index, args.partitions, args.partition_index):
                missing.append(start)
            if chunk_belongs_to_partition(chunk_index, args.partitions, args.partition_index):
                assigned_chunks += 1
        model = processor = None
        if missing or args.benchmark_only:
            seed_everything(int(config["flow_training"]["seed"]))
            model, processor, _ = load_model(qwen_config, variant="merger_trainable", device=device)
            load_qwen_adapter(model, adapter)
            if args.benchmark_only and complete:
                reference_start = complete[0]
                reference_rows = rows[reference_start : reference_start + args.chunk_records]
                reference_path = (
                    chunk_dir
                    / f"chunk_{reference_start:07d}_{reference_start+len(reference_rows):07d}.json"
                )
                reference_identity = chunk_identity(
                    reference_rows,
                    config_sha=config_sha,
                    qwen_config_sha=qwen_sha,
                    adapter_sha=adapter_sha,
                )
                reference_conditions = verify_chunk(
                    reference_path, reference_identity, reference_rows
                )
                optimization = validate_optimized_engine(
                    model=model,
                    processor=processor,
                    rows=reference_rows,
                    generation_records=center_generation_records(
                        reference_rows, int(qwen["crop_size"])
                    ),
                    existing_conditions=reference_conditions,
                    config=qwen_config,
                    rank=rank,
                    world=world,
                    device=device,
                )
                optimized_equivalent = bool(
                    isinstance(optimization, Mapping)
                    and optimization.get("exact_condition_equivalence") is True
                )
                if rank == 0:
                    print(
                        "FOUNDIR_PHYSICAL_V8_OPTIMIZATION_VALIDATION "
                        + json.dumps(optimization, sort_keys=True),
                        flush=True,
                    )
            elif args.benchmark_only and rank == 0:
                print(
                    "FOUNDIR_PHYSICAL_V8_OPTIMIZATION_FALLBACK "
                    "no_immutable_reference_chunk",
                    flush=True,
                )
            if args.benchmark_only:
                if not complete or not optimized_equivalent:
                    raise RuntimeError(
                        "optimized export benchmark did not prove exact condition equivalence"
                    )
                if rank == 0:
                    print("FOUNDIR_PHYSICAL_V8_OPTIMIZATION_BENCHMARK_PASS", flush=True)
                return
        for start in missing:
            selected = rows[start : start + args.chunk_records]
            generation_records = center_generation_records(selected, int(qwen["crop_size"]))
            outputs = generate_outputs(
                model, processor, generation_records, qwen_config,
                rank=rank, world=world, device=device,
            )
            identity = chunk_identity(selected, config_sha=config_sha, qwen_config_sha=qwen_sha, adapter_sha=adapter_sha)

            def publish() -> bool:
                conditions = [condition_row(row, raw) for row, raw in zip(selected, outputs, strict=True)]
                path = chunk_dir / f"chunk_{start:07d}_{start+len(selected):07d}.json"
                write_chunk_once(path, {**identity, "conditions": conditions})
                verify_chunk(path, identity, selected)
                return True

            broadcast_rank0(rank, publish)
            if rank == 0:
                print(f"FOUNDIR_PHYSICAL_V8_FULL_EXPORT completed={start+len(selected)}/{len(rows)}", flush=True)
        if args.partitions > 1:
            if rank == 0:
                print(
                    "FOUNDIR_PHYSICAL_V8_PARTITION_PASS "
                    + json.dumps(
                        {
                            "partition_index": args.partition_index,
                            "partitions": args.partitions,
                            "assigned_chunks": assigned_chunks,
                            "generated_chunks": len(missing),
                            "engine": "legacy_exact_semantics",
                            "protected_foundir_test_accessed": False,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            return
        payload = broadcast_rank0(
            rank,
            lambda: publish_csv(
                rows,
                chunk_dir=chunk_dir,
                chunk_records=args.chunk_records,
                output_csv=output_csv,
                output_json=output_json,
                config_path=config_path,
                qwen_config_path=qwen_config_path,
                adapter=adapter,
                manifest_path=manifest_path,
            ),
        )
        if rank == 0:
            print("FOUNDIR_PHYSICAL_V8_FULL_EXPORT_PASS " + json.dumps(payload, sort_keys=True))
    finally:
        if initialized and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
