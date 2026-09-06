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
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence

import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lowlight_restoration.qwen_lora import load_qwen_adapter  # noqa: E402
from lowlight_restoration.qwen_multidegradation import PER_TYPE_DIALOGUE_SCHEMA  # noqa: E402
from smoke.export_foundir_five_type_qwen_conditions import (  # noqa: E402
    condition_row,
    load_manifest,
    sha256,
)
from smoke.train_qwen_multidegradation_lora_ddp import (  # noqa: E402
    broadcast_rank0,
    generate_outputs,
    load_model,
    seed_everything,
    setup_distributed,
    validate_runtime,
)


def workspace_path(value: str | Path) -> Path:
    path = (ROOT / value).resolve()
    path.relative_to(ROOT.resolve())
    return path


def center_generation_records(rows: Sequence[Mapping[str, str]], crop_size: int) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for row in rows:
        width, height = int(row["width"]), int(row["height"])
        if min(width, height) < crop_size:
            raise RuntimeError(f"Qwen center crop is larger than {row['record_id']}: {width}x{height}")
        left, top = (width - crop_size) // 2, (height - crop_size) // 2
        output.append(
            {
                **row,
                "output_path": row["lq_path"],
                "crop_box": [left, top, left + crop_size, top + crop_size],
            }
        )
    return output


def chunk_identity(
    rows: Sequence[Mapping[str, str]], *, config_sha: str, qwen_config_sha: str, adapter_sha: str
) -> dict[str, object]:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(
            f"{row['record_id']}\0{row['lq_sha256']}\0{row['gt_sha256']}\n".encode()
        )
    return {
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
    args = parser.parse_args()
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
        for start in range(0, len(rows), args.chunk_records):
            selected = rows[start : start + args.chunk_records]
            path = chunk_dir / f"chunk_{start:07d}_{start+len(selected):07d}.json"
            if path.exists():
                verify_chunk(path, chunk_identity(selected, config_sha=config_sha, qwen_config_sha=qwen_sha, adapter_sha=adapter_sha), selected)
            else:
                missing.append(start)
        model = processor = None
        if missing:
            seed_everything(int(config["flow_training"]["seed"]))
            model, processor, _ = load_model(qwen_config, variant="merger_trainable", device=device)
            load_qwen_adapter(model, adapter)
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
