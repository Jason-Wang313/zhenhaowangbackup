#!/usr/bin/env python3
"""Resume-safe eight-GPU Joint-Qwen export for all FoundIR development rows."""

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

import torch
import torch.distributed as dist
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lowlight_restoration.qwen_lora import load_qwen_adapter  # noqa: E402
from lowlight_restoration.foundir_flux2_protocol import resolve_workspace_path  # noqa: E402
from lowlight_restoration.qwen_multidegradation import (  # noqa: E402
    FIXED_FIVE_SLOT_SCHEMA,
    prompt_text,
)
from smoke.export_foundir_five_type_qwen_conditions import (  # noqa: E402
    condition_row,
    load_manifest,
    sha256,
)
from smoke.train_qwen_multidegradation_lora_ddp import (  # noqa: E402
    load_model,
    seed_everything,
    setup_distributed,
    validate_runtime,
)


def append_rows(path: Path, rows: Sequence[Mapping[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), separators=(",", ":"), sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_shard(path: Path, assigned: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    if not path.exists():
        return []
    completed: list[dict[str, str]] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"incomplete Qwen shard line {path}:{line_number}") from exc
        if line_number > len(assigned):
            raise RuntimeError(f"Qwen shard has excess rows: {path}")
        expected = assigned[line_number - 1]
        if (
            row.get("record_id") != expected["record_id"]
            or row.get("lq_sha256") != expected["lq_sha256"]
            or row.get("gt_sha256") != expected["gt_sha256"]
        ):
            raise RuntimeError(f"Qwen shard prefix changed: {path}:{line_number}")
        # Reparse the cached decoder answer; valid JSON alone is insufficient.
        rebuilt = condition_row(expected, str(row.get("qwen_raw_output", "")))
        if row != rebuilt:
            raise RuntimeError(f"Qwen shard condition changed: {path}:{line_number}")
        completed.append(row)
    return completed


def open_images(records: Sequence[Mapping[str, str]]) -> list[Image.Image]:
    images = []
    for record in records:
        path = (ROOT / record["lq_path"]).resolve()
        path.relative_to(ROOT)
        with Image.open(path) as handle:
            images.append(handle.convert("RGB"))
    return images


def generate_local(
    model,
    processor,
    records: Sequence[Mapping[str, str]],
    *,
    batch_size: int,
    max_new_tokens: int,
    device: torch.device,
    shard: Path,
    starting_offset: int,
) -> None:
    prompt = prompt_text(processor, FIXED_FIVE_SLOT_SCHEMA)
    for offset in range(starting_offset, len(records), batch_size):
        chunk = records[offset : offset + batch_size]
        images = open_images(chunk)
        encoded = processor(
            text=[prompt] * len(chunk), images=images, padding=True, return_tensors="pt"
        )
        inputs = {
            key: value.to(device)
            for key, value in encoded.items()
            if isinstance(value, torch.Tensor)
        }
        input_length = int(inputs["input_ids"].shape[1])
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                use_cache=True,
            )
        rows = []
        for index, record in enumerate(chunk):
            raw = processor.tokenizer.decode(
                generated[index, input_length:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            ).strip()
            rows.append(condition_row(record, raw))
        append_rows(shard, rows)
        print(
            f"FOUNDIR_QWEN_SHARD rank={dist.get_rank()} completed={offset+len(chunk)}/{len(records)}",
            flush=True,
        )


def publish_file_once(temporary: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if sha256(destination) != sha256(temporary):
            raise RuntimeError(f"refusing to overwrite differing Qwen export: {destination}")
        temporary.unlink()
        return
    os.link(temporary, destination)
    temporary.unlink()


def merge_shards(
    *,
    manifest_rows: Sequence[Mapping[str, str]],
    shard_dir: Path,
    world: int,
    output_csv: Path,
    output_json: Path,
    config_path: Path,
    qwen_config_path: Path,
    manifest_path: Path,
    adapter: Path,
) -> dict[str, object]:
    by_id: dict[str, dict[str, str]] = {}
    for rank in range(world):
        assigned = [row for index, row in enumerate(manifest_rows) if index % world == rank]
        shard_rows = load_shard(shard_dir / f"rank_{rank:02d}.jsonl", assigned)
        if len(shard_rows) != len(assigned):
            raise RuntimeError(f"Qwen shard {rank} is incomplete")
        for row in shard_rows:
            if row["record_id"] in by_id:
                raise RuntimeError("duplicate record across Qwen shards")
            by_id[row["record_id"]] = row
    if len(by_id) != len(manifest_rows):
        raise RuntimeError("merged Qwen cache cardinality changed")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", newline="", dir=output_csv.parent, prefix=f".{output_csv.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
        first = by_id[manifest_rows[0]["record_id"]]
        writer = csv.DictWriter(handle, fieldnames=list(first), lineterminator="\n")
        writer.writeheader()
        for record in manifest_rows:
            writer.writerow(by_id[record["record_id"]])
        handle.flush()
        os.fsync(handle.fileno())
    split_counts = Counter(row["split"] for row in manifest_rows)
    category_counts = Counter(row["category"] for row in manifest_rows)
    payload = {
        "schema_version": 1,
        "status": "frozen_joint_qwen_foundir_all_development_conditions",
        "records": len(manifest_rows),
        "split_counts": dict(sorted(split_counts.items())),
        "category_counts": dict(sorted(category_counts.items())),
        "world_size": world,
        "manifest": manifest_path.relative_to(ROOT).as_posix(),
        "manifest_sha256": sha256(manifest_path),
        "qwen_config": qwen_config_path.relative_to(ROOT).as_posix(),
        "qwen_config_sha256": sha256(qwen_config_path),
        "qwen_adapter": adapter.relative_to(ROOT).as_posix(),
        "qwen_adapter_sha256": sha256(adapter),
        "restoration_config": config_path.relative_to(ROOT).as_posix(),
        "restoration_config_sha256": sha256(config_path),
        "conditions_csv": output_csv.relative_to(ROOT).as_posix(),
        "conditions_csv_sha256": sha256(temporary),
        "protected_foundir_test_accessed": False,
    }
    publish_file_once(temporary, output_csv)
    content = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    if output_json.exists() and output_json.read_bytes() != content:
        raise RuntimeError("refusing to overwrite differing Qwen export summary")
    if not output_json.exists():
        output_json.write_bytes(content)
    return payload


def verify_existing(
    *,
    output_csv: Path,
    output_json: Path,
    manifest: Path,
    qwen_config: Path,
    adapter: Path,
    config: Path,
) -> dict[str, object]:
    payload = json.loads(output_json.read_text())
    if (
        payload.get("status") != "frozen_joint_qwen_foundir_all_development_conditions"
        or payload.get("manifest_sha256") != sha256(manifest)
        or payload.get("qwen_config_sha256") != sha256(qwen_config)
        or payload.get("qwen_adapter_sha256") != sha256(adapter)
        or payload.get("restoration_config_sha256") != sha256(config)
        or payload.get("conditions_csv_sha256") != sha256(output_csv)
        or payload.get("protected_foundir_test_accessed") is not False
    ):
        raise RuntimeError("full Qwen condition export identity changed")
    print("FOUNDIR_QWEN_FULL_EXPORT_VERIFY_PASS " + json.dumps(payload, sort_keys=True))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--qwen-config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--shard-dir", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--verify-existing", action="store_true")
    args = parser.parse_args()
    args.config = resolve_workspace_path(ROOT, args.config)
    args.qwen_config = resolve_workspace_path(ROOT, args.qwen_config)
    args.manifest = resolve_workspace_path(ROOT, args.manifest)
    args.shard_dir = resolve_workspace_path(ROOT, args.shard_dir)
    args.output_csv = resolve_workspace_path(ROOT, args.output_csv)
    args.output_json = resolve_workspace_path(ROOT, args.output_json)
    restoration = json.loads(args.config.read_text())
    qwen_config = json.loads(args.qwen_config.read_text())
    adapter = ROOT / restoration["qwen_condition"]["adapter"]
    if args.verify_existing:
        verify_existing(
            output_csv=args.output_csv,
            output_json=args.output_json,
            manifest=args.manifest,
            qwen_config=args.qwen_config,
            adapter=adapter,
            config=args.config,
        )
        return
    rows = load_manifest(args.manifest)
    if len(rows) != int(restoration["data"]["expected_five_type_records"]):
        raise RuntimeError("full Qwen export manifest cardinality changed")
    if sha256(args.qwen_config) != restoration["qwen_condition"]["training_config_sha256"]:
        raise RuntimeError("frozen Joint-Qwen config changed")
    if sha256(adapter) != restoration["qwen_condition"]["adapter_sha256"]:
        raise RuntimeError("frozen Joint-Qwen adapter changed")
    if qwen_config["prompt"]["target_schema"] != FIXED_FIVE_SLOT_SCHEMA:
        raise RuntimeError("frozen Joint-Qwen schema changed")

    initialized = False
    try:
        rank, world, local_rank, device = setup_distributed(qwen_config)
        initialized = True
        if world != int(restoration["distributed"]["required_world_size"]):
            raise RuntimeError("full Qwen export requires exactly eight GPUs")
        validate_runtime(qwen_config, local_rank)
        assigned = [row for index, row in enumerate(rows) if index % world == rank]
        shard = args.shard_dir / f"rank_{rank:02d}.jsonl"
        completed = load_shard(shard, assigned)
        if len(completed) < len(assigned):
            os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
            seed_everything(int(restoration["flow_training"]["seed"]))
            model, processor, _ = load_model(qwen_config, variant="merger_trainable", device=device)
            load_qwen_adapter(model, adapter)
            model.eval()
            model.gradient_checkpointing_disable()
            model.config.use_cache = True
            generate_local(
                model,
                processor,
                assigned,
                batch_size=int(qwen_config["generation"]["batch_size_per_rank"]),
                max_new_tokens=int(qwen_config["generation"]["max_new_tokens"]),
                device=device,
                shard=shard,
                starting_offset=len(completed),
            )
        dist.barrier()
        if rank == 0:
            payload = merge_shards(
                manifest_rows=rows,
                shard_dir=args.shard_dir,
                world=world,
                output_csv=args.output_csv,
                output_json=args.output_json,
                config_path=args.config,
                qwen_config_path=args.qwen_config,
                manifest_path=args.manifest,
                adapter=adapter,
            )
            print("FOUNDIR_QWEN_FULL_EXPORT_PASS " + json.dumps(payload, sort_keys=True), flush=True)
        dist.barrier()
    finally:
        if initialized and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
