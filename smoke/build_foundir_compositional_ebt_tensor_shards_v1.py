#!/usr/bin/env python3
"""Build resumable contiguous uint8 tensor shards for lower-order EBT training."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/foundir_compositional_ebt_tensor_shards_v1.json"
DEGRADATIONS = ("blur", "noise", "rain", "haze", "low_light")
RECORDS: list[dict[str, Any]] = []
TARGETS: dict[str, dict[str, Any]] = {}
CONFIG: dict[str, Any] = {}
CONFIG_SHA = ""
OUTPUT_ROOT: Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_seed(*parts: object) -> int:
    payload = ":".join(map(str, parts)).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def canonical_line(payload: object) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()


def write_or_verify(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise RuntimeError(f"refusing to overwrite differing artifact: {path}")
        return
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sample_dtype(crop: int) -> np.dtype:
    return np.dtype([
        ("degraded", np.uint8, (3, crop, crop)),
        ("target", np.uint8, (3, crop, crop)),
        ("condition", np.uint8, (len(DEGRADATIONS),)),
        ("record_index", "<u4"),
        ("view_index", np.uint8),
    ], align=False)


def paths_for(shard_index: int) -> tuple[Path, Path]:
    stem = f"shard_{shard_index:04d}"
    return OUTPUT_ROOT / f"{stem}.npy", OUTPUT_ROOT / f"{stem}.json"


def build_shard(shard_index: int) -> dict[str, Any]:
    crop = int(CONFIG["crop_size"])
    views = int(CONFIG["views_per_record"])
    per_shard = int(CONFIG["records_per_shard"])
    first = shard_index * per_shard
    last = min(len(RECORDS), first + per_shard)
    records = RECORDS[first:last]
    shard_path, sidecar_path = paths_for(shard_index)
    if sidecar_path.is_file() and shard_path.is_file():
        sidecar = json.loads(sidecar_path.read_text())
        if sidecar.get("config_sha256") != CONFIG_SHA or sidecar.get("shard_index") != shard_index:
            raise RuntimeError(f"stale shard sidecar: {sidecar_path}")
        if sha256(shard_path) != sidecar.get("sha256"):
            raise RuntimeError(f"existing shard hash mismatch: {shard_path}")
        return sidecar
    if sidecar_path.exists() != shard_path.exists():
        raise RuntimeError(f"incomplete existing shard pair: {shard_path}")

    shard_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = shard_path.with_name(shard_path.name + f".tmp.{os.getpid()}.npy")
    array = np.lib.format.open_memmap(
        temporary, mode="w+", dtype=sample_dtype(crop), shape=(len(records) * views,)
    )
    cursor = 0
    for local_index, record in enumerate(records):
        record_id = str(record["record_id"])
        target_row = TARGETS[record_id]
        with Image.open(ROOT / str(record["output_path"])) as handle:
            degraded = np.asarray(handle.convert("RGB"), dtype=np.uint8).copy()
        with Image.open(ROOT / str(target_row["target_path"])) as handle:
            target = np.asarray(handle.convert("RGB"), dtype=np.uint8).copy()
        if degraded.shape != target.shape or min(degraded.shape[:2]) < crop:
            raise RuntimeError(f"bad source pair: {record_id}")
        height, width = degraded.shape[:2]
        present = set(map(str, record["degradations"]))
        condition = np.asarray([int(name in present) for name in DEGRADATIONS], dtype=np.uint8)
        for view_index in range(views):
            rng = random.Random(stable_seed(CONFIG["seed"], record_id, view_index))
            y = rng.randrange(height - crop + 1)
            x = rng.randrange(width - crop + 1)
            d = degraded[y : y + crop, x : x + crop]
            t = target[y : y + crop, x : x + crop]
            if rng.randrange(2):
                d, t = np.flip(d, 1), np.flip(t, 1)
            if rng.randrange(2):
                d, t = np.flip(d, 0), np.flip(t, 0)
            if rng.randrange(2):
                d, t = np.swapaxes(d, 0, 1), np.swapaxes(t, 0, 1)
            array[cursor]["degraded"] = np.ascontiguousarray(d.transpose(2, 0, 1))
            array[cursor]["target"] = np.ascontiguousarray(t.transpose(2, 0, 1))
            array[cursor]["condition"] = condition
            array[cursor]["record_index"] = first + local_index
            array[cursor]["view_index"] = view_index
            cursor += 1
    array.flush()
    del array
    os.replace(temporary, shard_path)
    digest = sha256(shard_path)
    sidecar = {
        "schema_version": 1,
        "shard_index": shard_index,
        "path": shard_path.relative_to(ROOT).as_posix(),
        "sha256": digest,
        "samples": len(records) * views,
        "records": len(records),
        "first_record_index": first,
        "last_record_index_exclusive": last,
        "first_record_id": records[0]["record_id"],
        "last_record_id": records[-1]["record_id"],
        "config_sha256": CONFIG_SHA,
    }
    write_or_verify(sidecar_path, (json.dumps(sidecar, indent=2, sort_keys=True) + "\n").encode())
    return sidecar


def validate(config: dict[str, Any], rows: list[dict[str, Any]], *, full_hash: bool) -> dict[str, Any]:
    if len(rows) != int(config["expected_shards"]):
        raise RuntimeError("shard count changed")
    if [int(row["shard_index"]) for row in rows] != list(range(len(rows))):
        raise RuntimeError("shard indices changed")
    if sum(int(row["samples"]) for row in rows) != int(config["total_samples"]):
        raise RuntimeError("sample count changed")
    if sum(int(row["records"]) for row in rows) != int(config["train_records"]):
        raise RuntimeError("record coverage changed")
    expected_first = 0
    for row in rows:
        if int(row["first_record_index"]) != expected_first:
            raise RuntimeError("record range gap/overlap")
        expected_first = int(row["last_record_index_exclusive"])
        path = ROOT / str(row["path"])
        if not path.is_file():
            raise RuntimeError(f"missing shard: {path}")
        if full_hash and sha256(path) != row["sha256"]:
            raise RuntimeError(f"shard hash mismatch: {path}")
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if array.shape != (int(row["samples"]),) or array.dtype != sample_dtype(int(config["crop_size"])):
            raise RuntimeError(f"shard shape/dtype changed: {path}")
    return {
        "shards": len(rows),
        "records": int(config["train_records"]),
        "samples": int(config["total_samples"]),
        "views_per_record": int(config["views_per_record"]),
        "full_hash_verification": full_hash,
        "triple_records": 0,
        "official_foundir_test_accessed": False,
        "sealed_internal_accessed": False,
        "gpu_used": False,
    }


def prepare_globals() -> None:
    global RECORDS, TARGETS, CONFIG, CONFIG_SHA, OUTPUT_ROOT
    CONFIG = json.loads(CONFIG_PATH.read_text())
    CONFIG_SHA = sha256(CONFIG_PATH)
    record_manifest = ROOT / CONFIG["record_manifest"]
    target_manifest = ROOT / CONFIG["target_manifest"]
    if sha256(record_manifest) != CONFIG["record_manifest_sha256"] or sha256(target_manifest) != CONFIG["target_manifest_sha256"]:
        raise RuntimeError("input manifest identity changed")
    RECORDS = sorted(
        [row for row in load_jsonl(record_manifest) if row["split"] == "train"],
        key=lambda row: str(row["record_id"]),
    )
    TARGETS = {str(row["record_id"]): row for row in load_jsonl(target_manifest) if row["split"] == "train"}
    if len(RECORDS) != int(CONFIG["train_records"]) or {str(row["record_id"]) for row in RECORDS} != set(TARGETS):
        raise RuntimeError("train/target coverage changed")
    if any(int(row["composition_size"]) not in {1, 2} for row in RECORDS):
        raise RuntimeError("triple entered tensor shard source")
    OUTPUT_ROOT = ROOT / CONFIG["output_root"]


def build(workers: int) -> dict[str, Any]:
    prepare_globals()
    shard_count = math.ceil(len(RECORDS) / int(CONFIG["records_per_shard"]))
    if shard_count != int(CONFIG["expected_shards"]):
        raise RuntimeError("expected shard count changed")
    started = time.perf_counter()
    rows = []
    with mp.get_context("fork").Pool(processes=workers) as pool:
        for completed, row in enumerate(pool.imap(build_shard, range(shard_count), chunksize=1), start=1):
            rows.append(row)
            if completed == 1 or completed % 10 == 0 or completed == shard_count:
                elapsed = time.perf_counter() - started
                print("EBT_TENSOR_SHARD_PROGRESS " + json.dumps({
                    "completed_shards": completed,
                    "total_shards": shard_count,
                    "elapsed_seconds": elapsed,
                    "estimated_total_seconds": elapsed / completed * shard_count,
                }, sort_keys=True), flush=True)
    rows.sort(key=lambda row: int(row["shard_index"]))
    checks = validate(CONFIG, rows, full_hash=True)
    manifest_payload = b"".join(canonical_line(row) for row in rows)
    write_or_verify(ROOT / CONFIG["output_manifest"], manifest_payload)
    metadata = {
        "schema_version": 1,
        "status": "foundir_compositional_ebt_tensor_shards_v1_ready",
        "config_sha256": CONFIG_SHA,
        "record_manifest_sha256": CONFIG["record_manifest_sha256"],
        "target_manifest_sha256": CONFIG["target_manifest_sha256"],
        "output_manifest": CONFIG["output_manifest"],
        "output_manifest_sha256": hashlib.sha256(manifest_payload).hexdigest(),
        "output_root": CONFIG["output_root"],
        "elapsed_seconds": time.perf_counter() - started,
        "workers": workers,
        "checks": checks,
        "triple_accessed": False,
        "official_foundir_test_accessed": False,
        "sealed_internal_accessed": False,
        "gpu_used": False,
    }
    write_or_verify(ROOT / CONFIG["output_metadata"], (json.dumps(metadata, indent=2, sort_keys=True) + "\n").encode())
    return metadata


def check(full_hash: bool) -> dict[str, Any]:
    prepare_globals()
    metadata = json.loads((ROOT / CONFIG["output_metadata"]).read_text())
    if metadata["config_sha256"] != CONFIG_SHA:
        raise RuntimeError("tensor-shard config identity changed")
    manifest = ROOT / metadata["output_manifest"]
    if sha256(manifest) != metadata["output_manifest_sha256"]:
        raise RuntimeError("tensor-shard manifest identity changed")
    rows = load_jsonl(manifest)
    return {**metadata, "recheck": validate(CONFIG, rows, full_hash=full_hash)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--full-hash", action="store_true")
    args = parser.parse_args()
    config = json.loads(CONFIG_PATH.read_text())
    if args.check:
        result = check(args.full_hash)
    else:
        workers = args.workers if args.workers is not None else int(config["default_workers"])
        if workers < 1 or workers > 32:
            parser.error("workers must be in [1, 32]")
        result = build(workers)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
