#!/usr/bin/env python3
"""Build/verify immutable rendered targets for the 66k lower-order EBT corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CONFIG_PATH = ROOT / "configs/foundir_compositional_ebt_scaled_targets_v1.json"

HEADLESS_CV2_ROOT = "/usr/local/lib/python3.12/dist-packages"
sys.path.insert(0, HEADLESS_CV2_ROOT)
try:
    import cv2  # noqa: E402
finally:
    sys.path.remove(HEADLESS_CV2_ROOT)

from lowlight_restoration.multidegradation_data import generate_record_stages  # noqa: E402


RECORDS: list[dict[str, Any]] = []
OUTPUT_ROOT: Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def target_path(record: dict[str, Any]) -> Path:
    return OUTPUT_ROOT / str(record["split"]) / f"{record['record_id']}.png"


def build_one(index: int) -> dict[str, Any]:
    record = RECORDS[index]
    if int(record["composition_size"]) not in {1, 2} or record["split"] not in {"train", "validation"}:
        raise RuntimeError(f"prohibited record entered target cache: {record['record_id']}")
    image = generate_record_stages(record, application_order=[])[0]
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError(f"target PNG encode failed: {record['record_id']}")
    payload = encoded.tobytes()
    output = target_path(record)
    write_or_verify(output, payload)
    digest = hashlib.sha256(payload).hexdigest()
    if sha256(output) != digest:
        raise RuntimeError(f"target write verification failed: {output}")
    return {
        "record_id": record["record_id"],
        "split": record["split"],
        "composition_size": record["composition_size"],
        "degradations": record["degradations"],
        "target_path": output.relative_to(ROOT).as_posix(),
        "target_sha256": digest,
        "height": int(image.shape[0]),
        "width": int(image.shape[1]),
        "channels": int(image.shape[2]),
    }


def validate_rows(config: dict[str, Any], rows: list[dict[str, Any]], *, full_hash: bool) -> dict[str, Any]:
    if len(rows) != int(config["expected_records"]):
        raise RuntimeError("target cache record count changed")
    counts = Counter(str(row["split"]) for row in rows)
    if dict(counts) != config["expected_split_records"]:
        raise RuntimeError(f"target cache split counts changed: {counts}")
    identifiers = [str(row["record_id"]) for row in rows]
    if len(set(identifiers)) != len(identifiers):
        raise RuntimeError("duplicate target cache record id")
    for row in rows:
        if int(row["composition_size"]) not in {1, 2}:
            raise RuntimeError("triple entered target cache")
        if (int(row["height"]), int(row["width"]), int(row["channels"])) != (448, 448, 3):
            raise RuntimeError(f"target shape changed: {row['record_id']}")
        path = ROOT / str(row["target_path"])
        if not path.is_file():
            raise RuntimeError(f"missing target: {path}")
        if full_hash and sha256(path) != row["target_sha256"]:
            raise RuntimeError(f"target hash mismatch: {path}")
    return {
        "records": len(rows),
        "split_records": dict(counts),
        "full_hash_verification": full_hash,
        "triple_records": 0,
        "official_foundir_test_accessed": False,
        "sealed_internal_accessed": False,
        "gpu_used": False,
    }


def build(workers: int) -> dict[str, Any]:
    global RECORDS, OUTPUT_ROOT
    config = json.loads(CONFIG_PATH.read_text())
    if cv2.__version__ != config["headless_cv2_version"] or not cv2.__file__.startswith(config["headless_cv2_module_root"]):
        raise RuntimeError(f"headless OpenCV identity changed: {cv2.__file__} {cv2.__version__}")
    manifest = ROOT / config["input_manifest"]
    metadata = ROOT / config["input_metadata"]
    if sha256(manifest) != config["input_manifest_sha256"] or sha256(metadata) != config["input_metadata_sha256"]:
        raise RuntimeError("input data contract identity changed")
    RECORDS = load_jsonl(manifest)
    OUTPUT_ROOT = ROOT / config["output_root"]
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    context = mp.get_context("fork")
    with context.Pool(processes=workers) as pool:
        for completed, row in enumerate(pool.imap(build_one, range(len(RECORDS)), chunksize=8), start=1):
            rows.append(row)
            if completed == 1 or completed % 1000 == 0 or completed == len(RECORDS):
                elapsed = time.perf_counter() - started
                print("EBT_TARGET_CACHE_PROGRESS " + json.dumps({
                    "completed": completed,
                    "total": len(RECORDS),
                    "elapsed_seconds": elapsed,
                    "estimated_total_seconds": elapsed / completed * len(RECORDS),
                }, sort_keys=True), flush=True)
    rows.sort(key=lambda row: (str(row["split"]), str(row["record_id"])))
    checks = validate_rows(config, rows, full_hash=True)
    payload = b"".join(canonical_line(row) for row in rows)
    output_manifest = ROOT / config["output_manifest"]
    write_or_verify(output_manifest, payload)
    result = {
        "schema_version": 1,
        "status": "foundir_compositional_ebt_scaled_targets_v1_ready",
        "config_sha256": sha256(CONFIG_PATH),
        "input_manifest": config["input_manifest"],
        "input_manifest_sha256": config["input_manifest_sha256"],
        "output_manifest": config["output_manifest"],
        "output_manifest_sha256": hashlib.sha256(payload).hexdigest(),
        "output_root": config["output_root"],
        "headless_cv2_version": cv2.__version__,
        "workers": workers,
        "elapsed_seconds": time.perf_counter() - started,
        "checks": checks,
        "triple_accessed": False,
        "official_foundir_test_accessed": False,
        "sealed_internal_accessed": False,
        "gpu_used": False,
    }
    write_or_verify(ROOT / config["output_metadata"], (json.dumps(result, indent=2, sort_keys=True) + "\n").encode())
    return result


def check(full_hash: bool) -> dict[str, Any]:
    global OUTPUT_ROOT
    config = json.loads(CONFIG_PATH.read_text())
    OUTPUT_ROOT = ROOT / config["output_root"]
    metadata = json.loads((ROOT / config["output_metadata"]).read_text())
    if metadata["config_sha256"] != sha256(CONFIG_PATH):
        raise RuntimeError("target cache config identity changed")
    manifest = ROOT / metadata["output_manifest"]
    if sha256(manifest) != metadata["output_manifest_sha256"]:
        raise RuntimeError("target cache manifest identity changed")
    checks = validate_rows(config, load_jsonl(manifest), full_hash=full_hash)
    return {**metadata, "recheck": checks}


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
        if workers < 1 or workers > 64:
            parser.error("workers must be in [1, 64]")
        result = build(workers)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
