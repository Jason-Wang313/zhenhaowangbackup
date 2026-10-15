#!/usr/bin/env python3
"""Reuse lower-order targets and render only missing triple targets."""

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
CONFIG_PATH = ROOT / "configs/foundir_vlm_energy_full_order_targets_v1.json"
HEADLESS_CV2_ROOT = "/usr/local/lib/python3.12/dist-packages"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, HEADLESS_CV2_ROOT)
try:
    import cv2  # noqa: E402
finally:
    sys.path.remove(HEADLESS_CV2_ROOT)

from lowlight_restoration.multidegradation_data import generate_record_stages  # noqa: E402


CONFIG: dict[str, Any] = {}
TRIPLES: list[dict[str, Any]] = []
OUTPUT_ROOT: Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def canonical(payload: object) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()


def write_once(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise RuntimeError(f"refusing to overwrite differing artifact: {path}")
        return
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def output_path(row: dict[str, Any]) -> Path:
    return OUTPUT_ROOT / str(row["split"]) / f"{row['record_id']}.png"


def build_one(index: int) -> dict[str, Any]:
    row = TRIPLES[index]
    if int(row["composition_size"]) != 3 or row["split"] not in {"train", "validation"}:
        raise RuntimeError("non-development triple entered target renderer")
    path = output_path(row)
    if path.is_file():
        target = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if target is None:
            raise RuntimeError(f"existing target decode failed: {path}")
        digest = sha256(path)
    else:
        target = generate_record_stages(row, application_order=[])[0]
        ok, encoded = cv2.imencode(".png", target)
        if not ok:
            raise RuntimeError(f"target encoding failed: {row['record_id']}")
        payload = encoded.tobytes()
        write_once(path, payload)
        digest = hashlib.sha256(payload).hexdigest()
        if sha256(path) != digest:
            raise RuntimeError(f"target verification failed: {path}")
    return {
        "record_id": row["record_id"],
        "split": row["split"],
        "composition_size": 3,
        "degradations": row["degradations"],
        "target_path": path.relative_to(ROOT).as_posix(),
        "target_sha256": digest,
        "height": int(target.shape[0]),
        "width": int(target.shape[1]),
        "channels": int(target.shape[2]),
    }


def prepare() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    global CONFIG, TRIPLES, OUTPUT_ROOT
    CONFIG = json.loads(CONFIG_PATH.read_text())
    if cv2.__version__ != CONFIG["headless_cv2_version"] or not cv2.__file__.startswith(CONFIG["headless_cv2_module_root"]):
        raise RuntimeError(f"headless OpenCV identity changed: {cv2.__file__} {cv2.__version__}")
    full_manifest = ROOT / CONFIG["input_manifest"]
    lower_manifest = ROOT / CONFIG["lower_order_target_manifest"]
    if sha256(full_manifest) != CONFIG["input_manifest_sha256"]:
        raise RuntimeError("full-order manifest identity changed")
    if sha256(lower_manifest) != CONFIG["lower_order_target_manifest_sha256"]:
        raise RuntimeError("lower-order target manifest identity changed")
    records = load_jsonl(full_manifest)
    lower = load_jsonl(lower_manifest)
    TRIPLES = [row for row in records if int(row["composition_size"]) == 3]
    OUTPUT_ROOT = ROOT / CONFIG["output_root"]
    if len(records) != int(CONFIG["expected_records"]):
        raise RuntimeError("full-order record count changed")
    if len(lower) != int(CONFIG["expected_reused_lower_order"]):
        raise RuntimeError("lower-order target count changed")
    if len(TRIPLES) != int(CONFIG["expected_generated_triple"]):
        raise RuntimeError("triple target count changed")
    record_ids = {str(row["record_id"]) for row in records}
    if any(str(row["record_id"]) not in record_ids for row in lower):
        raise RuntimeError("lower-order target is outside full-order contract")
    return records, lower


def validate(rows: list[dict[str, Any]], *, full_hash: bool) -> dict[str, Any]:
    if len(rows) != int(CONFIG["expected_records"]):
        raise RuntimeError("unified target count changed")
    identifiers = [str(row["record_id"]) for row in rows]
    if len(set(identifiers)) != len(identifiers):
        raise RuntimeError("duplicate target record")
    counts = Counter((str(row["split"]), int(row["composition_size"])) for row in rows)
    expected = {
        ("train", 1): 30000, ("train", 2): 30000, ("train", 3): 30000,
        ("validation", 1): 3000, ("validation", 2): 3000, ("validation", 3): 3000,
    }
    if counts != expected:
        raise RuntimeError(f"target split/order counts changed: {counts}")
    for row in rows:
        if (int(row["height"]), int(row["width"]), int(row["channels"])) != (448, 448, 3):
            raise RuntimeError(f"target shape changed: {row['record_id']}")
        path = ROOT / str(row["target_path"])
        if not path.is_file():
            raise RuntimeError(f"missing target: {path}")
        if full_hash and sha256(path) != row["target_sha256"]:
            raise RuntimeError(f"target hash mismatch: {path}")
    return {
        "records": len(rows),
        "reused_lower_order": int(CONFIG["expected_reused_lower_order"]),
        "generated_triple": int(CONFIG["expected_generated_triple"]),
        "full_hash_verification": full_hash,
        "parent_test_records": 0,
        "official_foundir_test_accessed": False,
        "sealed_internal_accessed": False,
    }


def build(workers: int) -> dict[str, Any]:
    _, lower = prepare()
    started = time.perf_counter()
    triples = []
    with mp.get_context("fork").Pool(workers) as pool:
        for completed, row in enumerate(pool.imap(build_one, range(len(TRIPLES)), chunksize=8), start=1):
            triples.append(row)
            if completed == 1 or completed % 1000 == 0 or completed == len(TRIPLES):
                elapsed = time.perf_counter() - started
                print("VLM_ENERGY_TARGET_PROGRESS " + json.dumps({
                    "completed": completed,
                    "total": len(TRIPLES),
                    "elapsed_seconds": elapsed,
                    "estimated_total_seconds": elapsed / completed * len(TRIPLES),
                }, sort_keys=True), flush=True)
    rows = sorted(lower + triples, key=lambda row: (str(row["split"]), str(row["record_id"])))
    # The reused 66k rows are already covered by the immutable lower-order
    # target manifest, whose build performed full SHA-256 verification.  Each
    # new triple is hashed in build_one.  Reopening all 99k files here would
    # duplicate evidence while imposing several minutes of metadata I/O.
    checks = validate(rows, full_hash=False)
    checks["reused_lower_order_hash_evidence"] = CONFIG["lower_order_target_manifest_sha256"]
    checks["generated_triples_hashed_during_build"] = True
    payload = b"".join(canonical(row) for row in rows)
    write_once(ROOT / CONFIG["output_manifest"], payload)
    result = {
        "schema_version": 1,
        "status": "foundir_vlm_energy_full_order_targets_v1_ready",
        "config_sha256": sha256(CONFIG_PATH),
        "input_manifest_sha256": CONFIG["input_manifest_sha256"],
        "lower_order_target_manifest_sha256": CONFIG["lower_order_target_manifest_sha256"],
        "output_manifest": CONFIG["output_manifest"],
        "output_manifest_sha256": hashlib.sha256(payload).hexdigest(),
        "output_root": CONFIG["output_root"],
        "workers": workers,
        "elapsed_seconds": time.perf_counter() - started,
        "checks": checks,
        "training_launched": False,
        "official_foundir_test_accessed": False,
        "sealed_internal_accessed": False,
    }
    write_once(ROOT / CONFIG["output_metadata"], (json.dumps(result, indent=2, sort_keys=True) + "\n").encode())
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int)
    args = parser.parse_args()
    config = json.loads(CONFIG_PATH.read_text())
    workers = int(args.workers or config["default_workers"])
    if not 1 <= workers <= 64:
        parser.error("workers must be in [1, 64]")
    print(json.dumps(build(workers), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
