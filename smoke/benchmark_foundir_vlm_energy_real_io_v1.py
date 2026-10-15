#!/usr/bin/env python3
"""Read-only paired-image I/O benchmark over the frozen real train split."""

from __future__ import annotations

import csv
import concurrent.futures
import hashlib
import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/foundir_vlm_energy_real_generalist_v1.json"


def decode_pair(task: tuple[str, str, str, int]) -> tuple[int, int]:
    record_id, lq_text, gt_text, crop = task
    decoded_bytes = checksum = 0
    arrays = []
    for text in (lq_text, gt_text):
        with Image.open(ROOT / text) as handle:
            array = np.asarray(handle.convert("RGB"), dtype=np.uint8)
        if min(array.shape[:2]) < crop:
            raise RuntimeError(f"undersized real pair: {record_id}")
        y = (array.shape[0] - crop) // 2
        x = (array.shape[1] - crop) // 2
        arrays.append(np.ascontiguousarray(array[y:y+crop, x:x+crop]))
        decoded_bytes += array.nbytes
    checksum = int(arrays[0][0, 0, 0]) + int(arrays[1][-1, -1, -1])
    return decoded_bytes, checksum


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    cfg = json.loads(CONFIG.read_text())
    manifest = ROOT / cfg["data"]["manifest"]
    if sha256(manifest) != cfg["data"]["manifest_sha256"]:
        raise RuntimeError("manifest identity mismatch")
    groups: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    with manifest.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["split"] == "train":
                groups[row["category"]].append((row["record_id"], row["lq_path"], row["gt_path"]))
    count = int(os.environ.get("VLM_ENERGY_REAL_IO_RECORDS", "4096"))
    rng = random.Random(2026101101)
    categories = sorted(groups)
    selected = []
    for index in range(count):
        category = categories[index % len(categories)]
        selected.append((category, *groups[category][rng.randrange(len(groups[category]))]))
    rng.shuffle(selected)
    crop = int(cfg["data"]["crop_size"])
    workers = int(os.environ.get("VLM_ENERGY_REAL_IO_WORKERS", "32"))
    started = time.perf_counter()
    decoded_bytes = checksum = 0
    tasks = [(record_id, lq_text, gt_text, crop) for _, record_id, lq_text, gt_text in selected]
    if workers == 1:
        results = map(decode_pair, tasks)
        for pair_bytes, pair_checksum in results:
            decoded_bytes += pair_bytes
            checksum = (checksum + pair_checksum) % (1 << 32)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            for pair_bytes, pair_checksum in executor.map(decode_pair, tasks, chunksize=1):
                decoded_bytes += pair_bytes
                checksum = (checksum + pair_checksum) % (1 << 32)
    elapsed = time.perf_counter() - started
    throughput = len(selected) / elapsed
    gate_pass = throughput >= 10.0
    result = {
        "status": (
            "foundir_vlm_energy_real_io_benchmark_pass"
            if gate_pass else "foundir_vlm_energy_real_io_benchmark_fail"
        ),
        "records": len(selected),
        "workers": workers,
        "paired_images": 2 * len(selected),
        "elapsed_seconds": elapsed,
        "pairs_per_second": throughput,
        "minimum_required_pairs_per_second": 10.0,
        "throughput_gate_pass": gate_pass,
        "decoded_gib_per_second": decoded_bytes / elapsed / (1 << 30),
        "checksum": checksum,
        "hostname": os.uname().nodename,
        "manifest_sha256": sha256(manifest),
        "official_foundir_test_accessed": False,
        "sealed_internal_accessed": False,
        "internal_test_accessed": False,
    }
    run_id = os.environ.get("VLM_ENERGY_REAL_IO_RUN_ID", os.uname().nodename)
    output = ROOT / "checkpoints/foundir_vlm_energy_real_generalist_v1/io_benchmark" / f"{run_id}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not gate_pass:
        raise RuntimeError("real-data I/O throughput gate failed")


if __name__ == "__main__":
    main()
