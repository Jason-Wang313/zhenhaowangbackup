#!/usr/bin/env python3
"""Inherit frozen non-Noise v4 outputs and leave exactly 100 Noise records pending."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
V4 = ROOT / "checkpoints/foundir_official_sow600_qwen_v4_noise_mse"
V5 = ROOT / "checkpoints/foundir_official_sow600_qwen_v5_noise_mse_lpips"
MANIFEST = ROOT / "manifests/foundir_official_sow600_qwen_v1.csv"
CONFIG = ROOT / "configs/foundir_official_sow600_qwen_v5_noise_mse_lpips.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def link_once(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if sha256(source) != sha256(target):
            raise RuntimeError(f"existing inherited artifact differs: {target}")
        return
    os.link(source, target)


def main() -> None:
    if not (V4 / "official_result.json").is_file():
        raise RuntimeError("v4 parent result is missing")
    with MANIFEST.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    inherited = pending = 0
    for row in rows:
        record_id = row["record_id"]
        if row["category"] == "05Noise":
            pending += 1
            continue
        for directory, suffix in (("restored", ".png"), ("restoration_sidecars", ".json")):
            link_once(V4 / directory / f"{record_id}{suffix}", V5 / directory / f"{record_id}{suffix}")
        inherited += 1
    if inherited != 500 or pending != 100:
        raise RuntimeError("v5 preparation escaped the one-category Noise change")
    print(json.dumps({
        "status": "prepared",
        "config_sha256": sha256(CONFIG),
        "inherited_outputs": inherited,
        "pending_noise_outputs": pending,
        "alpha": 1.0,
        "alpha_sweep": False,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
