#!/usr/bin/env python3
"""Prepare all-in-one v3 by swapping only the Noise path to Restormer."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "checkpoints/foundir_official_sow600_qwen_v2_rain_haze"
V3 = ROOT / "checkpoints/foundir_official_sow600_qwen_v3_noise_restormer"
MANIFEST = ROOT / "manifests/foundir_official_sow600_qwen_v1.csv"
CONFIG = ROOT / "configs/foundir_official_sow600_qwen_v3_noise_restormer.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise RuntimeError(f"refusing to replace differing artifact: {path}")
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    temporary.write_bytes(payload)
    os.link(temporary, path)
    temporary.unlink()


def atomic_json(path: Path, payload: dict) -> None:
    atomic_bytes(path, (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode())


def link_once(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if sha256(source) != sha256(target):
            raise RuntimeError(f"existing inherited artifact differs: {target}")
        return
    os.link(source, target)


def main() -> None:
    if not (V2 / "official_result.json").is_file():
        raise RuntimeError("successful Rain/Haze v2 parent result is missing")
    config_sha = sha256(CONFIG)
    with MANIFEST.open(newline="") as handle:
        manifest = list(csv.DictReader(handle))
    with (V2 / "router_decisions.csv").open(newline="") as handle:
        decisions = list(csv.DictReader(handle))
    if len(manifest) != 600 or len(decisions) != 600:
        raise RuntimeError("parent artifacts are incomplete")

    fallback_ids = {
        row["record_id"]
        for row in decisions
        if row["status"] == "abstain" and row["five_proposal"] == "noise"
    }
    if len(fallback_ids) != 50:
        raise RuntimeError(f"expected 50 Qwen Noise fallbacks, got {len(fallback_ids)}")

    fields = tuple(decisions[0])
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    counts: Counter[str] = Counter()
    per_category: dict[str, Counter[str]] = defaultdict(Counter)
    for source in decisions:
        row = dict(source)
        if row["record_id"] in fallback_ids:
            row["route"] = "noise"
            row["expert_plan"] = "noise"
            row["status"] = "fallback"
            row["fallback_applied"] = "True"
        writer.writerow(row)
        counts[row["status"]] += 1
        per_category[row["category"]][row["status"]] += 1
    atomic_bytes(V3 / "router_decisions.csv", buffer.getvalue().encode())
    atomic_json(V3 / "router_decisions.json", {
        "schema_version": 3,
        "status": "official_sow600_qwen_feedback_routing_complete",
        "records": 600,
        "counts": dict(counts),
        "per_category": {key: dict(value) for key, value in per_category.items()},
        "noise_fallback_records": sorted(fallback_ids),
        "noise_fallback_rule": "status=abstain AND five_proposal=noise",
        "truth_used_as_router_input": False,
        "category_path_used_as_router_input": False,
        "config_sha256": config_sha,
    })

    inherited = 0
    pending = 0
    for row in manifest:
        record_id = row["record_id"]
        if row["category"] == "05Noise":
            pending += 1
            continue
        for directory, suffix in (("restored", ".png"), ("restoration_sidecars", ".json")):
            link_once(V2 / directory / f"{record_id}{suffix}", V3 / directory / f"{record_id}{suffix}")
        inherited += 1
    if inherited != 500 or pending != 100:
        raise RuntimeError("v3 inheritance escaped the one-category Noise change")
    print(json.dumps({
        "status": "prepared",
        "inherited_outputs": inherited,
        "pending_noise_outputs": pending,
        "noise_fallbacks": len(fallback_ids),
        "config_sha256": config_sha,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
