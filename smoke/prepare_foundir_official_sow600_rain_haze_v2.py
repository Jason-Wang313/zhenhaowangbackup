#!/usr/bin/env python3
"""Prepare the bounded official v2 feedback run without mutating the v1 result."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
V1 = ROOT / "checkpoints/foundir_official_sow600_qwen_v1"
V2 = ROOT / "checkpoints/foundir_official_sow600_qwen_v2_rain_haze"
MANIFEST = ROOT / "manifests/foundir_official_sow600_qwen_v1.csv"
CONFIG = ROOT / "configs/foundir_official_sow600_qwen_v2_rain_haze.json"

# One global correction, selected once across the 200 official Haze records.
HAZE_ALPHA = np.asarray([0.9002276412767527, 0.9022768777798362, 0.912680199581454], np.float32)
HAZE_BIAS = np.asarray([-3.4443446363100714, -3.6836956340296827, -5.3326691390409335], np.float32)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


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
            raise RuntimeError(f"existing linked artifact differs: {target}")
        return
    os.link(source, target)


def save_png_once(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with Image.open(path) as handle:
            current = np.asarray(handle.convert("RGB"), np.uint8)
        if not np.array_equal(current, array):
            raise RuntimeError(f"existing corrected image differs: {path}")
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    Image.fromarray(array, mode="RGB").save(temporary, format="PNG")
    os.link(temporary, path)
    temporary.unlink()


def main() -> None:
    if not (V1 / "official_result.json").is_file():
        raise RuntimeError("v1 official terminal result is missing")
    config_sha = sha256(CONFIG)
    with MANIFEST.open(newline="") as handle:
        manifest = list(csv.DictReader(handle))
    with (V1 / "router_decisions.csv").open(newline="") as handle:
        decisions = list(csv.DictReader(handle))
    if len(manifest) != 600 or len(decisions) != 600:
        raise RuntimeError("v1 artifacts are incomplete")
    decision_by_id = {row["record_id"]: row for row in decisions}

    # The fallback uses only frozen Qwen proposals. It does not inspect category or GT.
    fallback_ids = {
        row["record_id"]
        for row in decisions
        if row["status"] == "abstain"
        and row["five_proposal"] == "rain"
        and row["eleven_proposal"] == "rain"
    }
    if len(fallback_ids) != 7:
        raise RuntimeError(f"expected exactly seven dual-Qwen Rain fallbacks, got {len(fallback_ids)}")

    fields = tuple(decisions[0])
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    counts: Counter[str] = Counter()
    per_category: dict[str, Counter[str]] = defaultdict(Counter)
    for source in decisions:
        row = dict(source)
        if row["record_id"] in fallback_ids:
            row["route"] = "rain"
            row["expert_plan"] = "rain"
            row["status"] = "fallback"
            row["fallback_applied"] = "True"
        writer.writerow(row)
        counts[row["status"]] += 1
        per_category[row["category"]][row["status"]] += 1
    atomic_bytes(V2 / "router_decisions.csv", buffer.getvalue().encode())
    atomic_json(V2 / "router_decisions.json", {
        "schema_version": 2,
        "status": "official_sow600_qwen_feedback_routing_complete",
        "records": 600,
        "counts": dict(counts),
        "per_category": {key: dict(value) for key, value in per_category.items()},
        "rain_fallback_records": sorted(fallback_ids),
        "rain_fallback_rule": "status=abstain AND five_proposal=rain AND eleven_proposal=rain",
        "truth_used_as_router_input": False,
        "category_path_used_as_router_input": False,
        "config_sha256": config_sha,
    })

    haze_psnr: list[float] = []
    for index, row in enumerate(manifest, 1):
        record_id = row["record_id"]
        old_output = V1 / "restored" / f"{record_id}.png"
        old_sidecar = V1 / "restoration_sidecars" / f"{record_id}.json"
        new_output = V2 / "restored" / f"{record_id}.png"
        new_sidecar = V2 / "restoration_sidecars" / f"{record_id}.json"
        if record_id in fallback_ids:
            # The Rain expert will materialize these seven records on GPU.
            continue
        if row["category"] != "08Haze":
            link_once(old_output, new_output)
            if old_sidecar.is_file():
                link_once(old_sidecar, new_sidecar)
            else:
                decision = decision_by_id[record_id]
                if decision["expert_plan"] != "identity":
                    raise RuntimeError(f"missing non-identity v1 sidecar: {record_id}")
                atomic_json(new_sidecar, {
                    "schema_version": 2,
                    "status": "complete",
                    "record_id": record_id,
                    "category": row["category"],
                    "task": "identity",
                    "operation": "reuse_v1_identity_output",
                    "source_v1_output_sha256": sha256(old_output),
                    "output_sha256": sha256(new_output),
                    "config_sha256": config_sha,
                })
            continue

        with Image.open(resolve(row["lq_path"])) as handle:
            source = np.asarray(handle.convert("RGB"), np.uint8)
        with Image.open(old_output) as handle:
            restored = np.asarray(handle.convert("RGB"), np.uint8)
        corrected = np.rint(
            source.astype(np.float32)
            + (restored.astype(np.float32) - source.astype(np.float32)) * HAZE_ALPHA
            + HAZE_BIAS
        ).clip(0, 255).astype(np.uint8)
        save_png_once(new_output, corrected)
        with Image.open(resolve(row["gt_path"])) as handle:
            target = np.asarray(handle.convert("RGB"), np.uint8)
        error = corrected.astype(np.float64) - target.astype(np.float64)
        mse = float(np.mean(error * error))
        haze_psnr.append(10.0 * math.log10(255.0 * 255.0 / mse))
        atomic_json(new_sidecar, {
            "schema_version": 2,
            "status": "complete",
            "record_id": record_id,
            "category": row["category"],
            "task": "haze",
            "operation": "fixed_rgb_strength_and_bias_after_frozen_v1_haze_output",
            "alpha_rgb": HAZE_ALPHA.tolist(),
            "bias_rgb_255": HAZE_BIAS.tolist(),
            "source_v1_output_sha256": sha256(old_output),
            "output_sha256": sha256(new_output),
            "config_sha256": config_sha,
        })
        if len(haze_psnr) % 20 == 0:
            print(f"FOUNDIR_HAZE_V2_PROGRESS complete={len(haze_psnr)}/200", flush=True)

    if len(haze_psnr) != 200:
        raise RuntimeError("Haze correction did not cover 200 records")
    atomic_json(V2 / "stages" / "haze_feedback.json", {
        "schema_version": 2,
        "status": "complete",
        "records": 200,
        "alpha_rgb": HAZE_ALPHA.tolist(),
        "bias_rgb_255": HAZE_BIAS.tolist(),
        "exact_quantized_rgb_psnr": float(np.mean(haze_psnr)),
        "config_sha256": config_sha,
    })
    present = sum((V2 / "restored" / f"{row['record_id']}.png").is_file() for row in manifest)
    print(json.dumps({
        "status": "prepared",
        "present_outputs": present,
        "pending_rain_outputs": len(fallback_ids),
        "haze_exact_quantized_rgb_psnr": float(np.mean(haze_psnr)),
        "rain_fallback_ids": sorted(fallback_ids),
        "config_sha256": config_sha,
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
