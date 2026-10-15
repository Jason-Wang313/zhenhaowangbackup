#!/usr/bin/env python3
"""Evaluate the frozen conservative Qwen router on a fresh training holdout."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lowlight_restoration.qwen_five_class_gate import CLASSES, routing_metrics
from smoke.run_foundir_qwen_five_class_gate_ddp import (
    extract,
    feature_metadata,
    feature_path,
    load_feature,
    probabilities,
    sha256,
    workspace_path,
)

def load_contract(path: Path):
    config = json.loads(path.read_text())
    config_sha = sha256(path)
    if config.get("schema_version") != 1:
        raise ValueError("bad v2 gate config schema")
    data = config["data"]
    manifest = workspace_path(data["manifest"])
    v1_manifest = workspace_path(data["excluded_v1_manifest"])
    head = workspace_path(config["router"]["head"])
    v1_result = workspace_path(config["router"]["v1_result"])
    qwen_config = workspace_path(config["qwen"]["training_config"])
    adapter = workspace_path(config["qwen"]["adapter"])
    checks = (
        (manifest, data["manifest_sha256"]),
        (v1_manifest, data["excluded_v1_manifest_sha256"]),
        (head, config["router"]["head_sha256"]),
        (v1_result, config["router"]["v1_result_sha256"]),
        (qwen_config, config["qwen"]["training_config_sha256"]),
        (adapter, config["qwen"]["adapter_sha256"]),
    )
    for artifact, expected in checks:
        if sha256(artifact) != expected:
            raise RuntimeError(f"frozen artifact changed: {artifact}")
    with manifest.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    with v1_manifest.open(newline="") as handle:
        used_sources = {row["source_group"] for row in csv.DictReader(handle)}
    if len(rows) != 300 or any(row["head_split"] != "fresh_holdout" for row in rows):
        raise RuntimeError("fresh holdout identity changed")
    if {row["source_group"] for row in rows} & used_sources:
        raise RuntimeError("fresh holdout overlaps a v1 source")
    if any(sum(row["label"] == label for row in rows) != 60 for label in CLASSES):
        raise RuntimeError("fresh holdout is not class balanced")
    if float(config["router"]["threshold"]) <= 0.925:
        raise RuntimeError("v2 threshold was not made more conservative")
    return config, config_sha, rows


def evaluate(config, config_sha, rows):
    result_path = workspace_path(config["outputs"]["result"])
    if result_path.exists():
        raise FileExistsError("refusing to replace v2 terminal result")
    features = np.stack(
        [
            load_feature(
                feature_path(config, row["record_id"]),
                feature_metadata(config, config_sha, row),
                int(config["features"]["feature_dim"]),
            ).numpy()
            for row in rows
        ]
    )
    head_path = workspace_path(config["router"]["head"])
    with safe_open(head_path, framework="pt", device="cpu") as handle:
        coefficient = handle.get_tensor("coefficient").numpy()
        intercept = handle.get_tensor("intercept").numpy()
        mean = handle.get_tensor("feature_mean").numpy()
        scale = handle.get_tensor("feature_scale").numpy()
    logits = ((features - mean) / scale) @ coefficient.T + intercept
    values = probabilities(logits, float(config["router"]["temperature"]))
    labels = [CLASSES.index(row["label"]) for row in rows]
    metrics = routing_metrics(
        values.tolist(), labels, threshold=float(config["router"]["threshold"])
    )
    gates = config["gates"]
    overall = metrics["overall"]
    per_class = metrics["per_class"]
    checks = {
        "overall_wrong_rate": overall["wrong_rate"] <= gates["maximum_wrong_rate"],
        "per_class_wrong_rate": all(
            value["wrong_rate"] <= gates["maximum_per_class_wrong_rate"]
            for value in per_class.values()
        ),
        "overall_coverage": overall["coverage"] >= gates["minimum_coverage"],
        "per_class_coverage": all(
            value["coverage"] >= gates["minimum_per_class_coverage"]
            for value in per_class.values()
        ),
    }
    payload = {
        "schema_version": 1,
        "status": "fresh_training_source_holdout_complete",
        "gate_pass": all(checks.values()),
        "checks": checks,
        "metrics": metrics,
        "gates": gates,
        "config_sha256": config_sha,
        "manifest_sha256": config["data"]["manifest_sha256"],
        "head_sha256": config["router"]["head_sha256"],
        "adapter_sha256": config["qwen"]["adapter_sha256"],
        "v1_source_overlap": 0,
        "development_or_internal_test_used": False,
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print("FOUNDIR_QWEN_FIVE_CLASS_FRESH_HOLDOUT_V2 " + json.dumps(payload, sort_keys=True), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    config, config_sha, rows = load_contract(args.config.resolve())
    if args.check_only:
        print(f"FOUNDIR_QWEN_FIVE_CLASS_FRESH_HOLDOUT_V2_PREFLIGHT_PASS records={len(rows)} config_sha256={config_sha}")
        return
    if any(name not in os.environ for name in ("RANK", "WORLD_SIZE", "LOCAL_RANK")):
        raise RuntimeError("launch with torchrun")
    rank, world, local_rank = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), int(os.environ["LOCAL_RANK"])
    if world != 8 or not torch.cuda.is_available() or torch.cuda.device_count() != 8:
        raise RuntimeError("fresh holdout requires exactly 8 visible GPUs")
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dist.init_process_group("nccl", device_id=device)
    extract(config, config_sha, rows, rank, world, device)
    dist.barrier()
    if rank == 0:
        evaluate(config, config_sha, rows)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
