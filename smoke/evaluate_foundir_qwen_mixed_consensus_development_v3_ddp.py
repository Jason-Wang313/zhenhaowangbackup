#!/usr/bin/env python3
"""Evaluate the frozen five/eleven-head consensus router on complete development."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lowlight_restoration.qwen_eleven_class_gate import (
    CLASSES,
    MIXED_CLASSES,
    SINGLETON_CLASSES,
    consensus_metrics,
    consensus_route,
)
from smoke.run_foundir_qwen_five_class_gate_ddp import (
    candidate_key,
    extract,
    feature_metadata,
    feature_path,
    load_feature,
    probabilities,
    sha256,
    workspace_path,
)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def load_contract(path: Path):
    config = json.loads(path.read_text())
    config_sha = sha256(path)
    if config.get("schema_version") != 1 or tuple(config.get("classes", ())) != CLASSES:
        raise ValueError("bad consensus config")
    artifacts = (
        (workspace_path(config["data"]["manifest"]), config["data"]["manifest_sha256"]),
        (workspace_path(config["data"]["singleton_manifest"]), config["data"]["singleton_manifest_sha256"]),
        (workspace_path(config["qwen"]["training_config"]), config["qwen"]["training_config_sha256"]),
        (workspace_path(config["qwen"]["adapter"]), config["qwen"]["adapter_sha256"]),
        (workspace_path(config["router"]["five_head"]), config["router"]["five_head_sha256"]),
        (workspace_path(config["router"]["eleven_head"]), config["router"]["eleven_head_sha256"]),
    )
    for artifact, expected in artifacts:
        if sha256(artifact) != expected:
            raise RuntimeError(f"frozen artifact changed: {artifact}")
    mixed = read_rows(artifacts[0][0])
    singleton = read_rows(artifacts[1][0])
    if len(mixed) != int(config["data"]["records"]) or len(singleton) != int(config["data"]["singleton_records"]):
        raise RuntimeError("development cardinality changed")
    if set(row["label"] for row in mixed) != set(MIXED_CLASSES):
        raise RuntimeError("mixed development labels changed")
    if set(row["label"] for row in singleton) != set(SINGLETON_CLASSES):
        raise RuntimeError("singleton development labels changed")
    if len({row["record_id"] for row in mixed + singleton}) != len(mixed) + len(singleton):
        raise RuntimeError("development record identity overlap")
    if config["data"].get("internal_or_official_test_allowed") is not False:
        raise RuntimeError("internal/official test must remain forbidden")
    return config, config_sha, mixed, singleton


def load_raw_feature(path: Path, row: dict[str, str], config: dict) -> torch.Tensor:
    if not path.exists():
        raise FileNotFoundError(f"missing frozen singleton feature: {path}")
    with safe_open(path, framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
        tensor = handle.get_tensor("feature")
    required = {
        "adapter_sha256": str(config["qwen"]["adapter_sha256"]),
        "record_id": row["record_id"],
        "lq_sha256": row["lq_sha256"],
        "pooling": "qwen_visual_token_mean_and_std",
        "feature_dim": str(config["features"]["feature_dim"]),
    }
    if any(metadata.get(key) != value for key, value in required.items()):
        raise RuntimeError(f"singleton feature identity changed: {path}")
    if tensor.shape != (int(config["features"]["feature_dim"]),):
        raise RuntimeError(f"singleton feature shape changed: {path}")
    return tensor


def head_probabilities(features: np.ndarray, path: Path, temperature: float) -> np.ndarray:
    with safe_open(path, framework="pt", device="cpu") as handle:
        tensors = {name: handle.get_tensor(name).numpy() for name in handle.keys()}
    standardized = (features - tensors["feature_mean"]) / tensors["feature_scale"]
    logits = standardized @ tensors["coefficient"].T + tensors["intercept"]
    return probabilities(logits, temperature)


def evaluate(config, config_sha, mixed, singleton):
    result_path = workspace_path(config["outputs"]["result"])
    predictions_path = workspace_path(config["outputs"]["predictions"])
    if result_path.exists() or predictions_path.exists():
        raise FileExistsError("refusing to replace consensus evaluation")
    dim = int(config["features"]["feature_dim"])
    mixed_features = [
        load_feature(feature_path(config, row["record_id"]), feature_metadata(config, config_sha, row), dim)
        for row in mixed
    ]
    if any(value is None for value in mixed_features):
        raise RuntimeError("mixed feature extraction incomplete")
    singleton_dir = workspace_path(config["features"]["singleton_feature_dir"])
    singleton_features = [
        load_raw_feature(singleton_dir / f"{row['record_id']}.safetensors", row, config)
        for row in singleton
    ]
    rows = singleton + mixed
    features = np.stack([value.numpy() for value in singleton_features + mixed_features if value is not None])
    five = head_probabilities(features, workspace_path(config["router"]["five_head"]), float(config["router"]["five_temperature"]))
    eleven = head_probabilities(features, workspace_path(config["router"]["eleven_head"]), float(config["router"]["eleven_temperature"]))
    routes = [
        consensus_route(
            five_row,
            eleven_row,
            five_threshold=float(config["router"]["five_threshold"]),
            eleven_threshold=float(config["router"]["eleven_threshold"]),
        )
        for five_row, eleven_row in zip(five, eleven, strict=True)
    ]
    truths = [row["label"] for row in rows]
    combined = consensus_metrics(routes, truths)
    singleton_metrics = consensus_metrics(routes[: len(singleton)], truths[: len(singleton)])
    mixed_metrics = consensus_metrics(routes[len(singleton) :], truths[len(singleton) :])
    gate_pass = candidate_key(combined, config["gates"])[0] == 1.0
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    with predictions_path.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=(
            "record_id", "truth", "route", "accepted", "five_proposal", "five_confidence",
            "eleven_proposal", "eleven_confidence",
        ))
        writer.writeheader()
        for row, truth, route, p5, p11 in zip(rows, truths, routes, five, eleven, strict=True):
            i5, i11 = int(np.argmax(p5)), int(np.argmax(p11))
            writer.writerow({
                "record_id": row["record_id"], "truth": truth, "route": route or "abstain",
                "accepted": route is not None, "five_proposal": SINGLETON_CLASSES[i5],
                "five_confidence": f"{p5[i5]:.9f}", "eleven_proposal": CLASSES[i11],
                "eleven_confidence": f"{p11[i11]:.9f}",
            })
    payload = {
        "schema_version": 1,
        "status": "complete_development_consensus_evaluated",
        "gate_pass": gate_pass,
        "combined": combined,
        "singleton": singleton_metrics,
        "mixed": mixed_metrics,
        "gates": config["gates"],
        "router": config["router"],
        "config_sha256": config_sha,
        "mixed_manifest_sha256": config["data"]["manifest_sha256"],
        "singleton_manifest_sha256": config["data"]["singleton_manifest_sha256"],
        "predictions": predictions_path.relative_to(ROOT).as_posix(),
        "predictions_sha256": sha256(predictions_path),
        "internal_or_official_test_used": False,
    }
    result_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print("FOUNDIR_QWEN_MIXED_CONSENSUS_RESULT " + json.dumps(payload, sort_keys=True), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    config, config_sha, mixed, singleton = load_contract(args.config.resolve())
    if args.check_only:
        print(f"FOUNDIR_QWEN_MIXED_CONSENSUS_PREFLIGHT_PASS mixed={len(mixed)} singleton={len(singleton)} config_sha256={config_sha}")
        return
    if any(name not in os.environ for name in ("RANK", "WORLD_SIZE", "LOCAL_RANK")):
        raise RuntimeError("launch with torchrun")
    rank, world, local_rank = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), int(os.environ["LOCAL_RANK"])
    if world != 8 or not torch.cuda.is_available() or torch.cuda.device_count() != 8:
        raise RuntimeError("consensus evaluation requires exactly 8 visible GPUs")
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dist.init_process_group("nccl", device_id=device)
    started = time.monotonic()
    extract(config, config_sha, mixed, rank, world, device)
    dist.barrier()
    if rank == 0:
        evaluate(config, config_sha, mixed, singleton)
        print(f"FOUNDIR_QWEN_MIXED_CONSENSUS_COMPLETE elapsed_seconds={time.monotonic()-started:.2f}", flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
