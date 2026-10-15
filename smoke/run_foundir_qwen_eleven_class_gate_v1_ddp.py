#!/usr/bin/env python3
"""Fit a fail-closed eleven-class head on frozen Qwen visual features."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from safetensors import safe_open
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lowlight_restoration.qwen_eleven_class_gate import CLASSES, SINGLETON_CLASSES, eleven_class_metrics
from smoke.run_foundir_qwen_five_class_gate_ddp import (
    candidate_key,
    extract,
    feature_metadata,
    feature_path,
    load_feature,
    probabilities,
    save_feature,
    sha256,
    workspace_path,
)


def load_contract(path: Path):
    config = json.loads(path.read_text())
    config_sha = sha256(path)
    if config.get("schema_version") != 1 or tuple(config.get("classes", ())) != CLASSES:
        raise ValueError("bad eleven-class gate config")
    manifest = workspace_path(config["data"]["manifest"])
    singleton_manifest = workspace_path(config["data"]["singleton_manifest"])
    qwen_config = workspace_path(config["qwen"]["training_config"])
    adapter = workspace_path(config["qwen"]["adapter"])
    for artifact, expected in (
        (manifest, config["data"]["manifest_sha256"]),
        (singleton_manifest, config["data"]["singleton_manifest_sha256"]),
        (qwen_config, config["qwen"]["training_config_sha256"]),
        (adapter, config["qwen"]["adapter_sha256"]),
    ):
        if sha256(artifact) != expected:
            raise RuntimeError(f"frozen artifact changed: {artifact}")
    with manifest.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    quotas = config["data"]["records_per_class"]
    if len(rows) != sum(int(quotas[name]) for name in quotas) * len(CLASSES):
        raise RuntimeError("eleven-class manifest cardinality changed")
    for split, quota in quotas.items():
        for label in CLASSES:
            if sum(row["head_split"] == split and row["label"] == label for row in rows) != int(quota):
                raise RuntimeError(f"unbalanced eleven-class partition: {split}/{label}")
    source_splits: dict[str, set[str]] = {}
    for row in rows:
        source_splits.setdefault(row["source_group"], set()).add(row["head_split"])
    if any(len(value) != 1 for value in source_splits.values()):
        raise RuntimeError("source leakage in eleven-class partition")
    return config, config_sha, rows


def seed_reused_features(config, config_sha, rows):
    if "reused_feature_dirs" in config["features"]:
        source_dirs = [workspace_path(value) for value in config["features"]["reused_feature_dirs"]]
        eligible = lambda row: True
    else:
        source_dirs = [workspace_path(config["features"]["reused_singleton_feature_dir"])]
        eligible = lambda row: row["label"] in SINGLETON_CLASSES
    feature_dim = int(config["features"]["feature_dim"])
    reused = 0
    for row in rows:
        if not eligible(row):
            continue
        target = feature_path(config, row["record_id"])
        expected = feature_metadata(config, config_sha, row)
        if load_feature(target, expected, feature_dim) is not None:
            reused += 1
            continue
        source = next(
            (directory / f"{row['record_id']}.safetensors" for directory in source_dirs if (directory / f"{row['record_id']}.safetensors").exists()),
            None,
        )
        if source is None:
            continue
        with safe_open(source, framework="pt", device="cpu") as handle:
            metadata = handle.metadata() or {}
            tensor = handle.get_tensor("feature")
        required = {
            "adapter_sha256": str(config["qwen"]["adapter_sha256"]),
            "record_id": row["record_id"],
            "lq_sha256": row["lq_sha256"],
            "pooling": "qwen_visual_token_mean_and_std",
            "feature_dim": str(feature_dim),
        }
        if any(metadata.get(key) != value for key, value in required.items()):
            raise RuntimeError(f"reused singleton feature identity changed: {source}")
        save_feature(target, tensor, expected)
        reused += 1
    expected_reused = int(config["features"].get("expected_reused_features", 3150))
    if reused != expected_reused:
        raise RuntimeError(f"expected to reuse {expected_reused} features, got {reused}")
    print(f"QWEN_ELEVEN_CLASS_FEATURES_REUSED count={reused}", flush=True)


def fit_head(config, config_sha, rows):
    from sklearn.linear_model import LogisticRegression

    output = workspace_path(config["outputs"]["result"])
    head_path = workspace_path(config["outputs"]["head"])
    if output.exists() or head_path.exists():
        raise FileExistsError("refusing to replace eleven-class result")
    realized = [
        load_feature(
            feature_path(config, row["record_id"]),
            feature_metadata(config, config_sha, row),
            int(config["features"]["feature_dim"]),
        )
        for row in rows
    ]
    if any(feature is None for feature in realized):
        raise RuntimeError("eleven-class feature extraction is incomplete")
    features = np.stack([feature.numpy() for feature in realized if feature is not None])
    labels = np.asarray([CLASSES.index(row["label"]) for row in rows], dtype=np.int64)
    splits = np.asarray([row["head_split"] for row in rows])
    fit_mask, calibration_mask, holdout_mask = splits == "fit", splits == "calibration", splits == "holdout"
    mean = features[fit_mask].mean(0)
    scale = features[fit_mask].std(0).clip(1e-6)
    x = (features - mean) / scale
    best = None
    for c_value in config["head"]["candidate_c"]:
        classifier = LogisticRegression(C=float(c_value), solver="lbfgs", max_iter=int(config["head"]["maximum_iterations"]), random_state=int(config["head"]["seed"]))
        classifier.fit(x[fit_mask], labels[fit_mask])
        logits = classifier.decision_function(x[calibration_mask])
        for temperature in config["head"]["candidate_temperatures"]:
            values = probabilities(logits, float(temperature))
            for threshold in config["head"]["candidate_thresholds"]:
                metrics = eleven_class_metrics(values.tolist(), labels[calibration_mask].tolist(), threshold=float(threshold))
                item = {"key": candidate_key(metrics, config["gates"]), "c": float(c_value), "temperature": float(temperature), "threshold": float(threshold), "metrics": metrics, "classifier": classifier}
                if best is None or item["key"] > best["key"]:
                    best = item
    assert best is not None
    classifier = best.pop("classifier")
    holdout_values = probabilities(classifier.decision_function(x[holdout_mask]), best["temperature"])
    holdout = eleven_class_metrics(holdout_values.tolist(), labels[holdout_mask].tolist(), threshold=best["threshold"])
    gate_pass = candidate_key(holdout, config["gates"])[0] == 1.0
    head_path.parent.mkdir(parents=True, exist_ok=True)
    save_file({
        "coefficient": torch.from_numpy(classifier.coef_.astype(np.float32)),
        "intercept": torch.from_numpy(classifier.intercept_.astype(np.float32)),
        "feature_mean": torch.from_numpy(mean.astype(np.float32)),
        "feature_scale": torch.from_numpy(scale.astype(np.float32)),
    }, head_path, metadata={
        "config_sha256": config_sha,
        "manifest_sha256": str(config["data"]["manifest_sha256"]),
        "adapter_sha256": str(config["qwen"]["adapter_sha256"]),
        "classes": ",".join(CLASSES),
        "temperature": str(best["temperature"]),
        "threshold": str(best["threshold"]),
    })
    payload = {
        "schema_version": 1,
        "status": "eleven_class_training_only_holdout_complete",
        "gate_pass": gate_pass,
        "selected": best,
        "holdout": holdout,
        "gates": config["gates"],
        "config_sha256": config_sha,
        "manifest_sha256": config["data"]["manifest_sha256"],
        "adapter_sha256": config["qwen"]["adapter_sha256"],
        "head": head_path.relative_to(ROOT).as_posix(),
        "head_sha256": sha256(head_path),
        "development_or_internal_test_used": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print("FOUNDIR_QWEN_ELEVEN_CLASS_TRAIN_ONLY_GATE " + json.dumps(payload, sort_keys=True), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    config, config_sha, rows = load_contract(args.config.resolve())
    if args.check_only:
        print(f"FOUNDIR_QWEN_ELEVEN_CLASS_PREFLIGHT_PASS records={len(rows)} config_sha256={config_sha}")
        return
    required = ("RANK", "WORLD_SIZE", "LOCAL_RANK")
    if any(name not in os.environ for name in required):
        raise RuntimeError("launch with torchrun")
    rank, world, local_rank = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), int(os.environ["LOCAL_RANK"])
    if world != 8 or not torch.cuda.is_available() or torch.cuda.device_count() != 8:
        raise RuntimeError("eleven-class gate requires exactly 8 visible GPUs")
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dist.init_process_group("nccl", device_id=device)
    started = time.monotonic()
    if rank == 0:
        seed_reused_features(config, config_sha, rows)
    dist.barrier()
    extract(config, config_sha, rows, rank, world, device)
    dist.barrier()
    if rank == 0:
        fit_head(config, config_sha, rows)
        print(f"FOUNDIR_QWEN_ELEVEN_CLASS_COMPLETE elapsed_seconds={time.monotonic()-started:.2f}", flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
