#!/usr/bin/env python3
"""Run the frozen five-class Qwen router on all singleton development records."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter
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
        raise ValueError("bad development gate config schema")
    manifest = workspace_path(config["data"]["manifest"])
    head = workspace_path(config["router"]["head"])
    qwen_config = workspace_path(config["qwen"]["training_config"])
    adapter = workspace_path(config["qwen"]["adapter"])
    for artifact, expected in (
        (manifest, config["data"]["manifest_sha256"]),
        (head, config["router"]["head_sha256"]),
        (qwen_config, config["qwen"]["training_config_sha256"]),
        (adapter, config["qwen"]["adapter_sha256"]),
    ):
        if sha256(artifact) != expected:
            raise RuntimeError(f"frozen artifact changed: {artifact}")
    with manifest.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected = {name: int(count) for name, count in config["data"]["counts"].items()}
    counts = Counter(row["label"] for row in rows)
    if len(rows) != int(config["data"]["records"]) or dict(counts) != expected:
        raise RuntimeError("full development inventory changed")
    if any(row["head_split"] != "development" for row in rows):
        raise RuntimeError("non-development record entered the frozen view")
    if float(config["router"]["threshold"]) != 0.925:
        raise RuntimeError("the empirically dominant frozen threshold changed")
    return config, config_sha, rows


def evaluate(config, config_sha, rows):
    result_path = workspace_path(config["outputs"]["result"])
    predictions_path = workspace_path(config["outputs"]["predictions"])
    if result_path.exists() or predictions_path.exists():
        raise FileExistsError("refusing to replace full development artifacts")
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
    with safe_open(workspace_path(config["router"]["head"]), framework="pt", device="cpu") as handle:
        coefficient = handle.get_tensor("coefficient").numpy()
        intercept = handle.get_tensor("intercept").numpy()
        mean = handle.get_tensor("feature_mean").numpy()
        scale = handle.get_tensor("feature_scale").numpy()
    logits = ((features - mean) / scale) @ coefficient.T + intercept
    values = probabilities(logits, float(config["router"]["temperature"]))
    labels = [CLASSES.index(row["label"]) for row in rows]
    threshold = float(config["router"]["threshold"])
    metrics = routing_metrics(values.tolist(), labels, threshold=threshold)
    prediction_rows = []
    confusion = {truth: {predicted: 0 for predicted in (*CLASSES, "abstain")} for truth in CLASSES}
    for row, truth_index, probs in zip(rows, labels, values, strict=True):
        predicted_index = int(probs.argmax())
        confidence = float(probs[predicted_index])
        accepted = confidence >= threshold
        route = CLASSES[predicted_index] if accepted else "abstain"
        confusion[CLASSES[truth_index]][route] += 1
        prediction_rows.append(
            {
                "record_id": row["record_id"],
                "source_group": row["source_group"],
                "true_label": CLASSES[truth_index],
                "route": route,
                "confidence": f"{confidence:.10f}",
                **{f"p_{name}": f"{float(probs[index]):.10f}" for index, name in enumerate(CLASSES)},
            }
        )
    gates = config["gates"]
    overall, per_class = metrics["overall"], metrics["per_class"]
    checks = {
        "overall_wrong_rate": overall["wrong_rate"] <= gates["maximum_wrong_rate"],
        "per_class_wrong_rate": all(value["wrong_rate"] <= gates["maximum_per_class_wrong_rate"] for value in per_class.values()),
        "overall_coverage": overall["coverage"] >= gates["minimum_coverage"],
        "per_class_coverage": all(value["coverage"] >= gates["minimum_per_class_coverage"] for value in per_class.values()),
    }
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    with predictions_path.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(prediction_rows[0]))
        writer.writeheader()
        writer.writerows(prediction_rows)
    payload = {
        "schema_version": 1,
        "status": "full_five_class_development_complete",
        "gate_pass": all(checks.values()),
        "checks": checks,
        "metrics": metrics,
        "confusion": confusion,
        "gates": gates,
        "config_sha256": config_sha,
        "manifest_sha256": config["data"]["manifest_sha256"],
        "head_sha256": config["router"]["head_sha256"],
        "adapter_sha256": config["qwen"]["adapter_sha256"],
        "predictions": predictions_path.relative_to(ROOT).as_posix(),
        "predictions_sha256": sha256(predictions_path),
        "internal_test_used": False,
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print("FOUNDIR_QWEN_FIVE_CLASS_DEVELOPMENT " + json.dumps(payload, sort_keys=True), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    config, config_sha, rows = load_contract(args.config.resolve())
    if args.check_only:
        print(f"FOUNDIR_QWEN_FIVE_CLASS_DEVELOPMENT_PREFLIGHT_PASS records={len(rows)} config_sha256={config_sha}")
        return
    if any(name not in os.environ for name in ("RANK", "WORLD_SIZE", "LOCAL_RANK")):
        raise RuntimeError("launch with torchrun")
    rank, world, local_rank = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), int(os.environ["LOCAL_RANK"])
    if world != 8 or not torch.cuda.is_available() or torch.cuda.device_count() != 8:
        raise RuntimeError("full development gate requires exactly 8 visible GPUs")
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
