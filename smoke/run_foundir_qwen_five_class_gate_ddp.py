#!/usr/bin/env python3
"""Extract frozen Qwen features and fit a fail-closed five-class linear head."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from safetensors import safe_open
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lowlight_restoration.qwen_five_class_gate import (
    CLASSES,
    image_embedding_sequence,
    routing_metrics,
)
from lowlight_restoration.qwen_lora import load_qwen_adapter
from smoke.train_qwen_multidegradation_lora_ddp import load_model


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def workspace_path(value: str) -> Path:
    path = (ROOT / value).resolve()
    if ROOT not in path.parents:
        raise ValueError(f"path escapes workspace: {value}")
    return path


def load_contract(path: Path) -> tuple[dict[str, object], str, list[dict[str, str]]]:
    config = json.loads(path.read_text())
    config_sha = sha256(path)
    if config.get("schema_version") != 1:
        raise ValueError("bad five-class gate config schema")
    manifest = workspace_path(str(config["data"]["manifest"]))
    if sha256(manifest) != config["data"]["manifest_sha256"]:
        raise RuntimeError("frozen training-only manifest hash changed")
    with manifest.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected = sum(int(config["data"]["records_per_class"][name]) for name in ("fit", "calibration", "holdout")) * len(CLASSES)
    if len(rows) != expected:
        raise RuntimeError("training-only manifest cardinality changed")
    if set(row["head_split"] for row in rows) != {"fit", "calibration", "holdout"}:
        raise RuntimeError("head split identities changed")
    if set(row["label"] for row in rows) != set(CLASSES):
        raise RuntimeError("five-class labels changed")
    source_splits: dict[str, set[str]] = {}
    for row in rows:
        source_splits.setdefault(row["source_group"], set()).add(row["head_split"])
    if any(len(splits) != 1 for splits in source_splits.values()):
        raise RuntimeError("source leakage across head splits")
    adapter = workspace_path(str(config["qwen"]["adapter"]))
    qwen_config = workspace_path(str(config["qwen"]["training_config"]))
    if sha256(adapter) != config["qwen"]["adapter_sha256"] or sha256(qwen_config) != config["qwen"]["training_config_sha256"]:
        raise RuntimeError("frozen Qwen artifacts changed")
    return config, config_sha, rows


def feature_path(config: Mapping[str, object], record_id: str) -> Path:
    return workspace_path(str(config["outputs"]["feature_dir"])) / f"{record_id}.safetensors"


def feature_metadata(config: Mapping[str, object], config_sha: str, row: Mapping[str, str]) -> dict[str, str]:
    return {
        "schema_version": "1",
        "config_sha256": config_sha,
        "manifest_sha256": str(config["data"]["manifest_sha256"]),
        "adapter_sha256": str(config["qwen"]["adapter_sha256"]),
        "record_id": row["record_id"],
        "lq_sha256": row["lq_sha256"],
        "pooling": "qwen_visual_token_mean_and_std",
        "feature_dim": str(config["features"]["feature_dim"]),
    }


def load_feature(path: Path, metadata: Mapping[str, str], feature_dim: int) -> torch.Tensor | None:
    if not path.exists():
        return None
    with safe_open(path, framework="pt", device="cpu") as handle:
        if handle.metadata() != dict(metadata) or set(handle.keys()) != {"feature"}:
            raise RuntimeError(f"feature cache identity changed: {path}")
        tensor = handle.get_tensor("feature")
    if tensor.shape != (feature_dim,) or tensor.dtype != torch.float32 or not bool(torch.isfinite(tensor).all()):
        raise RuntimeError(f"bad cached feature: {path}")
    return tensor


def save_feature(path: Path, tensor: torch.Tensor, metadata: Mapping[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        realized = load_feature(path, metadata, int(tensor.numel()))
        if realized is None or not torch.equal(realized, tensor.cpu()):
            raise RuntimeError(f"existing feature differs: {path}")
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    save_file({"feature": tensor.cpu().float().contiguous()}, temporary, metadata=dict(metadata))
    try:
        os.link(temporary, path)
    except FileExistsError:
        realized = load_feature(path, metadata, int(tensor.numel()))
        if realized is None or not torch.equal(realized, tensor.cpu()):
            raise RuntimeError(f"racing feature differs: {path}")
    finally:
        temporary.unlink(missing_ok=True)


def extract(config: Mapping[str, object], config_sha: str, rows: list[dict[str, str]], rank: int, world: int, device: torch.device) -> None:
    assigned = rows[rank::world]
    feature_dim = int(config["features"]["feature_dim"])
    missing = [row for row in assigned if load_feature(feature_path(config, row["record_id"]), feature_metadata(config, config_sha, row), feature_dim) is None]
    if not missing:
        print(f"QWEN_FIVE_CLASS_FEATURES_REUSED rank={rank} count={len(assigned)}", flush=True)
        return
    qwen_config = json.loads(workspace_path(str(config["qwen"]["training_config"])).read_text())
    model, processor, _ = load_model(qwen_config, variant="merger_trainable", device=device)
    load_qwen_adapter(model, workspace_path(str(config["qwen"]["adapter"])))
    model.requires_grad_(False).eval()
    batch_size = int(config["features"]["batch_size_per_gpu"])
    for start in range(0, len(missing), batch_size):
        batch = missing[start : start + batch_size]
        images = []
        for row in batch:
            path = workspace_path(row["lq_path"])
            if sha256(path) != row["lq_sha256"]:
                raise RuntimeError(f"input image hash changed: {row['record_id']}")
            with Image.open(path) as handle:
                images.append(handle.convert("RGB"))
        inputs = processor.image_processor(images=images, return_tensors="pt")
        with torch.inference_mode():
            vision_output = model.get_image_features(
                inputs.pixel_values.to(device), inputs.image_grid_thw.to(device)
            )
            embeddings = image_embedding_sequence(vision_output)
        if len(embeddings) != len(batch):
            raise RuntimeError("Qwen visual feature batch cardinality changed")
        for row, embedding in zip(batch, embeddings, strict=True):
            feature = torch.cat(
                (embedding.float().mean(0), embedding.float().std(0, unbiased=False)), dim=0
            ).cpu()
            if feature.shape != (feature_dim,) or not bool(torch.isfinite(feature).all()):
                raise RuntimeError("Qwen pooled feature shape changed")
            save_feature(
                feature_path(config, row["record_id"]),
                feature,
                feature_metadata(config, config_sha, row),
            )
        print(f"QWEN_FIVE_CLASS_FEATURE_PROGRESS rank={rank} complete={min(start + len(batch), len(missing))}/{len(missing)}", flush=True)


def probabilities(logits: np.ndarray, temperature: float) -> np.ndarray:
    values = logits / temperature
    values -= values.max(axis=1, keepdims=True)
    exp = np.exp(values)
    return exp / exp.sum(axis=1, keepdims=True)


def candidate_key(metrics: Mapping[str, object], gates: Mapping[str, object]) -> tuple[float, ...]:
    overall = metrics["overall"]
    per_class = metrics["per_class"]
    qualifies = (
        overall["wrong_rate"] <= float(gates["maximum_wrong_rate"])
        and overall["coverage"] >= float(gates["minimum_coverage"])
        and all(value["wrong_rate"] <= float(gates["maximum_per_class_wrong_rate"]) for value in per_class.values())
        and all(value["coverage"] >= float(gates["minimum_per_class_coverage"]) for value in per_class.values())
    )
    return (
        float(qualifies),
        -float(overall["wrong_rate"]),
        float(overall["correct_rate"]),
        float(overall["coverage"]),
        min(float(value["coverage"]) for value in per_class.values()),
    )


def fit_head(config: Mapping[str, object], config_sha: str, rows: list[dict[str, str]]) -> None:
    from sklearn.linear_model import LogisticRegression

    output = workspace_path(str(config["outputs"]["result"]))
    head_path = workspace_path(str(config["outputs"]["head"]))
    if output.exists() or head_path.exists():
        raise FileExistsError("refusing to replace five-class gate result")
    features = np.stack([
        load_feature(feature_path(config, row["record_id"]), feature_metadata(config, config_sha, row), int(config["features"]["feature_dim"])).numpy()
        for row in rows
    ])
    labels = np.asarray([CLASSES.index(row["label"]) for row in rows], dtype=np.int64)
    split = np.asarray([row["head_split"] for row in rows])
    fit_mask, calibration_mask, holdout_mask = split == "fit", split == "calibration", split == "holdout"
    mean = features[fit_mask].mean(0)
    scale = features[fit_mask].std(0).clip(1e-6)
    standardized = (features - mean) / scale
    gates = config["gates"]
    best = None
    for c_value in config["head"]["candidate_c"]:
        classifier = LogisticRegression(C=float(c_value), solver="lbfgs", max_iter=int(config["head"]["maximum_iterations"]), random_state=int(config["head"]["seed"]))
        classifier.fit(standardized[fit_mask], labels[fit_mask])
        logits = classifier.decision_function(standardized[calibration_mask])
        for temperature in config["head"]["candidate_temperatures"]:
            values = probabilities(logits, float(temperature))
            for threshold in config["head"]["candidate_thresholds"]:
                metrics = routing_metrics(values.tolist(), labels[calibration_mask].tolist(), threshold=float(threshold))
                item = {
                    "key": candidate_key(metrics, gates),
                    "c": float(c_value),
                    "temperature": float(temperature),
                    "threshold": float(threshold),
                    "metrics": metrics,
                    "classifier": classifier,
                }
                if best is None or item["key"] > best["key"]:
                    best = item
    assert best is not None
    classifier = best.pop("classifier")
    holdout_values = probabilities(classifier.decision_function(standardized[holdout_mask]), best["temperature"])
    holdout = routing_metrics(holdout_values.tolist(), labels[holdout_mask].tolist(), threshold=best["threshold"])
    gate_pass = candidate_key(holdout, gates)[0] == 1.0
    head_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {
            "coefficient": torch.from_numpy(classifier.coef_.astype(np.float32)),
            "intercept": torch.from_numpy(classifier.intercept_.astype(np.float32)),
            "feature_mean": torch.from_numpy(mean.astype(np.float32)),
            "feature_scale": torch.from_numpy(scale.astype(np.float32)),
        },
        head_path,
        metadata={
            "config_sha256": config_sha,
            "manifest_sha256": str(config["data"]["manifest_sha256"]),
            "adapter_sha256": str(config["qwen"]["adapter_sha256"]),
            "classes": ",".join(CLASSES),
            "temperature": str(best["temperature"]),
            "threshold": str(best["threshold"]),
        },
    )
    payload = {
        "schema_version": 1,
        "status": "training_only_holdout_complete",
        "gate_pass": gate_pass,
        "config_sha256": config_sha,
        "manifest_sha256": config["data"]["manifest_sha256"],
        "adapter_sha256": config["qwen"]["adapter_sha256"],
        "selected": best,
        "holdout": holdout,
        "gates": gates,
        "head": head_path.relative_to(ROOT).as_posix(),
        "head_sha256": sha256(head_path),
        "development_or_internal_test_used": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print("FOUNDIR_QWEN_FIVE_CLASS_TRAIN_ONLY_GATE " + json.dumps(payload, sort_keys=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    config, config_sha, rows = load_contract(args.config.resolve())
    if args.check_only:
        print(f"FOUNDIR_QWEN_FIVE_CLASS_PREFLIGHT_PASS records={len(rows)} config_sha256={config_sha}")
        return
    required = ("RANK", "WORLD_SIZE", "LOCAL_RANK")
    if any(name not in os.environ for name in required):
        raise RuntimeError("launch with torchrun")
    rank, world, local_rank = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), int(os.environ["LOCAL_RANK"])
    if world != 8 or not torch.cuda.is_available() or torch.cuda.device_count() != 8:
        raise RuntimeError("five-class feature pilot requires exactly 8 visible GPUs")
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dist.init_process_group("nccl", device_id=device)
    started = time.monotonic()
    extract(config, config_sha, rows, rank, world, device)
    dist.barrier()
    if rank == 0:
        fit_head(config, config_sha, rows)
        print(f"FOUNDIR_QWEN_FIVE_CLASS_COMPLETE elapsed_seconds={time.monotonic()-started:.2f}", flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
