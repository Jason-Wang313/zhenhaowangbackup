#!/usr/bin/env python3
"""Fit and evaluate five independent component heads on frozen Qwen features."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lowlight_restoration.qwen_compositional_router import (
    COMPONENTS,
    component_targets,
    compositional_metrics,
    compositional_route,
)
from lowlight_restoration.qwen_eleven_class_gate import CLASSES, MIXED_CLASSES, SINGLETON_CLASSES
from smoke.run_foundir_qwen_five_class_gate_ddp import candidate_key, sha256, workspace_path


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def load_contract(path: Path):
    config = json.loads(path.read_text())
    config_sha = sha256(path)
    if config.get("schema_version") != 1:
        raise ValueError("bad compositional router schema")
    data = config["data"]
    paths = {
        "fit_calibration": workspace_path(data["fit_calibration_manifest"]),
        "singleton": workspace_path(data["singleton_development_manifest"]),
        "mixed": workspace_path(data["mixed_development_manifest"]),
    }
    expected_hashes = {
        "fit_calibration": data["fit_calibration_manifest_sha256"],
        "singleton": data["singleton_development_manifest_sha256"],
        "mixed": data["mixed_development_manifest_sha256"],
    }
    for name, artifact in paths.items():
        if sha256(artifact) != expected_hashes[name]:
            raise RuntimeError(f"frozen manifest changed: {artifact}")
    fit_calibration = [
        row for row in read_rows(paths["fit_calibration"]) if row["head_split"] in {"fit", "calibration"}
    ]
    singleton = read_rows(paths["singleton"])
    mixed = read_rows(paths["mixed"])
    counts = Counter((row["head_split"], row["label"]) for row in fit_calibration)
    for label in CLASSES:
        if counts[("fit", label)] != int(data["fit_records_per_class"]):
            raise RuntimeError(f"fit count changed: {label}")
        if counts[("calibration", label)] != int(data["calibration_records_per_class"]):
            raise RuntimeError(f"calibration count changed: {label}")
    if set(row["label"] for row in singleton) != set(SINGLETON_CLASSES):
        raise RuntimeError("singleton labels changed")
    if set(row["label"] for row in mixed) != set(MIXED_CLASSES):
        raise RuntimeError("mixed labels changed")
    if data.get("internal_or_official_test_allowed") is not False:
        raise RuntimeError("internal/official test must remain forbidden")
    return config, config_sha, fit_calibration, singleton, mixed


def load_feature(directory: Path, row: dict[str, str], config: dict) -> np.ndarray:
    path = directory / f"{row['record_id']}.safetensors"
    if not path.exists():
        raise FileNotFoundError(path)
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
        raise RuntimeError(f"feature identity changed: {path}")
    if tensor.shape != (int(config["features"]["feature_dim"]),) or tensor.dtype != torch.float32:
        raise RuntimeError(f"feature shape changed: {path}")
    return tensor.numpy()


def predict_probabilities(x: np.ndarray, coefficients: np.ndarray, intercepts: np.ndarray) -> np.ndarray:
    logits = x @ coefficients.T + intercepts
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    config, config_sha, train_rows, singleton_rows, mixed_rows = load_contract(args.config.resolve())
    if args.check_only:
        print(
            f"FOUNDIR_QWEN_COMPOSITIONAL_PREFLIGHT_PASS train_cal={len(train_rows)} "
            f"development={len(singleton_rows)+len(mixed_rows)} config_sha256={config_sha}"
        )
        return

    from sklearn.linear_model import LogisticRegression

    result_path = workspace_path(config["outputs"]["result"])
    head_path = workspace_path(config["outputs"]["head"])
    predictions_path = workspace_path(config["outputs"]["predictions"])
    if any(path.exists() for path in (result_path, head_path, predictions_path)):
        raise FileExistsError("refusing to replace compositional router artifacts")
    started = time.monotonic()
    train_dir = workspace_path(config["features"]["fit_calibration_dir"])
    train_features = np.stack([load_feature(train_dir, row, config) for row in train_rows])
    labels = [row["label"] for row in train_rows]
    targets = np.asarray(component_targets(labels), dtype=np.int64)
    splits = np.asarray([row["head_split"] for row in train_rows])
    fit_mask, calibration_mask = splits == "fit", splits == "calibration"
    mean = train_features[fit_mask].mean(0)
    scale = train_features[fit_mask].std(0).clip(1e-6)
    standardized = (train_features - mean) / scale
    coefficients, intercepts = [], []
    for index, component in enumerate(COMPONENTS):
        classifier = LogisticRegression(
            C=float(config["head"]["c"]),
            class_weight="balanced",
            solver="lbfgs",
            max_iter=int(config["head"]["maximum_iterations"]),
            random_state=int(config["head"]["seed"]),
        )
        classifier.fit(standardized[fit_mask], targets[fit_mask, index])
        coefficients.append(classifier.coef_[0].astype(np.float32))
        intercepts.append(np.float32(classifier.intercept_[0]))
        print(f"FOUNDIR_QWEN_COMPONENT_FIT component={component}", flush=True)
    coefficient = np.stack(coefficients)
    intercept = np.asarray(intercepts)
    calibration_probabilities = predict_probabilities(
        standardized[calibration_mask], coefficient, intercept
    )
    calibration_labels = [label for label, keep in zip(labels, calibration_mask, strict=True) if keep]
    selected = None
    for margin in config["head"]["candidate_margins"]:
        metrics = compositional_metrics(
            calibration_probabilities.tolist(), calibration_labels, margin=float(margin)
        )
        item = {"margin": float(margin), "metrics": metrics, "key": candidate_key(metrics, config["gates"])}
        if selected is None or item["key"] > selected["key"]:
            selected = item
    assert selected is not None
    print(
        "FOUNDIR_QWEN_COMPONENT_CALIBRATION "
        + json.dumps({"margin": selected["margin"], "overall": selected["metrics"]["overall"]}, sort_keys=True),
        flush=True,
    )

    development_rows = singleton_rows + mixed_rows
    singleton_dir = workspace_path(config["features"]["singleton_development_dir"])
    mixed_dir = workspace_path(config["features"]["mixed_development_dir"])
    development_features = np.stack(
        [load_feature(singleton_dir, row, config) for row in singleton_rows]
        + [load_feature(mixed_dir, row, config) for row in mixed_rows]
    )
    development_probabilities = predict_probabilities(
        (development_features - mean) / scale, coefficient, intercept
    )
    development_labels = [row["label"] for row in development_rows]
    routes = [
        compositional_route(row, margin=selected["margin"]) for row in development_probabilities
    ]
    development = compositional_metrics(
        development_probabilities.tolist(), development_labels, margin=selected["margin"]
    )
    singleton = compositional_metrics(
        development_probabilities[: len(singleton_rows)].tolist(),
        development_labels[: len(singleton_rows)],
        margin=selected["margin"],
    )
    mixed = compositional_metrics(
        development_probabilities[len(singleton_rows) :].tolist(),
        development_labels[len(singleton_rows) :],
        margin=selected["margin"],
    )
    gate_pass = candidate_key(development, config["gates"])[0] == 1.0
    head_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {
            "coefficient": torch.from_numpy(coefficient),
            "intercept": torch.from_numpy(intercept),
            "feature_mean": torch.from_numpy(mean.astype(np.float32)),
            "feature_scale": torch.from_numpy(scale.astype(np.float32)),
        },
        head_path,
        metadata={
            "config_sha256": config_sha,
            "adapter_sha256": str(config["qwen"]["adapter_sha256"]),
            "components": ",".join(COMPONENTS),
            "margin": str(selected["margin"]),
        },
    )
    with predictions_path.open("x", newline="") as handle:
        fields = ["record_id", "truth", "route", "accepted", *[f"p_{name}" for name in COMPONENTS]]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row, truth, route, values in zip(
            development_rows, development_labels, routes, development_probabilities, strict=True
        ):
            item = {
                "record_id": row["record_id"], "truth": truth, "route": route or "abstain",
                "accepted": route is not None,
            }
            item.update({f"p_{name}": f"{value:.9f}" for name, value in zip(COMPONENTS, values, strict=True)})
            writer.writerow(item)
    payload = {
        "schema_version": 1,
        "status": "compositional_router_development_diagnostic_complete",
        "gate_pass": gate_pass,
        "selected": selected,
        "development": development,
        "singleton": singleton,
        "mixed": mixed,
        "gates": config["gates"],
        "head": head_path.relative_to(ROOT).as_posix(),
        "head_sha256": sha256(head_path),
        "predictions": predictions_path.relative_to(ROOT).as_posix(),
        "predictions_sha256": sha256(predictions_path),
        "config_sha256": config_sha,
        "elapsed_seconds": time.monotonic() - started,
        "gpu_hours": 0,
        "internal_or_official_test_used": False,
        "development_reused_after_prior_router_diagnosis": True,
    }
    result_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print("FOUNDIR_QWEN_COMPOSITIONAL_RESULT " + json.dumps(payload, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
