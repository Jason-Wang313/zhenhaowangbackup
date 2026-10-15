#!/usr/bin/env python3
"""Apply one bounded low-light-conditional noise correction to router v1."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lowlight_restoration.qwen_compositional_router import COMPONENTS, compositional_metrics, compositional_route
from smoke.run_foundir_qwen_compositional_router_v1 import load_contract, load_feature, predict_probabilities
from smoke.run_foundir_qwen_five_class_gate_ddp import candidate_key, sha256, workspace_path


def load_base(config):
    path = workspace_path(config["base"]["head"])
    if sha256(path) != config["base"]["head_sha256"]:
        raise RuntimeError("base compositional head changed")
    with safe_open(path, framework="pt", device="cpu") as handle:
        return {name: handle.get_tensor(name).numpy() for name in handle.keys()}


def corrected_probabilities(base_probabilities, conditional_probabilities, *, base_margin):
    corrected = base_probabilities.copy()
    lowlight_index, noise_index = COMPONENTS.index("low_light"), COMPONENTS.index("noise")
    eligible = corrected[:, lowlight_index] >= base_margin
    corrected[eligible, noise_index] = conditional_probabilities[eligible]
    return corrected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    config, config_sha, train_rows, singleton_rows, mixed_rows = load_contract(args.config.resolve())
    base = load_base(config)
    if args.check_only:
        print(f"FOUNDIR_QWEN_COMPOSITIONAL_V2_PREFLIGHT_PASS config_sha256={config_sha}")
        return
    from sklearn.linear_model import LogisticRegression

    outputs = [workspace_path(config["outputs"][key]) for key in ("head", "predictions", "result")]
    if any(path.exists() for path in outputs):
        raise FileExistsError("refusing to replace compositional v2 artifacts")
    head_path, predictions_path, result_path = outputs
    started = time.monotonic()
    train_dir = workspace_path(config["features"]["fit_calibration_dir"])
    train_features = np.stack([load_feature(train_dir, row, config) for row in train_rows])
    labels = np.asarray([row["label"] for row in train_rows])
    splits = np.asarray([row["head_split"] for row in train_rows])
    standardized = (train_features - base["feature_mean"]) / base["feature_scale"]
    eligible_classes = set(config["conditional_head"]["eligible_classes"])
    positive_classes = set(config["conditional_head"]["positive_classes"])
    eligible = np.asarray([label in eligible_classes for label in labels])
    fit = eligible & (splits == "fit")
    calibration = splits == "calibration"
    targets = np.asarray([int(label in positive_classes) for label in labels])
    classifier = LogisticRegression(
        C=float(config["conditional_head"]["c"]), class_weight="balanced", solver="lbfgs",
        max_iter=int(config["conditional_head"]["maximum_iterations"]),
        random_state=int(config["conditional_head"]["seed"]),
    )
    classifier.fit(standardized[fit], targets[fit])
    conditional_coefficient = classifier.coef_[0].astype(np.float32)
    conditional_intercept = np.float32(classifier.intercept_[0])
    base_calibration = predict_probabilities(
        standardized[calibration], base["coefficient"], base["intercept"]
    )
    conditional_calibration = predict_probabilities(
        standardized[calibration], conditional_coefficient[None, :], np.asarray([conditional_intercept])
    )[:, 0]
    corrected_calibration = corrected_probabilities(
        base_calibration, conditional_calibration, base_margin=float(config["base"]["margin"])
    )
    calibration_labels = labels[calibration].tolist()
    selected = None
    noise_index = COMPONENTS.index("noise")
    for margin in config["conditional_head"]["candidate_margins"]:
        candidate = corrected_calibration.copy()
        # Re-encode conditional uncertainty so the shared router margin implements this margin.
        lowlight_active = base_calibration[:, COMPONENTS.index("low_light")] >= float(config["base"]["margin"])
        q = conditional_calibration
        candidate[lowlight_active & (q >= float(margin)), noise_index] = 1.0
        candidate[lowlight_active & (q <= 1.0 - float(margin)), noise_index] = 0.0
        uncertain = lowlight_active & (q > 1.0 - float(margin)) & (q < float(margin))
        candidate[uncertain, noise_index] = 0.5
        metrics = compositional_metrics(candidate.tolist(), calibration_labels, margin=float(config["base"]["margin"]))
        item = {"conditional_margin": float(margin), "metrics": metrics, "key": candidate_key(metrics, config["gates"])}
        if selected is None or item["key"] > selected["key"]:
            selected = item
    assert selected is not None
    print("FOUNDIR_QWEN_COMPOSITIONAL_V2_CALIBRATION " + json.dumps({"margin": selected["conditional_margin"], "overall": selected["metrics"]["overall"]}, sort_keys=True), flush=True)

    development_rows = singleton_rows + mixed_rows
    singleton_dir = workspace_path(config["features"]["singleton_development_dir"])
    mixed_dir = workspace_path(config["features"]["mixed_development_dir"])
    development_features = np.stack(
        [load_feature(singleton_dir, row, config) for row in singleton_rows]
        + [load_feature(mixed_dir, row, config) for row in mixed_rows]
    )
    x = (development_features - base["feature_mean"]) / base["feature_scale"]
    base_probabilities = predict_probabilities(x, base["coefficient"], base["intercept"])
    conditional = predict_probabilities(x, conditional_coefficient[None, :], np.asarray([conditional_intercept]))[:, 0]
    probabilities = base_probabilities.copy()
    lowlight_active = probabilities[:, COMPONENTS.index("low_light")] >= float(config["base"]["margin"])
    margin = selected["conditional_margin"]
    probabilities[lowlight_active & (conditional >= margin), noise_index] = 1.0
    probabilities[lowlight_active & (conditional <= 1.0 - margin), noise_index] = 0.0
    uncertain = lowlight_active & (conditional > 1.0 - margin) & (conditional < margin)
    probabilities[uncertain, noise_index] = 0.5
    labels_development = [row["label"] for row in development_rows]
    routes = [compositional_route(row, margin=float(config["base"]["margin"])) for row in probabilities]
    development = compositional_metrics(probabilities.tolist(), labels_development, margin=float(config["base"]["margin"]))
    singleton = compositional_metrics(probabilities[:len(singleton_rows)].tolist(), labels_development[:len(singleton_rows)], margin=float(config["base"]["margin"]))
    mixed = compositional_metrics(probabilities[len(singleton_rows):].tolist(), labels_development[len(singleton_rows):], margin=float(config["base"]["margin"]))
    gate_pass = candidate_key(development, config["gates"])[0] == 1.0
    head_path.parent.mkdir(parents=True, exist_ok=True)
    save_file({
        **{name: torch.from_numpy(value) for name, value in base.items()},
        "conditional_noise_coefficient": torch.from_numpy(conditional_coefficient),
        "conditional_noise_intercept": torch.from_numpy(np.asarray([conditional_intercept])),
    }, head_path, metadata={"config_sha256": config_sha, "conditional_margin": str(margin), "base_head_sha256": config["base"]["head_sha256"]})
    with predictions_path.open("x", newline="") as handle:
        fields = ("record_id", "truth", "route", "accepted", "conditional_noise_probability")
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for row, truth, route, q in zip(development_rows, labels_development, routes, conditional, strict=True):
            writer.writerow({"record_id": row["record_id"], "truth": truth, "route": route or "abstain", "accepted": route is not None, "conditional_noise_probability": f"{q:.9f}"})
    payload = {
        "schema_version": 1, "status": "conditional_compositional_development_diagnostic_complete",
        "gate_pass": gate_pass, "selected": selected, "development": development,
        "singleton": singleton, "mixed": mixed, "gates": config["gates"],
        "head": head_path.relative_to(ROOT).as_posix(), "head_sha256": sha256(head_path),
        "predictions": predictions_path.relative_to(ROOT).as_posix(), "predictions_sha256": sha256(predictions_path),
        "config_sha256": config_sha, "elapsed_seconds": time.monotonic()-started, "gpu_hours": 0,
        "internal_or_official_test_used": False, "development_reused_after_prior_router_diagnosis": True,
    }
    result_path.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")
    print("FOUNDIR_QWEN_COMPOSITIONAL_V2_RESULT "+json.dumps(payload, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
