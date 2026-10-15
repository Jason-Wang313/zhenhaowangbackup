#!/usr/bin/env python3
"""One-shot evaluation of the physical guard on the last untouched source groups."""
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lowlight_restoration.noise_lowlight_physical_guard import physical_features, physical_route
from lowlight_restoration.qwen_compositional_router import compositional_route
from lowlight_restoration.qwen_eleven_class_gate import consensus_metrics
from smoke.run_foundir_qwen_compositional_router_v1 import predict_probabilities
from smoke.run_foundir_qwen_five_class_gate_ddp import (
    extract,
    feature_metadata,
    feature_path,
    load_feature,
    sha256,
    workspace_path,
)
from smoke.run_foundir_qwen_lowlight_noise_dual_nonlinear_v2 import apply_dual, model_from, tensors


def contract(path: Path):
    config = json.loads(path.read_text())
    config_sha = sha256(path)
    artifacts = (
        (workspace_path(config["data"]["manifest"]), config["data"]["manifest_sha256"]),
        (workspace_path(config["qwen"]["training_config"]), config["qwen"]["training_config_sha256"]),
        (workspace_path(config["qwen"]["adapter"]), config["qwen"]["adapter_sha256"]),
        (workspace_path(config["router"]["base_head"]), config["router"]["base_head_sha256"]),
        (workspace_path(config["router"]["lowlight_head"]), config["router"]["lowlight_head_sha256"]),
        (workspace_path(config["router"]["noise_head"]), config["router"]["noise_head_sha256"]),
        (workspace_path(config["physical_guard"]["rule"]), config["physical_guard"]["rule_sha256"]),
    )
    for artifact, expected in artifacts:
        if sha256(artifact) != expected:
            raise RuntimeError(f"artifact changed: {artifact}")
    with artifacts[0][0].open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 12 or any(sum(row["label"] == label for row in rows) != 6 for label in ("noise", "noise+low_light")):
        raise RuntimeError("physical holdout cardinality changed")
    if len({row["source_group"] for row in rows}) != 6:
        raise RuntimeError("expected six paired source groups")
    if config["data"].get("development_internal_or_official_test_allowed") is not False:
        raise RuntimeError("protected test access forbidden")
    rule = json.loads(artifacts[-1][0].read_text())
    if rule["status"] != "physical_rule_frozen_before_v3_holdout_access":
        raise RuntimeError("physical rule was not prospectively frozen")
    return config, config_sha, rows, rule["rule"]


def evaluate(config: dict, config_sha: str, rows: list[dict[str, str]], rule: dict) -> None:
    result_path = workspace_path(config["outputs"]["result"])
    predictions_path = workspace_path(config["outputs"]["predictions"])
    if result_path.exists() or predictions_path.exists():
        raise FileExistsError("refusing to overwrite terminal holdout result")
    features = [
        load_feature(
            feature_path(config, row["record_id"]),
            feature_metadata(config, config_sha, row),
            int(config["features"]["feature_dim"]),
        )
        for row in rows
    ]
    if any(value is None for value in features):
        raise RuntimeError("Qwen features incomplete")
    matrix = np.stack([value.numpy() for value in features])
    base_tensors = {
        name: value.numpy() for name, value in tensors(workspace_path(config["router"]["base_head"])).items()
    }
    standardized = ((matrix - base_tensors["feature_mean"]) / base_tensors["feature_scale"]).astype(np.float32)
    base_probabilities = predict_probabilities(standardized, base_tensors["coefficient"], base_tensors["intercept"])
    linear = predict_probabilities(
        standardized,
        base_tensors["conditional_noise_coefficient"][None, :],
        base_tensors["conditional_noise_intercept"],
    )[:, 0]
    lowlight_model = model_from(workspace_path(config["router"]["lowlight_head"]), standardized.shape[1])
    noise_model = model_from(workspace_path(config["router"]["noise_head"]), standardized.shape[1])
    with torch.inference_mode():
        lowlight_q = torch.sigmoid(lowlight_model(torch.from_numpy(standardized))).numpy()
        noise_q = torch.sigmoid(noise_model(torch.from_numpy(standardized))).numpy()
    shim = {
        "base": {
            "route_margin": config["router"]["route_margin"],
            "conditional_positive_margin": config["router"]["conditional_positive_margin"],
            "conditional_negative_margin": config["router"]["conditional_negative_margin"],
        },
        "lowlight_side": {"margin": config["router"]["lowlight_margin"]},
    }
    values, _, _ = apply_dual(
        base_probabilities,
        linear,
        lowlight_q,
        noise_q,
        shim,
        config["router"]["noise_margin"],
    )
    qwen_routes = [compositional_route(value, margin=config["router"]["route_margin"]) for value in values]
    physical = [physical_features(workspace_path(row["lq_path"]), int(rule["maximum_side"])) for row in rows]
    routes: list[str | None] = []
    physical_routes: list[str | None] = []
    for qwen_route, item in zip(qwen_routes, physical, strict=True):
        physical_decision = None
        route = qwen_route
        if qwen_route == "noise":
            physical_decision = physical_route(
                item,
                float(rule["low_exposure"]),
                float(rule["high_exposure"]),
                float(rule["dark_noise_floor"]),
            )
            route = physical_decision
        physical_routes.append(physical_decision)
        routes.append(route)

    truth = [row["label"] for row in rows]
    metrics = consensus_metrics(routes, truth)
    present = {key: value for key, value in metrics["per_class"].items() if value["records"]}
    gates = config["gates"]
    overall = metrics["overall"]
    passed = (
        overall["wrong_rate"] <= gates["maximum_wrong_rate"]
        and overall["coverage"] >= gates["minimum_coverage"]
        and all(
            value["wrong_rate"] <= gates["maximum_per_class_wrong_rate"]
            and value["coverage"] >= gates["minimum_per_class_coverage"]
            for value in present.values()
        )
    )
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    with predictions_path.open("x", newline="") as handle:
        fields = ("record_id", "source_group", "truth", "qwen_route", "physical_route", "final_route", "exposure", "dark_noise")
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row, label, qwen_route, physical_decision, final_route, item in zip(rows, truth, qwen_routes, physical_routes, routes, physical, strict=True):
            writer.writerow({
                "record_id": row["record_id"], "source_group": row["source_group"], "truth": label,
                "qwen_route": qwen_route or "abstain", "physical_route": physical_decision or "not_used_or_abstain",
                "final_route": final_route or "abstain", "exposure": item["exposure"], "dark_noise": item["dark_noise"],
            })
    payload = {
        "schema_version": 1,
        "status": "terminal_physical_holdout_complete",
        "gate_pass": passed,
        "metrics": {"overall": overall, "per_class": present},
        "records": len(rows),
        "paired_source_groups": 6,
        "effective_sample_size_warning": "Only six untouched paired source groups remained.",
        "rule_sha256": config["physical_guard"]["rule_sha256"],
        "predictions": predictions_path.relative_to(ROOT).as_posix(),
        "predictions_sha256": sha256(predictions_path),
        "config_sha256": config_sha,
        "development_internal_or_official_test_used": False,
    }
    result_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print("FOUNDIR_PHYSICAL_HOLDOUT_RESULT " + json.dumps(payload, sort_keys=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    config, config_sha, rows, rule = contract(args.config.resolve())
    if args.check_only:
        print(f"FOUNDIR_PHYSICAL_HOLDOUT_PREFLIGHT_PASS records={len(rows)} config_sha256={config_sha}")
        return
    rank, world, local = map(int, (os.environ["RANK"], os.environ["WORLD_SIZE"], os.environ["LOCAL_RANK"]))
    if world != 8 or not torch.cuda.is_available() or torch.cuda.device_count() != 8:
        raise RuntimeError("requires exactly 8 visible GPUs")
    torch.cuda.set_device(local)
    device = torch.device(f"cuda:{local}")
    dist.init_process_group("nccl", device_id=device)
    started = time.monotonic()
    extract(config, config_sha, rows, rank, world, device)
    dist.barrier()
    if rank == 0:
        evaluate(config, config_sha, rows, rule)
        print(f"FOUNDIR_PHYSICAL_HOLDOUT_COMPLETE elapsed_seconds={time.monotonic()-started:.2f}", flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
