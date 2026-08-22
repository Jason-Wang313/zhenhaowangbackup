#!/usr/bin/env python3
"""Evaluate the frozen four-way Gate 3 causal Qwen matrix."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import io
import json
import os
import sys
from pathlib import Path
from typing import Mapping

import numpy as np
from skimage.metrics import structural_similarity
import torch
import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from smoke.evaluate_foundir_lowlight_gate3_compact import aggregate, literal_psnr, load_rgb_path
from smoke.export_foundir_lowlight_gate3_causal_qwen import read_contract
from smoke.run_foundir_lowlight_gate3_causal_adapter import VARIANTS, write_once
from smoke.train_foundir_lowlight_gate3_compact import sha256, workspace_path


def metric_models(device: torch.device):
    import lpips
    import pyiqa

    contract = json.loads(workspace_path("configs/foundir_lowlight_metric_contract_v2b.json").read_text())
    for package, expected in contract["environment"]["packages"].items():
        actual = importlib.metadata.version(package)
        if package in {"torch", "torchvision"}:
            if not actual.startswith(expected.split("+")[0]):
                raise RuntimeError(f"metric package changed: {package} {actual}")
        elif actual != expected:
            raise RuntimeError(f"metric package changed: {package} {actual}")
    return (
        pyiqa.create_metric("psnr", device=device),
        pyiqa.create_metric("lpips", device=device),
        lpips.LPIPS(net="alex", version="0.1").to(device).eval(),
    )


def tensor(image: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(image.transpose(2, 0, 1).copy()).unsqueeze(0).to(device=device, dtype=torch.float32).div(255.0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config, config_sha, development, _ = read_contract(args.config.resolve())
    outer = {row["record_id"]: row for row in development if row["role"] == config["data"]["outer_role"]}
    manifest_path = workspace_path(config["outputs"]["prediction_manifest"])
    with manifest_path.open(newline="") as handle:
        predictions = list(csv.DictReader(handle))
    if len(predictions) != len(outer) * len(VARIANTS):
        raise RuntimeError("causal prediction manifest coverage changed")
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    if world != int(config["runtime"]["world_size"]):
        raise RuntimeError("causal evaluator world size changed")
    local_rank = int(os.environ["LOCAL_RANK"]); torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    pyiqa_psnr, pyiqa_lpips, direct_lpips = metric_models(device)
    grouped: dict[str, list[dict[str, str]]] = {}
    for prediction in predictions:
        grouped.setdefault(prediction["record_id"], []).append(prediction)
    local_rows = []
    record_ids = sorted(outer)
    with torch.inference_mode():
        for record_index in range(rank, len(record_ids), world):
            record_id = record_ids[record_index]
            row = outer[record_id]
            target = load_rgb_path(workspace_path(row["target_path"]), row["stored_color_order"])
            target_tensor = tensor(target, device)
            for prediction_row in grouped[record_id]:
                path = workspace_path(prediction_row["output_path"])
                if sha256(path) != prediction_row["output_sha256"]:
                    raise RuntimeError(f"causal prediction changed: {record_id}")
                prediction = load_rgb_path(path)
                if prediction.shape != target.shape:
                    raise RuntimeError(f"causal output dimensions changed: {record_id}")
                prediction_tensor = tensor(prediction, device)
                literal = literal_psnr(target, prediction)
                evaluator = float(pyiqa_psnr(prediction_tensor, target_tensor).item())
                pyiqa_value = float(pyiqa_lpips(prediction_tensor, target_tensor).item())
                direct = float(direct_lpips(prediction_tensor * 2 - 1, target_tensor * 2 - 1).item())
                local_rows.append({
                    "candidate_id": prediction_row["variant"], "family": row["family"], "group_id": row["group_id"],
                    "record_id": record_id, "psnr_literal": literal, "psnr_pyiqa": evaluator,
                    "psnr_conservative": min(literal, evaluator), "lpips_pyiqa": pyiqa_value,
                    "lpips_direct": direct, "lpips_conservative": max(pyiqa_value, direct),
                    "ssim": float(structural_similarity(target, prediction, channel_axis=2, data_range=255)),
                    "target_access": prediction_row["variant"] == "oracle_plus_measured",
                })
            print(f"GATE3_CAUSAL_METRICS rank={rank} records={record_index + 1}/{len(record_ids)}", flush=True)
    gathered: list[object] = [None for _ in range(world)]
    dist.all_gather_object(gathered, local_rows)
    if rank == 0:
        rows = sorted((item for part in gathered for item in part), key=lambda row: (row["candidate_id"], row["record_id"]))
        if len(rows) != len(predictions):
            raise RuntimeError("causal metric coverage changed")
        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
        metric_content = buffer.getvalue().encode()
        metric_path = workspace_path(config["outputs"]["metric_rows"])
        write_once(metric_path, metric_content)
        summary = aggregate(rows)
        measured, oracle, qwen, shuffled = (summary[name] for name in VARIANTS)
        thresholds = config["promotion"]
        checks: dict[str, dict[str, float | bool]] = {}
        for family in config["data"]["outer_families"]:
            m, o, q, s = (candidate["families"][family] for candidate in (measured, oracle, qwen, shuffled))
            oracle_gain = o["psnr_conservative"]["estimate"] - m["psnr_conservative"]["estimate"]
            qwen_gain = q["psnr_conservative"]["estimate"] - m["psnr_conservative"]["estimate"]
            shuffle_gain = q["psnr_conservative"]["estimate"] - s["psnr_conservative"]["estimate"]
            recovery = qwen_gain / oracle_gain if oracle_gain > 0 else float("-inf")
            lpips_over_measured = q["lpips_conservative"]["estimate"] - m["lpips_conservative"]["estimate"]
            lpips_over_shuffled = q["lpips_conservative"]["estimate"] - s["lpips_conservative"]["estimate"]
            passed = (
                oracle_gain >= float(thresholds["minimum_oracle_psnr_gain_each_family"])
                and qwen_gain >= float(thresholds["minimum_qwen_psnr_gain_each_family"])
                and shuffle_gain >= float(thresholds["minimum_qwen_psnr_gain_each_family"])
                and recovery >= float(thresholds["minimum_oracle_increment_recovered_fraction"])
                and lpips_over_measured <= float(thresholds["lpips_max_regression_each_family"])
                and lpips_over_shuffled <= float(thresholds["lpips_max_regression_each_family"])
            )
            checks[family] = {
                "oracle_psnr_gain_over_measured": oracle_gain,
                "qwen_psnr_gain_over_measured": qwen_gain,
                "qwen_psnr_gain_over_shuffled": shuffle_gain,
                "oracle_increment_recovered_fraction": recovery,
                "qwen_lpips_change_over_measured": lpips_over_measured,
                "qwen_lpips_change_over_shuffled": lpips_over_shuffled,
                "passed": passed,
            }
        passed = all(bool(value["passed"]) for value in checks.values())
        payload = {
            "schema_version": 1, "status": "passed" if passed else "failed", "automatic_gate_pass": passed,
            "protected_foundir_pixels_used": False, "config_sha256": config_sha,
            "prediction_manifest_sha256": sha256(manifest_path), "metric_rows": str(metric_path.relative_to(ROOT)),
            "metric_rows_sha256": hashlib.sha256(metric_content).hexdigest(), "aggregate": summary,
            "family_checks": checks, "failure_action": None if passed else config["promotion"]["failure_action"],
        }
        result_path = workspace_path(config["outputs"]["result"])
        write_once(result_path, (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode())
        print(f"FOUNDIR_GATE3_CAUSAL_RESULT status={payload['status']} pass={passed}")
    dist.barrier(); dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
