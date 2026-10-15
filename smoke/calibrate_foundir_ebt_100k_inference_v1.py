#!/usr/bin/env python3
"""Frozen lower-order inference calibration for the 100k EBT checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import smoke.train_foundir_compositional_ebt_scaled_learning_smoke_v1 as base  # noqa: E402
import smoke.train_foundir_compositional_ebt_tensor_shard_100k_v5 as train100k  # noqa: E402


CALIBRATION_CONFIG = ROOT / "configs/foundir_ebt_100k_inference_calibration_v1.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    data = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise RuntimeError(f"refusing differing calibration artifact: {path}")
        return
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def restore(model, degraded: torch.Tensor, q: torch.Tensor, training: dict[str, Any], *, alpha: float, steps: int) -> torch.Tensor:
    candidate = degraded.clone()
    clamp = tuple(map(float, training["energy_descent"]["candidate_clamp"]))
    for _ in range(steps):
        candidate = model.descent_step(
            degraded, candidate, q, step_size=alpha, create_graph=False, clamp=clamp
        ).candidate.detach()
    return candidate


def metric_rows(prediction: torch.Tensor, target: torch.Tensor, lpips) -> tuple[list[float], list[float]]:
    pred01, target01 = prediction.add(1).mul(0.5), target.add(1).mul(0.5)
    mse = (pred01 - target01).square().flatten(1).mean(1).clamp_min(1e-12)
    psnr = (-10 * torch.log10(mse)).detach().cpu().tolist()
    with torch.inference_mode():
        values = lpips(pred01, target01).reshape(-1).detach().cpu().tolist()
    lpips.reset()
    return list(map(float, psnr)), list(map(float, values))


def summarize(psnr: list[float], lpips: list[float], sizes: list[int]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, accepted in (("overall", {1, 2}), ("singleton", {1}), ("double", {2})):
        indices = [i for i, size in enumerate(sizes) if size in accepted]
        result[key] = {
            "records": len(indices),
            "psnr": sum(psnr[i] for i in indices) / len(indices),
            "lpips": sum(lpips[i] for i in indices) / len(indices),
        }
    return result


def pareto(candidates: list[dict[str, Any]]) -> list[str]:
    selected = []
    for item in candidates:
        metric = item["metrics"]["overall"]
        dominated = any(
            other is not item
            and other["metrics"]["overall"]["psnr"] >= metric["psnr"]
            and other["metrics"]["overall"]["lpips"] <= metric["lpips"]
            and (
                other["metrics"]["overall"]["psnr"] > metric["psnr"]
                or other["metrics"]["overall"]["lpips"] < metric["lpips"]
            )
            for other in candidates
        )
        if not dominated:
            selected.append(item["candidate_id"])
    return selected


def run_arm(arm: str) -> None:
    calibration = json.loads(CALIBRATION_CONFIG.read_text())
    training_path = ROOT / calibration["training_config"]
    if sha256(training_path) != calibration["training_config_sha256"]:
        raise RuntimeError("training config identity changed")
    training = json.loads(training_path.read_text())
    base.CONFIG_PATH = training_path
    train100k.CONFIG_PATH = training_path
    _, validation = train100k.load_records(training)
    if len(validation) != int(calibration["records"]) or any(
        int(row["composition_size"]) not in calibration["allowed_composition_sizes"] for row in validation
    ):
        raise RuntimeError("calibration validation boundary changed")
    batches = []
    cursor = 0
    for degraded, target, q in train100k.loader(validation, training, train=False):
        sizes = [int(row["composition_size"]) for row in validation[cursor : cursor + degraded.shape[0]]]
        batches.append((degraded, target, q, sizes))
        cursor += degraded.shape[0]
    if cursor != len(validation):
        raise RuntimeError("calibration loader cardinality changed")

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("calibration arm requires exactly one visible GPU")
    device = torch.device("cuda:0")
    checkpoint = ROOT / calibration["arms"][arm]["checkpoint"]
    if sha256(checkpoint) != calibration["arms"][arm]["checkpoint_sha256"]:
        raise RuntimeError("checkpoint identity changed")
    model = base.build_model(training, arm, device)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval().requires_grad_(False)
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True, reduction="none").to(device).eval()

    baseline_psnr: list[float] = []
    baseline_lpips: list[float] = []
    all_sizes: list[int] = []
    for degraded, target, _, sizes in batches:
        p, l = metric_rows(degraded.to(device), target.to(device), lpips)
        baseline_psnr.extend(p); baseline_lpips.extend(l); all_sizes.extend(sizes)
    baseline = summarize(baseline_psnr, baseline_lpips, all_sizes)

    candidates = []
    for steps in map(int, calibration["grid"]["descent_steps"]):
        for alpha in map(float, calibration["grid"]["step_sizes"]):
            psnr_values: list[float] = []
            lpips_values: list[float] = []
            sizes_values: list[int] = []
            for degraded, target, q, sizes in batches:
                degraded, target, q = degraded.to(device), target.to(device), q.to(device)
                with torch.enable_grad():
                    prediction = restore(model, degraded, q, training, alpha=alpha, steps=steps)
                p, l = metric_rows(prediction, target, lpips)
                psnr_values.extend(p); lpips_values.extend(l); sizes_values.extend(sizes)
            metrics = summarize(psnr_values, lpips_values, sizes_values)
            candidate = {
                "candidate_id": f"steps{steps}_alpha{alpha:g}",
                "descent_steps": steps,
                "step_size": alpha,
                "metrics": metrics,
                "overall_minus_input": {
                    "psnr_db": metrics["overall"]["psnr"] - baseline["overall"]["psnr"],
                    "lpips": metrics["overall"]["lpips"] - baseline["overall"]["lpips"],
                },
            }
            candidates.append(candidate)
            print("EBT_CALIBRATION " + json.dumps({"arm": arm, **candidate}, sort_keys=True), flush=True)
    result = {
        "schema_version": 1,
        "status": "foundir_ebt_100k_inference_calibration_arm_complete",
        "arm": arm,
        "calibration_config_sha256": sha256(CALIBRATION_CONFIG),
        "training_config_sha256": calibration["training_config_sha256"],
        "checkpoint_sha256": calibration["arms"][arm]["checkpoint_sha256"],
        "input_baseline": baseline,
        "candidates": candidates,
        "pareto_candidate_ids": pareto(candidates),
        "triple_accessed": False,
        "official_foundir_test_accessed": False,
        "sealed_internal_accessed": False,
    }
    write_json(ROOT / calibration["output_root"] / f"{arm}.json", result)


def finalize() -> None:
    calibration = json.loads(CALIBRATION_CONFIG.read_text())
    output = ROOT / calibration["output_root"]
    arms = {arm: json.loads((output / f"{arm}.json").read_text()) for arm in calibration["arms"]}
    if any(value["calibration_config_sha256"] != sha256(CALIBRATION_CONFIG) for value in arms.values()):
        raise RuntimeError("calibration arm identity mismatch")
    result = {
        "schema_version": 1,
        "status": "foundir_ebt_100k_inference_calibration_complete",
        "arms": arms,
        "triple_accessed": False,
        "official_foundir_test_accessed": False,
        "sealed_internal_accessed": False,
    }
    write_json(output / "result.json", result)
    print("EBT_100K_INFERENCE_CALIBRATION_COMPLETE")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("plain", "compositional"))
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    if args.finalize:
        if args.arm: parser.error("--finalize does not accept --arm")
        finalize()
    else:
        if not args.arm: parser.error("arm run requires --arm")
        run_arm(args.arm)


if __name__ == "__main__":
    main()
