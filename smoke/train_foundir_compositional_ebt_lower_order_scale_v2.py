#!/usr/bin/env python3
"""Matched 10k-step lower-order scale test for Plain and Compositional EBT.

This program deliberately never opens a held-out-triple image.  It trains both
matched arms on the frozen R0 singleton/double train parents, freezes both
final checkpoints, and then evaluates one fixed crop from the source-disjoint
singleton/double development parents.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lowlight_restoration.compositional_ebt import (  # noqa: E402
    DEGRADATIONS,
    RestorationEnergyTransformer,
    count_trainable_parameters,
)


CONFIG_PATH = ROOT / "configs/foundir_compositional_ebt_lower_order_scale_v2.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != encoded:
            raise RuntimeError(f"refusing to overwrite differing artifact: {path}")
        return
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_bytes(encoded)
    os.replace(temporary, path)


def load_rgb(path: Path) -> Tensor:
    with Image.open(path) as handle:
        pixels = np.asarray(handle.convert("RGB"), dtype=np.uint8).copy()
    return torch.from_numpy(pixels).permute(2, 0, 1).float().div_(127.5).sub_(1.0)


def node_name(record: dict[str, Any]) -> str:
    order = list(map(str, record["application_order"]))
    return "+".join(order) if order else "clean"


def q_for(record: dict[str, Any]) -> Tensor:
    present = set(map(str, record["degradations"]))
    return torch.tensor([float(name in present) for name in DEGRADATIONS])


class LowerOrderData:
    def __init__(self, config: dict[str, Any]) -> None:
        data = config["data"]
        manifest_path = ROOT / data["protocol_manifest"]
        metadata_path = ROOT / data["protocol_metadata"]
        cache_manifest_path = ROOT / data["cache_manifest"]
        expected_hashes = (
            (manifest_path, data["protocol_manifest_sha256"]),
            (metadata_path, data["protocol_metadata_sha256"]),
            (cache_manifest_path, data["cache_manifest_sha256"]),
        )
        for path, expected in expected_hashes:
            if sha256_file(path) != expected:
                raise RuntimeError(f"frozen input hash mismatch: {path}")

        records = load_jsonl(manifest_path)
        self.records = {
            role: sorted(
                [row for row in records if row["protocol_role"] == role],
                key=lambda row: str(row["parent_record_id"]),
            )
            for role in ("train", "development")
        }
        expected_counts = {
            "train": int(data["gradient_parents"]),
            "development": int(data["development_parents"]),
        }
        if {role: len(rows) for role, rows in self.records.items()} != expected_counts:
            raise RuntimeError("lower-order parent counts changed")
        if any(
            int(row["composition_size"]) not in {1, 2}
            for rows in self.records.values()
            for row in rows
        ):
            raise RuntimeError("triple leakage into lower-order train/development")

        relevant = {
            (str(row["parent_record_id"]), name)
            for rows in self.records.values()
            for row in rows
            for name in ("clean", node_name(row))
        }
        cache_rows: dict[tuple[str, str], dict[str, Any]] = {}
        for row in load_jsonl(cache_manifest_path):
            key = (str(row["parent_record_id"]), str(row["node_id"]))
            if key in relevant:
                if row["protocol_role"] not in {"train", "development"}:
                    raise RuntimeError("held-out cache row entered lower-order lookup")
                cache_rows[key] = row
        if set(cache_rows) != relevant:
            raise RuntimeError("lower-order rendered cache is incomplete")

        self.images: dict[tuple[str, str], Tensor] = {}
        for key, row in cache_rows.items():
            path = ROOT / str(row["path"])
            if "heldout_triple" in path.parts:
                raise RuntimeError("refusing held-out-triple image path")
            if sha256_file(path) != row["sha256"]:
                raise RuntimeError(f"rendered image hash mismatch: {path}")
            self.images[key] = load_rgb(path)

    def pair(self, record: dict[str, Any]) -> tuple[Tensor, Tensor, Tensor]:
        parent = str(record["parent_record_id"])
        return self.images[(parent, node_name(record))], self.images[(parent, "clean")], q_for(record)


def crop_pair(
    degraded: Tensor,
    clean: Tensor,
    *,
    crop_size: int,
    seed: int,
    augment: bool,
) -> tuple[Tensor, Tensor]:
    if degraded.shape != clean.shape:
        raise RuntimeError("degraded/clean shape mismatch")
    height, width = degraded.shape[-2:]
    if crop_size > min(height, width):
        raise RuntimeError("crop exceeds source image")
    if augment:
        generator = random.Random(seed)
        y = generator.randrange(height - crop_size + 1)
        x = generator.randrange(width - crop_size + 1)
    else:
        y = (height - crop_size) // 2
        x = (width - crop_size) // 2
    outputs = [image[:, y : y + crop_size, x : x + crop_size] for image in (degraded, clean)]
    if augment:
        if generator.randrange(2):
            outputs = [image.flip(-1) for image in outputs]
        if generator.randrange(2):
            outputs = [image.flip(-2) for image in outputs]
        if generator.randrange(2):
            outputs = [image.transpose(-2, -1) for image in outputs]
    return outputs[0], outputs[1]


def build_model(config: dict[str, Any], mode: str, device: torch.device) -> RestorationEnergyTransformer:
    model_config = config["model"]
    torch.manual_seed(int(model_config["initialization_seed"]))
    if device.type == "cuda":
        torch.cuda.manual_seed(int(model_config["initialization_seed"]))
    model = RestorationEnergyTransformer(
        image_size=int(model_config["image_size"]),
        patch_size=int(model_config["patch_size"]),
        hidden_size=int(model_config["hidden_size"]),
        depth=int(model_config["depth"]),
        num_heads=int(model_config["num_heads"]),
        mlp_ratio=float(model_config["mlp_ratio"]),
        mode=mode,
        zero_init_energy_head=bool(model_config["zero_init_energy_head"]),
    ).to(device)
    if count_trainable_parameters(model) != int(model_config["trainable_parameters_per_arm"]):
        raise RuntimeError("model parameter count changed")
    return model


def restore(
    model: RestorationEnergyTransformer,
    degraded: Tensor,
    q: Tensor,
    config: dict[str, Any],
    *,
    create_graph: bool,
) -> Tensor:
    descent = config["energy_descent"]
    candidate = degraded.clone()
    for _ in range(int(descent["training_steps"] if create_graph else descent["inference_steps"])):
        step = model.descent_step(
            degraded,
            candidate,
            q,
            step_size=float(descent["candidate_step_size"]),
            create_graph=create_graph,
            clamp=tuple(map(float, descent["candidate_clamp"])),
        )
        candidate = step.candidate if create_graph else step.candidate.detach()
    return candidate


def make_batch(
    data: LowerOrderData,
    records: list[dict[str, Any]],
    config: dict[str, Any],
    step: int,
) -> tuple[Tensor, Tensor, Tensor]:
    batch_size = int(config["training"]["batch_size"])
    crop_size = int(config["model"]["image_size"])
    base_seed = int(config["model"]["initialization_seed"]) + 100000
    degraded_batch, clean_batch, q_batch = [], [], []
    for offset in range(batch_size):
        presentation = (step - 1) * batch_size + offset
        record = records[presentation % len(records)]
        degraded, clean, q = data.pair(record)
        degraded, clean = crop_pair(
            degraded,
            clean,
            crop_size=crop_size,
            seed=base_seed + presentation,
            augment=True,
        )
        degraded_batch.append(degraded)
        clean_batch.append(clean)
        q_batch.append(q)
    return torch.stack(degraded_batch), torch.stack(clean_batch), torch.stack(q_batch)


def train_arm(
    mode: str,
    data: LowerOrderData,
    config: dict[str, Any],
    device: torch.device,
    output_root: Path,
) -> dict[str, Any]:
    model = build_model(config, mode, device)
    training = config["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        betas=tuple(map(float, training["betas"])),
        weight_decay=float(training["weight_decay"]),
    )
    total_steps = int(training["optimizer_steps_per_arm"])
    metrics: list[dict[str, float | int | str]] = []
    started = time.perf_counter()
    model.train()
    for step in range(1, total_steps + 1):
        degraded, clean, q = make_batch(data, data.records["train"], config, step)
        degraded = degraded.to(device)
        clean = clean.to(device)
        q = q.to(device)
        optimizer.zero_grad(set_to_none=True)
        restored = restore(model, degraded, q, config, create_graph=True)
        loss = F.smooth_l1_loss(restored, clean)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"nonfinite loss: arm={mode} step={step}")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(training["gradient_clip_norm"])
        )
        if not bool(torch.isfinite(grad_norm)):
            raise RuntimeError(f"nonfinite gradient: arm={mode} step={step}")
        optimizer.step()
        row = {
            "arm": mode,
            "step": step,
            "loss": float(loss.detach().cpu()),
            "gradient_norm": float(grad_norm.detach().cpu()),
        }
        metrics.append(row)
        if step == 1 or step % 10 == 0 or step == total_steps:
            elapsed = time.perf_counter() - started
            estimate = elapsed / step * total_steps
            print(
                "FOUNDIR_EBT_LOWER_ORDER_TRAIN "
                + json.dumps({**row, "elapsed_seconds": elapsed, "arm_estimated_seconds": estimate}, sort_keys=True),
                flush=True,
            )

    checkpoint = output_root / mode / f"step_{total_steps:06d}.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    temporary = checkpoint.with_name(checkpoint.name + f".tmp.{os.getpid()}")
    torch.save(
        {
            "schema_version": 1,
            "mode": mode,
            "step": total_steps,
            "config_sha256": sha256_file(CONFIG_PATH),
            "model": model.state_dict(),
        },
        temporary,
    )
    os.replace(temporary, checkpoint)
    window = min(25, len(metrics))
    first = sum(float(row["loss"]) for row in metrics[:window]) / window
    last = sum(float(row["loss"]) for row in metrics[-window:]) / window
    return {
        "mode": mode,
        "checkpoint": checkpoint.relative_to(ROOT).as_posix(),
        "checkpoint_sha256": sha256_file(checkpoint),
        "optimizer_steps": total_steps,
        "elapsed_seconds": time.perf_counter() - started,
        "first_25_mean_loss": first,
        "last_25_mean_loss": last,
        "last_over_first_loss_ratio": last / first if first > 0 else math.inf,
        "training_metrics": metrics,
    }


def evaluate_arm(
    mode: str,
    checkpoint_path: Path,
    data: LowerOrderData,
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    model = build_model(config, mode, device)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if payload["mode"] != mode or payload["config_sha256"] != sha256_file(CONFIG_PATH):
        raise RuntimeError("checkpoint identity mismatch")
    model.load_state_dict(payload["model"], strict=True)
    model.requires_grad_(False).eval()
    input_mse, output_mse = [], []
    input_smooth, output_smooth, output_means = [], [], []
    crop_size = int(config["model"]["image_size"])
    per_record = []
    for record in data.records["development"]:
        degraded, clean, q = data.pair(record)
        degraded, clean = crop_pair(
            degraded, clean, crop_size=crop_size, seed=0, augment=False
        )
        degraded = degraded.unsqueeze(0).to(device)
        clean = clean.unsqueeze(0).to(device)
        q = q.unsqueeze(0).to(device)
        with torch.enable_grad():
            restored = restore(model, degraded, q, config, create_graph=False)
        if not bool(torch.isfinite(restored).all()):
            raise RuntimeError(f"nonfinite development output: {mode}")
        in_mse = float(F.mse_loss(degraded, clean).detach().cpu()) / 4.0
        out_mse = float(F.mse_loss(restored, clean).detach().cpu()) / 4.0
        in_smooth = float(F.smooth_l1_loss((degraded + 1) / 2, (clean + 1) / 2).detach().cpu())
        out_smooth = float(F.smooth_l1_loss((restored + 1) / 2, (clean + 1) / 2).detach().cpu())
        mean = float(((restored + 1) / 2).mean().detach().cpu())
        input_mse.append(in_mse)
        output_mse.append(out_mse)
        input_smooth.append(in_smooth)
        output_smooth.append(out_smooth)
        output_means.append(mean)
        per_record.append(
            {
                "parent_record_id": record["parent_record_id"],
                "composition_size": record["composition_size"],
                "degradations": record["degradations"],
                "input_psnr": -10.0 * math.log10(max(in_mse, 1e-12)),
                "output_psnr": -10.0 * math.log10(max(out_mse, 1e-12)),
                "input_smooth_l1": in_smooth,
                "output_smooth_l1": out_smooth,
                "output_mean_0_1": mean,
            }
        )
    aggregate_input_mse = sum(input_mse) / len(input_mse)
    aggregate_output_mse = sum(output_mse) / len(output_mse)
    input_psnr = -10.0 * math.log10(max(aggregate_input_mse, 1e-12))
    output_psnr = -10.0 * math.log10(max(aggregate_output_mse, 1e-12))
    return {
        "records": len(per_record),
        "input_psnr": input_psnr,
        "output_psnr": output_psnr,
        "psnr_gain_over_input": output_psnr - input_psnr,
        "input_smooth_l1": sum(input_smooth) / len(input_smooth),
        "output_smooth_l1": sum(output_smooth) / len(output_smooth),
        "smooth_l1_improvement": (sum(input_smooth) - sum(output_smooth)) / len(input_smooth),
        "output_mean_0_1": sum(output_means) / len(output_means),
        "per_record": per_record,
    }


def preflight(config: dict[str, Any], data: LowerOrderData) -> dict[str, Any]:
    device = torch.device("cpu")
    batch = make_batch(data, data.records["train"], config, 1)
    checks = {}
    reference_state = None
    for mode in config["training"]["arms"]:
        model = build_model(config, mode, device)
        state = {name: tensor.detach().clone() for name, tensor in model.named_parameters()}
        if reference_state is None:
            reference_state = state
        elif any(not torch.equal(tensor, reference_state[name]) for name, tensor in state.items()):
            raise RuntimeError("matched arms do not have identical initialization")
        degraded, clean, q = batch
        restored = restore(model, degraded[:1], q[:1], config, create_graph=True)
        loss = F.smooth_l1_loss(restored, clean[:1])
        loss.backward()
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        checks[mode] = {
            "loss": float(loss.detach()),
            "finite_output": bool(torch.isfinite(restored).all()),
            "finite_gradients": bool(gradients) and all(bool(torch.isfinite(g).all()) for g in gradients),
            "parameters_with_gradient": len(gradients),
        }
    passed = all(item["finite_output"] and item["finite_gradients"] for item in checks.values())
    return {
        "schema_version": 1,
        "status": "cpu_preflight_pass" if passed else "cpu_preflight_fail",
        "config_sha256": sha256_file(CONFIG_PATH),
        "train_parents": len(data.records["train"]),
        "development_parents": len(data.records["development"]),
        "heldout_triple_images_accessed": False,
        "official_foundir_test_accessed": False,
        "sealed_internal_accessed": False,
        "arms": checks,
    }


def finalize(
    config: dict[str, Any],
    data: LowerOrderData,
    device: torch.device,
    output_root: Path,
) -> dict[str, Any]:
    started = time.time()
    arms = []
    for mode in config["training"]["arms"]:
        arm_result_path = output_root / f"{mode}_train_result.json"
        if not arm_result_path.is_file():
            raise RuntimeError(f"missing frozen arm result: {arm_result_path}")
        arm_result = json.loads(arm_result_path.read_text())
        if arm_result.get("status") != "arm_training_complete":
            raise RuntimeError(f"arm did not complete: {mode}")
        if arm_result.get("config_sha256") != sha256_file(CONFIG_PATH):
            raise RuntimeError(f"arm config mismatch: {mode}")
        arms.append(arm_result["arm"])
    # Both final checkpoints exist and are hashed before any development image is evaluated.
    for arm in arms:
        checkpoint = ROOT / arm["checkpoint"]
        if not checkpoint.is_file() or sha256_file(checkpoint) != arm["checkpoint_sha256"]:
            raise RuntimeError("checkpoint freeze verification failed")
    evaluations = {
        arm["mode"]: evaluate_arm(arm["mode"], ROOT / arm["checkpoint"], data, config, device)
        for arm in arms
    }
    gate = config["lower_order_gate"]
    mean_min, mean_max = map(float, gate["output_mean_range_0_1"])
    arm_pass = {}
    for arm in arms:
        metrics = evaluations[arm["mode"]]
        arm_pass[arm["mode"]] = bool(
            metrics["psnr_gain_over_input"] >= float(gate["each_arm_development_psnr_gain_over_input_min_db"])
            and metrics["smooth_l1_improvement"] > float(gate["each_arm_development_smooth_l1_improvement_over_input_min"])
            and mean_min <= metrics["output_mean_0_1"] <= mean_max
        )
    passed = all(arm_pass.values())
    return {
        "schema_version": 1,
        "status": "lower_order_learnability_pass" if passed else "lower_order_learnability_fail",
        "automatic_gate_pass": passed,
        "config_sha256": sha256_file(CONFIG_PATH),
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "wall_seconds": time.time() - started,
        "optimizer_steps_total": sum(int(arm["optimizer_steps"]) for arm in arms),
        "arms": arms,
        "development": evaluations,
        "arm_gate_pass": arm_pass,
        "heldout_triple_images_accessed": False,
        "foundir_60_image_triple_accessed": False,
        "official_foundir_test_accessed": False,
        "sealed_internal_accessed": False,
        "next_action": "register separate one-time held-out-triple evaluation" if passed else "stop without triple access",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--result", type=Path)
    parser.add_argument("--arm", choices=("plain", "compositional"))
    parser.add_argument("--finalize", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(CONFIG_PATH.read_text())
    data = LowerOrderData(config)
    if args.preflight:
        result = preflight(config, data)
    else:
        device = torch.device(args.device)
        if device.type != "cuda" or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("pilot requires exactly one visible CUDA GPU")
        gpu_name = torch.cuda.get_device_name(device)
        if "H100" not in gpu_name and "H200" not in gpu_name:
            raise RuntimeError(f"unsupported GPU: {gpu_name}")
        if args.output_root is None:
            raise RuntimeError("--output-root is required for training")
        if args.arm is not None and args.finalize:
            raise RuntimeError("choose either --arm or --finalize")
        if args.arm is not None:
            arm = train_arm(args.arm, data, config, device, args.output_root)
            result = {
                "schema_version": 1,
                "status": "arm_training_complete",
                "config_sha256": sha256_file(CONFIG_PATH),
                "gpu_name": gpu_name,
                "arm": arm,
                "heldout_triple_images_accessed": False,
                "official_foundir_test_accessed": False,
                "sealed_internal_accessed": False,
            }
        elif args.finalize:
            result = finalize(config, data, device, args.output_root)
        else:
            raise RuntimeError("training requires exactly one of --arm or --finalize")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    if args.result is not None:
        write_json_atomic(args.result, result)
    if result["status"].endswith("fail"):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
