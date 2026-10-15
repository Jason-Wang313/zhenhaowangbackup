#!/usr/bin/env python3
"""Matched real lower-order data smoke for 26M Plain/Compositional EBT arms."""

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
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# The frozen renderer needs OpenCV, but the project environment's GUI-linked
# wheel requires an unavailable libGL. Load the host's headless-compatible
# OpenCV with the required project interpreter, then restore normal import
# precedence before importing the renderer and SciPy dependencies.
HEADLESS_CV2_ROOT = "/usr/local/lib/python3.12/dist-packages"
sys.path.insert(0, HEADLESS_CV2_ROOT)
try:
    import cv2 as _cv2  # noqa: F401, E402
finally:
    sys.path.remove(HEADLESS_CV2_ROOT)

from lowlight_restoration.compositional_ebt import (  # noqa: E402
    DEGRADATIONS,
    RestorationEnergyTransformer,
    count_trainable_parameters,
)
from lowlight_restoration.multidegradation_data import generate_record_stages  # noqa: E402


CONFIG_PATH = ROOT / "configs/foundir_compositional_ebt_scaled_learning_smoke_v1.json"


def sha256(path: Path) -> str:
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


def stable_key(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


class LowerOrderDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(self, records: list[dict[str, Any]], *, crop: int, seed: int, augment: bool) -> None:
        self.records = records
        self.crop = crop
        self.seed = seed
        self.augment = augment

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        record = self.records[index]
        with Image.open(ROOT / str(record["output_path"])) as handle:
            degraded = np.asarray(handle.convert("RGB"), dtype=np.uint8).copy()
        # The deterministic renderer works internally in OpenCV BGR. Convert
        # its clean/no-nuisance target to RGB to match PIL-loaded degraded PNGs.
        clean = generate_record_stages(record, application_order=[])[0][..., ::-1].copy()
        if degraded.shape != clean.shape or min(degraded.shape[:2]) < self.crop:
            raise RuntimeError(f"bad pair shape: {record['record_id']}")
        height, width = degraded.shape[:2]
        rng = random.Random(self.seed + index)
        if self.augment:
            y = rng.randrange(height - self.crop + 1)
            x = rng.randrange(width - self.crop + 1)
        else:
            y = (height - self.crop) // 2
            x = (width - self.crop) // 2
        degraded = degraded[y : y + self.crop, x : x + self.crop]
        clean = clean[y : y + self.crop, x : x + self.crop]
        if self.augment:
            if rng.randrange(2):
                degraded, clean = np.flip(degraded, 1), np.flip(clean, 1)
            if rng.randrange(2):
                degraded, clean = np.flip(degraded, 0), np.flip(clean, 0)
            if rng.randrange(2):
                degraded, clean = np.swapaxes(degraded, 0, 1), np.swapaxes(clean, 0, 1)
        degraded = np.ascontiguousarray(degraded)
        clean = np.ascontiguousarray(clean)
        d = torch.from_numpy(degraded).permute(2, 0, 1).float().div_(127.5).sub_(1)
        c = torch.from_numpy(clean).permute(2, 0, 1).float().div_(127.5).sub_(1)
        present = set(map(str, record["degradations"]))
        q = torch.tensor([float(name in present) for name in DEGRADATIONS])
        return d, c, q


def load_records(config: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    data = config["data"]
    manifest = ROOT / data["manifest"]
    metadata = ROOT / data["metadata"]
    if sha256(manifest) != data["manifest_sha256"] or sha256(metadata) != data["metadata_sha256"]:
        raise RuntimeError("scaled lower-order data identity changed")
    rows = load_jsonl(manifest)
    allowed = set(map(int, data["allowed_composition_sizes"]))
    if any(int(row["composition_size"]) not in allowed for row in rows):
        raise RuntimeError("non-lower-order record entered smoke")
    train = [row for row in rows if row["split"] == "train"]
    validation = [row for row in rows if row["split"] == "validation"]
    if len(train) != int(data["train_records"]) or len(validation) != int(data["validation_records"]):
        raise RuntimeError("scaled lower-order counts changed")
    seed = int(config["model"]["initialization_seed"])
    train.sort(key=lambda row: stable_key(seed, str(row["record_id"])))
    validation.sort(key=lambda row: stable_key(seed + 1, str(row["record_id"])))
    needed = int(config["training"]["optimizer_steps_per_arm"]) * int(config["training"]["batch_size_per_arm"])
    if needed > len(train):
        train = [train[index % len(train)] for index in range(needed)]
    else:
        train = train[:needed]
    validation = validation[: int(data["validation_subset_records"])]
    return train, validation


def loader(records: list[dict[str, Any]], config: dict[str, Any], *, train: bool) -> DataLoader:
    settings = config["training"]
    workers = int(settings["num_workers_per_arm"])
    return DataLoader(
        LowerOrderDataset(
            records,
            crop=int(config["data"]["crop_size"]),
            seed=int(config["model"]["initialization_seed"]) + (1000 if train else 2000),
            augment=train,
        ),
        batch_size=int(settings["batch_size_per_arm"]) if train else 32,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        drop_last=train,
        persistent_workers=workers > 0,
        prefetch_factor=int(settings["prefetch_factor"]) if workers > 0 else None,
    )


def build_model(config: dict[str, Any], mode: str, device: torch.device) -> RestorationEnergyTransformer:
    model_config = config["model"]
    torch.manual_seed(int(model_config["initialization_seed"]))
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


def restore(model: RestorationEnergyTransformer, degraded: torch.Tensor, q: torch.Tensor, config: dict[str, Any], *, create_graph: bool) -> torch.Tensor:
    candidate = degraded.clone()
    descent = config["energy_descent"]
    for _ in range(int(descent["steps"])):
        candidate = model.descent_step(
            degraded, candidate, q,
            step_size=float(descent["candidate_step_size"]),
            create_graph=create_graph,
            clamp=tuple(map(float, descent["candidate_clamp"])),
        ).candidate
        if not create_graph:
            candidate = candidate.detach()
    return candidate


def evaluate(model: RestorationEnergyTransformer, batches: DataLoader, config: dict[str, Any], device: torch.device) -> dict[str, float | int]:
    model.eval()
    input_squared = output_squared = input_smooth = output_smooth = 0.0
    pixels = values = records = 0
    for degraded, clean, q in batches:
        degraded = degraded.to(device, non_blocking=True)
        clean = clean.to(device, non_blocking=True)
        q = q.to(device, non_blocking=True)
        with torch.enable_grad():
            output = restore(model, degraded, q, config, create_graph=False)
        d01, c01, o01 = (tensor.add(1).mul(0.5) for tensor in (degraded, clean, output))
        input_squared += float((d01 - c01).square().sum())
        output_squared += float((o01 - c01).square().sum())
        input_smooth += float(F.smooth_l1_loss(d01, c01, reduction="sum"))
        output_smooth += float(F.smooth_l1_loss(o01, c01, reduction="sum"))
        records += degraded.shape[0]
        values += degraded.numel()
        pixels += degraded.shape[0] * degraded.shape[2] * degraded.shape[3]
    input_mse, output_mse = input_squared / values, output_squared / values
    return {
        "records": records,
        "input_psnr": -10 * math.log10(max(input_mse, 1e-12)),
        "output_psnr": -10 * math.log10(max(output_mse, 1e-12)),
        "psnr_gain_db": -10 * math.log10(max(output_mse, 1e-12)) + 10 * math.log10(max(input_mse, 1e-12)),
        "input_smooth_l1": input_smooth / values,
        "output_smooth_l1": output_smooth / values,
    }


def train_arm(mode: str, output: Path) -> None:
    config = json.loads(CONFIG_PATH.read_text())
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("each arm requires exactly one visible CUDA GPU")
    gpu = torch.cuda.get_device_name(0)
    if "H100" not in gpu and "H200" not in gpu:
        raise RuntimeError(f"unsupported GPU: {gpu}")
    device = torch.device("cuda:0")
    train_records, validation_records = load_records(config)
    train_loader = loader(train_records, config, train=True)
    validation_loader = loader(validation_records, config, train=False)
    model = build_model(config, mode, device)
    before = evaluate(model, validation_loader, config, device)
    settings = config["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(settings["peak_learning_rate"]),
        betas=tuple(map(float, settings["betas"])), weight_decay=float(settings["weight_decay"]),
    )
    steps = int(settings["optimizer_steps_per_arm"])
    warmup = int(settings["warmup_steps"])
    losses: list[float] = []
    data_seconds = compute_seconds = 0.0
    started = time.perf_counter()
    iterator = iter(train_loader)
    model.train()
    for step in range(1, steps + 1):
        data_started = time.perf_counter()
        degraded, clean, q = next(iterator)
        data_seconds += time.perf_counter() - data_started
        degraded = degraded.to(device, non_blocking=True)
        clean = clean.to(device, non_blocking=True)
        q = q.to(device, non_blocking=True)
        learning_rate = float(settings["peak_learning_rate"]) * min(1.0, step / max(1, warmup))
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        compute_started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        restored = restore(model, degraded, q, config, create_graph=True)
        loss = F.smooth_l1_loss(restored, clean)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"nonfinite loss at step {step}")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(settings["gradient_clip_norm"]))
        if not bool(torch.isfinite(norm)):
            raise RuntimeError(f"nonfinite gradient at step {step}")
        optimizer.step()
        torch.cuda.synchronize()
        compute_seconds += time.perf_counter() - compute_started
        losses.append(float(loss.detach()))
        if step == 1 or step % 25 == 0 or step == steps:
            elapsed = time.perf_counter() - started
            print("EBT_SCALED_REAL_SMOKE " + json.dumps({
                "arm": mode, "step": step, "loss": losses[-1], "gradient_norm": float(norm),
                "learning_rate": learning_rate, "elapsed_seconds": elapsed,
                "estimated_total_seconds": elapsed / step * steps,
            }, sort_keys=True), flush=True)
    after = evaluate(model, validation_loader, config, device)
    elapsed = time.perf_counter() - started
    window = min(25, len(losses))
    checkpoint_saved = bool(settings.get("checkpoint_saved", False))
    checkpoint_path: Path | None = None
    checkpoint_sha256: str | None = None
    if checkpoint_saved:
        checkpoint_path = output.parent / f"{mode}_step_{steps:06d}.pt"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = checkpoint_path.with_name(checkpoint_path.name + f".tmp.{os.getpid()}")
        torch.save({
            "schema_version": 1,
            "arm": mode,
            "optimizer_steps": steps,
            "config_sha256": sha256(CONFIG_PATH),
            "model_state_dict": model.state_dict(),
        }, temporary)
        os.replace(temporary, checkpoint_path)
        checkpoint_sha256 = sha256(checkpoint_path)
    payload = {
        "schema_version": 1,
        "status": "scaled_lower_order_learning_smoke_arm_complete",
        "arm": mode,
        "config_sha256": sha256(CONFIG_PATH),
        "gpu": gpu,
        "optimizer_steps": steps,
        "batch_size": int(settings["batch_size_per_arm"]),
        "presentations": steps * int(settings["batch_size_per_arm"]),
        "elapsed_seconds_including_validation": elapsed,
        "data_wait_seconds_training": data_seconds,
        "compute_seconds_training": compute_seconds,
        "mean_training_images_per_second": steps * int(settings["batch_size_per_arm"]) / elapsed,
        "first_25_mean_loss": sum(losses[:window]) / window,
        "last_25_mean_loss": sum(losses[-window:]) / window,
        "before": before,
        "after": after,
        "validation_psnr_change_db": float(after["output_psnr"]) - float(before["output_psnr"]),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "checkpoint_saved": checkpoint_saved,
        "checkpoint": checkpoint_path.relative_to(ROOT).as_posix() if checkpoint_path else None,
        "checkpoint_sha256": checkpoint_sha256,
        "triple_accessed": False,
        "official_foundir_test_accessed": False,
        "sealed_internal_accessed": False,
    }
    write_json_atomic(output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


def finalize(run_id: str) -> None:
    config = json.loads(CONFIG_PATH.read_text())
    root = ROOT / config["runtime"]["output_root"] / run_id
    arms = [json.loads((root / f"{arm}.json").read_text()) for arm in config["training"]["arms"]]
    if any(arm["config_sha256"] != sha256(CONFIG_PATH) for arm in arms):
        raise RuntimeError("arm config identity mismatch")
    for arm in arms:
        if arm.get("checkpoint_saved"):
            checkpoint = ROOT / str(arm["checkpoint"])
            if not checkpoint.is_file() or sha256(checkpoint) != arm.get("checkpoint_sha256"):
                raise RuntimeError(f"checkpoint identity mismatch for {arm['arm']}")
    passed = all(
        arm["status"] == "scaled_lower_order_learning_smoke_arm_complete"
        and arm["validation_psnr_change_db"] > 0
        for arm in arms
    )
    result = {
        "schema_version": 1,
        "status": "scaled_lower_order_learning_smoke_pass" if passed else "scaled_lower_order_learning_smoke_fail",
        "automatic_gate_pass": passed,
        "run_id": run_id,
        "config_sha256": sha256(CONFIG_PATH),
        "arms": arms,
        "full_100k_training_launched": False,
        "checkpoint_saved": all(bool(arm.get("checkpoint_saved")) for arm in arms),
        "triple_accessed": False,
        "official_foundir_test_accessed": False,
        "sealed_internal_accessed": False,
    }
    write_json_atomic(root / "result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("plain", "compositional"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--finalize", action="store_true")
    parser.add_argument("--run-id")
    args = parser.parse_args()
    if args.finalize:
        if not args.run_id or args.arm or args.output:
            parser.error("--finalize requires only --run-id")
        finalize(args.run_id)
    else:
        if not args.arm or args.output is None or args.run_id:
            parser.error("arm run requires --arm and --output")
        train_arm(args.arm, args.output)


if __name__ == "__main__":
    main()
