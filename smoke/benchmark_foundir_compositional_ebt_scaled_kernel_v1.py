#!/usr/bin/env python3
"""Bounded synthetic full-update benchmark for credible larger restoration EBTs."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lowlight_restoration.compositional_ebt import (  # noqa: E402
    RestorationEnergyTransformer,
    count_trainable_parameters,
)


CONFIG_PATH = ROOT / "configs/foundir_compositional_ebt_scaled_kernel_benchmark_v1.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def build_model(config: dict[str, Any], model_name: str, device: torch.device) -> RestorationEnergyTransformer:
    shared = config["shared_model"]
    profile = config["model_profiles"][model_name]
    model = RestorationEnergyTransformer(
        image_size=int(shared["image_size"]),
        patch_size=int(shared["patch_size"]),
        hidden_size=int(profile["hidden_size"]),
        depth=int(profile["depth"]),
        num_heads=int(profile["num_heads"]),
        mlp_ratio=float(shared["mlp_ratio"]),
        mode=str(shared["mode"]),
        zero_init_energy_head=bool(shared["zero_init_energy_head"]),
    ).to(device)
    actual = count_trainable_parameters(model)
    if actual != int(profile["expected_trainable_parameters"]):
        raise RuntimeError(f"parameter count changed: expected={profile['expected_trainable_parameters']} actual={actual}")
    return model


def synthetic_batch(batch: int, size: int, device: torch.device, seed: int) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator(device=device).manual_seed(seed)
    degraded = torch.rand((batch, 3, size, size), generator=generator, device=device).mul_(2).sub_(1)
    target = (degraded + torch.randn(degraded.shape, generator=generator, device=device) * 0.05).clamp_(-1, 1)
    q = torch.zeros((batch, 5), device=device)
    for index in range(batch):
        first = index % 5
        q[index, first] = 1
        if index % 2:
            q[index, (first + 1 + (index // 2) % 4) % 5] = 1
    return degraded, target, q


def optimizer_step(
    model: RestorationEnergyTransformer,
    optimizer: torch.optim.Optimizer,
    degraded: torch.Tensor,
    target: torch.Tensor,
    q: torch.Tensor,
    *,
    descent_steps: int,
    step_size: float,
    clamp: tuple[float, float],
    clip: float,
) -> tuple[float, float]:
    optimizer.zero_grad(set_to_none=True)
    candidate = degraded.clone()
    for _ in range(descent_steps):
        candidate = model.descent_step(
            degraded,
            candidate,
            q,
            step_size=step_size,
            create_graph=True,
            clamp=clamp,
        ).candidate
    loss = F.smooth_l1_loss(candidate, target)
    if not bool(torch.isfinite(loss)):
        raise RuntimeError("nonfinite benchmark loss")
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    if not bool(torch.isfinite(norm)):
        raise RuntimeError("nonfinite benchmark gradient")
    optimizer.step()
    return float(loss.detach()), float(norm.detach())


def run_profile(profile_name: str, output: Path) -> None:
    config = json.loads(CONFIG_PATH.read_text())
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("each benchmark worker requires exactly one visible CUDA GPU")
    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(0)
    if "H200" not in gpu_name and "H100" not in gpu_name:
        raise RuntimeError(f"benchmark requires NVIDIA H100/H200, got {gpu_name}")
    try:
        model_name, descent_name = profile_name.split("__", 1)
        descent_steps = int(config["descent_profiles"][descent_name])
        config["model_profiles"][model_name]
    except (KeyError, ValueError) as exc:
        raise ValueError(f"unknown benchmark profile: {profile_name}") from exc

    settings = config["benchmark"]
    results: list[dict[str, Any]] = []
    torch.manual_seed(int(settings["seed"]))
    torch.cuda.manual_seed(int(settings["seed"]))
    for batch in map(int, settings["batch_sizes"]):
        model = optimizer = degraded = target = q = None
        try:
            torch.cuda.empty_cache()
            model = build_model(config, model_name, device)
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=float(settings["learning_rate"]),
                weight_decay=float(settings["weight_decay"]),
            )
            degraded, target, q = synthetic_batch(
                batch,
                int(config["shared_model"]["image_size"]),
                device,
                int(settings["seed"]) + batch,
            )
            for _ in range(int(settings["warmup_optimizer_steps"])):
                optimizer_step(
                    model, optimizer, degraded, target, q,
                    descent_steps=descent_steps,
                    step_size=float(settings["candidate_step_size"]),
                    clamp=tuple(map(float, settings["candidate_clamp"])),
                    clip=float(settings["gradient_clip_norm"]),
                )
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            durations, losses, norms = [], [], []
            for _ in range(int(settings["timed_optimizer_steps"])):
                started = time.perf_counter()
                loss, norm = optimizer_step(
                    model, optimizer, degraded, target, q,
                    descent_steps=descent_steps,
                    step_size=float(settings["candidate_step_size"]),
                    clamp=tuple(map(float, settings["candidate_clamp"])),
                    clip=float(settings["gradient_clip_norm"]),
                )
                torch.cuda.synchronize()
                durations.append(time.perf_counter() - started)
                losses.append(loss)
                norms.append(norm)
            results.append({
                "batch_size": batch,
                "status": "pass",
                "mean_full_optimizer_step_seconds": sum(durations) / len(durations),
                "images_per_second": batch * len(durations) / sum(durations),
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
                "losses": losses,
                "gradient_norms": norms,
            })
            print("EBT_SCALED_KERNEL_BENCHMARK " + json.dumps(results[-1], sort_keys=True), flush=True)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
            if not isinstance(exc, torch.cuda.OutOfMemoryError) and "out of memory" not in str(exc).lower():
                raise
            results.append({"batch_size": batch, "status": "oom", "error": str(exc)})
            print("EBT_SCALED_KERNEL_BENCHMARK " + json.dumps(results[-1], sort_keys=True), flush=True)
            if bool(settings["stop_larger_batches_after_oom"]):
                break
        finally:
            del model, optimizer, degraded, target, q
            gc.collect()
            torch.cuda.empty_cache()

    passing = [row for row in results if row["status"] == "pass"]
    payload = {
        "schema_version": 1,
        "status": "scaled_ebt_kernel_profile_complete" if passing else "scaled_ebt_kernel_profile_no_passing_batch",
        "profile": profile_name,
        "model_profile": model_name,
        "descent_profile": descent_name,
        "descent_steps": descent_steps,
        "trainable_parameters": int(config["model_profiles"][model_name]["expected_trainable_parameters"]),
        "config_sha256": sha256(CONFIG_PATH),
        "gpu": gpu_name,
        "gpu_total_memory_bytes": int(torch.cuda.get_device_properties(0).total_memory),
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "results": results,
        "checkpoint_saved": False,
        "training_manifest_accessed": False,
        "triple_accessed": False,
        "official_foundir_test_accessed": False,
        "sealed_internal_accessed": False,
    }
    write_json_atomic(output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


def finalize(run_id: str) -> None:
    config = json.loads(CONFIG_PATH.read_text())
    root = ROOT / config["runtime"]["output_root"] / run_id
    profiles = [
        f"{model}__{descent}"
        for model in config["model_profiles"]
        for descent in config["descent_profiles"]
    ]
    payloads = []
    for profile in profiles:
        path = root / f"{profile}.json"
        item = json.loads(path.read_text())
        if item["config_sha256"] != sha256(CONFIG_PATH) or item["profile"] != profile:
            raise RuntimeError(f"profile identity mismatch: {path}")
        if item["status"] != "scaled_ebt_kernel_profile_complete":
            raise RuntimeError(f"profile has no passing batch: {profile}")
        payloads.append(item)
    result = {
        "schema_version": 1,
        "status": "foundir_compositional_ebt_scaled_kernel_benchmark_v1_complete",
        "run_id": run_id,
        "config_sha256": sha256(CONFIG_PATH),
        "profiles": payloads,
        "scientific_result": False,
        "full_training_launched": False,
        "checkpoint_saved": False,
        "training_manifest_accessed": False,
        "triple_accessed": False,
        "official_foundir_test_accessed": False,
        "sealed_internal_accessed": False,
    }
    write_json_atomic(root / "result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--finalize", action="store_true")
    parser.add_argument("--run-id")
    args = parser.parse_args()
    if args.finalize:
        if not args.run_id or args.profile or args.output:
            parser.error("--finalize requires only --run-id")
        finalize(args.run_id)
    else:
        if not args.profile or args.output is None or args.run_id:
            parser.error("profile run requires --profile and --output")
        run_profile(args.profile, args.output)


if __name__ == "__main__":
    main()
