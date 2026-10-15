#!/usr/bin/env python3
"""Real-model 3/4/5-node DDP throughput and optimizer benchmark."""

from __future__ import annotations

import json
import math
import os
import socket
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from smoke.autotune_foundir_vlm_energy_intersection_v1_8gpu import load_crop, objective  # noqa: E402
from smoke.preflight_foundir_vlm_energy_intersection_v1_8gpu import build_model, load_jsonl, sha256  # noqa: E402


TRAIN_CONFIG = ROOT / "configs/foundir_vlm_energy_intersection_v1_train.json"


def gpu_family(name: str) -> str:
    """Return the accelerator family without depending on vendor name aliases."""
    upper = name.upper()
    for family in ("H200", "H100"):
        if family in upper:
            return family
    return "UNSUPPORTED"


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    local_world = int(os.environ["LOCAL_WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    config = json.loads(TRAIN_CONFIG.read_text())
    profiles = config["distributed"]["profiles"]
    if str(world) not in ("24", "32", "40") or str(world) not in profiles:
        raise RuntimeError(f"benchmark requires world size 24, 32, or 40; got {world}")
    profile = profiles[str(world)]
    if local_world != 8 or world != int(profile["nodes"]) * local_world:
        raise RuntimeError(f"requires {profile['nodes']} distinct 8-GPU nodes")
    batch = int(profile["batch_per_gpu"])
    if batch * world != int(profile["global_batch"]):
        raise RuntimeError("invalid batch profile")

    properties = torch.cuda.get_device_properties(device)
    identity = {
        "rank": rank,
        "local_rank": local_rank,
        "host": socket.gethostname(),
        "gpu": properties.name,
        "gpu_family": gpu_family(properties.name),
        "total_memory_bytes": properties.total_memory,
        "compute_capability": f"{properties.major}.{properties.minor}",
    }
    identities: list[dict[str, Any] | None] = [None] * world
    dist.all_gather_object(identities, identity)
    run_id = os.environ.get("VLM_ENERGY_MULTINODE_BENCHMARK_RUN_ID", "benchmark_v1")
    if rank == 0:
        print(
            "VLM_ENERGY_MULTINODE_GPU_IDENTITIES "
            + json.dumps(identities, sort_keys=True),
            flush=True,
        )
        audit_output = (
            ROOT
            / "checkpoints/foundir_vlm_energy_intersection_v1/multinode_benchmark"
            / run_id
            / "hardware_identity.json"
        )
        write_json_atomic(audit_output, {
            "schema_version": 1,
            "run_id": run_id,
            "world_size": world,
            "gpu_identities": identities,
            "official_foundir_test_accessed": False,
            "parent_test_accessed": False,
            "sealed_internal_accessed": False,
        })
    host_counts: dict[str, int] = {}
    for item in identities:
        assert item is not None
        host_counts[item["host"]] = host_counts.get(item["host"], 0) + 1
    if len(host_counts) != int(profile["nodes"]) or set(host_counts.values()) != {8}:
        raise RuntimeError(f"physical-node layout mismatch: {host_counts}")
    families = {item["gpu_family"] for item in identities if item}
    capabilities = {item["compute_capability"] for item in identities if item}
    minimum_memory = min(int(item["total_memory_bytes"]) for item in identities if item)
    if "UNSUPPORTED" in families:
        raise RuntimeError(
            "unsupported GPU family: "
            + json.dumps(identities, sort_keys=True)
        )
    if len(capabilities) != 1 or minimum_memory < 80_000_000_000:
        raise RuntimeError(
            "incompatible accelerator capability or memory: "
            + json.dumps(identities, sort_keys=True)
        )

    base_path = ROOT / config["base_config"]
    if sha256(base_path) != config["base_config_sha256"]:
        raise RuntimeError("base config changed")
    base = json.loads(base_path.read_text())
    rows = [row for row in load_jsonl(ROOT / base["data"]["manifest"]) if row["split"] == "train"]
    targets = {str(row["record_id"]): row for row in load_jsonl(ROOT / base["data"]["target_manifest"])}
    grouped = {size: [row for row in rows if int(row["composition_size"]) == size] for size in (1, 2, 3)}
    samples = []
    for index in range(batch):
        order = (rank + index) % 3 + 1
        row = grouped[order][rank * batch + index]
        samples.append(load_crop(
            row, targets[str(row["record_id"])], int(base["data"]["crop_size"]),
            int(base["model"]["initialization_seed"]) + rank * 100 + index,
        ))
    degraded = torch.stack([item[0] for item in samples]).to(device)
    clean = torch.stack([item[1] for item in samples]).to(device)
    descriptor = torch.stack([item[2] for item in samples]).to(device)

    model = build_model(base).to(device)
    distributed = DDP(
        model, device_ids=[local_rank], broadcast_buffers=False,
        find_unused_parameters=False,
        gradient_as_bucket_view=bool(config["distributed"]["gradient_as_bucket_view"]),
    )
    perceptual = LearnedPerceptualImagePatchSimilarity(
        net_type="alex", normalize=False
    ).to(device).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(distributed.parameters(), lr=1e-4, weight_decay=0.01)

    # Two unmeasured steps initialize CUDA/NCCL and DDP buckets.
    measurements = []
    total_steps = 10
    warmup_steps = 2
    started_all = time.perf_counter()
    for step in range(1, total_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        descent_steps = (step - 1) % 3 + 1
        full = step % 4 == 0
        generator = torch.Generator(device=device).manual_seed(
            int(base["model"]["initialization_seed"]) + rank * 1000 + step
        )
        scales = torch.rand(batch, generator=generator, device=device).add_(0.5)
        perturbation = torch.randn(
            degraded.shape, generator=generator, device=device, dtype=degraded.dtype
        ) * 0.025
        dist.barrier()
        started = time.perf_counter()
        loss, _ = objective(
            distributed, perceptual, base, degraded, clean, descriptor,
            steps=descent_steps, precision="float32", full=full,
            step_scale=scales, perturbation=perturbation,
        )
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"non-finite loss at step {step}")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(distributed.parameters(), 1.0)
        if not bool(torch.isfinite(norm)):
            raise RuntimeError(f"non-finite gradient at step {step}")
        optimizer.step()
        torch.cuda.synchronize(device)
        local_elapsed = time.perf_counter() - started
        elapsed_tensor = torch.tensor(local_elapsed, dtype=torch.float64, device=device)
        dist.all_reduce(elapsed_tensor, op=dist.ReduceOp.MAX)
        maximum_elapsed = float(elapsed_tensor.item())
        if step > warmup_steps:
            measurements.append({
                "step": step,
                "descent_steps": descent_steps,
                "full_objective": full,
                "seconds": maximum_elapsed,
                "global_images_per_second": batch * world / maximum_elapsed,
            })
        if rank == 0:
            print("VLM_ENERGY_MULTINODE_BENCHMARK_STEP " + json.dumps({
                "step": step, "total_steps": total_steps,
                "descent_steps": descent_steps, "full_objective": full,
                "seconds": maximum_elapsed,
                "global_images_per_second": batch * world / maximum_elapsed,
            }, sort_keys=True), flush=True)

    dist.barrier()
    if rank == 0:
        core = [item for item in measurements if not item["full_objective"]]
        mean_core = sum(float(item["seconds"]) for item in core) / len(core)
        mean_schedule = sum(float(item["seconds"]) for item in measurements) / len(measurements)
        train_records = int(base["data"]["train_records"])
        steps_per_epoch = math.ceil(train_records / (batch * world))
        core_hours = steps_per_epoch * int(config["schedule"]["joint_epochs"]) * mean_core / 3600
        schedule_hours = steps_per_epoch * int(config["schedule"]["joint_epochs"]) * mean_schedule / 3600
        result = {
            "schema_version": 1,
            "status": "foundir_vlm_energy_multinode_benchmark_pass",
            "run_id": run_id,
            "world_size": world,
            "nodes": host_counts,
            "gpu": identities[0]["gpu"],
            "gpu_family": identities[0]["gpu_family"],
            "gpu_families": sorted(families),
            "heterogeneous_gpu_families": len(families) > 1,
            "gpu_identities": identities,
            "batch_per_gpu": batch,
            "global_batch": batch * world,
            "measurements": measurements,
            "mean_core_seconds_per_step": mean_core,
            "mean_measured_schedule_seconds_per_step": mean_schedule,
            "steps_per_90000_image_epoch": steps_per_epoch,
            "estimated_20_joint_epoch_wall_hours_core_only": core_hours,
            "estimated_20_joint_epoch_gpu_hours_core_only": core_hours * world,
            "estimated_20_joint_epoch_wall_hours_measured_schedule": schedule_hours,
            "estimated_20_joint_epoch_gpu_hours_measured_schedule": schedule_hours * world,
            "elapsed_benchmark_seconds": time.perf_counter() - started_all,
            "train_config_sha256": sha256(TRAIN_CONFIG),
            "parent_test_accessed": False,
            "official_foundir_test_accessed": False,
            "sealed_internal_accessed": False,
        }
        output = ROOT / "checkpoints/foundir_vlm_energy_intersection_v1/multinode_benchmark" / run_id / "result.json"
        write_json_atomic(output, result)
        print("VLM_ENERGY_MULTINODE_BENCHMARK_RESULT " + json.dumps(result, sort_keys=True), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
