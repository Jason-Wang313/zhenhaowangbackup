#!/usr/bin/env python3
"""Three-node NCCL correctness and bandwidth smoke without model or dataset access."""

from __future__ import annotations

import json
import os
import platform
import socket
import time
from datetime import timedelta

import torch
import torch.distributed as dist


EXPECTED_NODES = 3
EXPECTED_LOCAL_WORLD = 8
EXPECTED_WORLD = EXPECTED_NODES * EXPECTED_LOCAL_WORLD
SIZES_MIB = (16, 64, 256)
WARMUP = 3
ITERATIONS = 10


def maximum_seconds(local_seconds: float, device: torch.device) -> float:
    value = torch.tensor([local_seconds], dtype=torch.float64, device=device)
    dist.all_reduce(value, op=dist.ReduceOp.MAX)
    return float(value.item())


def main() -> None:
    required = ("RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE")
    if any(name not in os.environ for name in required):
        raise RuntimeError("launch with torchrun")
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    local_world = int(os.environ["LOCAL_WORLD_SIZE"])
    if world != EXPECTED_WORLD or local_world != EXPECTED_LOCAL_WORLD:
        raise RuntimeError(
            f"expected 3x8=24 ranks, got world={world} local_world={local_world}"
        )
    if not torch.cuda.is_available() or torch.cuda.device_count() != EXPECTED_LOCAL_WORLD:
        raise RuntimeError("each node must expose exactly eight CUDA GPUs")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", timeout=timedelta(minutes=5), device_id=device)
    try:
        identity = {
            "rank": rank,
            "hostname": socket.gethostname(),
            "fqdn": socket.getfqdn(),
            "gpu": torch.cuda.get_device_name(device),
            "cuda": torch.version.cuda,
            "pytorch": torch.__version__,
            "python": platform.python_version(),
        }
        identities: list[dict | None] = [None] * world
        dist.all_gather_object(identities, identity)
        host_counts: dict[str, int] = {}
        for item in identities:
            assert item is not None
            host_counts[item["hostname"]] = host_counts.get(item["hostname"], 0) + 1
        if len(host_counts) != EXPECTED_NODES or set(host_counts.values()) != {8}:
            raise RuntimeError(f"expected three distinct 8-rank hosts, got {host_counts}")
        if len({item["gpu"] for item in identities if item is not None}) != 1:
            raise RuntimeError("mixed GPU models are forbidden in this smoke")

        correctness = torch.tensor([rank + 1.0], dtype=torch.float64, device=device)
        dist.all_reduce(correctness, op=dist.ReduceOp.SUM)
        expected_sum = world * (world + 1) / 2
        if float(correctness.item()) != expected_sum:
            raise RuntimeError(
                f"all-reduce correctness failed: {correctness.item()} != {expected_sum}"
            )

        measurements = []
        for size_mib in SIZES_MIB:
            elements = size_mib * 1024 * 1024 // 4
            value = torch.ones(elements, dtype=torch.float32, device=device)
            for _ in range(WARMUP):
                dist.all_reduce(value)
            torch.cuda.synchronize(device)
            dist.barrier()
            started = time.perf_counter()
            for _ in range(ITERATIONS):
                dist.all_reduce(value)
            torch.cuda.synchronize(device)
            elapsed = maximum_seconds(time.perf_counter() - started, device)
            seconds_per_iteration = elapsed / ITERATIONS
            algorithm_gbps = (size_mib / 1024) / seconds_per_iteration
            bus_gbps = algorithm_gbps * (2 * (world - 1) / world)
            measurements.append(
                {
                    "size_mib": size_mib,
                    "iterations": ITERATIONS,
                    "seconds_per_all_reduce": seconds_per_iteration,
                    "algorithm_bandwidth_GiB_s": algorithm_gbps,
                    "estimated_bus_bandwidth_GiB_s": bus_gbps,
                }
            )
            del value
            torch.cuda.empty_cache()

        payload = {
            "schema_version": 1,
            "status": "pass",
            "world_size": world,
            "nodes": host_counts,
            "gpu": identities[0]["gpu"],
            "backend": dist.get_backend(),
            "nccl_ib_disable": os.environ.get("NCCL_IB_DISABLE"),
            "nccl_socket_ifname": os.environ.get("NCCL_SOCKET_IFNAME"),
            "master_addr": os.environ.get("MASTER_ADDR"),
            "master_port": os.environ.get("MASTER_PORT"),
            "all_reduce_sum": float(correctness.item()),
            "measurements": measurements,
            "model_or_dataset_accessed": False,
        }
        if rank == 0:
            print("MULTINODE_NCCL_SMOKE_RESULT " + json.dumps(payload, sort_keys=True), flush=True)
        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
