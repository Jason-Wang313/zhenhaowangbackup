#!/usr/bin/env python3
"""No-update eight-rank NCCL and DDP sanity gate for CURE launchers."""

from __future__ import annotations

import os

import torch
from torch import distributed as dist
from torch import nn


def main() -> None:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    if world != 8 or not torch.cuda.is_available():
        raise RuntimeError("eight CUDA ranks required")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend="nccl", device_id=device)
    value = torch.tensor(float(rank + 1), device=device)
    dist.all_reduce(value)
    if value.item() != 36:
        raise RuntimeError(f"wrong NCCL all-reduce: {value.item()}")
    model = nn.parallel.DistributedDataParallel(nn.Linear(4, 4).to(device), device_ids=[local_rank])
    output = model(torch.ones(2, 4, device=device)).square().mean()
    output.backward()
    if not all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters()):
        raise RuntimeError("DDP gradient missing or nonfinite")
    dist.barrier()
    if rank == 0:
        print("CURE_NCCL_DDP_REHEARSAL_PASS world=8 optimizer_steps=0", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
