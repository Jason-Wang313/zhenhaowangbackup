#!/usr/bin/env python3
"""Exercise every selected GPU and NCCL collective before long FoundIR work."""

from __future__ import annotations

import datetime
import json
import os

import torch
import torch.distributed as dist


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(
        backend="nccl",
        timeout=datetime.timedelta(minutes=5),
        device_id=device,
    )
    try:
        rank = dist.get_rank()
        world = dist.get_world_size()
        if world != 8 or torch.cuda.device_count() != 8:
            raise RuntimeError(f"distributed canary requires exactly eight GPUs: {world}")
        properties = torch.cuda.get_device_properties(device)
        if not any(name in properties.name for name in ("H100", "H200")):
            raise RuntimeError(f"unsupported GPU in distributed canary: {properties.name}")

        reduced = torch.tensor([rank + 1.0], device=device)
        dist.all_reduce(reduced)
        if reduced.item() != 36.0:
            raise RuntimeError(f"NCCL all-reduce changed: {reduced.item()}")

        source = torch.randn((256, 256), device=device, dtype=torch.bfloat16, requires_grad=True)
        loss = (source @ source.T).float().square().mean()
        loss.backward()
        if source.grad is None or not torch.isfinite(source.grad).all():
            raise RuntimeError("GPU forward/backward canary produced invalid gradients")

        objects: list[object | None] = [None] * world
        dist.all_gather_object(objects, {"rank": rank, "gpu": properties.name})
        if sorted(int(item["rank"]) for item in objects if isinstance(item, dict)) != list(range(world)):
            raise RuntimeError(f"NCCL object collective is incomplete: {objects}")
        dist.barrier()
        if rank == 0:
            print(
                "FOUNDIR_FIVE_TYPE_DISTRIBUTED_CANARY_PASS "
                + json.dumps({"world_size": world, "inventory": objects}, sort_keys=True),
                flush=True,
            )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
