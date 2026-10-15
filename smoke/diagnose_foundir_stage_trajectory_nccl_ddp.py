#!/usr/bin/env python3
"""No-update NCCL/DDP rehearsal for the stage-trajectory adapter."""

from __future__ import annotations

import json
import os
import sys
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lowlight_restoration.stage_trajectory_compatibility import StageTrajectoryCompatibilityAdapter


RESULT = ROOT / "checkpoints/foundir_stage_trajectory_compatibility_v1/nccl_rehearsal.json"


def main() -> None:
    if os.environ.get("NCCL_NVLS_ENABLE") != "0" or os.environ.get("NCCL_P2P_DISABLE") != "0":
        raise RuntimeError("frozen NCCL recovery environment is missing")
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if world != 8:
        raise RuntimeError("NCCL rehearsal requires eight ranks")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", timeout=timedelta(minutes=5), device_id=device)
    scalar = torch.tensor(float(rank + 1), device=device)
    dist.all_reduce(scalar, op=dist.ReduceOp.SUM)
    if float(scalar.item()) != 36.0:
        raise RuntimeError("scalar NCCL all-reduce mismatch")

    torch.manual_seed(2026092703 + rank)
    torch.cuda.manual_seed(2026092703 + rank)
    bare = StageTrajectoryCompatibilityAdapter().to(device)
    model = DistributedDataParallel(
        bare,
        device_ids=[local_rank],
        output_device=local_rank,
        gradient_as_bucket_view=True,
    )
    states = [torch.rand(1, 3, 64, 64, device=device) for _ in range(4)]
    presence = torch.tensor([[1.0, 1.0, 1.0]], device=device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = model(*states, presence)
        loss = output.float().mean()
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters()]
    if any(gradient is None or not torch.isfinite(gradient).all() for gradient in gradients):
        raise RuntimeError("missing or nonfinite adapter gradient")
    gradient_sum = torch.stack([gradient.float().abs().sum() for gradient in gradients]).sum()
    dist.all_reduce(gradient_sum, op=dist.ReduceOp.SUM)
    if float(gradient_sum.item()) <= 0.0:
        raise RuntimeError("zero adapter gradient across all ranks")
    names: list[str | None] = [None] * world
    dist.all_gather_object(names, torch.cuda.get_device_name(device))
    dist.barrier()
    if rank == 0:
        if RESULT.exists():
            raise FileExistsError(f"refusing to overwrite {RESULT}")
        RESULT.parent.mkdir(parents=True, exist_ok=True)
        RESULT.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "status": "stage_trajectory_nccl_ddp_rehearsal_pass",
                    "world_size": world,
                    "gpu_names": names,
                    "scalar_all_reduce": float(scalar.item()),
                    "finite_full_model_gradients": True,
                    "gradient_absolute_sum_all_ranks": float(gradient_sum.item()),
                    "optimizer_steps": 0,
                    "model_weight_updates": False,
                    "nccl_nvls_enable": os.environ["NCCL_NVLS_ENABLE"],
                    "nccl_p2p_disable": os.environ["NCCL_P2P_DISABLE"],
                    "official_foundir_test_accessed": False,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        print("STAGE_TRAJECTORY_NCCL_DDP_REHEARSAL_PASS", flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
