#!/usr/bin/env python3
"""No-data, no-weight-update DDP recovery check for the Low-light+Haze run."""

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

from lowlight_restoration.task_native_restormer import load_task_native_restormer  # noqa: E402
from smoke.evaluate_foundir_five_type_flux2_ddp import write_once  # noqa: E402


def main() -> None:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    if world != 8 or int(os.environ.get("LOCAL_WORLD_SIZE", -1)) != 8:
        raise RuntimeError("NCCL recovery diagnostic requires one 8-GPU node")
    if (
        torch.cuda.device_count() != 8
        or os.environ.get("NCCL_P2P_DISABLE") != "0"
        or os.environ.get("NCCL_NVLS_ENABLE") != "0"
    ):
        raise RuntimeError(
            "NCCL recovery diagnostic requires eight GPUs, P2P enabled, and NVLS disabled"
        )
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if "H200" not in torch.cuda.get_device_properties(device).name:
        raise RuntimeError("this bounded recovery diagnostic is authorized only for H200")
    dist.init_process_group("nccl", timeout=timedelta(minutes=5), device_id=device)
    try:
        scalar = torch.tensor(float(rank + 1), device=device)
        dist.all_reduce(scalar, op=dist.ReduceOp.SUM)
        expected = float(world * (world + 1) // 2)
        if float(scalar.item()) != expected:
            raise RuntimeError(
                f"scalar NCCL all-reduce mismatch: {float(scalar.item())} != {expected}"
            )
        dist.barrier()

        model, metadata = load_task_native_restormer(
            "deraining", ROOT / "models/restormer-task-native"
        )
        model = model.to(device)
        distributed = DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False
        )
        source = torch.full((1, 3, 64, 64), 0.25 + rank / 100.0, device=device)
        target = torch.full_like(source, 0.5)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = torch.nn.functional.l1_loss(distributed(source), target)
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(distributed.parameters(), 1.0)
        values = torch.tensor([float(loss.item()), float(gradient.item())], device=device)
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
        values /= world
        if not bool(torch.isfinite(values).all().item()):
            raise RuntimeError("NCCL recovery diagnostic produced non-finite values")
        if rank == 0:
            payload = {
                "schema_version": 1,
                "status": "pass",
                "world_size": world,
                "gpu": torch.cuda.get_device_properties(device).name,
                "nccl_p2p_disable": os.environ["NCCL_P2P_DISABLE"],
                "nccl_nvls_enable": os.environ["NCCL_NVLS_ENABLE"],
                "explicit_process_group_device_binding": True,
                "scalar_collective": "pass",
                "ddp_parameter_verification": "pass",
                "forward_backward_gradient_allreduce": "pass",
                "mean_loss": float(values[0].item()),
                "mean_gradient_norm": float(values[1].item()),
                "model": metadata,
                "optimizer_step_performed": False,
                "model_or_dataset_pixels_accessed": False,
                "official_foundir_test_accessed": False,
            }
            path = ROOT / "checkpoints/foundir_lowlight_haze_restormer_v1/nccl_recovery/result.json"
            write_once(path, (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode())
            print("FOUNDIR_LOWLIGHT_HAZE_NCCL_RECOVERY_PASS " + json.dumps(payload, sort_keys=True), flush=True)
        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
