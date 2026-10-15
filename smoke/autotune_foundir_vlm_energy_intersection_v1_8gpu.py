#!/usr/bin/env python3
"""Profile and optimizer-smoke the intersection model in one 8-GPU run."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import socket
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lowlight_restoration.vlm_energy_objectives import (  # noqa: E402
    edge_reconstruction,
    trajectory_calibrated_objectives,
)
from lowlight_restoration.vlm_energy_restoration import DEGRADATIONS  # noqa: E402
from smoke.preflight_foundir_vlm_energy_intersection_v1_8gpu import build_model, load_jsonl, sha256  # noqa: E402


AUTOTUNE_PATH = ROOT / "configs/foundir_vlm_energy_intersection_v1_autotune.json"


def write_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_bytes(encoded)
    os.replace(temporary, path)


def load_crop(
    row: dict[str, Any], target: dict[str, Any], crop: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    with Image.open(ROOT / str(row["output_path"])) as handle:
        degraded = np.asarray(handle.convert("RGB"), dtype=np.uint8)
    with Image.open(ROOT / str(target["target_path"])) as handle:
        clean = np.asarray(handle.convert("RGB"), dtype=np.uint8)
    rng = random.Random(seed)
    y = rng.randrange(degraded.shape[0] - crop + 1)
    x = rng.randrange(degraded.shape[1] - crop + 1)
    degraded = np.ascontiguousarray(degraded[y:y+crop, x:x+crop])
    clean = np.ascontiguousarray(clean[y:y+crop, x:x+crop])
    present = set(map(str, row["degradations"]))
    descriptor = torch.tensor([float(name in present) for name in DEGRADATIONS])
    return (
        torch.from_numpy(degraded).permute(2, 0, 1).float().div_(127.5).sub_(1),
        torch.from_numpy(clean).permute(2, 0, 1).float().div_(127.5).sub_(1),
        descriptor,
    )


def objective(
    distributed: DDP,
    perceptual: LearnedPerceptualImagePatchSimilarity,
    base: dict[str, Any],
    degraded: torch.Tensor,
    clean: torch.Tensor,
    descriptor: torch.Tensor,
    *,
    steps: int,
    precision: str,
    full: bool,
    step_scale: torch.Tensor | float = 1.0,
    perturbation: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    enabled = precision == "bfloat16"
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=enabled):
        trajectory = distributed(
            degraded,
            descriptor,
            steps=steps,
            create_graph=True,
            step_scale=step_scale,
            initial_perturbation=perturbation,
        )
        terms = trajectory_calibrated_objectives(
            distributed.module,
            degraded,
            clean,
            descriptor,
            trajectory,
            ranking_margin_scale=float(base["landscape_training"]["ranking_margin_scale"]),
            compute_quality_ranking=full,
        )
        weights = base["loss"]
        value = (
            float(weights["multi_scale_reconstruction"]) * terms.reconstruction
            + float(weights["edge_reconstruction"]) * edge_reconstruction(trajectory.output, clean)
            + float(weights["vector_direction"]) * terms.vector_direction
            + float(weights["vector_progress"]) * terms.vector_progress
            + float(weights["quality_ranking"]) * terms.quality_ranking
        )
    lpips_value = value.new_zeros(())
    if full:
        # Run the frozen perceptual network in fp32; its input gradient remains
        # connected to the higher-order restoration trajectory.
        with torch.autocast(device_type="cuda", enabled=False):
            lpips_value = perceptual(trajectory.output.float(), clean.float())
        value = value + float(base["loss"]["lpips_alex"]) * lpips_value
    return value, {
        "reconstruction": float(terms.reconstruction.detach()),
        "vector_direction": float(terms.vector_direction.detach()),
        "vector_progress": float(terms.vector_progress.detach()),
        "quality_ranking": float(terms.quality_ranking.detach()),
        "lpips": float(lpips_value.detach()),
    }


def main() -> None:
    dist.init_process_group("nccl")
    rank, world, local_rank = dist.get_rank(), dist.get_world_size(), int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    tune = json.loads(AUTOTUNE_PATH.read_text())
    base_path = ROOT / tune["base_config"]
    if sha256(base_path) != tune["base_config_sha256"]:
        raise RuntimeError("base config changed")
    if sha256(ROOT / "lowlight_restoration/vlm_energy_restoration.py") != tune["implementation_sha256"]:
        raise RuntimeError("model implementation changed")
    if sha256(ROOT / "lowlight_restoration/vlm_energy_objectives.py") != tune["objectives_sha256"]:
        raise RuntimeError("objective implementation changed")
    base = json.loads(base_path.read_text())
    if world != int(tune["required_visible_gpus"]):
        raise RuntimeError(f"requires exactly 8 ranks, got {world}")
    gpu = torch.cuda.get_device_name(local_rank)
    if not any(name in gpu for name in ("H100", "H200")):
        raise RuntimeError(f"unsupported GPU: {gpu}")
    alexnet = ROOT / base["loss"]["alexnet_weights"]
    if sha256(alexnet) != base["loss"]["alexnet_weights_sha256"]:
        raise RuntimeError("offline LPIPS cache changed")

    records = [row for row in load_jsonl(ROOT / base["data"]["manifest"]) if row["split"] == "train"]
    targets = {str(row["record_id"]): row for row in load_jsonl(ROOT / base["data"]["target_manifest"])}
    grouped = {size: [row for row in records if int(row["composition_size"]) == size] for size in (1, 2, 3)}
    maximum_batch = max(map(int, tune["profile"]["batch_candidates"]))
    samples = []
    for index in range(maximum_batch):
        order = (rank + index) % 3 + 1
        row = grouped[order][rank * maximum_batch + index]
        samples.append(load_crop(
            row, targets[str(row["record_id"])], int(base["data"]["crop_size"]),
            int(base["model"]["initialization_seed"]) + rank * 100 + index,
        ))
    degraded_all = torch.stack([item[0] for item in samples]).to(device)
    clean_all = torch.stack([item[1] for item in samples]).to(device)
    descriptor_all = torch.stack([item[2] for item in samples]).to(device)

    model = build_model(base).to(device)
    distributed = DDP(
        model,
        device_ids=[local_rank],
        broadcast_buffers=False,
        find_unused_parameters=False,
        gradient_as_bucket_view=True,
        static_graph=False,
    )
    perceptual = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=False).to(device).eval()
    perceptual.requires_grad_(False)

    def run_case(name: str, batch: int, steps: int, precision: str, full: bool, repeats: int) -> dict[str, Any]:
        local: dict[str, Any] = {"name": name, "batch_per_gpu": batch, "steps": steps, "precision": precision, "full_objective": full}
        try:
            elapsed = 0.0
            loss_number = gradient_number = math.nan
            terms: dict[str, float] = {}
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            for repeat in range(repeats):
                distributed.zero_grad(set_to_none=True)
                dist.barrier()
                started = time.perf_counter()
                loss, terms = objective(
                    distributed, perceptual, base,
                    degraded_all[:batch], clean_all[:batch], descriptor_all[:batch],
                    steps=steps, precision=precision, full=full,
                )
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError("non-finite loss")
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(distributed.parameters(), 1.0)
                if not bool(torch.isfinite(norm)):
                    raise RuntimeError("non-finite gradient")
                torch.cuda.synchronize(device)
                elapsed = time.perf_counter() - started
                loss_number, gradient_number = float(loss.detach()), float(norm)
            local.update({
                "status": "pass",
                "elapsed_seconds": elapsed,
                "global_images_per_second": batch * world / elapsed,
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                "loss": loss_number,
                "gradient_norm": gradient_number,
                "terms": terms,
            })
        except torch.OutOfMemoryError as error:
            local.update({"status": "oom", "error": str(error)})
            distributed.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
        except RuntimeError as error:
            local.update({"status": "fail", "error": str(error)})
            distributed.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
        gathered: list[dict[str, Any] | None] = [None] * world if rank == 0 else []
        dist.gather_object(local, gathered if rank == 0 else None, dst=0)
        if rank == 0:
            passed = [row for row in gathered if row and row["status"] == "pass"]
            if len(passed) == world:
                merged = dict(local)
                merged["elapsed_seconds"] = max(float(row["elapsed_seconds"]) for row in passed)
                merged["global_images_per_second"] = batch * world / merged["elapsed_seconds"]
                merged["peak_allocated_bytes"] = max(int(row["peak_allocated_bytes"]) for row in passed)
                merged["loss_range"] = [min(float(row["loss"]) for row in passed), max(float(row["loss"]) for row in passed)]
                merged["gradient_norm_range"] = [min(float(row["gradient_norm"]) for row in passed), max(float(row["gradient_norm"]) for row in passed)]
            else:
                merged = {"name": name, "batch_per_gpu": batch, "steps": steps, "precision": precision, "full_objective": full, "status": "fail", "rank_results": gathered}
            print("VLM_ENERGY_AUTOTUNE_CASE " + json.dumps(merged, sort_keys=True), flush=True)
        else:
            merged = {}
        objects = [merged]
        dist.broadcast_object_list(objects, src=0)
        return objects[0]

    cases = []
    repeats = int(tune["profile"]["repeats"])
    # Precision validity is tested with the complete objective.
    for precision in tune["profile"]["precision_candidates"]:
        cases.append(run_case(f"precision_{precision}", 1, 2, precision, True, repeats))
    valid_precisions = [row for row in cases if row["name"].startswith("precision_") and row["status"] == "pass"]
    if not valid_precisions:
        raise RuntimeError("no precision candidate passed the full objective")
    selected_precision = str(max(valid_precisions, key=lambda row: float(row["global_images_per_second"]))["precision"])
    # Batch scaling uses the core objective that runs on most optimizer steps.
    for batch in tune["profile"]["batch_candidates"]:
        cases.append(run_case(f"batch_{batch}", int(batch), 2, selected_precision, False, repeats))
    safe = [
        row for row in cases
        if row["name"].startswith("batch_") and row["status"] == "pass"
        and int(row["peak_allocated_bytes"]) * 1.7 <= int(tune["profile"]["maximum_projected_bytes_for_three_steps"])
    ]
    if not safe:
        raise RuntimeError("no batch candidate satisfies the three-step memory guard")
    selected_batch = int(max(safe, key=lambda row: float(row["global_images_per_second"]))["batch_per_gpu"])
    # Validate the chosen batch at the expensive endpoints. Back off on failure.
    while selected_batch >= 1:
        three = run_case("selected_three_step", selected_batch, 3, selected_precision, False, 1)
        full = run_case("selected_full_objective", selected_batch, 2, selected_precision, True, 1)
        cases.extend((three, full))
        if three["status"] == "pass" and full["status"] == "pass":
            break
        selected_batch //= 2
    if selected_batch < 1:
        raise RuntimeError("selected profile failed endpoint validation")

    # A real bounded optimizer smoke uses the final selection and the declared
    # randomized landscape recipe. It saves a checkpoint only after all steps.
    settings = tune["optimizer_smoke"]
    optimizer = torch.optim.AdamW(
        distributed.parameters(), lr=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )
    smoke_losses = []
    smoke_started = time.perf_counter()
    initial_checksum = sum(float(parameter.detach().float().sum()) for parameter in distributed.module.parameters())
    for step in range(1, int(settings["steps"]) + 1):
        optimizer.zero_grad(set_to_none=True)
        descent_steps = (step - 1) % int(tune["profile"]["maximum_steps"]) + 1
        full = step % int(tune["profile"]["full_objective_interval"]) == 0
        generator = torch.Generator(device=device).manual_seed(int(base["model"]["initialization_seed"]) + rank * 1000 + step)
        low, high = map(float, settings["random_step_scale"])
        scales = torch.rand(selected_batch, generator=generator, device=device).mul_(high - low).add_(low)
        sigma_low, sigma_high = map(float, settings["langevin_sigma"])
        sigma = float(torch.rand((), generator=generator, device=device).mul_(sigma_high - sigma_low).add_(sigma_low))
        perturbation = torch.randn(
            degraded_all[:selected_batch].shape,
            generator=generator,
            device=device,
            dtype=degraded_all.dtype,
        ) * sigma
        loss, terms = objective(
            distributed, perceptual, base,
            degraded_all[:selected_batch], clean_all[:selected_batch], descriptor_all[:selected_batch],
            steps=descent_steps, precision=selected_precision, full=full,
            step_scale=scales, perturbation=perturbation,
        )
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"non-finite optimizer loss at step {step}")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(distributed.parameters(), float(settings["gradient_clip_norm"]))
        if not bool(torch.isfinite(norm)):
            raise RuntimeError(f"non-finite optimizer gradient at step {step}")
        optimizer.step()
        smoke_losses.append(float(loss.detach()))
        if rank == 0:
            print("VLM_ENERGY_OPTIMIZER_SMOKE " + json.dumps({
                "step": step, "steps": int(settings["steps"]), "descent_steps": descent_steps,
                "full_objective": full, "loss": smoke_losses[-1], "gradient_norm": float(norm),
                "batch_per_gpu": selected_batch, "precision": selected_precision,
                "elapsed_seconds": time.perf_counter() - smoke_started,
            }, sort_keys=True), flush=True)
    dist.barrier()
    final_checksum = sum(float(parameter.detach().float().sum()) for parameter in distributed.module.parameters())
    changed = not math.isclose(initial_checksum, final_checksum, rel_tol=0.0, abs_tol=1e-8)
    changed_tensor = torch.tensor(int(changed), device=device)
    dist.all_reduce(changed_tensor, op=dist.ReduceOp.MIN)
    if not int(changed_tensor):
        raise RuntimeError("optimizer smoke did not update parameters on every rank")

    if rank == 0:
        output_root = ROOT / tune["runtime"]["output_root"]
        output_root.mkdir(parents=True, exist_ok=True)
        checkpoint = output_root / "optimizer_smoke_step_000012.pt"
        temporary = checkpoint.with_name(checkpoint.name + f".tmp.{os.getpid()}")
        torch.save({
            "schema_version": 1,
            "role": "optimizer_smoke_only_not_a_model_candidate",
            "autotune_config_sha256": sha256(AUTOTUNE_PATH),
            "optimizer_steps": int(settings["steps"]),
            "model_state_dict": distributed.module.state_dict(),
        }, temporary)
        os.replace(temporary, checkpoint)
        successful_batches = [row for row in cases if row["name"].startswith("batch_") and row["status"] == "pass"]
        selected_core = next(row for row in successful_batches if int(row["batch_per_gpu"]) == selected_batch)
        seconds_per_step = float(selected_core["elapsed_seconds"])
        train_records = int(base["data"]["train_records"])
        global_batch = selected_batch * world
        steps_per_epoch = math.ceil(train_records / global_batch)
        result = {
            "schema_version": 1,
            "status": "foundir_vlm_energy_intersection_v1_autotune_pass",
            "autotune_config_sha256": sha256(AUTOTUNE_PATH),
            "host": socket.gethostname(),
            "gpu": gpu,
            "world_size": world,
            "cases": cases,
            "selection": {
                "precision": selected_precision,
                "batch_per_gpu": selected_batch,
                "global_batch": global_batch,
                "maximum_training_descent_steps": int(tune["profile"]["maximum_steps"]),
                "full_objective_interval": int(tune["profile"]["full_objective_interval"]),
                "measured_core_seconds_per_step": seconds_per_step,
                "steps_per_90000_image_epoch": steps_per_epoch,
                "estimated_20_epoch_wall_hours_core_only": steps_per_epoch * 20 * seconds_per_step / 3600,
                "estimated_20_epoch_gpu_hours_core_only": steps_per_epoch * 20 * seconds_per_step / 3600 * world,
            },
            "optimizer_smoke": {
                "steps": int(settings["steps"]),
                "losses": smoke_losses,
                "elapsed_seconds": time.perf_counter() - smoke_started,
                "parameters_changed": True,
                "checkpoint": checkpoint.relative_to(ROOT).as_posix(),
                "checkpoint_sha256": sha256(checkpoint),
                "checkpoint_is_model_candidate": False,
            },
            "parent_test_accessed": False,
            "official_foundir_test_accessed": False,
            "sealed_internal_accessed": False,
        }
        write_atomic(output_root / "result.json", result)
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
