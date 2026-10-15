#!/usr/bin/env python3
"""Train and select a development-only condition-routed Blur pixel expert."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, Sampler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lowlight_restoration.deblur_expert import DeblurPixelExpert, deblur_expert_parameter_count  # noqa: E402
from lowlight_restoration.flux_prompt_cache import file_sha256  # noqa: E402
from lowlight_restoration.foundir_flux2_protocol import resolve_workspace_path, stable_digest  # noqa: E402
from lowlight_restoration.pixel_refinement import spatial_highpass  # noqa: E402
from smoke.evaluate_foundir_five_type_flux2_ddp import (  # noqa: E402
    load_aligned_crop,
    selected_from_contract,
    torchrun_worker_identity,
    write_once,
)
from smoke.train_foundir_five_type_flux2_full_ddp import load_rows, prepare_aligned_pair  # noqa: E402


def validate_config(config: Mapping[str, object], config_path: Path) -> tuple[dict, Path]:
    if (
        config.get("schema_version") != 1
        or config.get("status") != "development_only_blur_capacity_gate"
        or config.get("checkpoint_selection_uses_validation_only") is not True
        or config.get("sealed_internal_test_accessed") is not False
        or config.get("protected_foundir_test_accessed") is not False
    ):
        raise RuntimeError("Blur gate contract is invalid")
    parent_path = resolve_workspace_path(ROOT, config["parent_config"])
    conditions_path = resolve_workspace_path(ROOT, config["conditions"])
    if file_sha256(parent_path) != config.get("parent_config_sha256"):
        raise RuntimeError("Blur gate parent config changed")
    if file_sha256(conditions_path) != config.get("conditions_sha256"):
        raise RuntimeError("Blur gate conditions changed")
    training = config["training"]
    if (
        int(training["world_size"]) != 8
        or int(training["global_batch_size"])
        != 8 * int(training["batch_size_per_rank"]) * int(training["gradient_accumulation_steps"])
    ):
        raise RuntimeError("Blur gate global batch contract changed")
    checkpoints = [int(step) for step in training["checkpoint_steps"]]
    if checkpoints != sorted(set(checkpoints)) or checkpoints[-1] != int(training["optimizer_steps"]):
        raise RuntimeError("Blur gate checkpoint schedule is invalid")
    return json.loads(parent_path.read_text()), conditions_path


def model_from_config(config: Mapping[str, object]) -> DeblurPixelExpert:
    model = config["model"]
    return DeblurPixelExpert(
        width=int(model["width"]),
        encoder_blocks=model["encoder_blocks"],
        middle_blocks=int(model["middle_blocks"]),
        decoder_blocks=model["decoder_blocks"],
        residual_bound=float(model["residual_bound"]),
    )


class BlurPresentationDataset(Dataset[dict[str, object]]):
    def __init__(
        self,
        blur_rows: Sequence[Mapping[str, str]],
        noise_rows: Sequence[Mapping[str, str]],
        config: Mapping[str, object],
    ) -> None:
        self.blur_rows = tuple(blur_rows)
        self.noise_rows = tuple(noise_rows)
        self.config = config

    def __len__(self) -> int:
        return 2**31 - 1

    def __getitem__(self, presentation: int) -> dict[str, object]:
        data = self.config["data"]
        seed = int(data["seed"])
        noise_identity = (
            int(stable_digest(seed, "noise-identity", presentation)[:16], 16) / 2**64
            < float(data["noise_identity_presentation_probability"])
        )
        pool = self.noise_rows if noise_identity else self.blur_rows
        record_index = int(stable_digest(seed, "training-record", presentation)[:16], 16) % len(pool)
        row = pool[record_index]
        with Image.open(resolve_workspace_path(ROOT, row["lq_path"])) as handle:
            source = np.asarray(handle.convert("RGB"), dtype=np.uint8).copy()
        with Image.open(resolve_workspace_path(ROOT, row["gt_path"])) as handle:
            target = np.asarray(handle.convert("RGB"), dtype=np.uint8).copy()
        crop = int(data["crop_size"])
        source, target = prepare_aligned_pair(source, target, crop, row["record_id"])
        if noise_identity:
            target = source.copy()
        digest = bytes.fromhex(stable_digest(seed, "blur-crop", presentation, row["record_id"]))
        y = int.from_bytes(digest[:8], "big") % (source.shape[0] - crop + 1)
        x = int.from_bytes(digest[8:16], "big") % (source.shape[1] - crop + 1)
        source = source[y : y + crop, x : x + crop]
        target = target[y : y + crop, x : x + crop]
        if data["d4_augmentation"]:
            transform = int.from_bytes(digest[16:24], "big") % 8
            turns, reflect = transform % 4, transform // 4
            source = np.rot90(source, turns)
            target = np.rot90(target, turns)
            if reflect:
                source = source[:, ::-1]
                target = target[:, ::-1]
        identity = noise_identity or (
            int(stable_digest(seed, "identity", presentation)[:16], 16) / 2**64
            < float(data["identity_probability"])
        )
        if identity and not noise_identity:
            source = target.copy()

        def tensor(value: np.ndarray) -> torch.Tensor:
            return torch.from_numpy(np.ascontiguousarray(value.transpose(2, 0, 1))).float().div_(255.0)

        return {
            "source": tensor(source),
            "target": tensor(target),
            "identity": identity,
            "presentation": presentation,
        }


class RankPresentationSampler(Sampler[int]):
    def __init__(self, *, start: int, stop: int, rank: int, world: int) -> None:
        if start % world or stop % world or not 0 <= start <= stop:
            raise ValueError("rank presentation range is not world aligned")
        self.start, self.stop, self.rank, self.world = start, stop, rank, world

    def __len__(self) -> int:
        return (self.stop - self.start) // self.world

    def __iter__(self):
        return iter(range(self.start + self.rank, self.stop, self.world))


def collate(items: Sequence[Mapping[str, object]]) -> dict[str, object]:
    return {
        "source": torch.stack([item["source"] for item in items]),
        "target": torch.stack([item["target"] for item in items]),
        "identity": torch.tensor([item["identity"] for item in items], dtype=torch.bool),
    }


def learning_rate(step: int, config: Mapping[str, object]) -> float:
    training = config["training"]
    peak = float(training["learning_rate"])
    floor = float(training["minimum_learning_rate"])
    warmup = int(training["warmup_steps"])
    total = int(training["optimizer_steps"])
    if step <= warmup:
        return peak * step / warmup
    progress = (step - warmup) / (total - warmup)
    return floor + (peak - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))


def loss_terms(
    prediction: torch.Tensor,
    target: torch.Tensor,
    source: torch.Tensor,
    identity: torch.Tensor,
    config: Mapping[str, object],
    lpips_metric,
    *,
    step: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    contract = config["training"]["loss"]
    difference = prediction.float() - target.float()
    mse = difference.square().mean()
    charbonnier = torch.sqrt(difference.square() + float(contract["charbonnier_epsilon"]) ** 2).mean()
    edge = F.l1_loss(spatial_highpass(prediction.float()), spatial_highpass(target.float()))
    identity_l1 = (
        F.l1_loss(prediction[identity].float(), source[identity].float())
        if identity.any()
        else prediction.new_zeros((), dtype=torch.float32)
    )
    use_lpips = step % int(contract["lpips_every_steps"]) == 0
    perceptual = lpips_metric(prediction.float(), target.float()) if use_lpips else prediction.new_zeros((), dtype=torch.float32)
    loss = (
        mse * float(contract["rgb_mse_weight"])
        + charbonnier * float(contract["charbonnier_weight"])
        + edge * float(contract["edge_l1_weight"])
        + perceptual * float(contract["lpips_weight"])
        + identity_l1 * float(contract["identity_l1_weight"])
    )
    return loss, {
        "loss": float(loss.detach()),
        "mse": float(mse.detach()),
        "charbonnier": float(charbonnier.detach()),
        "edge_l1": float(edge.detach()),
        "lpips": float(perceptual.detach()),
        "identity_l1": float(identity_l1.detach()),
        "lpips_active": float(use_lpips),
    }


def save_checkpoint(
    path: Path,
    model: DeblurPixelExpert,
    optimizer: torch.optim.Optimizer,
    *,
    step: int,
    config_sha: str,
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": 1,
            "config_sha256": config_sha,
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "protected_foundir_test_accessed": False,
        },
        temporary,
    )
    os.replace(temporary, path)


def resume(
    output: Path,
    model: DeblurPixelExpert,
    optimizer: torch.optim.Optimizer,
    config_sha: str,
) -> int:
    candidates = sorted(output.glob("step_*.pt"))
    if not candidates:
        return 0
    path = candidates[-1]
    payload = torch.load(path, map_location="cpu", weights_only=False)
    step = int(path.stem.split("_", 1)[1])
    if (
        payload.get("schema_version") != 1
        or payload.get("config_sha256") != config_sha
        or payload.get("step") != step
        or payload.get("protected_foundir_test_accessed") is not False
    ):
        raise RuntimeError(f"invalid Blur gate resume checkpoint: {path}")
    model.load_state_dict(payload["model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    return step


def scalar_metrics(metric, reference: torch.Tensor, prediction: torch.Tensor) -> dict[str, float]:
    prediction = prediction.float().clamp(0, 1)
    reference = reference.float().clamp(0, 1)
    mse = float((prediction - reference).square().mean())
    psnr = 100.0 if mse <= 1e-12 else -10.0 * math.log10(mse)
    with torch.inference_mode():
        lpips = float(metric(prediction, reference).item())
        metric.reset()
    return {"psnr": psnr, "lpips": lpips}


def mean_metrics(rows: Sequence[Mapping[str, object]], category: str, path: str) -> dict[str, float]:
    selected = [row[path] for row in rows if row["category"] == category]
    if not selected:
        raise RuntimeError(f"missing validation rows for {category}/{path}")
    return {
        "records": float(len(selected)),
        "psnr": sum(float(row["psnr"]) for row in selected) / len(selected),
        "lpips": sum(float(row["lpips"]) for row in selected) / len(selected),
    }


def evaluate(
    model: DeblurPixelExpert,
    rows: Sequence[Mapping[str, str]],
    config: Mapping[str, object],
    parent: Mapping[str, object],
    metric,
    device: torch.device,
    rank: int,
    world: int,
) -> list[dict[str, object]]:
    model.eval()
    local = []
    for index in range(rank, len(rows), world):
        row = rows[index]
        lq, gt, _ = load_aligned_crop(
            row,
            crop_size=int(parent["data"]["crop_size"]),
            seed=int(parent["data"]["split_seed"]),
        )
        source = ((lq.unsqueeze(0).to(device) + 1.0) * 0.5).float()
        target = ((gt.unsqueeze(0).to(device) + 1.0) * 0.5).float()
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            expert = model(source).float()
        qwen_blur_active = int(json.loads(row["qwen_json"])["blur"]) > 1
        routed = expert if qwen_blur_active else source
        local.append(
            {
                "record_id": row["record_id"],
                "category": row["category"],
                "qwen_blur_active": qwen_blur_active,
                "input": scalar_metrics(metric, target, source),
                "oracle_expert": scalar_metrics(metric, target, expert),
                "qwen_routed": scalar_metrics(metric, target, routed),
            }
        )
    gathered: list[list[dict[str, object]] | None] = [None for _ in range(world)]
    dist.all_gather_object(gathered, local)
    model.train()
    return [row for shard in gathered if shard is not None for row in shard]


def evaluation_summary(rows: Sequence[Mapping[str, object]], *, step: int) -> dict[str, object]:
    return {
        "schema_version": 1,
        "status": "blur_gate_development_evaluation_complete",
        "step": step,
        "records": len(rows),
        "blur": {
            "input": mean_metrics(rows, "01Blur", "input"),
            "oracle_expert": mean_metrics(rows, "01Blur", "oracle_expert"),
            "qwen_routed": mean_metrics(rows, "01Blur", "qwen_routed"),
            "qwen_active_records": sum(row["qwen_blur_active"] for row in rows if row["category"] == "01Blur"),
        },
        "noise": {
            "input": mean_metrics(rows, "05Noise", "input"),
            "qwen_routed": mean_metrics(rows, "05Noise", "qwen_routed"),
            "qwen_false_blur_records": sum(row["qwen_blur_active"] for row in rows if row["category"] == "05Noise"),
        },
        "split": "validation",
        "sealed_internal_test_accessed": False,
        "protected_foundir_test_accessed": False,
    }


def selection_score(summary: Mapping[str, object], gate: Mapping[str, object]) -> float:
    blur = summary["blur"]["qwen_routed"]
    noise = summary["noise"]["qwen_routed"]
    return min(
        float(blur["psnr"]) - float(gate["blur_psnr_minimum"]),
        (float(gate["blur_lpips_maximum"]) - float(blur["lpips"])) / 0.01,
        float(noise["psnr"]) - float(gate["noise_psnr_minimum"]),
        (float(gate["noise_lpips_maximum"]) - float(noise["lpips"])) / 0.01,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config_path = resolve_workspace_path(ROOT, args.config)
    config = json.loads(config_path.read_text())
    config_sha = file_sha256(config_path)
    parent, conditions_path = validate_config(config, config_path)
    rank, world, local_rank = torchrun_worker_identity(required_world=int(config["training"]["world_size"]))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend="nccl")
    if not all("H100" in torch.cuda.get_device_properties(index).name or "H200" in torch.cuda.get_device_properties(index).name for index in range(8)):
        raise RuntimeError("Blur gate requires one homogeneous H100/H200 node")
    rows = load_rows(conditions_path, int(parent["data"]["expected_five_type_records"]))
    training_rows = [
        row
        for row in rows
        if row["split"] == config["data"]["training_split"]
        and row["category"] == config["data"]["training_category"]
    ]
    noise_identity_rows = [
        row
        for row in rows
        if row["split"] == config["data"]["training_split"]
        and row["category"] == config["data"]["noise_identity_category"]
    ]
    if not training_rows or not noise_identity_rows:
        raise RuntimeError("Blur gate training or Noise identity pool is empty")
    validation = selected_from_contract(
        parent,
        rows,
        split="validation",
        config_sha=file_sha256(resolve_workspace_path(ROOT, config["parent_config"])),
        conditions_sha=file_sha256(conditions_path),
    )
    validation = [row for row in validation if row["category"] in config["data"]["validation_categories"]]
    counts = defaultdict(int)
    for row in validation:
        counts[row["category"]] += 1
    if any(counts[category] != int(config["data"]["validation_records_per_category"]) for category in config["data"]["validation_categories"]):
        raise RuntimeError(f"Blur gate validation coverage changed: {dict(counts)}")

    torch.manual_seed(int(config["data"]["seed"]))
    model = model_from_config(config).to(device)
    wrapped = DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False)
    optimizer = torch.optim.AdamW(
        wrapped.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
        fused=True,
    )
    output = resolve_workspace_path(ROOT, config["output"])
    output.mkdir(parents=True, exist_ok=True)
    step = resume(output, wrapped.module, optimizer, config_sha)
    loaded_step = torch.tensor([step], device=device)
    dist.broadcast(loaded_step, src=0)
    if int(loaded_step.item()) != step:
        raise RuntimeError("Blur gate ranks resumed different steps")

    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

    lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(device).eval()
    lpips_metric.requires_grad_(False)
    dataset = BlurPresentationDataset(training_rows, noise_identity_rows, config)
    total_steps = int(config["training"]["optimizer_steps"])
    global_batch = int(config["training"]["global_batch_size"])
    sampler = RankPresentationSampler(
        start=step * global_batch,
        stop=total_steps * global_batch,
        rank=rank,
        world=world,
    )
    loader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=int(config["training"]["batch_size_per_rank"]),
        num_workers=int(config["training"]["num_workers_per_rank"]),
        pin_memory=True,
        persistent_workers=int(config["training"]["num_workers_per_rank"]) > 0,
        collate_fn=collate,
    )
    iterator = iter(loader)
    accumulation = int(config["training"]["gradient_accumulation_steps"])
    checkpoint_steps = set(int(value) for value in config["training"]["checkpoint_steps"])
    optimizer.zero_grad(set_to_none=True)
    while step < total_steps:
        started = time.perf_counter()
        totals = defaultdict(float)
        for micro in range(accumulation):
            batch = next(iterator)
            source = batch["source"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            identity = batch["identity"].to(device, non_blocking=True)
            sync = wrapped.no_sync() if micro + 1 < accumulation else torch.enable_grad()
            with sync:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    prediction = wrapped(source)
                loss, terms = loss_terms(
                    prediction,
                    target,
                    source,
                    identity,
                    config,
                    lpips_metric,
                    step=step + 1,
                )
                (loss / accumulation).backward()
            lpips_metric.reset()
            for key, value in terms.items():
                totals[key] += value / accumulation
        torch.nn.utils.clip_grad_norm_(wrapped.parameters(), float(config["training"]["max_grad_norm"]))
        step += 1
        lr = learning_rate(step, config)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if step % 25 == 0 or step in checkpoint_steps:
            values = torch.tensor([totals[key] for key in sorted(totals)], device=device)
            dist.all_reduce(values)
            values /= world
            if rank == 0:
                report = " ".join(f"{key}={float(value):.6f}" for key, value in zip(sorted(totals), values))
                print(
                    f"FOUNDIR_BLUR_GATE step={step}/{total_steps} lr={lr:.8f} "
                    f"seconds={time.perf_counter() - started:.3f} {report}",
                    flush=True,
                )
        if step in checkpoint_steps:
            if rank == 0:
                save_checkpoint(output / f"step_{step:06d}.pt", wrapped.module, optimizer, step=step, config_sha=config_sha)
            dist.barrier()
            result_path = output / f"development_step_{step:06d}.json"
            if not result_path.exists():
                result_rows = evaluate(wrapped.module, validation, config, parent, lpips_metric, device, rank, world)
                summary = evaluation_summary(result_rows, step=step)
                summary.update(
                    config_sha256=config_sha,
                    checkpoint=(output / f"step_{step:06d}.pt").relative_to(ROOT).as_posix(),
                    checkpoint_sha256=file_sha256(output / f"step_{step:06d}.pt"),
                    model_parameters=deblur_expert_parameter_count(wrapped.module),
                )
                if rank == 0:
                    write_once(result_path, (json.dumps(summary, indent=2, sort_keys=True) + "\n").encode())
                    print("FOUNDIR_BLUR_GATE_EVAL " + json.dumps(summary, sort_keys=True), flush=True)
            dist.barrier()

    if rank == 0:
        candidates = [
            json.loads((output / f"development_step_{candidate_step:06d}.json").read_text())
            for candidate_step in sorted(checkpoint_steps)
        ]
        winner = max(candidates, key=lambda item: (selection_score(item, config["gate"]), -int(item["step"])))
        blur = winner["blur"]["qwen_routed"]
        noise = winner["noise"]["qwen_routed"]
        checks = {
            "blur_psnr": float(blur["psnr"]) >= float(config["gate"]["blur_psnr_minimum"]),
            "blur_lpips": float(blur["lpips"]) <= float(config["gate"]["blur_lpips_maximum"]),
            "noise_psnr": float(noise["psnr"]) >= float(config["gate"]["noise_psnr_minimum"]),
            "noise_lpips": float(noise["lpips"]) <= float(config["gate"]["noise_lpips_maximum"]),
        }
        result = {
            "schema_version": 1,
            "status": "blur_pixel_expert_gate_complete",
            "config": config_path.relative_to(ROOT).as_posix(),
            "config_sha256": config_sha,
            "training_records": len(training_rows),
            "noise_identity_records": len(noise_identity_rows),
            "sample_presentations": total_steps * global_batch,
            "model_parameters": deblur_expert_parameter_count(wrapped.module),
            "selection_rule": config["gate"]["selection"],
            "selected_step": winner["step"],
            "checkpoint": winner["checkpoint"],
            "checkpoint_sha256": winner["checkpoint_sha256"],
            "selected_development_metrics": {"blur": winner["blur"], "noise": winner["noise"]},
            "checks": checks,
            "gate_pass": all(checks.values()),
            "sealed_internal_test_accessed": False,
            "protected_foundir_test_accessed": False,
        }
        write_once(output / "result.json", (json.dumps(result, indent=2, sort_keys=True) + "\n").encode())
        print("FOUNDIR_BLUR_PIXEL_EXPERT_GATE_PASS " + json.dumps(result, sort_keys=True), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
