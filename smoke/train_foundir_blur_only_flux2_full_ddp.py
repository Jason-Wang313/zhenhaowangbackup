#!/usr/bin/env python3
"""Advisor-authorized resume-safe FLUX.2 Blur-only full training on eight GPUs."""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, Sampler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party/flux2/src"))

from lowlight_restoration.flow import build_reference_flow_batch  # noqa: E402
from lowlight_restoration.lora import inject_lora, load_lora_checkpoint, save_lora_checkpoint  # noqa: E402
from smoke.train_foundir_blur_only_flux2_overfit_ddp import (  # noqa: E402
    encode_prompt,
    evaluate_snapshot,
    load_crop,
    prepare_records,
    sha256,
)
from smoke.train_foundir_five_type_flux2_full_ddp import (  # noqa: E402
    flow_random_inputs,
    learning_rate_scale,
    prepare_aligned_pair,
    synchronized_loader_batches,
)


def stable_digest(*values: object) -> str:
    return hashlib.sha256(":".join(map(str, values)).encode()).hexdigest()


def write_once_json(path: Path, payload: Mapping[str, object]) -> None:
    content = (json.dumps(dict(payload), indent=2, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise RuntimeError(f"refusing to overwrite differing artifact: {path}")
        return
    with path.open("xb") as handle:
        handle.write(content)


def validate_config(config: Mapping[str, object]) -> None:
    if config.get("status") != "advisor_authorized_blur_only_full_training":
        raise RuntimeError("config is not authorized for Blur-only full training")
    scope = config["scope"]
    forbidden = (
        "qwen_degradation_conditioning",
        "mixed_degradations",
        "pixel_refiner",
        "official_foundir_test_accessed",
        "sealed_internal_accessed",
    )
    if scope.get("category") != "01Blur" or any(scope.get(key) is not False for key in forbidden):
        raise RuntimeError("Blur-only full-training scope changed")
    if config["authorization"].get("hold_id") != "blur_only_tiny_overfit-184f61640dde957e":
        raise RuntimeError("advisor authorization identity changed")
    if config["model"].get("name") != "flux.2-klein-base-4b":
        raise RuntimeError("model family changed")
    if config["lora"].get("initialization") != "fresh_zero_lora":
        raise RuntimeError("full training must not initialize from the memorized tiny-overfit LoRA")
    training = config["training"]
    if training.get("objective") != "pure_noise_to_clean_rectified_flow_with_native_reference":
        raise RuntimeError("Blur-only flow objective changed")
    distributed = config["distributed"]
    if int(distributed["required_world_size"]) != 8:
        raise RuntimeError("Blur-only full training requires eight ranks")
    if distributed.get("allowed_gpu_name_fragments") != ["H100", "H200"]:
        raise RuntimeError("this full-training route must support both H100 and H200")
    profiles = distributed.get("hardware_profiles")
    if not isinstance(profiles, dict) or set(profiles) != {"H100", "H200"}:
        raise RuntimeError("H100/H200 hardware profiles changed")
    expected_profiles = {"H100": (2, 2), "H200": (4, 1)}
    for hardware, (micro_batch, accumulation) in expected_profiles.items():
        profile = profiles[hardware]
        if (
            int(profile.get("micro_batch_size_per_rank", -1)) != micro_batch
            or int(profile.get("gradient_accumulation_steps", -1)) != accumulation
            or int(training["global_batch_size"])
            != int(distributed["required_world_size"]) * micro_batch * accumulation
        ):
            raise RuntimeError(f"{hardware} profile does not preserve global batch 32")
    checkpoints = [int(value) for value in training["checkpoint_steps"]]
    if checkpoints != sorted(set(checkpoints)) or checkpoints[-1] != int(training["max_steps"]):
        raise RuntimeError("checkpoint schedule must be unique, sorted, and end at max_steps")
    inference = config["inference"]
    if inference.get("schedule") != "official_flux2_shifted_schedule" or int(inference["num_steps"]) != 50:
        raise RuntimeError("official FLUX.2 Base 50-step shifted inference is required")
    if float(inference["guidance"]) != 1.0:
        raise RuntimeError("guidance must match the conditional training branch")
    if [int(value) for value in inference["evaluate_steps"]] != [0, *checkpoints]:
        raise RuntimeError("the fixed validation triptychs must cover initialization and every checkpoint")
    if inference.get("selection_protocol") != "human_visual_screen_fixed_triptychs_before_psnr_lpips":
        raise RuntimeError("checkpoint selection must put visual review before metrics")
    if inference.get("automatic_checkpoint_selection") is not False:
        raise RuntimeError("automatic checkpoint selection is forbidden")


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def load_manifests(config: Mapping[str, object], *, verify_paths: bool) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    data = config["data"]
    source = ROOT / str(data["source_manifest"])
    if sha256(source) != data["source_manifest_sha256"]:
        raise RuntimeError("FoundIR source manifest identity changed")
    rows = _load_csv(source)
    isolated_blur = [
        row for row in rows
        if row["category"] == "01Blur" and row["composition_size"] == "1"
    ]
    train = [row for row in isolated_blur if row["split"] == data["training_split"]]
    development = [row for row in isolated_blur if row["split"] == data["development_split"]]
    expected = (
        (train, int(data["expected_training_records"]), int(data["expected_training_source_groups"]), "train"),
        (development, int(data["expected_development_records"]), int(data["expected_development_source_groups"]), "development"),
    )
    for selected, records, groups, label in expected:
        if len(selected) != records or len({row["source_group"] for row in selected}) != groups:
            raise RuntimeError(f"isolated Blur {label} cardinality changed")
        if len({row["record_id"] for row in selected}) != len(selected):
            raise RuntimeError(f"duplicate isolated Blur {label} record")
    train_groups = {row["source_group"] for row in train}
    development_groups = {row["source_group"] for row in development}
    if train_groups & development_groups:
        raise RuntimeError("Blur train/development source-group leakage")

    fixed_path = ROOT / str(data["fixed_validation_manifest"])
    if sha256(fixed_path) != data["fixed_validation_manifest_sha256"]:
        raise RuntimeError("fixed Blur validation manifest identity changed")
    fixed = [
        row for row in _load_csv(fixed_path)
        if row["role"] == data["fixed_validation_role"]
    ]
    if (
        len(fixed) != int(data["fixed_validation_records"])
        or len({row["source_group"] for row in fixed}) != int(data["fixed_validation_source_groups"])
        or {row["source_group"] for row in fixed} & train_groups
    ):
        raise RuntimeError("fixed Blur validation subset changed or leaks training sources")
    development_ids = {row["record_id"] for row in development}
    if any(
        row["record_id"] not in development_ids
        or row["split"] != data["development_split"]
        or row["category"] != "01Blur"
        or row["composition_size"] != "1"
        for row in fixed
    ):
        raise RuntimeError("fixed visual records are not isolated Blur development records")
    if verify_paths:
        for row in [*train, *development]:
            width, height = int(row["width"]), int(row["height"])
            if width < int(data["crop_size"]) or height < int(data["crop_size"]):
                raise RuntimeError(f"undersized Blur record: {row['record_id']}")
            if not (ROOT / row["lq_path"]).is_file() or not (ROOT / row["gt_path"]).is_file():
                raise RuntimeError(f"missing Blur pair: {row['record_id']}")
        for row in fixed:
            for field, hash_field in (("lq_path", "lq_sha256"), ("gt_path", "gt_sha256")):
                if sha256(ROOT / row[field]) != row[hash_field]:
                    raise RuntimeError(f"fixed validation image identity changed: {row['record_id']} {field}")
            load_crop(row, int(data["crop_size"]))
    return train, fixed


class BlurPresentationDataset(Dataset[dict[str, object]]):
    def __init__(self, rows: Sequence[Mapping[str, str]], *, crop_size: int, seed: int) -> None:
        self.rows = rows
        self.crop_size = crop_size
        self.seed = seed
        self._epoch = -1
        self._permutation: torch.Tensor | None = None

    def __len__(self) -> int:
        return 2**31 - 1

    def _record_index(self, position: int) -> int:
        epoch, offset = divmod(position, len(self.rows))
        if epoch != self._epoch:
            generator = torch.Generator(device="cpu").manual_seed(self.seed + epoch)
            self._permutation = torch.randperm(len(self.rows), generator=generator)
            self._epoch = epoch
        assert self._permutation is not None
        return int(self._permutation[offset])

    def __getitem__(self, position: int) -> dict[str, object]:
        row = self.rows[self._record_index(position)]
        with Image.open(ROOT / row["lq_path"]) as handle:
            lq = np.asarray(handle.convert("RGB"), dtype=np.uint8).copy()
        with Image.open(ROOT / row["gt_path"]) as handle:
            gt = np.asarray(handle.convert("RGB"), dtype=np.uint8).copy()
        lq, gt = prepare_aligned_pair(lq, gt, self.crop_size, str(row["record_id"]))
        height, width = gt.shape[:2]
        digest = bytes.fromhex(stable_digest(self.seed, row["record_id"], position, "crop"))
        y = int.from_bytes(digest[:8], "big") % (height - self.crop_size + 1)
        x = int.from_bytes(digest[8:16], "big") % (width - self.crop_size + 1)

        def tensor(array: np.ndarray) -> torch.Tensor:
            crop = array[y : y + self.crop_size, x : x + self.crop_size]
            return torch.from_numpy(crop.copy()).permute(2, 0, 1).float().div_(127.5).sub_(1.0)

        return {
            "record_id": str(row["record_id"]),
            "position": position,
            "lq": tensor(lq),
            "gt": tensor(gt),
        }


class RankPositionSampler(Sampler[int]):
    def __init__(self, *, start: int, stop: int, rank: int, world: int) -> None:
        if start % world or stop % world:
            raise RuntimeError("global presentation boundaries must divide world size")
        self.start = start + rank
        self.stop = stop
        self.world = world

    def __iter__(self):
        return iter(range(self.start, self.stop, self.world))

    def __len__(self) -> int:
        return max(0, (self.stop - self.start + self.world - 1) // self.world)


def collate(items: Sequence[Mapping[str, object]]) -> dict[str, object]:
    return {
        "record_id": [str(item["record_id"]) for item in items],
        "position": [int(item["position"]) for item in items],
        "lq": torch.stack([item["lq"] for item in items]),
        "gt": torch.stack([item["gt"] for item in items]),
    }


def checkpoint_paths(output: Path, step: int) -> tuple[Path, Path]:
    checkpoint = output / f"step_{step:06d}.safetensors"
    return checkpoint, checkpoint.with_suffix(".optimizer.pt")


def save_checkpoint(
    model, optimizer, output: Path, *, step: int, config_sha: str, hardware_profile: str
) -> Path:
    checkpoint, optimizer_path = checkpoint_paths(output, step)
    if checkpoint.exists() or optimizer_path.exists():
        raise RuntimeError(f"refusing to overwrite checkpoint: {checkpoint}")
    save_lora_checkpoint(model, checkpoint, {
        "step": step,
        "config_sha256": config_sha,
        "scope": "advisor_authorized_isolated_blur_full_training",
    })
    with optimizer_path.open("xb") as handle:
        torch.save({
            "step": step,
            "config_sha256": config_sha,
            "checkpoint_sha256": sha256(checkpoint),
            "hardware_profile": hardware_profile,
            "optimizer": optimizer.state_dict(),
        }, handle)
    return checkpoint


def resume(model, optimizer, output: Path, *, config_sha: str) -> tuple[int, str | None]:
    candidates: list[tuple[int, Path]] = []
    for checkpoint in output.glob("step_*.safetensors"):
        try:
            step = int(checkpoint.stem.split("_")[-1])
        except ValueError:
            continue
        optimizer_path = checkpoint.with_suffix(".optimizer.pt")
        if not optimizer_path.is_file():
            raise RuntimeError(f"partial Blur-only checkpoint: {checkpoint}")
        candidates.append((step, checkpoint))
    if not candidates:
        return 0, None
    step, checkpoint = max(candidates)
    load_lora_checkpoint(model, checkpoint)
    payload = torch.load(checkpoint.with_suffix(".optimizer.pt"), map_location="cpu", weights_only=False)
    if (
        int(payload.get("step", -1)) != step
        or payload.get("config_sha256") != config_sha
        or payload.get("checkpoint_sha256") != sha256(checkpoint)
    ):
        raise RuntimeError("resume checkpoint identity changed")
    optimizer.load_state_dict(payload["optimizer"])
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(next(model.parameters()).device)
    previous_profile = payload.get("hardware_profile")
    if previous_profile not in {"H100", "H200"}:
        raise RuntimeError("resume checkpoint lacks a valid H100/H200 profile")
    return step, str(previous_profile)


def snapshot_complete(output: Path, step: int, records: int) -> bool:
    directory = output / "fixed_validation" / "training_triptychs" / f"step_{step:06d}"
    metrics = directory / "metrics.json"
    return metrics.is_file() and len(list(directory.glob("*.png"))) == records


def run_snapshot(model, autoencoder, records, cfg_context, cfg_ids, pack_image, metric, *, step: int, config, output: Path, rank: int, world: int, device: torch.device) -> None:
    complete = snapshot_complete(output, step, len(records))
    flags: list[object | None] = [None] * world
    dist.all_gather_object(flags, complete)
    if not all(flag is True for flag in flags):
        if any(flag is True for flag in flags):
            raise RuntimeError("ranks disagree about fixed-validation snapshot completeness")
        evaluate_snapshot(
            model, autoencoder, records, cfg_context, cfg_ids, pack_image, metric,
            step=step, limit=len(records), config=config,
            output_dir=output / "fixed_validation", rank=rank, world=world, device=device,
        )
    dist.barrier()


def preflight(config_path: Path) -> None:
    config = json.loads(config_path.read_text())
    validate_config(config)
    train, fixed = load_manifests(config, verify_paths=True)
    for key, hash_key in (("flow_weights", "flow_weights_sha256"), ("autoencoder_weights", "autoencoder_weights_sha256")):
        path = ROOT / config["model"][key]
        if sha256(path) != config["model"][hash_key]:
            raise RuntimeError(f"model artifact identity changed: {path}")
    decision = json.loads((ROOT / config["authorization"]["decision"]).read_text())
    if (
        decision.get("hold_id") != config["authorization"]["hold_id"]
        or decision.get("decision") != "approve_documented_change"
        or decision.get("decided_by") != "advisor"
        or "可以试试" not in str(decision.get("statement", ""))
    ):
        raise RuntimeError("matching advisor authorization is absent")
    print("FOUNDIR_BLUR_ONLY_FULL_PREFLIGHT_PASS " + json.dumps({
        "config_sha256": sha256(config_path),
        "training_records": len(train),
        "training_source_groups": len({row["source_group"] for row in train}),
        "fixed_validation_records": len(fixed),
        "train_validation_source_group_overlap": 0,
        "qwen_degradation_conditioning": False,
        "mixed_degradations": False,
        "pixel_refiner": False,
        "sealed_internal_accessed": False,
        "protected_foundir_test_accessed": False,
    }, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    config_path = args.config.resolve()
    if args.preflight:
        preflight(config_path)
        return 0
    config = json.loads(config_path.read_text())
    config["_config_path"] = str(config_path)
    validate_config(config)
    train_rows, fixed_rows = load_manifests(config, verify_paths=False)
    fixed_records = prepare_records(fixed_rows, int(config["data"]["crop_size"]))
    output = ROOT / config["runtime"]["output_dir"]
    output.mkdir(parents=True, exist_ok=True)
    if (output / "training_summary.json").exists():
        raise RuntimeError("Blur-only full training already completed")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend=str(config["distributed"]["backend"]), device_id=device)
    started = time.monotonic()
    try:
        rank, world = dist.get_rank(), dist.get_world_size()
        if world != 8 or torch.cuda.device_count() != 8:
            raise RuntimeError("Blur-only full training requires exactly eight visible GPUs")
        properties = torch.cuda.get_device_properties(device)
        if properties.total_memory < int(config["distributed"]["minimum_memory_gib"]) * 1024**3:
            raise RuntimeError("GPU memory is below contract")
        matching_profiles = [
            fragment
            for fragment in config["distributed"]["allowed_gpu_name_fragments"]
            if fragment in properties.name
        ]
        if len(matching_profiles) != 1:
            raise RuntimeError(f"unsupported GPU: {properties.name}")
        hardware_profile = matching_profiles[0]
        gathered_profiles: list[object | None] = [None] * world
        dist.all_gather_object(gathered_profiles, hardware_profile)
        if set(gathered_profiles) != {hardware_profile}:
            raise RuntimeError(f"mixed GPU hardware is forbidden: {gathered_profiles}")
        seed = int(config["training"]["seed"])
        random.seed(seed + rank)
        np.random.seed(seed + rank)
        torch.manual_seed(seed + rank)
        torch.cuda.manual_seed_all(seed + rank)
        os.environ.update(
            HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
            TORCH_HOME=str(ROOT / config["runtime"]["torch_home"]),
            KLEIN_4B_BASE_MODEL_PATH=str(ROOT / config["model"]["flow_weights"]),
            AE_MODEL_PATH=str(ROOT / config["model"]["autoencoder_weights"]),
            QWEN3_MODEL_PATH=str(ROOT / config["model"]["text_encoder"]),
        )
        context, context_ids, cfg_context, cfg_ids = encode_prompt(config, device)
        from flux2.sampling import batched_prc_img
        from flux2.util import load_ae, load_flow_model
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

        autoencoder = load_ae(config["model"]["name"], device=device).eval().requires_grad_(False)
        model = load_flow_model(config["model"]["name"], device=device).eval().requires_grad_(False)
        names = inject_lora(
            model, list(config["lora"]["target_patterns"]), rank=int(config["lora"]["rank"]),
            alpha=float(config["lora"]["alpha"]), dropout=float(config["lora"]["dropout"]),
        )
        if len(names) != int(config["lora"]["expected_module_count"]):
            raise RuntimeError("FLUX LoRA target count changed")
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        trainable_count = sum(parameter.numel() for parameter in trainable)
        wrapped = DistributedDataParallel(
            model, device_ids=[local_rank], broadcast_buffers=False,
            find_unused_parameters=False, gradient_as_bucket_view=True,
        )
        training = config["training"]
        optimizer = torch.optim.AdamW(
            trainable, lr=float(training["learning_rate"]),
            weight_decay=float(training["weight_decay"]), fused=True,
        )
        config_sha = sha256(config_path)
        start_step, previous_profile = resume(
            wrapped.module, optimizer, output, config_sha=config_sha
        )
        if rank == 0:
            print(
                f"FOUNDIR_BLUR_ONLY_HARDWARE profile={hardware_profile} name={properties.name} "
                f"resume_from_profile={previous_profile or 'none'}",
                flush=True,
            )
        candidates = {int(value) for value in training["checkpoint_steps"]}
        if start_step not in {0, *candidates}:
            raise RuntimeError("resume step is not a frozen candidate checkpoint")

        metric = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(device).eval()
        if start_step == 0:
            run_snapshot(
                wrapped, autoencoder, fixed_records, cfg_context, cfg_ids, batched_prc_img, metric,
                step=0, config=config, output=output, rank=rank, world=world, device=device,
            )
        elif not snapshot_complete(output, start_step, len(fixed_records)):
            run_snapshot(
                wrapped, autoencoder, fixed_records, cfg_context, cfg_ids, batched_prc_img, metric,
                step=start_step, config=config, output=output, rank=rank, world=world, device=device,
            )

        max_steps = int(training["max_steps"])
        runtime_profile = config["distributed"]["hardware_profiles"][hardware_profile]
        micro_batch = int(runtime_profile["micro_batch_size_per_rank"])
        accumulation = int(runtime_profile["gradient_accumulation_steps"])
        global_batch = int(training["global_batch_size"])
        start_position = start_step * global_batch
        stop_position = max_steps * global_batch
        losses: list[float] = []
        if start_step < max_steps:
            dataset = BlurPresentationDataset(
                train_rows, crop_size=int(config["data"]["crop_size"]), seed=int(config["data"]["crop_seed"]),
            )
            sampler = RankPositionSampler(start=start_position, stop=stop_position, rank=rank, world=world)
            workers = int(training["data_loader_workers_per_rank"])
            loader = DataLoader(
                dataset, batch_size=micro_batch, sampler=sampler, num_workers=workers,
                prefetch_factor=3 if workers else None, persistent_workers=bool(workers),
                pin_memory=True, collate_fn=collate, drop_last=True, timeout=120 if workers else 0,
            )
            iterator = iter(loader)
            optimizer.zero_grad(set_to_none=True)
            for step in range(start_step, max_steps):
                batches, data_seconds = synchronized_loader_batches(iterator, accumulation, device=device)
                accumulated = 0.0
                step_started = time.perf_counter()
                for micro, batch in enumerate(batches):
                    gt = batch["gt"].to(device, non_blocking=True)
                    lq = batch["lq"].to(device, non_blocking=True)
                    positions = batch["position"]
                    with torch.no_grad():
                        clean, degraded = autoencoder.encode(torch.cat([gt, lq])).chunk(2)
                    noise, timesteps = flow_random_inputs(clean.shape, positions, seed=seed, device=device)
                    flow = build_reference_flow_batch(
                        clean.to(torch.bfloat16), degraded.to(torch.bfloat16), noise, timesteps,
                        batched_prc_img, reference_time_coordinate=int(training["reference_time_coordinate"]),
                    )
                    batch_context = context.expand(gt.shape[0], *context.shape[1:]).contiguous()
                    batch_context_ids = context_ids.expand(gt.shape[0], *context_ids.shape[1:]).contiguous()
                    sync = contextlib.nullcontext() if micro + 1 == accumulation else wrapped.no_sync()
                    with sync, torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        prediction = wrapped(
                            x=flow.model_tokens, x_ids=flow.model_ids, timesteps=flow.timesteps,
                            ctx=batch_context, ctx_ids=batch_context_ids, guidance=None,
                        )[:, : flow.target_token_count]
                        loss = torch.nn.functional.mse_loss(prediction.float(), flow.target_velocity.float())
                        (loss / accumulation).backward()
                    accumulated += float(loss.detach())
                completed = step + 1
                torch.nn.utils.clip_grad_norm_(trainable, float(training["max_grad_norm"]))
                scale = learning_rate_scale(
                    completed, max_steps, int(training["warmup_steps"]), float(training["minimum_lr_ratio"]),
                )
                for group in optimizer.param_groups:
                    group["lr"] = float(training["learning_rate"]) * scale
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                value = torch.tensor([accumulated / accumulation], device=device)
                dist.all_reduce(value)
                mean_loss = float(value.item() / world)
                losses.append(mean_loss)
                if rank == 0 and (completed == start_step + 1 or completed % 25 == 0):
                    print(
                        f"FOUNDIR_BLUR_ONLY_FULL step={completed}/{max_steps} loss={mean_loss:.8f} "
                        f"lr={optimizer.param_groups[0]['lr']:.8g} step_seconds={time.perf_counter()-step_started:.6f} "
                        f"data_seconds={data_seconds:.6f}", flush=True,
                    )
                if completed in candidates:
                    dist.barrier()
                    if rank == 0:
                        save_checkpoint(
                            wrapped.module,
                            optimizer,
                            output,
                            step=completed,
                            config_sha=config_sha,
                            hardware_profile=hardware_profile,
                        )
                    dist.barrier()
                    run_snapshot(
                        wrapped, autoencoder, fixed_records, cfg_context, cfg_ids, batched_prc_img, metric,
                        step=completed, config=config, output=output, rank=rank, world=world, device=device,
                    )
            if hasattr(iterator, "_shutdown_workers"):
                iterator._shutdown_workers()

        dist.barrier()
        if rank == 0:
            final_checkpoint, _ = checkpoint_paths(output, max_steps)
            if not final_checkpoint.is_file() or not snapshot_complete(output, max_steps, len(fixed_records)):
                raise RuntimeError("final checkpoint or fixed-validation triptychs are incomplete")
            revision = subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True,
            ).stdout.strip()
            payload = {
                "schema_version": 1,
                "status": "training_complete_awaiting_human_visual_review",
                "automatic_checkpoint_selection": False,
                "human_visual_review": "pending",
                "selection_protocol": config["inference"]["selection_protocol"],
                "candidate_steps": sorted(candidates),
                "fixed_validation_records": len(fixed_records),
                "training_records": len(train_rows),
                "training_presentations": max_steps * global_batch,
                "optimizer_steps": max_steps,
                "global_batch_size": global_batch,
                "hardware_profile": hardware_profile,
                "gpu_name": properties.name,
                "micro_batch_size_per_rank": micro_batch,
                "gradient_accumulation_steps": accumulation,
                "mean_loss_this_process_segment": float(np.mean(losses)) if losses else None,
                "final_checkpoint": final_checkpoint.relative_to(ROOT).as_posix(),
                "final_checkpoint_sha256": sha256(final_checkpoint),
                "config": config_path.relative_to(ROOT).as_posix(),
                "config_sha256": config_sha,
                "code_revision": revision,
                "qwen_degradation_conditioning": False,
                "mixed_degradations": False,
                "pixel_refiner": False,
                "sealed_internal_accessed": False,
                "protected_foundir_test_accessed": False,
                "wall_seconds_this_process": time.monotonic() - started,
            }
            write_once_json(output / "training_summary.json", payload)
            print("FOUNDIR_BLUR_ONLY_FULL_TRAINING_COMPLETE_STOP_FOR_VISUAL_REVIEW", flush=True)
        dist.barrier()
        return 0
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
