#!/usr/bin/env python3
"""Resume-safe eight-GPU curriculum training for the unified FLUX restorer."""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import math
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

from lowlight_restoration.flow import build_degraded_anchored_flow_batch  # noqa: E402
from lowlight_restoration.flux_prompt_cache import PromptContextCache, file_sha256  # noqa: E402
from lowlight_restoration.foundir_flux2_protocol import (  # noqa: E402
    build_curriculum_pools,
    curriculum_record_index,
    resolve_workspace_path,
    stable_digest,
)
from lowlight_restoration.lora import (  # noqa: E402
    inject_lora,
    load_restoration_checkpoint,
    save_restoration_checkpoint,
)
from lowlight_restoration.pixel_refinement import (  # noqa: E402
    BoundedPixelRefiner,
    pixel_refinement_losses,
    pixel_refiner_parameter_count,
)
from smoke.train_foundir_five_type_flux2_overfit_ddp import unpack_tokens  # noqa: E402


TRAINING_UNDERSIZED_PAIR_POLICY = "bicubic_short_side_to_512_then_aligned_crop_v1"


def write_once(path: Path, payload: Mapping[str, object]) -> None:
    content = (json.dumps(dict(payload), indent=2, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise RuntimeError(f"refusing to overwrite differing training artifact: {path}")
        return
    with path.open("xb") as handle:
        handle.write(content)


def load_rows(path: Path, expected: int) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    required = {
        "record_id", "split", "source_group", "category", "composition_size",
        "lq_path", "gt_path", "qwen_text", "generic_text",
    }
    if reader.fieldnames is None or not required.issubset(reader.fieldnames):
        raise RuntimeError("full condition cache schema changed")
    if len(rows) != expected or len({row["record_id"] for row in rows}) != expected:
        raise RuntimeError("full condition cache cardinality changed")
    return rows


class PresentationDataset(Dataset[dict[str, object]]):
    def __init__(
        self,
        rows: Sequence[Mapping[str, str]],
        *,
        crop_size: int,
        seed: int,
        identity_probability: float,
        generic_probability: float,
    ) -> None:
        self.rows = rows
        self.crop_size = crop_size
        self.seed = seed
        self.identity_probability = identity_probability
        self.generic_probability = generic_probability

    def __len__(self) -> int:
        return 2**31 - 1

    def __getitem__(self, item: tuple[int, int]) -> dict[str, object]:
        record_index, presentation_index = item
        row = self.rows[record_index]
        with Image.open(ROOT / row["lq_path"]) as handle:
            lq = np.asarray(handle.convert("RGB"), dtype=np.uint8).copy()
        with Image.open(ROOT / row["gt_path"]) as handle:
            gt = np.asarray(handle.convert("RGB"), dtype=np.uint8).copy()
        lq, gt = prepare_aligned_pair(lq, gt, self.crop_size, str(row["record_id"]))
        height, width = gt.shape[:2]
        digest = bytes.fromhex(stable_digest(self.seed, row["record_id"], presentation_index, "crop"))
        y = int.from_bytes(digest[:8], "big") % (height - self.crop_size + 1)
        x = int.from_bytes(digest[8:16], "big") % (width - self.crop_size + 1)
        lq = lq[y : y + self.crop_size, x : x + self.crop_size]
        gt = gt[y : y + self.crop_size, x : x + self.crop_size]
        identity_unit = int(stable_digest(self.seed, presentation_index, "identity")[:16], 16) / 2**64
        generic_unit = int(stable_digest(self.seed, presentation_index, "generic")[:16], 16) / 2**64
        identity = identity_unit < self.identity_probability
        generic = identity or generic_unit < self.generic_probability
        if identity:
            lq = gt.copy()

        def tensor(array: np.ndarray) -> torch.Tensor:
            return torch.from_numpy(array.copy()).permute(2, 0, 1).float().div_(127.5).sub_(1.0)

        return {
            "record_id": row["record_id"],
            "presentation_index": presentation_index,
            "gt": tensor(gt),
            "lq": tensor(lq),
            "prompt": row["generic_text"] if generic else row["qwen_text"],
            "identity": identity,
            "generic_dropout": generic and not identity,
        }


def resized_pair_dimensions(width: int, height: int, crop_size: int) -> tuple[int, int]:
    """Upscale only undersized pairs while preserving their aspect ratio."""

    if width < 1 or height < 1 or crop_size < 1:
        raise ValueError("image and crop dimensions must be positive")
    short_side = min(width, height)
    if short_side >= crop_size:
        return width, height
    return (
        max(crop_size, (width * crop_size + short_side - 1) // short_side),
        max(crop_size, (height * crop_size + short_side - 1) // short_side),
    )


def prepare_aligned_pair(
    lq: np.ndarray,
    gt: np.ndarray,
    crop_size: int,
    record_id: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Make a valid aligned crop source without changing normal-size pairs.

    FoundIR contains a very small number of legitimate panoramic Rain pairs
    whose short side is below the 512-pixel training crop.  Upscaling LQ and
    GT with the exact same deterministic geometry retains their alignment and
    avoids silently dropping a frozen-manifest record.
    """

    if lq.ndim != 3 or gt.ndim != 3 or lq.shape[2:] != (3,) or gt.shape[2:] != (3,):
        raise RuntimeError(f"non-RGB FoundIR pair: {record_id}")
    if lq.shape != gt.shape:
        raise RuntimeError(
            f"unaligned FoundIR pair: {record_id}: lq={lq.shape} gt={gt.shape}"
        )
    height, width = gt.shape[:2]
    resized_width, resized_height = resized_pair_dimensions(width, height, crop_size)
    if (resized_width, resized_height) == (width, height):
        return lq, gt
    size = (resized_width, resized_height)
    lq = np.asarray(
        Image.fromarray(lq, mode="RGB").resize(size, Image.Resampling.BICUBIC),
        dtype=np.uint8,
    ).copy()
    gt = np.asarray(
        Image.fromarray(gt, mode="RGB").resize(size, Image.Resampling.BICUBIC),
        dtype=np.uint8,
    ).copy()
    return lq, gt


class CurriculumSampler(Sampler[tuple[int, int]]):
    def __init__(
        self,
        rows: Sequence[Mapping[str, str]],
        *,
        maximum_composition_size: int,
        stage_presentations: int,
        stage_presentation_offset: int,
        stage_sampler_offset: int,
        rank: int,
        world: int,
        seed: int,
    ) -> None:
        self.pools = build_curriculum_pools(rows)
        self.maximum_composition_size = maximum_composition_size
        self.stage_presentations = stage_presentations
        self.stage_presentation_offset = stage_presentation_offset
        self.stage_sampler_offset = stage_sampler_offset
        self.rank = rank
        self.world = world
        self.seed = seed
        if stage_presentations % world:
            raise ValueError("stage presentations must divide world size")

    def __len__(self) -> int:
        return self.stage_presentations // self.world

    def __iter__(self):
        for local in range(len(self)):
            stage_position = local * self.world + self.rank
            sampler_position = self.stage_sampler_offset + stage_position
            record_index = curriculum_record_index(
                self.pools,
                maximum_composition_size=self.maximum_composition_size,
                presentation_index=sampler_position,
                seed=self.seed,
            )
            yield record_index, self.stage_presentation_offset + stage_position


def collate(items: Sequence[Mapping[str, object]]) -> dict[str, object]:
    return {
        "gt": torch.stack([item["gt"] for item in items]),
        "lq": torch.stack([item["lq"] for item in items]),
        "prompt": [str(item["prompt"]) for item in items],
        "record_id": [str(item["record_id"]) for item in items],
        "presentation_index": [int(item["presentation_index"]) for item in items],
        "identity": [bool(item["identity"]) for item in items],
        "generic_dropout": [bool(item["generic_dropout"]) for item in items],
    }


def contexts_for(cache: PromptContextCache, prompts: Sequence[str], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    values = [cache.get(prompt, device) for prompt in prompts]
    return torch.cat([value[0] for value in values]), torch.cat([value[1] for value in values])


def flow_random_inputs(
    latent_shape: Sequence[int],
    presentation_indices: Sequence[int],
    *,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate per-presentation randomness independent of microbatch grouping."""

    if len(latent_shape) < 1 or int(latent_shape[0]) != len(presentation_indices):
        raise ValueError("latent batch and presentation indices differ")
    sample_shape = tuple(int(value) for value in latent_shape[1:])
    noises = []
    timesteps = []
    for presentation_index in presentation_indices:
        generator = torch.Generator(device=device).manual_seed(
            seed + int(presentation_index) * 1000003
        )
        noises.append(
            torch.randn(sample_shape, device=device, dtype=torch.bfloat16, generator=generator)
        )
        timesteps.append(
            torch.rand((), device=device, dtype=torch.bfloat16, generator=generator)
        )
    return torch.stack(noises), torch.stack(timesteps).clamp_(1e-4, 1 - 1e-4)


def synchronized_loader_batches(
    iterator,
    count: int,
    *,
    device: torch.device,
) -> tuple[list[Mapping[str, object]], float]:
    """Load a whole optimizer step on every rank before entering DDP forward.

    A worker exception on one rank must be reported by all ranks before any of
    them launches gradient collectives.  Otherwise healthy ranks can sit in a
    NCCL all-reduce until the watchdog expires, obscuring the actual data error.
    """

    started = time.perf_counter()
    batches: list[Mapping[str, object]] = []
    error: str | None = None
    try:
        for _ in range(count):
            batches.append(next(iterator))
    except Exception as exc:  # propagated collectively below
        error = f"{type(exc).__name__}: {exc}"
    synchronized_rank_error(error, device=device, label="DataLoader failure before DDP forward")
    return batches, time.perf_counter() - started


def synchronized_rank_error(
    error: str | None,
    *,
    device: torch.device,
    label: str,
) -> None:
    """Raise the same diagnostic on every rank before entering model collectives."""

    ready = torch.tensor([0 if error else 1], device=device, dtype=torch.int32)
    dist.all_reduce(ready, op=dist.ReduceOp.MIN)
    if int(ready.item()) != 1:
        errors: list[object | None] = [None] * dist.get_world_size()
        dist.all_gather_object(errors, error)
        details = "; ".join(
            f"rank {rank}: {message}" for rank, message in enumerate(errors) if message
        )
        raise RuntimeError(f"synchronized {label}: {details}")


def learning_rate_scale(step: int, total: int, warmup: int, minimum: float) -> float:
    if step <= warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return minimum + (1.0 - minimum) * 0.5 * (1.0 + math.cos(math.pi * progress))


def checkpoint_paths(output: Path, step: int) -> tuple[Path, Path]:
    checkpoint = output / f"step_{step:06d}.safetensors"
    return checkpoint, checkpoint.with_suffix(".optimizer.pt")


def move_optimizer_state_to_device(optimizer: torch.optim.Optimizer) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(optimizer.param_groups[0]["params"][0].device)


def resume(
    model,
    pixel_refiner,
    optimizer,
    output: Path,
    config_sha: str,
) -> tuple[int, int, list[dict[str, object]], dict[str, object] | None]:
    candidates = []
    for path in output.glob("step_*.safetensors"):
        try:
            step = int(path.stem.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        optimizer_path = path.with_suffix(".optimizer.pt")
        metadata_path = path.with_suffix(path.suffix + ".json")
        if not optimizer_path.is_file() or not metadata_path.is_file():
            raise RuntimeError(f"partial full-training checkpoint: {path}")
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("config_sha256") != config_sha or int(metadata.get("step", -1)) != step:
            raise RuntimeError(f"full-training checkpoint contract changed: {path}")
        candidates.append((step, path))
    if not candidates:
        return 0, 0, [], None
    step, path = max(candidates)
    load_restoration_checkpoint(model, pixel_refiner, path)
    payload = torch.load(path.with_suffix(".optimizer.pt"), map_location="cpu", weights_only=False)
    if (
        payload.get("config_sha256") != config_sha
        or int(payload.get("global_step", -1)) != step
        or payload.get("checkpoint_sha256") != file_sha256(path)
    ):
        raise RuntimeError("full-training optimizer contract changed")
    optimizer.load_state_dict(payload["optimizer"])
    move_optimizer_state_to_device(optimizer)
    return (
        step,
        int(payload["presentations_completed"]),
        list(payload.get("completed_stage_history", payload.get("stage_history", []))),
        payload.get("current_stage_progress"),
    )


def save_checkpoint(
    model,
    pixel_refiner,
    optimizer,
    output: Path,
    *,
    step: int,
    presentations_completed: int,
    stage_history: Sequence[Mapping[str, object]],
    current_stage_progress: Mapping[str, object],
    config_sha: str,
    trainable_parameters: int,
) -> None:
    checkpoint, optimizer_path = checkpoint_paths(output, step)
    if checkpoint.exists() or optimizer_path.exists():
        raise RuntimeError(f"refusing to overwrite full checkpoint: {checkpoint}")
    save_restoration_checkpoint(
        model,
        pixel_refiner,
        checkpoint,
        {
            "step": step,
            "presentations_completed": presentations_completed,
            "config_sha256": config_sha,
            "trainable_parameters": trainable_parameters,
        },
    )
    checkpoint_sha = file_sha256(checkpoint)
    with optimizer_path.open("xb") as handle:
        torch.save(
            {
                "global_step": step,
                "presentations_completed": presentations_completed,
                "config_sha256": config_sha,
                "optimizer": optimizer.state_dict(),
                "completed_stage_history": list(stage_history),
                "current_stage_progress": dict(current_stage_progress),
                "checkpoint_sha256": checkpoint_sha,
            },
            handle,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--benchmark-output", type=Path)
    parser.add_argument("--benchmark-name")
    parser.add_argument("--benchmark-steps", type=int, default=0)
    parser.add_argument("--benchmark-warmup-steps", type=int, default=5)
    parser.add_argument("--micro-batch-size-per-rank", type=int)
    parser.add_argument("--gradient-accumulation-steps", type=int)
    parser.add_argument("--data-loader-workers", type=int)
    parser.add_argument("--data-loader-timeout-seconds", type=int, default=120)
    parser.add_argument("--approved-speed-selection", type=Path)
    args = parser.parse_args()
    benchmark = args.benchmark_output is not None
    if benchmark != (args.benchmark_steps > 0):
        parser.error("--benchmark-output and a positive --benchmark-steps are required together")
    if benchmark != bool(args.benchmark_name):
        parser.error("--benchmark-name is required exactly in benchmark mode")
    if not benchmark and any(
        value is not None
        for value in (
            args.micro_batch_size_per_rank,
            args.gradient_accumulation_steps,
            args.data_loader_workers,
        )
    ):
        parser.error("runtime schedule overrides are benchmark-only")
    if benchmark and args.approved_speed_selection is not None:
        parser.error("an approved speed selection is full-training-only")
    if args.benchmark_warmup_steps < 0 or (
        benchmark and args.benchmark_warmup_steps >= args.benchmark_steps
    ):
        parser.error("benchmark warmup must be non-negative and smaller than benchmark steps")
    if args.data_loader_timeout_seconds < 1:
        parser.error("DataLoader timeout must be positive")
    args.config = resolve_workspace_path(ROOT, args.config)
    if args.benchmark_output is not None:
        args.benchmark_output = resolve_workspace_path(ROOT, args.benchmark_output)
    if args.approved_speed_selection is not None:
        args.approved_speed_selection = resolve_workspace_path(ROOT, args.approved_speed_selection)
    config = json.loads(args.config.read_text())
    config_sha = file_sha256(args.config)
    current_git_revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    speed_selection = (
        json.loads(args.approved_speed_selection.read_text())
        if args.approved_speed_selection is not None
        else None
    )
    full = config["full_training"]
    flow = config["flow_training"]
    refinement = config["pixel_refinement"]
    if refinement.get("enabled") is not True:
        raise RuntimeError("pixel refinement must be enabled after the VAE ceiling failure")
    distributed_config = config["distributed"]
    conditions_path = ROOT / config["full_conditions"]["csv"]
    conditions_summary = ROOT / config["full_conditions"]["summary"]
    condition_identity = json.loads(conditions_summary.read_text())
    if condition_identity.get("conditions_csv_sha256") != file_sha256(conditions_path):
        raise RuntimeError("full Qwen condition cache changed")
    rows = load_rows(conditions_path, int(config["data"]["expected_five_type_records"]))
    prompt_index = ROOT / config["prompt_cache"]["index"]
    prompt_identity = json.loads(prompt_index.read_text())
    if (
        prompt_identity.get("config_sha256") != config_sha
        or prompt_identity.get("conditions_sha256") != file_sha256(conditions_path)
        or prompt_identity.get("protected_foundir_test_accessed") is not False
    ):
        raise RuntimeError("frozen FLUX prompt cache belongs to another run")
    required_prompts = {
        row[key]
        for row in rows
        for key in ("qwen_text", "generic_text")
    }
    required_prompt_hashes = {
        hashlib.sha256(prompt.encode()).hexdigest() for prompt in required_prompts
    }
    if not required_prompt_hashes.issubset(prompt_identity.get("entries", {})):
        raise RuntimeError("frozen prompt cache is missing a full-training condition")
    evaluation_contract = json.loads((ROOT / config["evaluation"]["contract_json"]).read_text())
    if (
        evaluation_contract.get("status") != "frozen_before_flux_full_training"
        or evaluation_contract.get("config_sha256") != config_sha
        or evaluation_contract.get("conditions_sha256") != file_sha256(conditions_path)
        or evaluation_contract.get("sealed_internal_test_accessed") is not False
    ):
        raise RuntimeError("evaluation protocol was not frozen before full training")
    stage0 = (
        ROOT
        / config["runtime"]["output_dir"]
        / config["overfit"]["output_subdirectory"]
        / "result.json"
    )
    stage0_payload = json.loads(stage0.read_text())
    if (
        stage0_payload.get("automatic_gate_pass") is not True
        or stage0_payload.get("config_sha256") != config_sha
    ):
        raise RuntimeError("full training is forbidden before native-reference overfit passes")
    dist.init_process_group(backend=str(distributed_config["backend"]))
    started = time.monotonic()
    try:
        rank = dist.get_rank()
        world = dist.get_world_size()
        local_rank = int(os.environ["LOCAL_RANK"])
        if world != 8 or torch.cuda.device_count() != 8:
            raise RuntimeError("full FLUX training requires exactly eight visible GPUs")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        properties = torch.cuda.get_device_properties(device)
        family = "H200" if "H200" in properties.name else "H100" if "H100" in properties.name else None
        inventory: list[object | None] = [None] * world
        dist.all_gather_object(inventory, family)
        if family is None or set(inventory) != {family}:
            raise RuntimeError(f"full training requires homogeneous H100 or H200 GPUs: {inventory}")
        hardware = full["hardware_schedule"][family]
        if speed_selection is not None:
            comparisons = {
                str(value["benchmark_name"]): value
                for value in speed_selection.get("comparisons", [])
                if isinstance(value, dict) and "benchmark_name" in value
            }
            selected_comparison = comparisons.get(str(speed_selection.get("selected")))
            if (
                speed_selection.get("status") != "speed_benchmark_selection_complete"
                or speed_selection.get("config_sha256") != config_sha
                or speed_selection.get("git_revision") != current_git_revision
                or speed_selection.get("gpu_family") != family
                or int(speed_selection.get("world_size", -1)) != world
                or int(speed_selection.get("global_batch_size", -1))
                != int(distributed_config["global_batch_size"])
                or speed_selection.get("protected_foundir_test_accessed") is not False
                or not isinstance(selected_comparison, dict)
                or selected_comparison.get("short_run_loss_trajectory_equivalent") is not True
                or int(speed_selection.get("selected_micro_batch_size_per_rank", -1))
                != int(selected_comparison.get("micro_batch_size_per_rank", -2))
                or int(speed_selection.get("selected_gradient_accumulation_steps", -1))
                != int(selected_comparison.get("gradient_accumulation_steps", -2))
                or int(speed_selection.get("selected_data_loader_workers_per_rank", -1))
                != int(selected_comparison.get("data_loader_workers_per_rank", -2))
                or float(selected_comparison.get("maximum_peak_memory_fraction", 2.0)) > 0.9
            ):
                raise RuntimeError("approved speed benchmark does not match this full run")
        micro_batch = int(
            args.micro_batch_size_per_rank
            if args.micro_batch_size_per_rank is not None
            else speed_selection["selected_micro_batch_size_per_rank"]
            if speed_selection is not None
            else hardware["micro_batch_size_per_rank"]
        )
        accumulation = int(
            args.gradient_accumulation_steps
            if args.gradient_accumulation_steps is not None
            else speed_selection["selected_gradient_accumulation_steps"]
            if speed_selection is not None
            else hardware["gradient_accumulation_steps"]
        )
        loader_workers = int(
            args.data_loader_workers
            if args.data_loader_workers is not None
            else speed_selection["selected_data_loader_workers_per_rank"]
            if speed_selection is not None
            else 4
        )
        if micro_batch < 1 or accumulation < 1 or loader_workers < 1:
            raise RuntimeError("microbatch, accumulation, and DataLoader workers must be positive")
        global_batch = micro_batch * accumulation * world
        if not benchmark and global_batch != int(distributed_config["global_batch_size"]):
            raise RuntimeError("hardware schedule changed the scientific global batch")
        seed = int(flow["seed"])
        random.seed(seed + rank)
        np.random.seed(seed + rank)
        torch.manual_seed(seed + rank)
        torch.cuda.manual_seed_all(seed + rank)
        os.environ.update(
            HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
            KLEIN_4B_BASE_MODEL_PATH=str(ROOT / config["model"]["flow_weights"]),
            AE_MODEL_PATH=str(ROOT / config["model"]["autoencoder_weights"]),
        )
        prompt_cache = PromptContextCache(
            prompt_index,
            maximum_resident=int(config["prompt_cache"]["maximum_resident_contexts_per_rank"]),
        )
        prompt_preload_error: str | None = None
        try:
            for prompt in sorted(required_prompts):
                prompt_cache.get(prompt, torch.device("cpu"))
        except Exception as exc:
            prompt_preload_error = f"{type(exc).__name__}: {exc}"
        synchronized_rank_error(
            prompt_preload_error,
            device=device,
            label="prompt-cache preload failure before DDP construction",
        )
        if rank == 0:
            print(
                f"FOUNDIR_FLUX_PROMPT_CACHE_PRELOAD_PASS prompts={len(required_prompts)}",
                flush=True,
            )
        from flux2.sampling import batched_prc_img
        from flux2.util import load_ae, load_flow_model

        autoencoder = load_ae(config["model"]["name"], device=device).eval().requires_grad_(False)
        model = load_flow_model(config["model"]["name"], device=device).eval().requires_grad_(False)
        torch.cuda.reset_peak_memory_stats(device)
        with torch.inference_mode():
            canary = torch.zeros(
                (2 * micro_batch, 3, int(config["data"]["crop_size"]), int(config["data"]["crop_size"])),
                device=device,
                dtype=torch.float32,
            )
            canary_latent = autoencoder.encode(canary)
            canary_reconstruction = autoencoder.decode(canary_latent)
            if canary_reconstruction.shape != canary.shape:
                raise RuntimeError("RGB auxiliary memory canary changed tensor shape")
            del canary, canary_latent, canary_reconstruction
        canary_peak_bytes = int(torch.cuda.max_memory_allocated(device))
        torch.cuda.empty_cache()
        names = inject_lora(
            model,
            list(config["lora"]["target_patterns"]),
            rank=int(config["lora"]["rank"]),
            alpha=float(config["lora"]["alpha"]),
            dropout=float(config["lora"]["dropout"]),
        )
        if len(names) != int(config["lora"]["expected_module_count"]):
            raise RuntimeError("FLUX LoRA module count changed")
        pixel_refiner = BoundedPixelRefiner(
            width=int(refinement["width"]),
            blocks=int(refinement["blocks"]),
            maximum_residual_m11=float(refinement["maximum_residual_m11"]),
        ).to(device=device, dtype=torch.bfloat16)
        refiner_count = pixel_refiner_parameter_count(pixel_refiner)
        if refiner_count != int(refinement["expected_trainable_parameters"]):
            raise RuntimeError(f"pixel-refiner parameter count changed: {refiner_count}")
        trainable = [
            *(parameter for parameter in model.parameters() if parameter.requires_grad),
            *pixel_refiner.parameters(),
        ]
        trainable_count = sum(parameter.numel() for parameter in trainable)
        wrapped = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            broadcast_buffers=False,
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
        )
        wrapped_refiner = DistributedDataParallel(
            pixel_refiner,
            device_ids=[local_rank],
            broadcast_buffers=False,
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
        )
        optimizer = torch.optim.AdamW(
            trainable,
            lr=float(full["learning_rate"]),
            weight_decay=float(full["weight_decay"]),
            fused=True,
        )
        output = ROOT / config["runtime"]["output_dir"] / "full_curriculum"
        if benchmark:
            global_step, presentations_completed, stage_history, resumed_stage_progress = 0, 0, [], None
        else:
            output.mkdir(parents=True, exist_ok=True)
            global_step, presentations_completed, stage_history, resumed_stage_progress = resume(
                wrapped.module,
                wrapped_refiner.module,
                optimizer,
                output,
                config_sha,
            )
        total_presentations = sum(int(stage["presentations"]) for stage in full["curriculum"])
        total_steps = total_presentations // global_batch
        expected_step = presentations_completed // global_batch
        if global_step != expected_step:
            raise RuntimeError("resume optimizer step/presentation offset changed")
        dataset = PresentationDataset(
            rows,
            crop_size=int(config["data"]["crop_size"]),
            seed=seed,
            identity_probability=float(full["identity_pair_probability"]),
            generic_probability=float(full["generic_condition_probability"]),
        )
        stage_offset = 0
        loss_config = full["loss"]
        benchmark_rank_step_seconds: list[float] = []
        benchmark_rank_data_seconds: list[float] = []
        benchmark_mean_losses: list[float] = []
        benchmark_done = False
        torch.cuda.reset_peak_memory_stats(device)
        for stage in full["curriculum"]:
            stage_presentations = int(stage["presentations"])
            stage_end = stage_offset + stage_presentations
            if presentations_completed >= stage_end:
                if resumed_stage_progress is not None and resumed_stage_progress.get("stage") == stage["name"]:
                    if int(resumed_stage_progress.get("presentations_completed", -1)) != stage_end:
                        raise RuntimeError("completed-stage resume progress has the wrong boundary")
                    stage_history.append(
                        {
                            "stage": stage["name"],
                            "global_step": int(resumed_stage_progress["global_step"]),
                            "presentations_completed": stage_end,
                            "mean_loss": float(resumed_stage_progress["mean_loss_so_far"]),
                        }
                    )
                    resumed_stage_progress = None
                stage_offset = stage_end
                continue
            if presentations_completed < stage_offset:
                raise RuntimeError("resume state skips a curriculum stage")
            consumed_in_stage = presentations_completed - stage_offset
            remaining = stage_presentations - consumed_in_stage
            sampler = CurriculumSampler(
                rows,
                maximum_composition_size=int(stage["maximum_composition_size"]),
                stage_presentations=remaining,
                stage_presentation_offset=presentations_completed,
                stage_sampler_offset=consumed_in_stage,
                rank=rank,
                world=world,
                seed=seed + int(stage["maximum_composition_size"]) * 1000,
            )
            loader = DataLoader(
                dataset,
                batch_size=micro_batch,
                sampler=sampler,
                num_workers=loader_workers,
                prefetch_factor=3,
                persistent_workers=True,
                pin_memory=True,
                collate_fn=collate,
                drop_last=True,
                timeout=args.data_loader_timeout_seconds,
            )
            iterator = iter(loader)
            optimizer.zero_grad(set_to_none=True)
            stage_loss_sum = 0.0
            stage_steps = 0
            if resumed_stage_progress is not None:
                if (
                    resumed_stage_progress.get("stage") != stage["name"]
                    or int(resumed_stage_progress.get("presentations_completed", -1)) != presentations_completed
                ):
                    raise RuntimeError("resumed stage progress does not match checkpoint position")
                stage_loss_sum = float(resumed_stage_progress.get("loss_sum", 0.0))
                stage_steps = int(resumed_stage_progress.get("steps", 0))
                resumed_stage_progress = None
            steps_this_stage = remaining // global_batch
            for _ in range(steps_this_stage):
                optimizer_step = global_step + 1
                accumulated = 0.0
                torch.cuda.synchronize(device)
                optimizer_step_started = time.perf_counter()
                step_batches, data_seconds = synchronized_loader_batches(
                    iterator,
                    accumulation,
                    device=device,
                )
                for micro, batch in enumerate(step_batches):
                    gt = batch["gt"].to(device, non_blocking=True)
                    lq = batch["lq"].to(device, non_blocking=True)
                    context, context_ids = contexts_for(prompt_cache, batch["prompt"], device)
                    with torch.no_grad():
                        clean_latent, degraded_latent = autoencoder.encode(torch.cat([gt, lq])).chunk(2)
                    noise, timesteps = flow_random_inputs(
                        clean_latent.shape,
                        batch["presentation_index"],
                        seed=seed,
                        device=device,
                    )
                    flow_batch = build_degraded_anchored_flow_batch(
                        clean_latent.to(torch.bfloat16),
                        degraded_latent.to(torch.bfloat16),
                        noise,
                        timesteps,
                        batched_prc_img,
                        anchor_noise_scale=float(flow["anchor_noise_scale"]),
                        reference_time_coordinate=int(flow["reference_time_coordinate"]),
                    )
                    rgb_auxiliary = optimizer_step % int(loss_config["rgb_auxiliary_every_steps"]) == 0
                    sync_model = contextlib.nullcontext() if micro + 1 == accumulation else wrapped.no_sync()
                    sync_refiner = (
                        contextlib.nullcontext()
                        if micro + 1 == accumulation or not rgb_auxiliary
                        else wrapped_refiner.no_sync()
                    )
                    with sync_model, sync_refiner:
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            prediction = wrapped(
                                x=flow_batch.model_tokens,
                                x_ids=flow_batch.model_ids,
                                timesteps=flow_batch.timesteps,
                                ctx=context,
                                ctx_ids=context_ids,
                                guidance=None,
                            )[:, : flow_batch.target_token_count]
                            flow_loss = torch.nn.functional.mse_loss(
                                prediction.float(), flow_batch.target_velocity.float()
                            )
                            velocity_latent = unpack_tokens(prediction, clean_latent.shape)
                            source_latent = degraded_latent + float(flow["anchor_noise_scale"]) * noise
                            predicted_clean = source_latent - velocity_latent
                            latent_loss = torch.nn.functional.l1_loss(
                                predicted_clean.float(), clean_latent.float()
                            )
                            loss = (
                                float(loss_config["flow_mse_weight"]) * flow_loss
                                + float(loss_config["predicted_clean_latent_l1_weight"]) * latent_loss
                            )
                            if rgb_auxiliary:
                                decoded = autoencoder.decode(predicted_clean).float().clamp(-1, 1)
                                refined = wrapped_refiner(decoded, lq)
                                refinement_terms = pixel_refinement_losses(
                                    refined,
                                    decoded,
                                    gt,
                                    charbonnier_epsilon=float(refinement["charbonnier_epsilon"]),
                                )
                                loss = (
                                    loss
                                    + float(loss_config["rgb_charbonnier_weight"])
                                    * refinement_terms["charbonnier"]
                                    + float(refinement["edge_loss_weight"])
                                    * refinement_terms["edge"]
                                    + float(refinement["residual_penalty_weight"])
                                    * refinement_terms["residual"]
                                )
                        (loss / accumulation).backward()
                    accumulated += float(loss.detach())
                torch.nn.utils.clip_grad_norm_(trainable, float(full["max_grad_norm"]))
                global_step += 1
                scale = learning_rate_scale(
                    global_step,
                    total_steps,
                    int(full["warmup_steps"]),
                    float(full["minimum_learning_rate_ratio"]),
                )
                for group in optimizer.param_groups:
                    group["lr"] = float(full["learning_rate"]) * scale
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                presentations_completed += global_batch
                value = torch.tensor([accumulated / accumulation], device=device)
                dist.all_reduce(value)
                mean_loss = float(value.item() / world)
                torch.cuda.synchronize(device)
                step_seconds = time.perf_counter() - optimizer_step_started
                stage_loss_sum += mean_loss
                stage_steps += 1
                if benchmark:
                    benchmark_rank_step_seconds.append(step_seconds)
                    benchmark_rank_data_seconds.append(data_seconds)
                    benchmark_mean_losses.append(mean_loss)
                if rank == 0 and (
                    benchmark or global_step % 25 == 0 or stage_steps == 1
                ):
                    print(
                        f"FOUNDIR_FLUX_FULL stage={stage['name']} step={global_step}/{total_steps} "
                        f"loss={mean_loss:.8f} lr={optimizer.param_groups[0]['lr']:.8g} "
                        f"step_seconds={step_seconds:.6f} data_seconds={data_seconds:.6f}",
                        flush=True,
                    )
                if benchmark and global_step >= args.benchmark_steps:
                    benchmark_done = True
                    break
                checkpoint_due = (
                    global_step % int(full["checkpoint_every_steps"]) == 0
                    or presentations_completed == stage_end
                )
                if checkpoint_due:
                    dist.barrier()
                    if rank == 0:
                        progress = {
                            "stage": stage["name"],
                            "global_step": global_step,
                            "presentations_completed": presentations_completed,
                            "loss_sum": stage_loss_sum,
                            "steps": stage_steps,
                            "mean_loss_so_far": stage_loss_sum / stage_steps,
                        }
                        save_checkpoint(
                            wrapped.module,
                            wrapped_refiner.module,
                            optimizer,
                            output,
                            step=global_step,
                            presentations_completed=presentations_completed,
                            stage_history=stage_history,
                            current_stage_progress=progress,
                            config_sha=config_sha,
                            trainable_parameters=trainable_count,
                        )
                    dist.barrier()
            if benchmark_done:
                if hasattr(iterator, "_shutdown_workers"):
                    iterator._shutdown_workers()
                del iterator, loader
                break
            stage_history.append(
                {
                    "stage": stage["name"],
                    "global_step": global_step,
                    "presentations_completed": presentations_completed,
                    "mean_loss": stage_loss_sum / max(1, stage_steps),
                }
            )
            if hasattr(iterator, "_shutdown_workers"):
                iterator._shutdown_workers()
            del iterator, loader
            stage_offset = stage_end
        if benchmark:
            if not benchmark_done or len(benchmark_rank_step_seconds) != args.benchmark_steps:
                raise RuntimeError("benchmark did not execute the requested optimizer steps")
            rank_result = {
                "rank": rank,
                "step_seconds": benchmark_rank_step_seconds,
                "data_seconds": benchmark_rank_data_seconds,
                "peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                "peak_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
                "total_memory_bytes": int(properties.total_memory),
            }
            rank_results: list[object | None] = [None] * world
            dist.all_gather_object(rank_results, rank_result)
            if rank == 0:
                concrete = [dict(value) for value in rank_results if isinstance(value, dict)]
                if len(concrete) != world:
                    raise RuntimeError("benchmark rank results are incomplete")
                global_step_seconds = [
                    max(float(value["step_seconds"][index]) for value in concrete)
                    for index in range(args.benchmark_steps)
                ]
                global_data_seconds = [
                    max(float(value["data_seconds"][index]) for value in concrete)
                    for index in range(args.benchmark_steps)
                ]
                measured_step_seconds = global_step_seconds[args.benchmark_warmup_steps :]
                measured_seconds = sum(measured_step_seconds)
                measured_steps = len(measured_step_seconds)
                payload = {
                    "schema_version": 1,
                    "status": "speed_benchmark_complete",
                    "benchmark_name": args.benchmark_name,
                    "config": args.config.relative_to(ROOT).as_posix(),
                    "config_sha256": config_sha,
                    "git_revision": current_git_revision,
                    "gpu_family": family,
                    "world_size": world,
                    "global_batch_size": global_batch,
                    "micro_batch_size_per_rank": micro_batch,
                    "gradient_accumulation_steps": accumulation,
                    "data_loader_workers_per_rank": loader_workers,
                    "data_loader_timeout_seconds": args.data_loader_timeout_seconds,
                    "optimizer_steps": args.benchmark_steps,
                    "warmup_steps_excluded": args.benchmark_warmup_steps,
                    "measured_steps": measured_steps,
                    "measured_seconds": measured_seconds,
                    "mean_seconds_per_step": measured_seconds / measured_steps,
                    "global_samples_per_second": global_batch * measured_steps / measured_seconds,
                    "step_seconds": global_step_seconds,
                    "data_seconds": global_data_seconds,
                    "mean_losses": benchmark_mean_losses,
                    "peak_memory_allocated_bytes_per_rank": [
                        int(value["peak_memory_allocated_bytes"]) for value in concrete
                    ],
                    "peak_memory_reserved_bytes_per_rank": [
                        int(value["peak_memory_reserved_bytes"]) for value in concrete
                    ],
                    "total_memory_bytes_per_rank": [
                        int(value["total_memory_bytes"]) for value in concrete
                    ],
                    "training_undersized_pair_policy": TRAINING_UNDERSIZED_PAIR_POLICY,
                    "per_presentation_randomness_independent_of_microbatch_grouping": True,
                    "checkpoint_written": False,
                    "protected_foundir_test_accessed": False,
                }
                if not all(math.isfinite(float(value)) for value in benchmark_mean_losses):
                    raise RuntimeError("benchmark produced a non-finite loss")
                write_once(args.benchmark_output, payload)
                print("FOUNDIR_FLUX_SPEED_BENCHMARK_COMPLETE " + json.dumps(payload, sort_keys=True), flush=True)
            dist.barrier()
            return
        if resumed_stage_progress is not None:
            raise RuntimeError("checkpoint stage progress was not consumed by the curriculum")
        if rank == 0:
            checkpoint, _ = checkpoint_paths(output, global_step)
            payload = {
                "schema_version": 1,
                "status": "training_complete_pending_development_selection",
                "config_sha256": config_sha,
                "world_size": world,
                "gpu_family": family,
                "global_batch_size": global_batch,
                "micro_batch_size_per_rank": micro_batch,
                "gradient_accumulation_steps": accumulation,
                "speed_benchmark_selection": (
                    args.approved_speed_selection.relative_to(ROOT).as_posix()
                    if args.approved_speed_selection is not None
                    else None
                ),
                "speed_benchmark_selection_sha256": (
                    file_sha256(args.approved_speed_selection)
                    if args.approved_speed_selection is not None
                    else None
                ),
                "training_undersized_pair_policy": TRAINING_UNDERSIZED_PAIR_POLICY,
                "per_presentation_randomness_independent_of_microbatch_grouping": True,
                "completed_steps": global_step,
                "presentations_completed": presentations_completed,
                "trainable_parameters": trainable_count,
                "pixel_refiner_parameters": refiner_count,
                "pixel_refinement_every_steps": int(loss_config["rgb_auxiliary_every_steps"]),
                "maximum_pixel_residual_m11": float(refinement["maximum_residual_m11"]),
                "rgb_memory_canary_peak_bytes_per_rank": canary_peak_bytes,
                "stage_history": stage_history,
                "final_checkpoint": checkpoint.relative_to(ROOT).as_posix(),
                "final_checkpoint_sha256": file_sha256(checkpoint),
                "wall_seconds": time.monotonic() - started,
                "protected_foundir_test_accessed": False,
            }
            write_once(output / "training_summary.json", payload)
            print("FOUNDIR_FLUX_FULL_TRAINING_COMPLETE " + json.dumps(payload, sort_keys=True), flush=True)
        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
