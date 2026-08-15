#!/usr/bin/env python3
"""Train a resumable FLUX.2 reference-conditioned low-light LoRA baseline."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party/flux2/src"))

from lowlight_restoration.config import TrainConfig, load_train_config
from lowlight_restoration.authorization import validate_run_authorization
from lowlight_restoration.data import LowLightPairDataset, load_pair_manifest
from lowlight_restoration.flow import build_reference_flow_batch
from lowlight_restoration.lora import (
    inject_lora,
    load_lora_checkpoint,
    lora_state_dict,
    save_lora_checkpoint,
)


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content)
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_finite_numbers(value: object, location: str = "root") -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise RuntimeError(f"non-finite training metric at {location}: {value}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            require_finite_numbers(item, f"{location}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            require_finite_numbers(item, f"{location}.{key}")


def publish_once(temporary: Path, destination: Path) -> None:
    """Publish a completed file without ever replacing an existing artifact."""

    try:
        os.link(temporary, destination)
    except FileExistsError as exc:
        raise RuntimeError(f"refusing to overwrite existing artifact: {destination}") from exc
    temporary.unlink()


def atomic_torch_save(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def deterministic_index(position: int, length: int, seed: int) -> int:
    if position < 0 or length <= 0:
        raise ValueError("position must be non-negative and length must be positive")
    epoch, offset = divmod(position, length)
    generator = torch.Generator(device="cpu").manual_seed(seed + epoch)
    return int(torch.randperm(length, generator=generator)[offset])


def collate(dataset: LowLightPairDataset, positions: list[int]) -> dict[str, object]:
    items = [dataset[position] for position in positions]
    return {
        "clean": torch.stack([item["clean"] for item in items]),
        "degraded": torch.stack([item["degraded"] for item in items]),
        "pair_id": [str(item["pair_id"]) for item in items],
        "project_level": [str(item["project_level"]) for item in items],
        "crop_xy": torch.stack([item["crop_xy"] for item in items]),
    }


def prompt_context_path(output_dir: Path) -> Path:
    return output_dir / "prompt_context.safetensors"


def load_or_create_prompt_context(config: TrainConfig, output_dir: Path, device: torch.device):
    from flux2.sampling import batched_prc_txt
    from flux2.util import load_text_encoder

    model = config.section("model")
    path = prompt_context_path(output_dir)
    expected_prompt_hash = hashlib.sha256(str(model["prompt"]).encode()).hexdigest()
    if path.exists():
        tensors = load_file(str(path), device="cpu")
        metadata = json.loads(path.with_suffix(path.suffix + ".json").read_text())
        if metadata.get("prompt_sha256") != expected_prompt_hash:
            raise RuntimeError("existing prompt context belongs to a different prompt")
        required = {"context", "context_ids", "empty_context", "empty_context_ids"}
        if set(tensors) != required:
            raise RuntimeError(f"prompt context has unexpected tensors: {set(tensors)}")
        return tensors["context"].to(device), tensors["context_ids"].to(device)

    text_encoder = load_text_encoder(str(model["name"]), device=device)
    text_encoder.eval().requires_grad_(False)
    with torch.no_grad():
        contexts = text_encoder(["", str(model["prompt"])]).to(torch.bfloat16)
        contexts, context_ids = batched_prc_txt(contexts)
        empty_context, context = contexts[0:1], contexts[1:2]
        empty_context_ids, prompt_context_ids = context_ids[0:1], context_ids[1:2]
    cpu_tensors = {
        "context": context.cpu().contiguous(),
        "context_ids": prompt_context_ids.cpu().contiguous(),
        "empty_context": empty_context.cpu().contiguous(),
        "empty_context_ids": empty_context_ids.cpu().contiguous(),
    }
    temporary = path.with_name(f".{path.name}.tmp")
    save_file(cpu_tensors, str(temporary))
    os.replace(temporary, path)
    atomic_write_text(
        path.with_suffix(path.suffix + ".json"),
        json.dumps({"prompt": model["prompt"], "prompt_sha256": expected_prompt_hash}, indent=2) + "\n",
    )
    del text_encoder
    gc.collect()
    torch.cuda.empty_cache()
    return context.to(device), prompt_context_ids.to(device)


def checkpoint_paths(output_dir: Path, step: int) -> tuple[Path, Path, Path]:
    checkpoint = output_dir / f"step_{step:06d}.safetensors"
    return (
        checkpoint,
        checkpoint.with_suffix(".optimizer.pt"),
        checkpoint.with_suffix(checkpoint.suffix + ".json"),
    )


def load_checkpoint_record(path: Path, config: TrainConfig) -> dict[str, object]:
    optimizer_path = path.with_suffix(".optimizer.pt")
    metadata_path = path.with_suffix(path.suffix + ".json")
    if not path.is_file() or not optimizer_path.is_file() or not metadata_path.is_file():
        raise RuntimeError(f"incomplete checkpoint artifact set: {path}")
    record = json.loads(metadata_path.read_text())
    if not isinstance(record, dict):
        raise RuntimeError(f"checkpoint metadata is not an object: {metadata_path}")
    try:
        expected_step = int(path.stem.split("_", 1)[1])
    except (IndexError, ValueError) as exc:
        raise RuntimeError(f"bad checkpoint filename: {path}") from exc
    if record.get("step") != expected_step or record.get("config_sha256") != config.sha256:
        raise RuntimeError(f"checkpoint metadata does not match run: {path}")
    if record.get("checkpoint_sha256") != sha256(path):
        raise RuntimeError(f"checkpoint SHA-256 mismatch: {path}")
    if record.get("optimizer_sha256") != sha256(optimizer_path):
        raise RuntimeError(f"optimizer SHA-256 mismatch: {optimizer_path}")
    metric = record.get("metric")
    if not isinstance(metric, dict) or metric.get("step") != expected_step:
        raise RuntimeError(f"checkpoint metadata has no matching metric: {metadata_path}")
    require_finite_numbers(metric)
    return record


def latest_checkpoint(output_dir: Path, config: TrainConfig) -> tuple[int, Path] | None:
    steps: set[int] = set()
    for pattern in ("step_*.safetensors", "step_*.optimizer.pt", "step_*.safetensors.json"):
        for path in output_dir.glob(pattern):
            name = path.name.removeprefix("step_").split(".", 1)[0]
            try:
                steps.add(int(name))
            except ValueError:
                continue
    candidates: list[tuple[int, Path]] = []
    for step in sorted(steps):
        path, optimizer_path, metadata_path = checkpoint_paths(output_dir, step)
        present = [item.exists() for item in (path, optimizer_path, metadata_path)]
        if not all(present):
            raise RuntimeError(f"partial checkpoint artifacts for step {step}: {present}")
        load_checkpoint_record(path, config)
        candidates.append((step, path))
    return max(candidates) if candidates else None


def append_metric_once(metrics_path: Path, metric: dict[str, object]) -> bool:
    require_finite_numbers(metric)
    existing: dict[int, dict[str, object]] = {}
    if metrics_path.exists():
        for line_number, line in enumerate(metrics_path.read_text().splitlines(), start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                step = int(item["step"])
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raise RuntimeError(f"invalid metrics entry at line {line_number}") from exc
            require_finite_numbers(item, f"metrics[{step}]")
            if step in existing and existing[step] != item:
                raise RuntimeError(f"conflicting duplicate metric for step {step}")
            existing[step] = item
    step = int(metric["step"])
    if step in existing:
        if existing[step] != metric:
            raise RuntimeError(f"existing metric differs for checkpoint step {step}")
        return False
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("a") as handle:
        handle.write(json.dumps(metric, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return True


def recover_checkpoint_metric(output_dir: Path, checkpoint: Path, config: TrainConfig) -> bool:
    record = load_checkpoint_record(checkpoint, config)
    return append_metric_once(output_dir / "metrics.jsonl", dict(record["metric"]))


def prepare_output_dir(config: TrainConfig, resume: bool) -> tuple[Path, int, Path | None]:
    output_dir = config.workspace_path("runtime", "output_dir")
    if output_dir.exists() and any(output_dir.iterdir()) and not resume:
        raise RuntimeError(f"output directory is non-empty; use --resume: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    frozen_config = output_dir / "config.json"
    current = config.path.read_bytes()
    if frozen_config.exists() and frozen_config.read_bytes() != current:
        raise RuntimeError("output directory contains a different frozen config")
    if not frozen_config.exists():
        temporary = frozen_config.with_name(f".{frozen_config.name}.tmp")
        temporary.write_bytes(current)
        publish_once(temporary, frozen_config)
    latest = latest_checkpoint(output_dir, config) if resume else None
    return output_dir, latest[0] if latest else 0, latest[1] if latest else None


def save_training_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    output_dir: Path,
    step: int,
    config: TrainConfig,
    metric: dict[str, object],
) -> Path:
    path, optimizer_path, metadata_path = checkpoint_paths(output_dir, step)
    if any(item.exists() for item in (path, optimizer_path, metadata_path)):
        raise RuntimeError(f"refusing to overwrite checkpoint artifacts for step {step}")
    metadata = {
        "step": step,
        "config_sha256": config.sha256,
        "base_model": config.section("model")["name"],
        "kind": "flux2_lowlight_restoration_lora",
        "metric": metric,
    }
    temporary_dir = Path(tempfile.mkdtemp(dir=output_dir, prefix=f".step_{step:06d}."))
    temporary_checkpoint = temporary_dir / path.name
    temporary_optimizer = temporary_dir / optimizer_path.name
    temporary_metadata = temporary_checkpoint.with_suffix(temporary_checkpoint.suffix + ".json")
    try:
        save_lora_checkpoint(model, temporary_checkpoint, metadata)
        atomic_torch_save(
            {"step": step, "optimizer": optimizer.state_dict(), "config_sha256": config.sha256},
            temporary_optimizer,
        )
        metadata["checkpoint_sha256"] = sha256(temporary_checkpoint)
        metadata["optimizer_sha256"] = sha256(temporary_optimizer)
        atomic_write_text(temporary_metadata, json.dumps(metadata, indent=2, sort_keys=True) + "\n")
        publish_once(temporary_optimizer, optimizer_path)
        publish_once(temporary_checkpoint, path)
        publish_once(temporary_metadata, metadata_path)
    finally:
        shutil.rmtree(temporary_dir, ignore_errors=True)
    record = load_checkpoint_record(path, config)
    if record["checkpoint_sha256"] != sha256(path):
        raise RuntimeError("published checkpoint failed SHA-256 verification")
    return path


def git_revision(repository: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def runtime_provenance(config: TrainConfig, output_dir: Path) -> dict[str, object]:
    cuda_available = torch.cuda.is_available()
    gpu: dict[str, object] | None = None
    if cuda_available:
        properties = torch.cuda.get_device_properties(0)
        gpu = {
            "name": properties.name,
            "total_memory_bytes": properties.total_memory,
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "device_count": torch.cuda.device_count(),
        }
    output_paths = {
        "output_dir": str(output_dir.relative_to(ROOT)),
        "frozen_config": str((output_dir / "config.json").relative_to(ROOT)),
        "metrics": str((output_dir / "metrics.jsonl").relative_to(ROOT)),
        "prompt_context": str(prompt_context_path(output_dir).relative_to(ROOT)),
        "training_summary": str((output_dir / "training_summary.json").relative_to(ROOT)),
    }
    if config.raw.get("run_name") == "flux2_lowlight_baseline_full":
        output_paths.update(
            validation_evaluation=str((output_dir / "validation_full").relative_to(ROOT)),
            test_evaluation=str((output_dir / "test_full").relative_to(ROOT)),
        )
    else:
        output_paths["evaluation_dir"] = str((output_dir / "evaluation").relative_to(ROOT))
    return {
        "git_revision": git_revision(ROOT),
        "flux_revision": git_revision(ROOT / "third_party/flux2"),
        "command": [sys.executable, *sys.argv],
        "environment": {
            "python": platform.python_version(),
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "pytorch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "cuda_available": cuda_available,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "gpu": gpu,
        },
        "config": config.raw,
        "output_paths": output_paths,
    }


def deterministic_probe(
    flow_model: torch.nn.Module,
    autoencoder: torch.nn.Module,
    dataset: LowLightPairDataset,
    context: torch.Tensor,
    context_ids: torch.Tensor,
    pack_image,
    *,
    batch_size: int,
    seed: int,
    reference_time: int,
    device: torch.device,
) -> dict[str, object]:
    """Evaluate one fixed batch/noise/timestep probe for comparable loss evidence."""

    probe_size = min(batch_size, len(dataset))
    positions = [deterministic_index(index, len(dataset), seed + 71_003) for index in range(probe_size)]
    batch = collate(dataset, positions)
    clean = batch["clean"].to(device=device, dtype=torch.float32)
    degraded = batch["degraded"].to(device=device, dtype=torch.float32)
    was_training = flow_model.training
    flow_model.eval()
    try:
        with torch.no_grad():
            latents = autoencoder.encode(torch.cat([clean, degraded], dim=0))
            clean_latents, degraded_latents = latents.chunk(2)
            clean_latents = clean_latents.to(torch.bfloat16)
            degraded_latents = degraded_latents.to(torch.bfloat16)
            noise_seed = seed + 91_009
            generator = torch.Generator(device=device).manual_seed(noise_seed)
            noise = torch.randn(
                clean_latents.shape,
                generator=generator,
                device=device,
                dtype=torch.bfloat16,
            )
            timesteps = torch.full(
                (probe_size,), 0.5, device=device, dtype=torch.bfloat16
            )
            flow_batch = build_reference_flow_batch(
                clean_latents,
                degraded_latents,
                noise,
                timesteps,
                pack_image,
                reference_time_coordinate=reference_time,
            )
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                prediction = flow_model(
                    x=flow_batch.model_tokens,
                    x_ids=flow_batch.model_ids,
                    timesteps=flow_batch.timesteps,
                    ctx=context.expand(probe_size, -1, -1),
                    ctx_ids=context_ids.expand(probe_size, -1, -1),
                    guidance=None,
                )[:, : flow_batch.target_token_count]
                loss = torch.nn.functional.mse_loss(
                    prediction.float(), flow_batch.target_velocity.float()
                )
    finally:
        flow_model.train(was_training)
    result = {
        "probe_loss": float(loss),
        "probe_pairs": list(batch["pair_id"]),
        "probe_positions": positions,
        "probe_noise_seed": noise_seed,
        "probe_timestep": 0.5,
    }
    require_finite_numbers(result)
    return result


def load_or_create_baseline_probe(
    output_dir: Path,
    config: TrainConfig,
    *,
    start_step: int,
    probe: dict[str, object] | None,
) -> dict[str, object]:
    path = output_dir / "probe_baseline.json"
    if path.exists():
        existing = json.loads(path.read_text())
        if (
            not isinstance(existing, dict)
            or existing.get("config_sha256") != config.sha256
            or existing.get("step") != 0
        ):
            raise RuntimeError("existing deterministic baseline probe does not match this run")
        require_finite_numbers(existing)
        return existing
    if start_step > 0:
        raise RuntimeError("cannot resume checkpoint without deterministic baseline probe")
    if probe is None:
        raise RuntimeError("baseline probe payload is required for a new run")
    payload = {"schema_version": 1, "config_sha256": config.sha256, "step": 0, **probe}
    require_finite_numbers(payload)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    publish_once(temporary, path)
    return payload


def write_training_summary(
    output_dir: Path,
    config: TrainConfig,
    trainable_count: int,
    completed_steps: int,
) -> dict[str, object]:
    final_checkpoint, final_optimizer, final_metadata = checkpoint_paths(output_dir, completed_steps)
    final_record = load_checkpoint_record(final_checkpoint, config)
    recover_checkpoint_metric(output_dir, final_checkpoint, config)
    metrics_path = output_dir / "metrics.jsonl"
    if not metrics_path.is_file():
        raise RuntimeError("cannot summarize training without checkpoint metrics")
    metrics = [json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()]
    by_step: dict[int, dict[str, object]] = {}
    for metric in metrics:
        if not isinstance(metric, dict):
            raise RuntimeError("checkpoint metric is not an object")
        require_finite_numbers(metric)
        step = int(metric["step"])
        if step in by_step and by_step[step] != metric:
            raise RuntimeError(f"conflicting duplicate metric for step {step}")
        by_step[step] = metric
    ordered = [by_step[step] for step in sorted(by_step)]
    if not ordered or int(ordered[-1]["step"]) != completed_steps:
        raise RuntimeError("checkpoint metrics do not reach the completed step")
    baseline_probe_path = output_dir / "probe_baseline.json"
    if not baseline_probe_path.is_file():
        raise RuntimeError("missing deterministic baseline probe")
    baseline_probe = json.loads(baseline_probe_path.read_text())
    if baseline_probe.get("config_sha256") != config.sha256 or baseline_probe.get("step") != 0:
        raise RuntimeError("deterministic baseline probe does not match this run")
    require_finite_numbers(baseline_probe)
    initial_probe_loss = float(baseline_probe["probe_loss"])
    final_probe_loss = float(ordered[-1]["probe_loss"])
    provenance = runtime_provenance(config, output_dir)
    summary: dict[str, object] = {
        "schema_version": 1,
        "run_name": config.raw["run_name"],
        "config_sha256": config.sha256,
        "completed_steps": completed_steps,
        "trainable_parameters": trainable_count,
        "first_checkpoint_step": int(ordered[0]["step"]),
        "initial_probe_loss": initial_probe_loss,
        "final_probe_loss": final_probe_loss,
        "loss_decreased": final_probe_loss < initial_probe_loss,
        "checkpoint": str(final_checkpoint.relative_to(ROOT)),
        "checkpoint_sha256": final_record["checkpoint_sha256"],
        "optimizer_checkpoint": str(final_optimizer.relative_to(ROOT)),
        "optimizer_sha256": final_record["optimizer_sha256"],
        "checkpoint_metadata": str(final_metadata.relative_to(ROOT)),
        "requires_inference_and_visual_gate": True,
        **provenance,
    }
    require_finite_numbers(summary)
    summary_path = output_dir / "training_summary.json"
    if summary_path.exists():
        existing = json.loads(summary_path.read_text())
        for key in (
            "config_sha256",
            "completed_steps",
            "checkpoint",
            "checkpoint_sha256",
            "optimizer_sha256",
            "initial_probe_loss",
            "final_probe_loss",
            "loss_decreased",
        ):
            if existing.get(key) != summary.get(key):
                raise RuntimeError(f"existing training summary differs at {key}")
        require_finite_numbers(existing)
        return existing
    atomic_write_text(summary_path, json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = load_train_config(args.config)
    try:
        authorized_scope = validate_run_authorization(config)
    except RuntimeError as exc:
        raise SystemExit(f"RUN_NOT_AUTHORIZED {exc}") from exc
    print(f"RUN_AUTHORIZATION_PASS scope={authorized_scope}", flush=True)

    os.environ.update(
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        KLEIN_4B_BASE_MODEL_PATH=str(config.workspace_path("model", "flow_weights")),
        AE_MODEL_PATH=str(config.workspace_path("model", "autoencoder_weights")),
        QWEN3_MODEL_PATH=str(config.workspace_path("model", "text_encoder")),
    )
    if not torch.cuda.is_available():
        raise SystemExit("CUDA_NOT_AVAILABLE")
    device = torch.device("cuda")
    training = config.section("training")
    data_config = config.section("data")
    lora_config = config.section("lora")
    seed = int(training["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    records = load_pair_manifest(config.workspace_path("data", "manifest"))
    dataset = LowLightPairDataset(
        records,
        split=str(data_config["split"]),
        crop_size=int(data_config["crop_size"]),
        seed=seed,
    )
    if len(dataset) != int(data_config["expected_records"]):
        raise SystemExit(f"BAD_TRAIN_RECORD_COUNT {len(dataset)}")

    output_dir, start_step, resume_path = prepare_output_dir(config, args.resume)
    context, context_ids = load_or_create_prompt_context(config, output_dir, device)

    from flux2.sampling import batched_prc_img
    from flux2.util import load_ae, load_flow_model

    model_name = str(config.section("model")["name"])
    autoencoder = load_ae(model_name, device=device).eval().requires_grad_(False)
    flow_model = load_flow_model(model_name, device=device).eval().requires_grad_(False)
    target_names = inject_lora(
        flow_model,
        list(lora_config["target_patterns"]),
        rank=int(lora_config["rank"]),
        alpha=float(lora_config["alpha"]),
        dropout=float(lora_config["dropout"]),
    )
    if len(target_names) != int(lora_config["expected_module_count"]):
        raise SystemExit(f"BAD_LORA_MODULE_COUNT {len(target_names)}")
    trainable = [parameter for parameter in flow_model.parameters() if parameter.requires_grad]
    trainable_count = sum(parameter.numel() for parameter in trainable)
    if trainable_count != 8_110_080:
        raise SystemExit(f"BAD_TRAINABLE_PARAMETER_COUNT {trainable_count}")
    batch_size = int(training["batch_size"])
    accumulation = int(training["gradient_accumulation_steps"])
    reference_time = int(training["reference_time_coordinate"])
    checkpoint_every = int(training["checkpoint_every"])
    log_every = int(training["log_every"])
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    baseline_probe_path = output_dir / "probe_baseline.json"
    new_probe = None
    if not baseline_probe_path.exists() and start_step == 0:
        new_probe = deterministic_probe(
            flow_model,
            autoencoder,
            dataset,
            context,
            context_ids,
            batched_prc_img,
            batch_size=batch_size,
            seed=seed,
            reference_time=reference_time,
            device=device,
        )
    baseline_probe = load_or_create_baseline_probe(
        output_dir,
        config,
        start_step=start_step,
        probe=new_probe,
    )
    print("BASELINE_PROBE " + json.dumps(baseline_probe, sort_keys=True), flush=True)
    if resume_path is not None:
        load_lora_checkpoint(flow_model, resume_path)
        optimizer_payload = torch.load(resume_path.with_suffix(".optimizer.pt"), map_location="cpu", weights_only=True)
        if optimizer_payload.get("config_sha256") != config.sha256:
            raise RuntimeError("resume optimizer belongs to a different config")
        optimizer.load_state_dict(optimizer_payload["optimizer"])
        recovered = recover_checkpoint_metric(output_dir, resume_path, config)
        print(f"RESUMED step={start_step} checkpoint={resume_path}", flush=True)
        if recovered:
            print(f"RECOVERED_CHECKPOINT_METRIC step={start_step}", flush=True)

    max_steps = int(training["max_steps"])
    if start_step >= max_steps:
        summary = write_training_summary(output_dir, config, trainable_count, max_steps)
        print("TRAINING_ALREADY_COMPLETE " + json.dumps(summary, sort_keys=True))
        return
    metrics_path = output_dir / "metrics.jsonl"
    optimizer.zero_grad(set_to_none=True)

    for step in range(start_step, max_steps):
        accumulated_loss = 0.0
        step_pairs: list[str] = []
        for micro in range(accumulation):
            base_position = (step * accumulation + micro) * batch_size
            positions = [
                deterministic_index(base_position + offset, len(dataset), seed)
                for offset in range(batch_size)
            ]
            batch = collate(dataset, positions)
            clean = batch["clean"].to(device=device, dtype=torch.float32)
            degraded = batch["degraded"].to(device=device, dtype=torch.float32)
            step_pairs.extend(batch["pair_id"])
            with torch.no_grad():
                latents = autoencoder.encode(torch.cat([clean, degraded], dim=0))
                clean_latents, degraded_latents = latents.chunk(2)
                clean_latents = clean_latents.to(torch.bfloat16)
                degraded_latents = degraded_latents.to(torch.bfloat16)
            generator = torch.Generator(device=device).manual_seed(seed + (step * accumulation + micro) * 1_000_003)
            noise = torch.randn(
                clean_latents.shape,
                generator=generator,
                device=device,
                dtype=torch.bfloat16,
            )
            timesteps = torch.rand(
                (batch_size,), generator=generator, device=device, dtype=torch.bfloat16
            ).clamp_(0.0001, 0.9999)
            flow_batch = build_reference_flow_batch(
                clean_latents,
                degraded_latents,
                noise,
                timesteps,
                batched_prc_img,
                reference_time_coordinate=reference_time,
            )
            batch_context = context.expand(batch_size, -1, -1)
            batch_context_ids = context_ids.expand(batch_size, -1, -1)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                prediction = flow_model(
                    x=flow_batch.model_tokens,
                    x_ids=flow_batch.model_ids,
                    timesteps=flow_batch.timesteps,
                    ctx=batch_context,
                    ctx_ids=batch_context_ids,
                    guidance=None,
                )[:, : flow_batch.target_token_count]
                loss = torch.nn.functional.mse_loss(
                    prediction.float(), flow_batch.target_velocity.float()
                )
                scaled_loss = loss / accumulation
            scaled_loss.backward()
            accumulated_loss += float(loss.detach())

        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, float(training["max_grad_norm"]))
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        completed_step = step + 1
        mean_loss = accumulated_loss / accumulation
        metric = {
            "step": completed_step,
            "loss": mean_loss,
            "grad_norm": float(grad_norm),
            "pairs": step_pairs,
            "max_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
        }
        if completed_step % log_every == 0:
            print("TRAIN " + json.dumps(metric, sort_keys=True), flush=True)
        if completed_step % checkpoint_every == 0 or completed_step == max_steps:
            metric.update(
                deterministic_probe(
                    flow_model,
                    autoencoder,
                    dataset,
                    context,
                    context_ids,
                    batched_prc_img,
                    batch_size=batch_size,
                    seed=seed,
                    reference_time=reference_time,
                    device=device,
                )
            )
            checkpoint = save_training_checkpoint(
                flow_model,
                optimizer,
                output_dir,
                completed_step,
                config,
                metric,
            )
            before = {key: value.clone() for key, value in lora_state_dict(flow_model).items()}
            load_lora_checkpoint(flow_model, checkpoint)
            after = lora_state_dict(flow_model)
            if any(not torch.equal(before[key], after[key]) for key in before):
                raise RuntimeError("LoRA checkpoint round-trip verification failed")
            append_metric_once(metrics_path, metric)
            print(f"CHECKPOINT_ROUNDTRIP_PASS {checkpoint}", flush=True)

    summary = write_training_summary(output_dir, config, trainable_count, max_steps)
    print("LOWLIGHT_OVERFIT_TRAINING_COMPLETE " + json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
