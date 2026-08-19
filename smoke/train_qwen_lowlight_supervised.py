#!/usr/bin/env python3
"""Train the full Qwen2.5-VL visual path on a 40-record overfit smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import re
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lowlight_restoration.data import PairRecord
from lowlight_restoration.qwen_supervised import (
    canonical_target,
    count_parameter_scope,
    evaluate_generated_outputs,
    open_degraded_images,
    parameter_group,
    prepare_supervised_batch,
    prompt_text,
    record_bundle_sha256,
)
from smoke.qwen_lowlight_supervised_preflight import (
    load_inventory,
    load_supervised_config,
    run_preflight,
    select_records,
    sha256,
    workspace_path,
)

CHECKPOINT_RE = re.compile(r"step_(\d{6})$")


def atomic_write_new_bytes(path: Path, content: bytes) -> None:
    if path.exists():
        if not path.is_file() or path.read_bytes() != content:
            raise RuntimeError(f"existing artifact differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(content)


def atomic_write_new_json(path: Path, payload: object) -> None:
    content = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    atomic_write_new_bytes(path, content)


def read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise RuntimeError(f"JSON artifact is not an object: {path}")
    return payload


def git_revision() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    revision = result.stdout.strip()
    if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise RuntimeError(f"bad Git revision: {revision!r}")
    return revision


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def move_batch(
    batch: dict[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=False) for key, value in batch.items()}


def cpu_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(item) for item in value)
    return value


def optimizer_to_parameter_devices(optimizer: torch.optim.Optimizer) -> None:
    for parameter, state in optimizer.state.items():
        for key, value in list(state.items()):
            if isinstance(value, torch.Tensor):
                state[key] = value.to(parameter.device)


def checkpoint_metadata(
    directory: Path,
    *,
    config_sha256: str,
    manifest_sha256: str,
    selection_sha256: str,
    model_revision: str,
    expected_step: int | None = None,
) -> dict[str, object]:
    metadata_path = directory / "metadata.json"
    if not metadata_path.is_file():
        raise RuntimeError(f"checkpoint metadata missing: {directory}")
    metadata = read_json(metadata_path)
    expected = {
        "config_sha256": config_sha256,
        "manifest_sha256": manifest_sha256,
        "selection_bundle_sha256": selection_sha256,
        "model_revision": model_revision,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise RuntimeError(f"checkpoint {key} differs: {directory}")
    step = metadata.get("step")
    if isinstance(step, bool) or not isinstance(step, int) or step <= 0:
        raise RuntimeError(f"bad checkpoint step: {directory}")
    if expected_step is not None and step != expected_step:
        raise RuntimeError(f"checkpoint step differs: {directory}")
    for name, key in (
        ("visual.safetensors", "visual_sha256"),
        ("optimizer.pt", "optimizer_sha256"),
    ):
        artifact = directory / name
        if not artifact.is_file() or sha256(artifact) != metadata.get(key):
            raise RuntimeError(f"checkpoint artifact differs: {artifact}")
    return metadata


def save_checkpoint(
    *,
    output_dir: Path,
    step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    config_sha256: str,
    manifest_sha256: str,
    selection_sha256: str,
    model_revision: str,
) -> tuple[Path, dict[str, object]]:
    checkpoint_root = output_dir / "checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    destination = checkpoint_root / f"step_{step:06d}"
    if destination.exists():
        return destination, checkpoint_metadata(
            destination,
            config_sha256=config_sha256,
            manifest_sha256=manifest_sha256,
            selection_sha256=selection_sha256,
            model_revision=model_revision,
            expected_step=step,
        )

    temporary = Path(
        tempfile.mkdtemp(prefix=f".step_{step:06d}.partial-", dir=checkpoint_root)
    )
    visual_path = temporary / "visual.safetensors"
    optimizer_path = temporary / "optimizer.pt"
    visual_state = {
        name: tensor.detach().cpu().contiguous()
        for name, tensor in model.visual.state_dict().items()
    }
    save_file(visual_state, visual_path)
    del visual_state
    state = {
        "step": step,
        "optimizer": cpu_tree(optimizer.state_dict()),
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all(),
    }
    torch.save(state, optimizer_path)
    del state
    metadata: dict[str, object] = {
        "schema_version": 1,
        "scope": "qwen2_5_vl_full_visual_path_only",
        "step": step,
        "config_sha256": config_sha256,
        "manifest_sha256": manifest_sha256,
        "selection_bundle_sha256": selection_sha256,
        "model_revision": model_revision,
        "visual_sha256": sha256(visual_path),
        "optimizer_sha256": sha256(optimizer_path),
    }
    atomic_write_new_json(temporary / "metadata.json", metadata)
    if destination.exists():
        raise RuntimeError(f"checkpoint appeared concurrently: {destination}")
    os.rename(temporary, destination)
    return destination, metadata


def latest_checkpoint(output_dir: Path, total_steps: int) -> Path | None:
    checkpoint_root = output_dir / "checkpoints"
    if not checkpoint_root.is_dir():
        return None
    candidates: list[tuple[int, Path]] = []
    for path in checkpoint_root.iterdir():
        match = CHECKPOINT_RE.fullmatch(path.name) if path.is_dir() else None
        if match is not None:
            step = int(match.group(1))
            if step > total_steps:
                raise RuntimeError(f"checkpoint exceeds configured steps: {path}")
            candidates.append((step, path))
    return max(candidates)[1] if candidates else None


def load_checkpoint(
    *,
    directory: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    config_sha256: str,
    manifest_sha256: str,
    selection_sha256: str,
    model_revision: str,
) -> tuple[int, dict[str, object]]:
    metadata = checkpoint_metadata(
        directory,
        config_sha256=config_sha256,
        manifest_sha256=manifest_sha256,
        selection_sha256=selection_sha256,
        model_revision=model_revision,
    )
    visual_state = load_file(directory / "visual.safetensors", device="cpu")
    missing, unexpected = model.visual.load_state_dict(visual_state, strict=True)
    if missing or unexpected:
        raise RuntimeError(
            f"visual checkpoint key mismatch missing={missing} unexpected={unexpected}"
        )
    del visual_state
    state = torch.load(
        directory / "optimizer.pt", map_location="cpu", weights_only=False
    )
    if not isinstance(state, dict) or state.get("step") != metadata["step"]:
        raise RuntimeError("optimizer checkpoint step mismatch")
    optimizer.load_state_dict(state["optimizer"])
    optimizer_to_parameter_devices(optimizer)
    random.setstate(state["python_random_state"])
    np.random.set_state(state["numpy_random_state"])
    torch.set_rng_state(state["torch_rng_state"])
    torch.cuda.set_rng_state_all(state["cuda_rng_state_all"])
    return int(metadata["step"]), metadata


def frozen_snapshot(
    model: torch.nn.Module,
) -> tuple[dict[str, int], dict[str, torch.Tensor]]:
    frozen = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if not parameter.requires_grad
    ]
    if not frozen:
        raise RuntimeError("no frozen Qwen parameters")
    versions = {name: parameter._version for name, parameter in frozen}
    by_group: dict[str, list[tuple[str, torch.nn.Parameter]]] = {
        "language_model": [],
        "lm_head": [],
    }
    for name, parameter in frozen:
        group = parameter_group(name)
        if group in by_group:
            by_group[group].append((name, parameter))
    sentinels: dict[str, torch.Tensor] = {}
    for group, entries in by_group.items():
        if not entries:
            raise RuntimeError(f"frozen sentinel group is empty: {group}")
        entries.sort(key=lambda item: item[0])
        for name, parameter in (entries[0], entries[len(entries) // 2], entries[-1]):
            flat = parameter.detach().reshape(-1)
            indices = torch.tensor(
                sorted({0, max(0, flat.numel() // 2), flat.numel() - 1}),
                device=flat.device,
            )
            sentinels[name] = flat.index_select(0, indices).cpu().clone()
    return versions, sentinels


def frozen_unchanged(
    model: torch.nn.Module,
    versions: dict[str, int],
    sentinels: dict[str, torch.Tensor],
) -> dict[str, object]:
    named = dict(model.named_parameters())
    changed_versions = [
        name for name, version in versions.items() if named[name]._version != version
    ]
    changed_sentinels: list[str] = []
    for name, expected in sentinels.items():
        flat = named[name].detach().reshape(-1)
        indices = torch.tensor(
            sorted({0, max(0, flat.numel() // 2), flat.numel() - 1}),
            device=flat.device,
        )
        actual = flat.index_select(0, indices).cpu()
        if not torch.equal(actual, expected):
            changed_sentinels.append(name)
    frozen_gradients = [
        name
        for name, parameter in named.items()
        if not parameter.requires_grad and parameter.grad is not None
    ]
    return {
        "frozen_parameter_count": sum(
            parameter.numel()
            for parameter in named.values()
            if not parameter.requires_grad
        ),
        "version_counters_checked": len(versions),
        "sentinel_tensors_checked": len(sentinels),
        "changed_version_counters": changed_versions,
        "changed_sentinels": changed_sentinels,
        "frozen_parameters_with_gradients": frozen_gradients,
        "passed": not changed_versions
        and not changed_sentinels
        and not frozen_gradients,
    }


def gradient_audit(model: torch.nn.Module) -> dict[str, object]:
    missing: list[str] = []
    nonfinite: list[str] = []
    frozen_with_grad: list[str] = []
    squared_norms: Counter[str] = Counter()
    trainable_parameters = 0
    tensors_with_grad = 0
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            trainable_parameters += parameter.numel()
            if parameter.grad is None:
                missing.append(name)
                continue
            tensors_with_grad += 1
            if not torch.isfinite(parameter.grad).all():
                nonfinite.append(name)
                continue
            group = parameter_group(name)
            squared_norms[group] += float(parameter.grad.float().square().sum().item())
        elif parameter.grad is not None:
            frozen_with_grad.append(name)
    group_norms = {group: math.sqrt(value) for group, value in squared_norms.items()}
    required_groups = {"visual.patch_embed", "visual.blocks", "visual.merger"}
    zero_groups = sorted(
        group for group in required_groups if group_norms.get(group, 0.0) <= 0.0
    )
    return {
        "trainable_parameters": trainable_parameters,
        "trainable_tensors_with_gradients": tensors_with_grad,
        "missing_trainable_gradients": missing,
        "nonfinite_trainable_gradients": nonfinite,
        "frozen_parameters_with_gradients": frozen_with_grad,
        "gradient_norm_by_group": group_norms,
        "zero_gradient_groups": zero_groups,
        "passed": not missing
        and not nonfinite
        and not frozen_with_grad
        and not zero_groups,
    }


def teacher_forced_loss(
    model: torch.nn.Module,
    processor: Any,
    records: Sequence[PairRecord],
    device: torch.device,
) -> float:
    was_training = model.training
    model.eval()
    weighted_loss = 0.0
    supervised_tokens = 0
    with torch.no_grad():
        for record in records:
            image = open_degraded_images([record])[0]
            batch, target_counts = prepare_supervised_batch(
                processor, [record], [image]
            )
            output = model(**move_batch(batch, device), use_cache=False)
            loss = float(output.loss.detach().float().item())
            if not math.isfinite(loss):
                raise RuntimeError(f"non-finite probe loss for {record.pair_id}")
            weighted_loss += loss * target_counts[0]
            supervised_tokens += target_counts[0]
    if was_training:
        model.train()
    return weighted_loss / supervised_tokens


def generate_outputs(
    model: torch.nn.Module,
    processor: Any,
    records: Sequence[PairRecord],
    device: torch.device,
    max_new_tokens: int,
) -> list[str]:
    model.eval()
    model.config.use_cache = True
    fixed_prompt = prompt_text(processor)
    outputs: list[str] = []
    for index, record in enumerate(records, start=1):
        image = open_degraded_images([record])[0]
        encoded = processor(
            text=[fixed_prompt], images=[image], padding=True, return_tensors="pt"
        )
        inputs = {
            key: value.to(device)
            for key, value in encoded.items()
            if isinstance(value, torch.Tensor)
        }
        input_length = int(inputs["input_ids"].shape[1])
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                use_cache=True,
            )
        continuation = generated[0, input_length:]
        raw = processor.tokenizer.decode(
            continuation,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()
        outputs.append(raw)
        print(
            f"QWEN_SUPERVISED_GENERATE {index}/{len(records)} "
            f"pair={record.pair_id} output={raw!r}",
            flush=True,
        )
    model.config.use_cache = False
    return outputs


def validate_runtime(config: dict[str, object]) -> dict[str, object]:
    runtime = config["runtime"]
    assert isinstance(runtime, dict)
    if (
        os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        != runtime["required_cublas_workspace_config"]
    ):
        raise SystemExit(
            f"CUBLAS_WORKSPACE_CONFIG_MUST_EQUAL_{runtime['required_cublas_workspace_config']}"
        )
    if not torch.cuda.is_available():
        raise SystemExit("CUDA_NOT_AVAILABLE")
    if torch.cuda.device_count() != int(runtime["required_visible_device_count"]):
        raise SystemExit(
            f"VISIBLE_CUDA_DEVICE_COUNT_MUST_EQUAL_{runtime['required_visible_device_count']}"
        )
    properties = torch.cuda.get_device_properties(0)
    total_gib = properties.total_memory / 1024**3
    if total_gib < float(runtime["minimum_gpu_memory_gib"]):
        raise SystemExit(f"GPU_MEMORY_TOO_SMALL_{total_gib:.2f}_GIB")
    import transformers

    actual_versions = {
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "transformers": transformers.__version__,
    }
    expected_versions = {
        "python": runtime["required_python"],
        "pytorch": runtime["required_pytorch"],
        "transformers": runtime["required_transformers"],
    }
    if actual_versions != expected_versions:
        raise SystemExit(f"RUNTIME_VERSION_MISMATCH_{actual_versions}")
    return {
        "gpu": torch.cuda.get_device_name(0),
        "gpu_total_memory_gib": total_gib,
        "visible_cuda_devices": torch.cuda.device_count(),
        "cuda": torch.version.cuda,
        "versions": actual_versions,
    }


def load_model_and_processor(
    config: dict[str, object], inventory: dict[str, object], device: torch.device
) -> tuple[torch.nn.Module, Any, dict[str, int]]:
    configured_model = config["model"]
    processor_config = config["processor"]
    inventory_model = inventory["model"]
    assert isinstance(configured_model, dict)
    assert isinstance(processor_config, dict) and isinstance(inventory_model, dict)
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    local_model = workspace_path(inventory_model["local_path"])
    processor = AutoProcessor.from_pretrained(
        local_model,
        local_files_only=True,
        use_fast=False,
        min_pixels=int(processor_config["min_pixels"]),
        max_pixels=int(processor_config["max_pixels"]),
    )
    processor.tokenizer.padding_side = str(processor_config["padding_side"])
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        local_model,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation=str(configured_model["attention_implementation"]),
        low_cpu_mem_usage=True,
    ).to(device)
    model.requires_grad_(False)
    model.visual.requires_grad_(True)
    model.config.use_cache = False
    groups = count_parameter_scope(list(model.named_parameters()))
    if groups != configured_model["expected_parameter_groups"]:
        raise RuntimeError(f"loaded Qwen parameter scope changed: {groups}")
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    frozen = sum(
        parameter.numel()
        for parameter in model.parameters()
        if not parameter.requires_grad
    )
    if trainable != configured_model["expected_trainable_parameters"]:
        raise RuntimeError(f"trainable parameter count changed: {trainable}")
    if frozen != configured_model["expected_frozen_parameters"]:
        raise RuntimeError(f"frozen parameter count changed: {frozen}")
    bad_trainable_names = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and not name.startswith("model.visual.")
    ]
    if bad_trainable_names:
        raise RuntimeError(
            f"trainable parameters escaped visual path: {bad_trainable_names[:5]}"
        )
    if any(parameter.device != device for parameter in model.parameters()):
        raise RuntimeError("not all Qwen parameters are on the selected CUDA device")
    if any(parameter.dtype != torch.bfloat16 for parameter in model.parameters()):
        raise RuntimeError("not all Qwen parameters loaded as bfloat16")
    return model, processor, groups


def selection_payload(
    records: Sequence[PairRecord], config_sha256: str
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "scope": "synthetic_train_split_overfit_only",
        "config_sha256": config_sha256,
        "selection_bundle_sha256": record_bundle_sha256(records),
        "sources": sorted({record.source_id for record in records}),
        "records": [
            {
                "pair_id": record.pair_id,
                "source_id": record.source_id,
                "level": record.level_index,
                "degraded_path": str(record.degraded_path.relative_to(ROOT)),
                "degraded_sha256": record.degraded_sha256,
                "target": canonical_target(record.level_index),
            }
            for record in records
        ],
        "test_split_used": False,
        "official_foundir_used": False,
    }


def completed_result(output_dir: Path, config_sha256: str) -> int | None:
    path = output_dir / "training_summary.json"
    if not path.exists():
        return None
    summary = read_json(path)
    if summary.get("config_sha256") != config_sha256:
        raise RuntimeError("existing completed result belongs to another config")
    status = summary.get("status")
    automatic_pass = summary.get("automatic_gate_pass")
    checks = summary.get("gate_checks")
    if not isinstance(checks, dict):
        raise RuntimeError("existing completed result has no gate checks")
    if status == "passed" and (automatic_pass is not True or not all(checks.values())):
        raise RuntimeError("existing passed result has inconsistent gate evidence")
    if status == "failed" and automatic_pass is not False:
        raise RuntimeError("existing failed result has inconsistent gate evidence")
    checkpoint_value = summary.get("final_checkpoint")
    checkpoint_summary = summary.get("final_checkpoint_metadata")
    if not isinstance(checkpoint_value, str) or not isinstance(
        checkpoint_summary, dict
    ):
        raise RuntimeError("existing completed result has no final checkpoint evidence")
    checkpoint = (output_dir / checkpoint_value).resolve()
    try:
        checkpoint.relative_to(output_dir.resolve())
    except ValueError as exc:
        raise RuntimeError(
            "existing final checkpoint path escapes its output directory"
        ) from exc
    metadata = read_json(checkpoint / "metadata.json")
    if metadata != checkpoint_summary:
        raise RuntimeError(
            "existing final checkpoint metadata differs from the summary"
        )
    for name, key in (
        ("visual.safetensors", "visual_sha256"),
        ("optimizer.pt", "optimizer_sha256"),
    ):
        if sha256(checkpoint / name) != metadata.get(key):
            raise RuntimeError(f"existing final checkpoint hash differs: {name}")
    if status == "passed":
        print("QWEN_SUPERVISED_OVERFIT40_PASS existing_verified_result", flush=True)
        return 0
    if status == "failed":
        print("QWEN_SUPERVISED_OVERFIT40_FAILED existing_verified_result", flush=True)
        return 2
    raise RuntimeError("existing training summary has no terminal status")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", choices=("auto", "never"), default="auto")
    parser.add_argument("--stop-after-step", type=int)
    args = parser.parse_args()

    config, config_sha, config_raw = load_supervised_config(args.config)
    training = config["training"]
    data = config["data"]
    generation = config["generation"]
    gate = config["gate"]
    runtime = config["runtime"]
    assert all(
        isinstance(section, dict)
        for section in (training, data, generation, gate, runtime)
    )
    output_dir = workspace_path(runtime["output_dir"])
    existing = completed_result(output_dir, config_sha)
    if existing is not None:
        return existing

    preflight = run_preflight(args.config, skip_large_model_hashes=False)
    print(
        "QWEN_SUPERVISED_EMBEDDED_PREFLIGHT_PASS "
        + json.dumps(preflight, sort_keys=True),
        flush=True,
    )
    inventory, _ = load_inventory(config)
    inventory_model = inventory["model"]
    assert isinstance(inventory_model, dict)
    selected, probe = select_records(config)
    selection_sha = record_bundle_sha256(selected)
    if selection_sha != preflight["selection_bundle_sha256"]:
        raise RuntimeError("preflight selection changed before training")

    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_new_bytes(output_dir / "config.json", config_raw)
    atomic_write_new_json(
        output_dir / "preflight.json", {"schema_version": 1, **preflight}
    )
    atomic_write_new_json(
        output_dir / "selection.json", selection_payload(selected, config_sha)
    )

    runtime_evidence = validate_runtime(config)
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    seed = int(training["seed"])
    seed_everything(seed)
    if runtime["deterministic_algorithms"] is True:
        torch.use_deterministic_algorithms(True)
    if runtime["sdpa_backend"] == "math":
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    device = torch.device("cuda:0")
    model, processor, parameter_groups = load_model_and_processor(
        config, inventory, device
    )
    if training["gradient_checkpointing"] is True:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    model.train()

    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=float(training["learning_rate"]),
        betas=(float(training["adam_beta1"]), float(training["adam_beta2"])),
        eps=float(training["adam_epsilon"]),
        weight_decay=float(training["weight_decay"]),
        fused=bool(training["fused_adamw"]),
    )
    optimizer_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    expected_optimizer_ids = {id(parameter) for parameter in trainable_parameters}
    if optimizer_ids != expected_optimizer_ids:
        raise RuntimeError(
            "optimizer scope does not exactly equal the trainable visual path"
        )

    total_steps = int(training["total_steps"])
    start_step = 0
    resumed_metadata: dict[str, object] | None = None
    if args.resume == "auto":
        checkpoint = latest_checkpoint(output_dir, total_steps)
        if checkpoint is not None:
            start_step, resumed_metadata = load_checkpoint(
                directory=checkpoint,
                model=model,
                optimizer=optimizer,
                config_sha256=config_sha,
                manifest_sha256=str(data["manifest_sha256"]),
                selection_sha256=selection_sha,
                model_revision=str(inventory_model["revision"]),
            )
            event = {
                "schema_version": 1,
                "config_sha256": config_sha,
                "resumed_from_step": start_step,
                "checkpoint_metadata_sha256": sha256(checkpoint / "metadata.json"),
            }
            atomic_write_new_json(
                output_dir / "resume_events" / f"step_{start_step:06d}.json", event
            )
            print(
                f"QWEN_SUPERVISED_RESUME_VERIFIED step={start_step} checkpoint={checkpoint}",
                flush=True,
            )
    elif latest_checkpoint(output_dir, total_steps) is not None:
        raise RuntimeError("checkpoints exist but --resume=never was requested")

    frozen_versions, frozen_sentinels = frozen_snapshot(model)
    initial_probe_path = output_dir / "initial_probe.json"
    if initial_probe_path.exists():
        initial_probe = read_json(initial_probe_path)
        if initial_probe.get("config_sha256") != config_sha or initial_probe.get(
            "probe_bundle_sha256"
        ) != record_bundle_sha256(probe):
            raise RuntimeError("existing initial probe differs")
    else:
        if start_step != 0:
            raise RuntimeError("resumed checkpoint has no initial probe evidence")
        initial_loss = teacher_forced_loss(model, processor, probe, device)
        initial_probe = {
            "schema_version": 1,
            "config_sha256": config_sha,
            "probe_bundle_sha256": record_bundle_sha256(probe),
            "records": len(probe),
            "teacher_forced_loss": initial_loss,
        }
        atomic_write_new_json(initial_probe_path, initial_probe)
        print(f"QWEN_SUPERVISED_INITIAL_PROBE loss={initial_loss:.8f}", flush=True)

    if args.stop_after_step is not None:
        if args.stop_after_step <= 0 or args.stop_after_step > total_steps:
            raise ValueError("--stop-after-step is outside the configured run")
        if start_step >= args.stop_after_step:
            required_audit = (
                output_dir
                / f"frozen_parameter_audit_step_{args.stop_after_step:06d}.json"
            )
            if not required_audit.is_file():
                raise RuntimeError(
                    "intentional-stop checkpoint exists without its frozen-parameter audit"
                )
            print(
                f"QWEN_SUPERVISED_INTENTIONAL_STOP_ALREADY_REACHED step={start_step}",
                flush=True,
            )
            return 0

    checkpoint_steps = {int(value) for value in training["checkpoint_steps"]}
    gradient_path = output_dir / "gradient_audit.json"
    run_started = time.monotonic()
    for step in range(start_step + 1, total_steps + 1):
        record = selected[(step - 1) % len(selected)]
        image = open_degraded_images([record])[0]
        batch, target_counts = prepare_supervised_batch(processor, [record], [image])
        optimizer.zero_grad(set_to_none=True)
        output = model(**move_batch(batch, device), use_cache=False)
        loss = output.loss
        if loss is None or not torch.isfinite(loss):
            raise RuntimeError(f"non-finite training loss at step {step}")
        loss.backward()
        if not gradient_path.exists():
            audit = {
                "schema_version": 1,
                "config_sha256": config_sha,
                "step": step,
                **gradient_audit(model),
            }
            atomic_write_new_json(gradient_path, audit)
            if audit["passed"] is not True:
                raise RuntimeError(f"gradient-scope audit failed: {audit}")
            print(
                "QWEN_SUPERVISED_GRADIENT_SCOPE_PASS "
                + json.dumps(audit, sort_keys=True),
                flush=True,
            )
        frozen_with_grad = [
            name
            for name, parameter in model.named_parameters()
            if not parameter.requires_grad and parameter.grad is not None
        ]
        if frozen_with_grad:
            raise RuntimeError(
                f"frozen decoder received gradients: {frozen_with_grad[:5]}"
            )
        grad_norm = torch.nn.utils.clip_grad_norm_(
            trainable_parameters, float(training["max_grad_norm"])
        )
        if not torch.isfinite(grad_norm):
            raise RuntimeError(f"non-finite gradient norm at step {step}")
        optimizer.step()
        metric = {
            "schema_version": 1,
            "config_sha256": config_sha,
            "step": step,
            "pair_id": record.pair_id,
            "source_id": record.source_id,
            "true_level": record.level_index,
            "target": canonical_target(record.level_index),
            "target_tokens": target_counts[0],
            "loss": float(loss.detach().float().item()),
            "gradient_norm_before_clip": float(grad_norm.detach().float().item()),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        atomic_write_new_json(
            output_dir / "step_metrics" / f"step_{step:06d}.json", metric
        )
        elapsed = time.monotonic() - run_started
        print(
            f"QWEN_SUPERVISED_STEP step={step}/{total_steps} pair={record.pair_id} "
            f"loss={metric['loss']:.8f} grad_norm={metric['gradient_norm_before_clip']:.6f} "
            f"elapsed_s={elapsed:.1f}",
            flush=True,
        )
        if args.stop_after_step == step:
            early_frozen_audit = frozen_unchanged(
                model, frozen_versions, frozen_sentinels
            )
            atomic_write_new_json(
                output_dir / f"frozen_parameter_audit_step_{step:06d}.json",
                {
                    "schema_version": 1,
                    "config_sha256": config_sha,
                    **early_frozen_audit,
                },
            )
            if early_frozen_audit["passed"] is not True:
                raise RuntimeError(
                    f"frozen parameters changed before intentional stop: {early_frozen_audit}"
                )
        if step in checkpoint_steps:
            directory, metadata = save_checkpoint(
                output_dir=output_dir,
                step=step,
                model=model,
                optimizer=optimizer,
                config_sha256=config_sha,
                manifest_sha256=str(data["manifest_sha256"]),
                selection_sha256=selection_sha,
                model_revision=str(inventory_model["revision"]),
            )
            print(
                f"QWEN_SUPERVISED_CHECKPOINT step={step} path={directory} "
                f"visual_sha256={metadata['visual_sha256']}",
                flush=True,
            )
        if args.stop_after_step == step:
            if step not in checkpoint_steps:
                raise RuntimeError("intentional stop step was not checkpointed")
            marker = {
                "schema_version": 1,
                "config_sha256": config_sha,
                "intentional_stop_step": step,
                "checkpoint": f"checkpoints/step_{step:06d}",
            }
            atomic_write_new_json(
                output_dir / f"intentional_stop_step_{step:06d}.json", marker
            )
            print(f"QWEN_SUPERVISED_INTENTIONAL_STOP step={step}", flush=True)
            return 0

    required_resume_step = int(gate["required_resume_step"])
    resume_event_path = (
        output_dir / "resume_events" / f"step_{required_resume_step:06d}.json"
    )
    resume_verified = resume_event_path.is_file()
    if resume_verified:
        resume_event = read_json(resume_event_path)
        resume_verified = (
            resume_event.get("config_sha256") == config_sha
            and resume_event.get("resumed_from_step") == required_resume_step
        )
    early_frozen_path = (
        output_dir / f"frozen_parameter_audit_step_{required_resume_step:06d}.json"
    )
    early_frozen_verified = False
    if early_frozen_path.is_file():
        early_frozen_result = read_json(early_frozen_path)
        early_frozen_verified = (
            early_frozen_result.get("config_sha256") == config_sha
            and early_frozen_result.get("passed") is True
        )

    model.gradient_checkpointing_disable()
    final_loss = teacher_forced_loss(model, processor, probe, device)
    raw_outputs = generate_outputs(
        model,
        processor,
        selected,
        device,
        max_new_tokens=int(generation["max_new_tokens"]),
    )
    prediction_rows, understanding_metrics = evaluate_generated_outputs(
        selected, raw_outputs
    )
    atomic_write_new_json(
        output_dir / "predictions.json",
        {
            "schema_version": 1,
            "config_sha256": config_sha,
            "selection_bundle_sha256": selection_sha,
            "records": prediction_rows,
        },
    )
    frozen_audit = frozen_unchanged(model, frozen_versions, frozen_sentinels)
    atomic_write_new_json(
        output_dir / "frozen_parameter_audit.json",
        {"schema_version": 1, "config_sha256": config_sha, **frozen_audit},
    )
    gradient_result = read_json(gradient_path)
    if gradient_result.get("config_sha256") != config_sha:
        raise RuntimeError("gradient audit belongs to another config")
    initial_loss = float(initial_probe["teacher_forced_loss"])
    final_to_initial = final_loss / initial_loss
    per_level_minimum = min(
        float(item["accuracy"]) for item in understanding_metrics["per_level"].values()
    )
    gate_checks = {
        "resume_checkpoint_verified": resume_verified,
        "gradient_scope_passed": gradient_result.get("passed") is True,
        "frozen_parameters_unchanged": early_frozen_verified
        and frozen_audit["passed"] is True,
        "final_probe_loss_passed": final_loss
        <= float(gate["maximum_final_probe_loss"]),
        "probe_loss_ratio_passed": final_to_initial
        <= float(gate["maximum_final_to_initial_probe_loss_ratio"]),
        "structured_output_passed": float(
            understanding_metrics["valid_structured_output_rate"]
        )
        >= float(gate["minimum_valid_structured_output_rate"]),
        "degradation_type_passed": float(
            understanding_metrics["degradation_type_accuracy"]
        )
        >= float(gate["minimum_degradation_type_accuracy"]),
        "severity_memorization_passed": float(
            understanding_metrics["severity_accuracy"]
        )
        >= float(gate["minimum_severity_accuracy"]),
        "every_level_memorized": per_level_minimum
        >= float(gate["minimum_per_level_accuracy"]),
    }
    automatic_pass = all(gate_checks.values())
    training_elapsed_seconds = time.monotonic() - run_started
    final_checkpoint = output_dir / "checkpoints" / f"step_{total_steps:06d}"
    final_checkpoint_meta = checkpoint_metadata(
        final_checkpoint,
        config_sha256=config_sha,
        manifest_sha256=str(data["manifest_sha256"]),
        selection_sha256=selection_sha,
        model_revision=str(inventory_model["revision"]),
        expected_step=total_steps,
    )
    summary: dict[str, object] = {
        "schema_version": 1,
        "status": "passed" if automatic_pass else "failed",
        "scope": "synthetic_lowlight_supervised_vlm_overfit40_code_gradient_gate",
        "config_sha256": config_sha,
        "git_revision": git_revision(),
        "model_repository": inventory_model["repository"],
        "model_revision": inventory_model["revision"],
        "selection_bundle_sha256": selection_sha,
        "manifest_sha256": data["manifest_sha256"],
        "parameter_groups": parameter_groups,
        "trainable_parameters": config["model"]["expected_trainable_parameters"],
        "frozen_parameters": config["model"]["expected_frozen_parameters"],
        "initial_probe_loss": initial_loss,
        "final_probe_loss": final_loss,
        "final_to_initial_probe_loss_ratio": final_to_initial,
        "understanding_metrics": understanding_metrics,
        "gradient_audit": gradient_result,
        "frozen_parameter_audit": frozen_audit,
        "early_frozen_parameter_audit_verified": early_frozen_verified,
        "resume_verified": resume_verified,
        "resumed_metadata": resumed_metadata,
        "optimizer_steps_this_process": total_steps - start_step,
        "training_elapsed_seconds_this_process": training_elapsed_seconds,
        "optimizer_steps_per_second_this_process": (total_steps - start_step)
        / training_elapsed_seconds,
        "final_checkpoint": str(final_checkpoint.relative_to(output_dir)),
        "final_checkpoint_metadata": final_checkpoint_meta,
        "gate_checks": gate_checks,
        "automatic_gate_pass": automatic_pass,
        "environment": {
            **runtime_evidence,
            "model_dtype": "bfloat16",
            "model_device": str(device),
            "maximum_allocated_gpu_memory_gib": torch.cuda.max_memory_allocated()
            / 1024**3,
            "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
            "sdpa_backend": runtime["sdpa_backend"],
        },
        "synthetic_internal_only": True,
        "scientific_pilot_result": False,
        "test_split_used": False,
        "official_foundir_used": False,
    }
    atomic_write_new_json(output_dir / "training_summary.json", summary)
    marker = (
        "QWEN_SUPERVISED_OVERFIT40_PASS"
        if automatic_pass
        else "QWEN_SUPERVISED_OVERFIT40_FAILED"
    )
    print(marker + " " + json.dumps(gate_checks, sort_keys=True), flush=True)
    return 0 if automatic_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
