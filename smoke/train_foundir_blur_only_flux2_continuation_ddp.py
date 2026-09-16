#!/usr/bin/env python3
"""Exact-optimizer low-LR continuation of the verified FLUX.2 Blur-only run."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Mapping

import torch
import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import smoke.train_foundir_blur_only_flux2_full_ddp as base  # noqa: E402
from lowlight_restoration.lora import load_lora_checkpoint  # noqa: E402

ORIGINAL_RESUME = base.resume


def validate_config(config: Mapping[str, object]) -> None:
    if config.get("status") != "user_authorized_blur_only_low_lr_continuation":
        raise RuntimeError("continuation status changed")
    scope = config["scope"]
    expected_scope = {
        "category": "01Blur",
        "qwen_degradation_conditioning": False,
        "mixed_degradations": False,
        "pixel_refiner": False,
        "official_foundir_test_accessed": False,
        "sealed_internal_accessed": False,
        "development_gt_optimization": False,
    }
    if scope != expected_scope:
        raise RuntimeError("continuation scope changed")
    authorization = config["authorization"]
    if (
        authorization.get("hold_id") != "flux2_blur_continuation_authorization-60565f4f75ef056b"
        or authorization.get("hold_sha256") != "52535877d8b0ede5424fb41c6191f7eb9969778ca42d15830de74f17003a628b"
    ):
        raise RuntimeError("continuation authorization changed")
    parent = config["parent"]
    if (
        int(parent.get("step", -1)) != 10000
        or parent.get("config_sha256") != "ba5e2975ba36c9300d454c573882312ddda7415a3de370efcef707899512b61c"
        or parent.get("checkpoint_sha256") != "e0c834d88fc69ba6916032ef7763cdf84047a60d2407eeef23eae8992359c439"
    ):
        raise RuntimeError("parent identity changed")
    if config["lora"].get("initialization") != "exact_parent_step_10000":
        raise RuntimeError("continuation must initialize from the exact parent")
    training = config["training"]
    if (
        training.get("objective") != "pure_noise_to_clean_rectified_flow_with_native_reference"
        or int(training.get("start_step", -1)) != 10000
        or int(training.get("max_steps", -1)) != 14000
        or int(training.get("global_batch_size", -1)) != 32
        or float(training.get("learning_rate", -1)) != 5e-6
        or float(training.get("continuation_final_learning_rate", -1)) != 1e-6
        or training.get("continuation_schedule") != "linear_absolute_steps_10000_to_14000"
        or [int(x) for x in training.get("checkpoint_steps", [])] != [10000, 11000, 12000, 13000, 14000]
    ):
        raise RuntimeError("continuation training contract changed")
    if config["data"].get("expected_training_records") != 105057:
        raise RuntimeError("training set cardinality changed")
    distributed = config["distributed"]
    expected_profiles = {
        "H100": {"micro_batch_size_per_rank": 2, "gradient_accumulation_steps": 2},
        "H200": {"micro_batch_size_per_rank": 4, "gradient_accumulation_steps": 1},
    }
    if (
        int(distributed.get("required_world_size", -1)) != 8
        or distributed.get("allowed_gpu_name_fragments") != ["H100", "H200"]
        or distributed.get("hardware_profiles") != expected_profiles
    ):
        raise RuntimeError("eight-GPU global-batch contract changed")
    inference = config["inference"]
    if (
        inference.get("schedule") != "official_flux2_shifted_schedule"
        or int(inference.get("num_steps", -1)) != 50
        or float(inference.get("guidance", -1)) != 1.0
        or inference.get("automatic_checkpoint_selection") is not False
    ):
        raise RuntimeError("continuation inference contract changed")


def continuation_lr_scale(completed: int, _max_steps: int, _warmup: int, _minimum: float) -> float:
    if not 10000 <= completed <= 14000:
        raise RuntimeError(f"continuation LR requested outside frozen range: {completed}")
    return 1.0 - 0.8 * ((completed - 10000) / 4000.0)


def _optimizer_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def trainable_digest(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            digest.update(name.encode())
            digest.update(parameter.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return digest.hexdigest()


def verify_rank_digests(model: torch.nn.Module) -> None:
    digest = trainable_digest(model)
    gathered: list[object | None] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, digest)
    if set(gathered) != {digest}:
        raise RuntimeError("ranks restored different trainable tensors")


def resume_exact_parent(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    output: Path,
    *,
    config_sha: str,
) -> tuple[int, str | None]:
    candidates = sorted(output.glob("step_*.safetensors"))
    if candidates:
        result = ORIGINAL_RESUME(model, optimizer, output, config_sha=config_sha)
        verify_rank_digests(model)
        return result

    config = json.loads(Path(sys.argv[sys.argv.index("--config") + 1]).resolve().read_text())
    parent = config["parent"]
    checkpoint = ROOT / parent["checkpoint"]
    optimizer_path = ROOT / parent["optimizer"]
    if base.sha256(checkpoint) != parent["checkpoint_sha256"]:
        raise RuntimeError("parent LoRA hash changed")
    load_lora_checkpoint(model, checkpoint)
    payload = torch.load(optimizer_path, map_location="cpu", weights_only=False)
    if (
        int(payload.get("step", -1)) != 10000
        or payload.get("config_sha256") != parent["config_sha256"]
        or payload.get("checkpoint_sha256") != parent["checkpoint_sha256"]
    ):
        raise RuntimeError("parent optimizer metadata changed")
    optimizer.load_state_dict(payload["optimizer"])
    _optimizer_to_device(optimizer, next(model.parameters()).device)
    verify_rank_digests(model)
    return 10000, str(payload.get("hardware_profile"))


def preflight(config_path: Path) -> None:
    config = json.loads(config_path.read_text())
    validate_config(config)
    train, fixed = base.load_manifests(config, verify_paths=True)
    authorization = config["authorization"]
    if base.sha256(ROOT / authorization["evidence"]) != authorization["evidence_sha256"]:
        raise RuntimeError("authorization evidence changed")
    decision = json.loads((ROOT / authorization["decision"]).read_text())
    if (
        decision.get("hold_id") != authorization["hold_id"]
        or decision.get("hold_sha256") != authorization["hold_sha256"]
        or decision.get("decision") != "approve_documented_change"
    ):
        raise RuntimeError("matching user authorization is absent")
    parent = config["parent"]
    for key, hash_key in (("config", "config_sha256"), ("checkpoint", "checkpoint_sha256")):
        if base.sha256(ROOT / parent[key]) != parent[hash_key]:
            raise RuntimeError(f"parent {key} identity changed")
    payload = torch.load(ROOT / parent["optimizer"], map_location="cpu", weights_only=False)
    if (
        int(payload.get("step", -1)) != 10000
        or payload.get("config_sha256") != parent["config_sha256"]
        or payload.get("checkpoint_sha256") != parent["checkpoint_sha256"]
        or not isinstance(payload.get("optimizer"), dict)
    ):
        raise RuntimeError("parent optimizer state is not an exact resumable match")
    for key, hash_key in (("flow_weights", "flow_weights_sha256"), ("autoencoder_weights", "autoencoder_weights_sha256")):
        if base.sha256(ROOT / config["model"][key]) != config["model"][hash_key]:
            raise RuntimeError(f"model artifact identity changed: {key}")
    print("FOUNDIR_BLUR_FLUX2_CONTINUATION_PREFLIGHT_PASS " + json.dumps({
        "config_sha256": base.sha256(config_path),
        "parent_step": 10000,
        "start_global_position": 320000,
        "training_records": len(train),
        "training_source_groups": len({row["source_group"] for row in train}),
        "fixed_validation_records": len(fixed),
        "development_records": int(config["data"]["expected_development_records"]),
        "lr_at_10000": 5e-6,
        "lr_at_12000": 3e-6,
        "lr_at_14000": 1e-6,
        "protected_data_accessed": False,
    }, sort_keys=True), flush=True)


def main() -> int:
    if "--preflight" in sys.argv:
        config_path = Path(sys.argv[sys.argv.index("--config") + 1]).resolve()
        preflight(config_path)
        return 0
    base.validate_config = validate_config
    base.resume = resume_exact_parent
    base.learning_rate_scale = continuation_lr_scale
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
