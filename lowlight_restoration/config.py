"""Strict JSON configuration for FLUX.2 low-light restoration runs."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class TrainConfig:
    path: Path
    raw: dict[str, object]

    def section(self, name: str) -> dict[str, object]:
        value = self.raw.get(name)
        if not isinstance(value, dict):
            raise ValueError(f"config section {name!r} must be an object")
        return value

    def workspace_path(self, section: str, key: str) -> Path:
        value = self.section(section).get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"config {section}.{key} must be a non-empty path string")
        resolved = (ROOT / value).resolve()
        try:
            resolved.relative_to(ROOT.resolve())
        except ValueError as exc:
            raise ValueError(f"config path escapes workspace: {section}.{key}={value}") from exc
        return resolved

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.path.read_bytes()).hexdigest()


def _positive_int(section: dict[str, object], key: str) -> int:
    value = section.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{key} must be a positive integer")
    return value


def _nonnegative_int(section: dict[str, object], key: str) -> int:
    value = section.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{key} must be a non-negative integer")
    return value


def load_train_config(path: Path) -> TrainConfig:
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise ValueError("config schema_version must be 1")
    config = TrainConfig(path=path.resolve(), raw=raw)
    model = config.section("model")
    data = config.section("data")
    training = config.section("training")
    lora = config.section("lora")
    runtime = config.section("runtime")
    inference = config.section("inference")

    if model.get("name") != "flux.2-klein-base-4b":
        raise ValueError("only flux.2-klein-base-4b is allowed for this gate")
    if model.get("dtype") != "bfloat16":
        raise ValueError("model.dtype must be bfloat16")
    if not isinstance(model.get("prompt"), str) or not model["prompt"]:
        raise ValueError("model.prompt must be non-empty")
    if data.get("split") != "train":
        raise ValueError("the baseline training config must use the train split")
    crop_size = _positive_int(data, "crop_size")
    if crop_size % 16:
        raise ValueError("data.crop_size must be divisible by 16")
    if data.get("degradation_type") != "low_light":
        raise ValueError("only low_light data is permitted")

    for key in (
        "seed",
        "max_steps",
        "batch_size",
        "gradient_accumulation_steps",
        "checkpoint_every",
        "log_every",
    ):
        _positive_int(training, key)
    _nonnegative_int(training, "reference_time_coordinate")
    _positive_int(data, "expected_records")
    _positive_int(data, "expected_unique_sources")
    learning_rate = training.get("learning_rate")
    if (
        not isinstance(learning_rate, (int, float))
        or isinstance(learning_rate, bool)
        or not 0 < float(learning_rate) < 1
    ):
        raise ValueError("training.learning_rate must be in (0, 1)")
    max_grad_norm = training.get("max_grad_norm")
    if (
        not isinstance(max_grad_norm, (int, float))
        or isinstance(max_grad_norm, bool)
        or float(max_grad_norm) <= 0
    ):
        raise ValueError("training.max_grad_norm must be positive")
    if training.get("timestep_sampling") != "uniform":
        raise ValueError("only documented uniform timestep sampling is currently implemented")
    weight_decay = training.get("weight_decay")
    if not isinstance(weight_decay, (int, float)) or isinstance(weight_decay, bool) or float(weight_decay) < 0:
        raise ValueError("training.weight_decay must be non-negative")

    _positive_int(lora, "rank")
    alpha = lora.get("alpha")
    if not isinstance(alpha, (int, float)) or isinstance(alpha, bool) or float(alpha) <= 0:
        raise ValueError("lora.alpha must be positive")
    patterns = lora.get("target_patterns")
    if not isinstance(patterns, list) or not patterns or not all(isinstance(item, str) for item in patterns):
        raise ValueError("lora.target_patterns must be a non-empty string list")
    _positive_int(lora, "expected_module_count")
    dropout = lora.get("dropout")
    if (
        not isinstance(dropout, (int, float))
        or isinstance(dropout, bool)
        or not 0 <= float(dropout) < 1
    ):
        raise ValueError("lora.dropout must be in [0, 1)")
    if runtime.get("device") != "cuda":
        raise ValueError("runtime.device must be cuda")
    if runtime.get("offline") is not True:
        raise ValueError("runtime.offline must be true")
    _positive_int(inference, "num_steps")
    _positive_int(inference, "num_samples")
    _positive_int(inference, "seed")
    _positive_int(runtime, "minimum_vram_gib")
    guidance = inference.get("guidance")
    if (
        not isinstance(guidance, (int, float))
        or isinstance(guidance, bool)
        or float(guidance) <= 0
    ):
        raise ValueError("inference.guidance must be positive")

    for section, key in (
        ("data", "manifest"),
        ("model", "flow_weights"),
        ("model", "autoencoder_weights"),
        ("model", "text_encoder"),
        ("runtime", "output_dir"),
    ):
        config.workspace_path(section, key)
    return config
