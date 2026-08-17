"""Combined FLUX LoRA and Lowlight condition-adapter checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import torch
from safetensors.torch import load_file, save_file
from torch import Tensor, nn

from .condition_adapter import LowLightLevelContext
from .lora import lora_state_dict

ADAPTER_KEY = "condition_adapter.embedding.weight"


def conditioned_state_dict(
    flow_model: nn.Module, adapter: LowLightLevelContext
) -> dict[str, Tensor]:
    state = lora_state_dict(flow_model)
    state[ADAPTER_KEY] = adapter.embedding.weight.detach().cpu().contiguous()
    return state


def save_conditioned_checkpoint(
    flow_model: nn.Module,
    adapter: LowLightLevelContext,
    path: Path,
    metadata: Mapping[str, object],
) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite conditioned checkpoint: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        conditioned_state_dict(flow_model, adapter),
        str(path),
        metadata={str(key): str(value) for key, value in metadata.items()},
    )


def load_conditioned_checkpoint(
    flow_model: nn.Module, adapter: LowLightLevelContext, path: Path
) -> None:
    expected = conditioned_state_dict(flow_model, adapter)
    loaded = load_file(str(path), device="cpu")
    if set(loaded) != set(expected):
        missing = sorted(set(expected) - set(loaded))
        extra = sorted(set(loaded) - set(expected))
        raise RuntimeError(
            f"conditioned checkpoint key mismatch missing={missing} extra={extra}"
        )
    flow_parameters = dict(flow_model.named_parameters())
    for key, tensor in loaded.items():
        expected_tensor = expected[key]
        if tensor.shape != expected_tensor.shape:
            raise RuntimeError(f"conditioned checkpoint shape mismatch: {key}")
        if key == ADAPTER_KEY:
            target = adapter.embedding.weight
        else:
            try:
                target = flow_parameters[key]
            except KeyError as exc:
                raise RuntimeError(f"missing conditioned target parameter: {key}") from exc
        target.data.copy_(tensor.to(device=target.device, dtype=target.dtype))


def trainable_parameters(
    flow_model: nn.Module, adapter: LowLightLevelContext
) -> list[nn.Parameter]:
    flow = [parameter for parameter in flow_model.parameters() if parameter.requires_grad]
    condition = list(adapter.parameters())
    if not flow or not condition or any(not parameter.requires_grad for parameter in condition):
        raise RuntimeError("conditioned trainable parameter scope is invalid")
    return [*flow, *condition]
