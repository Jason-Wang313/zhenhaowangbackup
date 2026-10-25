"""FP32 trainable state and a shared train/inference FLUX endpoint."""

from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor, nn


def promote_lora_to_fp32(model: nn.Module) -> list[nn.Parameter]:
    parameters = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if ".lora_A." not in name and ".lora_B." not in name:
            raise ValueError(f"unexpected trainable parameter: {name}")
        parameter.data = parameter.data.float()
        parameters.append(parameter)
    if not parameters:
        raise ValueError("no trainable LoRA parameters")
    return parameters


def assert_fp32_optimizer(optimizer: torch.optim.Optimizer) -> None:
    if not optimizer.state:
        raise RuntimeError("optimizer has not taken a step")
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            if parameter.dtype != torch.float32:
                raise RuntimeError("trainable parameter is not FP32")
            state = optimizer.state.get(parameter, {})
            for name in ("exp_avg", "exp_avg_sq"):
                if name not in state or state[name].dtype != torch.float32:
                    raise RuntimeError(f"optimizer {name} is missing or not FP32")


def endpoint_from_latent(
    model: nn.Module,
    autoencoder: nn.Module,
    degraded: Tensor,
    context: Tensor,
    context_ids: Tensor,
    pack_image: Callable,
    reference_time: int = 10,
) -> dict[str, Tensor]:
    """Keep the observation latent and subtraction FP32 in every caller.

    Model matmuls and decoder convolutions use autocast on CUDA. Only the
    model's token input is cast; the original observation is never replaced.
    """
    if degraded.dtype != torch.float32:
        raise ValueError("endpoint requires the original FP32 encoded observation")
    tokens, ids = pack_image(degraded.to(torch.bfloat16))
    reference_ids = ids.clone()
    reference_ids[..., 0] = reference_time
    timestep = torch.ones(degraded.shape[0], device=degraded.device, dtype=torch.bfloat16)
    with torch.autocast(degraded.device.type, dtype=torch.bfloat16, enabled=degraded.is_cuda):
        prediction = model(
            x=torch.cat((tokens, tokens), dim=1),
            x_ids=torch.cat((ids, reference_ids), dim=1),
            timesteps=timestep,
            ctx=context,
            ctx_ids=context_ids,
            guidance=None,
        )[:, : tokens.shape[1]]
    batch, channels, height, width = degraded.shape
    if prediction.shape != (batch, height * width, channels):
        raise RuntimeError("unexpected FLUX velocity shape")
    # This subtraction must not be moved back into a BF16-only path.
    clean = degraded - prediction.float().transpose(1, 2).reshape_as(degraded)
    with torch.autocast(degraded.device.type, dtype=torch.bfloat16, enabled=degraded.is_cuda):
        decoded = autoencoder.decode(clean)
    return {"degraded": degraded, "prediction": prediction, "predicted_clean": clean,
            "decoded": decoded.float().clamp(-1, 1)}


def restore_endpoint(model, autoencoder, observation, context, context_ids, pack_image, reference_time=10):
    with torch.no_grad(), torch.autocast(observation.device.type, enabled=False):
        degraded = autoencoder.encode(observation.float()).float()
    return endpoint_from_latent(model, autoencoder, degraded, context, context_ids, pack_image, reference_time)
