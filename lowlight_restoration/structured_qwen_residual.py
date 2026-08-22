"""Structured, monotone Qwen tone residual for Gate 3 correction 1.

The module deliberately cannot restore spatial detail.  It maps only the
canonical five-level posterior to bounded achromatic gain/gamma residuals.
Entropy, posterior margin, and distance from training support can only shrink
the correction.  The input-derived measured baseline residualizes the Qwen
severity before the correction is applied.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F


STRUCTURED_QWEN_DIM = 8


def structured_qwen(condition: Sequence[float]) -> np.ndarray:
    """Return and validate posterior, expected level, entropy, and margin."""

    value = np.asarray(condition, dtype=np.float32)[:STRUCTURED_QWEN_DIM]
    if value.shape != (STRUCTURED_QWEN_DIM,) or not np.isfinite(value).all():
        raise ValueError("structured Qwen condition must contain eight finite values")
    if np.any(value[:5] < 0.0) or not np.isclose(value[:5].sum(), 1.0, atol=2e-5):
        raise ValueError("Qwen posterior is invalid")
    expected = float(value[:5] @ (np.arange(1, 6, dtype=np.float32) / 5.0))
    if not np.isclose(expected, value[5], atol=2e-5):
        raise ValueError("Qwen expected severity disagrees with its posterior")
    if not 0.0 <= float(value[6]) <= 1.0 or not 0.0 <= float(value[7]) <= 1.0:
        raise ValueError("Qwen uncertainty fields are outside [0,1]")
    return value.copy()


def fit_measured_posterior_ridge(
    measured: np.ndarray,
    posterior: np.ndarray,
    penalty: float,
) -> dict[str, np.ndarray | float]:
    """Fit a strongly regularized measured-statistics posterior predictor."""

    x = np.asarray(measured, dtype=np.float64)
    y = np.asarray(posterior, dtype=np.float64)
    if x.ndim != 2 or y.shape != (x.shape[0], 5) or x.shape[0] < 2:
        raise ValueError("invalid measured-posterior ridge inputs")
    mean = x.mean(axis=0)
    scale = np.maximum(x.std(axis=0), 1e-4)
    standardized = (x - mean) / scale
    centered = y - y.mean(axis=0)
    gram = standardized.T @ standardized
    coefficient = np.linalg.solve(
        gram + np.eye(gram.shape[0], dtype=np.float64) * float(penalty),
        standardized.T @ centered,
    )
    return {
        "mean": mean.astype(np.float32),
        "scale": scale.astype(np.float32),
        "coefficient": coefficient.astype(np.float32),
        "intercept": y.mean(axis=0).astype(np.float32),
    }


def predict_measured_posterior(measured: np.ndarray, ridge: Mapping[str, object]) -> np.ndarray:
    x = np.asarray(measured, dtype=np.float32)
    mean = np.asarray(ridge["mean"], dtype=np.float32)
    scale = np.asarray(ridge["scale"], dtype=np.float32)
    coefficient = np.asarray(ridge["coefficient"], dtype=np.float32)
    intercept = np.asarray(ridge["intercept"], dtype=np.float32)
    if (
        x.shape[-1] != mean.size or mean.shape != scale.shape
        or coefficient.shape != (mean.size, 5) or intercept.shape != (5,)
    ):
        raise ValueError("measured posterior ridge dimensions changed")
    prediction = intercept + ((x - mean) / scale) @ coefficient
    prediction = np.clip(prediction, 1e-4, None)
    return (prediction / prediction.sum(axis=-1, keepdims=True)).astype(np.float32)


def support_reference(structured: np.ndarray, group_ids: Sequence[str], quantile: float) -> dict[str, object]:
    """Freeze diagonal scaling and a leave-group-out nearest-neighbour radius."""

    matrix = np.asarray(structured, dtype=np.float32)
    groups = np.asarray(group_ids)
    if matrix.ndim != 2 or matrix.shape[1] != STRUCTURED_QWEN_DIM or groups.shape != (matrix.shape[0],):
        raise ValueError("invalid support reference inputs")
    mean = matrix.mean(axis=0)
    scale = np.maximum(matrix.std(axis=0), 1e-4)
    standardized = (matrix - mean) / scale
    distances = []
    for index in range(len(matrix)):
        eligible = groups != groups[index]
        if not eligible.any():
            raise ValueError("support distance requires at least two groups")
        delta = standardized[eligible] - standardized[index]
        distances.append(float(np.sqrt(np.square(delta).mean(axis=1)).min()))
    radius = max(float(np.quantile(distances, float(quantile))), 1e-3)
    entropy_deficit_anchor = max(float(np.median(1.0 - matrix[:, 6])), 1e-3)
    margin_anchor = max(float(np.median(matrix[:, 7])), 1e-3)
    return {
        "mean": mean.astype(np.float32),
        "scale": scale.astype(np.float32),
        "radius": radius,
        "entropy_deficit_anchor": entropy_deficit_anchor,
        "margin_anchor": margin_anchor,
        "training": matrix.astype(np.float32),
        "training_group_ids": [str(value) for value in group_ids],
    }


def support_shrink(
    structured: np.ndarray,
    reference: Mapping[str, object],
    *,
    query_group_ids: Sequence[str] | None = None,
    exclude_same_group: bool = False,
) -> np.ndarray:
    """Return a [0,1] shrink factor based on nearest training support."""

    query = np.asarray(structured, dtype=np.float32)
    if query.ndim == 1:
        query = query[None]
    training = np.asarray(reference["training"], dtype=np.float32)
    mean = np.asarray(reference["mean"], dtype=np.float32)
    scale = np.asarray(reference["scale"], dtype=np.float32)
    if query.ndim != 2 or query.shape[1] != STRUCTURED_QWEN_DIM or training.shape[1] != query.shape[1]:
        raise ValueError("support query dimensions changed")
    query_z = (query - mean) / scale
    training_z = (training - mean) / scale
    result = []
    radius = float(reference["radius"])
    if exclude_same_group and (query_group_ids is None or len(query_group_ids) != len(query)):
        raise ValueError("group-excluded support requires one group ID per query")
    training_groups = np.asarray(reference.get("training_group_ids", []))
    for index, value in enumerate(query_z):
        eligible = np.ones(len(training_z), dtype=bool)
        if exclude_same_group:
            eligible = training_groups != str(query_group_ids[index])
        if not eligible.any():
            raise ValueError("support query has no eligible training group")
        distance = float(np.sqrt(np.square(training_z[eligible] - value).mean(axis=1)).min())
        result.append(1.0 / (1.0 + (distance / radius) ** 4))
    return np.asarray(result, dtype=np.float32)


class StructuredQwenToneResidual(nn.Module):
    """Six-parameter monotone exposure/gamma residual with fixed shrinkage."""

    def __init__(
        self,
        *,
        maximum_gain_parameter_delta: float,
        maximum_gamma_parameter_delta: float,
        entropy_deficit_anchor: float,
        margin_anchor: float,
        posterior_center: float,
        maximum_knot_adjustment: float,
        initial_raw_slope: float = -2.0,
        initial_raw_residual_scale: float = -2.0,
    ) -> None:
        super().__init__()
        self.maximum_gain_parameter_delta = float(maximum_gain_parameter_delta)
        self.maximum_gamma_parameter_delta = float(maximum_gamma_parameter_delta)
        self.entropy_deficit_anchor = float(entropy_deficit_anchor)
        self.margin_anchor = float(margin_anchor)
        self.posterior_center = float(posterior_center)
        self.maximum_knot_adjustment = float(maximum_knot_adjustment)
        self.raw_gain_slope = nn.Parameter(torch.full((3,), float(initial_raw_slope)))
        self.raw_gamma_slope = nn.Parameter(torch.full((3,), float(initial_raw_slope)))
        self.raw_knot_adjustment = nn.Parameter(torch.zeros(5))
        self.raw_residual_scale = nn.Parameter(torch.tensor(float(initial_raw_residual_scale)))

    def uncertainty_shrink(self, structured: Tensor) -> Tensor:
        entropy_reliability = ((1.0 - structured[:, 6]) / self.entropy_deficit_anchor).clamp(0.0, 1.0)
        margin_reliability = (structured[:, 7] / self.margin_anchor).clamp(0.0, 1.0)
        # A non-zero floor avoids declaring a calibrated high-entropy posterior
        # useless while still shrinking uncertain examples substantially.
        return 0.25 + 0.75 * torch.sqrt((entropy_reliability * margin_reliability).clamp_min(0.0))

    def monotone_knots(self) -> Tensor:
        adjustment = torch.tanh(self.raw_knot_adjustment) * self.maximum_knot_adjustment
        first = adjustment.new_tensor(0.2) + adjustment[0]
        intervals = adjustment.new_full((4,), 0.2) + adjustment[1:]
        return torch.cat((first[None], first + torch.cumsum(intervals, dim=0)))

    def forward(self, structured: Tensor, measured_posterior: Tensor, support: Tensor) -> Tensor:
        if structured.ndim != 2 or structured.shape[1] != STRUCTURED_QWEN_DIM:
            raise ValueError("structured Qwen tensor has the wrong shape")
        if measured_posterior.shape != (structured.shape[0], 5) or support.shape != (structured.shape[0],):
            raise ValueError("measured posterior or support shrink has the wrong shape")
        posterior = structured[:, :5]
        levels = posterior.new_tensor([0.2, 0.4, 0.6, 0.8, 1.0])
        expected = posterior @ levels
        reliability = self.uncertainty_shrink(structured) * support.clamp(0.0, 1.0)
        residual_score = ((posterior - measured_posterior) * self.monotone_knots()).sum(dim=1)
        signal = expected - self.posterior_center + F.softplus(self.raw_residual_scale) * residual_score
        signal = signal.clamp(-0.30, 0.30) * reliability
        # Gain must increase and gamma must decrease as severity increases.
        gain = torch.tanh(signal[:, None] * F.softplus(self.raw_gain_slope))
        gamma = -torch.tanh(signal[:, None] * F.softplus(self.raw_gamma_slope))
        gain = gain * self.maximum_gain_parameter_delta
        gamma = gamma * self.maximum_gamma_parameter_delta
        return torch.cat((gain, gamma, torch.zeros_like(gain)), dim=1)
