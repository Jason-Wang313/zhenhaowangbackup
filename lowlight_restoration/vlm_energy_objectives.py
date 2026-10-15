"""Objectives for a trajectory-calibrated restoration energy landscape."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F

from .vlm_energy_restoration import RestorationTrajectory, VLMConditionedEnergyRestorer


@dataclass(frozen=True)
class EnergyObjectiveTerms:
    reconstruction: Tensor
    vector_direction: Tensor
    vector_progress: Tensor
    quality_ranking: Tensor
    energy_target: Tensor
    energy_candidates: Tensor
    quality_candidates: Tensor


def charbonnier(prediction: Tensor, target: Tensor, epsilon: float = 1e-3) -> Tensor:
    return torch.sqrt((prediction - target).square() + epsilon * epsilon).mean()


def multi_scale_reconstruction(prediction: Tensor, target: Tensor) -> Tensor:
    loss = prediction.new_zeros(())
    weight = 1.0
    current_prediction, current_target = prediction, target
    for _ in range(3):
        loss = loss + weight * charbonnier(current_prediction, current_target)
        if min(current_prediction.shape[-2:]) < 4:
            break
        current_prediction = F.avg_pool2d(current_prediction, 2)
        current_target = F.avg_pool2d(current_target, 2)
        weight *= 0.5
    return loss


def edge_reconstruction(prediction: Tensor, target: Tensor) -> Tensor:
    pred_x = prediction[..., :, 1:] - prediction[..., :, :-1]
    true_x = target[..., :, 1:] - target[..., :, :-1]
    pred_y = prediction[..., 1:, :] - prediction[..., :-1, :]
    true_y = target[..., 1:, :] - target[..., :-1, :]
    return charbonnier(pred_x, true_x) + charbonnier(pred_y, true_y)


def _per_sample_mse(prediction: Tensor, target: Tensor) -> Tensor:
    return (prediction - target).square().flatten(1).mean(dim=1)


def _vector_losses(trajectory: RestorationTrajectory, target: Tensor) -> tuple[Tensor, Tensor]:
    if not trajectory.states:
        zero = target.new_zeros(())
        return zero, zero
    directions = []
    progresses = []
    for state, gradient, step_size in zip(
        trajectory.states, trajectory.gradients, trajectory.step_sizes, strict=True
    ):
        desired = target - state
        update = -step_size[:, None, None, None] * gradient
        desired_flat, update_flat = desired.flatten(1), update.flatten(1)
        cosine = F.cosine_similarity(update_flat, desired_flat, dim=1, eps=1e-8)
        directions.append((1.0 - cosine).mean())
        # Match only the scalar amount of useful progress.  Exact residual
        # matching would force every energy step to solve the image at once
        # and would destroy test-time scaling.
        desired_norm = desired_flat.norm(dim=1).clamp_min(1e-8)
        signed_progress = (update_flat * desired_flat).sum(dim=1) / desired_norm.square()
        target_fraction = torch.full_like(signed_progress, 1.0 / len(trajectory.states))
        progresses.append(F.smooth_l1_loss(signed_progress, target_fraction))
    return torch.stack(directions).mean(), torch.stack(progresses).mean()


def _quality_ranking_loss(energies: Tensor, qualities: Tensor, margin_scale: float) -> Tensor:
    """Require lower energy for every measurably better candidate."""

    # i is better than j when its detached reconstruction error is lower.
    better = qualities[:, :, None] + 1e-8 < qualities[:, None, :]
    quality_gap = (qualities[:, None, :] - qualities[:, :, None]).detach().clamp_min(0)
    margin = float(margin_scale) * quality_gap / qualities.detach().mean(dim=1, keepdim=True)[:, :, None].clamp_min(1e-6)
    violation = F.relu(margin + energies[:, :, None] - energies[:, None, :])
    if not bool(better.any()):
        return energies.new_zeros(())
    return violation[better].mean()


def trajectory_calibrated_objectives(
    model: VLMConditionedEnergyRestorer,
    observation: Tensor,
    target: Tensor,
    descriptor: Tensor,
    trajectory: RestorationTrajectory,
    *,
    ranking_margin_scale: float = 0.1,
    compute_quality_ranking: bool = True,
) -> EnergyObjectiveTerms:
    """Supervise restoration, vector field, and intrinsic verification jointly."""

    if compute_quality_ranking:
        condition = model.condition_encoder(descriptor)
        candidate_list = (
            [target, *trajectory.states, trajectory.candidates[-1]]
            if trajectory.candidates
            else [target, trajectory.proposal]
        )
        # The trajectory already evaluated proposal and every pre-update state.
        # Reuse those graph-connected energies rather than repeating an expensive
        # energy-model forward for every candidate.  Only GT and the final state
        # have not yet been scored.  This preserves the exact candidate ordering:
        # [GT, initial-state, step-1, ..., final-step].  The initial state
        # equals the proposal unless Langevin/replay perturbation is active.
        if trajectory.candidates:
            target_energy = model.energy(observation, target, condition)
            final_energy = model.energy(observation, trajectory.candidates[-1], condition)
            energies = torch.stack(
                [target_energy, *trajectory.energies, final_energy], dim=1
            )
        else:
            energies = torch.stack(
                [
                    model.energy(observation, target, condition),
                    model.energy(observation, trajectory.proposal, condition),
                ],
                dim=1,
            )
        qualities = torch.stack([_per_sample_mse(candidate, target) for candidate in candidate_list], dim=1)
        ranking = _quality_ranking_loss(energies, qualities, ranking_margin_scale)
        energy_target = energies[:, 0].mean()
    else:
        energies = target.new_empty((target.shape[0], 0))
        qualities = target.new_empty((target.shape[0], 0))
        ranking = target.new_zeros(())
        energy_target = target.new_zeros(())
    direction, progress = _vector_losses(trajectory, target)
    return EnergyObjectiveTerms(
        reconstruction=multi_scale_reconstruction(trajectory.output, target),
        vector_direction=direction,
        vector_progress=progress,
        quality_ranking=ranking,
        energy_target=energy_target,
        energy_candidates=energies,
        quality_candidates=qualities,
    )
