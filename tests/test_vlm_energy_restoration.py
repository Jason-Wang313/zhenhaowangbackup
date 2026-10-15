from __future__ import annotations

import torch

from lowlight_restoration.vlm_energy_restoration import (
    VLMConditionedEnergyRestorer,
    count_trainable_parameters,
)
from lowlight_restoration.vlm_energy_objectives import trajectory_calibrated_objectives


def tiny_model() -> VLMConditionedEnergyRestorer:
    return VLMConditionedEnergyRestorer(
        condition_dim=16,
        proposal_width=8,
        proposal_encoder_blocks=(1, 1),
        proposal_middle_blocks=1,
        energy_width=8,
        energy_blocks_per_scale=(1, 1),
        maximum_descent_steps=3,
        initial_step_size=4.0,
    )


def test_variable_step_trajectory_and_gradients() -> None:
    torch.manual_seed(1)
    model = tiny_model()
    observation = torch.randn(2, 3, 16, 16).clamp(-1, 1)
    descriptor = torch.tensor([[1, 0, 0, 1, 0], [0, 1, 0, 0, 1]], dtype=torch.float32)
    trajectory = model(
        observation,
        descriptor,
        steps=2,
        create_graph=True,
        step_scale=torch.tensor([0.75, 1.25]),
    )
    assert trajectory.output.shape == observation.shape
    assert len(trajectory.candidates) == 2
    assert len(trajectory.states) == 2
    assert len(trajectory.energies) == 2
    assert all(torch.isfinite(value).all() for value in trajectory.candidates)
    assert all(torch.isfinite(value).all() for value in trajectory.gradients)
    loss = (trajectory.output - torch.zeros_like(observation)).square().mean()
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
    assert any(value is not None and torch.isfinite(value).all() and value.abs().sum() > 0 for value in gradients)


def test_trajectory_calibrated_objective_has_second_order_gradients() -> None:
    torch.manual_seed(2)
    model = tiny_model()
    observation = torch.randn(2, 3, 16, 16).clamp(-1, 1)
    target = torch.randn(2, 3, 16, 16).clamp(-1, 1)
    descriptor = torch.tensor([[1, 0, 1, 0, 0], [0, 1, 0, 1, 1]], dtype=torch.float32)
    trajectory = model(observation, descriptor, steps=2, create_graph=True)
    terms = trajectory_calibrated_objectives(model, observation, target, descriptor, trajectory)
    loss = terms.reconstruction + terms.vector_direction + terms.vector_progress + terms.quality_ranking
    assert torch.isfinite(loss)
    loss.backward()
    assert model.energy.intro.weight.grad is not None
    assert torch.isfinite(model.energy.intro.weight.grad).all()
    assert model.energy.intro.weight.grad.abs().sum() > 0


def test_zero_steps_returns_conditioned_proposal() -> None:
    model = tiny_model()
    observation = torch.zeros(1, 3, 16, 16)
    descriptor = torch.zeros(1, 5)
    trajectory = model(observation, descriptor, steps=0, create_graph=False)
    assert trajectory.output.shape == observation.shape
    assert trajectory.output.data_ptr() == trajectory.proposal.data_ptr()
    assert not trajectory.candidates


def test_parameter_count_and_descriptor_contract() -> None:
    model = tiny_model()
    assert count_trainable_parameters(model) > 0
    bad = torch.zeros(1, 4)
    try:
        model(torch.zeros(1, 3, 16, 16), bad, steps=0, create_graph=False)
    except ValueError as error:
        assert "descriptor" in str(error)
    else:
        raise AssertionError("bad descriptor width was accepted")


def test_intrinsic_self_verification_uses_same_energy() -> None:
    model = tiny_model().eval()
    observation = torch.randn(2, 3, 16, 16).clamp(-1, 1)
    descriptor = torch.tensor([[1, 0, 0, 0, 0], [0, 1, 1, 0, 0]], dtype=torch.float32)
    first = observation.clone()
    second = (observation * 0.9).clamp(-1, 1)
    verified = model.verify_candidates(observation, descriptor, (first, second))
    assert verified.output.shape == observation.shape
    assert verified.candidate_energies.shape == (2, 2)
    assert verified.selected_index.shape == (2,)
    assert torch.allclose(
        verified.selected_energy,
        verified.candidate_energies.gather(1, verified.selected_index[:, None]).squeeze(1),
    )


def test_adaptive_inference_returns_best_executed_state() -> None:
    model = tiny_model().eval()
    observation = torch.randn(1, 3, 16, 16).clamp(-1, 1)
    descriptor = torch.tensor([[1, 1, 0, 0, 1]], dtype=torch.float32)
    with torch.no_grad():
        result = model.adaptive_inference(
            observation,
            descriptor,
            maximum_steps=3,
            minimum_steps=1,
            relative_energy_tolerance=1e6,
            patience=1,
        )
    assert result.executed_steps == 1
    assert result.candidate_energies.shape == (1, 2)
    assert result.output.shape == observation.shape
