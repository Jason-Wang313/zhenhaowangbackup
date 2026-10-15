import pytest
import torch

from lowlight_restoration.stage_trajectory_compatibility import (
    StageTrajectoryCompatibilityAdapter,
    parameter_count,
)


def test_exact_identity_initialization_and_gradients() -> None:
    torch.manual_seed(7)
    model = StageTrajectoryCompatibilityAdapter(channels=16, blocks=2)
    states = [torch.rand(2, 3, 24, 32) for _ in range(4)]
    presence = torch.tensor([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]])
    output = model(*states, presence)
    assert torch.equal(output, states[-1])
    output.mean().backward()
    assert model.head.weight.grad is not None
    assert torch.isfinite(model.head.weight.grad).all()
    assert model.head.weight.grad.abs().sum() > 0


def test_input_contract_and_parameter_bound() -> None:
    model = StageTrajectoryCompatibilityAdapter()
    assert 200_000 < parameter_count(model) < 500_000
    image = torch.zeros(1, 3, 16, 16)
    with pytest.raises(ValueError):
        model(image, image, image, image, torch.zeros(1, 2))
