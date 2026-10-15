from __future__ import annotations

import pytest
import torch
from torch import nn

from lowlight_restoration.blur_restormer_refinement import (
    BaseLQRestormerRefiner,
    BaseLQRestormerExpert,
    _convert_mlx_restormer_state_dict,
    base_lq_restormer_parameter_count,
)


class ResidualCore(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.output = nn.Conv2d(3, 3, 1, bias=False)
        nn.init.zeros_(self.output.weight)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.output(value)


def test_base_lq_wrapper_starts_as_exact_base_identity() -> None:
    model = BaseLQRestormerRefiner(ResidualCore())
    base = torch.rand(2, 3, 16, 20)
    lq = torch.rand_like(base)
    assert torch.equal(model(base, lq), base)
    assert base_lq_restormer_parameter_count(model) == sum(
        parameter.numel() for parameter in model.parameters()
    )


def test_motion_expert_starts_from_exact_raw_lq_input() -> None:
    model = BaseLQRestormerExpert(ResidualCore())
    base = torch.rand(2, 3, 16, 20)
    lq = torch.rand_like(base)
    assert torch.equal(model(base, lq), lq)
    assert base_lq_restormer_parameter_count(model) == sum(
        parameter.numel() for parameter in model.parameters()
    )


def test_mlx_conv_layout_conversion_uses_module_type_not_shape_guessing() -> None:
    class TinyCore(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = nn.Conv2d(3, 4, 3, bias=False)
            self.scale = nn.Parameter(torch.arange(4, dtype=torch.float32))

    model = TinyCore()
    expected = model.state_dict()
    source = {
        "proj.weight": expected["proj.weight"].permute(0, 2, 3, 1).contiguous(),
        "scale": expected["scale"].clone(),
    }
    converted, convolution_count = _convert_mlx_restormer_state_dict(model, source)
    assert convolution_count == 1
    assert torch.equal(converted["proj.weight"], expected["proj.weight"])
    assert torch.equal(converted["scale"], expected["scale"])


def test_fusion_is_created_on_core_device_and_dtype() -> None:
    core = ResidualCore().to(dtype=torch.float64)
    model = BaseLQRestormerRefiner(core)
    assert model.fusion.weight.device == core.output.weight.device
    assert model.fusion.weight.dtype == core.output.weight.dtype


def test_base_lq_wrapper_can_learn_from_lq_after_residual_head_moves() -> None:
    model = BaseLQRestormerRefiner(ResidualCore())
    with torch.no_grad():
        model.restormer.output.weight.fill_(0.1)
    base = torch.rand(1, 3, 8, 8)
    lq = torch.rand_like(base)
    model(base, lq).mean().backward()
    assert model.fusion.weight.grad is not None
    assert float(model.fusion.weight.grad[:, 3:].abs().sum()) > 0.0


@pytest.mark.parametrize(
    ("base", "lq"),
    [
        (torch.rand(3, 8, 8), torch.rand(3, 8, 8)),
        (torch.rand(1, 3, 8, 8), torch.rand(1, 3, 8, 9)),
        (torch.rand(1, 1, 8, 8), torch.rand(1, 1, 8, 8)),
    ],
)
def test_base_lq_wrapper_rejects_unaligned_inputs(
    base: torch.Tensor, lq: torch.Tensor
) -> None:
    model = BaseLQRestormerRefiner(ResidualCore())
    with pytest.raises(ValueError, match="aligned BCHW RGB"):
        model(base, lq)
