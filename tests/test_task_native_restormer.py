from __future__ import annotations

from pathlib import Path

import pytest
import torch

from lowlight_restoration.task_native_restormer import (
    TASK_NATIVE_RESTORMER_SPECS,
    Restormer,
    load_task_native_restormer,
)


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS = ROOT / "models/restormer-task-native"


@pytest.mark.parametrize(
    ("name", "expected_tensors", "expected_parameters", "layer_norm"),
    [
        ("deraining", 494, 26_126_644, "WithBias"),
        ("real_denoising", 406, 26_111_668, "BiasFree"),
        ("gaussian_color_denoising_blind", 406, 26_111_668, "BiasFree"),
    ],
)
def test_pinned_task_native_checkpoint_loads_strictly(
    name: str,
    expected_tensors: int,
    expected_parameters: int,
    layer_norm: str,
) -> None:
    model, metadata = load_task_native_restormer(name, CHECKPOINTS)
    assert len(model.state_dict()) == expected_tensors
    assert sum(value.numel() for value in model.state_dict().values()) == expected_parameters
    assert metadata["layer_norm_type"] == layer_norm
    assert metadata["checkpoint_sha256"] == TASK_NATIVE_RESTORMER_SPECS[name].sha256


def test_task_native_restormer_forward_contract() -> None:
    model = Restormer(
        dimension=8,
        num_blocks=(1, 1, 1, 1),
        refinement_blocks=1,
        heads=(1, 1, 1, 1),
        expansion=2.0,
        layer_norm_type="BiasFree",
    ).eval()
    source = torch.rand(1, 3, 16, 24)
    with torch.inference_mode():
        restored = model(source)
    assert restored.shape == source.shape
    assert torch.isfinite(restored).all()
    with pytest.raises(ValueError, match="divisible by 8"):
        model(torch.rand(1, 3, 15, 16))


def test_unknown_task_native_checkpoint_fails_closed() -> None:
    with pytest.raises(ValueError, match="unknown task-native"):
        load_task_native_restormer("not-a-task", CHECKPOINTS)
