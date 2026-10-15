from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch import nn

import smoke.train_foundir_noise_family_cascade_terminal_ddp as trainer


ROOT = Path(__file__).resolve().parents[1]


class ResidualToy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.output = nn.Conv2d(3, 3, 3, padding=1, bias=False)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return image + self.output(image)


def test_frozen_terminal_config_and_manifest_pass() -> None:
    path = ROOT / "configs/foundir_noise_family_cascade_terminal_v1.json"
    config = trainer.validate_config(path)
    rows = trainer.load_rows(config)
    assert len(rows["train"]) == 2048
    assert len(rows["validation"]) == 256


def test_exposure_initialization_is_exact_identity() -> None:
    model = ResidualToy()
    trainer.initialize_identity_exposure(model)
    image = torch.rand(2, 3, 16, 16)
    assert torch.equal(model(image), image)


def test_cascade_freezes_noise_and_trains_exposure_only() -> None:
    noise = ResidualToy()
    exposure = ResidualToy()
    trainer.initialize_identity_exposure(exposure)
    cascade = trainer.NoiseExposureCascade(noise, exposure)
    assert not any(parameter.requires_grad for parameter in cascade.noise.parameters())
    assert all(parameter.requires_grad for parameter in cascade.exposure.parameters())
    cascade.train()
    assert cascade.noise.training is False
    assert cascade.exposure.training is True


def test_terminal_contract_rejects_method_drift(tmp_path: Path) -> None:
    source = ROOT / "configs/foundir_noise_family_cascade_terminal_v1.json"
    config = json.loads(source.read_text())
    config["method"] = "single_shared_restormer"
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    with pytest.raises(RuntimeError, match="contract changed"):
        trainer.validate_config(path)
