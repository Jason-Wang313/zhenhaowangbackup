from __future__ import annotations

import json
from types import MethodType
from pathlib import Path

import torch

from lowlight_restoration.proposal_relative_energy_v2 import EnergyAndDirectControlsV2
from smoke.train_foundir_proposal_relative_energy_noise_v2_ddp import (
    descriptor_for_row,
    pairwise_ranking,
    spatial_ranking,
)


ROOT = Path(__file__).resolve().parents[1]


def test_v2_unrolls_three_spatial_energy_steps_with_second_order_gradients() -> None:
    torch.manual_seed(7)
    model = EnergyAndDirectControlsV2()
    observation = torch.rand(1, 3, 16, 16).mul(2).sub(1)
    proposal = observation.detach().clone()
    target = torch.rand_like(observation).mul(2).sub(1)
    descriptor = torch.tensor([[0.0, 0.75, 0.0, 0.0, 0.0]])
    hard_negatives = torch.rand(1, 4, 3, 16, 16).mul(2).sub(1)
    output = model(observation, proposal, target, descriptor, hard_negatives)
    assert len(output["trajectory_candidates"]) == 3
    assert output["contrastive_energy_maps"].shape == (1, 9, 1, 16, 16)
    assert output["contrastive_energies"].shape == (1, 9)
    loss = sum(candidate.mean() for candidate in output["trajectory_candidates"])
    loss = loss + output["contrastive_energies"].mean()
    loss.backward()
    assert all(parameter.grad is not None for parameter in model.energy_model.parameters())
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.energy_model.parameters())


def test_v2_adaptive_selection_can_keep_the_proposal() -> None:
    model = EnergyAndDirectControlsV2().energy_model
    observation = torch.zeros(2, 3, 16, 16)
    proposal = torch.zeros_like(observation)
    candidates = (torch.full_like(observation, 0.1), torch.full_like(observation, -0.1))
    descriptor = torch.tensor([[0.0, 0.5, 0.0, 0.0, 0.0]]).repeat(2, 1)
    def squared_candidate_score(self, observation, candidate, descriptor):
        del observation, descriptor
        return candidate.square().flatten(1).mean(1)
    model.score = MethodType(squared_candidate_score, model)
    selected, index, energies = model.select_lowest_energy(
        observation, proposal, candidates, descriptor
    )
    assert selected.shape == proposal.shape
    assert index.shape == (2,)
    assert energies.shape == (2, 3)
    assert index.tolist() == [0, 0]
    assert torch.equal(selected, proposal)


def test_v2_ranking_losses_reward_lower_error_with_lower_energy() -> None:
    energies = torch.tensor([[0.0, 1.0, 2.0]])
    qualities = torch.tensor([[0.0, 1.0, 2.0]])
    assert pairwise_ranking(energies, qualities, 0.05).item() == 0.0
    target = torch.zeros(1, 3, 16, 16)
    candidates = (target, torch.ones_like(target))
    maps = torch.stack((torch.zeros(1, 1, 16, 16), torch.ones(1, 1, 16, 16)), dim=1)
    assert spatial_ranking(maps, candidates, target, kernel=8, margin=0.02).item() == 0.0


def test_v2_config_freezes_only_the_six_ppt_corrections() -> None:
    config = json.loads((ROOT / "configs/foundir_proposal_relative_energy_noise_v2.json").read_text())
    assert config["experiment_id"] == "foundir_proposal_relative_energy_noise_v2"
    assert config["data"]["train_records"] == 54585
    assert config["data"]["validation_subset_records"] == 256
    assert config["data"]["vlm_levels_field"] == "qwen_levels_json"
    assert config["schedule"]["primary_steps"] == 8530
    assert config["hardware"]["profiles"] == {
        "H100": {"micro_batch_per_gpu": 1, "gradient_accumulation": 4},
        "H200": {"micro_batch_per_gpu": 4, "gradient_accumulation": 1},
    }
    assert config["frozen_contrastive_models"] == {
        "under_restored": "real_denoising",
        "wrong_expert": "deraining",
        "detail_damaged": "gaussian_color_denoising_blind",
        "over_restored": "repeat_adapted_noise_proposal",
        "checkpoint_dir": "models/restormer-task-native",
    }


def test_v2_uses_each_rows_frozen_vlm_severity() -> None:
    config = json.loads((ROOT / "configs/foundir_proposal_relative_energy_noise_v2.json").read_text())
    row = {
        "record_id": "example",
        "qwen_levels_json": json.dumps(
            {"blur": 1, "noise": 5, "rain": 2, "haze": 3, "low_light": 4}
        ),
    }
    descriptor = descriptor_for_row(row, config, torch.device("cpu"))
    assert torch.equal(descriptor, torch.tensor([0.0, 1.0, 0.25, 0.5, 0.75]))
