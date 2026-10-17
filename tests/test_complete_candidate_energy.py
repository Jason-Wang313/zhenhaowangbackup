import json
import unittest
from pathlib import Path

import torch

from lowlight_restoration.complete_candidate_energy import (
    CompleteCandidateEnergy,
    compose_selected_candidates,
    relative_risk_targets,
    select_candidates_fail_closed,
    tie_aware_gap_ranking,
)


ROOT = Path(__file__).resolve().parents[1]


class CompleteCandidateEnergyTest(unittest.TestCase):
    def test_shared_scorer_shapes_and_initial_fail_closed_selection(self):
        model = CompleteCandidateEnergy(widths=(16, 24, 32, 40, 48), blocks=1)
        observation = torch.rand(2, 3, 64, 96)
        proposal = torch.rand_like(observation)
        candidates = torch.rand(2, 7, 3, 64, 96)
        severity = torch.rand(2, 5)
        risks = model(observation, proposal, candidates, severity)
        self.assertEqual(risks.shape, (2, 7, 3, 2, 3))
        selected, accepted = select_candidates_fail_closed(
            risks, safe_index=1, uncertainty_scale=.25,
            minimum_mse_advantage=0, maximum_relative_perceptual_risk=.25,
        )
        self.assertTrue(torch.equal(selected, torch.ones_like(selected)))
        self.assertFalse(bool(accepted.any()))

    def test_relative_targets_and_composition(self):
        target = torch.zeros(1, 3, 64, 64)
        candidates = torch.stack((
            torch.full_like(target, .2), torch.full_like(target, .1), torch.zeros_like(target)
        ), dim=1)
        spatial_lpips = candidates.square().mean(2)
        mse, perceptual = relative_risk_targets(
            candidates, target, spatial_lpips, safe_index=1, patch_size=32
        )
        self.assertTrue(torch.equal(mse[:, 1], torch.zeros_like(mse[:, 1])))
        self.assertTrue(torch.equal(perceptual[:, 1], torch.zeros_like(perceptual[:, 1])))
        selected = torch.tensor([[[2, 1], [1, 2]]])
        output = compose_selected_candidates(candidates, selected, patch_size=32)
        self.assertTrue(torch.equal(output[..., :32, :32], target[..., :32, :32]))
        self.assertTrue(torch.equal(output[..., :32, 32:], candidates[:, 1, :, :32, 32:]))

    def test_ranking_prefers_correct_order_and_ignores_ties(self):
        target = torch.tensor([[[[0.0]], [[1.0]], [[1.005]]]])
        good = target.clone()
        bad = -target
        self.assertLess(
            float(tie_aware_gap_ranking(good, target, tie_epsilon=.01)),
            float(tie_aware_gap_ranking(bad, target, tie_epsilon=.01)),
        )

    def test_config_contract(self):
        config = json.loads((ROOT / "configs/foundir_complete_candidate_energy_noise_v1.json").read_text())
        self.assertEqual(config["safe_index"], 1)
        self.assertEqual(config["candidate_names"][1], "fidelity")
        self.assertEqual(config["model"]["patch_size"], 32)
        self.assertEqual(config["hardware"]["world_size"], 8)
        self.assertFalse(any(config["access"][key] for key in (
            "internal_test_accessed", "official_foundir_test_accessed",
            "sealed_internal_accessed", "frozen_660_accessed",
        )))


if __name__ == "__main__":
    unittest.main()
