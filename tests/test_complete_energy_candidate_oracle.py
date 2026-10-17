import unittest

import torch

from smoke.evaluate_foundir_complete_energy_candidate_oracle_noise_v1_ddp import (
    deployable_candidates,
    patch_oracle,
)


class CompleteEnergyCandidateOracleTest(unittest.TestCase):
    def test_deployable_candidate_contract(self):
        proposal = torch.zeros(1, 3, 32, 32)
        fidelity = torch.full_like(proposal, .4)
        perceptual = torch.full_like(proposal, .8)
        values = deployable_candidates(proposal, fidelity, perceptual)
        self.assertEqual(list(values), [
            "proposal", "fidelity", "perceptual", "under_fidelity_0.5",
            "over_fidelity_1.25", "fidelity_0.75_perceptual_0.25",
            "fidelity_0.5_perceptual_0.5",
        ])
        self.assertTrue(torch.allclose(values["under_fidelity_0.5"], torch.full_like(proposal, .2)))
        self.assertTrue(torch.allclose(values["over_fidelity_1.25"], torch.full_like(proposal, .5)))

    def test_patch_oracle_selects_different_candidates(self):
        first = torch.zeros(1, 3, 32, 64)
        second = torch.ones_like(first)
        target = torch.cat((torch.zeros(1, 3, 32, 32), torch.ones(1, 3, 32, 32)), dim=-1)
        selected, winner = patch_oracle(torch.stack((first, second), dim=1), target, 32)
        self.assertTrue(torch.equal(selected, target))
        self.assertTrue(torch.equal(winner, torch.tensor([[[0, 1]]])))


if __name__ == "__main__":
    unittest.main()
