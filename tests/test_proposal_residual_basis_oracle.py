import unittest

import torch

from smoke.evaluate_foundir_proposal_residual_basis_noise_v1_oracle_ddp import (
    patch_alpha_candidate,
    summarize,
)


class ProposalResidualBasisOracleTest(unittest.TestCase):
    def test_patch_oracle_recovers_known_amplitude(self):
        proposal = torch.zeros(1, 3, 8, 8)
        fidelity = torch.full_like(proposal, .25)
        target = torch.full_like(proposal, .375)
        candidate = patch_alpha_candidate(proposal, fidelity, target, 8, False)
        self.assertTrue(torch.allclose(candidate, target))

    def test_summary_detects_joint_gate(self):
        rows = [
            {"psnr": {"proposal": 40., "candidate": 40.4}, "lpips": {"proposal": .1, "candidate": .08}},
            {"psnr": {"proposal": 41., "candidate": 41.4}, "lpips": {"proposal": .1, "candidate": .08}},
        ]
        result = summarize(rows, ["proposal", "candidate"])
        self.assertTrue(result["any_fixed_candidate_simultaneous_gate_pass"])
        self.assertTrue(result["any_oracle_simultaneous_gate_pass"])


if __name__ == "__main__":
    unittest.main()
