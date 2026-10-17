import unittest

import torch

from lowlight_restoration.proposal_residual_basis import ProposalResidualBasisRestormer
from lowlight_restoration.task_native_restormer import Restormer


class ProposalResidualBasisTest(unittest.TestCase):
    def test_initialization_and_forward_contract(self):
        source = Restormer(layer_norm_type="BiasFree")
        model = ProposalResidualBasisRestormer()
        model.initialize_from_proposal(source.state_dict())
        patch = model.state_dict()["patch_embed.proj.weight"]
        self.assertTrue(torch.count_nonzero(patch[:, :3]) == 0)
        self.assertTrue(torch.equal(patch[:, 3:6], source.state_dict()["patch_embed.proj.weight"]))
        self.assertTrue(torch.count_nonzero(patch[:, 6:]) == 0)
        observation = torch.rand(1, 3, 16, 16)
        proposal = torch.rand_like(observation)
        fidelity, perception = model(observation, proposal)
        self.assertEqual(fidelity.shape, proposal.shape)
        self.assertEqual(perception.shape, proposal.shape)
        self.assertTrue(bool(((0 <= fidelity) & (fidelity <= 1)).all()))
        self.assertTrue(bool(((0 <= perception) & (perception <= 1)).all()))


if __name__ == "__main__":
    unittest.main()
