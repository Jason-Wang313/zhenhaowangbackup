import unittest

import torch

from lowlight_restoration.spatial_residual_amplitude import (
    SpatialResidualAmplitudePredictor,
    analytic_patch_amplitude,
    apply_patch_amplitude,
)


class SpatialResidualAmplitudeTest(unittest.TestCase):
    def test_initial_identity_and_shapes(self):
        model = SpatialResidualAmplitudePredictor(channels=16, blocks=1)
        observation = torch.rand(2, 3, 16, 24)
        proposal = torch.rand_like(observation)
        fidelity = torch.rand_like(observation)
        amplitude = model(observation, proposal, fidelity)
        self.assertEqual(amplitude.shape, (2, 1, 2, 3))
        self.assertTrue(torch.equal(amplitude, torch.ones_like(amplitude)))
        self.assertTrue(torch.allclose(apply_patch_amplitude(proposal, fidelity, amplitude), fidelity))

    def test_analytic_teacher_recovers_known_amplitude(self):
        proposal = torch.zeros(1, 3, 16, 16)
        fidelity = torch.full_like(proposal, .25)
        target = torch.full_like(proposal, .375)
        amplitude, confidence = analytic_patch_amplitude(proposal, fidelity, target)
        self.assertTrue(torch.allclose(amplitude, torch.full_like(amplitude, 1.5)))
        self.assertTrue(torch.all(confidence > 0))
        self.assertTrue(torch.allclose(apply_patch_amplitude(proposal, fidelity, amplitude), target))


if __name__ == "__main__":
    unittest.main()
