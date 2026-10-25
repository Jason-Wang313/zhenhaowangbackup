from __future__ import annotations

import unittest

import torch

from lowlight_restoration.pixel_refinement import ObservationAnchoredRainHead


class ObservationAnchoredRainHeadTests(unittest.TestCase):
    def test_initial_output_is_exact_observation(self) -> None:
        head = ObservationAnchoredRainHead(width=8, blocks=2)
        observation = torch.rand(2, 3, 16, 16) * 2 - 1
        guidance = torch.rand_like(observation) * 2 - 1
        self.assertTrue(torch.equal(head(observation, guidance), observation))
        self.assertTrue(torch.equal(head(observation, observation), observation))

    def test_bounded_change_and_gradient(self) -> None:
        head = ObservationAnchoredRainHead(width=8, blocks=2, maximum_residual_m11=0.2)
        observation = torch.zeros(1, 3, 16, 16)
        guidance = torch.ones_like(observation)
        target = torch.rand_like(observation)
        loss = (head(observation, guidance) - target).square().mean()
        loss.backward()
        self.assertIsNotNone(head.ending.weight.grad)
        self.assertGreater(float(head.ending.weight.grad.abs().sum()), 0)
        with torch.no_grad():
            head.ending.bias.fill_(10)
            output = head(observation, guidance)
        self.assertLessEqual(float((output - observation).abs().max()), 0.200001)

    def test_requires_aligned_rgb(self) -> None:
        head = ObservationAnchoredRainHead(width=8, blocks=2)
        observation = torch.zeros(1, 3, 16, 16)
        with self.assertRaises(ValueError):
            head(observation, torch.zeros(1, 3, 8, 8))


if __name__ == "__main__":
    unittest.main()
