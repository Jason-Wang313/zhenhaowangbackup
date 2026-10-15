from __future__ import annotations

import unittest

import torch

from lowlight_restoration.compositional_ebt import (
    NUM_ENERGY_COMPONENTS,
    RestorationEnergyTransformer,
    compositional_descriptor,
    count_trainable_parameters,
)


class CompositionalEBTTests(unittest.TestCase):
    def test_descriptor_has_atomic_and_normalized_pair_terms(self) -> None:
        q = torch.tensor([[1.0, 1.0, 0.0, 0.0, 1.0]])
        descriptor = compositional_descriptor(q)
        self.assertEqual(tuple(descriptor.shape), (1, NUM_ENERGY_COMPONENTS))
        self.assertEqual(float(descriptor[0, :6].sum()), 4.0)
        self.assertAlmostEqual(float(descriptor[0, 6:].sum()), 1.0, places=6)

    def test_plain_and_compositional_capacity_is_exactly_matched(self) -> None:
        torch.manual_seed(7)
        plain = RestorationEnergyTransformer(mode="plain")
        torch.manual_seed(7)
        compositional = RestorationEnergyTransformer(mode="compositional")
        self.assertEqual(
            count_trainable_parameters(plain),
            count_trainable_parameters(compositional),
        )
        self.assertEqual(set(plain.state_dict()), set(compositional.state_dict()))

    def test_candidate_gradient_and_second_order_parameter_gradients(self) -> None:
        torch.manual_seed(11)
        model = RestorationEnergyTransformer(
            image_size=16,
            patch_size=4,
            hidden_size=32,
            depth=1,
            num_heads=4,
            zero_init_energy_head=False,
        )
        observation = torch.rand(2, 3, 16, 16) * 2 - 1
        target = torch.rand(2, 3, 16, 16) * 2 - 1
        candidate = observation.clone().detach()
        q = torch.tensor([[1.0, 1.0, 0.0, 0.0, 0.0]]).expand(2, -1)
        step = model.descent_step(
            observation,
            candidate,
            q,
            step_size=1e-2,
            create_graph=True,
            clamp=None,
        )
        self.assertTrue(torch.isfinite(step.candidate_gradient).all())
        self.assertGreater(float(step.candidate_gradient.abs().max()), 0.0)
        loss = torch.nn.functional.smooth_l1_loss(step.candidate, target)
        loss.backward()
        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.requires_grad and parameter.grad is not None
        ]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))


if __name__ == "__main__":
    unittest.main()
