import unittest

import torch

from lowlight_restoration.local_action_energy import (
    LocalActionEnergy,
    apply_selected_actions,
    local_candidate_cost_targets,
    select_actions_fail_closed,
    soft_action_output,
)


CANDIDATES = (0.0, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
SAFE = 3


class LocalActionEnergyTest(unittest.TestCase):
    def test_initial_model_is_exact_safe_fallback(self):
        model = LocalActionEnergy(candidates=CANDIDATES, safe_index=SAFE, widths=(16, 24, 32), blocks=1)
        observation = torch.rand(2, 3, 16, 24)
        proposal = torch.rand_like(observation)
        fidelity = torch.rand_like(observation)
        costs = model(observation, proposal, fidelity)
        selected, accepted = select_actions_fail_closed(costs, safe_index=SAFE, decision_margin=0.05)
        output = apply_selected_actions(proposal, fidelity, selected, CANDIDATES)
        self.assertEqual(costs.shape, (2, 7, 2, 3))
        self.assertTrue(torch.equal(selected, torch.full_like(selected, SAFE)))
        self.assertFalse(bool(accepted.any()))
        self.assertTrue(torch.equal(output, fidelity))

    def test_targets_recover_known_local_action(self):
        proposal = torch.zeros(1, 3, 16, 16)
        fidelity = torch.full_like(proposal, 0.25)
        target = torch.full_like(proposal, 0.375)
        costs, best = local_candidate_cost_targets(proposal, fidelity, target, CANDIDATES, SAFE)
        self.assertTrue(torch.equal(best, torch.full_like(best, 5)))
        self.assertTrue(torch.all(costs[:, SAFE] == 0))

    def test_soft_action_is_differentiable(self):
        proposal = torch.zeros(1, 3, 8, 8)
        fidelity = torch.full_like(proposal, 0.25)
        costs = torch.zeros(1, 7, 1, 1, requires_grad=True)
        output, alpha = soft_action_output(proposal, fidelity, costs, CANDIDATES, temperature=0.25)
        output.mean().backward()
        self.assertEqual(alpha.shape, (1, 1, 1, 1))
        self.assertIsNotNone(costs.grad)
        self.assertTrue(bool(torch.isfinite(costs.grad).all()))


if __name__ == "__main__":
    unittest.main()
