from types import MethodType

import unittest
import torch

from lowlight_restoration.proposal_relative_energy_v2 import SpatialProposalRelativeEnergy
from lowlight_restoration.restoration_energy_numerics import (
    detached_local_mse_targets, ranking_score, refinement_objective,
)
from smoke.audit_foundir_vlm_and_novel_ppt_retry_v1 import matched_rows


def quadratic_gradient(size, batch, corrected):
    candidate = torch.full((batch, 3, size, size), .25, requires_grad=True)
    density = .5 * candidate.square().sum(1, keepdim=True)
    score = refinement_objective(density, reference_pixels=128**2) if corrected else ranking_score(density)
    return torch.autograd.grad(score.sum(), candidate)[0]


def test_legacy_mean_energy_has_explicit_resolution_scaling():
    small = quadratic_gradient(128, 1, False)
    large = quadratic_gradient(256, 1, False)
    assert large.mean().item() == small.mean().item() / 4


def test_actual_legacy_refine_uses_that_mean_convention():
    model = SpatialProposalRelativeEnergy()
    def score_map(self, observation, candidate, condition):
        return .5 * candidate.square().sum(1, keepdim=True)
    model.score_map_from_condition = MethodType(score_map, model)
    grads = []
    for size in (128, 256):
        proposal = torch.full((1, 3, size, size), .25)
        trajectory = model.refine(torch.zeros_like(proposal), proposal, torch.zeros(1, 5), steps=1, create_graph=False)
        grads.append(trajectory.gradients[0].mean().item())
    assert grads[1] == grads[0] / 4


def test_reference_area_preserves_nominal_update_without_batch_dependence():
    baseline = quadratic_gradient(128, 1, False)
    for size in (128, 192, 256):
        result = quadratic_gradient(size, 2, True)
        assert torch.allclose(result, torch.full_like(result, baseline.mean()), rtol=1e-6)


def test_candidate_order_unchanged_and_second_derivatives_exist():
    weight = torch.tensor(2., requires_grad=True)
    candidate = torch.stack((torch.full((3, 16, 16), .25),
                             torch.full((3, 16, 16), .5))).requires_grad_(True)
    density = weight * candidate.square().sum(1, keepdim=True)
    assert torch.equal(ranking_score(density).argsort(), refinement_objective(density, reference_pixels=64).argsort())
    grad = torch.autograd.grad(refinement_objective(density, reference_pixels=64).sum(), candidate, create_graph=True)[0]
    loss = (candidate - grad).square().mean()
    loss.backward()
    assert weight.grad is not None and torch.isfinite(weight.grad) and weight.grad.abs() > 0


def test_local_targets_do_not_move_with_candidate_graph():
    target = torch.zeros(1, 3, 16, 16, requires_grad=True)
    good = torch.full_like(target, .1, requires_grad=True)
    bad = torch.full_like(target, .3, requires_grad=True)
    quality = detached_local_mse_targets((good, bad), target, kernel=8)
    assert quality.shape == (1, 2, 2, 2)
    assert not quality.requires_grad
    assert (quality[:, 0] < quality[:, 1]).all()


def test_reject_invalid_energy_geometry():
    with unittest.TestCase().assertRaises(ValueError):
        ranking_score(torch.zeros(1, 3, 8, 8))
    with unittest.TestCase().assertRaises(ValueError):
        refinement_objective(torch.zeros(1, 1, 8, 8), reference_pixels=0)


def test_saved_metric_join_is_matched_not_full_vs_preview():
    row = {'record_id': 'a', 'source_group': 's', 'input_psnr': 1., 'input_lpips': .5}
    assert matched_rows([row], [row]) == [row]
    with unittest.TestCase().assertRaises(ValueError):
        matched_rows([row], [{**row, 'input_psnr': 2.}])


def load_tests(loader, tests, pattern):
    return unittest.TestSuite(unittest.FunctionTestCase(value) for name, value in globals().items()
                              if name.startswith('test_') and callable(value))


if __name__ == '__main__':
    unittest.main()
