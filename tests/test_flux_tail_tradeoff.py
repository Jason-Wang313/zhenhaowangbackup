import copy
import json
from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'third_party/flux2/src'))
from lowlight_restoration.flux_tail_tradeoff import surrogate_image, objective_directions, choose_safe_trial, joint_target
from smoke.qualify_foundir_flux_energy_v1 import deployed_proposal
from smoke.diagnose_foundir_flux_tail_tradeoff_v2 import validate_spec, arm_names, aggregate


class TradeoffTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_ste_forward_exact_random_and_rounding_boundaries(self):
        torch.manual_seed(7)
        obs = torch.randint(0, 256, (2, 3, 64, 64)).float().div(127.5).sub(1)
        raw = torch.rand_like(obs)*2.2-1.1
        for candidate in (raw, ((raw+1)*127.5).floor().add(.5).div(127.5).sub(1)):
            _, expected = deployed_proposal(obs, candidate)
            actual = surrogate_image(obs, candidate, 'double_round_ste')
            self.assertTrue(torch.equal(expected, actual))

    def test_ste_surrogate_gradient_and_clamp(self):
        raw = torch.tensor([[[[.11, 1.2, -1.2]]]], requires_grad=True)
        image = surrogate_image(torch.zeros_like(raw), raw, 'double_round_ste')
        gradient = torch.autograd.grad(image.sum(), raw)[0]
        self.assertTrue(torch.allclose(gradient, torch.tensor([[[[.3, 0., 0.]]]])))
        with self.assertRaises(ValueError):
            surrogate_image(raw, raw, 'unknown')

    def test_balanced_directions_have_common_first_order_descent(self):
        gm = torch.tensor([100., 0., 0.])
        gp = torch.tensor([-.5, 1., 0.])
        directions, info = objective_directions(gm, gp)
        self.assertLess(info['cosine'], 0)
        self.assertGreater(info['mse_to_lpips_gradient_rms_ratio'], 80)
        self.assertGreater(info['directions']['legacy_sum']['predicted_lpips_derivative_along_negative_direction'], 0)
        for metric in ('mse', 'lpips'):
            self.assertLess(info['directions']['balanced'][f'predicted_{metric}_derivative_along_negative_direction'], 0)
        for value in directions.values():
            self.assertAlmostEqual(float(value.square().mean().sqrt()), 1., places=6)

    def test_exact_opposition_is_not_fabricated(self):
        g = torch.tensor([1., 2.])
        directions, info = objective_directions(g, -g)
        self.assertIsNone(directions['balanced'])
        self.assertIsNone(directions['legacy_sum'])
        with self.assertRaises(RuntimeError):
            objective_directions(g, torch.zeros_like(g))

    def test_separate_and_joint_backward_through_real_small_flux(self):
        from flux2.model import Flux2, Flux2Params
        from flux2.sampling import batched_prc_img
        from lowlight_restoration.flux_tail_diagnostic import Tail, checkpoint_blocks
        torch.manual_seed(21)
        model = Flux2(Flux2Params(in_channels=4, context_in_dim=3, hidden_size=32, num_heads=4,
            depth=1, depth_single_blocks=1, axes_dim=[2, 2, 2, 2], mlp_ratio=2., use_guidance_embed=False))
        model.eval().requires_grad_(False)
        tokens, ids = batched_prc_img(torch.rand(1, 4, 2, 2)*.1)
        refs, ref_ids = batched_prc_img(torch.zeros(1, 4, 2, 2))
        class AE:
            def decode(self, x): return x
        tail = Tail(model, AE(), ids, refs, ref_ids, torch.randn(2, 2, 3),
                    torch.zeros(2, 2, 4, dtype=torch.long), [1., .1, 0.])
        tail.bf16 = False
        checkpoint_blocks(model, tail.calls)
        for surrogate in ('continuous', 'double_round_ste'):
            state = tokens.clone().requires_grad_(True)
            raw = tail.image(state, 1, 'rollout_full')
            image = surrogate_image(torch.zeros_like(raw), raw, surrogate)
            mse, perception = image.square().mean(), (image-.03).square().mean()
            legacy = torch.autograd.grad(mse+perception, state, retain_graph=True)[0]
            gm = torch.autograd.grad(mse, state, retain_graph=True)[0]
            gp = torch.autograd.grad(perception, state)[0]
            directions, info = objective_directions(gm, gp, legacy_gradient=legacy)
            self.assertTrue(all(value is None or value.isfinite().all() for value in directions.values()))
            self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_gt_line_search_rejects_either_harm_and_preserves_ties(self):
        baseline = dict(psnr=40., lpips=.1)
        trials = [dict(amplitude=.001, metrics=dict(psnr=40.5, lpips=.101)),
                  dict(amplitude=.002, metrics=dict(psnr=39.9, lpips=.08)),
                  dict(amplitude=.005, metrics=dict(psnr=40.2, lpips=.09))]
        self.assertEqual(choose_safe_trial(trials, baseline, baseline), 2)
        self.assertIsNone(choose_safe_trial(trials[:2], baseline, baseline))
        self.assertIsNone(choose_safe_trial([dict(amplitude=.001, metrics=baseline)], baseline, baseline))
        self.assertTrue(joint_target(dict(psnr=40.31, lpips=.09), baseline))
        self.assertFalse(joint_target(dict(psnr=40.31, lpips=.091), baseline))

    def test_scope_and_budget_frozen(self):
        spec = json.loads((ROOT/'configs/foundir_flux_tail_tradeoff_v2.json').read_text())
        validate_spec(spec)
        self.assertEqual(len(arm_names(spec)), 50)
        for count in (4, 8):
            self.assertEqual(count*spec['budget_wall_seconds_by_gpus'][str(count)]/3600, 4)
        for field, value in [('auto_train', True), ('deployable', True), ('trajectory_max_updates', 100),
                             ('start_index', 0), ('cache_indices', [0, 1]), ('new_field', 1)]:
            with self.assertRaises(RuntimeError):
                validate_spec(spec | {field: value})

    def test_aggregate_all_arms_no_false_derivative_or_generalization_claim(self):
        spec = json.loads((ROOT/'configs/foundir_flux_tail_tradeoff_v2.json').read_text())
        rows = []
        for i in range(8):
            arms = {name: dict(metrics=dict(psnr=40., lpips=.1), control='historical_bf16') for name in arm_names(spec)}
            rows.append(dict(cache_index=i, record_id=str(i), source_group=str(i), split='train',
                             arms=arms, baseline_parity_pass=True, ste_forward_exact=True))
        result = aggregate(rows, spec)
        self.assertFalse(result['deployable'])
        self.assertFalse(result['scientific_gate_evaluated'])
        self.assertFalse(result['energy_trained'])
        self.assertNotIn('finite_difference_pass', result)
        self.assertTrue(result['ste_is_surrogate_not_true_derivative'])
        bad = copy.deepcopy(rows)
        del bad[0]['arms'][next(iter(bad[0]['arms']))]
        with self.assertRaises(RuntimeError):
            aggregate(bad, spec)
        with self.assertRaises(RuntimeError):
            aggregate(rows[:7], spec)
        with self.assertRaises(RuntimeError):
            aggregate([rows[0] | {'split': 'internal_test'}]+rows[1:], spec)


if __name__ == '__main__':
    unittest.main()
