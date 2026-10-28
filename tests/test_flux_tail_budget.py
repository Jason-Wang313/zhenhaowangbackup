import copy
import json
from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'third_party/flux2/src'))
from lowlight_restoration.flux_tail_budget import mixed_directions, thresholds, choose_trial
from lowlight_restoration.flux_tail_tradeoff import objective_directions
from smoke.diagnose_foundir_flux_tail_budget_v3 import validate_spec, policies, arm_names, aggregate
from smoke.diagnose_foundir_flux_tail_oracle_v1 import assigned_indices


def trial(psnr, lpips, step=.001, weight=.5):
    return dict(metrics=dict(psnr=psnr, lpips=lpips), amplitude=step, lpips_weight=weight)


class BudgetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.spec = json.loads((ROOT/'configs/foundir_flux_tail_budget_v3.json').read_text())
        self.baseline = dict(psnr=40., lpips=.1)

    def test_psnr_budget_spends_excess_but_never_target(self):
        current = dict(psnr=40.8, lpips=.09)
        trials = [trial(40.4, .08), trial(40.29, .07)]
        self.assertIsNone(choose_trial(trials, current, self.baseline, 'monotone')[0])
        chosen, decision = choose_trial(trials, current, self.baseline, 'psnr_budget')
        self.assertEqual(chosen, 0)
        self.assertEqual(decision['psnr_floor'], 40.3)
        self.assertFalse(decision['trials'][1]['psnr_ok'])

    def test_before_target_psnr_cannot_backslide(self):
        current = dict(psnr=40.1, lpips=.09)
        trials = [trial(40.09, .08), trial(40.15, .089)]
        for rule in ('monotone', 'psnr_budget'):
            self.assertEqual(choose_trial(trials, current, self.baseline, rule)[0], 1)

    def test_lpips_never_worsens_and_historical_caps_are_enforced(self):
        current = dict(psnr=40.8, lpips=.09)
        for rule in ('monotone', 'psnr_budget'):
            self.assertIsNone(choose_trial([trial(41., .091)], current, self.baseline, rule)[0])
            self.assertIsNone(choose_trial([trial(40.8, .09)], current, self.baseline, rule)[0])
        imperfect_control = dict(psnr=39.999, lpips=.10001)
        self.assertEqual(thresholds(imperfect_control, self.baseline, 'psnr_budget'),
                         dict(psnr_floor=40., lpips_ceiling=.1))
        self.assertIsNone(choose_trial([trial(39.9995, .099)], imperfect_control, self.baseline, 'psnr_budget')[0])

    def test_target_does_not_trigger_early_stop(self):
        current = dict(psnr=40.5, lpips=.08)
        self.assertEqual(choose_trial([trial(40.4, .07)], current, self.baseline, 'psnr_budget')[0], 0)

    def test_reach_psnr_floor_before_maximizing_lpips_gain(self):
        trials = [trial(40.15, .05), trial(40.35, .09), trial(40.5, .095)]
        for rule in ('monotone', 'psnr_budget'):
            self.assertEqual(choose_trial(trials, self.baseline, self.baseline, rule)[0], 1)
            self.assertEqual(choose_trial([trial(40.1, .05), trial(40.2, .09)],
                                         self.baseline, self.baseline, rule)[0], 1)

    def test_ranking_and_ties_are_deterministic(self):
        current = dict(psnr=40.8, lpips=.09)
        trials = [trial(40.5, .08, .005), trial(40.6, .08, .002),
                  trial(40.6, .08, .001), trial(40.4, .081, .0005)]
        self.assertEqual(choose_trial(trials, current, self.baseline, 'psnr_budget')[0], 2)
        with self.assertRaises(RuntimeError):
            choose_trial([trial(float('nan'), .08)], current, self.baseline, 'psnr_budget')
        with self.assertRaises(ValueError):
            choose_trial([], current, self.baseline, 'unknown')

    def test_direction_normalization_and_v2_balanced_parity(self):
        torch.manual_seed(8)
        gm, gp = torch.randn(1, 16)*40, torch.randn(1, 16)
        values, info = mixed_directions(gm, gp, [0., .25, .5, .75, 1.])
        previous, _ = objective_directions(gm, gp)
        self.assertTrue(torch.equal(values[.5], previous['balanced']))
        for v in values.values():
            self.assertAlmostEqual(float(v.square().mean().sqrt()), 1., places=6)
        self.assertGreater(info['mse_to_lpips_gradient_rms_ratio'], 1.)
        opposed, _ = mixed_directions(gm, -gm, [0., .5, 1.])
        self.assertIsNone(opposed[.5])
        self.assertIsNotNone(opposed[0.])
        with self.assertRaises(ValueError):
            mixed_directions(gm, gp, [.5, .5])
        with self.assertRaises(ValueError):
            mixed_directions(gm, gp, [2.])

    def test_frozen_scope_and_four_eight_complete_shards(self):
        validate_spec(self.spec)
        self.assertEqual(len(policies(self.spec)), 4)
        self.assertEqual(len(arm_names(self.spec)), 22)
        for count in (4, 8):
            shards = [assigned_indices(rank, count) for rank in range(count)]
            self.assertEqual(sorted(sum(shards, [])), list(range(8)))
            self.assertEqual(count*self.spec['budget_wall_seconds_by_gpus'][str(count)]/3600, 4.)
        for k, v in [('max_updates', 100), ('auto_expand', True), ('auto_train', True),
                     ('stop_on_joint_target', True), ('psnr_gain_target', .1),
                     ('cache_indices', [0, 2, 7]), ('surrogate', 'double_round_ste')]:
            with self.assertRaises(RuntimeError):
                validate_spec(self.spec | {k: v})

    def fake_rows(self):
        rows = []
        for i in range(8):
            arms = {key: dict(metrics=dict(self.baseline), accepted_updates=0,
                             control='historical_bf16') for key in arm_names(self.spec)}
            trajectories = {}
            for name, direction_set, rule in policies(self.spec):
                trials = [trial(40., .1, a, w) for w in self.spec['direction_sets'][direction_set]
                          for a in self.spec['rms_latent_steps']]
                chosen, decisions = choose_trial(trials, self.baseline, self.baseline, rule)
                trajectories[name] = [dict(step=1, current_before=self.baseline,
                    current_after=self.baseline, trials=trials, degenerate_weights=[],
                    chosen_trial=chosen, decision=decisions, accepted_updates=0,
                    stop_reason='no_eligible_trial_in_frozen_grid')]
            rows.append(dict(cache_index=i, record_id=str(i), source_group=str(i), split='train',
                             arms=arms, trajectories=trajectories,
                             baseline_parity_pass=True, ste_forward_exact=True))
        return rows

    def test_aggregate_no_false_claims_and_all_trials_rechecked(self):
        rows = self.fake_rows()
        result = aggregate(rows, self.spec)
        for key in ('energy_trained', 'deployable', 'generalization_established',
                    'scientific_gate_evaluated', 'protected_pixels_accessed', 'ste_used_for_guidance'):
            self.assertFalse(result[key])
        self.assertNotIn('finite_difference_pass', result)
        self.assertTrue(result['all_trial_decisions_reverified'])
        mutations = []
        bad = copy.deepcopy(rows); bad[0]['split'] = 'internal_test'; mutations.append(bad)
        bad = copy.deepcopy(rows); bad[0]['cache_index'] = 1; mutations.append(bad)
        bad = copy.deepcopy(rows); bad[0]['source_group'] = '1'; mutations.append(bad)
        bad = copy.deepcopy(rows); bad[0]['arms'].pop('mixed_psnr_budget_step16'); mutations.append(bad)
        bad = copy.deepcopy(rows); bad[0]['trajectories']['mixed_psnr_budget'][0]['chosen_trial'] = 0; mutations.append(bad)
        bad = copy.deepcopy(rows); bad[0]['trajectories']['mixed_psnr_budget'][0]['trials'].pop(); mutations.append(bad)
        bad = copy.deepcopy(rows); bad[0]['arms']['balanced_monotone_step16']['metrics']['psnr'] = 41.; mutations.append(bad)
        bad = copy.deepcopy(rows); bad[0]['trajectories']['balanced_monotone'][0]['stop_reason'] = 'target_reached'; mutations.append(bad)
        for bad in mutations:
            with self.assertRaises(RuntimeError):
                aggregate(bad, self.spec)
        with self.assertRaises(RuntimeError):
            aggregate(rows[:7], self.spec)

    def test_real_small_flux_two_backward_then_budget_step(self):
        from flux2.model import Flux2, Flux2Params
        from flux2.sampling import batched_prc_img
        from lowlight_restoration.flux_tail_diagnostic import Tail, checkpoint_blocks
        torch.manual_seed(19)
        model = Flux2(Flux2Params(in_channels=4, context_in_dim=3, hidden_size=32,
            num_heads=4, depth=1, depth_single_blocks=1, axes_dim=[2, 2, 2, 2],
            mlp_ratio=2., use_guidance_embed=False)).eval().requires_grad_(False)
        tokens, ids = batched_prc_img(torch.randn(1, 4, 2, 2)*.1)
        refs, ref_ids = batched_prc_img(torch.zeros(1, 4, 2, 2))
        class AE:
            def decode(self, x): return x
        tail = Tail(model, AE(), ids, refs, ref_ids, torch.randn(2, 2, 3),
                    torch.zeros(2, 2, 4, dtype=torch.long), [1., .1, 0.])
        tail.bf16 = False
        checkpoint_blocks(model, tail.calls)
        state = tokens.clone().requires_grad_(True)
        raw = tail.image(state, 1, 'rollout_full')
        gm = torch.autograd.grad(raw.square().mean(), state, retain_graph=True)[0]
        gp = torch.autograd.grad((raw-.1).square().mean(), state)[0]
        directions, _ = mixed_directions(gm, gp, [0., .25, .5, .75, 1.])
        with torch.no_grad():
            for direction in directions.values():
                if direction is not None:
                    output = tail.image(state.detach()-.001*direction, 1, 'rollout_full')
                    self.assertTrue(bool(output.isfinite().all()))
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in model.parameters()))


if __name__ == '__main__':
    unittest.main()
