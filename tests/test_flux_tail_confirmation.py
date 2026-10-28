import copy
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from smoke import diagnose_foundir_flux_tail_confirmation_v1 as confirmation
from smoke import diagnose_foundir_flux_tail_budget_v3 as v3
from lowlight_restoration.flux_tail_budget import choose_trial


class ConfirmationTests(unittest.TestCase):
    def setUp(self):
        self.spec = json.loads(confirmation.CONFIG.read_text())

    def test_fixed_policy_and_primary_endpoint(self):
        confirmation.validate_spec(self.spec)
        self.assertEqual(v3.policies(self.spec), [('mixed_psnr_budget', 'mixed', 'psnr_budget')])
        self.assertEqual(len(v3.arm_names(self.spec)), 7)
        for field, value in [('max_updates', 32), ('primary_endpoint', 'mixed_psnr_budget_step8'),
                             ('auto_train', True), ('cache_indices', list(range(24))),
                             ('acceptance_rules', ['monotone', 'psnr_budget']),
                             ('psnr_gain_target', .2), ('lpips_relative_reduction_target', .05),
                             ('stop_on_joint_target', True)]:
            with self.assertRaises(RuntimeError):
                confirmation.validate_spec(self.spec | {field: value})

    def test_4_and_8_gpu_shards_cover_the_same_24(self):
        for count in (4, 8):
            shards = [confirmation.assigned_indices(rank, count) for rank in range(count)]
            self.assertEqual(sorted(sum(shards, [])), list(range(8, 32)))
            self.assertEqual([len(s) for s in shards], [24//count]*count)
            self.assertEqual(count*self.spec['budget_wall_seconds_by_gpus'][str(count)]/3600, 4.)
            for rank, shard in enumerate(shards):
                self.assertTrue(all((i-8) % count == rank for i in shard))
        for rank, count in ((0, 3), (-1, 4), (8, 8)):
            with self.assertRaises(ValueError):
                confirmation.assigned_indices(rank, count)

    def test_exclusion_by_source_and_target_not_just_record_id(self):
        rows = [dict(split='train', category='05Noise', record_id=str(i), source_group=f's{i}',
                     gt_sha256=f'g{i}', lq_sha256=f'l{i}') for i in range(32)]
        confirmation.validate_selection(rows)
        for field in ('record_id', 'source_group', 'gt_sha256', 'lq_sha256'):
            for other in (0, 9):
                bad = copy.deepcopy(rows)
                bad[8][field] = bad[other][field]
                with self.assertRaises(RuntimeError):
                    confirmation.validate_selection(bad)
        for field, value in [('split', 'validation'), ('category', '04Rain')]:
            bad = copy.deepcopy(rows); bad[31][field] = value
            with self.assertRaises(RuntimeError):
                confirmation.validate_selection(bad)
        with self.assertRaises(RuntimeError):
            confirmation.validate_selection(rows[:31])

    def fake_rows(self, improve=False):
        rows = []
        for i in range(8, 32):
            baseline = dict(psnr=30.+i/32, lpips=.2)
            current = dict(psnr=baseline['psnr']+.4, lpips=.16) if improve else baseline.copy()
            trials = [dict(metrics=current.copy(), amplitude=a, lpips_weight=w)
                      for w in self.spec['direction_sets']['mixed'] for a in self.spec['rms_latent_steps']]
            chosen, decisions = choose_trial(trials, baseline, baseline, 'psnr_budget')
            history = [dict(step=1, current_before=baseline, current_after=current,
                trials=trials, degenerate_weights=[], chosen_trial=chosen, decision=decisions,
                accepted_updates=int(improve), stop_reason=None if improve else 'no_eligible_trial_in_frozen_grid')]
            if improve:
                chosen, decisions = choose_trial(trials, current, baseline, 'psnr_budget')
                history.append(dict(step=2, current_before=current, current_after=current,
                    trials=copy.deepcopy(trials), degenerate_weights=[], chosen_trial=chosen, decision=decisions,
                    accepted_updates=1, stop_reason='no_eligible_trial_in_frozen_grid'))
            arms = {key: dict(metrics=current.copy(), accepted_updates=int(improve),
                             control='fp32_state_control_48') for key in v3.arm_names(self.spec)}
            for key in ('historical_bf16', 'fp32_state_control_48'):
                arms[key] = dict(metrics=baseline.copy(), control='historical_bf16')
            rows.append(dict(cache_index=i, record_id=str(i), source_group=str(i), split='train',
                arms=arms, trajectories=dict(mixed_psnr_budget=history),
                baseline_parity_pass=True, ste_forward_exact=True))
        return rows

    def test_aggregate_new_baseline_and_primary_only(self):
        for improve in (False, True):
            result = confirmation.aggregate(self.fake_rows(improve), self.spec)
            self.assertEqual(result['primary_endpoint'], 'mixed_psnr_budget_step16')
            self.assertEqual(result['primary_mean_joint_target'], improve)
            self.assertEqual(result['primary_per_image_pass_count'], 24 if improve else 0)
            self.assertEqual(len(result['means']), 7)
            self.assertAlmostEqual(result['means']['historical_bf16']['lpips'], .2)
            for key in ('deployable', 'energy_trained', 'scientific_gate_evaluated',
                        'generalization_established', 'protected_pixels_accessed', 'ste_used_for_guidance'):
                self.assertFalse(result[key])
            if improve:
                value = result['means'][result['primary_endpoint']]
                self.assertAlmostEqual(value['psnr_gain_vs_historical'], .4)
                self.assertAlmostEqual(value['lpips_reduction_vs_historical'], .2)

    def test_reject_bad_shards_arms_and_trial_traces(self):
        rows = self.fake_rows()
        mutations = [rows[:-1], rows+rows[:1]]
        bad = copy.deepcopy(rows); bad[0]['cache_index'] = 0; mutations.append(bad)
        bad = copy.deepcopy(rows); bad[0]['source_group'] = '9'; mutations.append(bad)
        bad = copy.deepcopy(rows); bad[0]['split'] = 'internal_test'; mutations.append(bad)
        bad = copy.deepcopy(rows); bad[0]['arms'].pop(self.spec['primary_endpoint']); mutations.append(bad)
        bad = copy.deepcopy(rows); bad[0]['arms']['historical_bf16']['metrics']['psnr'] = float('nan'); mutations.append(bad)
        bad = copy.deepcopy(rows); bad[0]['arms'][self.spec['primary_endpoint']]['control'] = 'historical_bf16'; mutations.append(bad)
        for key, value in [('chosen_trial', 0), ('stop_reason', 'target_reached'), ('accepted_updates', 1)]:
            bad = copy.deepcopy(rows)
            bad[0]['trajectories']['mixed_psnr_budget'][0][key] = value
            mutations.append(bad)
        bad = copy.deepcopy(rows); bad[0]['trajectories']['mixed_psnr_budget'][0]['trials'].pop(); mutations.append(bad)
        bad = copy.deepcopy(rows); bad[0]['arms'][self.spec['primary_endpoint']]['metrics']['psnr'] += .1; mutations.append(bad)
        for bad in mutations:
            with self.assertRaises(RuntimeError):
                confirmation.aggregate(bad, self.spec)

    def test_receipt_binds_identity_seed_crop_and_rank(self):
        for gpus in (4, 8):
            receipt = dict(world_size=gpus, source_sha256={'code': 's'}, config_sha256='c', identities=[])
            for i in range(8, 32):
                row = dict(cache_index=i, rank=(i-8) % gpus, source_sha256={'code': 's'},
                    config_sha256='c', record_id=str(i), source_group=str(i), split='train', seed=i, crop_xy=[2, 3])
                receipt['identities'].append(dict(cache_index=i, row=row, seed=i, crop_xy=[2, 3]))
                confirmation.validate_row_identity(row, receipt)
                for field, value in [('rank', gpus), ('seed', -1), ('source_group', 'bad'),
                                     ('crop_xy', [0, 0]), ('config_sha256', 'bad')]:
                    with self.assertRaises(RuntimeError):
                        confirmation.validate_row_identity(row | {field: value}, receipt)


if __name__ == '__main__':
    unittest.main()
