import json
from pathlib import Path
import unittest

import torch

from lowlight_restoration.novel_ppt_energy import NovelPPTEnergy, objective
from smoke.train_foundir_novel_ppt_trainfit_v1_ddp import (
    fixed_records, near_candidates, ranking_counts, training_candidates, unchanged_learning_rate,
)


class TrainFitTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(2703)

    def test_fixed_sources_and_crops_ignore_row_order_and_step(self):
        rows = [dict(split='train', record_id=str(i), source_group=str(i//2)) for i in range(80)]
        first = fixed_records(rows, 27)
        self.assertEqual(first, fixed_records(list(reversed(rows)), 27))
        self.assertEqual(len({r['source_group'] for r, _ in first}), 32)
        self.assertTrue(all(isinstance(seed, int) for _, seed in first))
        with self.assertRaises(ValueError):
            fixed_records([{**rows[0], 'split': 'validation'}, *rows[1:]], 27)

    def test_ranking_ties_are_not_claimed_correct_and_close_pairs_have_support(self):
        correct = ranking_counts([0., 1., 2.], [1., 1.05, 2.], 1.)
        wrong = ranking_counts([2., 1., 0.], [1., 1.05, 2.], 1.)
        tied = ranking_counts([0., 0., 0.], [1., 1.05, 2.], 1.)
        self.assertEqual(correct['all_correct_credit'], 3)
        self.assertEqual(correct['close_pairs'], 1)
        self.assertEqual(wrong['all_correct_credit'], 0)
        self.assertEqual(tied['all_correct_credit'], 1.5)
        self.assertEqual(ranking_counts([0., 1.], [1., 1.], 1.)['all_pairs'], 0)
        with self.assertRaises(ValueError):
            ranking_counts([float('nan'), 1.], [1., 2.], 1.)

    def test_learning_rate_preserves_parent_horizon(self):
        root = Path(__file__).resolve().parents[1]
        spec = json.loads((root/'configs/foundir_dual_retry_execution_v1.json').read_text())['ppt']
        self.assertEqual(unchanged_learning_rate(spec, 1), 1e-6)
        self.assertAlmostEqual(unchanged_learning_rate(spec, 25), 2.5e-5)
        self.assertEqual(unchanged_learning_rate(spec, 100), 1e-4)
        self.assertGreater(unchanged_learning_rate(spec, 300), .99e-4)

    def test_cached_candidates_allow_repeated_full_backward_without_mutation(self):
        model = NovelPPTEnergy(width=4, condition_dim=8)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        x = torch.rand(1, 3, 16, 16)*.5
        case = dict(observation=x, proposal=x+.02, fidelity=x+.01, target=x+.007,
                    perceptual=x+.009, negatives=(x+.04, x-.01, x+.03, x+.025),
                    descriptor=torch.tensor([[0., .25, 0., 0., 0.]]))
        saved = case['fidelity'].clone()
        weights = dict(mse=1., lpips=2., gradient_fit=1., global_rank=.1, local_rank=.1,
                       no_harm_mse=1., no_harm_lpips=1.)
        perceptual = lambda a, b: (a-b).square().mean((1, 2, 3))
        for _ in range(2):
            optimizer.zero_grad()
            result = model(case['observation'], case['proposal'], case['fidelity'],
                           case['descriptor'], training_candidates(case))
            loss, _, _ = objective(result, case['fidelity'], case['target'], perceptual,
                                   residual_scale=.02, weights=weights)
            loss.backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
            optimizer.step()
        self.assertTrue(torch.equal(saved, case['fidelity']))
        self.assertTrue(model.field.head.weight.abs().max() > 0)
        self.assertEqual(len(training_candidates(case)), 11)
        self.assertEqual(list(near_candidates(case))[0], 'fidelity')

    def test_diagnostic_config_cannot_claim_formal_or_test_results(self):
        root = Path(__file__).resolve().parents[1]
        cfg = json.loads((root/'configs/foundir_novel_ppt_trainfit_v1.json').read_text())
        self.assertEqual(cfg['steps'], 300)
        self.assertEqual(cfg['records'], cfg['global_batch'])
        self.assertFalse(cfg['may_seed_formal'])
        self.assertFalse(cfg['development_pixels_accessed'])
        self.assertFalse(cfg['auto_launch_formal'])


if __name__ == '__main__':
    unittest.main()
