from pathlib import Path
import json
import unittest

import torch
from torch.nn import functional as F

from lowlight_restoration.novel_ppt_energy import NovelPPTEnergy, objective, ranking_loss
from smoke.dual_retry_runtime import source_disjoint_training_probe


class RetryTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(27)

    def inputs(self, batch=1):
        x = torch.rand(batch, 3, 16, 16)*.5
        p = x + .02
        anchor = p - .01
        c = torch.zeros(batch, 5); c[:, 1] = .25
        return x, p, anchor, c

    def test_zero_potential_is_exact_identity_and_tie_selects_anchor(self):
        model = NovelPPTEnergy(width=4, condition_dim=8)
        x, p, anchor, c = self.inputs(2)
        trajectory, gradients, _ = model.trajectory(x, p, anchor, c)
        self.assertTrue(all(torch.equal(y, anchor) for y in trajectory))
        self.assertTrue(all(torch.count_nonzero(g) == 0 for g in gradients))
        chosen, indices, _ = model.select(x, p, anchor, c, trajectory)
        self.assertTrue(torch.equal(chosen, anchor)); self.assertEqual(indices.tolist(), [0, 0])

    def test_full_three_step_loss_has_finite_second_order_gradients_and_learns(self):
        model = NovelPPTEnergy(width=4, condition_dim=8)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        x, p, anchor, c = self.inputs()
        target = anchor - .003
        weights = dict(mse=1., lpips=2., gradient_fit=1., global_rank=.1, local_rank=.1, no_harm_mse=1., no_harm_lpips=1.)
        perceptual = lambda a, b: (a-b).square().mean((1, 2, 3))
        for _ in range(2):
            optimizer.zero_grad()
            out = model(x, p, anchor, c, (target, p, anchor, x))
            loss, _, diagnostic = objective(out, anchor, target, perceptual, residual_scale=.02, weights=weights)
            loss.backward()
            self.assertTrue(all(v.grad is not None and torch.isfinite(v.grad).all() for v in model.parameters()))
            optimizer.step()
        self.assertTrue(model.field.head.weight.abs().max() > 0)
        self.assertTrue(torch.isfinite(diagnostic['gradient_cosine']))
        trajectory, _, _ = model.trajectory(x, p, anchor, c)
        self.assertFalse(torch.equal(trajectory[-1], anchor))
        self.assertTrue((trajectory[-1]-anchor).abs().max() <= .080001)

    def test_ranking_labels_are_detached(self):
        scores = torch.tensor([[0., 0., 0.]], requires_grad=True)
        qualities = torch.tensor([[0., .1, .2]], requires_grad=True)
        ranking_loss(scores, qualities).backward()
        self.assertIsNone(qualities.grad)
        self.assertIsNotNone(scores.grad)

    def test_no_batch_division_in_energy_update(self):
        model = NovelPPTEnergy(width=4, condition_dim=8)
        torch.nn.init.normal_(model.field.head.weight, std=.01)
        x, p, anchor, c = self.inputs()
        single = model.trajectory(x, p, anchor, c)[0][-1]
        doubled = model.trajectory(*(v.repeat(2, 1, 1, 1) for v in (x, p, anchor)), c.repeat(2, 1))[0][-1]
        self.assertTrue(torch.allclose(single, doubled[:1], atol=1e-6, rtol=1e-5))

    def test_training_source_selection_never_accepts_other_splits(self):
        rows = [{'split': 'train', 'source_group': str(i//2), 'record_id': str(i)} for i in range(12)]
        subset = source_disjoint_training_probe(rows, 4, 27)
        self.assertEqual(len({r['source_group'] for r in subset}), 4)
        self.assertEqual(subset, source_disjoint_training_probe(list(reversed(rows)), 4, 27))
        with self.assertRaises(ValueError):
            source_disjoint_training_probe([{**rows[0], 'split': 'internal_test'}], 1, 27)

    def test_config_is_single_node_bounded_and_formal_is_not_automatic(self):
        root = Path(__file__).resolve().parents[1]
        config = json.loads((root/'configs/foundir_dual_retry_execution_v1.json').read_text())
        self.assertEqual(config['runtime']['world_size'], 8)
        self.assertEqual(config['ppt']['global_batch'], 8*config['ppt']['micro_batch']*config['ppt']['accumulation'])
        self.assertEqual(config['moe']['optimizer_steps'], 0)
        self.assertEqual(config['ppt']['qualification_steps'], 25)
        self.assertFalse(config['qualification_may_seed_formal'])
        script = (root/'smoke/run_foundir_dual_retry_v1_8gpu.sh').read_text()
        self.assertNotIn('--mode formal', script)
        self.assertIn('NCCL_NVLS_ENABLE=0', script)

    def test_gradient_probe_distinguishes_opposing_losses(self):
        from smoke.diagnose_foundir_joint_moe_train_gradients_v1 import gradient_summary
        p = torch.nn.Parameter(torch.tensor([1., 2., 3.]))
        unused = torch.nn.Parameter(torch.zeros(2))
        norms, cosine = gradient_summary({'increase': p.sum(), 'decrease': -p.sum()},
                                         ['used', 'unused'], [p, unused])
        self.assertAlmostEqual(cosine['increase__decrease'], -1.)
        self.assertTrue(norms['increase']['unused']['unused'])
        self.assertIsNone(p.grad)


if __name__ == '__main__':
    unittest.main()
