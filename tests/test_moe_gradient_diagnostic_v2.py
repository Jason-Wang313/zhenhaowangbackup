import unittest

import torch

from smoke.diagnose_foundir_joint_moe_train_gradients_v2 import gradient_summary, probe_indices


class IntegerGradientProbeTests(unittest.TestCase):
    def test_actual_expert_shape_float32_boundary_reproduced_and_fixed(self):
        numel = 3072 * 9216
        old = torch.linspace(0, numel-1, 16384).long()
        self.assertEqual(int(old[-1]), numel)
        indices = probe_indices(numel)
        self.assertEqual(indices.dtype, torch.int64)
        self.assertEqual(len(indices), 16384)
        self.assertEqual(int(indices[0]), 0)
        self.assertEqual(int(indices[-1]), numel-1)
        self.assertTrue(bool(((indices >= 0) & (indices < numel)).all()))
        self.assertTrue(bool((indices[1:] > indices[:-1]).all()))

    def test_small_and_single_element_tensors(self):
        for numel in (1, 2, 17, 16384):
            self.assertTrue(torch.equal(probe_indices(numel), torch.arange(numel)))
        self.assertEqual(probe_indices(100, limit=1).tolist(), [0])
        for numel, limit in ((0, 10), (10, 0)):
            with self.assertRaises(ValueError):
                probe_indices(numel, limit=limit)

    def test_large_sizes_do_not_use_float_coordinates(self):
        for numel in (2**24+1, 2**25, 2**26, 2**31+5):
            indices = probe_indices(numel)
            self.assertEqual(int(indices[-1]), numel-1)
            self.assertEqual(indices.unique().numel(), 16384)
            self.assertTrue(torch.equal(indices, probe_indices(numel)))

    def test_gradient_diagnosis_preserves_parameters(self):
        p = torch.nn.Parameter(torch.tensor([1., 2., 3.]))
        unused = torch.nn.Parameter(torch.zeros(2))
        before = p.detach().clone()
        norms, cosine = gradient_summary({'a': p.sum(), 'b': -p.sum()},
                                         ['used', 'unused'], [p, unused])
        self.assertAlmostEqual(cosine['a__b'], -1.)
        self.assertTrue(norms['a']['unused']['unused'])
        self.assertIsNone(p.grad)
        self.assertTrue(torch.equal(p, before))


if __name__ == '__main__':
    unittest.main()
