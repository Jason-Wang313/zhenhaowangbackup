import unittest
import numpy as np
from lowlight_restoration.flux_frequency_repair_v1 import apply, statistics, fit, band_map, assigned


class FrequencyRepairTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        self.x = rng.integers(0, 256, (64, 96, 3), dtype=np.uint8)
        self.y = rng.integers(0, 256, self.x.shape, dtype=np.uint8)

    def test_endpoints_and_identity(self):
        np.testing.assert_array_equal(apply(self.x, self.y, [0]*6), self.x)
        np.testing.assert_array_equal(apply(self.x, self.y, [1]*6), self.y)
        np.testing.assert_array_equal(apply(self.x, self.x, [.3]*6), self.x)

    def test_constant_gain_matches_rgb_blend_except_half_ties(self):
        expected = np.rint(self.x.astype(float)*.7 + self.y.astype(float)*.3)
        error = abs(apply(self.x, self.y, [.3]*6).astype(float)-expected)
        self.assertLessEqual(error.max(), 1)
        continuous = self.x.astype(float)*.7 + self.y.astype(float)*.3
        ties = np.isclose(continuous-np.floor(continuous), .5, rtol=0, atol=1e-10)
        self.assertTrue(np.all(error[~ties] == 0))

    def test_analytic_solution_recovers_distinct_dc_and_detail(self):
        x = np.full((64, 96, 3), 110, np.uint8)
        checker = np.indices((64, 96)).sum(0)%2
        y = np.repeat((130+20*checker)[..., None], 3, axis=2).astype(np.uint8)
        target = apply(x, y, [.2, .4, .4, .4, .4, .8])
        w = fit([statistics(x, y, target)], .3)
        self.assertLess(abs(w[0]-.2), .03)
        self.assertLess(abs(w[-1]-.8), .03)
        result = apply(x, y, w)
        self.assertLess(np.square(result.astype(float)-target).mean(), 1.)

    def test_zero_energy_falls_back_to_parent(self):
        np.testing.assert_allclose(fit([statistics(self.x, self.x, self.y)], .3), [.3]*6)

    def test_dc_is_only_one_coefficient(self):
        m = band_map(64, 96)
        self.assertEqual(int((m == 0).sum()), 1)
        self.assertEqual(set(np.unique(m)), set(range(6)))

    def test_nodes_no_missing_or_duplicate(self):
        for nodes in ([0], [2], [0, 1, 2], [1, 2]):
            rows = [i for n in nodes for local in (0, 1) for i in assigned(list(range(160)), n, local, nodes)[1]]
            self.assertEqual(sorted(rows), list(range(160)))

    def test_reject_invalid(self):
        for weights in ([float('nan')]*6, [2]*6, [-1]*6, [1]*5):
            with self.assertRaises(ValueError): apply(self.x, self.y, weights)
        with self.assertRaises(ValueError): assigned([], 0, 0, [0, 0])
        with self.assertRaises(ValueError): statistics(self.x, self.y[:32], self.x)


if __name__ == '__main__': unittest.main()
