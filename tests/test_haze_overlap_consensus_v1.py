import unittest
import numpy as np
from lowlight_restoration.haze_overlap_consensus_v1 import synchronize


class ConsensusTest(unittest.TestCase):
    def test_single_tile_identity(self):
        rng = np.random.default_rng(18)
        a = rng.integers(0, 256, (512, 512, 3), dtype=np.uint8)
        result, base, stats = synchronize({'height': 300, 'width': 400},
                                         [{'key': 'a', 'x': 0, 'y': 0}], {'a': a})
        np.testing.assert_array_equal(result, a[:300, :400])
        np.testing.assert_array_equal(result, base)
        self.assertEqual(stats['correction_RMS_255'], 0)

    def test_true_shared_scene_unchanged(self):
        scene = np.random.default_rng(10).integers(20, 230, (512, 896, 3), dtype=np.uint8)
        tiles = [{'key': 'a', 'x': 0, 'y': 0}, {'key': 'b', 'x': 384, 'y': 0}]
        result, base, _ = synchronize({'height': 512, 'width': 896}, tiles,
                                     {'a': scene[:, :512], 'b': scene[:, 384:]})
        np.testing.assert_array_equal(result, scene)
        np.testing.assert_array_equal(base, scene)

    def test_known_offsets_recovered_without_target(self):
        tiles = [{'key': str(i), 'x': x, 'y': y} for i, (x, y) in enumerate(
                 [(0, 0), (384, 0), (0, 384), (384, 384)])]
        images = {str(i): np.full((512, 512, 3), value, np.uint8)
                  for i, value in enumerate([112, 120, 136, 144])}
        result, _, stats = synchronize({'height': 896, 'width': 896}, tiles, images)
        np.testing.assert_array_equal(result, np.full_like(result, 128))
        self.assertLess(stats['overlap_constant_MSE_255_after'], 1e-18)
        self.assertLess(max(abs(v) for v in stats['mean_correction_RGB_255_before_clip']), 1e-8)

    def test_missing_tile_rejected(self):
        with self.assertRaises(ValueError):
            synchronize({'height': 512, 'width': 896},
                        [{'key': 'a', 'x': 0, 'y': 0}], {})


if __name__ == '__main__':
    unittest.main()
