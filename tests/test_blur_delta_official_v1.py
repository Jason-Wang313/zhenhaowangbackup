import unittest
import numpy as np
from smoke.evaluate_foundir_blur_delta_official_v1 import assigned, metric_summary, x


class OfficialTests(unittest.TestCase):
    def rows(self):
        return [{'row': {'record_id': f'official_01Blur_{i:04d}'}} for i in range(1, 151)]

    def records(self, psnr=25.9, lpips=.18):
        return [{'item': i, 'metrics': {'lq': {'psnr': 21., 'lpips': .35, 'ssim': .6},
                    'old50step': {'psnr': 22., 'lpips': .20, 'ssim': .7},
                    'candidate': {'psnr': psnr, 'lpips': lpips, 'ssim': .8}}} for i in self.rows()]

    def test_two_ranks_cover_exact_all150(self):
        rows = self.rows(); a, b = assigned(rows, 0), assigned(rows, 1)
        self.assertEqual(len(a), 75); self.assertEqual(len(b), 75)
        self.assertEqual({i['row']['record_id'] for i in a+b}, {i['row']['record_id'] for i in rows})
        with self.assertRaises(ValueError): assigned(rows[:-1], 0)
        with self.assertRaises(ValueError): assigned(rows[:-1]+rows[:1], 0)

    def test_joint_gate_uses_same_complete_candidate(self):
        self.assertTrue(metric_summary(self.records())['official_Blur_pass'])
        self.assertFalse(metric_summary(self.records(lpips=.181))['official_Blur_pass'])
        self.assertFalse(metric_summary(self.records(psnr=25.89))['official_Blur_pass'])
        with self.assertRaises(ValueError): metric_summary(self.records()[:-1])
        duplicate = self.records(); duplicate[-1] = duplicate[0]
        with self.assertRaises(ValueError): metric_summary(duplicate)

    def test_original_size_tile_identity_and_missing_tile_rejection(self):
        for h, w in ((1, 33), (513, 721), (1080, 1920)):
            row = {'record_id': 'synthetic', 'height': str(h), 'width': str(w)}
            image = np.random.default_rng(19).integers(0, 256, (h, w, 3), dtype=np.uint8)
            tiles = {t['key']: x.tile_array(image, t)[0] for t in x.tiles(row)}
            np.testing.assert_array_equal(x.stitch(row, tiles), image)
            tiles.pop(next(iter(tiles)))
            with self.assertRaises(ValueError): x.stitch(row, tiles)


if __name__ == '__main__': unittest.main()
