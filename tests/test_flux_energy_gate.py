import copy
import unittest

from lowlight_restoration.flux_energy_gate import compare_paired, MINIMUM_LPIPS_REDUCTION_FRACTION


class FluxEnergyGateTests(unittest.TestCase):
    def setUp(self):
        self.protocol = dict(manifest_sha256='manifest', crop_size=512, crop_seed=2026091602,
                             quantization='uint8', lpips_definition='alex normalize=True',
                             baseline_checkpoint_sha256='checkpoint', baseline_alpha=.3,
                             inference_steps=50, inference_seed_namespace='noise_scalar_full_development_v1')
        self.base = [dict(record_id='a', source_group='s', crop_xy=[0, 0],
                          input_uint8_sha256='a'*64, psnr=40., lpips=.1)]

    def compare(self, rows, protocol=None):
        return compare_paired(self.base, rows, baseline_protocol=self.protocol,
                              energy_protocol=self.protocol if protocol is None else protocol)

    def test_unchanged_strong_baseline_never_passes(self):
        self.base[0].update(psnr=43.39715355634689, lpips=.028955043449059303)
        out = self.compare(copy.deepcopy(self.base))
        self.assertFalse(out['joint_gate_pass'])
        self.assertFalse(out['lpips_pass'])
        self.assertAlmostEqual(out['psnr_target'], 43.69715355634689)
        self.assertAlmostEqual(out['lpips_target'], .02608035573895327)

    def test_relative_not_absolute_lpips_requirement(self):
        self.assertAlmostEqual(MINIMUM_LPIPS_REDUCTION_FRACTION, .09928107050377874)
        out = self.compare([{**self.base[0], 'psnr': 40.31, 'lpips': .09}])
        self.assertTrue(out['joint_gate_pass'])
        out = self.compare([{**self.base[0], 'psnr': 40.31, 'lpips': .095}])
        self.assertFalse(out['joint_gate_pass'])

    def test_psnr_and_lpips_must_both_pass(self):
        out = self.compare([{**self.base[0], 'psnr': 40.2, 'lpips': .08}])
        self.assertFalse(out['joint_gate_pass'])
        self.assertTrue(out['lpips_pass'])

    def test_mismatched_protocol_rejected(self):
        with self.assertRaises(ValueError):
            self.compare(self.base, {**self.protocol, 'crop_size': 256})

    def test_duplicate_missing_and_changed_identities_rejected(self):
        cases = [[], self.base*2, [{**self.base[0], 'record_id': 'b'}],
                 [{**self.base[0], 'crop_xy': [1, 0]}],
                 [{**self.base[0], 'source_group': 'other'}],
                 [{**self.base[0], 'input_uint8_sha256': 'b'*64}]]
        for case in cases:
            with self.subTest(case=case), self.assertRaises(ValueError):
                self.compare(case)

    def test_nonfinite_metrics_rejected(self):
        for bad in (float('nan'), float('inf'), True):
            with self.assertRaises(ValueError):
                self.compare([{**self.base[0], 'psnr': bad}])


if __name__ == '__main__':
    unittest.main()
