import copy
import unittest
from smoke.blur_fixed8_fit_contract_v1 import select_items, summarize, STEPS, SNAPSHOTS
from lowlight_restoration.blur_delta_velocity_v1 import sample_indices


def population():
    return [{'role': 'train', 'row': {'record_id': str(i), 'source_group': str(i), 'gt_sha256': str(i),
            'split': 'train', 'category': '01Blur', 'lq_path': 'data/foundir/MillionIRData/Train/01Blur/lq',
            'gt_path': 'data/foundir/MillionIRData/Train/01Blur/gt'}} for i in range(1024)]


class FixedFitTests(unittest.TestCase):
    def test_fixed_first8_reject_nontraining_or_duplicate_sources(self):
        rows = population(); self.assertEqual(select_items(rows), rows[:8])
        rows[0]['row']['gt_path'] = 'data/official/gt'
        with self.assertRaises(ValueError): select_items(rows)
        rows = population(); rows[1]['row']['source_group'] = '0'
        with self.assertRaises(ValueError): select_items(rows)

    def test_every_step_covers_each_image_exactly_once_global8(self):
        exposure = [0]*8
        for step in range(STEPS):
            a, b = [sample_indices(step, rank, count=8) for rank in (0, 1)]
            self.assertEqual(sorted(a+b), list(range(8)))
            for i in a+b: exposure[i] += 1
        self.assertEqual(exposure, [100]*8)
        self.assertEqual(SNAPSHOTS[-1], STEPS)

    def records(self):
        items = select_items(population())
        records = [{'index': i, 'item': item, 'rank': i % 2, 'step': 100,
                    'parameter_hash': 'p', 'receipt_sha256': 'r',
                    'metrics': {'lq': {'psnr': 20., 'lpips': .3, 'ssim': .7},
                                'candidate': {'psnr': 19. if i == 7 else 22., 'lpips': .2, 'ssim': .8}}}
                   for i, item in enumerate(items)]
        return records, items

    def test_complete_snapshot_reports_harm_without_turning_into_gate(self):
        records, items = self.records(); r = summarize(records, items, 100, 'p', 'r')
        self.assertEqual(r['both_improve_count'], 7); self.assertEqual(r['PSNR_worse_count'], 1)
        self.assertFalse(r['quality_gate']); self.assertEqual(r['means']['candidate']['psnr'], 21.625)
        with self.assertRaises(ValueError): summarize(records[:-1], items, 100, 'p', 'r')
        records[-1] = copy.deepcopy(records[0])
        with self.assertRaises(ValueError): summarize(records, items, 100, 'p', 'r')

    def test_snapshot_must_match_step_parameters_and_finite_values(self):
        records, items = self.records()
        with self.assertRaises(ValueError): summarize(records, items, 50, 'p', 'r')
        with self.assertRaises(ValueError): summarize(records, items, 100, 'wrong', 'r')
        records[0]['metrics']['candidate']['lpips'] = float('nan')
        with self.assertRaises(ValueError): summarize(records, items, 100, 'p', 'r')


if __name__ == '__main__': unittest.main()
