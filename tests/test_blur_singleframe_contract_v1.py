import copy
import unittest
from smoke.blur_singleframe_contract_v1 import views, key, summarize


def rows():
    row = {'record_id': 'foundir_train_0034314', 'split': 'train', 'category': '01Blur',
           'width': '1920', 'height': '1080', 'lq_path': 'data/foundir/MillionIRData/Train/01Blur/LQ/0034315.png',
           'gt_path': 'data/foundir/MillionIRData/Train/01Blur/GT/0034315.png'}
    return [{'row': row, 'role': 'train', 'crop_xy': [985, 489]}]*1024


class ContractTests(unittest.TestCase):
    def test_complete_deployment_coverage_and_original_crop(self):
        items = views(rows())
        self.assertEqual(len({key(i) for i in items}), 16)
        self.assertEqual({tuple(i['crop_xy']) for i in items[:15]},
                         {(x, y) for x in (0,384,768,1152,1408) for y in (0,384,568)})
        self.assertEqual(items[15]['crop_xy'], [985,489])

    def test_reject_official_source_and_duplicate_context(self):
        for field, value in [('split','test'), ('lq_path','data/foundir/Test/01Blur/LQ/0034315.png')]:
            data = copy.deepcopy(rows()); data[0]['row'][field] = value
            with self.assertRaises(ValueError): views(data)
        data = copy.deepcopy(rows()); data[0]['crop_xy'] = [0,0]
        with self.assertRaises(ValueError): views(data)

    def test_snapshots_reject_missing_duplicate_mixed_checkpoint(self):
        items = views(rows())
        records = [{'index': i, 'item': item, 'rank': i%2, 'step': 1792, 'scope': 'training_fit_only',
                    'parameter_hash': 'model', 'receipt_sha256': 'receipt',
                    'metrics': {'lq': {'psnr': 20., 'lpips': .4, 'ssim': .5},
                                'candidate': {'psnr': 21., 'lpips': .3, 'ssim': .6}}}
                   for i, item in enumerate(items)]
        result = summarize(records, items, 1792, 'model', 'receipt')
        self.assertFalse(result['quality_gate']); self.assertEqual(result['both_improve_count'],16)
        for bad in (records[:-1], records[:-1]+records[:1]):
            with self.assertRaises(ValueError): summarize(bad, items, 1792, 'model', 'receipt')
        records[3]['parameter_hash'] = 'other'
        with self.assertRaises(ValueError): summarize(records, items, 1792, 'model', 'receipt')


if __name__ == '__main__': unittest.main()
