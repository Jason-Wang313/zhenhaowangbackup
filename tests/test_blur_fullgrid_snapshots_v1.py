import copy
import unittest
from unittest.mock import patch
from smoke import blur_fullgrid_snapshots_v1 as snapshots


class SnapshotTests(unittest.TestCase):
    def records(self):
        b = {'rows':[{'row':{'record_id':f'train_{i}'}} for i in range(8)]}
        records = [{'source_index':i,'item':item,'rank':i%4,'step':3712,'parameter_hash':'model',
                    'receipt_sha256':'receipt','scope':'training_fit_only',
                    'metrics':{'lq':{'psnr':20.,'lpips':.4,'ssim':.5},
                               'candidate':{'psnr':21.,'lpips':.3,'ssim':.6}}}
                   for i,item in enumerate(b['rows'])]
        return b,records

    def test_mean_is_descriptive_not_quality_gate(self):
        b,records=self.records()
        with patch.object(snapshots.data,'receipt_sha',return_value='receipt'):
            result=snapshots.summarize(records,b,3712,'model')
        self.assertFalse(result['quality_gate']);self.assertEqual(result['both_better_than_LQ'],8)
        self.assertEqual(result['means']['candidate']['psnr'],21.)

    def test_missing_duplicate_and_mixed_weights_rejected(self):
        b,records=self.records(); mixed=copy.deepcopy(records);mixed[0]['parameter_hash']='other'
        with patch.object(snapshots.data,'receipt_sha',return_value='receipt'):
            for bad in (records[:-1],records[:-1]+records[:1],mixed):
                with self.assertRaises((ValueError,RuntimeError)):
                    snapshots.summarize(bad,b,3712,'model')
            with self.assertRaises((ValueError,RuntimeError)):
                snapshots.summarize(records,b,1992,'model')


if __name__=='__main__':unittest.main()
