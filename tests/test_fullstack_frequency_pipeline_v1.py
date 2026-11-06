import copy
import unittest
from smoke.run_foundir_fullstack_frequency_repair_v1 import validate_rows,gates

class PipelineTests(unittest.TestCase):
    def rows(self):
        return [{'role':'train' if i<128 else 'development','crop_xy':[0,0],
                 'row':dict(record_id=str(i),source_group=str(i),gt_sha256=str(i),category='05Noise',composition_size='1',
                   split='train' if i<128 else 'validation',width='512',height='512',
                   lq_path='data/foundir/MillionIRData/Train/05Noise/LQ/a',gt_path='data/foundir/MillionIRData/Train/05Noise/GT/a')} for i in range(160)]
    def test_disjoint_plan(self):validate_rows(self.rows(),'noise',set())
    def test_protected_and_cross_role_and_official_rejected(self):
        with self.assertRaises(ValueError):validate_rows(self.rows(),'noise',{'0'})
        rows=self.rows();rows[-1]['row']['gt_sha256']='0'
        with self.assertRaises(ValueError):validate_rows(rows,'noise',set())
        rows=self.rows();rows[0]['row']['lq_path']='data/official/LQ/a'
        with self.assertRaises(ValueError):validate_rows(rows,'noise',set())
    def test_lpips_and_worst_regression_reject_psnr_gain(self):
        s={role:dict(candidate=dict(psnr=31.,lpips=.09),baseline=dict(psnr=30.,lpips=.10),worst_psnr_loss=.5) for role in ('train','development')}
        self.assertTrue(gates(s));bad=copy.deepcopy(s);bad['development']['candidate']['lpips']=.11
        self.assertFalse(gates(bad));s['development']['worst_psnr_loss']=1.01;self.assertFalse(gates(s))

if __name__=='__main__':unittest.main()
