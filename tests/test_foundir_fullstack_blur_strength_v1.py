import unittest
import numpy as np
from smoke import audit_foundir_fullstack_blur_strength_v1 as b

class StrengthTests(unittest.TestCase):
    def test_identity_and_zero_return_exact_pixels(self):
        a=np.arange(24,dtype=np.uint8).reshape(2,4,3);p=255-a
        np.testing.assert_array_equal(b.residual_image(a,p,1,1),p)
        np.testing.assert_array_equal(b.residual_image(a,p,0,1),a)
    def test_integer_round_even_and_clipping_not_wraparound(self):
        a=np.array([[[0,0,240],[255,255,0]]],dtype=np.uint8)
        p=np.array([[[1,3,255],[0,254,255]]],dtype=np.uint8)
        np.testing.assert_array_equal(b.residual_image(a,p,1,2),[[[0,2,248],[128,254,128]]])
        r=b.residual_image(a,p,3,2);self.assertEqual(int(r[0,0,2]),255);self.assertEqual(int(r[0,1,0]),0)
    def test_invalid_dtype_or_strength_rejected(self):
        a=np.zeros((1,1,3),dtype=np.uint8)
        with self.assertRaises(Exception):b.residual_image(a.astype(float),a,1,2)
        with self.assertRaises(Exception):b.residual_image(a,a,1,0)
    def records(self):
        result=[]
        for cat,count in [('01Blur',150),('05Noise',100),('08Haze',200),('10Rain',100),('14Lowlight',50)]:
            for i in range(count):result.append({'item':{'row':{'record_id':f'{cat}_{i}','category':cat}},'arms':{name:{'metrics':{'psnr':30.,'lpips':.15,'ssim':.8}} for name in b.SCALES}})
        return result
    def test_all600_summary_keeps_thresholds_and_does_not_select_best(self):
        result=b.summarize(self.records());self.assertEqual(set(result),set(b.SCALES))
        for arm in result.values():
            self.assertFalse(arm['all_ten_pass']);self.assertTrue(arm['checks']['01Blur']['psnr']);self.assertFalse(arm['checks']['05Noise']['psnr'])
    def test_missing_or_duplicate_records_rejected(self):
        records=self.records()
        with self.assertRaises(Exception):b.summarize(records[:-1])
        records[-1]=records[0]
        with self.assertRaises(Exception):b.summarize(records)

if __name__=='__main__':unittest.main()
