import unittest
import numpy as np
from smoke.audit_foundir_fullstack_blur_translation_v1_cpu import spectrum,translation,common_psnr

class TranslationTests(unittest.TestCase):
    def test_recovers_inverse_shift_and_improves_correct_common_crop(self):
        rng=np.random.default_rng(2);ref=rng.random((80,96,3),dtype=np.float32);moving=np.roll(ref,(3,-2),(0,1))
        shift=translation(spectrum(ref,False),spectrum(moving,False),ref.shape[:2]);self.assertEqual(shift,[-3,2]);self.assertAlmostEqual(common_psnr(moving,ref,shift),80,places=6)
        self.assertLess(common_psnr(moving,ref),20)
    def test_equal_images_do_not_shift(self):
        a=np.random.default_rng(3).random((64,64,3),dtype=np.float32);self.assertEqual(translation(spectrum(a),spectrum(a),a.shape[:2]),[0,0])
    def test_outside_diagnostic_window_rejected(self):
        a=np.zeros((64,64,3),np.float32)
        with self.assertRaises(ValueError):common_psnr(a,a,(9,0))

if __name__=='__main__':unittest.main()
