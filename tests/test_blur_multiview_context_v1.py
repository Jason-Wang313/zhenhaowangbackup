import copy
import unittest
from smoke import diagnose_foundir_blur_multiview_context_v1 as d


def records():
    return [{'item': {'row': {'record_id':str(i)}, 'crop_xy':[30,40]},
             'metrics_training_fit_only': {arm:{'psnr':22.,'lpips':.2,'ssim':.8} for arm in d.ARMS}}
            for i in range(8)]


class ContextComparisonTests(unittest.TestCase):
    def test_gap_sign_and_reordering_preserve_matched_pixels(self):
        old=records(); new=copy.deepcopy(old)
        for rows, standalone, full in ((old,24.,22.),(new,25.,24.)):
            for row in rows:
                row['metrics_training_fit_only']['crop_recomputed']['psnr']=standalone
                row['metrics_training_fit_only']['crop_from_full']['psnr']=full
        r=d.paired_summary(new,list(reversed(old)))
        self.assertEqual(r['full_minus_standalone_same_pixel_gaps']['step1280']['crop']['psnr'],-2.)
        self.assertEqual(r['full_minus_standalone_same_pixel_gaps']['step1792']['crop']['psnr'],-1.)
        self.assertEqual(r['1792_minus1280']['crop_from_full']['psnr'],2.)
        self.assertFalse(r['new_quality_gate'])

    def test_missing_duplicate_or_other_crop_rejected(self):
        for mode in ('missing','duplicate','crop'):
            old=records();new=copy.deepcopy(old)
            if mode=='missing':new.pop()
            elif mode=='duplicate':new[-1]=copy.deepcopy(new[0])
            else:new[0]['item']['crop_xy']=[31,40]
            with self.subTest(mode=mode),self.assertRaises((RuntimeError,ValueError)):
                d.paired_summary(new,old)

    def test_lq_scoring_path_change_is_not_model_improvement(self):
        old=records();new=copy.deepcopy(old)
        new[0]['metrics_training_fit_only']['crop_lq']['psnr']+=.01
        with self.assertRaises((RuntimeError,ValueError)):d.paired_summary(new,old)


if __name__=='__main__':unittest.main()
