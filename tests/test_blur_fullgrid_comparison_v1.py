import copy
import math
import unittest
from smoke.compare_foundir_blur_fullgrid_official_v1_cpu import summarize


def fixtures():
    # Each model is good on a different metric; never combine their best numbers.
    scores = {'step200': (26., .21), 'step1280': (25.4, .17), 'step1792': (25.8, .19), 'step9472': (25.85, .185)}
    result = {}
    for name, (p, l) in scores.items():
        records = [{'item': {'row': {'record_id': f'official_{i}', 'gt_sha256':str(i)}},
                    'GT_input_or_selection': False,
                    'metrics': {'lq': {'psnr':21.,'lpips':.3,'ssim':.7},
                                'old50step': {'psnr':22.,'lpips':.2,'ssim':.8},
                                'candidate': {'psnr':p,'lpips':l,'ssim':.8}}} for i in range(150)]
        means = {arm: {k: math.fsum(r['metrics'][arm][k] for r in records)/150
                       for k in ('psnr','lpips','ssim')} for arm in records[0]['metrics']}
        result[name] = {'status':'complete_official_Blur150','exit_code':0,
                        'GT_used_in_training_or_selection':False,'records':records,'means':means,
                        'recipe': {'selected_step':int(name[4:]),'population':'official150',
                                   'inference':'fixed','model_calls_per_tile':2,'tile':512,'overlap':128,
                                   'stitch':'fixed','padding':'fixed','metrics':'fixed','route':'fixed',
                                   'gate':{'psnr_min':25.9,'lpips_max':.18}},
                        'gate_checks':{'psnr':p>=25.9,'lpips':l<=.18},'official_Blur_pass':p>=25.9 and l<=.18}
    return result


class ComparisonTests(unittest.TestCase):
    def test_does_not_combine_best_metrics_from_different_models(self):
        report=summarize(fixtures())
        self.assertFalse(report['official_Blur_pass'])
        self.assertFalse(report['latest_vs']['step1280']['both_means_improve'])
        self.assertEqual(report['latest_vs']['step1280']['per_image_both_better'],0)

    def test_rejects_missing_duplicate_or_changed_GT_records(self):
        for mode in ('missing','duplicate','changed_GT'):
            r=fixtures(); records=r['step9472']['records']
            if mode=='missing':records.pop()
            elif mode=='duplicate':records[-1]=copy.deepcopy(records[0])
            else:records[0]['item']['row']['gt_sha256']='different'
            with self.subTest(mode=mode),self.assertRaises(ValueError):summarize(r)

    def test_rejects_changed_means_or_baselines(self):
        r=fixtures(); r['step9472']['means']['candidate']['psnr']+=.01
        with self.assertRaises(ValueError):summarize(r)
        r=fixtures();r['step9472']['records'][0]['metrics']['lq']['psnr']+=1
        r['step9472']['means']['lq']['psnr']+=1/150
        with self.assertRaises(ValueError):summarize(r)


if __name__=='__main__':unittest.main()
