import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from smoke import diagnose_foundir_router_hard_counterfactual_v1 as d


class HardCounterfactualTest(unittest.TestCase):
    def setUp(self):
        self.report = json.loads((d.ROOT/d.BREAKDOWN).read_text())

    def test_real_report_pinned_and_fixed_four(self):
        self.assertEqual(d.sha(d.ROOT/d.BREAKDOWN), d.BREAKDOWN_SHA)
        plans = d.select(self.report)
        self.assertEqual(len(plans), 4)
        self.assertEqual([p['predicted_anchor'] for p in plans], ['blur', 'blur', 'blur', 'noise'])
        for p in plans:
            self.assertEqual(sum(p['hard_weights_from_model_argmax']), 1.)

    def test_label_cannot_replace_predicted_anchor(self):
        row = next(r for r in self.report['per_image'] if not r['step408']['correct'])
        row['step408']['anchor'] = row['task']
        with self.assertRaisesRegex(ValueError, 'changed argmax'):
            d.select(self.report)

    def test_missing_or_duplicate_case_rejected(self):
        self.report['per_image'][-1] = copy.deepcopy(self.report['per_image'][0])
        with self.assertRaises(ValueError): d.select(self.report)

    def test_crop_and_seed_manifest_mismatch_rejected(self):
        self.report['missing_counterfactual']['items'][0]['seed'] += 1
        with self.assertRaisesRegex(ValueError, 'list changed'): d.select(self.report)

    def test_nonfinite_probability_rejected(self):
        next(r for r in self.report['per_image'] if not r['step408']['correct'])['step408']['weights'][0] = float('nan')
        with self.assertRaises(ValueError): d.select(self.report)

    def test_signed_decomposition_including_mixture_improvement(self):
        row = {'key': 'example', 'task': 'noise', 'baseline': {'psnr': 40., 'lpips': .05},
               'step408': {'anchor': 'blur', 'metrics': {'psnr': 33., 'lpips': .08}}}
        result = d.decompose(row, {'psnr': 30., 'lpips': .1})
        self.assertEqual(result['choice_and_native_protocol_deficit']['psnr_loss_db'], 10.)
        self.assertEqual(result['soft_mixture_incremental_deficit']['psnr_loss_db'], -3.)
        self.assertAlmostEqual(result['soft_mixture_incremental_deficit']['lpips_increase'], -.02)
        self.assertAlmostEqual(result['total_soft_deficit']['lpips_increase'], .03)

    def test_nonfinite_metric_rejected(self):
        with self.assertRaises(ValueError):
            d.decompose(self.report['per_image'][0], {'psnr': float('inf'), 'lpips': .1})

    def test_four_plans_partition_two_workers_no_extra_render(self):
        plans = d.select(self.report)
        shards = [plans[r::2] for r in range(2)]
        self.assertEqual([len(s) for s in shards], [2, 2])
        self.assertEqual({p['key'] for s in shards for p in s}, d.EXPECTED_KEYS)

    def test_saved_result_resumes_and_detects_tamper_without_torch(self):
        plan = d.select(self.report)[0]
        with tempfile.TemporaryDirectory() as temp, patch.object(d, 'ROOT', Path(temp)):
            output = Path(temp)/d.OUTPUT
            output.mkdir(parents=True)
            (output/'receipt.json').write_text('{}')
            (output/'runtime.json').write_text('{}')
            (output/'tensor.bin').write_bytes(b'fixture, not a real tensor')
            self.assertIsNone(d.saved_result(plan))
            record = {'identity': {'plan': plan, 'receipt_sha256': d.sha(output/'receipt.json'),
                                  'runtime_sha256': d.sha(output/'runtime.json')},
                      'tensor': 'tensor.bin', 'files': {'tensor.bin': d.sha(output/'tensor.bin')},
                      'route': {'anchor': plan['predicted_anchor'], 'weights': [plan['hard_weights_from_model_argmax']]}}
            d.analysis.write_report(output/'hard'/(plan['key']+'.json'), record)
            self.assertEqual(d.saved_result(plan), record)
            (output/'tensor.bin').write_bytes(b'tampered')
            with self.assertRaisesRegex(ValueError, 'tensor changed'): d.saved_result(plan)

    def test_report_never_overwrites_existing_result(self):
        with tempfile.TemporaryDirectory() as temp:
            file = Path(temp)/'record.json'
            d.analysis.write_report(file, {'a': 1})
            d.analysis.write_report(file, {'a': 1})
            with self.assertRaises(ValueError): d.analysis.write_report(file, {'a': 2})

    def test_complete_four_result_summary_from_saved_files(self):
        plans = d.select(self.report)
        rows = {r['key']: r for r in self.report['per_image']}
        prompts = {t: 'fixture '+t for t in d.analysis.ORDER}
        with tempfile.TemporaryDirectory() as temp, patch.object(d, 'ROOT', Path(temp)):
            output = Path(temp)/d.OUTPUT; output.mkdir(parents=True)
            parent = Path(temp)/'parent'; parent.mkdir()
            d.analysis.write_report(output/'receipt.json', {'fixture': True})
            d.analysis.write_report(output/'runtime.json', {'fixture': True})
            d.analysis.write_report(parent/'receipt.json', {'parent': {'experts': {t: {'prompt': p} for t,p in prompts.items()}}})
            for plan in plans:
                anchor = plan['predicted_anchor']; i = d.analysis.ORDER.index(anchor)
                name = plan['key']+'.bin'; (output/name).write_bytes(b'test fixture')
                route = {'anchor': anchor, 'weights': [plan['hard_weights_from_model_argmax']],
                         'prompt': prompts[anchor], 'alpha': (.75,.30,1.,.925,1.)[i],
                         'second_quantization': anchor not in ('haze','rain'), 'schedule_steps': 50,
                         'model_calls': 26 if anchor == 'rain' else 50,
                         'readout': 'endpoint_at_call25' if anchor == 'rain' else 'final_flow',
                         'rain_after_call25': 'zero_and_renormalize_nonrain'}
                record = {'identity': {'plan': plan, 'receipt_sha256': d.sha(output/'receipt.json'),
                                      'runtime_sha256': d.sha(output/'runtime.json')},
                          'tensor': name, 'files': {name: d.sha(output/name)}, 'route': route,
                          'metrics': rows[plan['key']]['step408']['metrics'],
                          'GT_used_for_generation_or_selection': False, 'seconds': 1., 'peak_allocated_gib': 1.}
                d.analysis.write_report(output/'hard'/(plan['key']+'.json'), record)
            with patch.object(d, 'bound_inputs', return_value=({'plans': plans}, self.report, parent)):
                d.summarize()
                result = json.loads((Path(temp)/d.REPORT).read_text())
                self.assertEqual(result['count'], 4)
                self.assertEqual(result['four_case_mean']['soft_mixture_incremental_deficit'],
                                 {'psnr_loss_db': 0., 'lpips_increase': 0.})
                d.summarize()  # Idempotent, no model/tensor loads or overwrite.
                (output/'hard'/(plans[0]['key']+'.json')).unlink()
                with self.assertRaisesRegex(ValueError, 'four complete outputs'): d.summarize()


if __name__ == '__main__':
    unittest.main()
