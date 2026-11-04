import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from smoke import evaluate_foundir_supervised_router_restoration_v1 as e


class MatchedRestorationTest(unittest.TestCase):
    def setUp(self):
        self.cpu = e.c.read(e.ROOT/e.supervised.REPORT)
        self.old = e.c.read(e.ROOT/e.analysis.OUTPUT)
        self.hard = e.c.read(e.ROOT/e.prior_hard.REPORT)

    def plan(self):
        return e.make_plan(self.cpu,self.old,self.hard)

    def test_real_reports_and_exact_render_scope(self):
        self.assertEqual(e.c.sha(e.ROOT/e.supervised.REPORT),e.CPU_REPORT_SHA)
        self.assertEqual(e.c.sha(e.ROOT/e.prior_hard.REPORT),e.HARD_REPORT_SHA)
        self.assertEqual(e.c.sha(e.ROOT/e.analysis.OUTPUT),e.prior_hard.BREAKDOWN_SHA)
        p = self.plan()
        self.assertEqual(p['prediction_rows'],80)
        self.assertEqual(p['unique_prediction_routes'],47)
        self.assertEqual(p['reused_unique_prediction_routes'],43)
        self.assertEqual(p['reused_prediction_rows'],76)
        self.assertEqual({r['render_id'] for r in p['new_plans']},e.EXPECTED_NEW)
        self.assertEqual([len(p['new_plans'][i::2]) for i in range(2)],[2,2])
        self.assertEqual([sum(x['render_id'] in e.EXPECTED_NEW for x in p['arms'][a]) for a in e.ARMS],[3,1])

    def test_changed_seed_crop_argmax_and_weights_rejected(self):
        saved = copy.deepcopy(self.cpu)
        for field,value in [('seed',-1),('crop_xy',[-1,-1]),('predicted_anchor','invalid'),
                            ('hard_weights',[.2]*5),('logits',[float('nan')]*5),
                            ('task_for_reporting_only','invalid')]:
            with self.subTest(field=field):
                self.cpu = copy.deepcopy(saved)
                self.cpu['arms']['image_only']['development']['rows'][0][field] = value
                with self.assertRaises(ValueError): self.plan()

    def test_missing_duplicate_or_extra_population_rejected(self):
        rows = self.cpu['arms']['image_only']['development']['rows']
        rows[-1] = copy.deepcopy(rows[0])
        with self.assertRaises(ValueError): self.plan()

    def test_wrong_training_step_and_policy_rejected(self):
        for key,value in [('steps_per_arm',401),('deployment_policy','known_class'),('restoration_images_generated',1)]:
            old = self.cpu[key]; self.cpu[key] = value
            with self.assertRaises(ValueError): self.plan()
            self.cpu[key] = old

    def test_cache_choice_does_not_depend_on_gt_quality(self):
        plan = self.plan()
        for row in self.old['per_image']:
            row['baseline'] = {'psnr':-999.,'lpips':999.}
            row['step408']['metrics'] = {'psnr':999.,'lpips':0.}
        for row in self.hard['per_image']: row['hard'] = {'psnr':-777.,'lpips':777.}
        self.assertEqual(plan,self.plan())

    def test_aggregation_matched_means_and_signs(self):
        plan = self.plan()
        # Synthetic fixture: not measured scientific results.
        metrics = {k:{'psnr':40. if v['predicted_anchor'] == next(r['task'] for r in self.old['per_image'] if r['key'] == v['key']) else 30.,
                      'lpips':.05 if v['predicted_anchor'] == next(r['task'] for r in self.old['per_image'] if r['key'] == v['key']) else .15}
                   for k,v in plan['routes'].items()}
        arms,delta = e.aggregate(plan,self.old,metrics)
        self.assertEqual([arms[a]['overall']['correct'] for a in e.ARMS],[33,38])
        self.assertAlmostEqual(delta['overall']['psnr_gain_db'],1.25)
        self.assertAlmostEqual(delta['overall']['lpips_reduction'],.0125)
        for arm in e.ARMS:
            self.assertEqual(arms[arm]['overall']['count'],40)
            self.assertEqual({t:v['count'] for t,v in arms[arm]['per_task'].items()},{t:8 for t in e.ORDER})

    def test_aggregation_rejects_missing_duplicate_wrong_join_and_nonfinite(self):
        original = self.plan(); values = {r:{'psnr':30.,'lpips':.1} for r in original['routes']}
        missing = dict(values); missing.pop(next(iter(missing)))
        with self.assertRaises(ValueError): e.aggregate(original,self.old,missing)
        duplicate = copy.deepcopy(original)
        duplicate['arms']['image_only'][-1] = duplicate['arms']['image_only'][0]
        with self.assertRaises(ValueError): e.aggregate(duplicate,self.old,values)
        wrong = copy.deepcopy(original)
        wrong['arms']['image_only'][0]['render_id'] = wrong['arms']['image_only'][1]['render_id']
        with self.assertRaises(ValueError): e.aggregate(wrong,self.old,values)
        values[next(iter(values))]['psnr'] = float('nan')
        with self.assertRaises(ValueError): e.aggregate(original,self.old,values)

    @staticmethod
    def route(anchor):
        i = e.ORDER.index(anchor)
        return {'anchor':anchor,'weights':[[float(t == anchor) for t in e.ORDER]],'prompt':'fixture '+anchor,
                'alpha':(.75,.30,1.,.925,1.)[i],'second_quantization':anchor not in ('haze','rain'),
                'schedule_steps':50,'model_calls':26 if anchor == 'rain' else 50,
                'readout':'endpoint_at_call25' if anchor == 'rain' else 'final_flow',
                'rain_after_call25':'zero_and_renormalize_nonrain'}

    def fixture(self,root):
        old = copy.deepcopy(self.old); hard = copy.deepcopy(self.hard)
        parent = root/'checkpoints/foundir_frozen_expert_router_v1'/e.analysis.RUN
        previous = root/e.prior_hard.OUTPUT
        e.c.write_new(previous/'receipt.json',{'test_fixture':True})
        e.c.write_new(previous/'runtime.json',{'test_fixture':True})
        hard['receipt_sha256'] = e.c.sha(previous/'receipt.json'); hard['runtime'] = {'test_fixture':True}
        bound = {'items':[{'key':r['key'],'task':r['task'],'split':'validation'} for r in old['per_image']],
                 'parent':{'experts':{t:{'prompt':'fixture '+t} for t in e.ORDER}}}
        e.c.write_new(parent/'receipt.json',bound)
        for item in bound['items']:
            key = item['key']; row = next(r for r in old['per_image'] if r['key'] == key)
            name = key+'.bin'; (parent/name).write_bytes(b'synthetic uint8 artifact')
            record = {'identity':{'receipt_sha256':old['receipt_sha256'],'item':item,'arm':'evaluation/known_class_control'},
                      'known_class_control_not_deployed':True,'route':self.route(item['task']),'metrics':row['baseline'],
                      'tensor':name,'files':{name:e.c.sha(parent/name)}}
            file = parent/'evaluation/known_class_control'/(key+'.json'); e.c.write_new(file,record)
            old['input_json_sha256'][file.relative_to(root).as_posix()] = e.c.sha(file)
        for item in hard['per_image']:
            key = item['key']; name = key+'.bin'; (previous/name).write_bytes(b'synthetic hard artifact')
            plan = next(p for p in old['missing_counterfactual']['items'] if p['key'] == key)
            record = {'identity':{'plan':plan,'receipt_sha256':hard['receipt_sha256'],'runtime_sha256':e.c.sha(previous/'runtime.json')},
                      'route':self.route(item['predicted_anchor']),'metrics':item['hard'],
                      'tensor':name,'files':{name:e.c.sha(previous/name)}}
            file = previous/'hard'/(key+'.json'); e.c.write_new(file,record)
            hard['artifacts'][key] = {'record_sha256':e.c.sha(file),'files':record['files']}
        return old,hard,bound,parent

    def test_cache_catalog_44_exact_entries_and_artifact_corruption(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); old,hard,bound,_ = self.fixture(root)
            catalog = e.cache_catalog(root,bound,old,hard)
            self.assertEqual(len(catalog),44)
            self.assertEqual(set(self.plan()['routes'])-set(catalog),e.EXPECTED_NEW)
            (root/next(iter(catalog.values()))['tensor']).write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError,'artifact changed'): e.cache_catalog(root,bound,old,hard)

    def test_artifact_path_escape_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); folder = root/'nested'; folder.mkdir()
            (root/'outside.bin').write_bytes(b'fixture')
            with self.assertRaises(ValueError):
                e.validate_artifact(root,folder,{'tensor':'../outside.bin','files':{}})

    def new_fixture(self,root,plan):
        folder = root/e.OUTPUT; folder.mkdir(parents=True,exist_ok=True)
        name = plan['render_id']+'.bin'; (folder/name).write_bytes(b'synthetic new artifact')
        row = {'identity':{'plan':plan,'receipt_sha256':e.c.sha(folder/'receipt.json'),
                           'runtime_sha256':e.c.sha(folder/'runtime.json')},
               'route':self.route(plan['predicted_anchor']),'tensor':name,'files':{name:e.c.sha(folder/name)},
               'GT_used_for_generation_or_selection':False,'metrics':{'psnr':31.,'lpips':.13},
               'seconds':1.,'peak_allocated_gib':1.}
        e.c.write_new(folder/'new'/(plan['render_id']+'.json'),row)
        return row

    def test_new_output_resume_and_tamper(self):
        plan = self.plan()['new_plans'][0]
        with tempfile.TemporaryDirectory() as temp,patch.object(e,'ROOT',Path(temp)):
            output = e.ROOT/e.OUTPUT
            e.c.write_new(output/'receipt.json',{'test_fixture':True})
            e.c.write_new(output/'runtime.json',{'test_fixture':True})
            self.assertIsNone(e.saved_new(plan))
            row = self.new_fixture(e.ROOT,plan)
            self.assertEqual(e.saved_new(plan),row)
            (output/row['tensor']).write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError,'artifact changed'): e.saved_new(plan)

    def test_complete_summary_80_rows_idempotent_missing_rejected(self):
        plan = self.plan()
        with tempfile.TemporaryDirectory() as temp,patch.object(e,'ROOT',Path(temp)):
            old,hard,bound,parent = self.fixture(e.ROOT)
            e.c.write_new(e.ROOT/e.analysis.OUTPUT,old)
            e.c.write_new(e.ROOT/e.prior_hard.REPORT,hard)
            e.c.write_new(e.ROOT/e.supervised.REPORT,self.cpu)
            catalog = e.cache_catalog(e.ROOT,bound,old,hard)
            receipt = {'plan':plan,'catalog':catalog}
            e.c.write_new(e.ROOT/e.OUTPUT/'receipt.json',receipt)
            e.c.write_new(e.ROOT/e.OUTPUT/'runtime.json',{'test_fixture':True})
            for p in plan['new_plans']: self.new_fixture(e.ROOT,p)
            with patch.object(e,'bound_inputs',return_value=(receipt,parent)):
                e.summarize(); e.summarize()
                result = e.c.read(e.ROOT/e.REPORT)
                self.assertEqual(result['plan_counts']['reused_prediction_rows'],76)
                self.assertEqual([result['arms'][a]['overall']['correct'] for a in e.ARMS],[33,38])
                self.assertEqual(len(result['artifacts']),48)  # 44 controls/caches + 4 new; 47 used predictions.
                self.assertFalse(result['GT_selects_expert'])
                (e.ROOT/e.OUTPUT/'new'/(plan['new_plans'][0]['render_id']+'.json')).unlink()
                with self.assertRaisesRegex(ValueError,'exactly four'): e.summarize()


if __name__ == '__main__': unittest.main()
