import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch

from smoke import analyze_foundir_router_generalization_v1 as a


class GeneralizationAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def test_published_scores_and_two_different_errors(self):
        cpu,restored=a.reports(); result=a.score_audit(cpu,restored)
        self.assertEqual(result['combined_correct'],38)
        self.assertEqual(result['same_model_image_branch_correct'],20)
        self.assertEqual((result['flip_helped'],result['flip_hurt']),(19,1))
        errors={r['key']:r for r in result['errors']}
        blur=errors['validation_foundir_train_0007534']; low=errors['validation_foundir_train_0479048']
        self.assertTrue(blur['flip_hurt']); self.assertFalse(low['flip_hurt'])
        self.assertLess(blur['selected_minus_true_margin']['image'],0)
        self.assertGreater(blur['selected_minus_true_margin']['combined'],0)
        self.assertEqual(low['separately_trained_image_only'],'noise')

    def test_reject_changed_logit_sum_probability_or_population(self):
        cpu,restored=a.reports()
        for field in ('image_logits','soft_weights_diagnostic_only'):
            altered=copy.deepcopy(cpu); altered['arms']['image_plus_vlm']['development']['rows'][0][field][0]+=1
            with self.assertRaises(ValueError): a.score_audit(altered,restored)
        altered=copy.deepcopy(cpu); altered['arms']['image_plus_vlm']['development']['rows'].pop()
        with self.assertRaises(ValueError): a.score_audit(altered,restored)

    @staticmethod
    def metadata():
        return [{'key':f'{split}_foundir_train_{t}_{j}','task':t,'split':split,'seed':j,
                 'row':{'record_id':f'foundir_train_{t}_{j}','category':a.CATEGORIES[t],
                        'split':split,'composition_size':'1','width':'1024','height':'1024','crop_xy':[0,0],
                        'source_group':f'{split}_{t}_{j}','gt_sha256':f'{split}_{t}_{j}',
                        'lq_sha256':f'{split}_{t}_{j}'}}
                for split,n in [('train',128),('validation',8)] for t in a.ORDER for j in range(n)]

    def test_metadata_exact_coverage_and_no_overlap(self):
        rows=self.metadata(); report=a.metadata_audit(rows)
        self.assertEqual(report['exposures_per_training_record'],20)
        self.assertEqual(report['unique_sources'],{'train':640,'validation':40})
        self.assertFalse(report['visual_label_correctness_verified'])
        rows[-1]['row']['lq_sha256']=rows[0]['row']['lq_sha256']
        with self.assertRaisesRegex(ValueError,'overlap'): a.metadata_audit(rows)

    def test_metadata_bad_label_or_crop_rejected(self):
        for field,value in [('category','08Haze'),('crop_xy',[900,0])]:
            rows=self.metadata(); rows[0]['row'][field]=value
            with self.assertRaises(ValueError): a.metadata_audit(rows)

    @staticmethod
    def small_features():
        train=[{'key':f'train_{t}_{j}','task':t,'split':'train','row':{'source_group':f'{t}_{j//2}'}}
               for t in a.ORDER for j in range(4)]
        dev=[{'key':f'validation_{t}','task':t,'split':'validation','row':{'source_group':f'dev_{t}'}} for t in a.ORDER]
        g=torch.Generator().manual_seed(11)
        x=torch.randn(20,12,generator=g,dtype=torch.float64)
        # First two rows of each class share a source and identical features;
        # leave-source-out must not use their zero-distance pair.
        for j in range(0,20,4): x[j+1]=x[j]
        return train,x,dev,x[::4].clone()

    def test_distances_train_only_and_leave_source_out(self):
        train,x,dev,y=self.small_features(); result=a.feature_coverage(train,x,dev,y)
        for t,v in result['training_reference'].items():
            self.assertGreater(v['leave_source_out_same_class_distance_quantiles']['q10'],.01)
        for row in result['development_rows']:
            self.assertLess(row['top5_training_neighbors'][0]['distance'],1e-12)
            self.assertTrue(all(n['key'].startswith('train_') for n in row['top5_training_neighbors']))
            self.assertEqual(row['same_class_distance_percentile_in_training_reference'],0.)
        self.assertFalse(result['used_to_change_routing'])

    def test_centered_distance_invariant_to_positive_scale_and_shift(self):
        train,x,dev,y=self.small_features()
        base=a.feature_coverage(train,x,dev,y); changed=a.feature_coverage(train,x*3+2,dev,y*7-4)
        for r1,r2 in zip(base['development_rows'],changed['development_rows']):
            self.assertEqual([n['key'] for n in r1['top5_training_neighbors']],[n['key'] for n in r2['top5_training_neighbors']])
            for t in a.ORDER:
                self.assertAlmostEqual(r1['nearest_training_by_class'][t]['distance'],r2['nearest_training_by_class'][t]['distance'],places=12)

    def test_bad_features_and_source_overlap_rejected(self):
        train,x,dev,y=self.small_features()
        with self.assertRaises(ValueError): a.feature_coverage(train,torch.ones_like(x),dev,y)
        z=x.clone(); z[0,0]=float('nan')
        with self.assertRaises(ValueError): a.feature_coverage(train,z,dev,y)
        dev[0]['row']['source_group']=train[0]['row']['source_group']
        with self.assertRaisesRegex(ValueError,'overlap'): a.feature_coverage(train,x,dev,y)

    def test_cache_copy_identity_and_corruption(self):
        from safetensors.torch import save_file
        with tempfile.TemporaryDirectory() as temp,patch.object(a,'ROOT',Path(temp)):
            item=self.metadata()[0]; key=item['key']
            parent=a.ROOT/'checkpoints/foundir_frozen_expert_router_v1'/a.quality.analysis.RUN
            parent.mkdir(parents=True)
            value=torch.arange(12,dtype=torch.float32).reshape(1,12)
            save_file({'vlm':value},str(parent/'vlm.safetensors'))
            save_file({'vlm':value},str(parent/'features.safetensors'))
            vm=parent/'vlm'/(key+'.json')
            a.c.write_new(vm,{'identity':{'receipt_sha256':'receipt','item':item,'arm':'vlm'},'GT_input':False,
                             'tensor':'vlm.safetensors','files':{'vlm.safetensors':a.c.sha(parent/'vlm.safetensors')}})
            marker=parent/'features'/(key+'.json')
            a.c.write_new(marker,{'identity':{'receipt_sha256':'receipt','item':item,'arm':'features'},
                                 'GT_input':False,'vlm_receipt_sha256':a.c.sha(vm)})
            bound={'items':[item],'parent_receipt_sha256':'receipt','features':{key:{
                'path':(parent/'features.safetensors').relative_to(a.ROOT).as_posix(),'marker_sha256':a.c.sha(marker)}}}
            self.assertEqual(a.cache_identity_audit(bound)['verified_records'],1)
            save_file({'vlm':value+1},str(parent/'features.safetensors'))
            with self.assertRaisesRegex(ValueError,'different input'): a.cache_identity_audit(bound)
            (parent/'vlm.safetensors').write_bytes(b'corrupted')
            with self.assertRaisesRegex(ValueError,'artifact changed'): a.cache_identity_audit(bound)


if __name__=='__main__': unittest.main()
