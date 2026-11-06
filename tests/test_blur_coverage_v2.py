import ast
import copy
from pathlib import Path
import unittest
from smoke import run_foundir_fullstack_blur_coverage_v2 as t
from smoke import evaluate_foundir_blur_coverage_official_v2 as e
from lowlight_restoration import blur_delta_velocity_v1 as core


def rows(count=32):
    return [dict(record_id=str(i),source_group='g'+str(i),split='train',category='01Blur',composition_size='1',
      width='1920',height='1080',lq_path=f'data/foundir/MillionIRData/Train/01Blur/LQ/{i}.png',
      gt_path=f'data/foundir/MillionIRData/Train/01Blur/GT/{i}.png',lq_sha256='l'+str(i),gt_sha256='t'+str(i)) for i in range(count)]


class CoverageTests(unittest.TestCase):
    def test_selection_excludes_protected_and_nontraining_groups(self):
        data=rows(); data[0]['split']='validation'; data[1]['width']='3840'
        selected=t.select_rows(data,{'g2'},{'l3','t4'},count=8)
        self.assertEqual(len(selected),8)
        self.assertTrue(all(i['row']['record_id'] not in ('0','1','2','3','4') for i in selected))
        for item in selected:
            x,y=item['crop_xy'];self.assertTrue(0<=x<=1408 and 0<=y<=568)

    def test_selection_independent_of_manifest_order_and_representative_counts(self):
        data=rows(); duplicate=copy.deepcopy(data[3]);duplicate['record_id']='extra'
        a=t.select_rows(data+[duplicate],set(),set(),8)
        b=t.select_rows(list(reversed(data+[duplicate])),set(),set(),8)
        self.assertEqual(a,b);self.assertEqual(len({i['row']['source_group'] for i in a}),8)

    def test_bad_scope_and_insufficient_population_fail_closed(self):
        with self.assertRaises(ValueError):t.select_rows(rows(7),set(),set(),8)
        data=rows();data[0]['lq_path']='downloads/foundir_official/x.png'
        with self.assertRaises(ValueError):t.select_rows(data,set(),set(),8)
        data=rows(8);data[1]['gt_sha256']=data[0]['gt_sha256']
        with self.assertRaises(ValueError):t.select_rows(data,set(),set(),8)

    def test_1024_sampler_has_full_coverage_and_no_rank_overlap(self):
        seen=[]
        for step in range(128):
            a,b=[core.sample_indices(step,r,count=1024) for r in (0,1)]
            self.assertFalse(set(a)&set(b));seen+=a+b
        self.assertEqual(sorted(seen),list(range(1024)))
        self.assertEqual(core.sample_indices(128,0,count=1024),core.sample_indices(128,0,count=1024))
        self.assertNotEqual(core.sample_indices(0,0,count=1024),core.sample_indices(128,0,count=1024))

    def test_numerical_kernel_and_full_official_evaluation_preserved(self):
        def defs(name):
            tree=ast.parse((t.ROOT/'smoke'/name).read_text())
            return {f.name:ast.dump(f,include_attributes=False) for f in tree.body if isinstance(f,ast.FunctionDef)}
        before=defs('run_foundir_fullstack_blur_delta_v1.py');after=defs('run_foundir_fullstack_blur_coverage_v2.py')
        for name in ('setup','load_model','load_vae_prompt','rgb','cached','cache_items','prediction','cache_phase','optimizer_for'):
            self.assertEqual(before[name],after[name],name)
        before=defs('evaluate_foundir_blur_delta_official_v1.py');after=defs('evaluate_foundir_blur_coverage_official_v2.py')
        for name in ('assigned','metric_summary','run','finish'):
            self.assertEqual(before[name],after[name],name)
        self.assertEqual(t.RECIPE['train_steps'],200);self.assertEqual(t.RECIPE['development'],0)
        self.assertEqual(e.RECIPE['gate'],dict(psnr_min=25.9,lpips_max=.18))

if __name__=='__main__':unittest.main()
