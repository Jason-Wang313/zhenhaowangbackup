import copy
import json
from pathlib import Path
import tempfile
import subprocess
import sys
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch
from smoke import official_vlm_router_v4_contract as x
from smoke import evaluate_foundir_official_vlm_router_v4 as run


def rows_fixture():
    return [{'record_id':f'official_{cat}_{j:04d}','category':cat,'pair_id':f'{j:04d}',
             'width':'513','height':'517',
             **{side+'_path':f'downloads/foundir_official/extracted_{side}/{cat}/{j:04d}.png' for side in ('lq','gt')},
             'lq_sha256':'1'*64,'gt_sha256':'2'*64}
            for cat,n in x.COUNTS.items() for j in range(1,n+1)]


def decision(a,b):
    return x.decisions([[float(t==a) for t in x.ORDER]],[[float(t==b) for t in x.ORDER]])


class OfficialV4Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_two_and_four_node_contracts_keep_same_scope_seed_and_total_budget(self):
        results=[]
        for nodes in (2,4):
            code="from smoke import official_vlm_router_v4_contract as x; import json; print(json.dumps({'world':x.WORLD,'run':x.RUN,'recipe':x.RECIPE,'seed':x.tiles({'record_id':'official_01Blur_0001','height':'517','width':'513'})}))"
            value=json.loads(subprocess.check_output([sys.executable,'-S','-c',code],env={**os.environ,'FOUNDIR_OFFICIAL_NNODES':str(nodes)},text=True))
            self.assertEqual(value['world'],nodes*2)
            self.assertEqual(value['recipe']['route_GPU_cap_seconds_per_node']*value['world']/3600,12)
            results.append(value)
        self.assertEqual(results[0]['seed'],results[1]['seed'])
        self.assertNotEqual(results[0]['run'],results[1]['run'])
        invalid=subprocess.run([sys.executable,'-S','-c','from smoke import official_vlm_router_v4_contract'],env={**os.environ,'FOUNDIR_OFFICIAL_NNODES':'3'},capture_output=True)
        self.assertNotEqual(invalid.returncode,0)

    def test_only_exact_official600_scope(self):
        rows=rows_fixture();x.validate_rows(rows)
        for mutate in ('missing','duplicate','other_set','mixed','path_escape'):
            bad=copy.deepcopy(rows)
            if mutate=='missing':bad.pop()
            elif mutate=='duplicate':bad[0]=bad[1]
            elif mutate=='other_set':bad[0]['lq_path']='data/train/foo.png'
            elif mutate=='mixed':bad[0]['category']='02Blur_Noise'
            else:bad[0]['gt_path']='downloads/foundir_official/../secret.png'
            with self.assertRaises(ValueError):x.validate_rows(bad)

    def test_shards_complete_disjoint(self):
        rows=rows_fixture();parts=[rows[k::x.WORLD] for k in range(x.WORLD)]
        self.assertEqual([len(p) for p in parts],[600//x.WORLD]*x.WORLD)
        self.assertEqual(len({r['record_id'] for p in parts for r in p}),600)

    def test_origins_cover_and_no_duplicate_edges(self):
        for n in (1,2,511,512,513,1080,1920,3984):
            pos=x.origins(n);self.assertEqual(len(set(pos)),len(pos));covered=np.zeros(n,bool)
            for k in pos:covered[k:min(n,k+512)]=True
            self.assertTrue(covered.all());self.assertEqual(pos[-1],max(n-512,0))
        for bad in (0,-1,1.5):
            with self.assertRaises(ValueError):x.origins(bad)

    def test_identity_stitch_tiny_odd_and_large_shapes(self):
        rng=np.random.default_rng(8)
        for h,w in ((1,1),(1,17),(17,1),(7,13),(512,512),(517,513),(1080,1920)):
            array=rng.integers(0,256,(h,w,3),dtype=np.uint8)
            row={'record_id':'official_01Blur_0001','width':str(w),'height':str(h)}
            preds={t['key']:x.tile_array(array,t)[0] for t in x.tiles(row)}
            self.assertTrue(np.array_equal(x.stitch(row,preds),array))
            self.assertEqual(x.stitch(row,preds).shape,array.shape)

    def test_stitch_rejects_missing_extra_and_wrong_dtype(self):
        row=rows_fixture()[0];pred={t['key']:np.zeros((512,512,3),np.uint8) for t in x.tiles(row)}
        key=next(iter(pred));bad=dict(pred);bad.pop(key)
        with self.assertRaises(ValueError):x.stitch(row,bad)
        bad=dict(pred);bad['extra']=bad[key]
        with self.assertRaises(ValueError):x.stitch(row,bad)
        bad=dict(pred);bad[key]=bad[key].astype(np.float32)
        with self.assertRaises(ValueError):x.stitch(row,bad)

    def test_seeds_identical_across_arms_and_no_ground_truth_dependency(self):
        row=rows_fixture()[0];first=x.tiles(row)
        row['gt_sha256']='9'*64;row['category']='14Lowlight'
        self.assertEqual(x.tiles(row),first)
        self.assertTrue(all(0<=t['seed']<2**32 for t in first))

    def test_global_average_logits_and_reference_do_not_change_deployed(self):
        d=x.decisions([[2.,1.,0.,0.,0.],[0.,3.,0.,0.,0.]],[[0.,0.,2.,0.,0.],[0.,0.,1.,0.,0.]])
        self.assertEqual(d['image_only']['anchor'],'noise');self.assertEqual(d['conditional_v4']['anchor'],'haze')
        row=rows_fixture()[0];a=x.anchors(row,d);row['category']='10Rain';b=x.anchors(row,d)
        self.assertEqual(a['image_only'],b['image_only']);self.assertEqual(a['conditional_v4'],b['conditional_v4'])
        self.assertNotEqual(a['known_class_reference'],b['known_class_reference'])

    def test_saved_decision_tampering_and_nonfinite_rejected(self):
        d=decision('blur','noise');d['conditional_v4']['anchor']='rain'
        with self.assertRaises(ValueError):x.anchors(rows_fixture()[0],d)
        with self.assertRaises(ValueError):x.decisions([[float('nan')]*5],[[0.]*5])
        with self.assertRaises(ValueError):x.decisions([],[])

    def test_official_aggregation_image_weighting_not_class_macro(self):
        rows=rows_fixture();records={}
        for row in rows:
            label={v:k for k,v in x.CATEGORIES.items()}[row['category']]
            d=decision(label,label);value=10. if label=='haze' else 40.
            records[row['record_id']]={'decisions':d,'metrics':{label:{'psnr':value,'lpips':.01,'ssim':.9}}}
        result=x.aggregate(rows,records)
        self.assertAlmostEqual(result['arms']['conditional_v4']['overall']['psnr'],30.)
        self.assertEqual(result['arms']['conditional_v4']['correct_routes'],600)
        self.assertFalse(result['all_ten_SoW_checks_pass']);self.assertFalse(result['full_20_category_test'])
        self.assertFalse(result['independent_blind_test'])
        records.pop(next(iter(records)))
        with self.assertRaises(ValueError):x.aggregate(rows,records)

    def test_exact_three_arm_dedup_and_gt_only_after_saved_outputs(self):
        row=rows_fixture()[-1];row.update(width='7',height='9')
        d=decision('blur','noise');array=np.arange(9*7*3,dtype=np.uint8).reshape(9,7,3)
        bank=SimpleNamespace(load_bank=lambda *a:('model','gates','rules','ae','prompts'))
        cache={};calls=[]
        with tempfile.TemporaryDirectory() as temp,patch.object(run,'ROOT',Path(temp)):
            out=Path(temp)/x.OUTPUT
            x.c.write_new(out/'receipt.json',{});x.c.write_new(out/'runtime_node0.json',{})
            x.c.write_new(out/'render_plan.json',{})
            x.c.write_new(out/'render_budget.json',{'render_plan_sha256':x.c.sha(out/'render_plan.json')})
            dec={'receipt_sha256':x.c.sha(out/'receipt.json'),'record_id':row['record_id'],
                 'lq_sha256':row['lq_sha256'],'decisions':d,'GT_input':False,'class_label_input':False}
            x.c.write_new(out/'decisions'/(row['record_id']+'.json'),dec)
            def pixels(r,side):
                if side=='gt':self.assertEqual(len(list((out/'restored').glob('*.png'))),3)
                return array.copy()
            def restore(*args,**kw):
                weights=args[3];anchor=x.ORDER[int(weights.argmax())];calls.append((anchor,kw['seed']))
                lq=args[5];u8=((lq+1)*127.5).round().byte()
                return lq,u8,{'anchor':anchor,'model_calls':26 if anchor=='rain' else 50}
            def save(impl,folder,i,values,node,meta=None):cache[i['key']]=(meta,values)
            metric={k:(lambda p,g:torch.tensor(.1)) for k in ('psnr','lpips','ssim')}
            with patch.object(run,'setup',return_value=({'rows':[row],'original':{},'metric_identity':{}},bank,0,torch.device('cpu'))), \
                 patch.object(run,'rendering_plan',return_value={}),patch.object(run,'input_image',side_effect=pixels), \
                 patch.object(run,'metric_models',return_value=(metric,{})),patch.object(run,'save',side_effect=save), \
                 patch.object(run,'cache',side_effect=lambda impl,folder,i,node:cache.get(i['key'])), \
                 patch('smoke.expanded_vlm_h100_runtime_v4.bounded_vae',side_effect=lambda ae:ae), \
                 patch('lowlight_restoration.five_expert_flux.restore_five_expert',side_effect=restore), \
                 patch('torch.cuda.max_memory_allocated',return_value=0):
                run.render(0)
                self.assertEqual(len(calls),3);self.assertEqual(len({seed for _,seed in calls}),1)
                run.render(0);self.assertEqual(len(calls),3)  # Verified resume does not regenerate.
                rec=run.read_result(row,0);self.assertEqual(set(rec['outputs']),{'blur','noise','rain'})
                p=out/rec['outputs']['blur']['path']
                p.write_bytes(b'corrupt synthetic fixture')
                with self.assertRaises(ValueError):run.read_result(row,0)

    def test_budget_requires_fixed_plan_and_explicit_note(self):
        with tempfile.TemporaryDirectory() as temp,patch.object(run,'ROOT',Path(temp)),patch.object(run,'bound',return_value={}), \
             patch.object(run,'rendering_plan',return_value={'fixed':True}):
            out=Path(temp)/x.OUTPUT;x.c.write_new(out/'render_plan.json',{'fixed':True})
            for hours,note in ((0,'explicit user approval'),(float('nan'),'explicit user approval'),(10,'guess')):
                with self.assertRaises(ValueError):run.authorize_render(hours,note)
            run.authorize_render(120,'SYNTHETIC TEST ONLY explicit user approval')
            b=x.c.read(out/'render_budget.json');self.assertEqual(b['cap_seconds_per_node'],int(120*3600/x.WORLD))
            self.assertEqual(b['render_plan_sha256'],x.c.sha(out/'render_plan.json'))

    def test_route_plan_binds_all_workers_and_feature_tensors_before_GT(self):
        rows=rows_fixture()
        for row in rows:row.update(width='1',height='1')
        with tempfile.TemporaryDirectory() as temp,patch.object(run,'ROOT',Path(temp)):
            out=Path(temp)/x.OUTPUT;x.c.write_new(out/'receipt.json',{})
            receipt_hash=x.c.sha(out/'receipt.json')
            for node in range(x.NODES):x.c.write_new(out/f'runtime_node{node}.json',{'node':node})
            for rank in range(x.WORLD):
                rh=x.c.sha(out/f'runtime_node{rank//2}.json')
                assigned=rows[rank::x.WORLD]
                x.c.write_new(out/f'route_worker{rank}.json',{'receipt_sha256':receipt_hash,'runtime_sha256':rh,
                            'rank':rank,'records':[r['record_id'] for r in assigned],'peak_allocated_gib':0})
                for row in assigned:
                    tile=x.tiles(row)[0];file=out/'synthetic_tensors'/(tile['key']+'.bin');file.parent.mkdir(exist_ok=True)
                    file.write_bytes(b'synthetic fixture, not a real feature tensor')
                    name=file.relative_to(out).as_posix()
                    feature={'identity':{'receipt_sha256':receipt_hash,'item':run.item(row,tile),'arm':'features','runtime_sha256':rh},
                             'GT_input':False,'tensor':name,'files':{name:x.c.sha(file)}}
                    path=out/'features'/(tile['key']+'.json');x.c.write_new(path,feature)
                    d={'receipt_sha256':receipt_hash,'record_id':row['record_id'],'lq_sha256':row['lq_sha256'],
                       'GT_input':False,'class_label_input':False,'decisions':decision('blur','blur'),
                       'feature_receipts':{tile['key']:x.c.sha(path)}}
                    x.c.write_new(out/'decisions'/(row['record_id']+'.json'),d)
            result=run.rendering_plan({'rows':rows})
            self.assertEqual(result['unique_native_tile_generations'],1050)
            self.assertFalse(result['GT_pixels_read']);self.assertFalse(result['render_authorized'])
            file.write_bytes(b'changed synthetic tensor')
            with self.assertRaisesRegex(ValueError,'feature tensor changed'):run.rendering_plan({'rows':rows})

    def test_cost_accounting_includes_failed_attempts_and_rejects_orphans(self):
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp)
            keys=[('prepare',0),('plan',0)]+[(f'{p}_node{n}',2) for p in ('route','render') for n in range(x.NODES)]
            for key,gpu in keys:
                for j,status in enumerate((1,0)):
                    x.c.write_new(out/'attempts'/f'{key}_{j}.started.json',{})
                    x.c.write_new(out/'attempts'/f'{key}_{j}.ended.json',{'seconds':10,'status':status,'gpus':gpu,'used_gpu_seconds':10*gpu})
            r=run.costs(out);self.assertAlmostEqual(r['used_gpu_hours'],(x.NODES*80)/3600)
            x.c.write_new(out/'attempts'/'render_node0_3.started.json',{})
            with self.assertRaises(ValueError):run.costs(out)


if __name__=='__main__':unittest.main()
