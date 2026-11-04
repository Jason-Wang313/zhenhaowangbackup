import copy
from collections import Counter
from pathlib import Path
import tempfile
import os
import unittest
from unittest.mock import patch
import torch

from smoke import repair_foundir_frozen_image_router_v2 as r
from smoke import frozen_image_router_state_v2 as state
from lowlight_restoration.frozen_expert_router import FrozenExpertRouter


def candidates():
    rows=[]; old=[]
    for task in r.ORDER:
        for j in range(180):
            category=r.audit.CATEGORIES[task]; split='validation' if j==179 else 'train'
            record=f'foundir_train_{task}_{j}'
            row={'record_id':record,'split':split,'category':category,'source_group':f'{task}_{j}',
                 'gt_sha256':f'gt_{task}_{j}','lq_sha256':f'lq_{task}_{j}','composition_size':'1','width':'1024','height':'1024',
                 'lq_path':f'data/foundir/MillionIRData/Train/{category}/{record}_lq.png',
                 'gt_path':f'data/foundir/MillionIRData/Train/{category}/{record}_gt.png'}
            rows.append(row)
            if j==0 or split=='validation':
                old.append({'key':split+'_'+record,'split':split,'task':task,'seed':123,'row':{**row,'crop_xy':[3,4]}})
    return rows,old


def small_model():
    torch.manual_seed(11); m=FrozenExpertRouter(6,4,8)
    m.image.requires_grad_(False)
    return m,torch.optim.AdamW(m.vlm.parameters(),lr=.001,weight_decay=.01,foreach=False,fused=False)


class FrozenImageV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def test_selection_retains_original_and_never_uses_old_development(self):
        rows,old=candidates(); quotas={'fit':4,'calibration':3,'holdout':3}
        chosen=r.select(rows,old,quotas)
        self.assertEqual(len(chosen),50)
        self.assertEqual(sum(bool(i['reuse_original_key']) for i in chosen),5)
        self.assertFalse({i['key'] for i in chosen}&{i['key'] for i in old if i['split']=='validation'})
        self.assertEqual(chosen,r.select(list(reversed(rows)),old,quotas))
        r.validate_selection(chosen,old,quotas)

    def test_hash_or_source_cannot_cross_roles_and_protected_gt_excluded(self):
        rows,old=candidates()
        # Two records belonging to different source groups share an image identity.
        rows[2]['gt_sha256']=rows[1]['gt_sha256']
        rows[3]['gt_sha256']=old[-1]['row']['gt_sha256']
        chosen=r.select(rows,old,{'fit':4,'calibration':3,'holdout':3})
        self.assertNotIn(rows[3]['record_id'],{i['row']['record_id'] for i in chosen})
        for field in ('source_group','gt_sha256','lq_sha256'):
            owners={}
            for i in chosen:
                self.assertEqual(owners.setdefault(i['row'][field],i['role']),i['role'])

    def test_old_training_reuse_cannot_be_relabelled_or_held_out(self):
        rows,old=candidates(); q={'fit':4,'calibration':3,'holdout':3}; chosen=r.select(rows,old,q)
        selected=next(i for i in chosen if i['reuse_original_key'])
        selected['row']['crop_xy']=[0,0]
        # select retains old row identity; deepcopy old fixture before tamper validation.
        _,fresh_old=candidates()
        with self.assertRaises(ValueError): r.validate_selection(chosen,fresh_old,q)

    def test_insufficient_sources_rejected(self):
        rows,old=candidates()
        with self.assertRaisesRegex(ValueError,'insufficient'): r.select(rows[:10],old,{'fit':4,'calibration':3,'holdout':3})

    def test_training_samples_fit_only_exact_twenty_exposures(self):
        fit=[{'key':f'{t}_{j}','task':t,'role':'fit'} for t in r.ORDER for j in range(512)]
        counts=Counter(i for step in range(r.STEPS) for i in r.batch_indices(fit,step))
        self.assertEqual(len(counts),2560); self.assertEqual(set(counts.values()),{20})
        fit[0]['role']='holdout'
        with self.assertRaises(ValueError): r.batch_indices(fit,0)

    def test_eight_shards_complete_no_duplicates(self):
        records=list(range(2720)); shards=[records[i::8] for i in range(8)]
        self.assertEqual([len(x) for x in shards],[340]*8)
        self.assertEqual(sorted(j for x in shards for j in x),records)

    def test_update_freezes_image_and_trains_only_vlm(self):
        m,opt=small_model(); before=copy.deepcopy(m.image.state_dict()); vlm=copy.deepcopy(m.vlm.state_dict())
        x=torch.randn(32,6); v=torch.randn(32,4); y=torch.arange(32)%5
        for _ in range(3): r.update(m,opt,x,v,y)
        self.assertTrue(all(torch.equal(before[k],t) for k,t in m.image.state_dict().items()))
        self.assertTrue(any(not torch.equal(vlm[k],t) for k,t in m.vlm.state_dict().items()))
        self.assertTrue(all(p.grad is None for p in m.image.parameters()))

    def test_baseline_evaluation_is_resume_invariant(self):
        m,opt=small_model(); x=torch.randn(10,6); v=torch.randn(10,4); y=torch.arange(10)%5
        items=[{'key':str(j),'task':r.ORDER[j%5],'seed':j,'row':{'crop_xy':[0,0]}} for j in range(10)]
        data=(items,x,v,y); before=r.evaluate(m,data,False)
        r.update(m,opt,x,v,y)
        self.assertEqual(before,r.evaluate(m,data,False))

    def test_exact_next_update_resume_adam_rng_and_frozen_image(self):
        m,opt=small_model(); x=torch.randn(32,6); v=torch.randn(32,4); y=torch.arange(32)%5
        with tempfile.TemporaryDirectory() as temp:
            state.save(temp,m,opt,0,[],'identity')
            hist=[r.update(m,opt,x,v,y)]; state.save(temp,m,opt,1,hist,'identity')
            saved=state.load(temp,'identity'); expected_rand=torch.rand(4)
            r.update(m,opt,x,v,y)
            other,oo=small_model(); state.restore(other,oo,saved)
            self.assertTrue(torch.equal(expected_rand,torch.rand(4)))
            r.update(other,oo,x,v,y)
            self.assertTrue(all(torch.equal(p,other.state_dict()[k]) for k,p in m.state_dict().items()))
            for k,entry in opt.state_dict()['state'].items():
                for n,value in entry.items(): self.assertTrue(torch.equal(value,oo.state_dict()['state'][k][n]))

    def test_corrupt_state_or_missing_step_rejected(self):
        m,opt=small_model()
        with tempfile.TemporaryDirectory() as temp:
            state.save(temp,m,opt,0,[],'id'); marker=state.c.read(Path(temp)/'steps/step_000000.json')
            with self.assertRaises(ValueError): state.load(temp,'different')
            (Path(temp)/marker['file']).write_bytes(b'corrupt')
            with self.assertRaises(ValueError): state.load(temp,'id')
        with tempfile.TemporaryDirectory() as temp:
            state.save(temp,m,opt,1,[{}],'id')
            with self.assertRaisesRegex(ValueError,'gap'): state.load(temp,'id')

    def test_holdout_features_locked_before_selection(self):
        with tempfile.TemporaryDirectory() as temp,patch.object(r,'ROOT',Path(temp)):
            with self.assertRaisesRegex(ValueError,'locked'): r.load_role({'items':[]},{},'holdout')

    def test_corrected_evaluation_without_labels_as_model_input(self):
        m,_=small_model(); x=torch.randn(10,6); v=torch.randn(10,4); y=torch.arange(10)%5
        items=[{'key':str(j),'task':r.ORDER[j%5],'seed':j,'row':{'crop_xy':[0,0]}} for j in range(10)]
        first=r.evaluate(m,(items,x,v,y),True)
        second=r.evaluate(m,(items,x,v,(y+1)%5),True)
        self.assertEqual([i['logits'] for i in first['rows']],[i['logits'] for i in second['rows']])

    def test_full_1600_step_small_model_pipeline_and_locked_holdout(self):
        # Synthetic six/four-dimensional features, not a scientific GPU result.
        all_items=[]; datasets={}
        gen=torch.Generator().manual_seed(17)
        for role,count in r.QUOTAS.items():
            items=[{'key':f'{role}_{t}_{j}','task':t,'role':role,'seed':j,'row':{'crop_xy':[0,0]}}
                   for t in r.ORDER for j in range(count)]
            labels=torch.tensor([r.ORDER.index(i['task']) for i in items])
            datasets[role]=(items,torch.randn(len(items),6,generator=gen),torch.randn(len(items),4,generator=gen),labels)
            all_items.extend(items)
        loaded=[]
        with tempfile.TemporaryDirectory() as temp,patch.object(r,'ROOT',Path(temp)),patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':''}):
            output=r.ROOT/r.OUTPUT; r.c.write_new(output/'receipt.json',{'fixture':True})
            bound={'items':all_items}
            def load(_b,_i,role):
                if role=='holdout': self.assertTrue((output/'selection.json').is_file())
                loaded.append(role); return datasets[role]
            with patch.object(r,'receipt',return_value=bound),patch.object(r,'feature_index',return_value={}),\
                 patch.object(r,'load_role',side_effect=load),patch.object(r,'model',side_effect=lambda b:small_model()),\
                 patch('torch.set_num_interop_threads'),patch('torch.set_num_threads'),\
                 patch('shutil.disk_usage',return_value=type('Disk',(),{'free':100*1024**3})()):
                r.train(); r.verify()
                result=r.c.read(r.ROOT/r.REPORT)
                self.assertIn(result['selected_step'],r.CHECKS)
                self.assertEqual(len(list((output/'steps').glob('step_*.json'))),1601)
                self.assertEqual(loaded,['fit','calibration','holdout'])
                self.assertFalse(result['old40_pixels_or_features_accessed'])
                r.train(); self.assertEqual(loaded,['fit','calibration','holdout'])


if __name__=='__main__': unittest.main()
