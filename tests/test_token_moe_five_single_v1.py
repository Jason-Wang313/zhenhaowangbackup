import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from smoke.token_moe_five_single_v1 import (CONFIG, TASKS, CATEGORIES, FiveSamples,
    load_contract, verify_parent, balanced_preview, development_crop, expand_protocol)
from smoke.token_residual_moe_v1 import write_once
from smoke.train_foundir_token_residual_moe_v1 import training_objective, resume_training
from lowlight_restoration.joint_moe_state import save_state, rng_state
from test_token_residual_moe_v1 import model, token_conditioning

ROOT = Path(__file__).resolve().parents[1]


def row(key, task, split, group=None):
    return dict(record_id=key, source_group=group or key, split=split, category=CATEGORIES[task],
                composition_size='1', role='direct_endpoint_train', width='1920', height='1080',
                lq_path=f'data/foundir/MillionIRData/Train/{key}_lq.png',
                gt_path=f'data/foundir/MillionIRData/Train/{key}_gt.png',
                lq_sha256=key+'lq', gt_sha256=key+'gt')


class FiveTests(unittest.TestCase):
    def cfg(self):
        return json.loads((ROOT / CONFIG).read_text())

    def fixture(self):
        training = {t: [row('train_'+t, t, 'train')] for t in TASKS}
        development = {'rain': [row('rain_dev'+str(i), 'rain', 'validation') for i in range(100)]}
        for t in TASKS[1:]:
            development[t+'_preview'] = [row(t+'_dev'+str(i), t, 'validation', t+'_group'+str(i%10))
                                         for i in range(100)]
        return training, development

    def test_all_five_balanced_on_eight_ranks_and_resume_positions(self):
        cfg = self.cfg(); t,d = self.fixture(); s = FiveSamples('.',cfg,t,d)
        specs = s.all_specs()
        self.assertEqual(len(specs), 1300)
        self.assertEqual(len({r['key'] for r in specs}), 1300)
        self.assertEqual(len(s.evaluation_specs(specs)), 500)
        for task in TASKS:
            self.assertEqual(sum(r.get('task') == task for r in specs[:800]), 160)
        for step in range(2000, 2025, 5):
            period = [s.train_spec(i) for i in range(step*32,(step+5)*32)]
            for task in TASKS:
                self.assertEqual(sum(r['task'] == task for r in period), 32)
        resumed = [s.train_spec(i) for i in range(2010*32,2025*32)]
        self.assertEqual(resumed, specs[320:800])
        with self.assertRaises(ValueError): s.train_spec(1999*32)
        with self.assertRaises(RuntimeError): s.evaluation_specs(specs, full=True)

    def test_preview_is_deterministic_and_source_balanced(self):
        rows = [row('n'+str(i), 'noise', 'validation', 'g'+str(i%7)) for i in range(140)]
        a = balanced_preview(rows); b = balanced_preview(list(reversed(rows)))
        self.assertEqual(a,b)
        counts = [sum(r['source_group']=='g'+str(i) for r in a) for i in range(7)]
        self.assertLessEqual(max(counts)-min(counts),1)

    def test_historical_crop_formula_and_protected_rejection(self):
        r = row('example','haze','validation'); seed=2026091702
        c=development_crop(r,seed)
        raw=hashlib.sha256(b'2026091702:full_development_crop:example').digest()
        self.assertEqual(int(c['crop_y']),int.from_bytes(raw[:8],'big')%(1080-512+1))
        self.assertEqual(int(c['crop_x']),int.from_bytes(raw[8:16],'big')%(1920-512+1))
        with self.assertRaises(RuntimeError): development_crop(dict(r,split='internal_test'),seed)
        with self.assertRaises(RuntimeError): development_crop(dict(r,lq_path='/protected/test.png'),seed)

    def test_global_nontrain_exclusion_and_old_model_leak_rejection(self):
        train,dev = self.fixture()
        refs={t:{'data':{'expected_records':100,'expected_source_groups':10,'crop_seed':1}}
              for t in ('noise','haze','low_light')}
        manifest=[r for rows in train.values() for r in rows]+[r for rows in dev.values() for r in rows]
        blocked=row('internal_noise','noise','internal_test', 'forbidden')
        candidate=row('candidate','noise','train','forbidden')
        manifest += [blocked,candidate]
        oldtrain={t:train[t] for t in ('rain','blur')}
        olddev={'rain':dev['rain'],'blur_preview':dev['blur_preview'],'blur_full':dev['blur_preview']}
        t,d,meta=expand_protocol(manifest,oldtrain,olddev,refs)
        self.assertEqual(meta['excluded_nontrain_overlap_records']['noise'],1)
        self.assertEqual([r['record_id'] for r in t['noise']],['train_noise'])
        self.assertFalse(meta['protected_pixels_accessed'])
        bad=copy.deepcopy(oldtrain); bad['rain'][0]['source_group']='forbidden'
        with self.assertRaises(RuntimeError): expand_protocol(manifest,bad,olddev,refs)
        with self.assertRaises(RuntimeError): expand_protocol(manifest+[manifest[0]],oldtrain,olddev,refs)

    def test_parent_report_and_wrong_receipt(self):
        cfg=self.cfg(); result=verify_parent(ROOT,cfg,artifacts=False,sources=False)
        self.assertEqual(result['start_step'],2000)
        self.assertEqual(result['mode'],'formal')
        cfg['parent_receipt_sha256']='wrong'
        with self.assertRaises(RuntimeError): verify_parent(ROOT,cfg,artifacts=False,sources=False)

    def test_forbid_extra_config_and_automatic_long_run(self):
        for change in ({'steps':12000},{'automatic_formal':True},{'experts_per_block':5},{'router_ce_weight':.01}):
            cfg=self.cfg(); cfg.update(change)
            with tempfile.TemporaryDirectory() as folder:
                path=Path(folder)/'bad.json'; write_once(path,cfg)
                with self.assertRaises(RuntimeError): load_contract(ROOT,path)

    def test_five_class_id_never_enters_two_class_ce_and_gradients_flow(self):
        net=model()
        with patch('smoke.train_foundir_token_residual_moe_v1.router_classification_loss',side_effect=AssertionError('CE forbidden')):
            with token_conditioning(net,torch.randn(1,4)):
                loss=net(torch.randn(1,3,6)).square().mean()
                objective=training_objective(loss,net,{'task_id':4},{},self.cfg(),'cpu')
                self.assertIs(objective,loss)
                objective.backward()
        for name,p in net.named_parameters():
            self.assertIsNotNone(p.grad,name)
            self.assertTrue(torch.isfinite(p.grad).all(),name)
        self.assertGreater(float(net.block.vlm_projector[-1].weight.grad.abs().sum()),0)

    def test_explicit_formal_to_qualification_transition_preserves_adam(self):
        net=model(); opt=torch.optim.AdamW(net.parameters(),lr=1e-3,weight_decay=0)
        x,f=torch.randn(1,3,6),torch.randn(1,4)
        def step(n,o):
            o.zero_grad()
            with token_conditioning(n,f): n(x).square().mean().backward()
            o.step()
        for _ in range(3): step(net,opt)
        binding={'config_sha256':'parent','source_sha256':{'code':'parent'}}
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); write_once(root/'launch.json',{'binding':binding})
            save_state(net,opt,root,step=3,mode='formal',arm='token_residual',config_sha='parent',
                       sources=binding['source_sha256'],rank_rng=[rng_state() for _ in range(8)])
            n=model(); o=torch.optim.AdamW(n.parameters(),lr=1e-3,weight_decay=0)
            receipt=root/'step_000003.resume.json'
            with self.assertRaises(RuntimeError): resume_training(n,o,receipt,binding,0,mode='qualification',total_steps=28)
            self.assertEqual(resume_training(n,o,receipt,binding,0,mode='formal',total_steps=3),3)
            for a,b in zip(net.parameters(),n.parameters()): torch.testing.assert_close(a,b,rtol=0,atol=0)
            step(net,opt); step(n,o)
            for a,b in zip(net.parameters(),n.parameters()): torch.testing.assert_close(a,b,rtol=0,atol=0)
            new=root/'qualification'; write_once(new/'launch.json',{'binding':binding})
            save_state(n,o,new,step=4,mode='qualification',arm='token_residual',config_sha='parent',
                       sources=binding['source_sha256'],rank_rng=[rng_state() for _ in range(8)])
            other=model(); oo=torch.optim.AdamW(other.parameters(),lr=1e-3,weight_decay=0)
            self.assertEqual(resume_training(other,oo,new/'step_000004.resume.json',binding,0,
                                            mode='qualification',total_steps=28),4)


if __name__=='__main__': unittest.main()
