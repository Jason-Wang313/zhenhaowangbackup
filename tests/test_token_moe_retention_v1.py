import copy
import json
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np
import torch
from safetensors.torch import save_file

from smoke.token_moe_retention_v1 import (CONFIG, TERMS, TASKS, RetentionSamples, batch_slot,
    validate_config, load_contract, retention_objective, reference_binding, read_reference, save_reference)
from smoke.token_residual_moe_v1 import write_once, sha
from smoke.token_moe_five_single_v1 import verify_parent
from lowlight_restoration.token_moe_retention_state import save_state, resume, rng_state
from lowlight_restoration.rain_blur_image_moe import load_expert_state, trainable_state
from test_token_residual_moe_v1 import model, token_conditioning
import test_token_moe_five_single_v1 as five_tests

ROOT = Path(__file__).resolve().parents[1]


class RetentionTests(unittest.TestCase):
    def cfg(self):
        return json.loads((ROOT / CONFIG).read_text())

    def samples(self):
        train, dev = five_tests.FiveTests().fixture()
        return RetentionSamples('.', self.cfg(), train, dev)

    def test_five_tasks_in_every_optimizer_update(self):
        s = self.samples()
        specs = s.all_specs()
        self.assertEqual(len(specs), 1500)
        self.assertEqual(len({x['key'] for x in specs}), 1500)
        self.assertEqual(len(s.evaluation_specs(specs)), 500)
        for step in range(25):
            selected = []
            for micro in range(5):
                for rank in range(8):
                    offset, scale = batch_slot(step, micro, rank)
                    self.assertEqual(scale, .2)
                    spec = s.train_spec(offset)
                    self.assertEqual(spec['task'], TASKS[micro])
                    self.assertEqual(spec['position'], step*8+rank)
                    selected.append(spec)
            for task in TASKS:
                self.assertEqual(sum(x['task'] == task for x in selected), 8)
        self.assertEqual(specs[400:1000], [s.train_spec(i) for i in range(10*40,25*40)])
        for index in (-1,1000):
            with self.assertRaises(ValueError): s.train_spec(index)
        with self.assertRaises(ValueError): batch_slot(0,5,0)
        with self.assertRaises(ValueError): batch_slot(0,0,8)

    def test_full_metadata_and_old_scientific_sources_unchanged(self):
        cfg, parent, _, _, _, train, dev, data = load_contract(ROOT, ROOT/CONFIG)
        self.assertEqual(sum(len(v) for v in train.values()), 311921)
        self.assertEqual(parent['training']['expert_lr'], 1e-6)
        self.assertEqual(parent['training']['router_lr'], 1e-4)
        self.assertTrue(cfg['fresh_optimizer'])
        self.assertFalse(cfg['restore_old_optimizer'])
        self.assertEqual(cfg['training_start_step'],0)
        verify_parent(ROOT,cfg,artifacts=False,sources=True)
        # The latest failed qualification binds all old scientific/launcher code.
        report = json.loads((ROOT/'reports/foundir_token_moe_five_single_v1_terminal_2026-11-01.json').read_text())
        for name, value in report['full_result']['binding']['source_sha256'].items():
            self.assertEqual(sha(ROOT/name),value,name)
        train_groups = {r['source_group'] for rows in train.values() for r in rows}
        train_gt = {r['gt_sha256'] for rows in train.values() for r in rows}
        self.assertFalse(train_groups & {r['source_group'] for rows in dev.values() for r in rows})
        self.assertFalse(train_gt & {r['gt_sha256'] for rows in dev.values() for r in rows})
        self.assertFalse(data['protected_pixels_accessed'])

    def test_scope_rejects_old_batch_extra_modules_and_formal(self):
        validate_config(self.cfg())
        for change in ({'global_batch':32},{'steps':1000},{'mode':'formal'},
                       {'automatic_formal':True},{'formal_entry_enabled':True},
                       {'expert_lr':1e-5},{'restormer':True},{'output_root':'../outside'}):
            with self.assertRaises(RuntimeError): validate_config({**self.cfg(),**change})

    def loss(self, factor=1., ref=None):
        ref = ref or {k:.1 for k in TERMS}
        terms = {k:torch.tensor(ref[k]*factor,requires_grad=True) for k in TERMS}
        weights = dict(flow_mse=.1,latent_l1=.25,rgb_mse=10.,charbonnier=1.,highpass=.1,lpips=.05)
        loss, info = retention_objective(terms,ref,weights,self.cfg()['objective'])
        return loss,info,terms

    def test_at_parent_zero_harm_and_unit_relative_loss(self):
        loss,info,terms=self.loss()
        self.assertAlmostEqual(float(loss),1.,places=6)
        self.assertEqual(float(info['mse_regression']),0.)
        self.assertEqual(float(info['lpips_regression']),0.)
        loss.backward()
        for value in terms.values(): self.assertGreater(float(value.grad),0.)

    def test_fractional_error_scaling_not_absolute_task_scale(self):
        a,_,_=self.loss(.8)
        b,_,_=self.loss(.8, {k:10. for k in TERMS})
        self.assertAlmostEqual(float(a),.8,places=6)
        self.assertAlmostEqual(float(b),.8,places=6)

    def test_regression_penalizes_both_metrics_separately(self):
        base,_,t=self.loss(); base.backward()
        base_grad = {k:float(v.grad) for k,v in t.items()}
        loss,info,terms=self.loss(1.1); loss.backward()
        self.assertAlmostEqual(float(info['mse_regression']),.1,places=6)
        self.assertAlmostEqual(float(info['lpips_regression']),.1,places=6)
        self.assertAlmostEqual(float(loss),2.1,places=5)
        for k in ('rgb_mse','lpips'):
            self.assertGreater(float(terms[k].grad),base_grad[k])

    def test_near_zero_floors_finite_and_no_hidden_harm_allowance(self):
        ref={k:0. for k in TERMS}
        terms={k:torch.tensor(1e-8,requires_grad=True) for k in TERMS}
        loss,info=retention_objective(terms,ref,{k:1. for k in TERMS},self.cfg()['objective'])
        loss.backward()
        self.assertGreater(float(info['mse_regression']),0.)
        self.assertGreater(float(info['lpips_regression']),0.)
        self.assertTrue(all(torch.isfinite(v.grad) for v in terms.values()))

    def test_no_validation_internal_or_gt_features_in_reference(self):
        specs=self.samples().all_specs()
        for spec in (specs[-1], {**specs[0],'row':{**specs[0]['row'],'split':'internal_test'}},
                     {**specs[0],'row':{**specs[0]['row'],'gt_path':'data/official/test.png'}}):
            with self.assertRaises(RuntimeError): reference_binding(spec,{})
        identity=reference_binding(specs[0],{'parent':'2000'})
        ref={k:.1 for k in TERMS}
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'reference.json'
            save_reference(path,ref,identity)
            self.assertEqual(read_reference(path,identity),ref)
            with self.assertRaises(RuntimeError): read_reference(path,{**identity,'sample_sha256':'different'})
            with self.assertRaises(RuntimeError): save_reference(path,{**ref,'lpips':.2},identity)

    def test_nonfinite_and_trainable_reference_rejected(self):
        terms={k:torch.tensor(.1,requires_grad=True) for k in TERMS}
        ref={k:.1 for k in TERMS}
        for value in (float('nan'),-1.,torch.tensor(.1,requires_grad=True)):
            with self.assertRaises(RuntimeError):
                retention_objective(terms,{**ref,'lpips':value},{k:1 for k in TERMS},self.cfg()['objective'])

    def test_strict_weights_preserve_trained_vlm_and_reject_missing(self):
        old=model()
        with torch.no_grad(): old.block.vlm_projector[-1].weight.add_(.7)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'parent.safetensors'
            state=trainable_state(old); save_file(state,str(path))
            new=model(); load_expert_state(new,path)
            for a,b in zip(old.parameters(),new.parameters()): torch.testing.assert_close(a,b,rtol=0,atol=0)
            state.pop(next(k for k in state if 'vlm_projector' in k)); save_file(state,str(path))
            with self.assertRaises(RuntimeError): load_expert_state(new,path)

    def test_batch40_adam_rng_roundtrip_and_new_sample_position(self):
        net=model(); opt=torch.optim.AdamW(net.parameters(),lr=1e-3,weight_decay=0)
        x,f=torch.randn(1,3,6),torch.randn(1,4)
        def step(n,o):
            o.zero_grad()
            with token_conditioning(n,f): n(x).square().mean().backward()
            o.step()
        for _ in range(3): step(net,opt)
        binding={'mode':'qualification','source':'fixed','global_batch':40}
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); write_once(root/'launch.json',{'binding':binding})
            receipt=save_state(net,opt,root,step=3,binding=binding,rank_rng=[rng_state() for _ in range(8)])
            self.assertEqual(receipt['next_sample_position'],120)
            expected=(random.random(),np.random.rand(),torch.rand(4))
            path=root/'step_000003.resume.json'
            new=model(); o=torch.optim.AdamW(new.parameters(),lr=1e-3,weight_decay=0)
            self.assertEqual(resume(new,o,path,binding,3,25),3)
            self.assertEqual(random.random(),expected[0]); self.assertEqual(np.random.rand(),expected[1])
            torch.testing.assert_close(torch.rand(4),expected[2],rtol=0,atol=0)
            step(net,opt); step(new,o)
            for a,b in zip(net.parameters(),new.parameters()): torch.testing.assert_close(a,b,rtol=0,atol=0)
            for a,b in zip(opt.state.values(),o.state.values()):
                for key in ('step','exp_avg','exp_avg_sq'): torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)
            with self.assertRaises(RuntimeError): resume(new,o,path,{**binding,'mode':'formal'},0,1000)
            badopt=torch.optim.AdamW(new.parameters(),lr=1e-4,weight_decay=0)
            with self.assertRaises(RuntimeError): resume(new,badopt,path,binding,0,25)
            with self.assertRaises(RuntimeError): save_state(net,opt,root,step=3,binding=binding,rank_rng=[])

    def test_old_batch32_receipt_rejected(self):
        net=model(); opt=torch.optim.AdamW(net.parameters())
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'old.json'
            write_once(path,{'mode':'qualification','step':3,'next_sample_position':96,'global_batch':32})
            with self.assertRaises(RuntimeError): resume(net,opt,path,{'mode':'qualification'},0,25)


if __name__=='__main__': unittest.main()
