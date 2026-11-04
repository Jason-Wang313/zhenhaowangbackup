import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from lowlight_restoration.supervised_singleton_router import FrozenExpertRouter, infer, singleton_weights, train_batch
from smoke import router_supervised_state_v1 as state
from smoke import train_foundir_router_supervised_repair_v1 as runner


def items():
    return [{'key': f'{split}_{task}_{j}', 'task': task, 'split': split, 'seed': j,
             'row': {'split': split, 'source_group': f'{split}_{task}_{j}', 'gt_sha256': f'{split}_{task}_{j}',
                     'composition_size': '1', 'crop_xy': [0, 0]}}
            for split,n in [('train',128), ('validation',8)] for task in runner.ORDER for j in range(n)]


def small_model():
    torch.manual_seed(5)
    model = FrozenExpertRouter(6, 4, 8)
    return model, torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=0., foreach=False, fused=False)


class RepairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_split_and_exact_twenty_exposures(self):
        rows = items(); runner.validate_split(rows)
        train = [i for i in rows if i['split'] == 'train']
        from collections import Counter
        slots = Counter(j for step in range(400) for j in runner.training_indices(train, step))
        self.assertEqual(len(slots), 640)
        self.assertEqual(set(slots.values()), {20})

    def test_reject_development_in_training_and_source_overlap(self):
        rows = items()
        with self.assertRaises(ValueError): runner.training_indices(rows, 0)
        rows[-1]['row']['source_group'] = rows[0]['row']['source_group']
        with self.assertRaisesRegex(ValueError, 'overlap'): runner.validate_split(rows)

    def test_top1_uses_logits_only_preserves_ties_and_rejects_nan(self):
        logits = torch.tensor([[2., 2., -1., 0., 1.], [-1.,0.,1.,2.,3.]], requires_grad=True)
        weights = singleton_weights(logits)
        self.assertTrue(torch.equal(weights, torch.tensor([[1.,0.,0.,0.,0.], [0.,0.,0.,0.,1.]])))
        self.assertFalse(weights.requires_grad)
        with self.assertRaises(ValueError): singleton_weights(torch.full((1,5), float('nan')))

    def test_vlm_supervised_from_first_step(self):
        model,opt = small_model(); x=torch.randn(32,6); v=torch.randn(32,4); y=torch.arange(32)%5
        before = copy.deepcopy(model.vlm.state_dict())
        record = train_batch(model,opt,x,v,y,arm='image_plus_vlm')
        self.assertGreater(record['gradient_norms']['vlm'], 0)
        self.assertFalse(torch.equal(model.vlm[-1].weight, before['3.weight']))
        train_batch(model,opt,x,v,y,arm='image_plus_vlm')
        self.assertGreater(float(model.vlm[1].weight.grad.norm()),0)

    def test_image_only_control_never_updates_vlm(self):
        model,opt = small_model(); before=copy.deepcopy(model.vlm.state_dict())
        x=torch.randn(32,6); v=torch.randn(32,4); y=torch.arange(32)%5
        for _ in range(4):
            r=train_batch(model,opt,x,v,y,arm='image_only'); self.assertEqual(r['gradient_norms']['vlm'],0)
        self.assertTrue(all(torch.equal(p, before[k]) for k,p in model.vlm.state_dict().items()))
        a=infer(model,x,v,arm='image_only'); b=infer(model,x,v*100,arm='image_only')
        self.assertTrue(torch.equal(a['logits'], b['logits']))

    def test_both_arms_have_identical_fresh_initialization(self):
        cfg={'seed':5,'image_dim':6,'vlm_dim':4,'hidden':8,'learning_rate':.001,'weight_decay':0.}
        a,oa=runner.make_model(cfg); b,ob=runner.make_model(cfg)
        self.assertEqual(state.io.parameter_hash(state.model_state(a)),state.io.parameter_hash(state.model_state(b)))
        self.assertEqual(len(oa.state),0); self.assertEqual(len(ob.state),0)

    def test_exact_next_update_after_checkpoint_restore(self):
        model,opt=small_model(); x=torch.randn(32,6); v=torch.randn(32,4); y=torch.arange(32)%5
        with tempfile.TemporaryDirectory() as temp:
            state.save(temp,model,opt,step=0,history=[],receipt_hash='r',arm='image_plus_vlm')
            r=train_batch(model,opt,x,v,y,arm='image_plus_vlm')
            state.save(temp,model,opt,step=1,history=[r],receipt_hash='r',arm='image_plus_vlm')
            saved=state.load(temp,'r','image_plus_vlm')
            expected_random=torch.rand(5)
            train_batch(model,opt,x,v,y,arm='image_plus_vlm')
            other,oo=small_model(); state.restore(other,oo,saved)
            self.assertTrue(torch.equal(torch.rand(5),expected_random))
            train_batch(other,oo,x,v,y,arm='image_plus_vlm')
            self.assertTrue(all(torch.equal(p,dict(other.named_parameters())[n]) for n,p in model.named_parameters()))
            for key,entry in opt.state_dict()['state'].items():
                for name,value in entry.items(): self.assertTrue(torch.equal(value,oo.state_dict()['state'][key][name]))

    def test_corrupt_checkpoint_and_arm_mismatch_rejected(self):
        model,opt=small_model()
        with tempfile.TemporaryDirectory() as temp:
            marker=state.save(temp,model,opt,step=0,history=[],receipt_hash='r',arm='image_only')
            with self.assertRaises(ValueError): state.load(temp,'r','image_plus_vlm')
            with self.assertRaises(ValueError): state.load(temp,'other','image_only')
            (Path(temp)/marker['file']).write_bytes(b'corrupt test fixture')
            with self.assertRaisesRegex(ValueError,'hash/path'): state.load(temp,'r','image_only')

    def test_development_features_locked_before_two_terminal_steps(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(runner,'ROOT',Path(temp)):
            with self.assertRaisesRegex(ValueError,'locked'):
                runner.load_features({'config':{'arms':['image_only','image_plus_vlm']}},'validation')

    def test_cpu_loader_reads_only_image_and_vlm_not_reference(self):
        from safetensors.torch import save_file
        with tempfile.TemporaryDirectory() as temp, patch.object(runner,'ROOT',Path(temp)):
            root=Path(temp); file=root/'fixture.safetensors'
            save_file({'image':torch.zeros(1,6144),'vlm':torch.ones(1,3584),'reference':torch.full((2,),float('nan'))},str(file))
            item=items()[0]
            bound={'items':[item],'features':{item['key']:{'path':'fixture.safetensors','sha256':runner.c.sha(file)}}}
            selected,x,v,y=runner.load_features(bound,'train')
            self.assertEqual(selected,[item]); self.assertTrue(torch.isfinite(x).all() and torch.isfinite(v).all())

    def test_full_matched_400_step_flow_on_tiny_fixture_and_json_verification(self):
        cfg={'seed':5,'image_dim':6,'vlm_dim':4,'hidden':8,'learning_rate':.001,'weight_decay':0.,
             'steps_per_arm':400,'arms':['image_only','image_plus_vlm'],'deployment_policy':'predicted_top1_singleton_only'}
        rows=items(); bound={'config':cfg,'items':rows}; order=[]
        def dataset(b,split):
            if split == 'validation':
                self.assertTrue(all((runner.ROOT/runner.OUTPUT/a/'steps/step_000400.json').exists() for a in cfg['arms']))
            order.append(split); selected=[i for i in rows if i['split']==split]
            y=torch.tensor([runner.ORDER.index(i['task']) for i in selected])
            x=torch.nn.functional.one_hot(y,6).float(); v=x[:,:4].contiguous()
            return selected,x,v,y
        with tempfile.TemporaryDirectory() as temp, patch.object(runner,'ROOT',Path(temp)), \
                patch.object(runner,'load_bound',return_value=bound), patch.object(runner,'load_features',side_effect=dataset), \
                patch.object(runner,'cpu_runtime',return_value={'fixture':True}), patch('builtins.print'):
            out=runner.ROOT/runner.OUTPUT
            runner.c.write_new(out/'receipt.json',{'fixture':True}); runner.c.write_new(out/'runtime.json',{'fixture':True})
            runner.c.write_new(runner.ROOT/runner.analysis.OUTPUT,{'per_image':[
                {'key':i['key'],'step408':{'correct':True}} for i in rows if i['split']=='validation']})
            runner.train(); runner.verify()
            result=runner.c.read(runner.ROOT/runner.REPORT)
            self.assertEqual(order,['train','validation']); self.assertFalse(result['restoration_PSNR_LPIPS_verified'])
            self.assertFalse(result['GPU_used'])
            for arm in cfg['arms']:
                self.assertEqual(len(list((out/arm/'steps').glob('step_*.json'))),401)
                self.assertEqual(result['arms'][arm]['train']['count'],640)
                self.assertEqual(result['arms'][arm]['development']['count'],40)
            runner.train(); runner.verify()  # Real complete-run resume, no extra updates or overwritten result.


if __name__ == '__main__': unittest.main()
