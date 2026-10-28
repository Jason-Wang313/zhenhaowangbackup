import copy
import json
from pathlib import Path
import random
import tempfile
import unittest
import torch

from smoke import flux_tail_checkpoint_v2 as saves
from smoke.qualify_foundir_flux_tail_energy_resumable_v2 import import_cache_metadata
from smoke.manage_flux_tail_energy_jobs_v2 import matches


class DurableTailTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def receipt(self):return dict(spec=dict(learning_rate=1e-4),config_sha256='science',source_sha256={'a':'bound'})

    def setup_model(self):
        torch.manual_seed(101);random.seed(91)
        model=torch.nn.Sequential(torch.nn.Linear(3,8),torch.nn.SiLU(),torch.nn.Linear(8,1))
        return model,torch.optim.AdamW(model.parameters(),lr=1e-4)

    def step(self,model,opt):
        opt.zero_grad(set_to_none=True)
        x=torch.randn(4,3)*(1+random.random())
        model(x).square().mean().backward();opt.step()

    def save(self,root,model,opt,step):
        return saves.commit_step(root,model,opt,step=step,world=1,rank_rng=[saves.rng_state('cpu')],
            receipt=self.receipt(),log=[dict(step=i+1) for i in range(step)],
            initial_hash='initial',elapsed_seconds=1.,peak_memory_gib=[0.])

    def test_resume_preserves_adam_rng_and_next_update_exactly(self):
        with tempfile.TemporaryDirectory(prefix='tail-resume-test-') as tmp:
            root=Path(tmp);model,opt=self.setup_model();self.save(root,model,opt,0)
            self.step(model,opt);self.save(root,model,opt,1)
            self.step(model,opt);expected=copy.deepcopy(model.state_dict());expected_adam=copy.deepcopy(opt.state_dict())
            state,record=saves.load_latest(root,self.receipt(),1)
            self.assertEqual(state['step'],1);self.assertEqual(state['next_global_slot'],32)
            model,opt=self.setup_model();model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer'])
            saves.restore_rng(state['rank_rng'][0],'cpu');self.step(model,opt)
            self.assertTrue(all(torch.equal(v,expected[k]) for k,v in model.state_dict().items()))
            for key,value in opt.state_dict()['state'].items():
                self.assertTrue(all(torch.equal(v,expected_adam['state'][key][name]) for name,v in value.items()))

    def test_latest_pointer_can_lag_without_losing_committed_step(self):
        with tempfile.TemporaryDirectory(prefix='tail-latest-test-') as tmp:
            root=Path(tmp);model,opt=self.setup_model();old=self.save(root,model,opt,0)
            self.step(model,opt);self.save(root,model,opt,1)
            saves.atomic_json(root/'latest.json',old,replace=True)
            state,_=saves.load_latest(root,self.receipt(),1)
            self.assertEqual(state['step'],1)
            with self.assertRaises(RuntimeError):saves.load_latest(root,self.receipt(),2)
            with self.assertRaises(RuntimeError):saves.load_latest(root,self.receipt()|{'source_sha256':{'a':'changed'}},1)

    def test_corrupt_latest_is_not_silently_rolled_back(self):
        with tempfile.TemporaryDirectory(prefix='tail-corruption-test-') as tmp:
            root=Path(tmp);model,opt=self.setup_model();self.save(root,model,opt,0)
            self.step(model,opt);record=self.save(root,model,opt,1)
            with (root/record['file']).open('ab') as h:h.write(b'corrupt')
            with self.assertRaises(RuntimeError):saves.load_latest(root,self.receipt(),1)

    def test_atomic_failure_preserves_old_file_and_no_partial_final(self):
        with tempfile.TemporaryDirectory(prefix='tail-atomic-test-') as tmp:
            root=Path(tmp);destination=root/'state.json';saves.atomic_json(destination,{'step':1})
            with self.assertRaises(FileExistsError):saves.atomic_json(destination,{'step':2})
            def broken(h):h.write(b'partial');raise RuntimeError('simulated power loss before publish')
            with self.assertRaises(RuntimeError):saves.atomic_write(root/'new.pt',broken)
            self.assertFalse((root/'new.pt').exists())
            self.assertEqual(json.loads(destination.read_text()),{'step':1})

    def test_cache_import_preserves_producer_and_does_not_mutate_original(self):
        old=dict(source_sha256={'old':'hash'},identity={'index':3},sha256='pixels')
        saved=copy.deepcopy(old);new=import_cache_metadata(old,{'new':'execution'})
        self.assertEqual(old,saved);self.assertEqual(new['sha256'],'pixels')
        self.assertEqual(new['cache_producer_source_sha256'],old['source_sha256'])
        self.assertTrue(new['cache_reused_without_regeneration'])

    def test_stop_matcher_is_specific_to_legacy_training(self):
        good=['python','-m','torch.distributed.run','smoke/qualify_foundir_flux_tail_energy_v1.py',
              'train','--run-id','qualification_4gpu_v1','--gpus','4']
        self.assertTrue(matches(good))
        for original,replacement in [('train','evaluate'),('qualification_4gpu_v1','another_job'),('4','8'),
            ('smoke/qualify_foundir_flux_tail_energy_v1.py','smoke/unrelated.py')]:
            bad=list(good);bad[bad.index(original)]=replacement;self.assertFalse(matches(bad))
        self.assertFalse(matches(['bash','-c',' '.join(good)]))


if __name__=='__main__':unittest.main()
