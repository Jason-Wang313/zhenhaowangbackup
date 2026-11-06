import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'third_party/flux2/src'))
import torch
from lowlight_restoration.flux_lq_start_v1 import initial_latent,select_suffix,restore_lq_start

class LQStartTests(unittest.TestCase):
    def test_suffix_is_original_schedule_not_resampled(self):
        from flux2.sampling import get_schedule
        schedule=get_schedule(50,1024);index,suffix=select_suffix(schedule)
        self.assertEqual(suffix,schedule[index:]);self.assertLessEqual(suffix[0],.5);self.assertGreater(schedule[index-1],.5)
        self.assertEqual(select_suffix(schedule,1),(0,schedule))
    def test_noise_endpoint_and_interpolation(self):
        a=torch.tensor([1.,2.],dtype=torch.bfloat16);n=torch.tensor([-1.,0.],dtype=torch.bfloat16)
        self.assertTrue(torch.equal(initial_latent(a,n,1),n))
        self.assertTrue(torch.equal(initial_latent(a,n,.5),torch.tensor([0.,1.],dtype=torch.bfloat16)))
        self.assertTrue(torch.equal(a,torch.tensor([1.,2.],dtype=torch.bfloat16)))
    def test_bad_inputs_rejected(self):
        for schedule in [[1,.5,.5,0],[0,.5,1],[1,float('nan'),0]]:
            with self.assertRaises(ValueError):select_suffix(schedule)
        with self.assertRaises(ValueError):initial_latent(torch.ones(2),torch.ones(3),.5)
    def test_sampler_preserves_reference_context_seed_and_no_grad(self):
        class AE:
            def encode(self,x):return x
            def decode(self,x):return x
        lq=torch.full((1,3,2,2),.2);seen=[]
        prompt={'cfg_context':torch.zeros(2,2,3),'cfg_ids':torch.zeros(2,2,4,dtype=torch.long)}
        def model(**kw):
            self.assertFalse(torch.is_grad_enabled());seen.append(kw)
            self.assertTrue(torch.all(kw['x_ids'][:,4:,0]==10));self.assertTrue(torch.all(kw['x_ids'][:,:4,0]==0))
            self.assertTrue(torch.all(kw['x'][:,4:]==lq.to(torch.bfloat16).permute(0,2,3,1).reshape(1,4,3)))
            return torch.zeros_like(kw['x'])
        raw,meta=restore_lq_start(model,AE(),lq,prompt,seed=123)
        self.assertEqual(raw.shape,lq.shape);self.assertEqual(len(seen),meta['model_calls']);self.assertFalse(meta['GT_input'])
        raw2,meta2=restore_lq_start(model,AE(),lq,prompt,seed=123)
        self.assertTrue(torch.equal(raw,raw2));self.assertEqual(meta,meta2)

if __name__=='__main__':unittest.main()
