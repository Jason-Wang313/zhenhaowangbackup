import unittest
import torch
from smoke.expanded_vlm_h100_runtime_v4 import bounded_vae, shard
from smoke import expanded_vlm_router_contract_v4 as x

class MemoryPolicyTests(unittest.TestCase):
    def test_scoped_native_conv_keeps_values_and_dtype_restores_flags(self):
        class AE(torch.nn.Module):
            def __init__(self):
                super().__init__();self.conv=torch.nn.Conv2d(3,3,3,padding=1)
            def encode(self,v):
                assert not torch.backends.cudnn.enabled
                return self.conv(v)
            decode=encode
        ae=AE().eval();v=torch.randn(1,3,16,16)
        expected=ae.conv(v).detach();before={k:t.clone() for k,t in ae.state_dict().items()}
        with torch.backends.cudnn.flags(enabled=True):
            bounded_vae(ae)
            with torch.no_grad():
                self.assertTrue(torch.equal(ae.encode(v),expected))
                self.assertTrue(torch.equal(ae.decode(v),expected))
            self.assertTrue(torch.backends.cudnn.enabled)
            self.assertIs(bounded_vae(ae),ae)
        for k,t in ae.state_dict().items():self.assertTrue(torch.equal(t,before[k]))
    def test_exception_restores_flags(self):
        class AE:
            def encode(self,v):raise ValueError('failed')
            decode=encode
        with torch.backends.cudnn.flags(enabled=True):
            with self.assertRaisesRegex(ValueError,'failed'):bounded_vae(AE()).encode(None)
            self.assertTrue(torch.backends.cudnn.enabled)
    def test_four_workers_once_and_budget_preserved(self):
        self.assertEqual((x.NODES,x.LOCAL_GPUS,x.WORLD_SIZE),(2,2,4))
        for n in (11457,100,137,400):
            a=list(range(n));groups=[shard(a,r) for r in range(4)]
            self.assertEqual(sorted(v for g in groups for v in g),a)
            self.assertLessEqual(max(map(len,groups))-min(map(len,groups)),1)
        self.assertEqual(x.RECIPE['GPU_features_cap_per_node']*4/3600,12)
        self.assertEqual(x.RECIPE['GPU_render_cap_per_node']*4/3600,6)
        with self.assertRaises(ValueError):shard([],4)
if __name__=='__main__':unittest.main()
