import copy
import os
import unittest
from unittest import mock
import torch
from smoke import rain_hopper_memory_v1 as h

class HopperMemoryTests(unittest.TestCase):
    def test_workspace_environment(self):
        with mock.patch.dict(os.environ,{},clear=True):
            h.configure_environment()
            self.assertEqual(os.environ['CUDNN_CONV_WSCAP_DBG'],'256')

    def test_mixed_gpu_signature_keeps_software_strict(self):
        a={'gpu_name':'NVIDIA H200','torch':'2.7','nvidia_driver':'NVRM version: NVIDIA UNIX x86_64 Kernel Module  570.124.06  Wed Feb 26\nGCC x'}
        b={**a,'gpu_name':'NVIDIA H100 80GB HBM3'}
        self.assertEqual(h.execution_signature(a),h.execution_signature(b))
        self.assertNotEqual(h.execution_signature(a),h.execution_signature({**b,'torch':'2.8'}))
        with self.assertRaises(ValueError):h.execution_signature({**a,'gpu_name':'A100'})

    def test_chunked_attention_full_context_and_gradient(self):
        from flux2.autoencoder import AttnBlock
        torch.manual_seed(7)
        original=AttnBlock(32).double();chunked=copy.deepcopy(original)
        h.configure_vae(chunked,chunk_size=7)
        x=torch.randn(1,32,5,6,dtype=torch.float64,requires_grad=True);y=x.detach().clone().requires_grad_()
        a=original(x);b=chunked(y)
        torch.testing.assert_close(a,b,rtol=1e-12,atol=1e-12)
        a.square().sum().backward();b.square().sum().backward()
        torch.testing.assert_close(x.grad,y.grad,rtol=1e-10,atol=1e-10)
        for p,q in zip(original.parameters(),chunked.parameters()):torch.testing.assert_close(p.grad,q.grad,rtol=1e-10,atol=1e-10)

    def test_offload_context_does_not_change_cpu_gradient(self):
        x=torch.randn(4,4,requires_grad=True);y=x.detach().clone().requires_grad_()
        x.sin().square().sum().backward()
        with h.offload_activations():y.sin().square().sum().backward()
        torch.testing.assert_close(x.grad,y.grad,rtol=0,atol=0)

    def test_hardware_uses_75gib_hopper_guard(self):
        from smoke import replay_foundir_existing_flux_experts_v1 as replay
        with mock.patch.object(replay,'hardware',return_value={'gpu_name':'NVIDIA H100'}) as guard:
            self.assertEqual(h.hardware(True)['gpu_name'],'NVIDIA H100')
            guard.assert_called_once_with(8,require_idle=True)
