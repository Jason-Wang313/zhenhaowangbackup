"""Exact full-attention derivatives, including the backward's backward."""
import unittest
import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from lowlight_restoration.flux_native_attention import tiled_attention


class AttentionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1); torch.manual_seed(23)
        self.inputs = tuple(torch.randn(1,2,n,4,dtype=torch.float64,requires_grad=True) for n in (7,9,9))

    def test_output_and_first_second_derivatives_match(self):
        outputs=[]
        for tiled in (False,True):
            args=tuple(x.detach().clone().requires_grad_(True) for x in self.inputs)
            with sdpa_kernel(SDPBackend.MATH):
                y=tiled_attention(*args,chunk=3) if tiled else F.scaled_dot_product_attention(*args)
                first=torch.autograd.grad(y.sin().sum(),args,create_graph=True)
                second=torch.autograd.grad(sum(x.square().sum() for x in first),args)
            outputs.append((y,*first,*second))
        for a,b in zip(*outputs):torch.testing.assert_close(a,b,rtol=1e-10,atol=1e-11)

    def test_gradcheck_and_gradgradcheck(self):
        f=lambda *args:tiled_attention(*args,chunk=3)
        self.assertTrue(torch.autograd.gradcheck(f,self.inputs,fast_mode=True))
        self.assertTrue(torch.autograd.gradgradcheck(f,self.inputs,fast_mode=True))

    def test_bfloat16_values_and_gradients(self):
        outputs=[]
        for tiled in (False,True):
            args=tuple(x.detach().to(torch.bfloat16).requires_grad_(True) for x in self.inputs)
            with sdpa_kernel(SDPBackend.MATH):
                y=tiled_attention(*args,chunk=3) if tiled else F.scaled_dot_product_attention(*args)
                first=torch.autograd.grad(y.float().square().sum(),args,create_graph=True)
                second=torch.autograd.grad(sum(x.float().square().sum() for x in first),args)
            outputs.append((y,*first,*second))
        for i,(a,b) in enumerate(zip(*outputs)):
            error=float((a.float()-b.float()).norm()/a.float().norm().clamp_min(1e-12))
            print('BF16_ATTENTION_RELATIVE_L2',i,error,flush=True)
            self.assertLess(error,.01)

    def test_higher_order_graph_does_not_save_full_attention_matrix(self):
        saved=[]
        for tiled in (False,True):
            shapes=[]
            args=tuple(x.detach().clone().requires_grad_(True) for x in self.inputs)
            def pack(t):
                shapes.append(tuple(t.shape))
                return t
            with torch.autograd.graph.saved_tensors_hooks(pack,lambda t:t), sdpa_kernel(SDPBackend.MATH):
                y=tiled_attention(*args,chunk=3) if tiled else F.scaled_dot_product_attention(*args)
                gradients=torch.autograd.grad(y.square().sum(),args,create_graph=True)
            saved.append(shapes)
            sum(g.square().sum() for g in gradients).backward()
        self.assertTrue(any(s[-2:]==(7,9) for s in saved[0]))
        self.assertFalse(any(s[-2:]==(7,9) for s in saved[1]))

    def test_empty_or_invalid_rejected(self):
        with self.assertRaises(ValueError):tiled_attention(*self.inputs,chunk=0)
        with self.assertRaises(ValueError):tiled_attention(self.inputs[0][...,:0,:],*self.inputs[1:])


if __name__=='__main__':unittest.main()
