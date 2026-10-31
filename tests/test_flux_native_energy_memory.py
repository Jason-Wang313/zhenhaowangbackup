"""Numerical parity and retained-tensor checks; no CUDA capacity claim."""
import unittest
from unittest.mock import patch
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from tests import test_flux_native_energy as native_tests
from lowlight_restoration.flux_native_energy import (
    EnergyAdapterLinear, energy_value_and_gradient, unroll, direct_quality_loss,
    pareto_ranking_loss, energy_attention_mode,
)


class MemoryTests(unittest.TestCase):
    setUp = native_tests.NativeTests.setUp
    energy = native_tests.NativeTests.energy
    generation_args = native_tests.NativeTests.generation_args

    def rendered_loss(self, recompute):
        states, scores = unroll(self.energy, self.state, training=True,
                                recompute_derivative=recompute)
        def render(z):
            for _ in range(2):
                z = z-.015*self.model.generation(**self.generation_args(z))
            return z[:, :3].reshape(1, 3, 2, 4).tanh()
        images = [render(z) for z in states[1:]]
        with torch.no_grad():
            proposal = render(self.state)
        loss, _ = direct_quality_loss(images, proposal*.8, proposal,
                                      lambda x,y: (x-y).square().mean().reshape(1))
        costs = torch.tensor([[1.,1.],[.8,.8],[.7,.7],[.6,.6]], dtype=torch.double)
        loss = loss + .1*pareto_ranking_loss(torch.cat(scores), costs)
        # Endpoint t=0 ranking stays in the same alive graph as full unroll.
        args = dict(self.args, timestep=torch.zeros_like(self.args['timestep']))
        endpoint = torch.cat([self.model(self.state*k, **args) for k in (.8, 1.)])
        loss = loss + .1*pareto_ranking_loss(endpoint, costs[:2])
        return loss, states, scores

    def test_three_step_loss_all_parameter_gradients_and_adam_match(self):
        outputs = []
        for recompute in (False, True):
            self.setUp()
            # Test beyond zero-initialized adapters as well.
            with torch.no_grad():
                for m in self.model.modules():
                    if isinstance(m, EnergyAdapterLinear): m.energy_b.weight.normal_(std=.01)
            parameters = [p for p in self.model.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(parameters, lr=1e-4)
            from lowlight_restoration.flux_native_attention import tiled_attention
            tiled_small = lambda q,k,v: tiled_attention(q,k,v,chunk=3)
            with patch('lowlight_restoration.flux_native_attention.tiled_attention', new=tiled_small), energy_attention_mode(recompute), sdpa_kernel(SDPBackend.MATH):
                loss, states, scores = self.rendered_loss(recompute)
                loss.backward()
            gradients = {n:p.grad.clone() for n,p in self.model.named_parameters() if p.requires_grad}
            optimizer.step()
            outputs.append((loss.detach(), [x.detach() for x in states],
                            [x.detach() for x in scores], gradients, self.model.trainable_state()))
        for left, right in zip(*outputs):
            if isinstance(left, dict):
                for k in left: torch.testing.assert_close(left[k], right[k], rtol=1e-9, atol=1e-11)
            elif isinstance(left, list):
                for a,b in zip(left,right): torch.testing.assert_close(a,b,rtol=1e-10,atol=1e-11)
            else: torch.testing.assert_close(left,right,rtol=1e-10,atol=1e-11)

    def test_gradient_and_double_backward_match(self):
        results=[]
        for recompute in (False,True):
            self.setUp()
            z=self.state.clone().requires_grad_(True)
            with energy_attention_mode(recompute), sdpa_kernel(SDPBackend.MATH):
                score,g=energy_value_and_gradient(self.energy,z,training=True,recompute_derivative=recompute)
                first=torch.autograd.grad(g.square().sum()+score.sum(),z,create_graph=True)[0]
                second=torch.autograd.grad(first.sum(),z)[0]
            results.append((score,g,first,second))
        for a,b in zip(*results):torch.testing.assert_close(a,b,rtol=1e-9,atol=1e-11)

    def test_derivative_checkpoint_retains_less_storage(self):
        retained=[]
        for recompute in (False,True):
            self.setUp();storages={}
            def pack(t):
                storage=t.untyped_storage()
                storages[storage.data_ptr()]=storage.nbytes()
                return t
            z=self.state.clone().requires_grad_(True)
            with torch.autograd.graph.saved_tensors_hooks(pack,lambda t:t), sdpa_kernel(SDPBackend.MATH):
                score,g=energy_value_and_gradient(self.energy,z,training=True,recompute_derivative=recompute)
            retained.append(sum(storages.values()))
            (g.square().sum()+score.sum()).backward()
        self.assertLess(retained[1],retained[0]/4,retained)
        print('CPU_RETAINED_STORAGE_BYTES block_only=%s derivative_checkpoint=%s' % tuple(retained),flush=True)


if __name__=='__main__':unittest.main()
