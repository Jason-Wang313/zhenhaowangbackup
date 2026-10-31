"""CPU tests on real FLUX block code at small dimensions; not GPU qualification."""
import copy
import json
from pathlib import Path
import random
import tempfile
import unittest

import torch
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel

from flux2.model import Flux2, Flux2Params
from lowlight_restoration.lora import inject_lora
from lowlight_restoration.flux_native_energy import (
    FluxNativeEnergy, EnergyAdapterLinear, unroll, direct_quality_loss,
    pareto_ranking_loss, select_without_target,
)


class NativeTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(17)
        flow = Flux2(Flux2Params(in_channels=8, context_in_dim=12, hidden_size=32,
            num_heads=2, depth=1, depth_single_blocks=1, axes_dim=[4,4,4,4],
            mlp_ratio=2., use_guidance_embed=False)).double()
        inject_lora(flow, ['double_blocks.*.img_attn.qkv','double_blocks.*.img_attn.proj',
                          'single_blocks.*.linear1','single_blocks.*.linear2'], 2, 2, 0.)
        self.native = copy.deepcopy(flow).eval()
        self.model = FluxNativeEnergy(flow, 2, 2, expected_adapters=4).double()
        self.state = torch.randn(1,4,8,dtype=torch.double)
        ids = torch.zeros(1,4,4,dtype=torch.double)
        ids[0,:,2] = torch.arange(4)
        self.args = dict(ids=ids, timestep=torch.tensor([.03],dtype=torch.double),
            reference=torch.randn_like(self.state), reference_ids=ids.clone(),
            proposal_reference=torch.randn_like(self.state), proposal_ids=ids.clone(),
            context=torch.randn(1,3,12,dtype=torch.double), context_ids=torch.zeros(1,3,4,dtype=torch.double),
            descriptor=torch.tensor([[0.,.25,0.,0.,0.]],dtype=torch.double))
        self.args['reference_ids'][...,0] = 10
        self.args['proposal_ids'][...,0] = 20
        self.args['context_ids'][...,3] = torch.arange(3)

    def energy(self, state):
        return self.model(state, **self.args)

    def generation_args(self, state):
        return dict(x=state, x_ids=self.args['ids'], timesteps=self.args['timestep'],
                    ctx=self.args['context'], ctx_ids=self.args['context_ids'], guidance=None)

    def test_generator_is_unchanged_even_after_energy_training(self):
        kwargs = self.generation_args(self.state)
        with sdpa_kernel(SDPBackend.MATH):
            with torch.no_grad():
                wanted = self.native(**kwargs)
                self.assertTrue(torch.equal(wanted, self.model.generation(**kwargs)))
                for module in self.model.modules():
                    if isinstance(module, EnergyAdapterLinear):
                        module.energy_b.weight.normal_(std=.2)
            # Leave an energy graph alive while doing generator forward.
            score = self.energy(self.state.requires_grad_(True))
            self.assertTrue(torch.equal(wanted, self.model.generation(**kwargs)))
            score.sum().backward()
            self.model.architecture_receipt()

    def test_three_step_second_order_reaches_flux_lora(self):
        with sdpa_kernel(SDPBackend.MATH):
            states, scores = unroll(self.energy, self.state, training=True)
            def render_tail(state):
                # Exercise the real frozen native generator inside the same
                # live graph as energy; the small reshape is NOT a real VAE.
                for _ in range(2):
                    arguments = self.generation_args(torch.cat((state, self.args['reference']),dim=1))
                    arguments['x_ids'] = torch.cat((self.args['ids'],self.args['reference_ids']),dim=1)
                    state = state-.015*self.model.generation(**arguments)[:, :state.shape[1]]
                return state[:, :3].reshape(1,3,2,4).tanh()
            images = [render_tail(s) for s in states[1:]]
            with torch.no_grad():
                proposal = render_tail(self.state)
            target = proposal*.8
            perceptual = lambda x,y: ((x-y)**2).mean().reshape(1)
            loss, metrics = direct_quality_loss(images, target, proposal, perceptual)
            loss.backward()
            self.assertEqual(len(states),4)
            self.assertEqual(len(metrics),3)
            lora = [p.grad for n,p in self.model.named_parameters() if '.energy_b.' in n]
            self.assertTrue(any(g is not None and float(g.abs().sum()) > 1e-10 for g in lora))
            for name, parameter in self.model.named_parameters():
                if parameter.requires_grad:
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(bool(torch.isfinite(parameter.grad).all()), name)
                else:
                    self.assertIsNone(parameter.grad, name)
            self.model.architecture_receipt()

    def test_input_gradient_finite_difference(self):
        z = self.state.clone().requires_grad_(True)
        with sdpa_kernel(SDPBackend.MATH):
            gradient, = torch.autograd.grad(self.energy(z).sum(), z)
            direction = gradient/gradient.norm()
            with torch.no_grad():
                numerical = (self.energy(z+1e-5*direction)-self.energy(z-1e-5*direction))/(2e-5)
            self.assertAlmostEqual(float(numerical), float((gradient*direction).sum()), places=6)

    def test_trainable_state_roundtrip_and_rejection(self):
        saved = self.model.trainable_state()
        self.assertTrue(all('energy_' in n or n.startswith(('condition.','readout.')) for n in saved))
        self.model.load_trainable_state(saved)
        with self.assertRaises(ValueError):
            self.model.load_trainable_state({'cnn.weight': torch.zeros(1)})
        self.assertFalse(any(isinstance(m,nn.Conv2d) for m in self.model.modules()))

    def test_inference_update_matches_training_without_parameter_update(self):
        with sdpa_kernel(SDPBackend.MATH):
            training, _ = unroll(self.energy,self.state,training=True)
            inference, scores = unroll(self.energy,self.state,training=False)
            for a,b in zip(training,inference):
                self.assertTrue(torch.equal(a,b))
            self.assertIn(select_without_target(scores),range(4))

    def test_ranking_does_not_force_conflicting_metrics(self):
        scores = torch.tensor([1.,-1.],requires_grad=True)
        self.assertEqual(float(pareto_ranking_loss(scores,torch.tensor([[1.,2.],[2.,1.]]))),0.)
        self.assertGreater(float(pareto_ranking_loss(scores,torch.tensor([[1.,1.],[2.,2.]]))),0.)
        self.assertEqual(select_without_target([torch.tensor(x) for x in (0.,.1,.2,.3)]),0)

    def test_real_adam_and_rng_resume(self):
        import torch.distributed as dist
        from smoke.train_foundir_flux_native_energy_v1 import save_step,load_step
        from smoke.flux_native_energy_runtime_v1 import atomic_json
        with tempfile.TemporaryDirectory(prefix='flux-native-resume-') as directory:
            root=Path(directory)
            dist.init_process_group('gloo',init_method='file://'+str(root/'gloo'),rank=0,world_size=1)
            try:
                atomic_json(root/'execution.json',{'test_only':True})
                receipt={'spec':{'global_batch':8},'protocol_sha256':'cpu_test_only'}
                params=[p for p in self.model.parameters() if p.requires_grad]
                optimizer=torch.optim.AdamW(params,lr=1e-4)
                def update():
                    optimizer.zero_grad(set_to_none=True)
                    with sdpa_kernel(SDPBackend.MATH):
                        self.energy(self.state).square().sum().backward()
                    for p in params:
                        if p.grad is None:p.grad=torch.zeros_like(p)
                    optimizer.step()
                update()
                save_step(root,self.model,optimizer,1,receipt,0,1,'cpu',1.,[{'step':1}])
                wanted_random=(random.random(),torch.rand(2))
                update();wanted=self.model.trainable_state()
                step,elapsed,log=load_step(root,self.model,optimizer,receipt,0,1,'cpu')
                self.assertEqual(step,1)
                self.assertEqual(random.random(),wanted_random[0])
                self.assertTrue(torch.equal(torch.rand(2),wanted_random[1]))
                update()
                for name,value in wanted.items():
                    self.assertTrue(torch.equal(value,self.model.trainable_state()[name]),name)
            finally:dist.destroy_process_group()

    def test_global_stream_and_full_action_coverage(self):
        from smoke.train_foundir_flux_native_energy_v1 import sample_index,action_pair
        self.assertEqual({sample_index(i,3) for i in range(24)},set(range(24)))
        for world in (4,8):
            slots=sorted(slot for rank in range(world) for slot in range(rank,8,world))
            self.assertEqual(slots,list(range(8)))
        self.assertEqual({k for i in range(13) for k in action_pair(i)},set(range(8)))

    def test_atomic_artifacts_do_not_overwrite(self):
        from smoke.flux_native_energy_runtime_v1 import atomic_json
        with tempfile.TemporaryDirectory(prefix='flux-native-atomic-') as directory:
            path=Path(directory)/'artifact.json'
            atomic_json(path,{'a':1})
            with self.assertRaises(FileExistsError):atomic_json(path,{'a':2})
            self.assertEqual(json.loads(path.read_text()),{'a':1})


if __name__ == '__main__':
    unittest.main()
