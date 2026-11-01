import copy
import json
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn
from safetensors.torch import save_file

from test_token_residual_moe_v1 import model, token_conditioning
from smoke.token_moe_interference_v1 import (CONFIG, ARMS, capture_routes, replay_routes,
    checkpoint_deltas, four_cell_summary, diagnostic_config, summarize_terminal, load_diagnostic)
from smoke.token_residual_moe_v1 import write_once

ROOT = Path(__file__).resolve().parents[1]


class InterferenceTests(unittest.TestCase):
    def prepared(self):
        net = model().eval()
        with torch.no_grad():
            nn.init.normal_(net.block.vlm_projector[-1].weight)
        return net, torch.randn(2, 3, 6), torch.randn(2, 4)

    def test_native_self_replay_exact_and_weights_untouched(self):
        net, x, f = self.prepared()
        before = copy.deepcopy(net.state_dict())
        with torch.inference_mode():
            with token_conditioning(net, f):
                reference = net(x); gates = capture_routes(net)
            with replay_routes(net, gates):
                self.assertTrue(torch.equal(reference, net(x)))
        for name, value in net.state_dict().items():
            self.assertTrue(torch.equal(value, before[name]))
        self.assertNotIn('forward', net.block.__dict__)

    def test_fixed_gates_bypass_both_router_branches(self):
        old, x, f = self.prepared(); new = copy.deepcopy(old)
        with torch.inference_mode():
            with token_conditioning(old, f):
                expected = old(x); gates = capture_routes(old)
            for part in (new.block.router, new.block.vlm_projector):
                for p in part.parameters(): p.add_(3.)
            # Neither features nor labels are needed for replayed soft gates.
            with replay_routes(new, gates):
                self.assertTrue(torch.equal(expected, new(x)))
            new.block.experts[0].bias.add_(1.)
            with replay_routes(new, gates):
                self.assertFalse(torch.equal(expected, new(x)))

    def test_multiblock_replay_survives_changed_upstream_features(self):
        class Chain(nn.Module):
            def __init__(self):
                super().__init__(); self.a=model().block; self.b=model().block
            def forward(self,x): return self.b(self.a(x))
        old=Chain().eval()
        with torch.no_grad(): nn.init.normal_(old.b.router[-1].weight)
        new=copy.deepcopy(old)
        x,f=torch.randn(2,3,6),torch.randn(2,4)
        with torch.inference_mode():
            with token_conditioning(old,f):
                expected=old(x); gates=capture_routes(old)
            self.assertEqual(set(gates),{'a','b'})
            with replay_routes(old,gates): self.assertTrue(torch.equal(old(x),expected))
            # Nonuniform feature change: a uniform shift is removed by router LayerNorm.
            new.a.experts[0].bias[0].add_(2.)
            with token_conditioning(new,f):
                new(x); natural=capture_routes(new)
            self.assertFalse(torch.equal(gates['b'],natural['b']))
            with replay_routes(new,gates):
                new(x)
                self.assertTrue(torch.equal(new.b.last_weights,gates['b']))

    def test_preserves_actual_per_image_weights_not_dataset_mean(self):
        net, x, _ = self.prepared()
        gates = {'block': torch.tensor([[1.,0.], [0.,1.]])}
        with torch.inference_mode(), replay_routes(net, gates):
            result = net(x)
            torch.testing.assert_close(result[0],net.block.experts[0](x)[0],atol=1e-6,rtol=1e-6)
            torch.testing.assert_close(result[1],net.block.experts[1](x)[1],atol=1e-6,rtol=1e-6)

    def test_inference_only_guard_and_complete_block_coverage(self):
        net, x, _ = self.prepared(); good = {'block':torch.full((2,2),.5)}
        with self.assertRaises(RuntimeError), replay_routes(net, good): pass
        with torch.no_grad():
            net.train()
            with self.assertRaises(RuntimeError), replay_routes(net, good): pass
            net.eval()
            with self.assertRaises(RuntimeError), replay_routes(net, {}): pass
            with self.assertRaises(RuntimeError), replay_routes(net, {'other':good['block']}): pass
            with replay_routes(net, good):
                with self.assertRaises(RuntimeError): net(x[:1])

    def test_bad_or_trainable_probabilities_rejected(self):
        net, _, _ = self.prepared()
        for value in (torch.ones(2,2), torch.full((2,2),float('nan')),
                      torch.tensor([[-.2,1.2], [.5,.5]]), torch.ones(2,3)/3,
                      torch.full((2,2),.5,requires_grad=True),torch.full((2,2),.5,dtype=torch.float64)):
            with torch.no_grad(), self.assertRaises(RuntimeError), replay_routes(net, {'block':value}): pass

    def test_forward_is_restored_on_exception(self):
        net, x, f = self.prepared()
        with torch.no_grad():
            with token_conditioning(net, f):
                expected = net(x); gates = capture_routes(net)
            with self.assertRaises(ValueError), replay_routes(net,gates):
                raise ValueError('simulated failure')
            with token_conditioning(net,f): self.assertTrue(torch.equal(net(x),expected))
        self.assertNotIn('forward',net.block.__dict__)

    def test_capture_requires_native_forward(self):
        net, _, _ = self.prepared()
        with self.assertRaises(RuntimeError): capture_routes(net)

    def test_four_cell_contrasts_and_paired_identity(self):
        cells = {arm:[dict(record_id='a',source_group='s',split='rain',psnr=value,lpips=.1-value/1000)]
                 for arm,value in zip(ARMS,(30.,29.,27.,25.))}
        summary = four_cell_summary(cells)['contrasts']['rain']['psnr']
        self.assertEqual(summary,dict(gates_at_old_body=-1.,body_at_old_gates=-3.,combined_change=-5.,interaction=-1.))
        bad = copy.deepcopy(cells); bad[ARMS[-1]][0]['source_group']='different'
        with self.assertRaises(RuntimeError): four_cell_summary(bad)
        bad = copy.deepcopy(cells); bad.pop(ARMS[-1])
        with self.assertRaises(RuntimeError): four_cell_summary(bad)

    def test_checkpoint_deltas_group_exact_saved_tensors(self):
        keys = ('block.experts.0.weight','block.experts.1.weight','block.router.0.weight',
                'block.vlm_projector.1.weight','stream.lora_B.weight')
        old={k:torch.ones(2,2) for k in keys}; new={k:v+float(i) for i,(k,v) in enumerate(old.items())}
        with tempfile.TemporaryDirectory() as folder:
            a,b=str(Path(folder)/'a.safetensors'),str(Path(folder)/'b.safetensors')
            save_file(old,a); save_file(new,b)
            result=checkpoint_deltas(a,b)
            self.assertEqual(set(result),{'expert_0','expert_1','image_router','vlm_router','shared_lora'})
            self.assertEqual(result['expert_0']['changed_tensors'],0)
            self.assertEqual(result['shared_lora']['max_abs_delta'],4.)
            self.assertEqual(result['shared_lora']['delta_squared_norm'],64.)
            save_file({'wrong':torch.ones(1)},b)
            with self.assertRaises(RuntimeError): checkpoint_deltas(a,b)

    def test_actual_terminal_means_and_mse_conversion(self):
        report=json.loads((ROOT/'reports/foundir_token_moe_five_single_v1_terminal_2026-11-01.json').read_text())
        result=summarize_terminal(report)
        self.assertEqual(result['tasks']['low_light_preview']['source_groups'],21)
        self.assertGreater(result['tasks']['low_light_preview']['mean_rgb01_mse_before'],
                           result['tasks']['rain']['mean_rgb01_mse_before'])
        report['full_metrics']['2025']['rows'][0]['record_id']='wrong'
        with self.assertRaises(RuntimeError): summarize_terminal(report)

    def test_cannot_turn_diagnostic_into_training(self):
        raw=json.loads((ROOT/CONFIG).read_text())
        for change in ({'optimizer_updates':25},{'automatic_training':True},{'records_per_task':32},
                       {'self_replay_uint8_exact':False},{'arms':['best_candidate']}):
            with tempfile.TemporaryDirectory() as folder:
                path=Path(folder)/'bad.json'; write_once(path,{**raw,**change})
                with self.assertRaises(RuntimeError): diagnostic_config(ROOT,path)

    def test_full_metadata_and_bound_qualification_sources_without_pixels(self):
        result=load_diagnostic(ROOT,ROOT/CONFIG,artifacts=False)
        self.assertEqual(len(result['specs']),500)
        self.assertTrue(all(s['kind']=='validation' for s in result['specs']))
        self.assertEqual(len(result['binding']['qualification_binding']['source_sha256']),31)
        self.assertEqual(result['dc']['optimizer_updates'],0)


if __name__=='__main__': unittest.main()
