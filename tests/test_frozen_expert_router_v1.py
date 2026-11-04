import copy
from pathlib import Path
import random
import tempfile
import unittest
from unittest import mock

import torch
from torch import nn
from lowlight_restoration.frozen_expert_router import (
    FrozenExpertRouter, RoundUint8STE, quantized_output, run_steps, enable_routed_checkpointing, objective)
from lowlight_restoration.protocol_routed_flux import RouteBoundFlux, install_protocol_bank, ProtocolRouter, ExpertProtocol
from lowlight_restoration.existing_flux_experts import FLOW_CONTRACT
from lowlight_restoration.five_expert_flux import ORDER, plan
from smoke import frozen_expert_router_contract_v1 as contract
from smoke import frozen_expert_router_state_v1 as state


def rules():
    return ProtocolRouter([ExpertProtocol(t, 'original '+t, a, t not in ('haze', 'rain'))
        for t, a in zip(ORDER, (.75, .30, 1., .925, 1.))])


class Block(nn.Module):
    def __init__(self):
        super().__init__(); self.fc = nn.Linear(3, 3, bias=False)

    def forward_kv_extract(self, x, *, scale):
        return torch.sin(self.fc(x)*scale)


class Blocks(nn.Module):
    def __init__(self):
        super().__init__()
        self.double_blocks = nn.ModuleList([Block()])
        self.single_blocks = nn.ModuleList([Block()])

    def forward(self, x):
        for block in list(self.double_blocks)+list(self.single_blocks):
            x = block.forward_kv_extract(x, scale=.25)
        return x


def bank_fixture():
    torch.manual_seed(113)
    model = Blocks(); sites = ['double_blocks.0.fc', 'single_blocks.0.fc']
    experts = []
    for name in ORDER:
        tensors = {site+'.lora_'+ab+'.weight': torch.randn(2, 3) if ab == 'A' else torch.randn(3, 2)
                   for site in sites for ab in ('A', 'B')}
        experts.append((name, tensors, 2., FLOW_CONTRACT))
    gates = install_protocol_bank(model, modules=sites, expert_states=experts)
    return model, gates


def synthetic_rows():
    rows = []
    for task, category in contract.CATEGORIES.items():
        for split, count in [('train', 130), ('validation', 10), ('internal_test', 2)]:
            for i in range(count):
                key = f'{task}_{split}_{i}'
                rows.append({'record_id': key, 'source_group': contract.c.stable(key), 'gt_sha256': contract.c.stable('gt', key),
                    'lq_sha256': contract.c.stable('lq', key), 'width': '520', 'height': '530', 'composition_size': '1',
                    'split': split, 'category': category,
                    'lq_path': f'data/foundir/MillionIRData/Train/{category}/LQ/{key}.png',
                    'gt_path': f'data/foundir/MillionIRData/Train/{category}/GT/{key}.png'})
    return rows


class RouterTests(unittest.TestCase):
    def test_vlm_zero_initialization_and_both_heads_gradient(self):
        model = FrozenExpertRouter(6, 4, 8)
        x, v = torch.randn(3, 6), torch.randn(3, 4)
        logits, w, image, delta = model(x, v)
        self.assertTrue(torch.equal(logits, image)); self.assertEqual(int(torch.count_nonzero(delta)), 0)
        torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1, 4])).backward()
        self.assertTrue(all(p.grad is not None for p in model.parameters()))
        self.assertGreater(float(model.vlm[-1].weight.grad.norm()), 0)
        self.assertEqual(float(model.vlm[1].weight.grad.norm()), 0)

    def test_disabled_vlm_stays_zero_with_adam_state(self):
        model = FrozenExpertRouter(6, 4, 8); opt = torch.optim.AdamW(model.parameters(), weight_decay=0)
        before = copy.deepcopy(model.vlm.state_dict())
        model(torch.randn(1, 6), torch.randn(1, 4), vlm_enabled=False)[0].square().sum().backward(); opt.step()
        self.assertEqual(len(opt.state), len(list(model.parameters())))
        self.assertTrue(all(torch.equal(v, model.vlm.state_dict()[k]) for k, v in before.items()))

    def test_no_trainable_or_nonfinite_feature_inputs(self):
        model = FrozenExpertRouter(6, 4, 8)
        for x in (torch.randn(1, 6, requires_grad=True), torch.full((1, 6), float('nan'))):
            with self.assertRaises(ValueError): model(x, torch.randn(1, 4))

    def test_no_minimum_alternative_ratio(self):
        model = FrozenExpertRouter(6, 4, 8)
        with torch.no_grad():
            model.image[-1].weight.zero_(); model.image[-1].bias.copy_(torch.tensor([30., -30, -30, -30, -30]))
        self.assertGreater(float(model(torch.randn(1, 6), torch.randn(1, 4))[1][0, 0]), .99999)

    def test_quantization_matches_all_inherited_outputs(self):
        from smoke.replay_foundir_existing_flux_experts_v1 import quantize
        torch.manual_seed(5)
        lq, raw = [torch.rand(1, 3, 64, 64)*2-1 for _ in range(2)]
        for i, task in enumerate(ORDER):
            route = plan(rules(), torch.eye(5)[i:i+1])
            result = ((quantized_output(lq, raw, route)+1)*127.5).round().byte()
            expected = ((raw+1)*127.5).round().byte() if task == 'rain' else quantize(task, lq, raw, route['alpha'])
            self.assertTrue(torch.equal(result, expected), task)

    def test_ste_is_documented_surrogate_not_true_round_derivative(self):
        x = torch.tensor([-.333, .111, .999], requires_grad=True)
        y = RoundUint8STE.apply(x); y.sum().backward()
        self.assertTrue(torch.equal(x.grad, torch.ones_like(x)))
        self.assertFalse(torch.equal(x.detach(), y.detach()))

    def test_full_trajectory_checkpoint_exact_forward_gradient(self):
        schedule = torch.linspace(1, 0, 51).tolist()
        for anchor in ('blur', 'rain'):
            x = torch.randn(2, 2, 3)
            scores = torch.tensor([[3., 1, 0, -1, 4 if anchor == 'rain' else -2]], requires_grad=True)
            indices = []
            def predict(value, gates, index, t):
                indices.append(index)
                return torch.sin(value[:1]*.1 + (gates*torch.arange(5)).sum()*.1 + t*.02+index*.001)
            ordinary = run_steps(x, scores.softmax(-1), schedule, anchor, predict, checkpointed=False)
            reference = torch.autograd.grad(ordinary.square().sum(), scores)[0]
            indices.clear()
            actual = run_steps(x, scores.softmax(-1), schedule, anchor, predict, checkpointed=True)
            gradient = torch.autograd.grad(actual.square().sum(), scores)[0]
            self.assertTrue(torch.equal(ordinary, actual)); self.assertTrue(torch.equal(reference, gradient))
            self.assertGreater(float(gradient.abs().sum()), 0.)
            self.assertEqual(set(indices), set(range(26 if anchor == 'rain' else 50)))

    def test_early_step_gradient_not_detached(self):
        scores = torch.tensor([[3., 1., 0., -1., -2.]], dtype=torch.float64, requires_grad=True)
        x = torch.zeros(2, 1, 1, dtype=torch.float64); schedule = torch.linspace(1, 0, 51).tolist()
        def output(s):
            return run_steps(x, s.softmax(-1), schedule, 'blur',
                lambda x, g, i, t: g[:, 0:1, None] if i == 0 else x[:1]*0, checkpointed=True).sum()
        grad = torch.autograd.grad(output(scores), scores)[0]
        eps = 1e-5; plus = scores.detach().clone(); minus = plus.clone(); plus[0, 0] += eps; minus[0, 0] -= eps
        self.assertAlmostEqual(float(grad[0, 0]), float((output(plus)-output(minus))/(2*eps)), places=8)
        self.assertGreater(float(grad.abs().sum()), 0.)

    def test_rain_uses_bfloat16_rounded_time(self):
        x = torch.ones(2, 1, 1, dtype=torch.bfloat16); w = torch.eye(5)[4:5]
        schedule = [1-i*.01983 for i in range(51)]
        def predict(x, gate, i, t): return torch.ones_like(x[:1]) if i == 25 else torch.zeros_like(x[:1])
        value = run_steps(x, w, schedule, 'rain', predict, checkpointed=False)
        self.assertEqual(float(value), 1-float(torch.tensor(schedule[25], dtype=torch.bfloat16)))

    def test_nested_checkpoint_rebinds_after_sentinels_restored(self):
        model, gates = bank_fixture(); x = torch.randn(2, 2, 3)
        logits = torch.randn(1, 5, requires_grad=True)
        def full(check):
            return run_steps(x, logits.softmax(-1), torch.linspace(1, 0, 51).tolist(),
                ORDER[int(logits.argmax())], lambda z, w, i, t: RouteBoundFlux(model, gates, w)(z)[:1], checkpointed=check)
        plain = full(False); ref = torch.autograd.grad(plain.square().sum(), logits)[0]
        enable_routed_checkpointing(model)
        actual = full(True); grad = torch.autograd.grad(actual.square().sum(), logits)[0]
        self.assertTrue(torch.equal(plain, actual)); torch.testing.assert_close(grad, ref, rtol=0, atol=0)
        self.assertTrue(all(torch.isnan(model.get_buffer(n)).all() for n in gates))
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in model.parameters()))
        with self.assertRaises(ValueError): enable_routed_checkpointing(model)

    def test_real_flux_blocks_tiny_native_sampler_parity_and_backward(self):
        from flux2.model import Flux2, Flux2Params
        from lowlight_restoration.frozen_expert_router import differentiable_restore
        from lowlight_restoration.five_expert_flux import restore_five_expert
        class AE(nn.Module):
            def encode(self, x): return x.to(torch.bfloat16)
            def decode(self, z): return z.float().tanh()
        torch.manual_seed(17)
        model = Flux2(Flux2Params(in_channels=3, context_in_dim=4, hidden_size=16, num_heads=2,
            depth=1, depth_single_blocks=1, axes_dim=[2, 2, 2, 2], mlp_ratio=2., use_guidance_embed=False)).to(torch.bfloat16)
        sites = ['double_blocks.0.img_attn.qkv', 'double_blocks.0.img_attn.proj',
                 'single_blocks.0.linear1', 'single_blocks.0.linear2']
        states = []
        for task in ORDER:
            values = {}
            for site in sites:
                layer = model.get_submodule(site)
                # CPU sampler has no CUDA autocast. Mixed Rain FP32 is covered
                # by bank unit tests and the mandatory real GPU parity check.
                dtype = torch.bfloat16
                values[site+'.lora_A.weight'] = torch.randn(2, layer.in_features, dtype=dtype)*.1
                values[site+'.lora_B.weight'] = torch.randn(layer.out_features, 2, dtype=dtype)*.1
            states.append((task, values, 2., FLOW_CONTRACT))
        gates = install_protocol_bank(model, modules=sites, expert_states=states)
        model.eval().requires_grad_(False); ae = AE()
        prompts = {t: {'cfg_context': torch.randn(2, 1, 4, dtype=torch.bfloat16),
                       'cfg_ids': torch.zeros(2, 1, 4, dtype=torch.long)} for t in ORDER}
        lq = torch.rand(1, 3, 2, 2)*2-1
        original = {i: restore_five_expert(model, gates, rules(), torch.eye(5)[i:i+1], ae, lq, prompts, seed=22)
                    for i in (0, 4)}
        enable_routed_checkpointing(model)
        for i in (0, 4):
            w = torch.eye(5)[i:i+1].requires_grad_()
            raw, image, _ = differentiable_restore(model, gates, rules(), w, ae, lq, ae.encode(lq), prompts, seed=22)
            self.assertTrue(torch.equal(raw, original[i][0]))
            self.assertTrue(torch.equal(((image+1)*127.5).round().byte(), original[i][1]))
            image.square().mean().backward()
            self.assertTrue(torch.isfinite(w.grad).all()); self.assertGreater(float(w.grad.norm()), 0.)
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_source_selection_budget_and_no_protected_paths(self):
        rows = synthetic_rows(); cfg = contract.config(); items = contract.select(rows, cfg)
        self.assertEqual(len(items), 680); self.assertEqual(items, contract.select(reversed(rows), cfg))
        train = [i for i in items if i['split'] == 'train']; val = [i for i in items if i['split'] == 'validation']
        for key in ('source_group', 'gt_sha256'):
            self.assertFalse({i['row'][key] for i in train} & {i['row'][key] for i in val})
        self.assertEqual(len({contract.training_item(items, s)['key'] for s in range(640)}), 640)
        for step in (0, 399, 400, 407):
            slots = [step*32+micro*8+r for r in range(8) for micro in range(4)]
            self.assertEqual(sorted(slots), list(range(step*32, (step+1)*32)))
            self.assertTrue(all(contract.training_item(items, s)['split'] == 'train' for s in slots))
        with self.assertRaises(ValueError): contract.training_item(items, 408*32)
        rows[0]['lq_path'] = 'official/whatever.png'
        with self.assertRaises(ValueError): contract.select(rows, cfg)

    def test_nontrain_same_source_and_gt_block_training(self):
        rows = synthetic_rows(); row = dict(rows[0]); row['record_id'] = 'blocked'; row['split'] = 'internal_test'; rows.append(row)
        items = contract.select(rows, contract.config())
        self.assertNotIn(rows[0]['record_id'], {i['row']['record_id'] for i in items})

    def test_resume_exact_model_adam_and_rng_and_corruption(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); torch.manual_seed(4); random.seed(4)
            model = FrozenExpertRouter(6, 4, 8); opt = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=0)
            rng = lambda: [state.io.rng_state('cpu') for _ in range(8)]
            state.save(root, model, opt, step=0, history=[], rank_rng=rng(), receipt_hash='binding')
            def update(m, o):
                o.zero_grad(set_to_none=True)
                logits = m(torch.randn(1, 6), torch.randn(1, 4), vlm_enabled=False)[0]
                torch.nn.functional.cross_entropy(logits, torch.tensor([random.randrange(5)])).backward(); o.step()
            update(model, opt)
            state.save(root, model, opt, step=1, history=[{'step': 1}], rank_rng=rng(), receipt_hash='binding')
            saved = state.load(root, 'binding'); update(model, opt)
            fresh = FrozenExpertRouter(6, 4, 8); fresh_opt = torch.optim.AdamW(fresh.parameters())
            state.restore(fresh, fresh_opt, saved, 0, 'cpu'); update(fresh, fresh_opt)
            self.assertTrue(all(torch.equal(v, fresh.state_dict()[k]) for k, v in model.state_dict().items()))
            for key, entries in opt.state_dict()['state'].items():
                for name, value in entries.items():
                    self.assertTrue(torch.equal(value, fresh_opt.state_dict()['state'][key][name]))
            marker = state.c.read(root/'steps/step_000001.json')
            with (root/marker['file']).open('ab') as f: f.write(b'corrupt')
            with self.assertRaises(ValueError): state.load(root, 'binding')

    def test_gpu_launcher_has_bounded_training_no_auto_formal(self):
        root = Path(__file__).resolve().parents[1]
        text = (root/'smoke/run_foundir_frozen_expert_router_v1_gpu.sh').read_text()
        for part in ('WALL_CAP=10800', '--nnodes=4', '--nproc_per_node=2', 'CUDA_VISIBLE_DEVICES=0,1', '"$SCRIPT" train'):
            self.assertIn(part, text)
        for part in ('--max_restarts=1', 'agent-driven-world-model', 'pkill', 'restormer'):
            self.assertNotIn(part, text)
        self.assertFalse(contract.config()['automatic_formal'])


if __name__ == '__main__':
    unittest.main()
