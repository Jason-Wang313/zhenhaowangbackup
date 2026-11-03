import copy
from pathlib import Path
import unittest
from unittest import mock

import torch
from torch import nn

from lowlight_restoration.five_expert_flux import ORDER, InferenceTrajectory, plan, weights_for_call, restore_five_expert
from lowlight_restoration.protocol_routed_flux import ExpertProtocol, ProtocolRouter, RouteBoundFlux, install_protocol_bank
from lowlight_restoration.existing_flux_experts import FLOW_CONTRACT
from lowlight_restoration.lora import LoRALinear
from smoke import qualify_foundir_five_expert_integration_v1 as q


class Tiny(nn.Module):
    def __init__(self, dtype=torch.float32):
        super().__init__()
        self.fc = nn.Linear(3, 3, bias=False, dtype=dtype)

    def forward(self, x):
        return self.fc(x)


def fixture(dtype=torch.float32, mixed=False):
    torch.manual_seed(73)
    model = Tiny(dtype)
    base = copy.deepcopy(model)
    states = []
    for name in ORDER:
        typ = torch.float32 if mixed and name == 'rain' else dtype
        states.append((name, {'fc.lora_A.weight': torch.randn(2, 3, dtype=typ) * .1,
                             'fc.lora_B.weight': torch.randn(3, 2, dtype=typ) * .1}, 2., FLOW_CONTRACT))
    gates = install_protocol_bank(model, modules=['fc'], expert_states=states)
    return base, model, gates, states


def rules():
    return ProtocolRouter([ExpertProtocol(t, 'original ' + t, a, t not in ('haze', 'rain'))
                           for t, a in zip(ORDER, (.75, .30, 1., .925, 1.))])


class FiveExpertTests(unittest.TestCase):
    def test_one_hot_matches_actual_lora_all_five_and_mixed_precision(self):
        for dtype, mixed in ((torch.float32, False), (torch.bfloat16, True)):
            base, bank, gates, states = fixture(dtype, mixed)
            x = torch.randn(2, 4, 3, dtype=dtype)
            for i, (_, state, _, _) in enumerate(states):
                original = LoRALinear(copy.deepcopy(base.fc), 2, 2.)
                original.lora_A.weight = nn.Parameter(state['fc.lora_A.weight'].clone(), requires_grad=False)
                original.lora_B.weight = nn.Parameter(state['fc.lora_B.weight'].clone(), requires_grad=False)
                with torch.no_grad(), torch.autocast('cpu', dtype=torch.bfloat16, enabled=mixed):
                    actual = RouteBoundFlux(bank, gates, torch.eye(5)[i:i+1])(x)
                    self.assertTrue(torch.equal(original(x), actual))
            self.assertTrue(all(not p.requires_grad for p in bank.parameters()))
            self.assertEqual(bank.fc.bank.a[4].dtype, torch.float32)

    def test_horizon_mask_and_renormalization(self):
        w = torch.tensor([[.6, .1, .1, .1, .1]])
        self.assertTrue(torch.equal(weights_for_call(w, 25, 'blur'), w))
        late = weights_for_call(w, 26, 'blur')
        self.assertEqual(float(late[0, 4]), 0.)
        torch.testing.assert_close(late[0, :4], w[0, :4] / .9)
        self.assertAlmostEqual(float(late.sum()), 1., delta=1e-6)
        with self.assertRaises(ValueError):
            weights_for_call(torch.eye(5)[4:5], 26, 'rain')
        with self.assertRaises(ValueError):
            weights_for_call(w, 50, 'blur')
        with self.assertRaises(ValueError):
            weights_for_call(w, 0, 'rain')

    def test_original_four_one_hot_remain_unchanged_at_every_call(self):
        for i, task in enumerate(ORDER[:4]):
            for call in range(50):
                self.assertTrue(torch.equal(weights_for_call(torch.eye(5)[i:i+1], call, task), torch.eye(5)[i:i+1]))

    def test_anchor_protocol_not_averaged_or_gt_selected(self):
        for i, task in enumerate(ORDER):
            p = plan(rules(), torch.eye(5)[i:i+1])
            self.assertEqual(p['model_calls'], 26 if task == 'rain' else 50)
            self.assertEqual(p['schedule_steps'], 50)
            self.assertEqual(p['alpha'], (.75, .30, 1., .925, 1.)[i])
        self.assertEqual(plan(rules(), torch.full((1, 5), .2))['anchor'], 'blur')

    def test_fresh_trajectory_counter_and_no_training(self):
        _, model, gates, _ = fixture()
        trajectory = InferenceTrajectory(model, gates, torch.eye(5)[4:5], 'rain')
        x = torch.ones(1, 2, 3)
        with self.assertRaises(ValueError):
            trajectory(x)
        with torch.no_grad():
            for _ in range(26):
                trajectory(x)
            with self.assertRaises(ValueError):
                trajectory(x)
        self.assertEqual(InferenceTrajectory(model, gates, torch.eye(5)[4:5], 'rain').calls, 0)
        self.assertTrue(torch.isnan(model.fc.route_weights).all())

    def test_real_prefix_capture_preserves_50_schedule_but_calls_26(self):
        _, model, gates, _ = fixture()
        lq = torch.zeros(1, 3, 2, 3)
        prompts = {t: {'cfg_context': t, 'cfg_ids': t} for t in ORDER}
        sampler_calls = []
        def sampler(m, ae, image, ctx, ids, pack, *, seeds, steps, reference_time):
            sampler_calls.append((ctx, steps, reference_time, seeds))
            with torch.no_grad():
                for _ in range(steps):
                    raw = m(x=image)
            return raw
        with mock.patch('smoke.evaluate_foundir_blur_only_flux2_full_development_ddp.restore_batch', sampler), \
                mock.patch('lowlight_restoration.rain_onpolicy_midpoint.predict_endpoint', lambda m, inputs: m(**inputs)), \
                mock.patch('lowlight_restoration.rain_onpolicy_midpoint.decode_endpoint', lambda ae, z, checkpointed: z):
            for i, task in enumerate(ORDER):
                raw, u8, p = restore_five_expert(model, gates, rules(), torch.eye(5)[i:i+1], None, lq, prompts, seed=99)
                self.assertEqual(p['model_calls'], 26 if task == 'rain' else 50)
                self.assertEqual(u8.dtype, torch.uint8)
        self.assertEqual(sampler_calls, [(t, 50, 10, [99]) for t in ORDER])

    def test_rain_quantization_does_not_take_lq_subtraction_roundtrip(self):
        # Exercise the adapter with nontrivial LQ, including quantization boundaries.
        _, model, gates, _ = fixture()
        lq = torch.full((1, 3, 2, 3), -.9876543)
        raw = torch.full_like(lq, .1234567)
        prompts = {'rain': {}}
        def inputs(m, ae, image, prompt, seed):
            with torch.no_grad():
                for _ in range(25):
                    m(x=image)
            return {'x': image}
        def predict(m, inputs):
            m(**inputs)
            return raw
        with mock.patch('lowlight_restoration.rain_onpolicy_midpoint.generated_inputs', inputs), \
                mock.patch('lowlight_restoration.rain_onpolicy_midpoint.predict_endpoint', predict), \
                mock.patch('lowlight_restoration.rain_onpolicy_midpoint.decode_endpoint', lambda ae, z, checkpointed: z), \
                mock.patch.object(ProtocolRouter, 'output', side_effect=AssertionError('Rain must not use blend arithmetic')):
            _, image, _ = restore_five_expert(model, gates, rules(), torch.eye(5)[4:5], None, lq, prompts, seed=1)
        self.assertTrue(torch.equal(image, q.parent.quantize(raw)))

    def test_shards_complete_and_soft_never_on_development(self):
        cfg = q.config()
        self.assertEqual(cfg['world_size'], 8)
        bound = {'items': [{'key': str(i), 'training_fit_only': i < 16} for i in range(116)]}
        union = [i['key'] for rank in range(8) for i in q.items_for_rank(bound, rank)]
        self.assertEqual(len(set(union)), 116)
        self.assertEqual(sum(len(q.arms(i)) for i in bound['items']), 248)
        for item in bound['items'][16:]:
            self.assertNotIn('near_one_hot', q.arms(item))
            with self.assertRaises(ValueError):
                q.fixture(item, 'near_one_hot', 'cpu')

    def test_rain_prompt_text_matches_existing_native_prompt_not_missing_receipt_key(self):
        self.assertEqual(q.rain_prompt_text(),
            'Remove rain streaks from the reference image and restore a clean image. '
            'Preserve the exact content, geometry, identity, color, and texture.')

    def test_preflight_receipt_schema_without_destination_pixels_or_cuda(self):
        cfg = q.config()
        model = {'flow_weights': 'models/base.safetensors', 'flow_weights_sha256': 'basehash',
                 'autoencoder_weights': 'models/ae.safetensors', 'autoencoder_weights_sha256': 'aehash'}
        # Deliberately no model['prompt']; the real inherited receipt has none.
        four = {'sources': {}, 'assets': {'fixture/manifests/foundir_five_type_pairs_v1.csv': 'manifest',
                model['flow_weights']: 'basehash', model['autoencoder_weights']: 'aehash'},
                'experts': {t: {'checkpoint': {'path': f'weights/{t}', 'sha256': 's'},
                               'prompt': t, 'inference': {'frozen_alpha': a}}
                            for t, a in zip(ORDER[:4], (.75, .30, 1., .925))},
                'rows': {t: [{'record_id': f'{t}{i}', 'inference_seed': i} for i in range(2)] for t in ORDER[:4]}}
        rr = {'model': model, 'rain': {'prompt_context_cache': {'path': 'prompt', 'sha256': 's'}},
              'metric': {}, 'config': {'seed': 10, 'probe_seed': 20, 'memory_policy': {}},
              'rows': [{'category': '10Rain', 'usage': 'train_update', 'split': 'train',
                        'record_id': 'train', 'source_group': 'train', 'gt_sha256': 'train'}] * 64032
                      + [{'category': '10Rain', 'usage': 'development', 'split': 'validation',
                          'record_id': str(i), 'source_group': str(i), 'gt_sha256': str(i)} for i in range(100)]}
        rain_root = q.c.path(q.ROOT, cfg['rain_parent_run'])
        ck = rain_root / 'tensors/rain.safetensors'
        report = {'last_global_step': 800, 'development_gate_pass': True, 'nodes': [{'status': 0}] * 4,
                  'result_sha256': cfg['rain_result_sha256'], 'receipt_sha256': 's',
                  'final_lora': {'step': 800, 'tensor': 'tensors/rain.safetensors',
                                 'files': {'tensors/rain.safetensors': cfg['rain_lora_sha256']}}}
        def read(path):
            path = Path(path)
            if path == q.ROOT / cfg['rain_report']:
                return report
            if path.name.startswith('step_000800_image_'):
                return {'identity': {'index': int(path.stem.rsplit('_', 1)[1])}}
            if path.name == 'foundir_rain_flux2_source_diverse_pilot_v1.json':
                return {'model': {'prompt': 'bound rain prompt'}}
            raise AssertionError('unexpected read ' + str(path))
        def digest(path):
            if Path(path) == ck:
                return cfg['rain_lora_sha256']
            if Path(path) == rain_root / 'result.json':
                return cfg['rain_result_sha256']
            return 's'
        with mock.patch.object(q, 'config', return_value=cfg), \
                mock.patch.object(q.c, 'load_receipt', return_value=four), \
                mock.patch.object(q.parent.replay, 'prompt_info', return_value={}), \
                mock.patch.object(q.rain, 'load_receipt', return_value=({'sources': {}}, rr, rain_root)), \
                mock.patch.object(q.rain, 'image_identity', side_effect=lambda output, b, r, i, step: {'index': i}), \
                mock.patch.object(q.four_gt, 'select_gt', return_value={}), \
                mock.patch.object(q.c, 'sha', side_effect=digest), \
                mock.patch.object(q.c, 'read', side_effect=read):
            receipt = q.make_receipt()
        self.assertEqual(len(receipt['items']), 116)
        self.assertEqual(sum(i['training_fit_only'] for i in receipt['items']), 16)
        self.assertEqual(receipt['experts']['rain']['prompt'], 'bound rain prompt')
        self.assertFalse(receipt['official_pixels_read'])

    def test_no_original_source_edits_and_launcher_contract(self):
        root = Path(__file__).resolve().parents[1]
        script = (root / 'smoke/run_foundir_five_expert_integration_v1_gpu.sh').read_text()
        self.assertIn('--nnodes=4 --nproc_per_node=2', script)
        self.assertIn('CUDA_VISIBLE_DEVICES=0,1', script)
        self.assertIn('WALL_CAP=3600', script)
        self.assertIn('qualify_foundir_five_expert_integration_v1.py', script)
        self.assertNotIn('train_foundir_five', script)
        self.assertFalse(q.config()['automatic_training'])


if __name__ == '__main__':
    unittest.main()
