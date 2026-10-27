import copy
import ast
import json
from types import ModuleType
import unittest
from unittest.mock import patch
from typing import Sequence

import torch

from lowlight_restoration.novel_ppt_energy import NovelPPTEnergy, objective
from lowlight_restoration.novel_ppt_energy_nonzero_init import NonzeroInitNovelPPTEnergy
from lowlight_restoration.scalar_residual import quantized_scalar_residual
from smoke import flux_energy_qualification_contract_v1 as contract
from smoke.qualify_foundir_flux_energy_v1 import deployed_proposal, restore, synchronized_patch, training_candidates


class ContractTests(unittest.TestCase):
    def row(self, index=0, split='train'):
        return dict(record_id=str(index), source_group='s'+str(index), gt_sha256='g'+str(index),
                    split=split, category='05Noise', width='600', height='700',
                    qwen_levels_json=json.dumps(dict(blur=1, noise=3, rain=1, haze=1, low_light=1)))

    def test_selection_and_leakage(self):
        rows = [self.row(i) for i in range(6)] + [self.row(99, 'validation')]
        selected, _ = contract.select_training(rows, count=3, seed=7)
        self.assertEqual(selected, contract.select_training(list(reversed(rows)), count=3, seed=7)[0])
        self.assertTrue(all(r['split'] == 'train' for r in selected))
        for field in ('source_group', 'gt_sha256'):
            bad = copy.deepcopy(rows); bad[0][field] = bad[-1][field]
            with self.assertRaises(RuntimeError):
                contract.select_training(bad, count=3, seed=7)
        with self.assertRaises(RuntimeError):
            contract.select_training(rows+rows[:1], count=3, seed=7)

    def test_cache_identity(self):
        row = self.row()
        receipt = dict(config_sha256='c', plan={'baseline_checkpoint_sha256': 'b'},
                       spec={'wrong_expert_sha256': 'w'}, prompt_receipt_sha256='p')
        meta = dict(row=row, config_sha256='c', crop_xy=contract.crop_xy(row),
                    baseline_checkpoint_sha256='b', wrong_expert_sha256='w', prompt_receipt_sha256='p')
        contract.validate_cache_metadata(meta, row, receipt)
        for field in ('config_sha256', 'baseline_checkpoint_sha256', 'wrong_expert_sha256', 'prompt_receipt_sha256', 'crop_xy'):
            with self.assertRaises(RuntimeError):
                contract.validate_cache_metadata(meta | {field: None}, row, receipt)
        with self.assertRaises(RuntimeError):
            contract.validate_cache_metadata(meta, row | {'split': 'internal_test'}, receipt)

    def test_descriptor_and_paths(self):
        self.assertEqual(contract.descriptor(self.row()), [0., .5, 0., 0., 0.])
        with self.assertRaises(ValueError):
            contract.path('/tmp/forbidden_image.png')
        with self.assertRaises(ValueError):
            contract.crop_xy(self.row() | {'width': '200'})


class EnergyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_double_quantization(self):
        torch.manual_seed(10)
        x, raw = (torch.rand(2, 3, 8, 8)*2-1 for _ in range(2))
        _, q = quantized_scalar_residual(x, raw, 1.)
        expected = quantized_scalar_residual(x, q, .3)
        actual = deployed_proposal(x, raw)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(expected, actual)))

    def test_native_reference_matches_historical_function(self):
        # Execute only the three original pure inference functions, without
        # importing its training/data/CLI dependency chain or any real model.
        source = contract.ROOT / 'smoke/evaluate_foundir_blur_only_flux2_full_development_ddp.py'
        tree = ast.parse(source.read_text())
        names = {'deterministic_noise', 'expand_cfg_condition', 'restore_batch'}
        tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
        namespace = {'torch': torch, 'Sequence': Sequence}
        exec(compile(tree, str(source), 'exec'), namespace)
        sampling = ModuleType('flux2.sampling')
        sampling.batched_prc_img = lambda x: (x, torch.zeros(x.shape[0], 1, 4, dtype=torch.int64))
        sampling.get_schedule = lambda steps, length: [steps, length]
        sampling.scatter_ids = lambda x, ids: [x.unsqueeze(2)]
        calls = []
        def denoise(model, tokens, ids, context, context_ids, **kwargs):
            calls.append({**kwargs, 'context': context.clone(), 'ids': context_ids.clone(), 'noise': tokens.clone()})
            return tokens * .5 + kwargs['img_cond_seq'] * .2
        sampling.denoise_cfg = denoise
        class AE:
            def encode(self, x): return x
            def decode(self, x): return x.float()
        x, context, ids = torch.rand(1,3,8,8), torch.rand(2,4,8), torch.zeros(2,4,4)
        with patch.dict('sys.modules', {'flux2.sampling': sampling}), patch('torch.autocast', return_value=__import__('contextlib').nullcontext()):
            actual = restore(None, AE(), x, context, ids, 27)
            expected = namespace['restore_batch'](None, AE(), x, context, ids, sampling.batched_prc_img,
                                                  seeds=[27], steps=50, reference_time=10)
        self.assertTrue(torch.equal(actual, expected))
        for key in calls[0]:
            if isinstance(calls[0][key], torch.Tensor):
                self.assertTrue(torch.equal(calls[0][key], calls[1][key]), key)
            else:
                self.assertEqual(calls[0][key], calls[1][key])

    def test_synchronized_crops(self):
        image = torch.randint(0, 256, (3, 512, 512), dtype=torch.uint8)
        a = synchronized_patch(dict(observation=image, target=image.clone(), proposal=image.clone()), seed=9)
        b = synchronized_patch(dict(observation=image), seed=9)
        self.assertEqual(a['observation'].shape, (1, 3, 256, 256))
        self.assertTrue(all(torch.equal(a['observation'], x) for x in a.values()))
        self.assertTrue(torch.equal(a['observation'], b['observation']))

    def test_initial_identity_and_rng(self):
        torch.manual_seed(5); old = NovelPPTEnergy(width=4, condition_dim=8)
        old_rng = torch.get_rng_state()
        torch.manual_seed(5); new = NonzeroInitNovelPPTEnergy(width=4, condition_dim=8)
        self.assertTrue(torch.equal(old_rng, torch.get_rng_state()))
        for key, value in old.state_dict().items():
            if key.endswith(('.beta', '.gamma')):
                self.assertTrue(torch.allclose(new.state_dict()[key], torch.full_like(value, .1)))
            else:
                self.assertTrue(torch.equal(value, new.state_dict()[key]), key)
        x, p, c = torch.rand(1, 3, 16, 16), torch.rand(1, 3, 16, 16), torch.zeros(1, 5)
        trajectory, _, _ = new.trajectory(x, p, p, c)
        self.assertTrue(all(torch.equal(t, p) for t in trajectory))
        chosen, indices, _ = new.select(x, p, p, c, trajectory)
        self.assertTrue(torch.equal(chosen, p)); self.assertEqual(indices.item(), 0)

    def test_complete_three_step_backward(self):
        torch.manual_seed(13)
        model = NonzeroInitNovelPPTEnergy(width=4, condition_dim=8)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        x = torch.rand(1, 3, 16, 16)*.5
        images = dict(observation=x, proposal=x+.02, target=x+.007, wrong_expert=x+.1)
        p = images['proposal']; descriptor = torch.tensor([[0., .5, 0., 0., 0.]])
        weights = dict(mse=1., lpips=2., gradient_fit=1., global_rank=.1, local_rank=.1, no_harm_mse=1., no_harm_lpips=1.)
        for _ in range(2):
            optimizer.zero_grad()
            output = model(x, p, p, descriptor, training_candidates(images))
            self.assertEqual(len(output['rank_candidates']), 10)  # 7 actions + 3 generated states
            loss, _, _ = objective(output, p, images['target'], lambda a,b: (a-b).square().mean((1,2,3)),
                                   residual_scale=.02, weights=weights)
            loss.backward()
            self.assertTrue(all(p.grad is not None and bool(p.grad.isfinite().all()) for p in model.parameters()))
            optimizer.step()
        self.assertGreater(model.field.head.weight.detach().abs().max().item(), 0)

    def test_open_nonlinearity_has_input_dependent_head_gradient(self):
        torch.manual_seed(21)
        model = NonzeroInitNovelPPTEnergy(width=4, condition_dim=8)
        def mixed_gradient(x):
            p = x + .01
            trajectory, gradients, _ = model.trajectory(x, p, p, torch.zeros(1, 5), create_graph=True)
            return torch.autograd.grad(gradients[0].sum(), model.field.head.weight)[0]
        a = mixed_gradient(torch.rand(1,3,16,16)*.2)
        b = mixed_gradient(torch.rand(1,3,16,16)*.7)
        self.assertFalse(torch.allclose(a, b, atol=1e-6, rtol=1e-6))


if __name__ == '__main__':
    unittest.main()
