from __future__ import annotations

import copy
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from lowlight_restoration.lora import inject_lora, save_lora_checkpoint, load_lora_checkpoint
from lowlight_restoration.flux2_endpoint_precision import promote_lora_to_fp32
from lowlight_restoration.rain_full_data import RainFullDataset
from lowlight_restoration.rain_precision_resume import RainResumeDataset, restore_adam, validate_adam_payload
from smoke import train_foundir_rain_flux2_precision_resume_v1_ddp as resume
from tests import test_foundir_rain_precision_full as full_tests


def toy_model():
    torch.manual_seed(17)
    model = torch.nn.Sequential(torch.nn.Linear(5, 4), torch.nn.Linear(4, 3))
    inject_lora(model, ['0', '1'], rank=2, alpha=2, dropout=0)
    params = promote_lora_to_fp32(model)
    return model, torch.optim.AdamW(params, lr=1e-5, weight_decay=0, fused=False)


def update(model, optimizer, step):
    optimizer.zero_grad(set_to_none=True)
    values = torch.arange(10, dtype=torch.float32).reshape(2, 5) / (step + 1)
    model(values).square().mean().backward()
    optimizer.step()


class ResumeTests(unittest.TestCase):
    def test_interrupted_lora_adam_matches_uninterrupted_next_updates(self):
        model, optimizer = toy_model()
        for step in range(3):
            update(model, optimizer, step)
        payload = {'step': 3, 'config_sha256': 'toy', 'optimizer': optimizer.state_dict()}
        buffer = io.BytesIO()
        torch.save(payload, buffer)
        buffer.seek(0)
        saved = torch.load(buffer, weights_only=True)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'lora.safetensors'
            save_lora_checkpoint(model, path, {'step': 3})
            other, other_optimizer = toy_model()
            load_lora_checkpoint(other, path)
            restore_adam(other_optimizer, saved, config_sha256='toy', step=3, fused=False)
            for step in range(3, 6):
                update(model, optimizer, step)
                update(other, other_optimizer, step)
                for a, b in zip(model.parameters(), other.parameters()):
                    self.assertTrue(torch.equal(a, b))
                for a, b in zip(optimizer.state.values(), other_optimizer.state.values()):
                    for key in a:
                        self.assertTrue(torch.equal(a[key], b[key]))

    def test_corrupt_optimizer_rejected(self):
        model, optimizer = toy_model()
        update(model, optimizer, 0)
        payload = {'step': 1, 'config_sha256': 'toy', 'optimizer': optimizer.state_dict()}
        shapes = [p.shape for p in optimizer.param_groups[0]['params']]
        mutations = [
            lambda p: p.update(step=0),
            lambda p: p.update(config_sha256='different'),
            lambda p: p['optimizer']['param_groups'][0].update(lr=2e-5),
            lambda p: p['optimizer']['state'][0].pop('exp_avg'),
            lambda p: p['optimizer']['state'][0]['step'].zero_(),
            lambda p: p['optimizer']['state'][0]['exp_avg'].fill_(float('nan')),
            lambda p: p['optimizer']['state'][0]['exp_avg_sq'].fill_(-1),
            lambda p: p['optimizer']['state'][0].update(exp_avg=torch.zeros(1)),
            lambda p: p['optimizer']['param_groups'][0]['params'].reverse(),
        ]
        for mutation in mutations:
            changed = copy.deepcopy(payload)
            mutation(changed)
            with self.subTest(mutation=mutation), self.assertRaises(RuntimeError):
                validate_adam_payload(changed, config_sha256='toy', step=1, shapes=shapes, fused=False)

    def test_absolute_schedule_all_32000_presentations(self):
        rows = full_tests.rows_for_schedule()
        with patch('lowlight_restoration.rain_full_data.crop_pair', return_value=(None, None)):
            positions = set()
            for rank in range(8):
                original = RainFullDataset('.', rows, steps=4000, rank=rank, world=8, seed=19)
                continued = RainResumeDataset('.', rows, steps=1000, start_step=3000,
                                             rank=rank, world=8, seed=19)
                self.assertEqual(len(continued), 4000)
                for index in range(len(continued)):
                    a, b = original[12000 + index], continued[index]
                    self.assertEqual(a, b)
                    positions.add(b['position'])
                with self.assertRaises(IndexError):
                    continued[4000]
            self.assertEqual(positions, set(range(96000, 128000)))

    def test_crop_and_augmentation_exact_and_spawn_safe(self):
        with tempfile.TemporaryDirectory() as directory:
            row = full_tests.FullRainTests().make_row(directory)
            original = RainFullDataset(directory, [row], steps=4000, rank=7, world=8, seed=19)
            continued = RainResumeDataset(directory, [row], steps=1, start_step=3000,
                                         rank=7, world=8, seed=19)
            loader = torch.utils.data.DataLoader(continued, batch_size=1, num_workers=2,
                prefetch_factor=2, multiprocessing_context='spawn')
            for index, batch in enumerate(loader):
                expected = original[12000 + index]
                self.assertEqual(int(batch['position'][0]), expected['position'])
                self.assertTrue(torch.equal(batch['lq'][0], expected['lq']))
                self.assertTrue(torch.equal(batch['gt'][0], expected['gt']))

    def test_frozen_contract_and_real_checkpoint_audit(self):
        config, parent = resume.read_contract(resume.ROOT / 'configs/foundir_rain_flux2_precision_resume_v1.json')
        report = resume.audit_resume(config, parent)
        self.assertEqual(report['adam_step'], 3000)
        self.assertEqual(report['adam_parameters'], 100)
        self.assertEqual(config['modes']['continuation'], {'steps': 4000, 'milestones': [3500, 4000]})
        parent_config, _ = resume.full_impl.read_contract(resume.ROOT / config['resume']['parent_config'])
        for key in ('training', 'data', 'evaluation', 'distributed'):
            self.assertEqual(config[key], parent_config[key])

    def test_real_adam_state_restores_exactly_on_cpu(self):
        config, _ = resume.read_contract(resume.ROOT / 'configs/foundir_rain_flux2_precision_resume_v1.json')
        saved = torch.load(resume.ROOT / config['resume']['optimizer'], map_location='cpu', weights_only=True)
        params = [torch.nn.Parameter(torch.zeros_like(s['exp_avg'])) for s in saved['optimizer']['state'].values()]
        optimizer = torch.optim.AdamW(params, lr=1e-5, weight_decay=0, fused=True)
        restore_adam(optimizer, saved, config_sha256=config['resume']['parent_config_sha256'], step=3000)
        self.assertEqual(len(optimizer.state), 100)
        self.assertTrue(all(float(s['step']) == 3000 for s in optimizer.state.values()))

    def test_modified_config_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'wrong.json'
            path.write_text('{}')
            with self.assertRaisesRegex(RuntimeError, 'contract changed'):
                resume.read_contract(path)


if __name__ == '__main__':
    unittest.main()
