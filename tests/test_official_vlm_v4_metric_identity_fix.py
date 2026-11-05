import copy
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import torch
from smoke import official_vlm_v4_metric_identity_fix as fix
from smoke.rain_native50_continue_state_v1 import parameter_hash


class Metric(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            self.net.weight.fill_(0.25)
        self.dummy_param = torch.nn.Parameter(torch.empty(0))
        self.requires_grad_(False)
        self.device = 'cpu'

    def forward(self, value):
        return self.net(value.to(self.dummy_param.device))


def identity(models):
    return {'version': '0.1.16', 'defaults': True,
            'state_hashes': {name: parameter_hash(model.state_dict())
                             for name, model in models.items()}}


class MetricIdentityFixTests(unittest.TestCase):
    def test_whole_module_transfer_keeps_registered_dummy_on_meta(self):
        old = torch.nn.Module()
        old.dummy_param = torch.nn.Parameter(torch.empty(0)).to('meta')
        self.assertNotIn('dummy_param', old.state_dict())
        fixed = Metric().to('meta')
        self.assertIn('dummy_param', fixed.state_dict())
        self.assertIsInstance(fixed.dummy_param, torch.nn.Parameter)
        self.assertEqual(fixed.dummy_param.device.type, 'meta')

    def test_cpu_construction_full_state_and_forward_preserved(self):
        models = {'psnr': Metric(), 'lpips': Metric(), 'ssim': Metric()}
        expected = identity(models)
        values = torch.tensor([[2., 4.]])
        before = {name: model(values).clone() for name, model in models.items()}
        with patch.object(fix, 'ORIGINAL_METRIC_MODELS', return_value=(models, expected)) as factory:
            result, actual = fix.checked_metric_models('cpu', expected)
        factory.assert_called_once_with('cpu')
        self.assertEqual(actual, expected)
        for name, model in result.items():
            self.assertTrue(torch.equal(model(values), before[name]))
            self.assertFalse(any(p.requires_grad for p in model.parameters()))

    def test_changed_cpu_weights_rejected(self):
        models = {'lpips': Metric()}
        expected = identity(models)
        with torch.no_grad():
            models['lpips'].net.weight.add_(1)
        with patch.object(fix, 'ORIGINAL_METRIC_MODELS', return_value=(models, identity(models))):
            with self.assertRaisesRegex(ValueError, 'CPU metric weights changed'):
                fix.checked_metric_models('cpu', expected)

    def test_corruption_during_transfer_rejected_even_with_matching_cpu_identity(self):
        models = {'lpips': Metric()}
        expected = identity(models)
        def corrupt(device):
            with torch.no_grad():
                models['lpips'].net.weight.add_(1)
            return models['lpips']
        with patch.object(fix, 'ORIGINAL_METRIC_MODELS', return_value=(models, expected)), \
             patch.object(models['lpips'], 'to', side_effect=corrupt):
            with self.assertRaisesRegex(ValueError, 'state changed during device transfer'):
                fix.checked_metric_models('cpu', expected)

    def test_repair_attestation_rejects_execution_source_changes(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(fix, 'ROOT', Path(tmp)):
            out = Path(tmp) / fix.x.OUTPUT
            for name in fix.SOURCES:
                path = Path(tmp) / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('synthetic repair source\n')
            fix.c.write_new(out / 'receipt.json', {'metric_identity': {}})
            fix.c.write_new(out / 'unlimited_execution.json', {'synthetic': True})
            fix.c.write_new(out / 'metric_identity_fix.json', fix.binding(out))
            with patch.object(fix.unlimited, 'verify', return_value=out):
                self.assertEqual(fix.verify(), out)
                (Path(tmp) / fix.SOURCES[0]).write_text('changed source\n')
                with self.assertRaisesRegex(ValueError, 'repair binding changed'):
                    fix.verify()

    def test_shell_keeps_unlimited_accounting_locks_and_no_retry_loop(self):
        root = Path(__file__).resolve().parents[1]
        run = root / fix.SOURCES[1]
        start = root / fix.SOURCES[2]
        subprocess.run(['bash', '-n', str(run), str(start)], check=True)
        text = run.read_text()
        self.assertNotIn('timeout', text)
        self.assertNotIn('CAP=', text)
        self.assertIn('flock -n 9', text)
        self.assertIn('flock -n 8', text)
        self.assertIn('trap finish_attempt EXIT', text)
        self.assertIn('--max_restarts=0', text)
        self.assertIn('used_gpu_seconds', text)


if __name__ == '__main__':
    unittest.main()
