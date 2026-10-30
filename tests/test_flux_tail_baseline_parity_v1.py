import copy
import os
from pathlib import Path
import subprocess
import sys
import types
import unittest
from unittest.mock import patch

import torch

from smoke import diagnose_flux_tail_baseline_parity_v1 as diagnostic


class ParityDiagnosticsTests(unittest.TestCase):
    def test_uint8_difference_does_not_wrap_and_detects_one_level(self):
        result = diagnostic.difference(torch.tensor([0, 255], dtype=torch.uint8),
                                       torch.tensor([255, 254], dtype=torch.uint8))
        self.assertEqual(result['max_abs'], 255)
        self.assertEqual(result['mean_abs'], 128)
        self.assertEqual(result['signed_mean'], -127)
        self.assertEqual(result['different_elements'], 2)
        self.assertFalse(result['equal'])

    def test_first_difference_uses_execution_order_and_fails_on_bad_data(self):
        left = {'encoded': torch.tensor([1.]), 'noise': torch.tensor([2.]), 'output': torch.tensor([3.])}
        right = copy.deepcopy(left)
        right['noise'] += .01
        right['output'] += 1
        result = diagnostic.compare_traces(left, right, ['encoded', 'noise', 'output'])
        self.assertEqual(result['first_different_stage'], 'noise')
        with self.assertRaises(RuntimeError):
            diagnostic.compare_traces(left, right, ['encoded'])
        with self.assertRaises(RuntimeError):
            diagnostic.difference(torch.tensor([float('nan')]), torch.tensor([1.]))

    def test_rejects_other_sources_and_roles(self):
        identity = dict(index=0, role='train', row=dict(split='train', category='05Noise'))
        diagnostic.validate_identity(identity)
        for mutation in [dict(index=1), dict(role='engineering_evaluation'),
                         dict(row=dict(split='validation', category='05Noise'))]:
            with self.assertRaises(RuntimeError):
                diagnostic.validate_identity(identity | mutation)

    def test_trace_observes_fifty_steps_and_restores_hooks(self):
        class Model(torch.nn.Module):
            def forward(self, x, **kwargs):
                return x * .1

        ae = types.SimpleNamespace(encode=lambda x: x * .5, decode=lambda x: x * 2)
        flow = types.SimpleNamespace(ae=ae, model=Model(), context={'cfg_context': torch.ones(1), 'cfg_ids': torch.zeros(1)})
        original_encode, original_decode = ae.encode, ae.decode

        def fake_restore(model, autoencoder, image, context, ids, seed):
            value = autoencoder.encode(image)
            for i in range(50):
                value = value - model(x=value, x_ids=ids, timesteps=torch.tensor([i]), ctx=context, ctx_ids=ids)
            return autoencoder.decode(value).clamp(-1, 1)

        image = torch.full((1, 3, 2, 2), .2)
        expected = fake_restore(flow.model, ae, image, flow.context['cfg_context'], flow.context['cfg_ids'], 1)
        with patch('smoke.qualify_foundir_flux_energy_v1.restore', fake_restore):
            trace, order = diagnostic.traced_restore(flow, image, 1)
        self.assertTrue(torch.equal(trace['raw_clamped'], expected))
        self.assertIn('step_49_velocity', trace)
        self.assertEqual(len(order), len(set(order)))
        self.assertIs(ae.encode, original_encode)
        self.assertIs(ae.decode, original_decode)
        self.assertEqual(len(flow.model._forward_hooks), 0)
        with patch('smoke.qualify_foundir_flux_energy_v1.restore', side_effect=RuntimeError('simulated failure')):
            with self.assertRaises(RuntimeError):
                diagnostic.traced_restore(flow, image, 1)
        self.assertIs(ae.encode, original_encode)
        self.assertEqual(len(flow.model._forward_pre_hooks), 0)

    def test_launcher_ignores_stale_cluster_settings_and_rejects_duplicate_devices(self):
        script = Path(diagnostic.ROOT)/'smoke/run_flux_tail_baseline_parity_2gpu_v1.sh'
        env = dict(os.environ, FLUX_TAIL_V3_PYTHON=sys.executable, FLUX_TAIL_PARITY_DRY_RUN='1',
                   NNODES='3', NODE_RANK='2', MASTER_ADDR='10.1.1.1', CUDA_VISIBLE_DEVICES='0,1')
        result = subprocess.run(['bash', str(script)], env=env, cwd='/tmp', capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('NODES=1 GPUS=2 WALL_CAP=900', result.stdout)
        self.assertNotIn('torch.distributed', result.stdout)
        env['CUDA_VISIBLE_DEVICES'] = '0,0'
        result = subprocess.run(['bash', str(script)], env=env, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)


if __name__ == '__main__':
    unittest.main()
