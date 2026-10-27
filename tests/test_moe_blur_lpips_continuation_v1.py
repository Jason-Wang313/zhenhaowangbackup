from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from smoke import moe_blur_lpips_continuation_v1 as run
from lowlight_restoration.joint_moe_state import save_state, load_state, rng_state
from lowlight_restoration.rain_blur_moe_data import task_position
from tests.test_joint_vlm_moe import build, step


class ContinuationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_only_blur_lpips_changes(self):
        original = dict(flow_mse=.1, latent_l1=.25, rgb_mse=10., charbonnier=1., highpass=.1, lpips=.05)
        before = dict(original)
        for variant in run.VARIANTS:
            for task in (0, 1):
                value = run.loss_weights(original, variant, task)
                changes = [k for k in value if value[k] != original[k]]
                self.assertEqual(changes, ['lpips'] if (variant, task) == ('blur_lpips4x', 1) else [])
                self.assertEqual(value['lpips'], .20 if changes else .05)
        self.assertEqual(before, original)
        for variant, task in [('invalid', 0), ('control', 2)]:
            with self.assertRaises(ValueError): run.loss_weights(original, variant, task)

    def test_loss_gradient_changes_only_by_extra_lpips(self):
        from lowlight_restoration.joint_vlm_moe import restoration_loss
        output = torch.randn(1, 3, 8, 8, requires_grad=True)
        result = dict(decoded=output, degraded=torch.zeros_like(output), prediction=output, predicted_clean=output)
        weights = dict(flow_mse=.1, latent_l1=.25, rgb_mse=10., charbonnier=1., highpass=.1, lpips=.05)
        metric = lambda a, b: (a-b).square().mean()
        gt = torch.ones_like(output)
        losses = [restoration_loss(result, torch.zeros_like(output), gt, metric, lambda x: (x, None),
                                  run.loss_weights(weights, variant, 1)) for variant in run.VARIANTS]
        a, b = [torch.autograd.grad(loss, output, retain_graph=True)[0] for loss, _ in losses]
        extra = torch.autograd.grad(losses[0][1]['lpips'] * .15, output)[0]
        self.assertTrue(torch.allclose(b-a, extra, atol=1e-7))

    def test_completed_parent_restores_adam_rng_and_exact_next_update(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, ao = build('image_only', checkpointed=True)
            step(a, ao)
            for state in ao.state.values(): state['step'].fill_(10000)
            save_state(a, ao, tmp, step=10000, mode='formal', arm='image_only',
                       config_sha='parent', sources={'file': 'hash'}, rank_rng=[rng_state() for _ in range(8)])
            step(a, ao)
            b, bo = build('image_only', checkpointed=True)
            expected = [{k: v for k, v in g.items() if k != 'params'} for g in bo.param_groups]
            receipt = load_state(b, bo, Path(tmp)/'step_010000.resume.json', arm='image_only',
                config_sha='parent', sources={'file': 'hash'}, rank=0, total_steps=11000)
            run.verify_optimizer_groups(bo, expected)
            self.assertEqual(receipt['next_sample_position'], 320000)
            step(b, bo)
            for x, y in zip(a.parameters(), b.parameters()): self.assertTrue(torch.equal(x, y))
            for x, y in zip(ao.state.values(), bo.state.values()):
                self.assertTrue(torch.equal(x['exp_avg'], y['exp_avg']))
                self.assertTrue(torch.equal(x['exp_avg_sq'], y['exp_avg_sq']))
                self.assertEqual(int(y['step']), 10001)
            bo.param_groups[0]['lr'] *= 2
            with self.assertRaises(RuntimeError): run.verify_optimizer_groups(bo, expected)

    def test_balanced_continuation_positions(self):
        for step_number in (10000, 10499, 10999):
            counts = [0, 0]
            for micro in range(4):
                for rank in range(8):
                    index = step_number * 4 + micro
                    task, pos = task_position(index, rank, 8)
                    counts[task] += 1
                    self.assertEqual(task, micro % 2)
                    self.assertGreaterEqual(pos, 160000)
            self.assertEqual(counts, [16, 16])

    def test_config_and_sequential_launcher(self):
        cfg = run.read_config()
        self.assertEqual(cfg['training']['end_step'] - cfg['training']['start_step'], 1000)
        self.assertEqual(cfg['variants_in_order'], list(run.VARIANTS))
        launcher = (run.ROOT/'smoke/run_foundir_moe_blur_lpips_pair_v1_8gpu.sh').read_text()
        self.assertIn('for VARIANT in control blur_lpips4x; do', launcher)
        self.assertIn('--nproc_per_node=8', launcher)
        self.assertNotIn('--nproc_per_node=4', launcher)
        self.assertIn('flock -n 9', launcher)
        self.assertIn('flock -n 8', launcher)

    def fixture_results(self):
        def metric(step_number, psnr, lpips):
            return dict(step=step_number, route='learned', rows=[dict(record_id=str(i), source_group=str(i),
                input_psnr=20., input_lpips=.5, output_psnr=psnr, output_lpips=lpips) for i in range(3)])
        a = dict(status='complete', variant='control', engineering_pass=True, start_step=10000,
                 steps=11000, optimizer_steps_this_run=1000, this_launch_presentations=32000,
                 protected_pixels_accessed=False, config_sha256='cfg', source_sha256={'a': 'b'}, data={},
                 hardware={'hostname': 'node', 'gpu_uuids': ['id']}, gpu_names=['H100']*8,
                 parent_receipt_sha256='r', parent_weights_sha256='w', parent_optimizer_sha256='o',
                 sample_trace_sha256_by_rank=['trace']*8, optimizer_group_settings=[{'lr': 1e-5}],
                 initial_metrics={'rain': metric(10000, 34.7, .06), 'blur_preview': metric(10000, 26., .24)},
                 final_metrics={'rain': metric(11000, 34.7, .06), 'blur_full': metric(11000, 26., .23)})
        b = copy.deepcopy(a); b['variant'] = 'blur_lpips4x'
        b['final_metrics']['blur_full'] = metric(11000, 25.95, .21)
        return a, b

    def test_repair_signal_is_not_absolute_gate_or_novelty(self):
        a, b = self.fixture_results()
        result = run.compare_results(a, b)
        self.assertTrue(result['predeclared_repair_signal'])
        self.assertFalse(result['absolute_joint_gate_pass']['blur_lpips4x'])
        self.assertFalse(result['novelty_or_VLM_advantage_established'])
        b['final_metrics']['rain']['rows'][0]['output_lpips'] = .5
        self.assertFalse(run.compare_results(a, b)['predeclared_repair_signal'])

    def test_comparison_rejects_other_node_samples_and_incomplete_results(self):
        a, b = self.fixture_results()
        for field, bad in [('hardware', {'hostname': 'other'}), ('sample_trace_sha256_by_rank', ['wrong']),
                           ('steps', 10500), ('parent_optimizer_sha256', 'different')]:
            changed = copy.deepcopy(b); changed[field] = bad
            with self.assertRaises(ValueError): run.compare_results(a, changed)


if __name__ == '__main__':
    unittest.main()
