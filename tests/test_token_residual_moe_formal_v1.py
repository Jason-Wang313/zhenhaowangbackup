import copy
import json
from pathlib import Path
import tempfile
import unittest

import torch

from smoke.token_residual_moe_v1 import (CONFIG, FORMAL_CONFIG, Samples, load_run_config,
    verify_qualification, feature_binding, write_once, sha)
from smoke.train_foundir_token_residual_moe_v1 import resume_training
from lowlight_restoration.joint_moe_state import save_state, rng_state
from test_token_residual_moe_v1 import model, token_conditioning

ROOT = Path(__file__).resolve().parents[1]


class FormalTests(unittest.TestCase):
    def test_same_recipe_only_schedule_changes(self):
        q = load_run_config(ROOT)
        f = load_run_config(ROOT, ROOT / FORMAL_CONFIG)
        changed = {k for k in q if q[k] != f[k]}
        self.assertEqual(changed, {'mode', 'steps', 'save_every', 'milestones', 'full_evaluation_steps',
            'output_root', 'wall_hours_estimate', 'allocated_gpu_hours_estimate', 'hard_wall_seconds_per_launch'})
        self.assertEqual(f['steps'], 1000)
        self.assertIn('step_011000', f['initial_weights'])
        self.assertTrue(f['fresh_optimizer'])

    def test_no_hyperparameter_override(self):
        f = json.loads((ROOT / FORMAL_CONFIG).read_text())
        f['seed'] = 123
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / 'bad.json'
            write_once(p, f)
            with self.assertRaises(RuntimeError):
                load_run_config(ROOT, p)

    def test_actual_qualification_report_identity_and_failure_rejection(self):
        f = load_run_config(ROOT, ROOT / FORMAL_CONFIG)
        evidence = verify_qualification(ROOT, f, artifacts=False, sources=False)
        self.assertEqual(evidence['qualified_source_commit'][:7], '528383c')
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            report = json.loads((ROOT / f['qualification_report']).read_text())
            report['full_result']['status'] = 'failed'
            p = root / 'bad.json'; write_once(p, report)
            f.update(qualification_report='bad.json', qualification_report_sha256=sha(p))
            with self.assertRaises(RuntimeError):
                verify_qualification(root, f, artifacts=False, sources=False)

    def test_canonical_cache_has_full_dev_without_duplicate_preview(self):
        cfg = load_run_config(ROOT, ROOT / FORMAL_CONFIG)
        train_row = {'record_id': 't', 'source_group': 't', 'role': 'direct_endpoint_train', 'split': 'train'}
        rain = [{'record_id': f'r{i}', 'source_group': f'r{i}', 'split': 'validation'} for i in range(100)]
        blur = [{'record_id': f'b{i}', 'source_group': f'b{i}', 'split': 'validation'} for i in range(2595)]
        sample = Samples('.', cfg, {'rain': [train_row], 'blur': [train_row]},
                         {'rain': rain, 'blur_preview': blur[:100], 'blur_full': blur})
        specs = sample.all_specs()
        self.assertEqual(len(specs), 34695)
        self.assertEqual(len({s['key'] for s in specs}), 34695)
        preview = sample.evaluation_specs(specs)
        full = sample.evaluation_specs(specs, full=True)
        self.assertEqual((len(preview), len(full)), (200, 2695))
        canonical = {s['key']: s for s in full}
        for s in preview:
            self.assertIs(s, canonical[s['key']])
        counts = [sample.train_spec(i)['task_id'] for i in range(32)]
        self.assertEqual(counts.count(0), 16)
        self.assertEqual(sample.train_spec(31999)['key'], 'train_000383999')
        with self.assertRaises(ValueError):
            sample.train_spec(32000)

    def test_actual_small_receipt_binding_is_enforced(self):
        cfg = load_run_config(ROOT, ROOT / FORMAL_CONFIG)
        report = json.loads((ROOT / cfg['qualification_report']).read_text())
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            report['artifact_hashes'] = {'x/result.json': {'sha256': 'wrong'}}
            write_once(root / 'x/result.json', {'status': 'fake'})
            write_once(root / 'report.json', report)
            cfg.update(qualification_report='report.json', qualification_report_sha256=sha(root/'report.json'))
            with self.assertRaises(RuntimeError):
                verify_qualification(root, cfg, artifacts=True, sources=False)

    def test_checkpoint_formal_mode_enforced_and_final_eval_resume_supported(self):
        net = model(); opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
        with token_conditioning(net, torch.randn(1, 4)):
            net(torch.randn(1, 3, 6)).square().mean().backward()
        opt.step()
        binding = {'config_sha256': 'test', 'source_sha256': {'code': 'test'}}
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write_once(root / 'launch.json', {'binding': binding})
            save_state(net, opt, root, step=1, mode='formal', arm='token_residual',
                config_sha='test', sources=binding['source_sha256'], rank_rng=[rng_state() for _ in range(8)])
            receipt = root / 'step_000001.resume.json'
            restored = model(); restored_opt = torch.optim.AdamW(restored.parameters(), lr=1e-3)
            with self.assertRaises(RuntimeError):
                resume_training(restored, restored_opt, receipt, binding, 0, mode='qualification', total_steps=25)
            # total_steps=1 exercises evaluation-only restore at the final optimizer step.
            self.assertEqual(resume_training(restored, restored_opt, receipt, binding, 0, mode='formal', total_steps=1), 1)
            for a, b in zip(net.parameters(), restored.parameters()):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            bad = dict(binding, config_sha256='wrong')
            with self.assertRaises(RuntimeError):
                resume_training(restored, restored_opt, receipt, bad, 0, mode='formal', total_steps=1000)


if __name__ == '__main__':
    unittest.main()
