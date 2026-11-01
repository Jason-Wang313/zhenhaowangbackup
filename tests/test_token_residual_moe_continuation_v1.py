import copy
import json
from pathlib import Path
import tempfile
import unittest

import torch

from smoke.token_residual_moe_v1 import (load_run_config, FORMAL_CONFIG, CONTINUATION_CONFIG,
                                        Samples, write_once, sha)
from smoke.token_residual_moe_continuation_v1 import verify_continuation
from smoke.train_foundir_token_residual_moe_v1 import resume_training
from lowlight_restoration.joint_moe_state import save_state, rng_state
from test_token_residual_moe_v1 import model, token_conditioning

ROOT = Path(__file__).resolve().parents[1]


class ContinuationTests(unittest.TestCase):
    def cfg(self):
        return load_run_config(ROOT, ROOT / CONTINUATION_CONFIG)

    def test_recipe_and_real_report_binding(self):
        cfg = self.cfg()
        evidence = verify_continuation(ROOT, cfg, artifacts=False, sources=False)
        self.assertEqual(evidence['start_step'], 1000)
        self.assertEqual(evidence['parent_metrics']['blur_full']['lpips'], .18342951432234736)
        self.assertFalse(cfg['fresh_optimizer'])
        self.assertTrue(cfg['restore_old_optimizer'])
        self.assertEqual(cfg['steps'], 2000)

    def test_loss_change_rejected(self):
        cfg = self.cfg(); cfg['blur_lpips_weight'] *= 2
        with self.assertRaises(RuntimeError):
            verify_continuation(ROOT, cfg, artifacts=False, sources=False)

    def test_failed_parent_report_rejected(self):
        cfg = self.cfg()
        with tempfile.TemporaryDirectory() as folder:
            report = json.loads((ROOT / cfg['continuation']['report']).read_text())
            report['full_result']['status'] = 'failed'
            path = Path(folder) / 'report.json'; write_once(path, report)
            cfg['continuation'].update(report=str(path), report_sha256=sha(path))
            with self.assertRaises(RuntimeError):
                verify_continuation(ROOT, cfg, artifacts=False, sources=False)

    def test_wrong_receipt_binding_rejected(self):
        cfg = self.cfg(); cfg['continuation']['receipt_sha256'] = 'wrong'
        with self.assertRaises(RuntimeError):
            verify_continuation(ROOT, cfg, artifacts=False, sources=False)

    def test_next_samples_not_replay_and_full_evaluations(self):
        cfg = self.cfg()
        row = {'record_id': 't', 'source_group': 't', 'role': 'direct_endpoint_train', 'split': 'train'}
        rain = [{'record_id': f'r{i}', 'source_group': f'r{i}', 'split': 'validation'} for i in range(100)]
        blur = [{'record_id': f'b{i}', 'source_group': f'b{i}', 'split': 'validation'} for i in range(2595)]
        s = Samples('.', cfg, {'rain': [row], 'blur': [row]},
                    {'rain': rain, 'blur_preview': blur[:100], 'blur_full': blur})
        specs = s.all_specs()
        self.assertEqual(len(specs), 34695)
        self.assertEqual(len({v['key'] for v in specs}), 34695)
        self.assertEqual(specs[0]['key'], 'train_000384000')
        self.assertEqual(specs[31999]['key'], 'train_000415999')
        self.assertEqual(len(s.evaluation_specs(specs, full=True)), 2695)
        self.assertEqual(cfg['full_evaluation_steps'], [1000, 1500, 2000])

    def test_actual_adam_1000_transition_and_new_lineage_resume(self):
        torch.set_num_threads(1)
        net = model(); opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=0)
        x, f = torch.randn(1, 3, 6), torch.randn(1, 4)
        def step(n, o):
            o.zero_grad()
            with token_conditioning(n, f):
                n(x).square().mean().backward()
            o.step()
        for _ in range(1000):
            step(net, opt)
        old = {'config_sha256': 'old', 'source_sha256': {'code': 'old'}}
        new = {'config_sha256': 'new', 'source_sha256': {'code': 'new'}}
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); prior = root/'prior'; current = root/'current'
            write_once(prior/'launch.json', {'binding': old})
            save_state(net, opt, prior, step=1000, mode='formal', arm='token_residual',
                       config_sha='old', sources=old['source_sha256'], rank_rng=[rng_state() for _ in range(8)])
            state = copy.deepcopy(opt.state_dict())
            expected_draw = torch.rand(9)
            n = model(); o = torch.optim.AdamW(n.parameters(), lr=1e-3, weight_decay=0)
            receipt = prior/'step_001000.resume.json'
            with self.assertRaises(RuntimeError):
                resume_training(n, o, receipt, new, 0, mode='formal', total_steps=2000)
            self.assertEqual(resume_training(n, o, receipt, old, 0, mode='formal', total_steps=1000), 1000)
            torch.testing.assert_close(torch.rand(9), expected_draw, rtol=0, atol=0)
            for a, b in zip(net.parameters(), n.parameters()):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            for key, value in o.state_dict()['state'].items():
                for name, tensor in value.items():
                    torch.testing.assert_close(tensor, state['state'][key][name], rtol=0, atol=0)
            step(net, opt); step(n, o)
            for a, b in zip(net.parameters(), n.parameters()):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            write_once(current/'launch.json', {'binding': new})
            save_state(n, o, current, step=1001, mode='formal', arm='token_residual',
                       config_sha='new', sources=new['source_sha256'], rank_rng=[rng_state() for _ in range(8)])
            other = model(); other_opt = torch.optim.AdamW(other.parameters(), lr=1e-3, weight_decay=0)
            self.assertEqual(resume_training(other, other_opt, current/'step_001001.resume.json',
                                            new, 0, mode='formal', total_steps=2000), 1001)
            for a, b in zip(n.parameters(), other.parameters()):
                torch.testing.assert_close(a, b, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
