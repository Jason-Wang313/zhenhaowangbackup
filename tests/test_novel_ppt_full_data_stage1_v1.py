import copy
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from smoke import novel_ppt_full_data_state_v1 as stateio
from smoke.train_foundir_proposal_relative_energy_noise_v1_ddp import sample_global_batch


class FullDataStateTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(27)
        self.device = torch.device('cpu')

    def test_lr_keeps_full_horizon_and_no_rewarmup(self):
        spec = json.loads((stateio.ROOT/'configs/foundir_dual_retry_execution_v1.json').read_text())['ppt']
        self.assertEqual(stateio.learning_rate(spec, 100), .0001)
        self.assertGreater(stateio.learning_rate(spec, 1706), .00009)
        self.assertEqual(stateio.learning_rate(spec, 8530), .000001)
        self.assertNotEqual(stateio.learning_rate(spec, 101), stateio.learning_rate(spec, 1))

    def test_sampling_after_resume_exactly_matches_next_global_step(self):
        groups = {str(i): [dict(record_id=str(i), source_group=str(i), split='train')] for i in range(40)}
        before = sample_global_batch(groups, task='noise', seed=27, count=32, step=100)
        next_full = sample_global_batch(groups, task='noise', seed=27, count=32, step=101)
        self.assertNotEqual(before, next_full)
        resumed = sample_global_batch(groups, task='noise', seed=27, count=32, step=100+1)
        self.assertEqual(next_full, resumed)

    def toy(self):
        model = torch.nn.Linear(3, 2)
        return model, torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)

    def step(self, model, optimizer):
        optimizer.zero_grad()
        x = torch.rand(2, 3) + random.random() + float(np.random.rand())
        model(x).square().mean().backward()
        optimizer.step()

    def payload(self):
        model, optimizer = self.toy()
        self.step(model, optimizer)
        state = dict(mode=stateio.MODE, world_size=8, step=1, next_sample_step=2,
                     config_sha256='config', source_sha256={'trainer': 'hash'},
                     parameter_names=list(dict(model.named_parameters())),
                     model=copy.deepcopy(model.state_dict()), optimizer=copy.deepcopy(optimizer.state_dict()),
                     rank_rng=[stateio.capture_rng(self.device) for _ in range(8)])
        return model, optimizer, state

    def test_restored_adam_and_rng_produce_identical_next_update(self):
        model, optimizer, state = self.payload()
        stateio.validate_state(state, 'config', {'trainer': 'hash'})
        self.step(model, optimizer)
        restored, new_optimizer = self.toy()
        stateio.apply_state(restored, new_optimizer, state, 0, self.device)
        self.step(restored, new_optimizer)
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, restored.state_dict()[key]))
        for left, right in zip(optimizer.state.values(), new_optimizer.state.values(), strict=True):
            for key in left:
                self.assertTrue(torch.equal(left[key], right[key]))

    def test_rejects_trainfit_wrong_source_and_broken_sample_state(self):
        _, _, state = self.payload()
        for field, value in [('mode', 'trainfit'), ('mode', 'qualification'), ('world_size', 4),
                             ('next_sample_step', 1), ('rank_rng', []), ('config_sha256', 'other')]:
            bad = {**state, field: value}
            with self.assertRaises(RuntimeError):
                stateio.validate_state(bad, 'config', {'trainer': 'hash'})
        with self.assertRaises(RuntimeError):
            stateio.validate_state(state, 'config', {'trainer': 'changed'})

    def test_rejects_missing_adam_and_nonfinite_weights(self):
        _, _, state = self.payload()
        bad = copy.deepcopy(state)
        bad['optimizer']['state'].pop(next(iter(bad['optimizer']['state'])))
        with self.assertRaises(RuntimeError):
            stateio.validate_state(bad, 'config', {'trainer': 'hash'})
        bad = copy.deepcopy(state)
        bad['model']['weight'][0, 0] = float('nan')
        with self.assertRaises(RuntimeError):
            stateio.validate_state(bad, 'config', {'trainer': 'hash'})

    def test_atomic_receipt_roundtrip_and_corruption_guard(self):
        model, optimizer, _ = self.payload()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root/'config.json'
            config.write_text(json.dumps({'output_root': 'outputs'}))
            output = root/'outputs/run'; output.mkdir(parents=True)
            source = {'trainer': 'hash'}
            def gather(items, item):
                items[:] = [item]*8
            with patch.object(stateio, 'ROOT', root), patch.object(stateio, 'CONFIG', config), \
                 patch.object(stateio, 'binding', return_value=source), \
                 patch.object(stateio.dist, 'all_gather_object', side_effect=gather), \
                 patch.object(stateio.dist, 'broadcast_object_list'):
                receipt = stateio.save(output, model, optimizer, 1, stateio.sha(config), source, self.device, 0)
                loaded, _ = stateio.read_resume(root/receipt['receipt'], stateio.sha(config), source)
                self.assertEqual(loaded['next_sample_step'], 2)
                self.assertFalse(list(output.glob('*.partial')))
                with self.assertRaises(RuntimeError):
                    stateio.save(output, model, optimizer, 1, stateio.sha(config), source, self.device, 0)
                with (output/receipt['checkpoint']).open('ab') as stream:
                    stream.write(b'corrupt')
                with self.assertRaises(RuntimeError):
                    stateio.read_resume(root/receipt['receipt'], stateio.sha(config), source)

    def test_stage_is_bounded_and_never_initializes_from_trainfit(self):
        cfg = json.loads(stateio.CONFIG.read_text())
        self.assertEqual(cfg['steps'], 1706)
        self.assertEqual(cfg['learning_rate_horizon'], 8530)
        self.assertEqual(cfg['train_records'], 54585)
        self.assertEqual(cfg['development_records'], 256)
        self.assertFalse(cfg['qualification_or_trainfit_may_seed'])
        self.assertFalse(cfg['auto_continue_to_8530'])


if __name__ == '__main__':
    unittest.main()
