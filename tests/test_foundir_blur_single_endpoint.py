from __future__ import annotations

import copy
import json
import tempfile
import unittest
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from lowlight_restoration.blur_single_endpoint import (
    BlurOnlyDataset, save_state, load_state, rng_state, check_adam,
)
from lowlight_restoration.rain_blur_image_moe import inject_image_moe, training_routes
from lowlight_restoration.rain_blur_moe_data import RainBlurDataset, task_position
from smoke import train_foundir_blur_single_endpoint_v1_ddp as run
from tests.test_foundir_rain_blur_image_moe import ToyModel
from tests import test_foundir_rain_precision_full as data_tests


def model_and_optimizer():
    torch.manual_seed(73)
    model = ToyModel()
    inject_image_moe(model, indices=[0, 1], experts=1)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-5, weight_decay=0)
    return model, optimizer


def toy_step(model, optimizer):
    optimizer.zero_grad(set_to_none=True)
    for _ in range(2):
        (model(torch.randn(1, 7, 4)).square().mean() * .25).backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
    optimizer.step()


def ddp_worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method='file://' + rendezvous, rank=rank,
                            world_size=2, timeout=timedelta(seconds=120))
    try:
        model, optimizer = model_and_optimizer()
        wrapped = torch.nn.parallel.DistributedDataParallel(model, gradient_as_bucket_view=True)
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            for micro in range(2):
                with wrapped.no_sync() if micro == 0 else nullcontext():
                    (wrapped(torch.randn(1, 5, 4) + rank).square().mean() * .25).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
        check_adam(optimizer, 3)
        vector = torch.cat([p.detach().flatten() for p in model.parameters() if p.requires_grad])
        peers = [torch.empty_like(vector) for _ in range(2)]
        dist.all_gather(peers, vector)
        assert torch.equal(peers[0], peers[1])
    finally:
        dist.destroy_process_group()


class BlurSingleTests(unittest.TestCase):
    def test_exact_48000_positions_match_parent_blur_only(self):
        positions = []
        for step in range(3000):
            for micro in range(2):
                for rank in range(8):
                    task, old_position = task_position(step * 4 + micro * 2 + 1, rank, 8)
                    new_position = step * 16 + micro * 8 + rank
                    self.assertEqual(task, 1)
                    self.assertEqual(old_position, new_position)
                    positions.append(new_position)
        self.assertEqual(positions, list(range(48000)))

    def test_crop_and_source_selection_exactly_match_parent(self):
        with tempfile.TemporaryDirectory() as directory:
            row = data_tests.FullRainTests().make_row(directory)
            old = RainBlurDataset(directory, {'rain': [row], 'blur': [row]}, steps=3, rank=5)
            new = BlurOnlyDataset(directory, [row], steps=3, rank=5)
            for i in range(6):
                a, b = old[2 * i + 1], new[i]
                self.assertEqual(a['record_id'], b['record_id'])
                self.assertTrue(torch.equal(a['lq'], b['lq']))
                self.assertTrue(torch.equal(a['gt'], b['gt']))
            resumed = BlurOnlyDataset(directory, [row], steps=3, rank=5, start_step=2)
            self.assertEqual(len(resumed), 2)
            self.assertEqual(resumed[0]['position'], new[4]['position'])
            self.assertTrue(torch.equal(resumed[0]['lq'], new[4]['lq']))

    def test_nontrain_rows_and_invalid_world_rejected(self):
        row = {'split': 'validation', 'role': 'direct_endpoint_train'}
        with self.assertRaises(ValueError):
            BlurOnlyDataset('.', [row], steps=25, rank=0)
        with self.assertRaises(ValueError):
            BlurOnlyDataset('.', [], steps=25, rank=0, world=4)

    def test_blur_gradient_matches_hard_routed_parent_scale(self):
        torch.manual_seed(3)
        moe = ToyModel()
        single = copy.deepcopy(moe)
        inject_image_moe(moe, indices=[0, 1], experts=2, hidden=8)
        inject_image_moe(single, indices=[0, 1], experts=1)
        for micro in range(4):
            x = torch.randn(1, 7, 4)
            with training_routes(moe, torch.tensor([micro % 2]), 1):
                (moe(x).square().mean() / 4).backward()
            if micro % 2:
                (single(x).square().mean() * .5 / 2).backward()
        reference = dict(moe.named_parameters())
        for name, param in single.named_parameters():
            if param.requires_grad:
                self.assertTrue(torch.allclose(param.grad, reference[name.replace('experts.0', 'experts.1')].grad, atol=1e-7))

    def test_resume_matches_uninterrupted_weights_adam_and_rng(self):
        a, oa = model_and_optimizer()
        toy_step(a, oa)
        with tempfile.TemporaryDirectory() as directory:
            save_state(a, oa, directory, step=1, mode='formal', config_sha='x', sources={},
                       rank_rng=[rng_state()], initial_parity_pass=True)
            toy_step(a, oa)
            expected_random = torch.randn(7)
            b, ob = model_and_optimizer()
            meta = load_state(b, ob, Path(directory) / 'step_000001.resume.json', config_sha='x',
                             sources={}, rank=0, world=1, total_steps=3)
            self.assertEqual(meta['next_sample_position'], 16)
            toy_step(b, ob)
            self.assertTrue(torch.equal(torch.randn(7), expected_random))
            for p, q in zip(a.parameters(), b.parameters()):
                self.assertTrue(torch.equal(p, q))
            for sa, sb in zip(oa.state.values(), ob.state.values()):
                for key in ('step', 'exp_avg', 'exp_avg_sq'):
                    self.assertTrue(torch.equal(sa[key], sb[key]))

    def test_resume_rejects_qualification_hash_source_and_sample_tampering(self):
        model, optim = model_and_optimizer()
        toy_step(model, optim)
        with tempfile.TemporaryDirectory() as directory:
            saved = save_state(model, optim, directory, step=1, mode='formal', config_sha='x', sources={},
                               rank_rng=[rng_state()], initial_parity_pass=True)
            receipt = Path(directory) / 'step_000001.resume.json'
            mutations = [{'mode': 'qualification'}, {'config_sha256': 'bad'}, {'source_sha256': {'bad': 'x'}},
                         {'next_sample_position': 32}, {'weights_sha256': 'bad'}, {'optimizer_sha256': 'bad'},
                         {'world_size': 8}, {'initial_parity_pass': False}, {'weights': '../elsewhere'}]
            for change in mutations:
                # Ephemeral fixture only: production writes use completion receipts.
                with patch.object(Path, 'read_text', return_value=json.dumps({**saved, **change})):
                    with self.assertRaises(RuntimeError):
                        load_state(model, optim, receipt, config_sha='x', sources={}, rank=0, world=1, total_steps=3)
            with self.assertRaisesRegex(RuntimeError, 'preserve existing'):
                save_state(model, optim, directory, step=1, mode='formal', config_sha='x', sources={},
                           rank_rng=[rng_state()], initial_parity_pass=True)

    def test_nonfinite_adam_rejected(self):
        model, optimizer = model_and_optimizer()
        toy_step(model, optimizer)
        next(iter(optimizer.state.values()))['exp_avg'].fill_(float('nan'))
        with self.assertRaisesRegex(RuntimeError, 'nonfinite'):
            check_adam(optimizer, 1)

    def test_actual_contract_and_no_protected_source_overlap(self):
        config, _, _, training, development, data = run.read_contract(run.ROOT / 'configs/foundir_blur_single_endpoint_v1.json')
        self.assertEqual(config['training']['objective_scale'], .5)
        self.assertEqual(len(training['blur']), 105057)
        dev = development['rain'] + development['blur_full']
        self.assertFalse({r['source_group'] for r in training['blur']} & {r['source_group'] for r in dev})
        self.assertFalse({r['gt_sha256'] for r in training['blur']} & {r['gt_sha256'] for r in dev})
        self.assertEqual(data['development_records']['blur_full'], 2595)

    def test_parity_rejects_changed_inputs_outputs(self):
        _, config, *_ = run.read_contract(run.ROOT / 'configs/foundir_blur_single_endpoint_v1.json')
        row = {'record_id': 'x', 'source_group': 'g', 'input_psnr': 20., 'input_lpips': .3,
               'output_psnr': 25., 'output_lpips': .2}
        reference = {'rows': [row]}
        run.check_blur_parity(reference, reference, config)
        for field in ('output_psnr', 'input_lpips'):
            bad = {'rows': [{**row, field: 1.}]}
            with self.assertRaises(RuntimeError):
                run.check_blur_parity(bad, reference, config)

    def test_formal_cannot_use_parent_or_incomplete_qualification(self):
        receipt = {'status': 'complete', 'mode': 'qualification', 'engineering_pass': True,
                   'initial_parity_pass': True, 'config_sha256': run.CONFIG_SHA256,
                   'source_sha256': {}, 'data': {}}
        run.verify_qualification(receipt, {}, {})
        for change in ({'mode': 'formal'}, {'initial_parity_pass': False}, {'engineering_pass': False},
                       {'config_sha256': run.paired.CONFIG_SHA256}, {'source_sha256': {'bad': 'x'}}):
            with self.assertRaises(RuntimeError):
                run.verify_qualification({**receipt, **change}, {}, {})

    def test_two_rank_accumulation_and_checkpointed_ffn(self):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(ddp_worker, args=(str(Path(directory) / 'rendezvous'),), nprocs=2, join=True)


if __name__ == '__main__':
    unittest.main()
