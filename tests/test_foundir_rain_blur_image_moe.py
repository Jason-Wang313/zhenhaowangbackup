from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from contextlib import nullcontext
from datetime import timedelta

from lowlight_restoration.rain_blur_image_moe import (
    ImageExpertFFN, inject_image_moe, training_routes, router_classification_loss,
    warmup_fraction, trainable_state, clear_routing_graphs, load_expert_state,
)
from lowlight_restoration.rain_blur_moe_data import task_position, RainBlurDataset
from tests import test_foundir_rain_precision_full as data_tests
from smoke import train_foundir_rain_blur_image_moe_v1_ddp as run


class ToyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_size = 4
        self.img_mlp = nn.Sequential(nn.Linear(4, 8, bias=False), nn.SiLU(), nn.Linear(8, 4, bias=False))


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.double_blocks = nn.ModuleList([ToyBlock(), ToyBlock()])
        self.frozen = nn.Linear(4, 4)

    def forward(self, x):
        x = self.frozen(x)
        for block in self.double_blocks:
            x = x + block.img_mlp(x)
        return x


def ddp_worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method='file://' + rendezvous, rank=rank,
                            world_size=2, timeout=timedelta(seconds=120))
    try:
        torch.manual_seed(23)
        model = ToyModel()
        inject_image_moe(model, indices=[0, 1], hidden=8)
        wrapped = torch.nn.parallel.DistributedDataParallel(model, gradient_as_bucket_view=True)
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=1e-3)
        for step in range(4):
            optimizer.zero_grad(set_to_none=True)
            for micro in range(4):
                labels = torch.tensor([micro % 2])
                with wrapped.no_sync() if micro < 3 else nullcontext():
                    with training_routes(model, labels, warmup_fraction(step, 1, 2)):
                        prediction = wrapped(torch.randn(1, 5, 4) + rank)
                        loss = (prediction - labels[:, None, None]).square().mean()
                        loss = loss + router_classification_loss(model, labels)
                        (loss / 4).backward()
                    clear_routing_graphs(model)
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
            optimizer.step()
        vector = torch.cat([p.detach().flatten() for p in params])
        peers = [torch.empty_like(vector) for _ in range(2)]
        dist.all_gather(peers, vector)
        assert torch.equal(peers[0], peers[1]), 'DDP expert/router states diverged'
    finally:
        dist.destroy_process_group()


class MoETests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(23)

    def test_insertion_exact_fp32_and_bfloat16(self):
        for dtype in (torch.float32, torch.bfloat16):
            original = ToyBlock().img_mlp.to(dtype).eval()
            module = ImageExpertFFN(original, 4, hidden=8).eval()
            x = torch.randn(2, 7, 4).to(dtype)
            with torch.autocast('cpu', dtype=torch.bfloat16, enabled=dtype == torch.bfloat16):
                before, after = original(x), module(x)
            self.assertTrue(torch.equal(before, after))
            self.assertTrue(torch.allclose(module.last_weights.sum(-1), torch.ones(2)))

    def test_training_label_gate_never_enters_inference(self):
        model = ToyModel()
        modules = inject_image_moe(model, indices=[0, 1], hidden=8)
        model.eval()
        with self.assertRaises(RuntimeError):
            with training_routes(model, torch.tensor([0]), 1):
                pass
        model.train()
        with training_routes(model, torch.tensor([0, 1]), 1):
            model(torch.randn(2, 5, 4))
            for module in modules.values():
                self.assertTrue(torch.equal(module.last_weights, torch.eye(2)))
        self.assertTrue(all(m.labels is None and m.teacher_fraction == 0 for m in modules.values()))
        model.eval()
        model(torch.randn(2, 5, 4))
        self.assertTrue(all(torch.all(m.last_weights > 0) for m in modules.values()))

    def test_hard_then_learned_training_updates_both_experts_and_router(self):
        model = ToyModel()
        frozen = copy.deepcopy(model.frozen.state_dict())
        modules = inject_image_moe(model, indices=[0, 1], hidden=8)
        model.train()
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=1e-3)
        initial = {name: p.detach().clone() for name, p in model.named_parameters() if p.requires_grad}
        for step in range(8):
            optimizer.zero_grad()
            labels = torch.tensor([0, 1])
            with training_routes(model, labels, warmup_fraction(step, 2, 4)):
                result = model(torch.randn(2, 6, 4))
                loss = (result - labels[:, None, None]).square().mean() + router_classification_loss(model, labels)
                loss.backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in params))
            optimizer.step()
            clear_routing_graphs(model)
        for part in ('experts.0', 'experts.1', 'router'):
            self.assertTrue(any(not torch.equal(p, initial[name]) for name, p in model.named_parameters()
                                if p.requires_grad and part in name))
        self.assertTrue(all(torch.equal(value, frozen[key]) for key, value in model.frozen.state_dict().items()))

    def test_checkpointed_and_uncheckpointed_gradients_match(self):
        a = ImageExpertFFN(ToyBlock().img_mlp, 4, hidden=8)
        b = copy.deepcopy(a)
        b.activation_checkpointing = False
        x = torch.randn(2, 5, 4)
        for module in (a, b):
            module.labels, module.teacher_fraction = torch.tensor([0, 1]), .5
            module(x).square().mean().backward()
        for pa, pb in zip(a.parameters(), b.parameters()):
            self.assertTrue(torch.allclose(pa.grad, pb.grad))

    def test_constant_expert_is_eval_only(self):
        module = ImageExpertFFN(ToyBlock().img_mlp, 4, hidden=8)
        module.force_expert = 0
        with self.assertRaises(RuntimeError):
            module(torch.randn(1, 5, 4))
        module.eval()
        x = torch.randn(1, 5, 4)
        self.assertTrue(torch.equal(module(x), module.experts[0](x)))

    def test_dense_control_initialization_and_no_router(self):
        original = ToyModel()
        reference = copy.deepcopy(original)
        inject_image_moe(original, indices=[0, 1], experts=1)
        x = torch.randn(2, 4, 4)
        self.assertTrue(torch.equal(original(x), reference(x)))
        self.assertEqual(float(router_classification_loss(original, torch.tensor([0, 1]))), 0)
        self.assertTrue(all('img_mlp.experts.0' in name for name in trainable_state(original)))

    def test_checkpoint_state_can_rehydrate_exactly(self):
        from safetensors.torch import save_file, load_file
        model = ToyModel()
        inject_image_moe(model, indices=[0, 1], hidden=8)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'state.safetensors'
            save_file(trainable_state(model), str(path))
            loaded = load_file(str(path))
            self.assertEqual(set(loaded), set(trainable_state(model)))
            other = copy.deepcopy(model)
            with torch.no_grad():
                for name, p in other.named_parameters():
                    if p.requires_grad:
                        p.zero_()
            load_expert_state(other, path)
            x = torch.randn(1, 4, 4)
            self.assertTrue(torch.equal(model(x), other(x)))

    def test_balanced_absolute_schedule_no_task_repeats(self):
        positions = {0: set(), 1: set()}
        for step in range(20):
            counts = {0: 0, 1: 0}
            for micro in range(4):
                for rank in range(8):
                    task, position = task_position(step * 4 + micro, rank, 8)
                    self.assertNotIn(position, positions[task])
                    positions[task].add(position)
                    counts[task] += 1
            self.assertEqual(counts, {0: 16, 1: 16})
        self.assertEqual(positions, {0: set(range(320)), 1: set(range(320))})

    def test_spawn_data_and_aligned_pair(self):
        with tempfile.TemporaryDirectory() as directory:
            row = data_tests.FullRainTests().make_row(directory)
            dataset = RainBlurDataset(directory, {'rain': [row], 'blur': [row]}, steps=1, rank=3)
            loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=2,
                                                multiprocessing_context='spawn')
            batches = list(loader)
            self.assertEqual([int(b['task_id'][0]) for b in batches], [0, 1, 0, 1])
            self.assertEqual([int(b['position'][0]) for b in batches], [3, 11, 19, 27])
            self.assertTrue(all(torch.equal(b['lq'], b['gt']) for b in batches))

    def test_actual_source_isolation_and_original_blur_crop(self):
        config, parent, training, development, metadata = run.read_contract(run.ROOT / 'configs/foundir_rain_blur_image_moe_v1.json')
        dev = development['rain'] + development['blur_full']
        train = training['rain'] + training['blur']
        self.assertFalse({r['source_group'] for r in train} & {r['source_group'] for r in dev})
        self.assertFalse({r['gt_sha256'] for r in train} & {r['gt_sha256'] for r in dev})
        self.assertEqual(len(development['blur_full']), 2595)
        self.assertEqual(len({r['source_group'] for r in development['blur_preview']}), 100)
        self.assertEqual(config['data_contract'], metadata)
        self.assertEqual(parent['training']['losses']['teacher_l1_weight'], .5)

    def test_real_parameter_counts_meta(self):
        from flux2.model import Flux2, Klein4BParams
        for experts, expected in [(1, 424673280), (2, 853313290)]:
            with torch.device('meta'):
                model = Flux2(Klein4BParams())
                inject_image_moe(model, experts=experts)
            self.assertEqual(sum(p.numel() for p in model.parameters() if p.requires_grad), expected)

    def test_warmup_does_not_persist_past_transition(self):
        self.assertEqual(warmup_fraction(0, 250, 500), 1)
        self.assertEqual(warmup_fraction(375, 250, 500), .5)
        self.assertEqual(warmup_fraction(500, 250, 500), 0)
        self.assertEqual(warmup_fraction(2999, 250, 500), 0)

    def test_bad_checkpoint_keys_rejected(self):
        from safetensors.torch import save_file
        model = ToyModel()
        inject_image_moe(model, indices=[0, 1], hidden=8)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'bad.safetensors'
            save_file({'wrong': torch.zeros(1)}, str(path))
            with self.assertRaisesRegex(RuntimeError, 'keys mismatch'):
                load_expert_state(model, path)

    def test_rain_parity_gate_rejects_damage(self):
        import json
        config, *_ = run.read_contract(run.ROOT / 'configs/foundir_rain_blur_image_moe_v1.json')
        reference = json.loads((run.ROOT / config['rain_metrics']).read_text())
        run.check_rain_parity(reference, config)
        bad = copy.deepcopy(reference)
        bad['rows'][0]['output_psnr'] -= .01
        with self.assertRaisesRegex(RuntimeError, 'changed working Rain'):
            run.check_rain_parity(bad, config)

    def test_comparator_rejects_qualification_and_exposes_capacity(self):
        from smoke.compare_foundir_rain_blur_image_moe_v1 import compare
        row = {'record_id': 'r', 'source_group': 'g', 'input_psnr': 20., 'input_lpips': .3,
               'output_psnr': 30., 'output_lpips': .1}
        entry = {'rows': [row], 'psnr': 30., 'lpips': .1, 'mean_expert_weights': {}}
        common = {'status': 'complete', 'mode': 'formal', 'steps': 3000, 'engineering_pass': True,
                  'initial_rain_parity_pass': True, 'protected_pixels_accessed': False,
                  'config_sha256': 'x', 'source_sha256': {}, 'data': {},
                  'history': [{'rain': entry, 'blur_full': entry}], 'both_task_gates_pass': False,
                  'allocated_gpu_hours': 1}
        moe = dict(common, arm='moe', trainable_parameters=20)
        dense = dict(common, arm='dense', trainable_parameters=10)
        self.assertFalse(compare(moe, dense)['capacity_matched'])
        with self.assertRaises(RuntimeError):
            compare(dict(moe, mode='qualification'), dense)

    def test_full_blur_crop_matches_original_digest(self):
        import hashlib
        _, _, _, development, _ = run.read_contract(run.ROOT / 'configs/foundir_rain_blur_image_moe_v1.json')
        for row in development['blur_full']:
            raw = hashlib.sha256(':'.join(map(str, [2026091111, 'full_development_crop', row['record_id']])).encode()).digest()
            self.assertEqual(int(row['crop_y']), int.from_bytes(raw[:8], 'big') % (int(row['height']) - 512 + 1))
            self.assertEqual(int(row['crop_x']), int.from_bytes(raw[8:16], 'big') % (int(row['width']) - 512 + 1))

    def test_two_rank_ddp_accumulation_checkpointing_and_warmup(self):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(ddp_worker, args=(str(Path(directory) / 'rendezvous'),), nprocs=2, join=True)


if __name__ == '__main__':
    unittest.main()
