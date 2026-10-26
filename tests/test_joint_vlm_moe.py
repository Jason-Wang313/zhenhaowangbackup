from __future__ import annotations

import copy
import csv
import json
import tempfile
import unittest
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from lowlight_restoration.lora import inject_lora
from lowlight_restoration.joint_vlm_moe import (
    predicted_levels, join_conditions, install_joint_moe, joint_conditioning,
    enable_block_checkpointing, JointDataset, restoration_loss,
)
from lowlight_restoration.joint_moe_state import save_state, load_state, rng_state
from lowlight_restoration.rain_blur_image_moe import image_moe_modules, router_classification_loss
from lowlight_restoration.rain_blur_moe_data import RainBlurDataset
from tests.test_foundir_rain_precision_full import FullRainTests
from smoke import train_foundir_rain_blur_joint_vlm_moe_v2_ddp as run


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_size = 4
        self.attention = nn.Linear(4, 4, bias=False)
        self.img_mlp = nn.Sequential(nn.Linear(4, 8, bias=False), nn.SiLU(), nn.Linear(8, 4, bias=False))

    def forward(self, x):
        x = x + self.attention(x)
        return x + self.img_mlp(x)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.double_blocks = nn.ModuleList([Block(), Block()])
        self.single_blocks = nn.ModuleList([])

    def forward(self, x):
        for block in self.double_blocks:
            x = block(x)
        return x


def build(arm='vlm_assisted', checkpointed=False):
    torch.manual_seed(73)
    model = Model()
    inject_lora(model, ['double_blocks.*.attention'], rank=2, alpha=2)
    # Model a nonzero pretrained shared adapter, as in the real Rain parent.
    for name, p in model.named_parameters():
        if '.lora_B.' in name:
            nn.init.normal_(p, std=.01)
    install_joint_moe(model, arm=arm, indices=[0, 1], hidden=8)
    if checkpointed:
        enable_block_checkpointing(model)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    return model, optimizer


def step(model, optimizer):
    model.train(); optimizer.zero_grad(set_to_none=True)
    labels = torch.tensor([0, 1])
    condition = torch.tensor([[0., 0, 1, 0, 0], [1., 0, 0, 0, 0]])
    with joint_conditioning(model, condition, labels=labels, fraction=.5):
        out = model(torch.randn(2, 5, 4))
        loss = out.square().mean() + router_classification_loss(model, labels)
        loss.backward()
    assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.requires_grad)
    optimizer.step()


def ddp_worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method='file://' + rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(seconds=120))
    try:
        model, optimizer = build(checkpointed=True)
        wrapped = torch.nn.parallel.DistributedDataParallel(model, gradient_as_bucket_view=True)
        for _ in range(3):
            model.train(); optimizer.zero_grad(set_to_none=True)
            for micro in range(2):
                labels = torch.tensor([rank])
                condition = torch.tensor([[float(rank), 0., float(1-rank), 0., 0.]])
                with wrapped.no_sync() if micro == 0 else nullcontext():
                    with joint_conditioning(model, condition, labels=labels, fraction=.5):
                        out = wrapped(torch.randn(1, 5, 4))
                        ((out.square().mean() + router_classification_loss(model, labels)) / 2).backward()
            assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.requires_grad)
            optimizer.step()
        vector = torch.cat([p.detach().flatten() for p in model.parameters() if p.requires_grad])
        peers = [torch.empty_like(vector) for _ in range(2)]
        dist.all_gather(peers, vector)
        assert torch.equal(peers[0], peers[1])
    finally:
        dist.destroy_process_group()


class JointTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_only_predicted_values_not_oracle(self):
        row = {'qwen_levels_json': '{"blur":2,"noise":1,"rain":4,"haze":1,"low_light":5}',
               'type_oracle_levels_json': '{"blur":5}', 'category': '01Blur'}
        self.assertEqual(predicted_levels(row), [.25, 0, .75, 0, 1])
        del row['qwen_levels_json']
        with self.assertRaises(KeyError): predicted_levels(row)
        for invalid in ('{}', '{"blur":0,"noise":1,"rain":1,"haze":1,"low_light":1}'):
            with self.assertRaises(ValueError): predicted_levels({'qwen_levels_json': invalid})

    def test_join_refuses_protected_mismatch_missing_and_duplicates(self):
        row = {'record_id': 'a', 'split': 'train', 'source_group': 's', 'lq_path': 'l', 'lq_sha256': 'lh',
               'gt_path': 'g', 'gt_sha256': 'gh', 'qwen_levels_json': json.dumps(dict(blur=3, noise=1, rain=1, haze=1, low_light=1))}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'pred.csv'
            def write(rows):
                with path.open('w', newline='') as f:
                    w = csv.DictWriter(f, fieldnames=list(row)); w.writeheader(); w.writerows(rows)
            write([row]); self.assertEqual(join_conditions(path, [row]), {'a': [.5, 0, 0, 0, 0]})
            for bad in ([dict(row, split='internal_test')], [dict(row, lq_sha256='wrong')], [], [row, row]):
                write(bad)
                with self.assertRaises(ValueError): join_conditions(path, [row])

    def test_same_initialization_and_zero_vlm_adjustment(self):
        a, _ = build('image_only'); b, _ = build('vlm_assisted')
        for k, v in a.state_dict().items(): self.assertTrue(torch.equal(v, b.state_dict()[k]))
        a.eval(); b.eval(); x = torch.randn(2, 6, 4)
        with joint_conditioning(a, None), joint_conditioning(b, torch.rand(2, 5)):
            self.assertTrue(torch.equal(a(x), b(x)))
            self.assertTrue(all(m.last_adjustment.count_nonzero() == 0 for m in image_moe_modules(b).values()))

    def test_shared_attention_is_trainable_base_is_not(self):
        model, _ = build()
        for name, p in model.named_parameters():
            if '.attention.base.' in name: self.assertFalse(p.requires_grad)
            elif '.lora_' in name: self.assertTrue(p.requires_grad)
        model, _ = build('image_only')
        self.assertTrue(all(not p.requires_grad for n, p in model.named_parameters() if '.vlm_projector.' in n))

    def test_all_branches_update_and_condition_cleanup(self):
        for arm in ('image_only', 'vlm_assisted'):
            model, optimizer = build(arm, checkpointed=True)
            initial = {n: p.detach().clone() for n, p in model.named_parameters()}
            for _ in range(4): step(model, optimizer)
            for part in ('experts.0', 'experts.1', '.router.', '.lora_A.', '.lora_B.'):
                self.assertTrue(any(not torch.equal(p, initial[n]) for n, p in model.named_parameters() if part in n))
            changed = any(not torch.equal(p, initial[n]) for n, p in model.named_parameters() if '.vlm_projector.' in n)
            self.assertEqual(changed, arm == 'vlm_assisted')
            self.assertTrue(all(m.condition is None and m.last_weights is None for m in image_moe_modules(model).values()))

    def test_checkpointed_gradients_equal(self):
        a, ao = build(checkpointed=False); b, bo = build(checkpointed=True)
        torch.manual_seed(12); step(a, ao)
        torch.manual_seed(12); step(b, bo)
        for x, y in zip(a.parameters(), b.parameters()): self.assertTrue(torch.allclose(x, y, atol=1e-7))

    def test_evaluation_never_uses_task_labels_and_requires_predictions(self):
        model, _ = build(); model.eval()
        with self.assertRaises(ValueError):
            with joint_conditioning(model, torch.rand(1, 5), labels=torch.tensor([0])): pass
        with self.assertRaises(ValueError):
            with joint_conditioning(model, None): model(torch.randn(1, 5, 4))
        with joint_conditioning(model, None, disable_vlm=True):
            model(torch.randn(1, 5, 4))
            self.assertTrue(all(m.last_adjustment.count_nonzero() == 0 for m in image_moe_modules(model).values()))

    def test_data_and_resume_match_original_schedule(self):
        with tempfile.TemporaryDirectory() as tmp:
            row = FullRainTests().make_row(tmp)
            training = {'rain': [row], 'blur': [row]}
            base = RainBlurDataset(tmp, training, steps=3, rank=4, seed=7)
            joint = JointDataset(tmp, training, conditions={row['record_id']: [1, 0, 0, 0, 0]}, steps=3, rank=4, seed=7, start_step=1)
            for i in range(len(joint)):
                a, b = base[i+4], joint[i]
                self.assertEqual(a['record_id'], b['record_id']); self.assertEqual(a['position'], b['position'])
                self.assertTrue(torch.equal(a['lq'], b['lq'])); self.assertTrue(torch.equal(a['gt'], b['gt']))

    def test_exact_resume_and_qualification_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, ao = build(checkpointed=True)
            step(a, ao)
            saved = save_state(a, ao, tmp, step=1, mode='formal', arm='vlm_assisted', config_sha='c', sources={'s':'h'}, rank_rng=[rng_state() for _ in range(8)])
            step(a, ao)
            b, bo = build(checkpointed=True)
            path = Path(tmp) / 'step_000001.resume.json'
            load_state(b, bo, path, arm='vlm_assisted', config_sha='c', sources={'s':'h'}, rank=0, total_steps=3)
            step(b, bo)
            for x, y in zip(a.parameters(), b.parameters()): self.assertTrue(torch.equal(x, y))
            for x, y in zip(ao.state.values(), bo.state.values()):
                self.assertTrue(torch.equal(x['exp_avg'], y['exp_avg']))
            for arm, config_sha in [('image_only', 'c'), ('vlm_assisted', 'wrong')]:
                with self.assertRaises(RuntimeError):
                    load_state(b, bo, path, arm=arm, config_sha=config_sha, sources={'s':'h'}, rank=0, total_steps=3)
            saved['mode'] = 'qualification'; path.write_text(json.dumps(saved))
            with self.assertRaises(RuntimeError):
                load_state(b, bo, path, arm='vlm_assisted', config_sha='c', sources={'s':'h'}, rank=0, total_steps=3)

    def test_task_neutral_loss_has_no_identity_or_teacher(self):
        output = torch.randn(1, 3, 8, 8, requires_grad=True)
        result = {'decoded': output, 'degraded': torch.zeros_like(output), 'prediction': output, 'predicted_clean': output}
        metric = lambda a, b: (a-b).abs().mean()
        weights = dict(flow_mse=.1, latent_l1=.25, rgb_mse=10, charbonnier=1, highpass=.1, lpips=.05)
        loss, terms = restoration_loss(result, torch.zeros_like(output), torch.ones_like(output), metric, lambda x: (x, None), weights)
        self.assertEqual(set(terms), set(weights)); loss.backward()
        self.assertTrue(bool(torch.isfinite(output.grad).all()))

    def test_contract_hash_and_both_qualification_requirement(self):
        path = run.ROOT / 'configs/foundir_rain_blur_joint_vlm_moe_v2.json'
        self.assertEqual(run.old.sha256(path), run.CONFIG_SHA256)
        with self.assertRaises(RuntimeError): run.require_qualifications(None, {}, {})
        with self.assertRaises(RuntimeError): run.require_qualifications([], {}, {})

    def test_full_flux_parameter_budget_without_allocating_weights(self):
        from flux2.model import Flux2, Klein4BParams
        from lowlight_restoration.flux2_endpoint_precision import promote_lora_to_fp32
        config = json.loads((run.ROOT / 'configs/foundir_rain_blur_joint_vlm_moe_v2.json').read_text())
        patterns = ['double_blocks.*.img_attn.qkv', 'double_blocks.*.img_attn.proj',
                    'single_blocks.*.linear1', 'single_blocks.*.linear2']
        for arm in ('image_only', 'vlm_assisted'):
            with torch.device('meta'):
                model = Flux2(Klein4BParams())
                self.assertEqual(len(inject_lora(model, patterns, rank=32, alpha=32)), 50)
                promote_lora_to_fp32(model)
                install_joint_moe(model, arm=arm)
            counts = {g['name']: sum(p.numel() for p in g['params']) for g in run.parameter_groups(model, config)}
            self.assertEqual(counts['shared_lora'], 32440320)
            self.assertEqual(counts['experts'], 849346560)
            self.assertEqual(sum(counts.values()), 885753610 + (10250 if arm == 'vlm_assisted' else 0))

    def test_comparator_pairs_records_and_detects_crop_drift(self):
        from smoke.compare_foundir_rain_blur_joint_vlm_moe_v2 import paired_summary
        a = {'rows': [{'record_id': str(i), 'source_group': str(i//2), 'input_psnr': 20.,
                       'input_lpips': .4, 'output_psnr': 25., 'output_lpips': .2} for i in range(6)]}
        b = copy.deepcopy(a)
        for r in b['rows']: r.update(output_psnr=25.5, output_lpips=.18)
        result = paired_summary(a, b)
        self.assertAlmostEqual(result['VLM_minus_image_PSNR'], .5)
        self.assertAlmostEqual(result['VLM_minus_image_LPIPS'], -.02)
        self.assertEqual(result['source_groups'], 3)
        b['rows'][0]['input_psnr'] += 1
        with self.assertRaises(ValueError): paired_summary(a, b)

    def test_two_rank_ddp_checkpointed_shared_adaptation(self):
        with tempfile.TemporaryDirectory() as tmp:
            mp.spawn(ddp_worker, args=(str(Path(tmp)/'rendezvous'),), nprocs=2, join=True)


if __name__ == '__main__':
    unittest.main()
