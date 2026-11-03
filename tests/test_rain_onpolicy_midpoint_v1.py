import ast
import copy
import inspect
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch
from smoke import rain_onpolicy_midpoint_state_v1 as state
from smoke import existing_flux_replay_contract_v1 as c
from smoke import train_foundir_rain_onpolicy_midpoint_v1 as impl
from smoke import rain_midpoint_readout_v1 as readout
from lowlight_restoration import rain_onpolicy_midpoint as logic
from lowlight_restoration.lora import inject_lora


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(3, 3)
        self.requires_grad_(False)
        inject_lora(self, ['fc'], 2, 2., 0.)
        self.calls = 0

    def forward(self, **kw):
        self.calls += 1
        return self.fc(kw['x'].float()).to(kw['x'].dtype) + kw['ctx'][:, 0, 0].reshape(-1, 1, 1)


class AE:
    def decode(self, z):
        return z.tanh()


class Metric:
    def __call__(self, a, b):
        return (a - b).abs().mean()


def sampler(m):
    from flux2.sampling import denoise_cfg, batched_prc_img
    # Real native sampler, BF16 CFG arithmetic, deterministic LQ/reference/noise.
    x, ids = batched_prc_img(torch.linspace(-.2, .2, 12).reshape(1, 3, 2, 2).to(torch.bfloat16))
    with torch.inference_mode():
        return denoise_cfg(m, x, ids, torch.tensor([[[.013]], [[.031]]], dtype=torch.bfloat16),
            torch.zeros(2, 1, 4, dtype=torch.long), timesteps=[1 - i / 50 for i in range(51)],
            guidance=1., img_cond_seq=x * .2, img_cond_seq_ids=ids)


def optimizer(model):
    return torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-5, weight_decay=0., fused=False)


def small_update(model, opt, value=None):
    opt.zero_grad()
    value = torch.randn(2, 3) if value is None else value
    model.fc(value).square().mean().backward()
    opt.step()
    opt.zero_grad()


class OnPolicyTests(unittest.TestCase):
    def test_detached_prefix_matches_existing_readout_and_has_only25_calls(self):
        torch.manual_seed(7)
        model = Toy()
        inputs = logic.detached_prefix(model, sampler)
        self.assertEqual(model.calls, 25)
        for value in inputs.values():
            if isinstance(value, torch.Tensor):
                self.assertFalse(value.requires_grad)
                self.assertFalse(torch.is_inference(value))
        z = logic.predict_endpoint(model, inputs)
        self.assertEqual(model.calls, 26)
        snap, calls = readout.run_prefix(model, sampler)
        self.assertEqual(calls, 26)
        self.assertTrue(torch.equal(z, readout.endpoint_latent(snap, 'cpu')))
        loss, terms = logic.objective(model, AE(), inputs, torch.zeros_like(z), Metric(),
                                      {'gt_rgb_mse': 10., 'gt_lpips': .05})
        loss.backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters()))
        self.assertTrue(all(p.grad is None for p in model.fc.base.parameters()))
        self.assertEqual(set(terms), {'gt_rgb_mse', 'gt_lpips'})

    def test_current_weights_refresh_prefix(self):
        torch.manual_seed(7)
        model = Toy()
        before = logic.detached_prefix(model, sampler)['x']
        with torch.no_grad():
            for name, value in model.named_parameters():
                if 'lora_B' in name:
                    value.add_(.1)
        after = logic.detached_prefix(model, sampler)['x']
        self.assertFalse(torch.equal(before, after))

    def test_sampler_errors_are_not_swallowed(self):
        with self.assertRaisesRegex(RuntimeError, 'original error'):
            logic.detached_prefix(Toy(), lambda m: (_ for _ in ()).throw(RuntimeError('original error')))
        with self.assertRaises(ValueError):
            logic.detached_prefix(Toy(), lambda m: None)

    def test_no_GT_teacher_or_metric_in_generation(self):
        self.assertEqual(list(inspect.signature(logic.generated_inputs).parameters),
                         ['model', 'ae', 'lq', 'prompt', 'seed'])
        self.assertEqual(list(inspect.signature(impl.inference).parameters),
                         ['model', 'ae', 'lq', 'prompt', 'seed'])
        self.assertIn('steps=50', inspect.getsource(logic.generated_inputs))
        for text in (inspect.getsource(logic), inspect.getsource(impl.inference)):
            self.assertNotIn("['teacher_rgb']", text)
            self.assertNotIn("['clean_latent']", text)

    def test_sampling_and_evaluation_are_fixed(self):
        positions = [logic.training_position(s, m, r) for s in range(400, 600) for m in range(4) for r in range(8)]
        self.assertEqual(positions, list(range(12800, 19200)))
        self.assertEqual(logic.training_position(400, 0, 0), 12800)
        for args in ((399, 0, 0), (600, 0, 0), (400, 4, 0), (400, 0, 8)):
            with self.assertRaises(ValueError):
                logic.training_position(*args)
        for step in (400, 450, 500, 600):
            indices = sorted(i for rank in range(8) for i in logic.probe_indices(step, rank))
            expected = list(range(12800, 12832)) + (list(range(64032, 64132)) if step == 600 else [])
            self.assertEqual(indices, expected)

    def test_training_guard_precedes_cache_read(self):
        with mock.patch.object(impl.parent, 'cached') as get:
            for index in (0, 64000, 64032, 19200, 12799):
                with self.assertRaises(ValueError):
                    impl.cached({}, {}, Path('/not-mounted'), index, 'cpu', training=True)
            get.assert_not_called()

    def test_resume_keeps400_Adam_RNG_and_next_update(self):
        torch.manual_seed(17)
        model = Toy()
        opt = optimizer(model)
        for _ in range(400):
            small_update(model, opt)
        with tempfile.TemporaryDirectory() as tmp:
            state.save_step(tmp, model, opt, step=400, world=1,
                            rank_rng=[state.base.rng_state('cpu')], receipt_hash='new', history=[{}] * 400)
            saved, record = state.load_latest(tmp, 'new', 1)
            self.assertEqual(saved['next_global_slot'], 12800)
            small_update(model, opt)
            expected = state.base.trainable_state(model)
            expected_opt = state.base.cpu_tree(opt.state_dict())
            # Frozen backbone is the same asset on resume, not newly randomized.
            torch.manual_seed(17)
            model2 = Toy()
            opt2 = optimizer(model2)
            state.base.restore(model2, opt2, saved, 0, 'cpu')
            small_update(model2, opt2)
            for key, value in state.base.trainable_state(model2).items():
                self.assertTrue(torch.equal(value, expected[key]))
            for idx, entries in expected_opt['state'].items():
                for key, value in entries.items():
                    self.assertTrue(torch.equal(value, opt2.state_dict()['state'][idx][key]))
            state.save_step(tmp, model2, opt2, step=401, world=1,
                            rank_rng=[state.base.rng_state('cpu')], receipt_hash='new', history=[{}] * 401)
            self.assertEqual(state.load_latest(tmp, 'new', 1)[0]['step'], 401)
            for wrong_hash, world in [('wrong', 1), ('new', 2)]:
                with self.assertRaises(ValueError):
                    state.load_latest(tmp, wrong_hash, world)
            (Path(tmp) / 'steps/step_000400.json').unlink()
            with self.assertRaisesRegex(ValueError, 'gap'):
                state.load_latest(tmp, 'new', 1)

    def test_corrupt_checkpoint_rejected_not_silent_rollback(self):
        model = Toy()
        opt = optimizer(model)
        for _ in range(400):
            small_update(model, opt)
        with tempfile.TemporaryDirectory() as tmp:
            record = state.save_step(tmp, model, opt, step=400, world=1,
                rank_rng=[state.base.rng_state('cpu')], receipt_hash='r', history=[{}] * 400)
            with (Path(tmp) / record['file']).open('ab') as f:
                f.write(b'corruption')
            with self.assertRaisesRegex(ValueError, 'hash'):
                state.load_latest(tmp, 'r', 1)

    def test_scope_and_launch(self):
        cfg = impl.config()
        self.assertEqual(cfg['new_updates'], 200)
        self.assertEqual(cfg['loss_weights'], {'gt_rgb_mse': 10., 'gt_lpips': .05})
        for key, value in [('new_updates', 2000), ('prefix_calls_without_grad', 24), ('development_steps', [450, 600])]:
            with mock.patch.object(impl.c, 'read', return_value={**cfg, key: value}):
                with self.assertRaises(ValueError):
                    impl.config()
        src = (impl.ROOT / 'smoke/run_foundir_rain_onpolicy_midpoint_v1_gpu.sh').read_text()
        for value in ['WALL_CAP=10800', '--nnodes=4 --nproc_per_node=2', 'CUDA_VISIBLE_DEVICES=0,1',
                      'zhenhaowang_existing_flux_replay.lock', 'publish --event verified', '29696']:
            self.assertIn(value, src)
        self.assertIn("c.write_new(output / 'result.json', result)", inspect.getsource(impl.summarize))
        self.assertNotIn('prepare_cache(', inspect.getsource(impl))

    def test_joint_gate_and_regressions(self):
        rows = [{'index': i, 'metrics': {'psnr': 34.3, 'lpips': .079}} for i in range(100)]
        before = {i: {'psnr': 33.129, 'lpips': .075364} for i in range(100)}
        out = impl.paired_summary(rows, before)
        self.assertTrue(out['joint_mean_target_pass'])
        self.assertEqual(out['lpips_worse_count'], 100)
        self.assertEqual(out['both_nonworse_count'], 0)
        rows[0]['metrics']['psnr'] = float('nan')
        with self.assertRaises(ValueError):
            impl.paired_summary(rows, before)
        with self.assertRaises(ValueError):
            impl.paired_summary(rows[:-1], before)

    def test_parent400_only_not_latest2000(self):
        src = inspect.getsource(impl.load_parent_state)
        self.assertIn('step_000400.json', src)
        self.assertNotIn('load_latest', src)
        self.assertIn('state_io.base.restore(model, optimizer, saved, rank, device)', inspect.getsource(impl.train))


if __name__ == '__main__':
    unittest.main()
