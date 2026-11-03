import copy
import inspect
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import torch
from smoke import train_foundir_rain_onpolicy_continue_v1 as impl
from smoke import rain_onpolicy_continue_state_v1 as state
from smoke import rain_onpolicy_midpoint_state_v1 as prior_state
from test_rain_onpolicy_midpoint_v1 import Toy, optimizer, small_update


class ContinueTests(unittest.TestCase):
    def test_recipe_unchanged_and_budget_bounded(self):
        cfg = impl.config()
        self.assertEqual(cfg['start_step'], 600)
        self.assertEqual(cfg['end_step'], 800)
        self.assertEqual(cfg['train_probe_steps'], [700, 800])
        self.assertEqual(cfg['development_steps'], [700, 800])
        self.assertIs(impl.logic, impl.previous.logic)
        original = impl.c.read
        for key, value in [('end_step', 2000), ('learning_rate', 2e-5), ('global_batch', 16),
                           ('loss_weights', {'gt_rgb_mse': 1., 'gt_lpips': .05})]:
            def read(path):
                return {**cfg, key: value} if Path(path) == impl.ROOT / impl.CONFIG else original(path)
            with mock.patch.object(impl.c, 'read', side_effect=read), self.assertRaises(ValueError):
                impl.config()

    def test_next_sample_no_repeat_and_full_evaluation_shards(self):
        slots = [impl.training_position(s, m, r) for s in range(600, 800) for m in range(4) for r in range(8)]
        self.assertEqual(slots, list(range(19200, 25600)))
        for args in [(599, 0, 0), (800, 0, 0), (600, 4, 0), (600, 0, 8)]:
            with self.assertRaises(ValueError):
                impl.training_position(*args)
        for step in (600, 700, 800):
            values = sorted(i for r in range(8) for i in impl.probe_indices(step, r))
            self.assertEqual(values, [*range(12800, 12832), *range(64032, 64132)])

    def test_training_guard_rejects_probes_and_old_slots(self):
        with mock.patch.object(impl.previous, 'cached') as reader:
            for i in (12800, 19199, 25600, 64000, 64032):
                with self.assertRaises(ValueError):
                    impl.cached({}, {}, None, i, 'cpu', training=True)
            reader.assert_not_called()
            impl.cached({'cache_source': 'checkpoints/source'}, {}, None, 19200, 'cpu', training=True)
            self.assertEqual(reader.call_args.args[3], 19200)

    def test_full_parent600_to601_exact_resume(self):
        torch.manual_seed(17)
        model = Toy()
        opt = optimizer(model)
        for _ in range(600):
            small_update(model, opt)
        with tempfile.TemporaryDirectory() as old, tempfile.TemporaryDirectory() as new:
            record = prior_state.save_step(old, model, opt, step=600, world=1,
                rank_rng=[state.base.rng_state('cpu')], receipt_hash='parent', history=[{}] * 600)
            saved = torch.load(Path(old) / record['file'], map_location='cpu', weights_only=True)
            state.validate_state(saved, record, 'parent', 1)
            state.save_step(new, model, opt, step=600, world=1, rank_rng=saved['rank_rng'],
                            receipt_hash='child', history=saved['history'])
            resumed, _ = state.load_latest(new, 'child', 1)
            self.assertEqual(resumed['next_global_slot'], 19200)
            small_update(model, opt)
            expected = state.base.trainable_state(model)
            expected_opt = state.base.cpu_tree(opt.state_dict())
            torch.manual_seed(17)
            restored = Toy()
            restored_opt = optimizer(restored)
            state.base.restore(restored, restored_opt, resumed, 0, 'cpu')
            small_update(restored, restored_opt)
            self.assertTrue(all(torch.equal(v, expected[k]) for k, v in state.base.trainable_state(restored).items()))
            for i, entries in expected_opt['state'].items():
                self.assertTrue(all(torch.equal(v, restored_opt.state_dict()['state'][i][k]) for k, v in entries.items()))
            state.save_step(new, restored, restored_opt, step=601, world=1,
                            rank_rng=[state.base.rng_state('cpu')], receipt_hash='child', history=[{}] * 601)
            self.assertEqual(state.load_latest(new, 'child', 1)[0]['next_global_slot'], 19232)
            with self.assertRaises(ValueError):
                state.load_latest(new, 'child', 2)
            with self.assertRaises(ValueError):
                state.load_latest(new, 'parent', 1)
            (Path(new) / 'steps/step_000600.json').unlink()
            with self.assertRaisesRegex(ValueError, 'gap'):
                state.load_latest(new, 'child', 1)

    def test_no_fresh_or_out_of_budget_state(self):
        model = Toy()
        opt = optimizer(model)
        with tempfile.TemporaryDirectory() as out:
            for step in (0, 400, 599, 801):
                with self.assertRaises(ValueError):
                    state.save_step(out, model, opt, step=step, world=1, rank_rng=[{}], receipt_hash='x', history=[{}] * step)

    def test_temporary_baseline_restores_model_on_success_and_error(self):
        model = Toy()
        opt = optimizer(model)
        small_update(model, opt)
        current = state.base.trainable_state(model)
        before_opt = state.base.cpu_tree(opt.state_dict())
        other = {k: v + .1 for k, v in current.items()}
        for fail in (False, True):
            try:
                with impl.temporary_weights(model, other):
                    self.assertTrue(all(torch.equal(v, other[k]) for k, v in state.base.trainable_state(model).items()))
                    if fail:
                        raise RuntimeError('test interruption')
            except RuntimeError:
                self.assertTrue(fail)
            self.assertTrue(all(torch.equal(v, current[k]) for k, v in state.base.trainable_state(model).items()))
            for i, entries in before_opt['state'].items():
                self.assertTrue(all(torch.equal(v, opt.state_dict()['state'][i][k]) for k, v in entries.items()))

    def test_generation_loss_backprop_unchanged(self):
        old = inspect.getsource(impl.previous.train)
        new = inspect.getsource(impl.train)
        a = old[old.index('            inputs = logic.generated_inputs'):old.index('        log = ')]
        b = new[new.index('            inputs = logic.generated_inputs'):new.index('        log = ')]
        self.assertEqual(a, b)
        self.assertNotIn('prepare_cache(', new)
        self.assertIn('state_io.base.restore(model, optimizer, saved, rank, device)', new)
        self.assertIn('step_000600.json', inspect.getsource(impl.load_parent_state))

    def test_report_and_launcher_contract(self):
        cfg = impl.config()
        self.assertEqual(impl.c.sha(impl.ROOT / cfg['parent_report']), cfg['parent_report_sha256'])
        report = impl.c.read(impl.ROOT / cfg['parent_report'])
        self.assertEqual(report['latest_checkpoint']['sha256'], cfg['parent_checkpoint_sha256'])
        self.assertEqual(report['receipt_sha256'], cfg['parent_receipt_sha256'])
        self.assertEqual(report['result_sha256'], cfg['parent_result_sha256'])
        src = (impl.ROOT / 'smoke/run_foundir_rain_onpolicy_continue_v1_gpu.sh').read_text()
        for value in ['WALL_CAP=10800', '--nnodes=4 --nproc_per_node=2', 'CUDA_VISIBLE_DEVICES=0,1',
                      'zhenhaowang_existing_flux_replay.lock', 'publish --event verified', '29698']:
            self.assertIn(value, src)
        self.assertIn("c.write_new(output / 'result.json', result)", inspect.getsource(impl.summarize))
        self.assertIn("saved['history'][600:]", inspect.getsource(impl.summarize))


if __name__ == '__main__':
    unittest.main()
