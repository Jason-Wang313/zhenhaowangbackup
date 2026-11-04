import copy
from collections import Counter
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from smoke import repair_foundir_vlm_preserve_router_v3 as r
from tests.test_frozen_image_router_v2 import candidates, small_model


class PreserveRouterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_safe_changed_scores_are_not_forced_to_copy_image(self):
        image = torch.tensor([[3., 0., 0., 0., 0.]], requires_grad=True)
        final = torch.tensor([[.5, .2, -.1, .1, -.2]], requires_grad=True)
        loss, ce, penalty, mask = r.objective(final, image, torch.tensor([0]))
        self.assertEqual(float(penalty), 0.)
        self.assertTrue(bool(mask[0]))
        loss.backward()
        self.assertIsNone(image.grad)
        self.assertTrue(torch.isfinite(final.grad).all())

    def test_harmful_override_has_gradient_toward_correct_expert(self):
        image = torch.tensor([[3., 0., 0., 0., 0.]])
        final = torch.tensor([[0., 1., -1., -1., -1.]], requires_grad=True)
        _, _, penalty, _ = r.objective(final, image, torch.tensor([0]))
        self.assertAlmostEqual(float(penalty), 1.2, places=6)
        penalty.backward()
        self.assertLess(float(final.grad[0, 0]), 0.)
        self.assertGreater(float(final.grad[0, 1]), 0.)

    def test_wrong_image_route_is_not_protected(self):
        image = torch.tensor([[0., 3., 0., 0., 0.]])
        final = torch.tensor([[0., 1., -1., -1., -1.]], requires_grad=True)
        loss, ce, penalty, mask = r.objective(final, image, torch.tensor([0]))
        self.assertFalse(bool(mask[0]))
        self.assertEqual(float(penalty), 0.)
        self.assertEqual(float(loss), float(ce))
        loss.backward()
        self.assertLess(float(final.grad[0, 0]), 0.)

    def test_low_image_margin_not_artificially_inflated_and_batch_normalization(self):
        im = torch.tensor([[.05, 0., -1., -1., -1.], [0., 3., -1., -1., -1.]])
        z = torch.tensor([[0., .1, -1., -1., -1.], [0., 1., -1., -1., -1.]], requires_grad=True)
        _, _, penalty, _ = r.objective(z, im, torch.tensor([0, 0]))
        self.assertAlmostEqual(float(penalty), .075, places=6)

    def test_ties_and_zero_grad_cases_are_finite(self):
        z = torch.zeros(5, 5, requires_grad=True)
        loss, _, penalty, _ = r.objective(z, torch.zeros_like(z), torch.arange(5))
        self.assertEqual(float(penalty), 0.)
        loss.backward()
        self.assertTrue(torch.isfinite(z.grad).all())

    def test_frozen_image_unchanged_vlm_updates_and_exact_resume(self):
        m, opt = small_model()
        before = copy.deepcopy(m.image.state_dict())
        x, v, y = torch.randn(32, 6), torch.randn(32, 4), torch.arange(32) % 5
        initial = copy.deepcopy(m.vlm.state_dict())
        with tempfile.TemporaryDirectory() as temp:
            r.state.save(temp, m, opt, 0, [], 'id')
            history = [r.update(m, opt, x, v, y)]
            r.state.save(temp, m, opt, 1, history, 'id')
            value = r.state.load(temp, 'id')
            random_reference = torch.rand(3)
            r.update(m, opt, x, v, y)
            other, oo = small_model()
            r.state.restore(other, oo, value)
            self.assertTrue(torch.equal(random_reference, torch.rand(3)))
            r.update(other, oo, x, v, y)
            for k, tensor in m.state_dict().items():
                self.assertTrue(torch.equal(tensor, other.state_dict()[k]))
            for k, entry in opt.state_dict()['state'].items():
                for name, tensor in entry.items():
                    self.assertTrue(torch.equal(tensor, oo.state_dict()['state'][k][name]))
        self.assertTrue(all(torch.equal(before[k], t) for k, t in m.image.state_dict().items()))
        self.assertTrue(any(not torch.equal(initial[k], t) for k, t in m.vlm.state_dict().items()))

    def test_forbidden_roles_rejected_before_io(self):
        for role in ('holdout', 'future_confirmation', 'official', 'validation'):
            with self.assertRaisesRegex(ValueError, 'must not be loaded'):
                r.load_role({}, role)

    def test_reserved_sources_exclude_all_old_roles_and_all_nontrain_hashes(self):
        rows, old = candidates()
        selected = r.parent.select(rows, old, {'fit': 4, 'calibration': 3, 'holdout': 3})
        rows[20]['gt_sha256'] = old[-1]['row']['gt_sha256']
        reserved = r.reserve_confirmation(rows, selected, old, 5)
        self.assertTrue(reserved['ready'])
        self.assertEqual(reserved, r.reserve_confirmation(list(reversed(rows)), selected, old, 5))
        for field in ('source_group', 'gt_sha256', 'lq_sha256'):
            ids = [i['row'][field] for i in reserved['items']]
            excluded = {i['row'][field] for i in selected + old} | {a[field] for a in rows if a['split'] != 'train'}
            self.assertEqual(len(ids), len(set(ids)))
            self.assertFalse(set(ids) & excluded)
        self.assertFalse(reserved['pixels_or_features_read'])
        self.assertFalse(reserved['automatic_extraction'])

    def test_insufficient_reserve_cannot_silently_reuse_previous_images(self):
        rows, old = candidates()
        reserve = r.reserve_confirmation(rows[:2], old, old, 20)
        self.assertFalse(reserve['ready'])
        self.assertEqual(reserve['requested_per_class'], 20)

    @staticmethod
    def summary(predictions):
        # All examples have true class blur; compact fixture for selection only.
        rows = [{'key': str(j), 'task_for_reporting_only': 'blur', 'predicted_anchor': t,
                 'seed': j, 'crop_xy': [0, 0], 'image_logits': [2, 1, 0, 0, 0]}
                for j, t in enumerate(predictions)]
        return {'rows': rows, 'correct': predictions.count('blur'), 'ce': .3,
                'per_task': {t: {'count': len(rows) if t == 'blur' else 0,
                                'correct': predictions.count('blur') if t == 'blur' else 0} for t in r.ORDER}}

    def test_selection_rejects_disabling_vlm_even_with_zero_harms(self):
        base = self.summary(['blur', 'blur', 'noise', 'noise', 'noise'])
        old = self.summary(['noise', 'blur', 'blur', 'blur', 'noise'])
        safe = self.summary(['blur', 'blur', 'blur', 'blur', 'noise'])
        self.assertFalse(r.eligible(base, old, base))
        self.assertTrue(r.eligible(base, old, safe))
        milestones = {400: base, 800: safe, 1200: safe, 1600: old}
        selected = r.choose(base, old, milestones)
        self.assertEqual(selected['step'], 800)
        self.assertTrue(selected['calibration_screen_pass'])
        self.assertFalse(selected['deploy_or_restoration_authorized'])

    def test_no_eligible_candidate_is_diagnostic_only(self):
        base = self.summary(['blur', 'blur', 'noise'])
        old = self.summary(['noise', 'blur', 'blur'])
        selected = r.choose(base, old, {k: old for k in r.CHECKS})
        self.assertFalse(selected['calibration_screen_pass'])
        self.assertEqual(selected['step'], 400)

    def test_transition_population_or_reference_change_rejected(self):
        base = self.summary(['blur', 'noise'])
        bad = copy.deepcopy(base)
        bad['rows'][0]['image_logits'][0] = 3
        with self.assertRaisesRegex(ValueError, 'reference changed'):
            r.transitions(base, bad)

    def test_full_pipeline_1600_steps_cpu_only_no_holdout_and_verified_repeat(self):
        # Synthetic tiny-dimensional features: engineering test, NOT real metrics.
        datasets, items = {}, []
        gen = torch.Generator().manual_seed(52)
        for role, n in [('fit', 512), ('calibration', 80)]:
            group = [{'key': f'{role}_{t}_{j}', 'task': t, 'role': role, 'seed': j,
                      'row': {'crop_xy': [0, 0]}} for t in r.ORDER for j in range(n)]
            datasets[role] = (group, torch.randn(len(group), 6, generator=gen), torch.randn(len(group), 4, generator=gen),
                              torch.tensor([r.ORDER.index(i['task']) for i in group]))
            items.extend(group)
        m, _ = small_model()
        base = r.parent.evaluate(m, datasets['calibration'], False)
        old = r.parent.evaluate(m, datasets['calibration'], True)
        b = {'items': items, 'future_confirmation': {'items': [], 'ready': False}}
        loaded = []
        with tempfile.TemporaryDirectory() as temp, patch.object(r, 'ROOT', Path(temp)), \
             patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': ''}):
            out = r.ROOT / r.OUTPUT
            r.c.write_new(out / 'receipt.json', {'fixture': True})
            r.c.write_new(r.ROOT / r.parent.REPORT, {'calibration': {'image_only': base, 'corrected': old}})
            def load(_b, role):
                self.assertIn(role, ('fit', 'calibration'))
                loaded.append(role)
                return datasets[role]
            with patch.object(r, 'receipt', return_value=b), patch.object(r, 'load_role', side_effect=load), \
                 patch.object(r.parent, 'model', side_effect=lambda b: small_model()), \
                 patch('torch.set_num_interop_threads'), patch('torch.set_num_threads'), \
                 patch('shutil.disk_usage', return_value=type('Disk', (), {'free': 100 * 1024**3})()):
                r.train()
                r.verify()
                result = r.c.read(r.ROOT / r.REPORT)
                self.assertEqual(loaded, ['fit', 'calibration'])
                self.assertFalse(result['fresh_confirmation_evaluated'])
                self.assertFalse(result['old_holdout_features_accessed'])
                self.assertEqual(len(list((out / 'steps').glob('step_*.json'))), 1601)
                r.train()
                self.assertEqual(loaded, ['fit', 'calibration'])
                # Tampering with summaries is caught independently of the tensor.
                result['transitions']['corrected']['harm'] += 1
                with patch.object(r.c, 'read', wraps=r.c.read) as read:
                    real = read._mock_wraps
                    read.side_effect = lambda p: result if Path(p) == r.ROOT / r.REPORT else real(p)
                    with self.assertRaisesRegex(ValueError, 'accounting mismatch'):
                        r.verify()


if __name__ == '__main__':
    unittest.main()
