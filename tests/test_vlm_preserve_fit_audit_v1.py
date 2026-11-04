import copy
import math
import os
from pathlib import Path
import platform
import tempfile
import unittest
from unittest.mock import patch

import torch

from smoke import audit_foundir_vlm_preserve_fit_v1 as a
from tests.test_frozen_image_router_v2 import small_model


def summaries():
    m, _ = small_model()
    gen = torch.Generator().manual_seed(71)
    items = [{'key': str(j), 'task': a.ORDER[j % 5], 'seed': j, 'row': {'crop_xy': [0, 0]}} for j in range(20)]
    data = (items, torch.randn(20, 6, generator=gen), torch.randn(20, 4, generator=gen), torch.arange(20) % 5)
    return items, a.v3.parent.evaluate(m, data, False)


class FitAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_summary_recomputed_and_tampering_rejected(self):
        items, summary = summaries()
        a.validate_summary(summary, {i['key']: i for i in items})
        bad = copy.deepcopy(summary)
        bad['correct'] += 1
        with self.assertRaisesRegex(ValueError, 'metrics mismatch'):
            a.validate_summary(bad, {i['key']: i for i in items})
        bad = copy.deepcopy(summary)
        bad['rows'][0]['seed'] += 1
        with self.assertRaisesRegex(ValueError, 'identity mismatch'):
            a.validate_summary(bad, {i['key']: i for i in items})

    def test_duplicate_and_nonfinite_rows_rejected(self):
        items, summary = summaries()
        bad = copy.deepcopy(summary)
        bad['rows'][1] = bad['rows'][0]
        with self.assertRaisesRegex(ValueError, 'population mismatch'):
            a.validate_summary(bad, {i['key']: i for i in items})
        bad = copy.deepcopy(summary)
        bad['rows'][0]['logits'][0] = math.nan
        with self.assertRaisesRegex(ValueError, 'invalid five'):
            a.validate_summary(bad, {i['key']: i for i in items})

    def test_loss_matches_training_objective(self):
        _, base = summaries()
        changed = copy.deepcopy(base)
        for r in changed['rows']:
            delta = [float(j) / 3 for j in range(5)]
            r['vlm_delta'] = delta
            r['logits'] = [x + d for x, d in zip(r['image_logits'], delta)]
            r['predicted_anchor'] = a.ORDER[max(range(5), key=lambda k: r['logits'][k])]
        d = a.diagnostics(base, changed)
        z = torch.tensor([r['logits'] for r in changed['rows']])
        im = torch.tensor([r['image_logits'] for r in changed['rows']])
        y = torch.tensor([a.ORDER.index(r['task_for_reporting_only']) for r in changed['rows']])
        _, _, penalty, _ = a.v3.objective(z, im, y)
        self.assertAlmostEqual(d['preserve_penalty_full_population_mean'], float(penalty), places=6)
        self.assertEqual(d['protected_count'], base['correct'])

    def test_harm_rate_uses_image_correct_not_population(self):
        _, base = summaries()
        corrected = copy.deepcopy(base)
        for r in corrected['rows']:
            if r['predicted_anchor'] == r['task_for_reporting_only']:
                truth = a.ORDER.index(r['task_for_reporting_only'])
                z = [0.] * 5
                z[(truth + 1) % 5] = 10.
                r['logits'] = z
                r['vlm_delta'] = [v-b for v,b in zip(z,r['image_logits'])]
                r['predicted_anchor'] = a.ORDER[(truth+1) % 5]
                break
        d = a.diagnostics(base, corrected)
        self.assertEqual(d['transitions']['harm'], 1)
        self.assertAlmostEqual(d['harm_rate_among_image_correct'], 1/base['correct'])

    def test_zero_denominators_are_explicit_not_nan(self):
        _, base = summaries()
        all_wrong = copy.deepcopy(base)
        for r in all_wrong['rows']:
            truth = a.ORDER.index(r['task_for_reporting_only'])
            z = [0.] * 5
            z[(truth+1) % 5] = 1.
            r.update(logits=z, image_logits=z, vlm_delta=[0.]*5, predicted_anchor=a.ORDER[(truth+1)%5])
        all_wrong['correct'] = 0
        d = a.diagnostics(all_wrong, all_wrong)
        self.assertIsNone(d['harm_rate_among_image_correct'])
        self.assertIsNone(d['minimum_protected_final_margin'])
        self.assertEqual(d['preserve_penalty_full_population_mean'], 0.)

    def test_history_reports_tail_only_and_rejects_missing_fields(self):
        row = dict(loss=2., ce=1., preserve_penalty=1., protected_training_rows=20,
                   harmful_training_overrides=2, vlm_grad_norm_before_clip=.5)
        history = [dict(row, loss=50.)] * 700 + [row] * 100
        result = a.history_summary(history, 800)
        self.assertEqual(result['first_step'], 701)
        self.assertEqual(result['mean']['loss'], 2.)
        with self.assertRaisesRegex(ValueError, 'step mismatch'):
            a.history_summary(history, 1600)
        history[-1] = {}
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            a.history_summary(history, 800)

    def test_end_to_end_real_saved_tiny_checkpoints_read_only_and_repeat(self):
        # A tiny synthetic model is trained to create fixture checkpoints, NOT a project experiment.
        gen = torch.Generator().manual_seed(23)
        datasets, items = {}, []
        for role, n in [('fit', 512), ('calibration', 80)]:
            group = [{'key': f'{role}_{t}_{j}', 'task': t, 'role': role, 'seed': j,
                      'row': {'crop_xy': [0, 0]}} for t in a.ORDER for j in range(n)]
            datasets[role] = (group, torch.randn(len(group), 6, generator=gen), torch.randn(len(group), 4, generator=gen),
                              torch.tensor([a.ORDER.index(i['task']) for i in group]))
            items.extend(group)
        m, opt = small_model()
        base = a.v3.parent.evaluate(m, datasets['calibration'], False)
        history = []
        with tempfile.TemporaryDirectory() as temp, patch.object(a, 'ROOT', Path(temp)), \
             patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': ''}):
            parent = a.ROOT/a.v3.OUTPUT
            a.c.write_new(parent/'receipt.json', {'fixture': True})
            for n in a.SOURCES:
                a.c.write_new(a.ROOT/n, {'fixture_source': True})
            for step in range(1600):
                ids = torch.tensor(a.v3.parent.batch_indices(datasets['fit'][0], step))
                fit = datasets['fit']
                history.append(a.v3.update(m, opt, fit[1][ids], fit[2][ids], fit[3][ids]))
                if step+1 in a.STEPS:
                    a.v3.state.save(parent, m, opt, step+1, history, 'fixture_identity')
                    a.c.write_new(parent/f'calibration_{step+1}.json', a.v3.parent.evaluate(m, datasets['calibration'], True))
            markers = {str(k): a.c.read(parent/f'steps/step_{k:06d}.json') for k in a.STEPS}
            runtime = {'torch': str(torch.__version__), 'build': a.c.digest(torch.__config__.show()),
                       'python': platform.python_version(), 'threads': 8, 'interop': 1, 'deterministic': True,
                       'cpu': next(l for l in Path('/proc/cpuinfo').read_text().splitlines() if l.startswith('model name'))}
            old = {'runtime': runtime, 'frozen_image_hash': a.v3.state.io.parameter_hash(dict(m.image.named_parameters())),
                   'calibration': {'image_only': base}, 'selection': {'calibration_sha256':
                   {str(k): a.c.sha(parent/f'calibration_{k}.json') for k in a.STEPS}}}
            a.c.write_new(a.ROOT/a.v3.REPORT, old)
            b = {'items': items}
            protected = {str(p): a.c.sha(p) for p in parent.rglob('*') if p.is_file()}
            loaded = []
            def load(_b, role):
                self.assertEqual(role, 'fit')
                loaded.append(role)
                return datasets[role]
            with patch.object(a, 'context', return_value=(b, old, 'fixture_identity', markers)), \
                 patch.object(a.v3, 'load_role', side_effect=load), \
                 patch.object(a.v3.parent, 'model', side_effect=lambda b: small_model()), \
                 patch('torch.set_num_interop_threads'), patch('torch.set_num_threads'), \
                 patch.object(a.v3, 'update', side_effect=AssertionError('audit cannot train')), \
                 patch.object(a.v3.state, 'save', side_effect=AssertionError('audit cannot save checkpoints')):
                a.audit()
                a.verify()
                result = a.c.read(a.ROOT/a.REPORT)
                self.assertFalse(result['training_performed'])
                self.assertFalse(result['calibration_features_loaded'])
                self.assertEqual(result['checkpoints']['800']['fit']['count'], 2560)
                self.assertEqual(loaded, ['fit'])
                a.audit()
                self.assertEqual(loaded, ['fit'])
                self.assertEqual(protected, {str(p): a.c.sha(p) for p in parent.rglob('*') if p.is_file()})
                broken = copy.deepcopy(result)
                broken['checkpoints']['800']['fit']['correct'] += 1
                original_read = a.c.read
                with patch.object(a.c, 'read', side_effect=lambda p: broken if Path(p) == a.ROOT/a.REPORT else original_read(p)):
                    with self.assertRaisesRegex(ValueError, 'differs'):
                        a.verify()


if __name__ == '__main__':
    unittest.main()
