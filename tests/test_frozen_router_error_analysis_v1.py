import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from smoke import analyze_foundir_frozen_router_errors_v1 as a


def fixture():
    items = []; records = {arm: {} for arm in a.ARMS}
    hashes = {400: 'router400', 408: 'router408'}
    prompts = {t: 'prompt '+t for t in a.ORDER}
    for task in a.ORDER:
        for index in range(8):
            key = task+'_'+str(index)
            item = {'key': key, 'task': task, 'split': 'validation', 'seed': index,
                'row': {'record_id': key, 'source_group': 'source_'+key, 'split': 'validation', 'crop_xy': [0, 0]}}
            items.append(item)
            for arm in a.ARMS:
                control = arm == 'known_class_control'; stage = 0 if control else int(arm[-3:])
                wrong = (not control and (task in ('noise', 'lowlight') and index < 2
                    or stage == 400 and (task == 'blur' and index == 0 or task == 'rain' and index < 2)))
                anchor = a.ORDER[(a.ORDER.index(task)+1)%5] if wrong else task
                weights = [float(t == anchor) if control else (.96 if t == anchor else .01) for t in a.ORDER]
                ident = {'receipt_sha256': 'receipt', 'item': item, 'arm': 'evaluation/'+arm}
                if not control: ident['router_hash'] = hashes[stage]
                metrics = {'psnr': 30. if control else (26. if wrong else 29.),
                           'lpips': .1 if control else (.14 if wrong else .11)}
                if task == 'blur' and index == 1 and not control: metrics['psnr'] = 31.
                records[arm][key] = {'identity': ident, 'metrics': metrics, 'known_class_control_not_deployed': control,
                    'route': {'anchor': anchor, 'weights': [weights], 'prompt': prompts[anchor],
                        'alpha': (.75, .30, 1., .925, 1.)[a.ORDER.index(anchor)],
                        'second_quantization': anchor not in ('haze', 'rain'), 'schedule_steps': 50,
                        'model_calls': 26 if anchor == 'rain' else 50,
                        'readout': 'endpoint_at_call25' if anchor == 'rain' else 'final_flow',
                        'rain_after_call25': 'zero_and_renormalize_nonrain'}}
    result = {}
    for step, field in [(400, 'before_quality'), (408, 'after_quality')]:
        groups = {}
        for task in a.ORDER:
            group = {'count': 8, 'means': {}, 'router_anchor_correct': {}, 'regressions_vs_known_class': {}}
            selected = [i for i in items if i['task'] == task]
            for arm in ('known_class_control', 'router_step'+str(step)):
                rows = [records[arm][i['key']] for i in selected]
                group['means'][arm] = {m: a.mean(r['metrics'][m] for r in rows) for m in a.METRICS}
                group['router_anchor_correct'][arm] = sum(r['route']['anchor'] == task for r in rows)
                if arm != 'known_class_control':
                    group['regressions_vs_known_class'][arm] = {m: sum(
                        r['metrics'][m] < 30. if m == 'psnr' else r['metrics'][m] > .1 for r in rows) for m in a.METRICS}
            groups[task] = group
        result[field] = {'groups': groups}
    return items, records, result, 'receipt', hashes, prompts


class ErrorAnalysisTests(unittest.TestCase):
    def test_partition_sums_and_causal_limit(self):
        report = a.analyze(*fixture()); groups = report['stage408']
        correct = groups['correct_anchor_soft_vs_same_anchor_onehot']; wrong = groups['wrong_anchor_choice_and_mixing_combined']
        self.assertEqual(correct['count'], 36); self.assertEqual(wrong['count'], 4)
        self.assertEqual(len(report['per_image']), 40)
        for metric in ('psnr_loss_db', 'lpips_increase'):
            self.assertAlmostEqual(correct['deficit'][metric]['contribution_to_full_population_mean'] +
                wrong['deficit'][metric]['contribution_to_full_population_mean'],
                groups['all']['deficit'][metric]['conditional_mean'])
        self.assertGreater(correct['deficit']['psnr_loss_db']['positive_harm_sum'],
            correct['deficit']['psnr_loss_db']['conditional_mean']*36)
        self.assertFalse(report['GPU_used']); self.assertFalse(report['checkpoint_tensors_read'])
        self.assertFalse(report['missing_counterfactual']['GT_used_to_choose_counterfactual_anchor'])

    def test_transitions_not_inferred_from_counts(self):
        report = a.analyze(*fixture())
        self.assertEqual(report['transitions']['became_correct']['count'], 3)
        self.assertEqual(report['transitions']['stayed_wrong']['count'], 4)
        self.assertEqual(report['transitions']['became_wrong']['count'], 0)
        self.assertEqual(report['transitions']['kept_correct']['count'], 33)

    def test_missing_hard_outputs_only_for_model_chosen_wrong_anchor(self):
        report = a.analyze(*fixture()); pending = report['missing_counterfactual']
        self.assertEqual(pending['count'], 4); self.assertFalse(pending['launched'])
        for item in pending['items']:
            original = next(r for r in report['per_image'] if r['key'] == item['key'])
            self.assertNotEqual(item['predicted_anchor'], original['task'])
            self.assertEqual(sum(item['hard_weights_from_model_argmax']), 1.)
            self.assertEqual(a.ORDER[item['hard_weights_from_model_argmax'].index(1.)], original['step408']['anchor'])

    def test_empty_wrong_group_has_zero_contribution_not_nan(self):
        report = a.analyze(*fixture())
        group = report['per_task']['haze']['408']['wrong_anchor_choice_and_mixing_combined']
        self.assertEqual(group['count'], 0)
        self.assertIsNone(group['deficit']['psnr_loss_db']['conditional_mean'])
        self.assertEqual(group['deficit']['psnr_loss_db']['contribution_to_full_population_mean'], 0)
        json.dumps(report, allow_nan=False)

    def test_reject_missing_extra_and_wrong_item_identity(self):
        for mutation in ('missing', 'extra', 'identity', 'checkpoint', 'control'):
            args = list(fixture()); records = args[1]
            if mutation == 'missing': records['router_step408'].pop('blur_0')
            if mutation == 'extra': records['router_step408']['alien'] = records['router_step408']['blur_0']
            if mutation == 'identity': records['router_step408']['blur_0']['identity'] = {}
            if mutation == 'checkpoint': records['router_step408']['blur_0']['identity']['router_hash'] = 'wrong'
            if mutation == 'control': records['known_class_control']['blur_0']['known_class_control_not_deployed'] = False
            with self.assertRaises(ValueError, msg=mutation): a.analyze(*args)

    def test_reject_wrong_route_protocol_probability_and_nonfinite(self):
        for field, value in [('anchor', 'rain'), ('alpha', .9), ('model_calls', 26), ('weights', [[1., -1., 1., 0., 0.]])]:
            args = list(fixture()); args[1]['router_step408']['blur_0']['route'][field] = value
            with self.assertRaises(ValueError): a.analyze(*args)
        args = list(fixture()); args[1]['router_step408']['blur_0']['metrics']['psnr'] = float('nan')
        with self.assertRaises(ValueError): a.analyze(*args)

    def test_reject_mean_and_count_that_disagree_with_terminal(self):
        for mutation in ('mean', 'correct', 'regressions'):
            args = list(fixture()); group = args[2]['after_quality']['groups']['haze']
            if mutation == 'mean': group['means']['router_step408']['psnr'] += .001
            if mutation == 'correct': group['router_anchor_correct']['router_step408'] = 7
            if mutation == 'regressions': group['regressions_vs_known_class']['router_step408']['lpips'] = 0
            with self.assertRaises(ValueError): a.analyze(*args)

    def test_read_only_inputs_and_immutable_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); output = root/'report.json'
            a.write_report(output, {'result': 1}); old = output.read_bytes()
            a.write_report(output, {'result': 1})
            with self.assertRaises(ValueError): a.write_report(output, {'result': 2})
            self.assertEqual(old, output.read_bytes())
            with self.assertRaises(ValueError): a.read_json(root/'model.pt')
            output.write_text('{"bad": NaN}')
            with self.assertRaises(ValueError): a.read_json(output)

    def test_real_github_aggregate_has_only_supported_conclusions(self):
        terminal, sha = a.read_json(a.ROOT/a.REPORT)
        self.assertEqual(sha, a.REPORT_SHA)
        report = a.aggregate_only(terminal)
        self.assertEqual(report['missing_final_predicted_hard_outputs'], 4)
        self.assertEqual(report['status'], 'aggregate_only_per_image_breakdown_pending')
        self.assertTrue(report['per_task']['haze']['all_correct_group_means_isolate_mixing'])
        self.assertFalse(report['per_task']['noise']['all_correct_group_means_isolate_mixing'])
        self.assertAlmostEqual(report['per_task']['haze']['gap_final_vs_onehot']['psnr_loss_db'], .9583437961)

    def test_complete_metadata_reader_without_any_images_or_tensor_files(self):
        items, records, result, _, hashes, prompts = fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); run = root/'checkpoints/foundir_frozen_expert_router_v1'/a.RUN
            receipt = {'items': items, 'parent': {'experts': {t: {'prompt': prompts[t]} for t in a.ORDER}}}
            def write(path, value):
                path.parent.mkdir(parents=True, exist_ok=True)
                raw = (json.dumps(value, sort_keys=True)+'\n').encode(); path.write_bytes(raw)
                return hashlib.sha256(raw).hexdigest()
            receipt_sha = write(run/'receipt.json', receipt)
            markers = {}
            for step in (400, 408):
                markers[step] = {'step': step, 'parameter_hash': hashes[step], 'receipt_hash': receipt_sha,
                                 'file': 'never_read_checkpoint.pt', 'sha256': 'unused_tensor_digest'}
                write(run/'steps'/f'step_{step:06d}.json', markers[step])
            result.update(step=408, frozen_experts=True, receipt_sha256=receipt_sha, final_checkpoint=markers[408])
            for arm in a.ARMS:
                for key, entry in records[arm].items():
                    entry['identity']['receipt_sha256'] = receipt_sha
                    # Deliberately no tensor or image assets on disk.
                    entry['tensor'] = 'never_read_image.safetensors'
                    write(run/'evaluation'/arm/(key+'.json'), entry)
            report = a.analyze(items, records, result, receipt_sha, hashes, prompts)
            terminal = {'run_id': a.RUN, 'status': 'completed_all_four_nodes_status0',
                'attempts': [{'node': n, 'status': 0} for n in range(4)],
                'result': result, 'result_sha256': write(run/'result.json', result),
                'final_checkpoint': markers[408], 'regressions_step408_vs400': report['regressions_step408_vs400']}
            terminal_sha = write(root/a.REPORT, terminal)
            before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*.json')}
            with mock.patch.object(a, 'REPORT_SHA', terminal_sha):
                checked = a.diagnose(root)
            self.assertEqual(checked['missing_counterfactual']['count'], 4)
            self.assertEqual(len(checked['input_json_sha256']), 123)
            self.assertEqual(before, {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*.json')})

    def test_no_heavy_dependencies_or_pixel_readers(self):
        import ast
        tree = ast.parse(Path(a.__file__).read_text())
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import): imports.extend(n.name for n in node.names)
            if isinstance(node, ast.ImportFrom): imports.append(node.module)
        self.assertFalse({'torch', 'PIL', 'safetensors', 'numpy'} & set(imports))


if __name__ == '__main__': unittest.main()
