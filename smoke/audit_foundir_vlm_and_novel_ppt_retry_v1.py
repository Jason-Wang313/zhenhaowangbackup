#!/usr/bin/env python3
"""Read existing metadata/results only; never open images or run a model.

This audit is descriptive. It must not choose a checkpoint or a new threshold
using the development outcomes. JSON is written exclusively, never replaced.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
ATOMS = ('blur', 'noise', 'rain', 'haze', 'low_light')
MOE = 'checkpoints/foundir_rain_blur_joint_vlm_moe_v2'
PAIRWISE = 'checkpoints/foundir_pairwise_safe_candidate_gate_noise_v1/formal1'
V2 = 'checkpoints/foundir_proposal_relative_energy_noise_v2/noise_energy_v2_formal1'


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def matched_rows(full, subset):
    indexed = {r['record_id']: r for r in full}
    if len(indexed) != len(full):
        raise ValueError('duplicate full-result IDs')
    ids = [r['record_id'] for r in subset]
    if len(set(ids)) != len(ids) or not set(ids) <= indexed.keys():
        raise ValueError('invalid result subset')
    for row in subset:
        reference = indexed[row['record_id']]
        for field in ('source_group', 'input_psnr', 'input_lpips'):
            if reference[field] != row[field]:
                raise ValueError(f'paired result mismatch: {field}')
    return [indexed[key] for key in ids]


def metrics(rows):
    return {'records': len(rows),
            'psnr': mean(r['output_psnr'] for r in rows),
            'lpips': mean(r['output_lpips'] for r in rows)}


def routing(rows, target_expert):
    result = {}
    for block in sorted(rows[0]['expert_weights']):
        weights = [r['expert_weights'][block][target_expert] for r in rows]
        result[block] = {'mean_task_expert_weight': mean(weights),
                         'task_expert_argmax_fraction': mean(w > .5 for w in weights),
                         'task_expert_weight_above_099_fraction': mean(w > .99 for w in weights)}
    return result


def condition_summary(rows, conditions, task):
    levels = [conditions[r['record_id']] for r in rows]
    groups = defaultdict(list)
    for row, level in zip(rows, levels, strict=True):
        groups[str(level[task])].append(row)
    return {'target_severity_counts': dict(sorted(Counter(str(v[task]) for v in levels).items())),
            'all_five_absent_fraction': mean(all(v[a] == 1 for a in ATOMS) for v in levels),
            'unique_five_level_descriptors': len({tuple(v[a] for a in ATOMS) for v in levels}),
            'quality_by_predicted_severity_descriptive_not_ground_truth': {
                k: metrics(v) for k, v in sorted(groups.items())}}


def audit(root):
    bindings = {}

    def read(relative):
        path = root / relative
        bindings[relative] = sha256(path)
        return json.loads(path.read_text())

    config = read('configs/foundir_rain_blur_joint_vlm_moe_v2.json')
    condition_path = root / config['conditions']['path']
    actual = sha256(condition_path)
    if actual != config['conditions']['sha256']:
        raise ValueError('condition manifest changed')
    bindings[config['conditions']['path']] = actual
    full = {}
    required_ids = set()
    for arm in ('image_only', 'vlm_assisted'):
        full[arm] = {}
        for task, split in (('rain', 'rain'), ('blur', 'blur_full')):
            obj = read(f'{MOE}/{arm}/formal_v1/metrics/{split}_010000_learned.json')
            if obj['GT_used_for_routing'] or obj['step'] != 10000:
                raise ValueError('invalid final routing evaluation')
            full[arm][task] = obj['rows']
            required_ids.update(r['record_id'] for r in obj['rows'])
    conditions = {}
    with condition_path.open(newline='') as stream:
        for row in csv.DictReader(stream):
            if row['record_id'] not in required_ids:
                continue  # metadata only, no pixel paths are opened
            if row['split'] != 'validation' or row['category'] not in ('01Blur', '10Rain'):
                raise ValueError('unexpected condition scope')
            level = json.loads(row['qwen_levels_json'])
            if set(level) != set(ATOMS) or any(type(x) is not int or not 1 <= x <= 5 for x in level.values()):
                raise ValueError('invalid predicted condition')
            if row['record_id'] in conditions:
                raise ValueError('duplicate condition')
            conditions[row['record_id']] = level
    if conditions.keys() != required_ids:
        raise ValueError('missing conditions')

    arms = {}
    for arm, tasks in full.items():
        arms[arm] = {}
        for task, rows in tasks.items():
            arm_result = {'full_final': metrics(rows),
                          'routing': routing(rows, 0 if task == 'rain' else 1),
                          'conditions': condition_summary(rows, conditions, task)}
            split = 'rain' if task == 'rain' else 'blur_preview'
            routes = ('constant_0', 'constant_1', 'vlm_off') if arm == 'vlm_assisted' else ('constant_0', 'constant_1')
            comparisons = {}
            for route in routes:
                obj = read(f'{MOE}/{arm}/formal_v1/metrics/{split}_010000_{route}.json')
                subset = matched_rows(rows, obj['rows'])
                normal, changed = metrics(subset), metrics(obj['rows'])
                comparisons[route] = {'matched_normal': normal, 'changed_route': changed,
                                      'changed_minus_normal_psnr': changed['psnr'] - normal['psnr'],
                                      'changed_minus_normal_lpips': changed['lpips'] - normal['lpips']}
            arm_result['fixed_weights_route_interventions_not_retraining_controls'] = comparisons
            arms[arm][task] = arm_result

    safe_history = {}
    for step in (1706, 3412, 5118, 6824, 8530):
        obj = read(f'{PAIRWISE}/validation_step_{step:06d}.json')
        safe_history[str(step)] = obj['validation']
    terminal = read(f'{PAIRWISE}/terminal_result.json')
    v2_history = {}
    for step in (1706, 3412, 5118):
        v2_history[str(step)] = read(f'{V2}/validation_step_{step:06d}.json')['validation']
    log_path = root / 'logs/foundir_proposal_relative_energy_noise_v2_noise_energy_v2_formal1.log'
    crop_logs = defaultdict(list)
    with log_path.open() as stream:
        for line in stream:
            if not line.startswith('{'):
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if 'direction' in row and 'step' in row:
                crop = '128' if row['step'] < 1706 else '192' if row['step'] < 5118 else '256'
                crop_logs[crop].append(row)
    log_summary = {crop: {'last_logged_step': rows[-1]['step'], 'logged_updates': len(rows),
                          'last_20_logged_mean_gradient_cosine': mean(1 - r['direction'] for r in rows[-20:])}
                   for crop, rows in crop_logs.items()}
    return {'schema_version': 1, 'status': 'existing_artifact_diagnosis_not_new_experiment',
            'new_training_or_inference': False, 'images_opened': 0, 'protected_pixels_accessed': False,
            'moe': arms, 'ppt_v2_existing_validation': v2_history,
            'ppt_v2_train_direction_logs': log_summary,
            'pairwise_gate_existing_validation': safe_history,
            'pairwise_gate_terminal': terminal,
            'interpretation': {
                'pairwise_joint_gate_is_absolute_not_improvement_over_fidelity': True,
                'oracle_headroom_does_not_establish_observable_learnable_ranking': True,
                'VLM_severity_is_prediction_not_ground_truth': True,
                'no_claim_that_router_or_VLM_are_useless_on_unmeasured_mixed_tasks': True},
            'artifact_sha256': bindings}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = audit(args.root)
    content = json.dumps(result, indent=2, allow_nan=False) + '\n'
    if args.output:
        with args.output.open('x') as stream:
            stream.write(content)
        print(f'EXISTING_ARTIFACT_AUDIT_WRITTEN {args.output}')
    else:
        print(content, end='')


if __name__ == '__main__':
    main()
