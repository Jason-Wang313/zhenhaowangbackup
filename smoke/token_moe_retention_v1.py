"""Train-only parent-relative retention objective; no new restoration network.

Old protocol/code is immutable. This is a new batch-40, fresh-Adam phase, not
an exact continuation of the old batch-32 optimizer trajectory.
"""
import json
import math
from pathlib import Path

import torch

from smoke.token_moe_five_single_v1 import (TASKS, FiveSamples, load_contract as old_contract,
                                          verify_parent, experiment_sources)
from smoke.token_residual_moe_v1 import sha, digest, write_once

CONFIG = 'configs/foundir_token_moe_retention_qualification_v1.json'
PROTOCOL = 'configs/foundir_token_moe_five_single_qualification_v1.json'
TERMS = ('flow_mse', 'latent_l1', 'rgb_mse', 'charbonnier', 'highpass', 'lpips')


def validate_config(raw):
    expected = {'experiment', 'protocol_config', 'mode', 'seed', 'steps', 'global_batch',
                'accumulation', 'save_every', 'milestones', 'shared_lora_lr', 'expert_lr',
                'router_lr', 'objective', 'output_root', 'minimum_free_gib',
                'wall_hours_estimate', 'allocated_gpu_hours_estimate', 'hard_wall_seconds_per_launch',
                'automatic_formal', 'qualification_may_seed_formal', 'planned_formal_steps',
                'formal_entry_enabled'}
    if (set(raw) != expected or raw['experiment'] != 'five_single_retention_v1'
            or raw['protocol_config'] != PROTOCOL or raw['mode'] != 'qualification'
            or (raw['seed'], raw['steps'], raw['global_batch'], raw['accumulation'], raw['save_every'])
                != (2026110105, 25, 40, 5, 5)
            or raw['milestones'] != [5, 25] or raw['planned_formal_steps'] != 1000
            or raw['output_root'] != 'checkpoints/foundir_token_moe_retention_qualification_v1'
            or raw['minimum_free_gib'] != 80 or raw['hard_wall_seconds_per_launch'] != 5400
            or raw['automatic_formal'] or raw['qualification_may_seed_formal'] or raw['formal_entry_enabled']
            or (raw['shared_lora_lr'], raw['expert_lr'], raw['router_lr']) != (1e-6, 1e-6, 1e-4)
            or raw['objective'] != dict(weighted_reference_floor=1e-4, rgb_mse_floor=4e-6,
                                        lpips_floor=1e-3, regression_weight=5.0)):
        raise RuntimeError('retention qualification contract changed; no hidden formal/mixed launch')


def load_contract(root, path):
    root = Path(root)
    raw = json.loads(Path(path).read_text())
    validate_config(raw)
    cfg, parent, base, qwen, qcfg, training, development, data = old_contract(root, root / PROTOCOL)
    # Keep the old parent lineage keys. Only optimizer, loss and sampling change.
    cfg = {**cfg, **raw, 'training_start_step': 0, 'fresh_optimizer': True,
           'restore_old_optimizer': False, 'full_evaluation_steps': []}
    parent = {**parent, 'training': {**parent['training'], **{k: raw[k] for k in
              ('shared_lora_lr', 'expert_lr', 'router_lr')}}}
    return cfg, parent, base, qwen, qcfg, training, development, data


def source_hashes(root, relative, lineage):
    names = set(experiment_sources(root, PROTOCOL, lineage)) | {
        relative, 'smoke/token_moe_retention_v1.py',
        'smoke/train_foundir_token_moe_retention_v1.py',
        'smoke/run_foundir_token_moe_retention_v1_4x2.sh',
        'lowlight_restoration/token_moe_retention_state.py'}
    return {name: sha(Path(root) / name) for name in sorted(names)}


def batch_slot(step, micro, rank):
    if step < 0 or not 0 <= micro < 5 or not 0 <= rank < 8:
        raise ValueError('batch40 requires 5 accumulation slots on eight ranks')
    return step * 40 + micro * 8 + rank, .2


class RetentionSamples(FiveSamples):
    def train_spec(self, offset):
        if not 0 <= offset < self.cfg['steps'] * 40:
            raise ValueError('sample outside new batch40 phase')
        micro, rank = divmod(offset, 8)
        task_id = micro % 5
        task = TASKS[task_id]
        position = (micro // 5) * 8 + rank
        row = self.training[task][self.schedules[task].index(position)]
        return {'key': f'retention_train_{offset:09d}', 'kind': 'train', 'task': task,
                'task_id': task_id, 'position': position, 'seed': self.cfg['seed'] + task_id, 'row': row}

    def all_specs(self):
        specs = [self.train_spec(i) for i in range(self.cfg['steps'] * 40)]
        for task in TASKS:
            split = 'rain' if task == 'rain' else task + '_preview'
            for row in self.development[split]:
                if row['split'] != 'validation':
                    raise RuntimeError('evaluation rejects non-development pixels')
                specs.append({'key': f'{split}_{row["record_id"]}', 'kind': 'validation',
                              'task': task, 'split': split, 'row': row})
        return specs


def assert_train_reference(spec):
    row = spec['row']
    if spec['kind'] != 'train' or row['split'] != 'train' or row['role'] != 'direct_endpoint_train':
        raise RuntimeError('reference targets may use TRAINING pixels only')
    for key in ('lq_path', 'gt_path'):
        if not row[key].startswith('data/foundir/MillionIRData/Train/') or '..' in Path(row[key]).parts:
            raise RuntimeError('reference path outside approved training collection')


def validate_terms(values):
    if set(values) != set(TERMS) or any(type(v) not in (float, int) or not math.isfinite(v) or v < 0
                                      for v in values.values()):
        raise RuntimeError('invalid parent loss scalars')


def reference_binding(spec, binding):
    assert_train_reference(spec)
    return {'binding_sha256': digest(binding), 'sample_sha256': digest(spec),
            'kind': 'training_only_parent_loss_scalars'}


def read_reference(path, expected):
    if not Path(path).exists():
        return None
    data = json.loads(Path(path).read_text())
    if set(data) != {'identity', 'terms'} or data['identity'] != expected:
        raise RuntimeError('reference identity mismatch')
    validate_terms(data['terms'])
    return data['terms']


def save_reference(path, values, identity):
    validate_terms(values)
    write_once(path, {'identity': identity, 'terms': values})


def retention_objective(terms, reference, weights, options):
    """Relative total loss + separate positive MSE and LPIPS regression costs.

    Reference contains detached scalars from the same TRAINING crop/D4 and
    frozen parent. Neither old outputs nor GT become model/router inputs.
    Floors bound normalization at near-perfect predictions; harm is measured
    against the real reference, not the floored denominator.
    """
    validate_terms(reference)
    if set(terms) != set(TERMS) or set(weights) != set(TERMS):
        raise RuntimeError('loss term contract changed')
    if any(t.numel() != 1 or not bool(torch.isfinite(t).all()) for t in terms.values()):
        raise RuntimeError('nonfinite or non-scalar training loss')
    parent_total = sum(reference[k] * weights[k] for k in TERMS)
    normalized = sum(terms[k] * weights[k] for k in TERMS) / max(parent_total, options['weighted_reference_floor'])
    harm = {k: (terms[k] - reference[k]).relu() / max(reference[k], options[k + '_floor'])
            for k in ('rgb_mse', 'lpips')}
    loss = normalized + options['regression_weight'] * sum(harm.values())
    if not bool(torch.isfinite(loss).all()):
        raise RuntimeError('nonfinite relative training objective')
    return loss, {'relative_total': normalized, 'mse_regression': harm['rgb_mse'],
                  'lpips_regression': harm['lpips']}
