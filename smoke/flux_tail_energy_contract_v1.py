"""Frozen, training-pool-only source partition and cache identity."""
from __future__ import annotations
import csv
import json
from pathlib import Path

from smoke import flux_energy_qualification_contract_v1 as common
from smoke import diagnose_foundir_flux_tail_confirmation_v1 as confirmation
from smoke import diagnose_foundir_flux_tail_oracle_v1 as base

ROOT = common.ROOT
CONFIG = ROOT/'configs/foundir_flux_tail_energy_qualification_v1.json'
FILES = ['configs/foundir_flux_tail_energy_qualification_v1.json',
    'lowlight_restoration/flux_tail_learned_energy.py',
    'smoke/flux_tail_energy_contract_v1.py', 'smoke/flux_tail_energy_runtime_v1.py',
    'smoke/qualify_foundir_flux_tail_energy_v1.py',
    'smoke/run_foundir_flux_tail_energy_qualification_v1_gpu.sh',
    'smoke/start_foundir_flux_tail_energy_qualification_v1_tmux.sh']


def binding():
    return confirmation.binding() | {p: common.sha(ROOT/p) for p in FILES}


def validate_spec(s):
    expected = dict(mode='fresh_source_tail_energy_engineering_qualification',
        parent_result='checkpoints/foundir_flux_tail_confirmation_v1/confirm24_4gpu_v1/result.json',
        parent_result_sha256='37f13289195eb6abd4fc4127f797008f7e6843ebd11957123b7245fcfa735137',
        output_root='checkpoints/foundir_flux_tail_energy_qualification_v1', allowed_world_sizes=[4, 8],
        selection_seed=2026102809, seed=2026102810, exclude_previous_32_sources=True,
        reserved_calibration_sources=128, reserved_development_sources=128,
        qualification_train_sources=24, qualification_evaluation_sources=8, crop=512,
        global_batch_pairs=32, microbatch_pairs=1, training_steps=25, learning_rate=.0001,
        warmup_steps=5, weight_decay=.0001, gradient_clip=1., energy=dict(width=32, condition_dim=96),
        loss=dict(gradient_weight=1., pair_weight=.25), cache_teacher_max_updates=16,
        cache_random_rms_amplitudes=[.002, .01], guidance_max_updates=16,
        milestones=[1, 2, 4, 8, 16], start_index=48, total_steps=50,
        direction_weights=[0., .25, .5, .75, 1.], rms_latent_steps=[.0005, .001, .002, .005, .01],
        psnr_gain_target=.3, lpips_relative_reduction_target=.09928107050377874,
        budget_wall_seconds_by_gpus={'4': 5400, '8': 2700}, budget_used_gpu_hours_cap=6,
        auto_formal=False, qualification_may_seed_formal=False, protected_pixels_allowed=False)
    if s != expected:
        raise RuntimeError('frozen tail energy qualification scope changed')


def assigned(indices, rank, world):
    if world not in (4, 8) or not 0 <= rank < world:
        raise ValueError('single-node 4/8 ranks required')
    return list(indices)[rank::world]


def training_slots(step, rank, world):
    assigned([], rank, world)
    if step < 1:
        raise ValueError('one-based optimizer steps required')
    return list(range((step-1)*32+rank, step*32, world))


def partition(rows, old_rows, seed):
    protected_sources = {r['source_group'] for r in rows if r['split'] != 'train'}
    protected_gt = {r['gt_sha256'] for r in rows if r['split'] != 'train'}
    old_sources = {r['source_group'] for r in old_rows}
    old_gt = {r['gt_sha256'] for r in old_rows}
    old_lq = {r['lq_sha256'] for r in old_rows}
    blocked = old_sources | {r['source_group'] for r in rows if r['gt_sha256'] in old_gt or r['lq_sha256'] in old_lq}
    groups = {}
    for r in rows:
        if r['split'] != 'train' or r['category'] != '05Noise':
            continue
        if r['source_group'] in protected_sources or r['gt_sha256'] in protected_gt:
            raise RuntimeError('training/protected source overlap')
        if r['source_group'] in blocked or min(int(r['width']), int(r['height'])) < 512:
            continue
        common.descriptor(r)
        groups.setdefault(r['source_group'], []).append(r)
    keys = sorted(groups, key=lambda k: common.digest(seed, 'partition', k))
    if len(keys) < 280:
        raise RuntimeError('insufficient independent sources')
    roles = {k: ('calibration' if i < 128 else 'engineering_evaluation' if i < 256 else 'train')
             for i, k in enumerate(keys)}
    content_roles = {}
    for k, records in groups.items():
        for r in records:
            for field in ('gt_sha256', 'lq_sha256'):
                identity = (field, r[field])
                if identity in content_roles and content_roles[identity] != roles[k]:
                    raise RuntimeError('training/calibration/evaluation content overlap')
                content_roles[identity] = roles[k]
    selected = []
    for role, subset in [('train', keys[256:280]), ('engineering_evaluation', keys[128:136])]:
        for k in subset:
            row = min(groups[k], key=lambda r: common.digest(seed, 'record', r['record_id']))
            selected.append(dict(role=role, row=row))
    for field in ('source_group', 'record_id', 'gt_sha256', 'lq_sha256'):
        if len({v['row'][field] for v in selected}) != 32:
            raise RuntimeError('duplicate qualification identity: '+field)
    counts = {role: sum(v == role for v in roles.values()) for role in set(roles.values())}
    return selected, counts, roles


def prepare(output, world, write=False):
    assigned([], 0, world)
    spec = json.loads(CONFIG.read_text()); validate_spec(spec)
    prior = json.loads(common.bound(spec['parent_result'], spec['parent_result_sha256']).read_text())
    if not prior['primary_mean_joint_target'] or prior['records'] != 24:
        raise RuntimeError('requires confirmed GT feasibility')
    for filename, expected in prior['source_sha256'].items(): common.bound(filename, expected)
    parent_spec = json.loads(base.CONFIG.read_text())
    parent, previous = base.parent_receipt(parent_spec)
    old = previous['selected']
    artifacts = {}
    for field in ('manifest', 'geometry_manifest', 'flow', 'ae'):
        p, sha = previous['spec'][field], previous['spec'][field+'_sha256']
        print('VERIFY '+field, flush=True); common.bound(p, sha); artifacts[p] = sha
    for p in (previous['plan']['baseline_checkpoint'], 'models/torch/hub/checkpoints/alexnet-owt-7be5be79.pth'):
        sha = previous['artifacts'][p]; common.bound(p, sha); artifacts[p] = sha
    geometry, blocked = {}, []
    with common.path(previous['spec']['geometry_manifest']).open(newline='') as h:
        count = 0
        for row in csv.DictReader(h):
            count += 1
            if row['category'] == '05Noise': geometry[row['record_id']] = row
            elif row['split'] != 'train': blocked.append(row)
        if count != 685495: raise RuntimeError('geometry cardinality changed')
    rows = []
    with common.path(previous['spec']['manifest']).open(newline='') as h:
        count = 0
        for row in csv.DictReader(h):
            count += 1
            if row['category'] != '05Noise': continue
            other = geometry.pop(row['record_id'])
            for k in ('split', 'source_group', 'category', 'lq_path', 'gt_path', 'lq_sha256', 'gt_sha256'):
                if row[k] != other[k]: raise RuntimeError('manifest join mismatch')
            keys = ('record_id', 'split', 'source_group', 'category', 'lq_path', 'gt_path',
                    'lq_sha256', 'gt_sha256', 'qwen_levels_json')
            rows.append({k: row[k] for k in keys} | {k: other[k] for k in ('width', 'height')})
        if count != 685495 or geometry: raise RuntimeError('VLM manifest cardinality changed')
    train = [r for r in rows if r['split'] == 'train']
    if len(train) != 54585 or len({r['source_group'] for r in train}) != 1755:
        raise RuntimeError('Noise training inventory changed')
    selected, counts, roles = partition(rows+blocked, old, spec['selection_seed'])
    for i, item in enumerate(selected):
        row = item['row']
        item.update(index=i, seed=int(common.digest(2026091604, 'noise_scalar_full_development_v1', row['record_id'])[:16], 16),
                    crop_xy=common.crop_xy(row))
        for field in ('lq_path', 'gt_path'):
            if not common.path(row[field]).is_file(): raise FileNotFoundError(row[field])
    receipt = dict(spec=spec, world_size=world, source_sha256=binding(), config_sha256=common.sha(CONFIG),
        parent_spec=parent_spec, parent_cache=str(parent.relative_to(ROOT)), selected=selected,
        base_spec=previous['spec'], plan=previous['plan'], artifacts=artifacts,
        partition_sources=counts, source_roles=roles, protected_pixels_accessed=False, images_decoded=0)
    if write:
        output.mkdir(parents=True, exist_ok=False); common.write_new(output/'preflight.json', receipt)
    print('TAIL_ENERGY_PREFLIGHT_PASS '+json.dumps(dict(world_size=world, train=24, evaluation=8,
          partition_sources=counts, excluded_previous_sources=len({r['source_group'] for r in old}), images_decoded=0)), flush=True)
    return receipt


def load(output, world):
    r = json.loads((output/'preflight.json').read_text()); validate_spec(r['spec'])
    if r['world_size'] != world or r['config_sha256'] != common.sha(CONFIG) or r['source_sha256'] != binding():
        raise RuntimeError('source/config/world changed after prepare')
    return r


def cache_meta(output, receipt, index, role):
    meta = json.loads((output/f'cache_{index:03d}.json').read_text())
    if (meta['identity'] != receipt['selected'][index] or meta['identity']['role'] != role
            or meta['config_sha256'] != receipt['config_sha256']
            or meta['source_sha256'] != receipt['source_sha256']
            or meta['file'] != f'cache_{index:03d}.safetensors'):
        raise RuntimeError('cache identity/role mismatch')
    common.bound(str(output/meta['file']), meta['sha256'])
    return meta
