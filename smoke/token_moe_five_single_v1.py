"""Five-singleton admission on the unchanged two-expert VLM-token FLUX MoE.

No mixed training yet. Audit all eleven categories' split identities now so
later mixed-data extension cannot silently leak into the frozen development.
"""
import csv
import hashlib
from collections import Counter, defaultdict
from pathlib import Path

from smoke.token_residual_moe_v1 import (contract, bound_json, sha, digest, Samples,
                                       source_hashes, CONTINUATION_CONFIG)
from lowlight_restoration.rain_full_data import RainSourceSchedule

CONFIG = 'configs/foundir_token_moe_five_single_qualification_v1.json'
TASKS = ('rain', 'blur', 'noise', 'haze', 'low_light')
CATEGORIES = {'rain': '10Rain', 'blur': '01Blur', 'noise': '05Noise',
              'haze': '08Haze', 'low_light': '14Lowlight'}
ALL_CATEGORIES = {'01Blur', '02Blur_Noise', '05Noise', '08Haze', '09Lowlight_Haze',
                  '10Rain', '13Rain_Haze', '14Lowlight', '15Lowlight_Blur',
                  '16Lowlight_Noise', '18Lowlight_Blur_Noise'}


def balanced_preview(rows, count=100):
    """Round-robin across source groups, using IDs only, never quality scores."""
    groups = defaultdict(list)
    for row in sorted(rows, key=lambda r: hashlib.sha256(r['record_id'].encode()).hexdigest()):
        groups[row['source_group']].append(row)
    names = sorted(groups, key=lambda g: hashlib.sha256(g.encode()).hexdigest())
    result = []
    for index in range(max(map(len, groups.values()), default=0)):
        for group in names:
            if index < len(groups[group]):
                result.append(groups[group][index])
                if len(result) == count:
                    return result
    raise RuntimeError('insufficient development records for the frozen preview')


def development_crop(row, seed):
    row = dict(row)
    w, h = int(row['width']), int(row['height'])
    if row['split'] != 'validation' or min(w, h) < 512:
        raise RuntimeError('only unchanged 512 development crops allowed')
    for side in ('lq_path', 'gt_path'):
        if not row[side].startswith('data/foundir/MillionIRData/Train/') or '..' in Path(row[side]).parts:
            raise RuntimeError('development pixels must come from the approved collection')
    raw = hashlib.sha256(f"{seed}:full_development_crop:{row['record_id']}".encode()).digest()
    row.update(crop_y=str(int.from_bytes(raw[:8], 'big') % (h-512+1)),
               crop_x=str(int.from_bytes(raw[8:16], 'big') % (w-512+1)), role='five_single_development')
    return row


def expand_protocol(rows, old_training, old_development, references):
    """Consume metadata only; exclude every non-train source and exact GT hash."""
    by_category = {v: [] for v in CATEGORIES.values()}
    dev = {v: [] for v in CATEGORIES.values()}
    blocked_groups, blocked_gt, seen = set(), set(), set()
    counts = Counter()
    for r in rows:
        if r['record_id'] in seen or r['category'] not in ALL_CATEGORIES:
            raise RuntimeError('duplicate or unexpected five-type record')
        seen.add(r['record_id'])
        if r['split'] not in {'train', 'validation', 'internal_test'}:
            raise RuntimeError('unknown split; never infer permission')
        counts[(r['category'], r['split'])] += 1
        if r['split'] != 'train':
            blocked_groups.add(r['source_group']); blocked_gt.add(r['gt_sha256'])
        if r['category'] in by_category:
            if int(r['composition_size']) != 1:
                raise RuntimeError('mixed record entered singleton category')
            if r['split'] == 'train':
                by_category[r['category']].append(dict(r))
            elif r['split'] == 'validation':
                dev[r['category']].append(dict(r))
    for rows_ in old_development.values():
        for r in rows_:
            blocked_groups.add(r['source_group']); blocked_gt.add(r['gt_sha256'])
    training, excluded = {}, {}
    for task, category in CATEGORIES.items():
        source = old_training[task] if task in old_training else by_category[category]
        # The existing two-task model must not already have seen new dev sources.
        if task in old_training and any(r['source_group'] in blocked_groups or r['gt_sha256'] in blocked_gt for r in source):
            raise RuntimeError('parent training overlaps expanded development; cannot claim source isolation')
        allowed = []
        for r in source:
            if r['split'] != 'train' or r['category'] != category:
                raise RuntimeError('invalid training pool')
            if r['source_group'] in blocked_groups or r['gt_sha256'] in blocked_gt:
                continue
            if min(int(r['width']), int(r['height'])) < 256:
                raise RuntimeError('unsupported training geometry')
            # Reject external/test paths before any pixel read.
            for side in ('lq_path', 'gt_path'):
                if not r[side].startswith('data/foundir/MillionIRData/Train/') or '..' in Path(r[side]).parts:
                    raise RuntimeError('pixels must come from the approved training collection')
            allowed.append(dict(r, role='direct_endpoint_train'))
        training[task] = sorted(allowed, key=lambda r: r['record_id'])
        excluded[task] = len(source)-len(allowed)
    development = dict(old_development)
    for task in ('noise', 'haze', 'low_light'):
        reference = references[task]['data']
        rows_ = sorted(dev[CATEGORIES[task]], key=lambda r: r['record_id'])
        if (len(rows_) != reference['expected_records'] or
                len({r['source_group'] for r in rows_}) != reference['expected_source_groups']):
            raise RuntimeError('historical full development identity changed: ' + task)
        full = [development_crop(r, reference['crop_seed']) for r in rows_]
        development[task + '_full'] = full
        development[task + '_preview'] = balanced_preview(full)
    metadata = {
        'train_records': {t: len(r) for t, r in training.items()},
        'train_sources': {t: len({x['source_group'] for x in r}) for t, r in training.items()},
        'excluded_nontrain_overlap_records': excluded,
        'training_digest': {t: digest(r) for t, r in training.items()},
        'development_records': {t: len(r) for t, r in development.items()},
        'development_sources': {t: len({x['source_group'] for x in r}) for t, r in development.items()},
        'development_digest': {t: digest(r) for t, r in development.items()},
        'eleven_category_split_counts': {c: {s: counts[(c, s)] for s in ('train', 'validation', 'internal_test')}
                                       for c in sorted(ALL_CATEGORIES)},
        'protected_pixels_accessed': False, 'mixed_pixels_accessed': False,
    }
    return training, development, metadata


def load_contract(root, config_path):
    import json
    root = Path(root)
    raw = json.loads(Path(config_path).read_text())
    expected = {'experiment', 'parent_config', 'parent_config_sha256', 'parent_report',
                'parent_report_sha256', 'parent_receipt', 'parent_receipt_sha256', 'mode',
                'seed', 'training_start_step', 'steps', 'save_every', 'milestones',
                'full_evaluation_steps', 'tasks', 'lpips_by_task', 'router_ce_weight',
                'preview_records_per_task', 'reference_protocols', 'output_root',
                'minimum_free_gib', 'wall_hours_estimate', 'allocated_gpu_hours_estimate',
                'hard_wall_seconds_per_launch', 'automatic_formal', 'qualification_may_seed_formal'}
    if (set(raw) != expected or raw['experiment'] != 'five_single_qualification_v1'
            or raw['parent_config'] != CONTINUATION_CONFIG or raw['mode'] != 'qualification'
            or (raw['training_start_step'], raw['steps'], raw['save_every']) != (2000, 2025, 5)
            or raw['milestones'] != [2025] or raw['full_evaluation_steps'] != []
            or raw['tasks'] != list(TASKS) or raw['preview_records_per_task'] != 100
            or raw['router_ce_weight'] != 0 or raw['automatic_formal'] or raw['qualification_may_seed_formal']
            or raw['lpips_by_task'] != dict(rain=.05, blur=.20, noise=.05, haze=.05, low_light=.05)):
        raise RuntimeError('five-single engineering scope changed; no automatic formal/mixed training')
    bound_json(root, raw['parent_config'], raw['parent_config_sha256'])
    cfg, parent, base, qwen, qcfg, old_training, old_dev, old_data = contract(root, root / raw['parent_config'])
    # These are explicitly a new dataset/objective schedule, not exact old-data continuation.
    cfg = {**cfg, **raw, 'fresh_optimizer': False, 'restore_old_optimizer': True}
    cfg.pop('continuation', None)
    refs = {t: bound_json(root, v['path'], v['sha256']) for t, v in raw['reference_protocols'].items()}
    manifest = bound_json(root, parent['parent_config'], parent['parent_config_sha256'])['source_manifest']
    # contract() has already hash-checked this exact CSV.
    with (root / manifest).open(newline='') as handle:
        training, development, data = expand_protocol(csv.DictReader(handle), old_training, old_dev, refs)
    data['parent_two_task_data'] = old_data
    for task in TASKS:
        if not training[task]:
            raise RuntimeError('empty admitted singleton pool')
    return cfg, parent, base, qwen, qcfg, training, development, data


def verify_parent(root, cfg, *, artifacts=True, sources=True, tensors=False):
    root = Path(root)
    report = bound_json(root, cfg['parent_report'], cfg['parent_report_sha256'])
    result = report['full_result']; binding = result['binding']
    if not (report['status'] == 'continuation_terminal_complete_absolute_gate_pass'
            and result['status'] == 'complete' and result['mode'] == 'formal' and result['steps'] == 2000
            and result['absolute_quality_gate_pass'] and report['final_checkpoint_sha_verified']
            and not result['protected_pixels_accessed']
            and binding['config_sha256'] == cfg['parent_config_sha256']):
        raise RuntimeError('requires the audited successful VLM-token step2000')
    if report['artifact_hashes'][cfg['parent_receipt']]['sha256'] != cfg['parent_receipt_sha256']:
        raise RuntimeError('wrong parent receipt binding')
    if sources:
        for name, value in binding['source_sha256'].items():
            if name != 'smoke/train_foundir_token_residual_moe_v1.py' and sha(root / name) != value:
                raise RuntimeError('scientific model implementation changed: ' + name)
    if artifacts:
        directory = Path(cfg['parent_receipt']).parent
        for filename in ('launch.json', 'result.json', 'step_002000.resume.json'):
            name = str(directory / filename)
            if sha(root / name) != report['artifact_hashes'][name]['sha256']:
                raise RuntimeError('parent evidence changed: ' + name)
        receipt = bound_json(root, cfg['parent_receipt'], cfg['parent_receipt_sha256'])
        expected = [r for r in result['checkpoints'] if r['step'] == 2000]
        if len(expected) != 1 or receipt != expected[0]:
            raise RuntimeError('parent full state differs from audited result')
        if tensors:
            for key in ('weights', 'optimizer'):
                if Path(receipt[key]).name != receipt[key] or sha(root / directory / receipt[key]) != receipt[key + '_sha256']:
                    raise RuntimeError('parent tensor identity changed')
    return {'binding': binding, 'receipt': cfg['parent_receipt'], 'start_step': 2000,
            'mode': 'formal', 'report_sha256': cfg['parent_report_sha256'],
            'parent_metrics': report['full_metrics']['2000']}


def experiment_sources(root, config_relative, lineage):
    names = set(source_hashes(root, config_relative)) | set(lineage['binding']['source_sha256'])
    names.update(['smoke/token_moe_five_single_v1.py', 'smoke/run_foundir_token_moe_five_single_v1_4x2.sh'])
    return {name: sha(root / name) for name in sorted(names)}


class FiveSamples(Samples):
    def __init__(self, root, cfg, training, development):
        self.root, self.cfg, self.training = Path(root), cfg, training
        self.schedules = {t: RainSourceSchedule(training[t], cfg['seed'] + i) for i, t in enumerate(TASKS)}
        self.development = development
        self.verified = {}

    def train_spec(self, offset):
        begin, end = self.cfg['training_start_step'] * 32, self.cfg['steps'] * 32
        if not begin <= offset < end:
            raise ValueError('sample outside this new five-task schedule')
        micro, rank = divmod(offset-begin, 8)
        task_id = micro % len(TASKS); task = TASKS[task_id]
        position = (micro // len(TASKS)) * 8 + rank
        row = self.training[task][self.schedules[task].index(position)]
        return {'key': f'five_train_{offset:09d}', 'kind': 'train', 'task': task, 'task_id': task_id,
                'position': position, 'seed': self.cfg['seed'] + task_id, 'row': row}

    def all_specs(self):
        specs = [self.train_spec(i) for i in range(self.cfg['training_start_step']*32, self.cfg['steps']*32)]
        for task in TASKS:
            split = 'rain' if task == 'rain' else task + '_preview'
            for row in self.development[split]:
                if row['split'] != 'validation':
                    raise RuntimeError('evaluation rejects non-development pixels')
                specs.append({'key': f'{split}_{row["record_id"]}', 'kind': 'validation',
                              'task': task, 'split': split, 'row': row})
        return specs

    def evaluation_specs(self, specs, *, full=False):
        if full:
            raise RuntimeError('25-step preview cannot masquerade as full five-type gate')
        return [s for s in specs if s['kind'] == 'validation']
