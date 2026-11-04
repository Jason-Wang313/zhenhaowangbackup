"""Source-separated frozen-expert router qualification. Metadata only here."""
import csv
import re
from collections import defaultdict
from pathlib import Path
from smoke import qualify_foundir_five_expert_integration_v1 as parent

c, ROOT, ORDER = parent.c, parent.ROOT, parent.ORDER
CONFIG = 'configs/foundir_frozen_expert_router_qualification_v1.json'
CATEGORIES = dict(zip(ORDER, ('01Blur', '05Noise', '08Haze', '14Lowlight', '10Rain')))
SOURCES = [CONFIG, 'lowlight_restoration/frozen_expert_router.py',
           'smoke/frozen_expert_router_contract_v1.py', 'smoke/frozen_expert_router_state_v1.py',
           'smoke/train_foundir_frozen_expert_router_v1.py',
           'smoke/run_foundir_frozen_expert_router_v1_gpu.sh', 'smoke/start_foundir_frozen_expert_router_v1_tmux.sh',
           'smoke/train_qwen_multidegradation_lora_ddp.py', 'smoke/qwen_multidegradation_lora_preflight.py',
           'lowlight_restoration/qwen_lora.py', 'lowlight_restoration/qwen_five_class_gate.py']


def config():
    cfg = c.read(ROOT / CONFIG)
    expected = dict(train_sources_per_task=128, development_sources_per_task=8, crop=512,
                    image_feature_dim=6144, vlm_feature_dim=3584, router_hidden=256,
                    warmup_steps=400, quality_steps=8, warmup_learning_rate=.001,
                    quality_learning_rate=.0001, global_batch=32, world_size=8, nodes=4,
                    local_gpus=2, wall_cap_seconds=10800, minimum_free_disk_gib=20,
                    official_pixels_allowed_in_this_run=False, may_seed_formal=False, automatic_formal=False)
    if any(cfg.get(k) != v for k, v in expected.items()):
        raise ValueError('bounded qualification scope changed')
    return cfg


def select(rows, cfg):
    rows = list(rows)
    if len({r['record_id'] for r in rows}) != len(rows):
        raise ValueError('duplicate manifest records')
    if any(r['split'] not in ('train', 'validation', 'internal_test') for r in rows):
        raise ValueError('unknown data split')
    blocked_sources = {r['source_group'] for r in rows if r['split'] != 'train'}
    blocked_gt = {r['gt_sha256'] for r in rows if r['split'] != 'train'}
    result = []
    for split, count in [('train', cfg['train_sources_per_task']), ('validation', cfg['development_sources_per_task'])]:
        for task, category in CATEGORIES.items():
            groups = defaultdict(list)
            for row in rows:
                if row['split'] != split or row['category'] != category or min(int(row['width']), int(row['height'])) < cfg['crop']:
                    continue
                if int(row['composition_size']) != 1:
                    raise ValueError('mixed record in singleton category')
                if not re.fullmatch(r'[A-Za-z0-9_-]+', row['record_id']):
                    raise ValueError('unsafe record key')
                if any(not re.fullmatch(r'[0-9a-f]{64}', row[k]) for k in ('source_group', 'gt_sha256', 'lq_sha256')):
                    raise ValueError('invalid image/source hashes')
                if split == 'train' and (row['source_group'] in blocked_sources or row['gt_sha256'] in blocked_gt):
                    continue
                for side in ('lq_path', 'gt_path'):
                    if not row[side].startswith(f'data/foundir/MillionIRData/Train/{category}/') or '..' in Path(row[side]).parts:
                        raise ValueError('pixels outside existing collection')
                groups[row['source_group']].append(row)
            names = sorted(groups, key=lambda g: c.stable(cfg['seed'], task, split, g))[:count]
            if len(names) != count:
                raise ValueError('not enough source-disjoint images: ' + task + '/' + split)
            for group in names:
                row = dict(min(groups[group], key=lambda r: c.stable(cfg['seed'], r['record_id'])))
                raw = bytes.fromhex(c.stable(cfg['seed'], 'crop', row['record_id']))
                row['crop_xy'] = [int.from_bytes(raw[:8], 'big') % (int(row['width'])-cfg['crop']+1),
                                  int.from_bytes(raw[8:16], 'big') % (int(row['height'])-cfg['crop']+1)]
                result.append({'key': split + '_' + row['record_id'], 'task': task, 'split': split, 'row': row,
                               'seed': int.from_bytes(raw[16:20], 'big')})
    return result


def training_item(items, slot):
    if type(slot) is not int or not 0 <= slot < 408 * 32:
        raise ValueError('outside warmup/qualification budget')
    task = ORDER[slot % 5]
    rows = [i for i in items if i['split'] == 'train' and i['task'] == task]
    if len(rows) != 128:
        raise ValueError('training selection changed')
    return rows[(slot // 5) % len(rows)]


def output_path(run_id):
    c.run_path(ROOT, run_id)
    return c.path(ROOT, 'checkpoints/foundir_frozen_expert_router_v1/' + run_id)


def make_receipt():
    cfg = config(); folder = c.path(ROOT, cfg['parent_run'])
    if c.sha(folder / 'receipt.json') != cfg['parent_receipt_sha256'] or c.sha(folder / 'result.json') != cfg['parent_result_sha256']:
        raise ValueError('wrong five-expert integration parent')
    prior = parent.load_receipt(folder)
    report = c.read(ROOT / cfg['parent_report'])
    if not (report['single_selection_parity_pass'] and report['rain_development_gate_pass']
            and report['rain_historical_pixel_reproduction_pass'] and report['result_sha256'] == cfg['parent_result_sha256']
            and len(report['nodes']) == 4 and all(n['status'] == 0 for n in report['nodes'])):
        raise ValueError('integration not qualified')
    if c.sha(ROOT / cfg['manifest']) != cfg['manifest_sha256'] or c.sha(ROOT / cfg['qwen_config']) != cfg['qwen_config_sha256']:
        raise ValueError('manifest/Qwen contract changed')
    with (ROOT / cfg['manifest']).open(newline='') as f:
        items = select(csv.DictReader(f), cfg)
    qwen = c.read(ROOT / cfg['qwen_config'])['qwen']
    qcfg = c.read(ROOT / qwen['training_config'])
    if c.sha(ROOT / qwen['training_config']) != qwen['training_config_sha256']:
        raise ValueError('Qwen training config changed')
    inv = qcfg['model']['inventory_config']
    if c.sha(ROOT / inv) != qcfg['model']['inventory_config_sha256']:
        raise ValueError('Qwen model inventory changed')
    inventory = c.read(ROOT / inv)['model']
    if inventory['local_path'] != qcfg['model']['local_path']:
        raise ValueError('Qwen model mismatch')
    assets = {qwen['adapter']: qwen['adapter_sha256'], cfg['manifest']: cfg['manifest_sha256']}
    for key in ('flow_weights', 'autoencoder_weights'):
        assets[prior['model'][key]] = prior['model'][key+'_sha256']
    assets.update({inventory['local_path'] + '/' + name: info['sha256'] for name, info in inventory['expected_files'].items()})
    stats = {}
    for name, digest in assets.items():
        file = c.path(ROOT, name)
        before = file.stat()
        if c.sha(file) != digest:
            raise ValueError('Qwen/data identity changed: ' + name)
        after = file.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError('asset changed during check')
        stats[name] = [after.st_size, after.st_mtime_ns]
    for item in items:
        for side in ('lq_path', 'gt_path'):
            if not c.path(ROOT, item['row'][side]).is_file():
                raise FileNotFoundError(item['row'][side])
    sources = {**prior['sources'], **{n: c.sha(ROOT / n) for n in SOURCES +
               [cfg['parent_report'], cfg['qwen_config'], qwen['training_config'], inv]}}
    return {'config': cfg, 'parent': prior, 'items': items, 'selection_digest': c.digest(items),
            'unique_sources_by_split': {s: len({i['row']['source_group'] for i in items if i['split'] == s})
                                        for s in ('train', 'validation')},
            'qwen': qwen, 'qcfg': qcfg, 'sources': sources, 'assets': assets, 'asset_stats': stats,
            'official_pixels_read': False, 'frozen_experts': True, 'trainable': 'router_image_and_vlm_MLP_only'}


def load_receipt(output):
    b = c.read(output / 'receipt.json')
    if b['config'] != config() or c.digest(b['items']) != b['selection_digest']:
        raise ValueError('bound config/selection changed')
    for name, digest in b['sources'].items():
        if c.sha(ROOT / name) != digest:
            raise ValueError('bound source changed: ' + name)
    for name, old in b['asset_stats'].items():
        stat = c.path(ROOT, name).stat()
        if [stat.st_size, stat.st_mtime_ns] != old:
            raise ValueError('asset changed since prepare: ' + name)
    return b
