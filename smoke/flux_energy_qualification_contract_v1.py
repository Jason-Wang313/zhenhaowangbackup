"""Offline metadata and cache identity contract; no pixel decoding at preflight."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'configs/foundir_flux_energy_qualification_v1.json'
ORDER = ('blur', 'noise', 'rain', 'haze', 'low_light')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def digest(*parts):
    return hashlib.sha256(':'.join(map(str, parts)).encode()).hexdigest()


def path(value):
    result = (ROOT / value).resolve()
    result.relative_to(ROOT)
    return result


def write_new(destination, value):
    with Path(destination).open('x') as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')


def bound(filename, expected):
    result = path(filename)
    if sha(result) != expected:
        raise RuntimeError('SHA-256 mismatch: ' + str(filename))
    return result


def run_path(run_id):
    if not re.fullmatch(r'[A-Za-z0-9_-]+', run_id):
        raise ValueError('invalid fresh run ID')
    return path(json.loads(CONFIG.read_text())['output_root']) / run_id


def source_binding():
    # Include the local import closure, not just the entrypoint. No old trainer is imported.
    local = ('__init__', 'data', 'flow', 'lora', 'scalar_residual', 'flux_energy_gate',
             'novel_ppt_energy', 'novel_ppt_energy_nonzero_init',
             'proposal_relative_energy', 'proposal_relative_energy_v2', 'vlm_energy_restoration')
    files = [ROOT / f'lowlight_restoration/{name}.py' for name in local]
    files += list((ROOT / 'third_party/flux2/src/flux2').glob('*.py'))
    files += [CONFIG, ROOT / 'smoke/flux_energy_qualification_contract_v1.py',
              ROOT / 'smoke/qualify_foundir_flux_energy_v1.py',
              ROOT / 'smoke/run_foundir_flux_energy_qualification_v1_8gpu.sh']
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(files)}


def descriptor(row):
    levels = json.loads(row['qwen_levels_json'])
    if set(levels) != set(ORDER) or any(type(v) is not int or not 1 <= v <= 5 for v in levels.values()):
        raise ValueError('invalid per-image VLM descriptor')
    return [(levels[k] - 1) / 4 for k in ORDER]


def select_training(rows, *, count, seed):
    """One hash-selected record per source; held-out metadata only for exclusion."""
    blocked_sources = {r['source_group'] for r in rows if r['split'] != 'train'}
    blocked_targets = {r['gt_sha256'] for r in rows if r['split'] != 'train'}
    if len({r['record_id'] for r in rows}) != len(rows):
        raise RuntimeError('duplicate record ID')
    groups = {}
    for row in rows:
        if row['split'] not in ('train', 'validation', 'internal_test'):
            raise RuntimeError('unknown split')
        if row['split'] != 'train' or row['category'] != '05Noise':
            continue
        if row['source_group'] in blocked_sources or row['gt_sha256'] in blocked_targets:
            raise RuntimeError('training / held-out source or target overlap')
        if min(int(row['width']), int(row['height'])) < 512:
            continue  # no resizing; excluded count recorded in receipt
        descriptor(row)
        groups.setdefault(row['source_group'], []).append(row)
    keys = sorted(groups, key=lambda k: digest(seed, 'source', k))[:count]
    if len(keys) != count:
        raise RuntimeError('insufficient eligible training sources')
    return [min(groups[k], key=lambda r: digest(seed, 'record', r['record_id'])) for k in keys], groups


def crop_xy(row, size=512, seed=2026091602):
    width, height = int(row['width']), int(row['height'])
    if min(width, height) < size:
        raise ValueError('no resizing allowed')
    data = bytes.fromhex(digest(seed, 'full_development_crop', row['record_id']))
    return [int.from_bytes(data[8:16], 'big') % (width-size+1),
            int.from_bytes(data[:8], 'big') % (height-size+1)]


def preflight():
    spec = json.loads(CONFIG.read_text())
    plan = json.loads(bound(spec['plan'], spec['plan_sha256']).read_text())
    if (spec['mode'] != 'training_sources_only_engineering_qualification' or spec['records'] != 32
            or spec['steps'] != 25 or spec['world_size'] != 8 or spec['global_batch'] != 32
            or spec['cache_crop'] != 512 or spec['train_crop'] != 256
            or spec['auto_formal'] or spec['qualification_may_seed_formal']
            or spec['sampling_guidance_implemented'] or spec['internal_residual_init'] != .1):
        raise RuntimeError('qualification scope changed')
    artifacts = {}
    for field in ('manifest', 'geometry_manifest', 'flow', 'ae', 'wrong_expert'):
        print('VERIFY ' + field, flush=True)
        bound(spec[field], spec[field + '_sha256'])
        artifacts[spec[field]] = spec[field + '_sha256']
    for field in ('baseline_config', 'baseline_checkpoint'):
        bound(plan[field], plan[field + '_sha256'])
        artifacts[plan[field]] = plan[field + '_sha256']
    qwen_files = ('config.json', 'generation_config.json', 'model.safetensors.index.json',
                  'model-00001-of-00002.safetensors', 'model-00002-of-00002.safetensors',
                  'tokenizer.json', 'tokenizer_config.json', 'merges.txt', 'vocab.json')
    for name in qwen_files:
        filename = str(Path(spec['qwen']) / name)
        artifacts[filename] = sha(path(filename))
    alex = 'models/torch/hub/checkpoints/alexnet-owt-7be5be79.pth'
    artifacts[alex] = sha(path(alex))
    # Join dimensions from the pair manifest: the VLM export has no width/height.
    with path(spec['geometry_manifest']).open(newline='') as handle:
        geometry = {r['record_id']: r for r in csv.DictReader(handle)}
    rows = []
    with path(spec['manifest']).open(newline='') as handle:
        for row in csv.DictReader(handle):
            other = geometry.pop(row['record_id'])
            for key in ('split', 'source_group', 'category', 'lq_path', 'gt_path', 'lq_sha256', 'gt_sha256'):
                if row[key] != other[key]:
                    raise RuntimeError('VLM/pairs metadata mismatch: ' + key)
            # Retain metadata only. Do not open any held-out image.
            rows.append({k: row[k] for k in ('record_id', 'split', 'source_group', 'category',
                         'lq_path', 'gt_path', 'lq_sha256', 'gt_sha256', 'qwen_levels_json')} |
                        {'width': other['width'], 'height': other['height']})
    if geometry or len(rows) != 685495:
        raise RuntimeError('full manifest cardinality mismatch')
    selected, groups = select_training(rows, count=spec['records'], seed=spec['selection_seed'])
    noise_train = [r for r in rows if r['category'] == '05Noise' and r['split'] == 'train']
    if len(noise_train) != 54585 or len({r['source_group'] for r in noise_train}) != 1755:
        raise RuntimeError('Noise training cardinality mismatch')
    for row in selected:
        # Path resolution / existence only; image hashes checked before actual decoding.
        for field in ('lq_path', 'gt_path'):
            if not path(row[field]).is_file():
                raise FileNotFoundError(row[field])
    return dict(spec=spec, plan=plan, config_sha256=sha(CONFIG), source_sha256=source_binding(),
                artifacts=artifacts, selected=selected, eligible_records=sum(map(len, groups.values())),
                eligible_sources=len(groups), protected_pixels_accessed=False, images_opened=0)


def load_receipt(output):
    receipt = json.loads((output / 'preflight.json').read_text())
    if receipt['config_sha256'] != sha(CONFIG) or receipt['source_sha256'] != source_binding():
        raise RuntimeError('source/config changed after preflight')
    return receipt


def validate_cache_metadata(meta, row, receipt):
    if row['split'] != 'train' or row['category'] != '05Noise':
        raise RuntimeError('only training Noise pixels allowed')
    if meta['row'] != row or meta['config_sha256'] != receipt['config_sha256']:
        raise RuntimeError('cache record/contract changed')
    if meta['crop_xy'] != crop_xy(row) or meta['baseline_checkpoint_sha256'] != receipt['plan']['baseline_checkpoint_sha256']:
        raise RuntimeError('cache crop/checkpoint mismatch')
    if meta['wrong_expert_sha256'] != receipt['spec']['wrong_expert_sha256']:
        raise RuntimeError('wrong expert identity mismatch')
    if meta['prompt_receipt_sha256'] != receipt['prompt_receipt_sha256']:
        raise RuntimeError('prompt cache identity mismatch')
