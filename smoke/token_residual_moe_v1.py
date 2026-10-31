"""Frozen data/initialization contract for the right-diagram MoE qualification."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path

import torch
from PIL import Image
from safetensors import safe_open
from safetensors.torch import save_file

from lowlight_restoration.rain_blur_moe_data import load_protocol, digest
from lowlight_restoration.rain_full_data import RainSourceSchedule, crop_pair

CONFIG = 'configs/foundir_token_residual_moe_v1.json'
TASKS = ('rain', 'blur')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def bound_json(root, path, expected):
    if sha(root / path) != expected:
        raise RuntimeError('artifact identity changed: ' + path)
    return json.loads((root / path).read_text())


def write_once(path, payload):
    """Publish atomically without replacing prior results or partial evidence."""
    path = Path(path)
    raw = (json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != raw:
            raise RuntimeError('refusing to replace ' + str(path))
        return
    import uuid
    temp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.partial')
    with temp.open('xb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temp, path)
    except FileExistsError:
        if path.read_bytes() != raw:
            raise RuntimeError('concurrent output conflict')
    finally:
        temp.unlink()


def contract(root, config_path=None):
    root = Path(root)
    # Optional separate config path is only for read-only CPU migration audits.
    cfg = json.loads(Path(config_path or root / CONFIG).read_text())
    if (cfg['steps'], cfg['global_batch'], cfg['world_size'], cfg['nodes'], cfg['gpus_per_node'],
            cfg['feature_dim'], cfg['sampling_start']) != (25, 32, 8, 4, 2, 3584, 352000):
        raise RuntimeError('qualification/topology changed')
    if (cfg['crop'] != 512 or cfg['vlm_scale'] != 1.0 or not cfg['vlm_output_zero_init']
            or cfg['experts_per_block'] != 2 or cfg['ffn_blocks'] != list(range(5))
            or cfg['automatic_formal'] or cfg['qualification_may_seed_formal']
            or cfg['restore_old_optimizer'] or not cfg['fresh_optimizer']):
        raise RuntimeError('right-diagram initialization/scope changed')
    parent = bound_json(root, cfg['parent_config'], cfg['parent_config_sha256'])
    pc = bound_json(root, parent['parent_config'], parent['parent_config_sha256'])
    base = bound_json(root, cfg['base_config'], cfg['base_config_sha256'])
    qwen = bound_json(root, cfg['qwen_config'], cfg['qwen_config_sha256'])
    qcfg = bound_json(root, qwen['qwen']['training_config'], qwen['qwen']['training_config_sha256'])
    receipt = bound_json(root, cfg['initial_receipt'], cfg['initial_receipt_sha256'])
    if receipt['step'] != 11000 or receipt['weights_sha256'] != cfg['initial_weights_sha256']:
        raise RuntimeError('wrong image-only starting checkpoint')
    for path, expected in ((base['data']['manifest'], base['data']['manifest_sha256']),
                           (pc['source_manifest'], pc['source_manifest_sha256'])):
        if sha(root / path) != expected:
            raise RuntimeError('data manifest changed')
    with (root / base['data']['manifest']).open(newline='') as handle:
        rain_rows = list(csv.DictReader(handle))
    rain_train = [r for r in rain_rows if r['role'] == 'direct_endpoint_train']
    rain_dev = [r for r in rain_rows if r['role'] == 'direct_endpoint_validation']
    if len(rain_train) != 38334 or len(rain_dev) != 100:
        raise RuntimeError('Rain split cardinality changed')
    training, development, data = load_protocol(root, rain_train, rain_dev, pc['source_manifest'])
    if data != pc['data_contract']:
        raise RuntimeError('parent source-disjoint data protocol changed')
    return cfg, parent, base, qwen, qcfg, training, development, data


def verify_assets(root, cfg, base, qwen):
    assets = [(cfg['initial_weights'], cfg['initial_weights_sha256']),
              (qwen['qwen']['adapter'], qwen['qwen']['adapter_sha256'])]
    for key in ('flow_weights', 'autoencoder_weights', 'prompt_context_cache'):
        assets.append((base['model'][key], base['model'][key + '_sha256']))
    for path, expected in assets:
        if sha(root / path) != expected:
            raise RuntimeError('model asset mismatch: ' + path)
    qcfg = bound_json(root, qwen['qwen']['training_config'], qwen['qwen']['training_config_sha256'])
    inventory = bound_json(root, qcfg['model']['inventory_config'], qcfg['model']['inventory_config_sha256'])
    if inventory['model']['local_path'] != qcfg['model']['local_path']:
        raise RuntimeError('Qwen base-model location mismatch')
    model_root = (root / qcfg['model']['local_path']).resolve()
    if root.resolve() not in model_root.parents:
        raise RuntimeError('Qwen loader rejects model symlinks outside the active workspace; relocate explicitly')
    for name, expected in inventory['model']['expected_files'].items():
        path = model_root / name
        if path.stat().st_size != expected['bytes'] or sha(path) != expected['sha256']:
            raise RuntimeError('frozen Qwen base/processor identity changed: ' + name)


class Samples:
    def __init__(self, root, cfg, training, development):
        self.root, self.cfg, self.training = Path(root), cfg, training
        self.schedules = {task: RainSourceSchedule(training[task], cfg['seed'] + i)
                          for i, task in enumerate(TASKS)}
        self.development = {'rain': development['rain'], 'blur_preview': development['blur_preview']}
        self.verified = {}

    def train_spec(self, offset):
        if not 0 <= offset < self.cfg['steps'] * 32:
            raise ValueError('training sample outside frozen run')
        absolute = self.cfg['sampling_start'] + offset
        global_micro, rank = divmod(absolute, 8)
        task_id = global_micro % 2
        position = (global_micro // 2) * 8 + rank
        task = TASKS[task_id]
        row = self.training[task][self.schedules[task].index(position)]
        return {'key': f'train_{absolute:09d}', 'kind': 'train', 'task_id': task_id,
                'position': position, 'seed': self.cfg['seed'] + task_id, 'row': row}

    def all_specs(self):
        specs = [self.train_spec(i) for i in range(self.cfg['steps'] * 32)]
        for split, rows in self.development.items():
            for row in rows:
                if row['split'] != 'validation':
                    raise RuntimeError('only frozen development rows may be evaluated')
                specs.append({'key': f'{split}_{row["record_id"]}', 'kind': 'validation',
                              'split': split, 'row': row})
        return specs

    def pair(self, spec):
        row = spec['row']
        if spec['kind'] == 'train':
            return crop_pair(self.root, row, crop_size=512, position=spec['position'],
                             seed=spec['seed'], verified=self.verified)
        # Reproduce the parent's exact development uint8 crop, no resizing.
        import numpy as np
        result = []
        x, y = int(row['crop_x']), int(row['crop_y'])
        for side in ('lq', 'gt'):
            path = self.root / row[side + '_path']
            if sha(path) != row[side + '_sha256']:
                raise RuntimeError('development image bytes changed')
            with Image.open(path) as image:
                if x < 0 or y < 0 or x + 512 > image.width or y + 512 > image.height:
                    raise RuntimeError('invalid frozen development crop')
                array = np.array(image.convert('RGB').crop((x, y, x + 512, y + 512)), copy=True)
            result.append(torch.from_numpy(array).permute(2, 0, 1).float() / 127.5 - 1)
        return tuple(result)


def feature_binding(spec, config_sha, qwen):
    # Includes exact crop position/augmentation or development coordinates,
    # but GT bytes are never an encoder input.
    return {'config_sha256': config_sha, 'sample_sha256': digest(spec),
            'adapter_sha256': qwen['qwen']['adapter_sha256'],
            'pooling': 'visual_merger_tokens_mean', 'dimension': '3584'}


def read_feature(path, binding):
    if not path.exists():
        return None
    with safe_open(path, framework='pt', device='cpu') as handle:
        if handle.metadata() != binding or set(handle.keys()) != {'feature'}:
            raise RuntimeError('token cache binding mismatch')
        feature = handle.get_tensor('feature')
    if feature.shape != (3584,) or feature.dtype != torch.float32 or not torch.isfinite(feature).all():
        raise RuntimeError('invalid pooled token feature')
    return feature


def save_feature(path, feature, binding):
    path.parent.mkdir(parents=True, exist_ok=True)
    import uuid
    temp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.partial')
    save_file({'feature': feature.cpu().float().contiguous()}, str(temp), metadata=binding)
    try:
        os.link(temp, path)
    except FileExistsError:
        if not torch.equal(read_feature(path, binding), feature.cpu().float()):
            raise RuntimeError('concurrent token cache mismatch')
    finally:
        temp.unlink()


def source_hashes(root):
    # Bind scientific implementation, feature extraction and native FLUX code.
    paths = [CONFIG, 'lowlight_restoration/token_residual_moe.py',
        'smoke/token_residual_moe_v1.py', 'smoke/run_foundir_token_residual_moe_v1_4x2.sh',
        'smoke/train_foundir_token_residual_moe_v1.py',
        'lowlight_restoration/joint_vlm_moe.py', 'lowlight_restoration/joint_moe_state.py',
        'lowlight_restoration/rain_blur_image_moe.py', 'lowlight_restoration/rain_blur_moe_data.py',
        'lowlight_restoration/rain_full_data.py', 'lowlight_restoration/lora.py',
        'lowlight_restoration/flux2_endpoint_precision.py', 'lowlight_restoration/blur_single_endpoint.py',
        'lowlight_restoration/qwen_lora.py', 'lowlight_restoration/qwen_five_class_gate.py',
        'smoke/train_qwen_multidegradation_lora_ddp.py', 'smoke/qwen_multidegradation_lora_preflight.py',
        'smoke/train_foundir_rain_flux2_precision_fit_v1_ddp.py',
        'lowlight_restoration/pixel_refinement.py',
        'third_party/flux2/src/flux2/model.py',
        'third_party/flux2/src/flux2/autoencoder.py', 'third_party/flux2/src/flux2/sampling.py',
        'third_party/flux2/src/flux2/util.py']
    return {p: sha(root / p) for p in paths}
