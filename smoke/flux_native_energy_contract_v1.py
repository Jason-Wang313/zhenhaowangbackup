"""New FLUX-only contract. Reads historical metadata, never old energy code/weights."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT/'configs/foundir_flux_native_energy_v1.json'
ORDER = ('blur','noise','rain','haze','low_light')
FILES = ('configs/foundir_flux_native_energy_v1.json',
         'lowlight_restoration/flux_native_energy.py', 'lowlight_restoration/lora.py',
         'lowlight_restoration/scalar_residual.py','lowlight_restoration/flux2_trajectory_endpoint.py',
         'smoke/flux_native_energy_contract_v1.py', 'smoke/flux_native_energy_runtime_v1.py',
         'smoke/train_foundir_flux_native_energy_v1.py',
         'smoke/run_foundir_flux_native_energy_v1_gpu.sh',
         'smoke/start_foundir_flux_native_energy_v1_tmux.sh','smoke/tmux_launcher_lib.sh')


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle,'sha256').hexdigest()


def path(value):
    result = (ROOT/value).resolve()
    result.relative_to(ROOT)
    return result


def bound(value, expected):
    target = path(value)
    if sha(target) != expected:
        raise RuntimeError('artifact SHA mismatch: '+str(value))
    return target


def descriptor(row):
    levels = json.loads(row['qwen_levels_json'])
    if set(levels) != set(ORDER) or any(type(v) is not int or not 1<=v<=5 for v in levels.values()):
        raise ValueError('invalid VLM descriptor')
    return [(levels[k]-1)/4 for k in ORDER]


def binding():
    names = list(FILES) + ['lowlight_restoration/'+n+'.py' for n in ('__init__','data','flow')]
    names += [str(p.relative_to(ROOT)) for p in sorted((ROOT/'third_party/flux2/src/flux2').glob('*.py'))]
    if not any(n.endswith('flux2/model.py') for n in names):
        raise FileNotFoundError('native FLUX source missing')
    return {n:sha(path(n)) for n in names}


def load_spec():
    spec=json.loads(CONFIG.read_text())
    if (spec['mode']!='qualification_only' or spec['steps']!=25 or spec['updates']!=3
        or spec['crop']!=512 or spec['global_batch']!=8 or spec['allowed_gpu_counts']!=[4,8]
        or spec['training_indices']!=list(range(24)) or spec['probe_indices']!=list(range(8))
        or spec['auto_trainfit'] or spec['qualification_may_seed_trainfit']
        or spec['cnn_or_restormer_allowed'] or spec['protected_pixels_allowed']):
        raise ValueError('qualification scope changed')
    return spec


def source_metadata():
    spec=load_spec()
    previous=json.loads(bound(spec['source_receipt'],spec['source_receipt_sha256']).read_text())
    selected=previous['selected'][:24]
    if len(selected)!=24:
        raise ValueError('missing training identities')
    for item in selected:
        row=item['row']
        if item['role']!='train' or row['split']!='train' or row['category']!='05Noise':
            raise RuntimeError('nontraining source refused before pixel access')
        if previous['source_roles'].get(row['source_group'])!='train':
            raise RuntimeError('reserved source refused')
        if not row['lq_path'].startswith('data/foundir/MillionIRData/Train/05Noise/LQ/'):
            raise RuntimeError('unexpected LQ asset location')
        if not row['gt_path'].startswith('data/foundir/MillionIRData/Train/05Noise/GT/'):
            raise RuntimeError('unexpected GT asset location')
        descriptor(row)
        x,y=item['crop_xy']
        if x<0 or y<0 or x+512>int(row['width']) or y+512>int(row['height']):
            raise ValueError('crop outside source')
    for key in ('source_group','record_id','lq_sha256','gt_sha256'):
        if len({v['row'][key] for v in selected})!=24:
            raise ValueError('duplicate training identity: '+key)
    return spec,previous,selected


def preflight(world):
    spec,previous,selected=source_metadata()
    if world not in spec['allowed_gpu_counts']:
        raise ValueError('one 4/8 GPU node required')
    assets={}
    for field in ('flow','ae','wrong_expert'):
        name=previous['base_spec'][field]
        assets[name]=previous['base_spec'][field+'_sha256']
    name=previous['plan']['baseline_checkpoint']
    assets[name]=previous['plan']['baseline_checkpoint_sha256']
    for name,expected in spec['prompt_hashes'].items():
        assets[str(Path(previous['parent_cache'])/name)]=expected
    alex='models/torch/hub/checkpoints/alexnet-owt-7be5be79.pth'
    if alex not in previous['artifacts']:
        raise ValueError('missing frozen LPIPS feature checkpoint binding')
    assets[alex]=previous['artifacts'][alex]
    for name,expected in assets.items():
        print('NATIVE_ENERGY_VERIFY '+name,flush=True)
        bound(name,expected)
    receipt=dict(spec=spec,selected=selected,assets=assets,base_spec=previous['base_spec'],
        plan=previous['plan'],prompt_root=previous['parent_cache'],world_size=world,
        sources=binding(),config_sha256=sha(CONFIG),protected_pixels_accessed=False,
        cnn_checkpoint_loaded=False,qualification_checkpoint_loaded=False)
    receipt['protocol_sha256']=hashlib.sha256(json.dumps(receipt,sort_keys=True).encode()).hexdigest()
    return receipt


def run_path(run_id):
    if not re.fullmatch(r'[A-Za-z0-9_-]+',run_id):
        raise ValueError('invalid run ID')
    return path(load_spec()['output_root'])/run_id
