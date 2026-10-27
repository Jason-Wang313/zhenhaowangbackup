"""Shared bounded execution contract for the two authorized retry jobs."""
from __future__ import annotations

from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'configs/foundir_dual_retry_execution_v1.json'


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def bound_json(relative, expected):
    path = ROOT / relative
    if sha(path) != expected:
        raise RuntimeError(f'bound JSON changed: {relative}')
    return json.loads(path.read_text())


def write_once(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def sources(extra=()):
    # Bound import closure plus launchers/configs. Historical source unchanged.
    names = [
        'smoke/dual_retry_runtime.py', 'smoke/run_foundir_dual_retry_v1_8gpu.sh',
        'smoke/start_foundir_dual_retry_v1_tmux.sh',
        'smoke/diagnose_foundir_joint_moe_train_gradients_v1.py',
        'smoke/train_foundir_novel_ppt_energy_v1_ddp.py',
        'lowlight_restoration/novel_ppt_energy.py',
        'lowlight_restoration/vlm_energy_restoration.py',
        'lowlight_restoration/proposal_relative_energy.py',
        'lowlight_restoration/proposal_relative_energy_v2.py',
        'lowlight_restoration/proposal_residual_basis.py',
        'lowlight_restoration/task_native_restormer.py',
        'lowlight_restoration/complete_candidate_energy.py',
        'lowlight_restoration/flux_prompt_cache.py',
        'smoke/train_foundir_proposal_relative_energy_noise_v1_ddp.py',
        'smoke/train_foundir_proposal_relative_energy_noise_v2_ddp.py',
        'smoke/train_foundir_complete_candidate_energy_noise_v1_ddp.py',
        'smoke/evaluate_foundir_complete_energy_candidate_oracle_noise_v1_ddp.py',
        'smoke/cure_nccl_rehearsal_v1.py',
        'smoke/tmux_launcher_lib.sh',
    ]
    return {name: sha(ROOT / name) for name in sorted(set(names) | set(extra))}


def initialize(branch, run_id):
    import torch
    import torch.distributed as dist
    config = json.loads(CONFIG.read_text())
    if branch not in ('moe', 'ppt') or not re.fullmatch(r'[A-Za-z0-9_-]+', run_id):
        raise ValueError('invalid branch or fresh run ID')
    rank = int(os.environ['RANK']); local = int(os.environ['LOCAL_RANK'])
    if (int(os.environ['WORLD_SIZE']) != 8 or int(os.environ['LOCAL_WORLD_SIZE']) != 8
            or torch.cuda.device_count() != 8):
        raise RuntimeError('requires exactly one eight-GPU node')
    torch.cuda.set_device(local)
    device = torch.device('cuda', local)
    prop = torch.cuda.get_device_properties(device)
    if prop.total_memory < 75*1024**3 or not any(s in prop.name for s in ('H100', 'H200')):
        raise RuntimeError('requires H100/H200 >=75 GiB')
    dist.init_process_group('nccl', device_id=device, timeout=timedelta(minutes=30))
    names = [None]*8; dist.all_gather_object(names, prop.name)
    if len(set(names)) != 1:
        raise RuntimeError('homogeneous GPUs required')
    output = ROOT / config['runtime']['output_root'] / branch / run_id
    error = [None]
    if rank == 0:
        try:
            output.mkdir(parents=True, exist_ok=False)
        except Exception as exc:
            error[0] = str(exc)
    dist.broadcast_object_list(error, src=0)
    if error[0]:
        raise RuntimeError(error[0])
    return config, rank, device, output, names


def source_disjoint_training_probe(rows, count, seed):
    """One record per train source; no scores or pixels used for selection."""
    if any(row['split'] != 'train' for row in rows):
        raise ValueError('training-only probe received another split')
    ordered = sorted(rows, key=lambda r: hashlib.sha256(f"{seed}:{r['source_group']}:{r['record_id']}".encode()).hexdigest())
    selected, used = [], set()
    for row in ordered:
        if row['source_group'] not in used:
            selected.append(row); used.add(row['source_group'])
            if len(selected) == count:
                return selected
    raise ValueError('not enough distinct training sources')
