"""Isolated single-change contract and paired analysis; no old sources mutated."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import socket
import subprocess

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'configs/foundir_moe_blur_lpips_continuation_v1.json'
CONFIG_SHA256 = 'f9840280eabc089567168ab55d7ce8f6431772f6fcb94ddbd208a3ff16449d20'
VARIANTS = ('control', 'blur_lpips4x')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def read_config():
    if sha(CONFIG) != CONFIG_SHA256:
        raise RuntimeError('frozen continuation configuration changed')
    return json.loads(CONFIG.read_text())


def write_once(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')


def loss_weights(original, variant, task_id):
    if variant not in VARIANTS or task_id not in (0, 1):
        raise ValueError('unknown variant/task')
    if original.get('lpips') != .05:
        raise RuntimeError('parent LPIPS coefficient changed')
    weights = dict(original)
    if variant == 'blur_lpips4x' and task_id == 1:
        weights['lpips'] = .20
    return weights


def hardware_identity():
    # UUIDs distinguish replacement nodes even if names/models happen to match.
    uuids = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True).splitlines()
    uuids = [v.strip() for v in uuids if v.strip()]
    if len(uuids) != 8 or len(set(uuids)) != 8:
        raise RuntimeError('expected eight unique physical GPU UUIDs')
    return {'hostname': socket.gethostname(), 'gpu_uuids': uuids}


def verify_parent(config, sources, *, verify_artifacts=True):
    path = ROOT / config['parent_receipt']
    if sha(path) != config['parent_receipt_sha256']:
        raise RuntimeError('parent receipt changed')
    receipt = json.loads(path.read_text())
    if not (receipt['mode'] == 'formal' and receipt['arm'] == 'image_only'
            and receipt['step'] == 10000 and receipt['next_sample_position'] == 320000
            and receipt['world_size'] == 8 and receipt['source_sha256'] == sources
            and receipt['config_sha256'] == config['parent_config_sha256']):
        raise RuntimeError('parent state/source mismatch')
    for key in ('weights', 'optimizer'):
        name = receipt[key]
        if Path(name).name != name or receipt[key + '_sha256'] != config['parent_' + key + '_sha256']:
            raise RuntimeError('invalid parent artifact binding')
        if verify_artifacts and sha(path.parent / name) != receipt[key + '_sha256']:
            raise RuntimeError('parent artifact bytes changed: ' + key)
    return receipt


def verify_optimizer_groups(optimizer, expected):
    """Restore Adam moments without silently changing any per-group settings."""
    actual = optimizer.param_groups
    if len(actual) != len(expected):
        raise RuntimeError('Adam parameter groups changed')
    for group, ref in zip(actual, expected, strict=True):
        for key in ('name', 'lr', 'betas', 'eps', 'weight_decay', 'amsgrad', 'fused'):
            if group.get(key) != ref.get(key):
                raise RuntimeError('restored Adam setting differs: ' + key)


def compare_results(control, treatment):
    # Reuse existing row-paired/source-bootstrap calculation, relabel the arms.
    from smoke.compare_foundir_rain_blur_joint_vlm_moe_v2 import paired_summary
    for result, variant in zip((control, treatment), VARIANTS, strict=True):
        if not (result['status'] == 'complete' and result['variant'] == variant
                and result['engineering_pass'] and result['start_step'] == 10000
                and result['steps'] == 11000 and result['optimizer_steps_this_run'] == 1000
                and result['this_launch_presentations'] == 32000
                and not result['protected_pixels_accessed']):
            raise ValueError('requires both complete, qualified continuation arms')
    for key in ('config_sha256', 'source_sha256', 'data', 'hardware', 'gpu_names',
                'parent_receipt_sha256', 'parent_weights_sha256', 'parent_optimizer_sha256',
                'sample_trace_sha256_by_rank', 'optimizer_group_settings'):
        if control[key] != treatment[key]:
            raise ValueError('unmatched continuation: ' + key)
    # Ensure starting restored outputs agree, not just the checkpoint filename.
    for split in ('rain', 'blur_preview'):
        start = paired_summary(control['initial_metrics'][split], treatment['initial_metrics'][split])
        if abs(start['VLM_minus_image_PSNR']) > 1e-6 or abs(start['VLM_minus_image_LPIPS']) > 1e-6:
            raise ValueError('starting output mismatch')
        left = {r['record_id']: r for r in control['initial_metrics'][split]['rows']}
        for row in treatment['initial_metrics'][split]['rows']:
            if any(abs(row['output_' + m] - left[row['record_id']]['output_' + m]) > 1e-6
                   for m in ('psnr', 'lpips')):
                raise ValueError('per-record starting output mismatch')
    summaries = {}
    for split in ('rain', 'blur_full'):
        left, right = control['final_metrics'][split], treatment['final_metrics'][split]
        if left['step'] != 11000 or right['step'] != 11000 or left['route'] != 'learned' or right['route'] != 'learned':
            raise ValueError('only fixed final deployed outputs may decide the result')
        summary = paired_summary(left, right)
        renames = {'image_only': 'control', 'vlm_assisted': 'blur_lpips4x',
                   'VLM_minus_image_PSNR': 'treatment_minus_control_PSNR',
                   'VLM_minus_image_LPIPS': 'treatment_minus_control_LPIPS'}
        summaries[split] = {renames.get(k, k): v for k, v in summary.items()}
    rain, blur = summaries['rain'], summaries['blur_full']
    b = blur['blur_lpips4x']; r = rain['blur_lpips4x']
    absolute = {variant: (rain[variant]['psnr'] >= 34.2 and rain[variant]['lpips'] <= .08
                         and blur[variant]['psnr'] >= 25.9 and blur[variant]['lpips'] <= .18)
                for variant in VARIANTS}
    signal = (blur['LPIPS_source_bootstrap_95pct'][1] < 0 and b['psnr'] >= 25.9
              and r['psnr'] >= 34.2 and r['lpips'] <= .08)
    return {'status': 'matched_continuation_comparison', 'per_task': summaries,
            'absolute_joint_gate_pass': absolute, 'predeclared_repair_signal': bool(signal),
            'automatic_extension': False, 'novelty_or_VLM_advantage_established': False,
            'protected_pixels_accessed': False,
            'interpretation': 'single_seed_same_node_paired_loss_ablation_not_proof_of_unique_bottleneck'}


def compare_pair(pair):
    read_config()
    results = [json.loads((pair / v / 'result.json').read_text()) for v in VARIANTS]
    for result, variant in zip(results, VARIANTS, strict=True):
        if result['config_sha256'] != CONFIG_SHA256:
            raise RuntimeError('result config changed')
        for name, expected in result['source_sha256'].items():
            if sha(ROOT / name) != expected:
                raise RuntimeError('result source changed: ' + name)
        receipt = result['checkpoints'][-1]
        for key in ('weights', 'optimizer'):
            if sha(pair / variant / receipt[key]) != receipt[key + '_sha256']:
                raise RuntimeError('final checkpoint artifact changed')
    comparison = compare_results(*results)
    write_once(pair / 'comparison.json', comparison)
    print('MOE_LPIPS_PAIR_COMPARISON ' + json.dumps(comparison), flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compare-pair', type=Path, required=True)
    compare_pair(parser.parse_args().compare_pair)
