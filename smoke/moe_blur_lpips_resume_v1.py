"""Explicit migration provenance; do not weaken the old same-node comparator."""
from __future__ import annotations

import json
from pathlib import Path

from smoke.moe_blur_lpips_continuation_v1 import ROOT, sha, write_once

CONFIG = ROOT / 'configs/foundir_moe_blur_lpips_resume_v1.json'
CONFIG_SHA256 = '39afd8616ada4b8a22cac7e41d7453e999ce81f3e5cab161693cf26e8f462f4d'


def read_migration():
    if sha(CONFIG) != CONFIG_SHA256:
        raise RuntimeError('frozen migration contract changed')
    return json.loads(CONFIG.read_text())


def verify_migration(sources, *, artifacts=True):
    cfg = read_migration()
    for path, digest in (('scientific_config', 'scientific_config_sha256'),
                         ('resume_receipt', 'resume_receipt_sha256'),
                         ('old_launch', 'old_launch_sha256'), ('control_result', 'control_result_sha256')):
        if sha(ROOT / cfg[path]) != cfg[digest]:
            raise RuntimeError('migration binding changed: ' + path)
    receipt = json.loads((ROOT / cfg['resume_receipt']).read_text())
    if not (receipt['arm'] == 'blur_lpips4x' and receipt['mode'] == 'formal'
            and receipt['step'] == 10500 and receipt['next_sample_position'] == 336000
            and receipt['world_size'] == 8 and receipt['source_sha256'] == sources
            and receipt['config_sha256'] == cfg['scientific_config_sha256']):
        raise RuntimeError('resume state does not match the interrupted treatment')
    for kind in ('weights', 'optimizer'):
        if (Path(receipt[kind]).name != receipt[kind]
                or receipt[kind + '_sha256'] != cfg[kind + '_sha256']):
            raise RuntimeError('unsafe or mismatched checkpoint artifact')
        if artifacts and sha((ROOT / cfg['resume_receipt']).parent / receipt[kind]) != cfg[kind + '_sha256']:
            raise RuntimeError('resume artifact bytes changed: ' + kind)
    old = json.loads((ROOT / cfg['old_launch']).read_text())
    if old['source_sha256'] != sources or old['config_sha256'] != cfg['scientific_config_sha256']:
        raise RuntimeError('old launch differs from the checkpoint')
    if (ROOT / cfg['old_launch']).with_name('result.json').exists():
        raise RuntimeError('old treatment has completed; do not duplicate it')
    return cfg, receipt, old


def trace_event(trace, position, task_id, record_id):
    trace.update(json.dumps([position, task_id, record_id], separators=(',', ':')).encode() + b'\n')


def reconstruct_trace_prefix(dataset, trace, stop_step):
    """Metadata only; original checkpoints did not save incremental hash state."""
    from lowlight_restoration.rain_blur_moe_data import task_position
    for index in range(10000 * 4, stop_step * 4):
        task_id, position = task_position(index, dataset.rank, dataset.world)
        task = dataset.tasks[task_id]
        row = dataset.training[task][dataset.schedules[task].index(position)]
        trace_event(trace, index * 8 + dataset.rank, task_id, row['record_id'])


def compare_migrated(control, resumed):
    from smoke.compare_foundir_rain_blur_joint_vlm_moe_v2 import paired_summary
    if not (control['status'] == resumed['status'] == 'complete'
            and control['engineering_pass'] and resumed['engineering_pass']
            and control['variant'] == 'control' and resumed['variant'] == 'blur_lpips4x'
            and control['steps'] == resumed['steps'] == 11000
            and resumed['start_step'] == 10500 and resumed['optimizer_steps_this_run'] == 500
            and resumed['this_launch_presentations'] == 16000
            and not control['protected_pixels_accessed'] and not resumed['protected_pixels_accessed']):
        raise ValueError('requires completed fixed-final control and migrated treatment')
    for key in ('config_sha256', 'data', 'parent_receipt_sha256', 'parent_weights_sha256',
                'parent_optimizer_sha256', 'sample_trace_sha256_by_rank', 'optimizer_group_settings'):
        if control[key] != resumed[key]:
            raise ValueError('scientific protocol mismatch: ' + key)
    if control['source_sha256'] != resumed['original_pair_source_sha256']:
        raise ValueError('original code lineage mismatch')
    summaries = {}
    for split in ('rain', 'blur_full'):
        a, b = control['final_metrics'][split], resumed['final_metrics'][split]
        if a['step'] != 11000 or b['step'] != 11000 or a['route'] != 'learned' or b['route'] != 'learned':
            raise ValueError('wrong step or inference rule')
        raw = paired_summary(a, b)
        names = {'image_only': 'control', 'vlm_assisted': 'blur_lpips4x',
                 'VLM_minus_image_PSNR': 'treatment_minus_control_PSNR',
                 'VLM_minus_image_LPIPS': 'treatment_minus_control_LPIPS'}
        summaries[split] = {names.get(k, k): v for k, v in raw.items()}
    rain, blur = summaries['rain'], summaries['blur_full']
    r, b = rain['blur_lpips4x'], blur['blur_lpips4x']
    return {'status': 'migration_qualified_final_comparison', 'per_task': summaries,
        'absolute_joint_gate_pass': {v: rain[v]['psnr'] >= 34.2 and rain[v]['lpips'] <= .08
                                      and blur[v]['psnr'] >= 25.9 and blur[v]['lpips'] <= .18
                                     for v in ('control', 'blur_lpips4x')},
        'predeclared_repair_signal': bool(blur['LPIPS_source_bootstrap_95pct'][1] < 0
                                         and b['psnr'] >= 25.9 and r['psnr'] >= 34.2 and r['lpips'] <= .08),
        'same_physical_node_for_full_pair': False,
        'same_GPU_model_after_migration': control['gpu_names'] == resumed['gpu_names'],
        'hardware_history': resumed['hardware_history'],
        'causal_caveat': 'node/device migration and execution interruption are additional differences; not the original strict same-node ablation',
        'auto_extend': False, 'VLM_advantage_or_novelty_established': False,
        'protected_pixels_accessed': False}


def compare_output(output):
    cfg = read_migration()
    path = ROOT / cfg['control_result']
    if sha(path) != cfg['control_result_sha256']:
        raise RuntimeError('control result changed')
    result = json.loads((output / 'result.json').read_text())
    if result['migration_config_sha256'] != CONFIG_SHA256:
        raise RuntimeError('migration result belongs to another contract')
    for name, expected in result['source_sha256'].items():
        if sha(ROOT / name) != expected:
            raise RuntimeError('migration source changed: ' + name)
    final = result['checkpoints'][-1]
    for kind in ('weights', 'optimizer'):
        if sha(output / final[kind]) != final[kind + '_sha256']:
            raise RuntimeError('final checkpoint changed')
    comparison = compare_migrated(json.loads(path.read_text()), result)
    write_once(output / 'comparison.json', comparison)
    print('MOE_MIGRATION_COMPARISON ' + json.dumps(comparison), flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compare-output', type=Path, required=True)
    compare_output(parser.parse_args().compare_output)
