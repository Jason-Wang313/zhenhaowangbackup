"""Explicit, hash-bound transition from completed step1000; not a relaxed resume."""
from pathlib import Path

from smoke.token_residual_moe_v1 import bound_json, sha, load_run_config, FORMAL_CONFIG


def verify_continuation(root, cfg, *, artifacts=True, sources=True, tensors=False):
    if 'continuation' not in cfg:
        return None
    root = Path(root)
    c = cfg['continuation']
    if set(c) != {'previous_config', 'previous_config_sha256', 'report', 'report_sha256',
                  'receipt', 'receipt_sha256'} or c['previous_config'] != FORMAL_CONFIG:
        raise RuntimeError('unsupported continuation lineage')
    bound_json(root, c['previous_config'], c['previous_config_sha256'])
    prior = load_run_config(root, root / c['previous_config'])
    allowed = {'steps', 'milestones', 'full_evaluation_steps', 'output_root',
               'wall_hours_estimate', 'allocated_gpu_hours_estimate',
               'fresh_optimizer', 'restore_old_optimizer'}
    if any(cfg[k] != value for k, value in prior.items() if k not in allowed):
        raise RuntimeError('continuation changed scientific recipe')
    report = bound_json(root, c['report'], c['report_sha256'])
    result = report['full_result']
    binding = result['binding']
    if not (result['status'] == 'complete' and result['mode'] == 'formal'
            and result['steps'] == 1000 and report['formal_run_completed']
            and report['final_checkpoint_sha_verified'] and report['baseline_sha_verified']
            and not result['protected_pixels_accessed']
            and binding['config_sha256'] == c['previous_config_sha256']):
        raise RuntimeError('requires the completed audited formal step1000')
    parent_dir = Path(c['receipt']).parent
    if (report['artifact_hashes'][c['receipt']]['sha256'] != c['receipt_sha256']
            or len([r for r in result['checkpoints'] if r['step'] == 1000]) != 1):
        raise RuntimeError('wrong parent receipt identity')
    if sources:
        for name, expected in binding['source_sha256'].items():
            if name not in {'smoke/token_residual_moe_v1.py', 'smoke/train_foundir_token_residual_moe_v1.py'}:
                if sha(root / name) != expected:
                    raise RuntimeError('parent scientific implementation changed: ' + name)
    if artifacts:
        for filename in ('launch.json', 'result.json', 'baseline.json', 'step_001000.resume.json'):
            name = str(parent_dir / filename)
            if sha(root / name) != report['artifact_hashes'][name]['sha256']:
                raise RuntimeError('parent evidence changed: ' + name)
        receipt = bound_json(root, c['receipt'], c['receipt_sha256'])
        expected = next(r for r in result['checkpoints'] if r['step'] == 1000)
        if receipt != expected:
            raise RuntimeError('parent receipt differs from audited state')
        if tensors:
            for key in ('weights', 'optimizer'):
                if Path(receipt[key]).name != receipt[key] or sha(root / parent_dir / receipt[key]) != receipt[key + '_sha256']:
                    raise RuntimeError('parent checkpoint bytes changed: ' + key)
    return {'binding': binding, 'receipt': c['receipt'], 'report_sha256': c['report_sha256'],
            'parent_metrics': report['metrics']['1000'], 'start_step': 1000}
