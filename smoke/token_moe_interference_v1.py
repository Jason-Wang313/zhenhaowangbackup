"""Read-only causal intervention: saved restoration weights x actual gate decisions.

Replaying per-image gates, rather than swapping router parameters, prevents
changed upstream image features from silently changing the supposedly fixed policy.
No GT or task label enters gate capture, replay, or checkpoint choice.
"""
from collections import defaultdict
from contextlib import contextmanager
import json
import math
from pathlib import Path
import statistics
from types import MethodType

import torch
from safetensors import safe_open

from smoke.token_residual_moe_v1 import bound_json, sha, digest, verify_assets
from smoke.token_moe_five_single_v1 import load_contract, verify_parent, FiveSamples, experiment_sources
from lowlight_restoration.token_residual_moe import TokenResidualFFN
from lowlight_restoration.rain_blur_image_moe import image_moe_modules

CONFIG = 'configs/foundir_token_moe_five_single_interference_v1.json'
ARMS = ('old_body_old_gates', 'old_body_new_gates', 'new_body_old_gates', 'new_body_new_gates')
SPLITS = ('rain', 'blur_preview', 'noise_preview', 'haze_preview', 'low_light_preview')


def diagnostic_config(root, path):
    cfg = json.loads(Path(path).read_text())
    expected = {'experiment', 'qualification_config', 'qualification_config_sha256',
        'terminal_report', 'terminal_report_sha256', 'old_step', 'new_step', 'records_per_task',
        'arms', 'self_replay_max_abs', 'self_replay_uint8_exact', 'output_root', 'minimum_free_gib',
        'wall_hours_estimate', 'allocated_gpu_hours_estimate', 'hard_wall_seconds_per_launch',
        'optimizer_updates', 'automatic_training'}
    if (set(cfg) != expected or cfg['experiment'] != 'five_single_interference_v1'
            or (cfg['old_step'], cfg['new_step'], cfg['records_per_task']) != (2000, 2025, 100)
            or cfg['arms'] != list(ARMS) or cfg['optimizer_updates'] != 0
            or cfg['automatic_training'] or cfg['self_replay_max_abs'] != 1e-5
            or cfg['self_replay_uint8_exact'] is not True):
        raise RuntimeError('read-only four-cell diagnostic contract changed')
    bound_json(Path(root), cfg['qualification_config'], cfg['qualification_config_sha256'])
    return cfg


def summarize_terminal(report):
    """Audit all embedded paired rows; compute MSE, not 10**(-mean PSNR/10)."""
    result = report['full_result']
    if not (report['status'] == 'qualification_terminal_engineering_pass_quality_regressions'
            and result['mode'] == 'qualification' and result['steps'] == 2025
            and result['start_step'] == 2000 and result['optimizer_updates_this_launch'] == 25
            and not result['protected_pixels_accessed'] and report['final_checkpoint_sha_verified']
            and len(report['logs']) == 4 and all(v['exit_status'] == 0 for v in report['logs'])):
        raise RuntimeError('unexpected qualification evidence')
    paired = {}
    for step in ('2000', '2025'):
        m = report['full_metrics'][step]
        rows = m['rows']
        if m['GT_used_for_routing'] or len(rows) != 500 or len({r['record_id'] for r in rows}) != 500:
            raise RuntimeError('invalid paired preview evidence')
        paired[step] = {r['record_id']: r for r in rows}
        for split in SPLITS:
            part = [r for r in rows if r['split'] == split]
            if len(part) != 100:
                raise RuntimeError('preview category changed')
            for metric in ('psnr', 'lpips'):
                values = [r[metric] for r in part]
                if not all(math.isfinite(v) for v in values) or abs(statistics.mean(values)-m['summary'][split][metric]) > 1e-10:
                    raise RuntimeError('reported mean differs from per-image results')
    if paired['2000'].keys() != paired['2025'].keys():
        raise RuntimeError('before/after images differ')
    for key, row in paired['2000'].items():
        after = paired['2025'][key]
        if (row['source_group'], row['split']) != (after['source_group'], after['split']):
            raise RuntimeError('before/after source identity differs')
    tasks = {}
    for split in SPLITS:
        old = [r for r in paired['2000'].values() if r['split'] == split]
        new = [paired['2025'][r['record_id']] for r in old]
        blocks = sorted(old[0]['weights'])
        tasks[split] = {
            'records': len(old), 'source_groups': len({r['source_group'] for r in old}),
            'metrics_before': report['full_metrics']['2000']['summary'][split],
            'metrics_after': report['full_metrics']['2025']['summary'][split],
            'mean_rgb01_mse_before': statistics.mean(10**(-r['psnr']/10) for r in old),
            'mean_rgb01_mse_after': statistics.mean(10**(-r['psnr']/10) for r in new),
            'psnr_improved_percent': 100*statistics.mean(b['psnr'] > a['psnr'] for a,b in zip(old,new)),
            'lpips_improved_percent': 100*statistics.mean(b['lpips'] < a['lpips'] for a,b in zip(old,new)),
            'median_psnr_delta': statistics.median(b['psnr']-a['psnr'] for a,b in zip(old,new)),
            'mean_expert0_weight_before': {b: statistics.mean(r['weights'][b][0] for r in old) for b in blocks},
            'mean_expert0_weight_after': {b: statistics.mean(r['weights'][b][0] for r in new) for b in blocks},
        }
    return {'status': 'paired_report_cpu_audit_pass', 'tasks': tasks,
            'optimizer_updates': 0, 'pixels_read': False,
            'gradient_dominance_proven': False, 'causal_component_proven': False}


def capture_routes(model):
    modules = image_moe_modules(model)
    if not modules or any(not isinstance(m, TokenResidualFFN) or m.last_weights is None for m in modules.values()):
        raise RuntimeError('capture requires a completed native token-router forward')
    return {n: m.last_weights.detach().cpu().clone() for n, m in modules.items()}


@contextmanager
def replay_routes(model, routes):
    """Diagnostic-only eval override; never patch the bound production source.

    Ignore BOTH image and VLM logits; use this image's recorded per-block
    soft weights. The two restoration experts and their exact blend are unchanged.
    """
    modules = image_moe_modules(model)
    if (model.training or torch.is_grad_enabled() or not modules
            or set(routes) != set(modules) or any(not isinstance(m, TokenResidualFFN) for m in modules.values())):
        raise RuntimeError('gate replay is inference-only and requires every token-router block')
    for name, value in routes.items():
        if (value.requires_grad or value.dtype != torch.float32 or value.ndim != 2 or value.shape[1] != 2
                or not torch.isfinite(value).all() or (value < 0).any() or (value > 1).any()
                or not torch.allclose(value.sum(-1), torch.ones(value.shape[0], device=value.device), atol=1e-6, rtol=0)):
            raise RuntimeError('invalid soft gate probabilities: ' + name)
    sentinel = object()
    original = {}
    try:
        for name, module in modules.items():
            original[name] = module.__dict__.get('forward', sentinel)
            weights = routes[name].detach().clone().to(next(module.parameters()).device)
            def forward(self, tokens, gates=weights):
                if self.training or torch.is_grad_enabled() or tokens.shape[0] != gates.shape[0]:
                    raise RuntimeError('gate replay cannot be used for training or a different batch')
                first = self.experts[0](tokens)
                second = self.experts[1](tokens)
                self.last_weights = gates
                return (first.float() + gates[:, 1, None, None] * (second.float()-first.float())).to(first.dtype)
            module.forward = MethodType(forward, module)
        yield
    finally:
        for name, previous in original.items():
            if previous is sentinel:
                delattr(modules[name], 'forward')
            else:
                modules[name].forward = previous


def parameter_group(name):
    if '.lora_A.' in name or '.lora_B.' in name:
        return 'shared_lora'
    for index in (0, 1):
        if f'.experts.{index}.' in name:
            return f'expert_{index}'
    if '.vlm_projector.' in name:
        return 'vlm_router'
    if '.router.' in name:
        return 'image_router'
    raise RuntimeError('unexpected saved trainable parameter: ' + name)


def checkpoint_deltas(old_path, new_path):
    totals = defaultdict(lambda: {'parameters': 0, 'changed_tensors': 0, 'tensors': 0,
                                 'old_squared_norm': 0., 'delta_squared_norm': 0., 'max_abs_delta': 0.})
    with safe_open(old_path, framework='pt', device='cpu') as old, safe_open(new_path, framework='pt', device='cpu') as new:
        if set(old.keys()) != set(new.keys()):
            raise RuntimeError('checkpoint parameter keys differ')
        for name in old.keys():
            a, b = old.get_tensor(name), new.get_tensor(name)
            if a.shape != b.shape or a.dtype != torch.float32 or b.dtype != torch.float32:
                raise RuntimeError('checkpoint shape/dtype changed')
            if not torch.isfinite(a).all() or not torch.isfinite(b).all():
                raise RuntimeError('nonfinite checkpoint')
            delta = b-a
            out = totals[parameter_group(name)]
            out['parameters'] += a.numel(); out['tensors'] += 1
            out['changed_tensors'] += int(bool(torch.count_nonzero(delta)))
            out['old_squared_norm'] += float(a.double().square().sum())
            out['delta_squared_norm'] += float(delta.double().square().sum())
            out['max_abs_delta'] = max(out['max_abs_delta'], float(delta.abs().max()))
    return dict(totals)


def four_cell_summary(cells):
    if set(cells) != set(ARMS):
        raise RuntimeError('all four predeclared cells are required')
    reference = None
    for rows in cells.values():
        if len({r['record_id'] for r in rows}) != len(rows):
            raise RuntimeError('duplicate evaluation record')
        identity = {(r['record_id'], r['source_group'], r['split']) for r in rows}
        if reference is not None and identity != reference:
            raise RuntimeError('counterfactual cells use different records')
        reference = identity
    splits = sorted({r['split'] for r in cells[ARMS[0]]})
    means = {a: {s: {m: statistics.mean(r[m] for r in rows if r['split']==s)
              for m in ('psnr', 'lpips')} for s in splits} for a, rows in cells.items()}
    contrasts = {}
    for s in splits:
        contrasts[s] = {}
        for m in ('psnr', 'lpips'):
            oo, on, no, nn = (means[a][s][m] for a in ARMS)
            contrasts[s][m] = {'gates_at_old_body': on-oo, 'body_at_old_gates': no-oo,
                'combined_change': nn-oo, 'interaction': nn-no-on+oo}
    return {'means': means, 'contrasts': contrasts,
            'meaning': 'metric differences, not independent/additive causal percentages; positive LPIPS is worse'}


def load_diagnostic(root, config_path, *, artifacts=True, model_assets=True):
    root = Path(root)
    dc = diagnostic_config(root, config_path)
    report = bound_json(root, dc['terminal_report'], dc['terminal_report_sha256'])
    audit = summarize_terminal(report)
    cfg, parent, base, qwen, qcfg, train, dev, data = load_contract(root, root / dc['qualification_config'])
    lineage = verify_parent(root, cfg, artifacts=artifacts)
    samples = FiveSamples(root, cfg, train, dev)
    all_specs = samples.all_specs()
    sources = experiment_sources(root, dc['qualification_config'], lineage)
    binding = {'config_sha256': sha(root / dc['qualification_config']), 'source_sha256': sources,
               'data': data, 'samples_sha256': digest(all_specs)}
    if binding != report['full_result']['binding']:
        raise RuntimeError('qualification scientific source/data identity changed')
    specs = samples.evaluation_specs(all_specs)
    expected = {(s['row']['record_id'], s['row']['source_group'], s['split']) for s in specs}
    actual = {(r['record_id'], r['source_group'], r['split']) for r in report['full_metrics']['2000']['rows']}
    if len(specs) != 500 or actual != expected:
        raise RuntimeError('must use the identical five-task development previews')
    old = bound_json(root, cfg['parent_report'], cfg['parent_report_sha256'])['full_result']['checkpoints'][-1]
    if old['step'] != 2000:
        raise RuntimeError('wrong old restoration weights')
    new = report['full_result']['checkpoints'][-1]
    if not (new['step'] == 2025 and new['mode'] == 'qualification' and new['arm'] == 'token_residual'
            and new['world_size'] == 8 and new['next_sample_position'] == 2025*32
            and new['config_sha256'] == binding['config_sha256']
            and new['source_sha256'] == binding['source_sha256']):
        raise RuntimeError('wrong new restoration weights')
    directories = {'old': Path(cfg['parent_receipt']).parent,
                   'new': Path(cfg['output_root']) / report['run_id']}
    receipts = {'old': old, 'new': new}
    weights = {key: root / directories[key] / receipt['weights'] for key, receipt in receipts.items()}
    for key, receipt in receipts.items():
        if Path(receipt['weights']).name != receipt['weights']:
            raise RuntimeError('unsafe checkpoint filename')
        if artifacts:
            path = directories[key] / f'step_{receipt["step"]:06d}.resume.json'
            if json.loads((root / path).read_text()) != receipt or sha(weights[key]) != receipt['weights_sha256']:
                raise RuntimeError('checkpoint differs from audited report')
    cache = root / cfg['output_root'] / 'token_cache' / digest(binding)
    feature_receipt = None
    if artifacts:
        if model_assets:
            verify_assets(root, cfg, base, qwen)
        feature_receipt = json.loads((cache / 'receipt.json').read_text())
        if feature_receipt['binding'] != binding or feature_receipt['records'] != 1300:
            raise RuntimeError('reuse only the completed qualification token cache')
    extra = [CONFIG, 'smoke/token_moe_interference_v1.py',
             'smoke/diagnose_foundir_token_moe_five_single_interference_v1.py',
             'smoke/run_foundir_token_moe_five_single_interference_v1_4x2.sh']
    diagnostic_binding = {'config_sha256': sha(config_path), 'qualification_binding': binding,
        'report_sha256': dc['terminal_report_sha256'], 'evaluation_specs_sha256': digest(specs),
        'checkpoint_sha256': {k: v['weights_sha256'] for k,v in receipts.items()},
        'diagnostic_sources': {n: sha(root / n) for n in extra}}
    if artifacts:
        diagnostic_binding['token_receipt_sha256'] = sha(cache / 'receipt.json')
    return dict(dc=dc, cfg=cfg, parent=parent, base=base, qwen=qwen, samples=samples, specs=specs,
                cache=cache, feature_receipt=feature_receipt, weights=weights, binding=diagnostic_binding,
                report=report, audit=audit)
