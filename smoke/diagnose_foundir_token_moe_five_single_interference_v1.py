#!/usr/bin/env python3
"""No-training 2x2 gate-decision / restoration-weight diagnostic, four nodes x two GPUs."""
import argparse
import json
import math
from pathlib import Path
import re
import shutil
import sys
import time
import os

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'third_party/flux2/src'))

import torch
import torch.distributed as dist
from safetensors.torch import load_file

from smoke.token_moe_interference_v1 import (CONFIG, ARMS, SPLITS, diagnostic_config,
    summarize_terminal, load_diagnostic, capture_routes, replay_routes,
    checkpoint_deltas, four_cell_summary)
from smoke.token_residual_moe_v1 import bound_json, sha, write_once, read_feature, feature_binding, verify_assets
from smoke.train_foundir_token_residual_moe_v1 import distributed_start, rank0
from lowlight_restoration.lora import inject_lora
from lowlight_restoration.joint_vlm_moe import install_joint_moe
from lowlight_restoration.rain_blur_image_moe import load_expert_state
from lowlight_restoration.token_residual_moe import attach_token_router, token_conditioning
from lowlight_restoration.flux2_endpoint_precision import promote_lora_to_fp32, restore_endpoint


def quantized(decoded):
    return ((decoded + 1) * 127.5).round().clamp(0, 255) / 127.5 - 1


def image_metrics(decoded, gt, metric):
    output = quantized(decoded)
    mse = float((((gt.float()-output.float())*.5).square()).mean())
    psnr = 100. if mse <= 1e-12 else -10.*math.log10(mse)
    lpips = float(metric((output+1)*.5, (gt+1)*.5))
    if not math.isfinite(psnr+lpips):
        raise RuntimeError('nonfinite diagnostic metrics')
    return {'psnr': psnr, 'lpips': lpips}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / CONFIG)
    parser.add_argument('--phase', choices=('report', 'preflight', 'evaluate'), required=True)
    parser.add_argument('--run-id', default='diagnosis_4x2_v1')
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.run_id):
        raise ValueError('unsafe run ID')
    if args.phase == 'report':
        dc = diagnostic_config(ROOT, args.config)
        print(json.dumps(summarize_terminal(bound_json(ROOT, dc['terminal_report'], dc['terminal_report_sha256'])),
                         ensure_ascii=False, indent=2, allow_nan=False))
        return
    c = load_diagnostic(ROOT, args.config, model_assets=args.phase == 'preflight')
    dc, cfg, base, qwen = (c[k] for k in ('dc', 'cfg', 'base', 'qwen'))
    if shutil.disk_usage(ROOT).free < dc['minimum_free_gib'] * 1024**3:
        raise RuntimeError('insufficient free space; do not delete existing assets')
    if args.phase == 'preflight':
        print(json.dumps({'status': 'cpu_preflight_pass', 'binding': c['binding'],
            'checkpoint_deltas': checkpoint_deltas(str(c['weights']['old']), str(c['weights']['new'])),
            'preview_records': len(c['specs']), 'optimizer_updates': 0, 'GPU_started': False,
            'protected_pixels_accessed': False}, ensure_ascii=False, allow_nan=False), flush=True)
        return
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
        TORCH_HOME=str(ROOT / base['runtime']['torch_home']),
        KLEIN_4B_BASE_MODEL_PATH=str(ROOT / base['model']['flow_weights']),
        AE_MODEL_PATH=str(ROOT / base['model']['autoencoder_weights']))
    output = ROOT / dc['output_root'] / args.run_id
    rank, local, device, hardware = distributed_start(
        {**c['binding'], 'run_id': args.run_id, 'phase': args.phase}, ROOT / dc['output_root'])
    started = time.monotonic()
    try:
        rank0(lambda: verify_assets(ROOT, cfg, base, qwen))
        def prepare():
            output.mkdir(parents=True, exist_ok=False)
            write_once(output / 'launch.json', {'binding': c['binding'], 'hardware': hardware,
                'optimizer_updates': 0, 'no_weight_output': True})
        rank0(prepare)
        from flux2.util import load_flow_model, load_ae
        from flux2.sampling import batched_prc_img
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
        ae = load_ae(base['model']['name'], device=device).eval().requires_grad_(False)
        model = load_flow_model(base['model']['name'], device=device).eval()
        names = inject_lora(model, base['lora']['target_patterns'], rank=32, alpha=32, dropout=0)
        if len(names) != 50:
            raise RuntimeError('shared LoRA structure changed')
        promote_lora_to_fp32(model)
        install_joint_moe(model, arm='image_only', indices=cfg['ffn_blocks'])
        attach_token_router(model, cfg['feature_dim'], cfg['router_hidden'])
        model.eval()
        prompt = load_file(str(ROOT / base['model']['prompt_context_cache']))
        context, ids = prompt['cfg_context'][:1].to(device), prompt['cfg_ids'][:1].to(device)
        del prompt
        metric = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True).to(device).eval().requires_grad_(False)
        specs = c['specs'][rank::8]
        rows = {a: [] for a in ARMS}
        routes = {'old': {}, 'new': {}}
        parity = {'old': {'max_abs': 0., 'checked': 0}, 'new': {'max_abs': 0., 'checked': 0}}

        def feature(spec):
            path = c['cache'] / (spec['key'] + '.safetensors')
            if sha(path) != c['feature_receipt']['files'][spec['key']]:
                raise RuntimeError('qualification feature bytes changed')
            value = read_feature(path, feature_binding(spec, c['binding']['qualification_binding']['config_sha256'], qwen))
            if value is None:
                raise RuntimeError('missing qualification features; no re-extraction or label replacement')
            return value[None].to(device)

        def forward(lq):
            return restore_endpoint(model, ae, lq, context, ids, batched_prc_img)['decoded']

        def record(spec, decoded, gt, gates):
            return {'record_id': spec['row']['record_id'], 'source_group': spec['row']['source_group'],
                    'split': spec['split'], **image_metrics(decoded, gt, metric),
                    'weights': {n: v[0].tolist() for n, v in gates.items()}}

        # One restoration model resident at a time. Reload old weights only once
        # after capturing both policies; never hold two 4B models on the GPU.
        for body in ('old', 'new', 'old_cross'):
            weight_key = 'old' if body == 'old_cross' else body
            load_expert_state(model, c['weights'][weight_key])
            model.eval()
            with torch.inference_mode():
                for index, spec in enumerate(specs):
                    key = spec['row']['record_id']
                    lq, gt = [v[None].to(device) for v in c['samples'].pair(spec)]
                    if body != 'old_cross':
                        with token_conditioning(model, feature(spec)):
                            native = forward(lq)
                            gates = capture_routes(model)
                        routes[body][key] = gates
                        with replay_routes(model, gates):
                            replayed = forward(lq)
                        error = float((native-replayed).abs().max())
                        if (not math.isfinite(error) or error > dc['self_replay_max_abs']
                                or not torch.equal(quantized(native), quantized(replayed))):
                            raise RuntimeError('own-gate replay failed; intervention results are invalid')
                        parity[body]['max_abs'] = max(parity[body]['max_abs'], error)
                        parity[body]['checked'] += 1
                        rows[f'{body}_body_{body}_gates'].append(record(spec, native, gt, gates))
                        del native, replayed
                    if body == 'new':
                        gates = routes['old'][key]
                        with replay_routes(model, gates):
                            decoded = forward(lq)
                        rows['new_body_old_gates'].append(record(spec, decoded, gt, gates))
                        del decoded
                    elif body == 'old_cross':
                        gates = routes['new'][key]
                        with replay_routes(model, gates):
                            decoded = forward(lq)
                        rows['old_body_new_gates'].append(record(spec, decoded, gt, gates))
                        del decoded
                    if index % 10 == 0 or index+1 == len(specs):
                        print(f'TOKEN_MOE_INTERFERENCE rank={rank} stage={body} records={index+1}/{len(specs)}', flush=True)
            # Preserve completed observations even if a later stage/node fails.
            write_once(output / f'rank_{rank}_{body}.json', {'binding': c['binding'],
                'rank': rank, 'stage': body, 'rows': rows, 'self_replay': parity})
        observed = [None] * 8
        dist.all_gather_object(observed, {'rank': rank, 'rows': rows, 'self_replay': parity,
            'peak_memory_gib': torch.cuda.max_memory_allocated()/1024**3})
        def finish():
            cells = {a: sorted([r for o in observed for r in o['rows'][a]], key=lambda r:r['record_id']) for a in ARMS}
            expected = {(s['row']['record_id'], s['row']['source_group'], s['split']) for s in c['specs']}
            for values in cells.values():
                if len(values) != 500 or {(r['record_id'],r['source_group'],r['split']) for r in values} != expected:
                    raise RuntimeError('incomplete or changed four-cell preview')
            checks = {k: {'checked': sum(o['self_replay'][k]['checked'] for o in observed),
                         'max_abs': max(o['self_replay'][k]['max_abs'] for o in observed)} for k in ('old','new')}
            if any(v['checked'] != 500 for v in checks.values()):
                raise RuntimeError('incomplete own-gate replay validation')
            for key, path in c['weights'].items():
                if sha(path) != c['binding']['checkpoint_sha256'][key]:
                    raise RuntimeError('source checkpoint bytes changed during diagnosis')
            summary = four_cell_summary(cells)
            historical = {}
            for step, arm in (('2000', 'old_body_old_gates'), ('2025', 'new_body_new_gates')):
                historical[step] = {s:{m:summary['means'][arm][s][m]-c['report']['full_metrics'][step]['summary'][s][m]
                                      for m in ('psnr','lpips')} for s in SPLITS}
            hours = (time.monotonic()-started)/3600
            write_once(output / 'result.json', {'status': 'diagnostic_complete', 'binding': c['binding'],
                'rows': cells, **summary, 'self_replay': checks, 'historical_native_differences': historical,
                'peak_memory_gib': max(o['peak_memory_gib'] for o in observed),
                'evaluation_wall_hours': hours, 'evaluation_allocated_gpu_hours': hours*8,
                'optimizer_updates': 0, 'automatic_training': False, 'GT_used_for_routing': False,
                'protected_pixels_accessed': False, 'full_five_type_gate': False,
                'counterfactuals_are_diagnostics_not_deployable_results': True})
            print('TOKEN_MOE_INTERFERENCE_COMPLETE ' + json.dumps(summary), flush=True)
        rank0(finish)
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
