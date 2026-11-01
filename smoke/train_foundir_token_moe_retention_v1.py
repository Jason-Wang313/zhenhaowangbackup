#!/usr/bin/env python3
"""Independent five-single retention qualification; four nodes x two GPUs."""
from contextlib import nullcontext
import argparse
import json
import math
import os
from pathlib import Path
import random
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'third_party/flux2/src'))

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from safetensors.torch import load_file

from smoke.token_moe_retention_v1 import (CONFIG, TASKS, load_contract, verify_parent, source_hashes,
    RetentionSamples, batch_slot, retention_objective, reference_binding, read_reference, save_reference)
from smoke.token_residual_moe_v1 import (sha, digest, write_once, verify_assets, feature_binding, read_feature)
from smoke.train_foundir_token_residual_moe_v1 import rank0, distributed_start, extract, parameter_groups
from lowlight_restoration.token_residual_moe import attach_token_router, token_conditioning
from lowlight_restoration.joint_vlm_moe import install_joint_moe, enable_block_checkpointing, restoration_loss
from lowlight_restoration.rain_blur_image_moe import load_expert_state
from lowlight_restoration.lora import inject_lora
from lowlight_restoration.flux2_endpoint_precision import promote_lora_to_fp32, restore_endpoint, assert_fp32_optimizer
from lowlight_restoration.token_moe_retention_state import save_state, resume, rng_state, restore_rng


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('preflight', 'features', 'reference', 'train'), required=True)
    parser.add_argument('--config', type=Path, default=ROOT / CONFIG)
    parser.add_argument('--run-id', default='qualification_4x2_v1')
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.run_id):
        raise ValueError('unsafe run ID')
    cfg, parent, base, qwen, qcfg, training, development, data = load_contract(ROOT, args.config)
    lineage = verify_parent(ROOT, cfg, tensors=args.phase == 'preflight')
    if lineage['binding']['data'] != data['parent_two_task_data']:
        raise RuntimeError('parent development/training identities changed')
    samples = RetentionSamples(ROOT, cfg, training, development)
    specs = samples.all_specs()
    config_sha = sha(args.config)
    sources = source_hashes(ROOT, str(args.config.resolve().relative_to(ROOT)), lineage)
    binding = {'config_sha256': config_sha, 'source_sha256': sources, 'data': data,
               'samples_sha256': digest(specs), 'mode': cfg['mode'], 'global_batch': 40,
               'parent_receipt_sha256': cfg['parent_receipt_sha256']}
    root = ROOT / cfg['output_root']
    output = root / args.run_id
    cache = root / 'cache' / digest(binding)
    tokens, references = cache / 'tokens', cache / 'reference'
    parent_path = ROOT / cfg['parent_receipt']
    parent_receipt = json.loads(parent_path.read_text())
    parent_weights = parent_path.parent / parent_receipt['weights']
    if Path(parent_receipt['weights']).name != parent_receipt['weights']:
        raise RuntimeError('unsafe parent artifact name')

    def assets():
        verify_assets(ROOT, cfg, base, qwen)
        if sha(parent_weights) != parent_receipt['weights_sha256']:
            raise RuntimeError('parent step2000 weights changed')

    def space():
        import shutil
        if shutil.disk_usage(ROOT).free < cfg['minimum_free_gib'] * 1024**3:
            raise RuntimeError(f"requires {cfg['minimum_free_gib']}GiB free; no automatic deletion")

    if args.phase == 'preflight':
        assets(); space()
        print(json.dumps({'status': 'retention_cpu_preflight_pass', 'binding': binding,
                          'samples': 1000, 'per_task': 200, 'preview': 500,
                          'fresh_Adam': True, 'parent_step': 2000, 'phase_steps': 25,
                          'GPU_started': False, 'protected_pixels_accessed': False}), flush=True)
        return
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
        TORCH_HOME=str(ROOT / base['runtime']['torch_home']),
        KLEIN_4B_BASE_MODEL_PATH=str(ROOT / base['model']['flow_weights']),
        AE_MODEL_PATH=str(ROOT / base['model']['autoencoder_weights']))
    rank, local, device, hardware = distributed_start(
        {**binding, 'phase': args.phase, 'run_id': args.run_id,
         'resume': str(args.resume) if args.resume else None}, root)
    started = time.monotonic()
    try:
        rank0(assets)
        if not args.resume:
            rank0(space)
        if args.phase == 'features':
            extract(samples, specs, tokens, config_sha, qwen, qcfg, rank, device, binding)
            return
        token_receipt = json.loads((tokens / 'receipt.json').read_text())
        if token_receipt['binding'] != binding or token_receipt['records'] != len(specs):
            raise RuntimeError('matching complete token cache required')
        seen = set()

        def feature(spec):
            path = tokens / (spec['key'] + '.safetensors')
            if spec['key'] not in seen:
                if sha(path) != token_receipt['files'][spec['key']]:
                    raise RuntimeError('token cache modified')
                seen.add(spec['key'])
            value = read_feature(path, feature_binding(spec, config_sha, qwen))
            if value is None:
                raise RuntimeError('missing exact-crop VLM feature')
            return value[None].to(device)

        random.seed(cfg['seed']); np.random.seed(cfg['seed']); torch.manual_seed(cfg['seed'])
        torch.cuda.manual_seed_all(cfg['seed'])
        from flux2.util import load_flow_model, load_ae
        from flux2.sampling import batched_prc_img
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
        prompt = load_file(str(ROOT / base['model']['prompt_context_cache']))
        context, ids = prompt['cfg_context'][:1].to(device), prompt['cfg_ids'][:1].to(device)
        del prompt
        ae = load_ae(base['model']['name'], device=device).eval().requires_grad_(False)
        model = load_flow_model(base['model']['name'], device=device).eval()
        names = inject_lora(model, base['lora']['target_patterns'], rank=32, alpha=32, dropout=0)
        if len(names) != 50:
            raise RuntimeError('shared LoRA structure changed')
        promote_lora_to_fp32(model)
        install_joint_moe(model, arm='image_only', indices=cfg['ffn_blocks'])
        modules = attach_token_router(model, cfg['feature_dim'], cfg['router_hidden'])
        # Strict loading includes previously trained VLM projectors. No zeroing,
        # old image-only initialization or old optimizer-state restoration.
        load_expert_state(model, parent_weights)
        metric = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True).to(device).eval().requires_grad_(False)

        def weights(spec):
            return {**parent['training']['losses'], 'lpips': cfg['lpips_by_task'][spec['task']]}

        train_specs = [s for s in specs if s['kind'] == 'train']
        if args.phase == 'reference':
            # One model only. Cache six GT-supervised TRAINING loss scalars,
            # never teacher pixels, GT features or a second resident model.
            model.eval().requires_grad_(False)
            for index, spec in enumerate(train_specs[rank::8]):
                identity = reference_binding(spec, binding)
                path = references / (spec['key'] + '.json')
                if read_reference(path, identity) is None:
                    lq, gt = [v[None].to(device) for v in samples.pair(spec)]
                    with torch.inference_mode(), token_conditioning(model, feature(spec)):
                        clean = ae.encode(gt.float()).float()
                        result = restore_endpoint(model, ae, lq, context, ids, batched_prc_img)
                        _, terms = restoration_loss(result, clean, gt, metric, batched_prc_img, weights(spec))
                        values = {k: float(v) for k, v in terms.items()}
                    save_reference(path, values, identity)
                if index % 20 == 0 or index + 1 == len(train_specs[rank::8]):
                    print(f'RETENTION_REFERENCE rank={rank} completed={index+1}/{len(train_specs[rank::8])}', flush=True)
            dist.barrier()

            def reference_receipt():
                files, summary = {}, {}
                for spec in train_specs:
                    path = references / (spec['key'] + '.json')
                    if read_reference(path, reference_binding(spec, binding)) is None:
                        raise RuntimeError('incomplete reference')
                    files[spec['key']] = sha(path)
                for task in TASKS:
                    values = [read_reference(references / (s['key'] + '.json'), reference_binding(s, binding))
                              for s in train_specs if s['task'] == task]
                    summary[task] = {k: float(np.mean([v[k] for v in values])) for k in values[0]}
                write_once(references / 'receipt.json', {'binding': binding, 'files': files,
                           'records': len(train_specs), 'training_only': True, 'mean_terms': summary,
                           'parent_weights_sha256': parent_receipt['weights_sha256']})
                print('RETENTION_REFERENCE_COMPLETE ' + json.dumps(summary), flush=True)
            rank0(reference_receipt)
            return

        reference_receipt = json.loads((references / 'receipt.json').read_text())
        if (reference_receipt['binding'] != binding or reference_receipt['records'] != len(train_specs)
                or not reference_receipt['training_only']
                or reference_receipt['parent_weights_sha256'] != parent_receipt['weights_sha256']):
            raise RuntimeError('matching train-only reference cache required')

        def reference(spec):
            path = references / (spec['key'] + '.json')
            if sha(path) != reference_receipt['files'][spec['key']]:
                raise RuntimeError('reference cache changed')
            values = read_reference(path, reference_binding(spec, binding))
            if values is None:
                raise RuntimeError('missing reference')
            return values

        def prepare():
            output.mkdir(parents=True, exist_ok=False)
            write_once(output / 'launch.json', {'binding': binding, 'hardware': hardware,
                'initialization': 'same_phase_full_resume' if args.resume else 'step2000_weights_fresh_Adam',
                'resume': str(args.resume) if args.resume else None, 'parent_weights_sha256': parent_receipt['weights_sha256'],
                'reference_receipt_sha256': sha(references / 'receipt.json'),
                'token_receipt_sha256': sha(tokens / 'receipt.json')})
            if args.resume:
                old_launch = json.loads((args.resume.parent / 'launch.json').read_text())
                for key, path in [('reference_receipt_sha256', references / 'receipt.json'),
                                  ('token_receipt_sha256', tokens / 'receipt.json')]:
                    if old_launch[key] != sha(path):
                        raise RuntimeError('resume cache receipt changed')
                baseline = json.loads((args.resume.parent / 'baseline.json').read_text())
                if sha(ROOT / baseline['path']) != baseline['sha256']:
                    raise RuntimeError('original stage baseline changed')
                write_once(output / 'baseline.json', baseline)
        rank0(prepare)
        enable_block_checkpointing(model)
        groups = parameter_groups(model, parent)
        optimizer = torch.optim.AdamW(groups, weight_decay=0, fused=True)
        start_step = resume(model, optimizer, args.resume, binding, rank, cfg['steps']) if args.resume else 0
        if not args.resume and optimizer.state:
            raise RuntimeError('new phase must use fresh Adam')
        initial_vlm = {n: p.detach().cpu().clone() for n, p in model.named_parameters() if '.vlm_projector.' in n}
        restored_rng = rng_state()
        wrapped = DDP(model, device_ids=[local], broadcast_buffers=False, gradient_as_bucket_view=True)
        params = [p for g in groups for p in g['params']]

        def evaluate(step):
            model.eval(); rows = []
            validation = samples.evaluation_specs(specs)
            for spec in validation[rank::8]:
                lq, gt = [v[None].to(device) for v in samples.pair(spec)]
                with torch.inference_mode(), token_conditioning(model, feature(spec)):
                    decoded = restore_endpoint(model, ae, lq, context, ids, batched_prc_img)['decoded']
                    quantized = ((decoded + 1) * 127.5).round().clamp(0, 255) / 127.5 - 1
                    mse = float((((gt.float() - quantized.float()) * .5).square()).mean())
                    psnr = 100. if mse <= 1e-12 else -10 * math.log10(mse)
                    lpips = float(metric((quantized + 1) * .5, (gt + 1) * .5))
                    if not math.isfinite(psnr + lpips):
                        raise RuntimeError('nonfinite evaluation')
                    rows.append({'record_id': spec['row']['record_id'], 'source_group': spec['row']['source_group'],
                                 'split': spec['split'], 'psnr': psnr, 'lpips': lpips,
                                 'weights': {k: m.last_weights[0].cpu().tolist() for k, m in modules.items()}})
            gathered = [None] * 8
            dist.all_gather_object(gathered, rows)

            def publish():
                all_rows = sorted([r for part in gathered for r in part], key=lambda r: r['record_id'])
                if (len(all_rows) != len(validation) or {r['record_id'] for r in all_rows}
                        != {s['row']['record_id'] for s in validation}):
                    raise RuntimeError('evaluation identity mismatch')
                summary = {split: {m: float(np.mean([r[m] for r in all_rows if r['split'] == split]))
                           for m in ('psnr', 'lpips')} for split in sorted({r['split'] for r in all_rows})}
                path = output / f'metrics_{step:06d}.json'
                write_once(path, {'phase_step': step, 'rows': all_rows, 'summary': summary,
                                 'GT_used_for_routing': False, 'full_five_type_gate': False})
                if step == 0:
                    write_once(output / 'baseline.json', {'path': str(path.relative_to(ROOT)), 'sha256': sha(path)})
                print('RETENTION_METRICS ' + json.dumps({'phase_step': step, **summary}), flush=True)
            rank0(publish)

        evaluate(start_step)
        restore_rng(restored_rng)
        times, checkpoints = [], []
        for step in range(start_step, cfg['steps']):
            tick = time.monotonic()
            model.train(); optimizer.zero_grad(set_to_none=True)
            telemetry = []
            for micro in range(5):
                offset, scale = batch_slot(step, micro, rank)
                spec = samples.train_spec(offset)
                lq, gt = [v[None].to(device) for v in samples.pair(spec)]
                ref = reference(spec)
                with torch.no_grad():
                    clean = ae.encode(gt.float()).float()
                with wrapped.no_sync() if micro < 4 else nullcontext():
                    # Keep condition installed through checkpoint-recomputed backward.
                    with token_conditioning(model, feature(spec)):
                        result = restore_endpoint(wrapped, ae, lq, context, ids, batched_prc_img)
                        _, terms = restoration_loss(result, clean, gt, metric, batched_prc_img, weights(spec))
                        loss, info = retention_objective(terms, ref, weights(spec), cfg['objective'])
                        (loss * scale).backward()
                        telemetry.append([float(loss.detach()), *[float(info[k].detach()) for k in
                                           ('relative_total', 'mse_regression', 'lpips_regression')]])
            if any(p.grad is None for p in params):
                raise RuntimeError('disconnected trainable parameter')
            group_norms = {g['name']: math.sqrt(sum(float(p.grad.float().square().sum()) for p in g['params']))
                           for g in groups}
            norm = torch.nn.utils.clip_grad_norm_(params, 1.)
            if not math.isfinite(float(norm)):
                raise RuntimeError('nonfinite gradient')
            optimizer.step(); assert_fp32_optimizer(optimizer)
            torch.cuda.synchronize(); times.append(time.monotonic() - tick)
            values = torch.tensor(telemetry, device=device)
            dist.all_reduce(values); values /= 8
            completed = step + 1
            if rank == 0:
                print('RETENTION_TRAIN ' + json.dumps({'phase_step': completed, 'steps': cfg['steps'],
                    'tasks': {task: dict(zip(('loss', 'relative_total', 'mse_regression', 'lpips_regression'), v))
                              for task, v in zip(TASKS, values.cpu().tolist())},
                    'grad_norms_before_clip': group_norms, 'seconds': times[-1],
                    'peak_memory_gib': torch.cuda.max_memory_allocated() / 1024**3}), flush=True)
            if completed % cfg['save_every'] == 0:
                states = [None] * 8
                dist.all_gather_object(states, rng_state())
                rank0(lambda: checkpoints.append(save_state(model, optimizer, output,
                     step=completed, binding=binding, rank_rng=states)))
            if completed in cfg['milestones']:
                before_eval = rng_state(); evaluate(completed); restore_rng(before_eval)

        observations = [None] * 8
        changed = any(not torch.equal(p.detach().cpu(), initial_vlm[n])
                      for n, p in model.named_parameters() if n in initial_vlm) if times else None
        dist.all_gather_object(observations, {'updates_this_launch': len(times), 'vlm_changed': changed,
            'per_task_presentations_this_launch': len(times),
            'peak_memory_gib': torch.cuda.max_memory_allocated() / 1024**3})

        def finish():
            if times and not all(o['vlm_changed'] for o in observations):
                raise RuntimeError('VLM branch did not update on every rank')
            baseline = json.loads((output / 'baseline.json').read_text())
            if sha(ROOT / baseline['path']) != baseline['sha256']:
                raise RuntimeError('baseline identity changed')
            initial = json.loads((ROOT / baseline['path']).read_text())['summary']
            final = json.loads((output / f'metrics_{cfg["steps"]:06d}.json').read_text())['summary']
            gain = {s: {m: final[s][m] - initial[s][m] for m in ('psnr', 'lpips')} for s in initial}
            hours = (time.monotonic() - started) / 3600
            write_once(output / 'result.json', {'status': 'engineering_pass', 'binding': binding,
                'start_phase_step': start_step, 'steps': cfg['steps'], 'parent_step': 2000,
                'gain_vs_step2000_same_run': gain, 'checkpoints': checkpoints, 'observations': observations,
                'resume_receipt': ({'path': str(args.resume), 'sha256': sha(args.resume)} if args.resume else None),
                'optimizer_updates_this_launch': len(times), 'total_presentations': cfg['steps'] * 40,
                'total_presentations_per_task': cfg['steps'] * 8,
                'median_training_seconds': float(np.median(times)) if times else None,
                'train_phase_wall_hours': hours, 'train_phase_allocated_gpu_hours': hours * 8,
                'full_five_type_gate': False, 'VLM_advantage_proven': False,
                'automatic_formal': False, 'qualification_may_seed_formal': False,
                'protected_pixels_accessed': False})
            print('TOKEN_MOE_RETENTION_QUALIFICATION_PASS', flush=True)
        rank0(finish)
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
