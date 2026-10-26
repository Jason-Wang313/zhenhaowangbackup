#!/usr/bin/env python3
"""One-step Blur-only, no-router diagnostic; same initialization/loss/exposures."""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import time
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'third_party/flux2/src'))
from smoke import train_foundir_rain_blur_image_moe_v1_ddp as paired
from lowlight_restoration.blur_single_endpoint import (
    BlurOnlyDataset, save_state, load_state, rng_state, restore_rng,
)
from lowlight_restoration.rain_blur_image_moe import inject_image_moe
from lowlight_restoration.flux2_endpoint_precision import (
    promote_lora_to_fp32, restore_endpoint, assert_fp32_optimizer,
)
from lowlight_restoration.lora import inject_lora, load_lora_checkpoint

old, write_json = paired.old, paired.write_json
CONFIG_SHA256 = 'a273b55ccfd9ca128b976ea5a6fa314ce862d77273a4127f0e5836e58957eefb'


def source_hashes():
    paths = [Path(__file__), ROOT / 'lowlight_restoration/blur_single_endpoint.py',
             ROOT / 'smoke/run_foundir_blur_single_endpoint_v1_8gpu.sh',
             ROOT / 'smoke/start_foundir_blur_single_endpoint_v1_tmux.sh']
    return {**paired.source_hashes(), **{str(p.relative_to(ROOT)): old.sha256(p) for p in paths}}


def read_contract(path):
    if old.sha256(path) != CONFIG_SHA256:
        raise RuntimeError('frozen Blur-only contract changed')
    config = json.loads(path.read_text())
    if old.sha256(ROOT / config['parent_config']) != config['parent_config_sha256']:
        raise RuntimeError('parent MoE contract changed')
    parent_config, parent, training, development, metadata = paired.read_contract(ROOT / config['parent_config'])
    evaluation = config['evaluation']
    if old.sha256(ROOT / evaluation['blur_initial_reference']) != evaluation['blur_initial_reference_sha256']:
        raise RuntimeError('initial Blur parity reference changed')
    return config, parent_config, parent, training, development, metadata


def preflight(path):
    config, _, _, training, development, data = read_contract(path)
    paired.preflight(ROOT / config['parent_config'])
    if len(training['blur']) != 105057 or len(development['blur_full']) != 2595:
        raise RuntimeError('Blur protocol mismatch')
    result = {'status': 'blur_single_endpoint_preflight_pass', 'config_sha256': CONFIG_SHA256,
              'source_sha256': source_hashes(), 'data': data,
              'formal_blur_presentations': 48000, 'protected_pixels_accessed': False,
              'gpu_qualification_completed': False}
    print('BLUR_SINGLE_PREFLIGHT ' + json.dumps(result), flush=True)
    return result


def check_blur_parity(actual, reference, parent_config):
    a = {r['record_id']: r for r in actual['rows']}
    b = {r['record_id']: r for r in reference['rows']}
    if a.keys() != b.keys():
        raise RuntimeError('initial Blur parity record mismatch')
    errors = {}
    for metric in ('psnr', 'lpips'):
        if any(a[k]['source_group'] != b[k]['source_group'] or
               abs(a[k]['input_' + metric] - b[k]['input_' + metric]) > 1e-6 for k in a):
            raise RuntimeError('initial Blur inputs changed')
        errors[metric] = max(abs(a[k]['output_' + metric] - b[k]['output_' + metric]) for k in a)
        if not math.isfinite(errors[metric]) or errors[metric] > parent_config['evaluation']['parity_max_' + metric + '_error']:
            raise RuntimeError('single-expert initialization changed Blur output')
    return {'status': 'initial_blur_parity_pass', 'maximum_metric_errors': errors}


def verify_qualification(q, sources, data):
    if not (q.get('status') == 'complete' and q.get('mode') == 'qualification'
            and q.get('engineering_pass') and q.get('initial_parity_pass')
            and q.get('config_sha256') == CONFIG_SHA256
            and q.get('source_sha256') == sources and q.get('data') == data):
        raise RuntimeError('formal requires the matching Blur-only engineering qualification')


def rank_zero_action(rank, callback):
    error, result = [None], None
    if rank == 0:
        try:
            result = callback()
        except Exception as exc:
            error[0] = str(exc)
    dist.broadcast_object_list(error, src=0)
    if error[0]:
        raise RuntimeError(error[0])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/foundir_blur_single_endpoint_v1.json')
    parser.add_argument('--mode', choices=('qualification', 'formal'), default='qualification')
    parser.add_argument('--run-id', default='qualification_v1')
    parser.add_argument('--qualification-result', type=Path)
    parser.add_argument('--resume-receipt', type=Path)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    if args.preflight:
        preflight(args.config)
        return
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.run_id):
        raise RuntimeError('unsafe run ID')
    if args.resume_receipt and args.mode != 'formal':
        raise RuntimeError('only formal continuation is supported')
    config, parent_config, parent, training, development, data = read_contract(args.config)
    sources = source_hashes()
    if args.mode == 'formal':
        if args.qualification_result is None:
            raise RuntimeError('formal requires qualification receipt')
        verify_qualification(json.loads(args.qualification_result.read_text()), sources, data)
    rank, world, local = (int(os.environ[k]) for k in ('RANK', 'WORLD_SIZE', 'LOCAL_RANK'))
    if world != 8 or int(os.environ.get('LOCAL_WORLD_SIZE', 8)) != 8 or torch.cuda.device_count() != 8:
        raise RuntimeError('requires one eight-GPU node')
    torch.cuda.set_device(local)
    device = torch.device('cuda', local)
    prop = torch.cuda.get_device_properties(device)
    if prop.total_memory < 75 * 1024**3 or not any(n in prop.name for n in ('H100', 'H200')):
        raise RuntimeError('requires H100/H200 >=75 GiB')
    dist.init_process_group('nccl', device_id=device, timeout=timedelta(minutes=30))
    output = ROOT / 'checkpoints/foundir_blur_single_endpoint_v1' / args.run_id
    started = time.monotonic()
    try:
        names = [None] * world
        dist.all_gather_object(names, prop.name)
        if len(set(names)) != 1:
            raise RuntimeError('mixed GPU hardware')
        def initialize_output():
            if output.exists():
                raise RuntimeError('preserve existing output; continuation needs a new run ID')
            preflight(args.config)
            output.mkdir(parents=True)
            write_json(output / 'receipt.json', {'config': config, 'config_sha256': CONFIG_SHA256,
                'source_sha256': sources, 'data': data, 'mode': args.mode,
                'resume_receipt': str(args.resume_receipt) if args.resume_receipt else None})
        rank_zero_action(rank, initialize_output)
        seed = config['training']['seed']
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
            TORCH_HOME=str(ROOT / parent['runtime']['torch_home']),
            KLEIN_4B_BASE_MODEL_PATH=str(ROOT / parent['model']['flow_weights']),
            AE_MODEL_PATH=str(ROOT / parent['model']['autoencoder_weights']))
        from flux2.util import load_ae, load_flow_model
        from flux2.sampling import batched_prc_img
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
        context, context_ids, _, _ = old.encode_prompt(parent, device)
        ae = load_ae(parent['model']['name'], device=device).eval().requires_grad_(False)
        model = load_flow_model(parent['model']['name'], device=device).eval().requires_grad_(False)
        targets = inject_lora(model, parent['lora']['target_patterns'], rank=32, alpha=32, dropout=0)
        if len(targets) != 50:
            raise RuntimeError('Rain LoRA targets changed')
        promote_lora_to_fp32(model)
        load_lora_checkpoint(model, ROOT / parent_config['rain_lora'])
        modules = inject_image_moe(model, indices=config['architecture']['blocks'], experts=1)
        params = [p for p in model.parameters() if p.requires_grad]
        count = sum(p.numel() for p in params)
        if count != config['architecture']['trainable_parameters'] or any(m.router is not None for m in modules.values()):
            raise RuntimeError('single-expert/no-router architecture mismatch')
        if any(p.dtype != torch.float32 for p in params):
            raise RuntimeError('trainable state must be FP32')
        optimizer = torch.optim.AdamW(params, lr=config['training']['expert_lr'], weight_decay=0, fused=True)
        metric = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True).to(device).eval().requires_grad_(False)
        stages = config['modes'][args.mode]
        start_step, resume = 0, None
        if args.resume_receipt:
            resume = load_state(model, optimizer, args.resume_receipt, config_sha=CONFIG_SHA256,
                sources=sources, rank=rank, world=world, total_steps=stages['steps'])
            start_step = resume['step']
        wrapped = DDP(model, device_ids=[local], broadcast_buffers=False, gradient_as_bucket_view=True)
        evaluation_args = dict(rank=rank, world=world, device=device, output=output)
        history, checkpoints, times = [], [], []
        def assess(step, final=False):
            split = 'blur_full' if final and args.mode == 'formal' else 'blur_preview'
            summary = paired.evaluate(model, ae, development[split], context, context_ids,
                batched_prc_img, metric, split=split, step=step, **evaluation_args)
            if rank == 0:
                history.append(summary)
            return summary
        if not resume:
            initial_blur = assess(0)
            initial_rain = paired.evaluate(model, ae, development['rain'], context, context_ids,
                batched_prc_img, metric, split='rain_initial_parity', step=0, **evaluation_args)
            def initial_parity():
                reference = json.loads((ROOT / config['evaluation']['blur_initial_reference']).read_text())
                parity = {'rain': paired.check_rain_parity(initial_rain, parent_config),
                          'blur': check_blur_parity(initial_blur, reference, parent_config)}
                write_json(output / 'initial_parity.json', parity)
                print('BLUR_SINGLE_INITIAL_PARITY ' + json.dumps(parity), flush=True)
            rank_zero_action(rank, initial_parity)
        elif start_step in stages['milestones']:
            # A stop may occur after saving but before milestone evaluation.
            assess(start_step)
        # Iterator startup consumes RNG; checkpoint restoration comes AFTER it.
        saved_rng = rng_state() if resume else None
        dataset = BlurOnlyDataset(ROOT, training['blur'], steps=stages['steps'], rank=rank,
            seed=config['training']['sample_crop_seed'], start_step=start_step)
        loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2,
            prefetch_factor=2, multiprocessing_context='spawn', pin_memory=True)
        iterator = iter(loader)
        if saved_rng:
            restore_rng(saved_rng)
        sampled_initial = {n: p.detach().flatten()[:64].cpu().clone() for n, p in model.named_parameters() if p.requires_grad}
        weights = dict(parent['training']['losses'], teacher_l1_weight=0.0)
        for step in range(start_step, stages['steps']):
            tick = time.monotonic()
            model.train()
            optimizer.zero_grad(set_to_none=True)
            totals = {}
            for micro in range(2):
                batch = next(iterator)
                if int(batch['position'][0]) != step * 16 + micro * world + rank:
                    raise RuntimeError('absolute Blur sample position mismatch')
                lq, gt = batch['lq'].to(device, non_blocking=True), batch['gt'].to(device, non_blocking=True)
                with torch.no_grad():
                    clean = ae.encode(gt.float()).float()
                with wrapped.no_sync() if micro == 0 else nullcontext():
                    prediction = restore_endpoint(wrapped, ae, lq, context, context_ids, batched_prc_img)
                    loss, terms = paired.full.training_losses(prediction, clean, lq, gt, (gt + 1) * .5,
                        metric, batched_prc_img, weights)
                    objective = loss * config['training']['objective_scale']
                    if not bool(torch.isfinite(objective)):
                        raise RuntimeError('nonfinite objective')
                    (objective / 2).backward()
                for name, value in {'objective': objective, **terms}.items():
                    totals[name] = totals.get(name, 0.0) + float(value.detach()) / 2
            norm = torch.nn.utils.clip_grad_norm_(params, config['training']['clip_grad_norm'])
            if not math.isfinite(float(norm)):
                raise RuntimeError('nonfinite gradient')
            optimizer.step()
            assert_fp32_optimizer(optimizer)
            vector = torch.tensor(list(totals.values()), device=device, dtype=torch.float64)
            dist.all_reduce(vector); vector /= world
            torch.cuda.synchronize()
            times.append(time.monotonic() - tick)
            completed = step + 1
            if rank == 0 and (completed == start_step + 1 or completed % 10 == 0 or completed in stages['milestones']):
                print('BLUR_SINGLE_TRAIN ' + json.dumps({'step': completed, 'max_steps': stages['steps'],
                    'losses': dict(zip(totals, vector.tolist())), 'grad_norm': float(norm),
                    'seconds': times[-1], 'peak_memory_gib': torch.cuda.max_memory_allocated() / 1024**3}), flush=True)
            if completed % config['training']['checkpoint_every'] == 0 or completed in stages['milestones']:
                all_rng = [None] * world
                dist.all_gather_object(all_rng, rng_state())
                checkpoint = rank_zero_action(rank, lambda: save_state(model, optimizer, output,
                    step=completed, mode=args.mode, config_sha=CONFIG_SHA256, sources=sources,
                    rank_rng=all_rng, initial_parity_pass=True))
                if rank == 0:
                    checkpoints.append(checkpoint)
                    print('BLUR_SINGLE_CHECKPOINT ' + json.dumps(checkpoint), flush=True)
            if completed in stages['milestones']:
                last = assess(completed, final=completed == stages['steps'])
        observed = {name: any(not torch.equal(p.detach().flatten()[:64].cpu(), sampled_initial[n])
                             for n, p in model.named_parameters() if p.requires_grad and n.startswith(name + '.'))
                    for name in modules}
        flags = torch.tensor(int(all(observed.values())), device=device)
        dist.all_reduce(flags, op=dist.ReduceOp.MIN)
        if not bool(flags):
            raise RuntimeError(f'missing expert block update: {observed}')
        def finalize():
            elapsed = time.monotonic() - started
            result = {'status': 'complete', 'mode': args.mode, 'engineering_pass': True,
                'initial_parity_pass': True, 'config_sha256': CONFIG_SHA256, 'source_sha256': sources,
                'data': data, 'start_step': start_step, 'steps': stages['steps'], 'history': history,
                'checkpoints': checkpoints, 'resumed_from': str(args.resume_receipt) if resume else None,
                'blur_full_gate_pass': args.mode == 'formal' and last['psnr'] >= 25.9 and last['lpips'] <= .18,
                'qualification_is_engineering_only': args.mode == 'qualification',
                'trainable_parameters': count, 'sample_presentations_total': stages['steps'] * 16,
                'observed_block_updates': observed,
                'sample_presentations_this_segment': (stages['steps'] - start_step) * 16,
                'wall_clock_hours_this_segment': elapsed / 3600,
                'allocated_gpu_hours_this_segment': elapsed * world / 3600,
                'median_training_step_seconds': float(np.median(times[1:] or times)),
                'peak_memory_gib': torch.cuda.max_memory_allocated() / 1024**3,
                'gpu_name': prop.name, 'protected_pixels_accessed': False, 'automatic_next_launch': False,
                'pure_router_causal_ablation': False}
            write_json(output / 'result.json', result)
            print('BLUR_SINGLE_END ' + json.dumps({k: v for k, v in result.items()
                  if k not in ('history', 'checkpoints', 'source_sha256', 'data')}), flush=True)
        rank_zero_action(rank, finalize)
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
