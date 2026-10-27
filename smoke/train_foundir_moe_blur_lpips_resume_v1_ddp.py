#!/usr/bin/env python3
"""Isolated new-node resume of interrupted Blur-LPIPS arm: 10500->11000."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
import hashlib
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

from smoke import train_foundir_rain_blur_joint_vlm_moe_v2_ddp as parent
from smoke import train_foundir_moe_blur_lpips_continuation_v1_ddp as previous
from smoke.moe_blur_lpips_resume_v1 import (
    CONFIG_SHA256 as MIGRATION_SHA256, verify_migration, trace_event, reconstruct_trace_prefix,
)
from smoke.moe_blur_lpips_continuation_v1 import (
    CONFIG_SHA256, VARIANTS, read_config, sha, write_once, loss_weights,
    verify_parent, verify_optimizer_groups, hardware_identity,
)
from lowlight_restoration.joint_moe_state import load_state, save_state, rng_state
from lowlight_restoration.blur_single_endpoint import restore_rng
from lowlight_restoration.joint_vlm_moe import (
    install_joint_moe, enable_block_checkpointing, joint_conditioning, JointDataset, restoration_loss,
)
from lowlight_restoration.lora import inject_lora, load_lora_checkpoint
from lowlight_restoration.flux2_endpoint_precision import promote_lora_to_fp32, restore_endpoint, assert_fp32_optimizer
from lowlight_restoration.rain_blur_image_moe import router_classification_loss


def source_hashes():
    names = ['smoke/moe_blur_lpips_resume_v1.py',
             'smoke/train_foundir_moe_blur_lpips_resume_v1_ddp.py',
             'smoke/run_foundir_moe_blur_lpips_resume_v1_8gpu.sh',
             'smoke/start_foundir_moe_blur_lpips_resume_v1_tmux.sh']
    return {**previous.source_hashes(), **{p: sha(ROOT / p) for p in names}}


def preflight():
    original = previous.preflight()
    migration, receipt, old_launch = verify_migration(previous.source_hashes())
    print('MOE_MIGRATION_PREFLIGHT ' + json.dumps({
        'status': 'pass', 'migration_config_sha256': MIGRATION_SHA256,
        'resume_receipt_sha256': migration['resume_receipt_sha256'],
        'source_sha256': source_hashes(), 'restored_step': receipt['step'],
        'next_sample_position': receipt['next_sample_position'], 'remaining_steps': 500,
        'images_opened': 0, 'GPU_execution_verified': False}), flush=True)
    return original


def group_settings(optimizer):
    return [{k: v for k, v in group.items() if k != 'params'} for group in optimizer.param_groups]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=('blur_lpips4x',), default='blur_lpips4x')
    parser.add_argument('--run-id', default='resume_v1')
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    if args.preflight:
        preflight()
        return
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.run_id):
        raise ValueError('invalid fresh pair ID')
    if os.environ.get('OLD_MOE_STOPPED') != '1':
        raise RuntimeError('confirm old job stopped with OLD_MOE_STOPPED=1; stale logs alone are not proof')
    cfg, original, pc, base, training, development, conditions, data = previous.contract()
    migration, resume_binding, old_launch = verify_migration(previous.source_hashes(), artifacts=False)
    sources = source_hashes()
    rank, local = int(os.environ['RANK']), int(os.environ['LOCAL_RANK'])
    if (int(os.environ['WORLD_SIZE']) != 8 or int(os.environ['LOCAL_WORLD_SIZE']) != 8
            or torch.cuda.device_count() != 8):
        raise RuntimeError('requires a single eight-GPU node')
    torch.cuda.set_device(local)
    device = torch.device('cuda', local)
    props = torch.cuda.get_device_properties(device)
    if props.total_memory < 75 * 1024**3 or not any(n in props.name for n in ('H100', 'H200')):
        raise RuntimeError('requires H100/H200 >=75 GiB')
    dist.init_process_group('nccl', device_id=device, timeout=timedelta(minutes=30))
    started = time.monotonic()
    output = ROOT / migration['output_root'] / args.run_id
    try:
        gpu_names = [None] * 8
        dist.all_gather_object(gpu_names, props.name)
        if len(set(gpu_names)) != 1:
            raise RuntimeError('requires homogeneous GPUs')
        hardware = hardware_identity()
        pair_binding = {'hardware': hardware, 'gpu_names': gpu_names,
                        'config_sha256': CONFIG_SHA256, 'source_sha256': sources}
        def prepare():
            def check_development(row):
                if row['split'] != 'validation':
                    raise RuntimeError('non-development evaluation row')
                for side in ('lq', 'gt'):
                    if sha(ROOT / row[side + '_path']) != row[side + '_sha256']:
                        raise RuntimeError('development bytes changed: ' + row['record_id'])
            with ThreadPoolExecutor(max_workers=8) as pool:
                list(pool.map(check_development, development['rain'] + development['blur_full']))
            output.mkdir(parents=True, exist_ok=False)
            write_once(output / 'launch.json', {**pair_binding, 'config': cfg, 'variant': args.variant,
                       'migration_config': migration, 'migration_config_sha256': MIGRATION_SHA256,
                       'old_launch_sha256': migration['old_launch_sha256'],
                       'old_hardware': old_launch['hardware'], 'old_gpu_names': old_launch['gpu_names'],
                       'old_job_stopped_is_user_ack_not_remotely_verified': True,
                       'data': data, 'parent_receipt_sha256': cfg['parent_receipt_sha256']})
        parent.rank_zero_action(rank, prepare)
        seed = original['training']['seed']
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
            TORCH_HOME=str(ROOT / base['runtime']['torch_home']),
            KLEIN_4B_BASE_MODEL_PATH=str(ROOT / base['model']['flow_weights']),
            AE_MODEL_PATH=str(ROOT / base['model']['autoencoder_weights']))
        from flux2.util import load_ae, load_flow_model
        from flux2.sampling import batched_prc_img
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
        cache = load_file(str(ROOT / base['model']['prompt_context_cache']))
        context, ids = cache['cfg_context'][:1].to(device), cache['cfg_ids'][:1].to(device)
        del cache
        ae = load_ae(base['model']['name'], device=device).eval().requires_grad_(False)
        model = load_flow_model(base['model']['name'], device=device).eval()
        names = inject_lora(model, base['lora']['target_patterns'], rank=32, alpha=32, dropout=0)
        if len(names) != 50:
            raise RuntimeError('shared LoRA structure changed')
        promote_lora_to_fp32(model)
        load_lora_checkpoint(model, ROOT / pc['rain_lora'])
        modules = install_joint_moe(model, arm='image_only', indices=original['architecture']['ffn_blocks'])
        enable_block_checkpointing(model)
        optimizer = torch.optim.AdamW(parent.parameter_groups(model, original), weight_decay=0, fused=True)
        expected_groups = group_settings(optimizer)
        # Restore the interrupted treatment, not the original step10000 control.
        resumed = load_state(model, optimizer, ROOT / migration['resume_receipt'], arm='blur_lpips4x',
                             config_sha=CONFIG_SHA256, sources=previous.source_hashes(),
                             rank=rank, total_steps=11000)
        verify_optimizer_groups(optimizer, expected_groups)
        if resumed['step'] != 10500 or resumed['next_sample_position'] != 336000:
            raise RuntimeError('wrong migration sampling offset')
        initial_rng = rng_state()
        params = [p for group in optimizer.param_groups for p in group['params']]
        wrapped = DDP(model, device_ids=[local], broadcast_buffers=False, gradient_as_bucket_view=True)
        metric = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True).to(device).eval().requires_grad_(False)
        def assess(step, full):
            results = {}
            for split in ('rain', 'blur_full' if full else 'blur_preview'):
                results[split] = parent.evaluate(model, ae, development[split], conditions, context, ids,
                    batched_prc_img, metric, rank=rank, world=8, device=device,
                    output=output, split=split, step=step)
            return results
        initial_metrics = assess(10500, True)
        dataset = JointDataset(ROOT, training, conditions=conditions, steps=11000,
                               rank=rank, seed=seed, start_step=10500)
        loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2,
            prefetch_factor=2, multiprocessing_context='spawn', pin_memory=True,
            generator=torch.Generator().manual_seed(seed + rank + 111))
        iterator = iter(loader)
        # LPIPS module construction/evaluation must not consume restored train RNG.
        restore_rng(initial_rng)
        trace = hashlib.sha256()
        reconstruct_trace_prefix(dataset, trace, 10500)
        checkpoints, history, times = [], [], []
        initial = {n: p.detach().flatten()[:64].cpu().clone() for n, p in model.named_parameters() if p.requires_grad}
        weight_error, max_adjustment = 0., 0.
        for step in range(10500, 11000):
            tick = time.monotonic()
            model.train(); optimizer.zero_grad(set_to_none=True)
            totals = {}
            for micro in range(4):
                batch = next(iterator)
                position = int(batch['position'][0])
                task_id = int(batch['task_id'][0])
                if position != step * 32 + micro * 8 + rank or task_id != micro % 2:
                    raise RuntimeError('parent balanced sample sequence changed')
                trace_event(trace, position, task_id, batch['record_id'][0])
                lq, gt = batch['lq'].to(device), batch['gt'].to(device)
                labels, condition = batch['task_id'].to(device), batch['condition'].to(device)
                weights = loss_weights(original['training']['losses'], args.variant, task_id)
                with torch.no_grad():
                    clean = ae.encode(gt.float()).float()
                with wrapped.no_sync() if micro < 3 else nullcontext():
                    with joint_conditioning(model, condition, labels=labels, fraction=0.):
                        restored = restore_endpoint(wrapped, ae, lq, context, ids, batched_prc_img)
                        loss, terms = restoration_loss(restored, clean, gt, metric, batched_prc_img, weights)
                        ce = router_classification_loss(model, labels)
                        objective = loss + original['training']['router_ce_weight'] * ce
                        if not bool(torch.isfinite(objective)):
                            raise RuntimeError('nonfinite objective')
                        for module in modules.values():
                            weight_error = max(weight_error, float((module.last_weights.detach().sum(-1) - 1).abs().max()))
                            max_adjustment = max(max_adjustment, float(module.last_adjustment.detach().abs().max()))
                        (objective / 4).backward()
                task = 'rain' if task_id == 0 else 'blur'
                for name, value in {'objective': objective, 'router_ce': ce, **terms}.items():
                    key = task + '_' + name
                    totals[key] = totals.get(key, 0.) + float(value.detach()) / 2
            if any(p.grad is None for p in params):
                raise RuntimeError('disconnected trainable parameter')
            norm = torch.nn.utils.clip_grad_norm_(params, original['training']['clip_grad_norm'])
            if not math.isfinite(float(norm)):
                raise RuntimeError('nonfinite gradients')
            optimizer.step(); assert_fp32_optimizer(optimizer)
            vector = torch.tensor(list(totals.values()), device=device, dtype=torch.float64)
            dist.all_reduce(vector); vector /= 8
            torch.cuda.synchronize(); times.append(time.monotonic() - tick)
            completed = step + 1
            if rank == 0 and (completed == 10501 or completed % 10 == 0 or completed == 10525):
                print('MOE_MIGRATION_TRAIN ' + json.dumps({'variant': args.variant, 'step': completed,
                    'additional_steps': completed - 10000, 'steps_after_resume': completed - 10500, 'max_steps': 11000,
                    'losses': dict(zip(totals, vector.tolist())), 'grad_norm': float(norm),
                    'seconds': times[-1], 'peak_memory_gib': torch.cuda.max_memory_allocated() / 1024**3}), flush=True)
            if completed in migration['save_steps']:
                states = [None] * 8
                dist.all_gather_object(states, rng_state())
                def save():
                    checkpoints.append(save_state(model, optimizer, output, step=completed, mode='formal',
                        arm=args.variant, config_sha=CONFIG_SHA256, sources=sources, rank_rng=states))
                parent.rank_zero_action(rank, save)
            if completed in migration['evaluation_steps']:
                eval_rng = rng_state()
                final = assess(completed, True)
                restore_rng(eval_rng)
                if rank == 0:
                    history.append(final)
        changes = {part: any(not torch.equal(p.detach().flatten()[:64].cpu(), initial[name])
                    for name, p in model.named_parameters() if p.requires_grad and part in name)
                   for part in ('experts.0', 'experts.1', '.router.', '.lora_A.', '.lora_B.')}
        peers = [None] * 8
        dist.all_gather_object(peers, {'changes': changes, 'sample_trace_sha256': trace.hexdigest(),
            'max_weight_error': weight_error, 'max_adjustment': max_adjustment,
            'peak_memory_gib': torch.cuda.max_memory_allocated() / 1024**3})
        def finish():
            engineering = (all(all(p['changes'].values()) for p in peers)
                           and max(p['max_weight_error'] for p in peers) <= 1e-5
                           and all(p['max_adjustment'] == 0 for p in peers))
            elapsed = (time.monotonic() - started) / 3600
            result = {'status': 'complete', 'variant': args.variant, 'model_arm': 'image_only',
                'engineering_pass': engineering, 'start_step': 10500, 'steps': 11000,
                'optimizer_steps_this_run': 500, 'this_launch_presentations': 16000,
                'next_sample_position': 352000, 'config_sha256': CONFIG_SHA256,
                'source_sha256': sources, 'data': data, 'hardware': hardware, 'gpu_names': gpu_names,
                'migration_config_sha256': MIGRATION_SHA256,
                'resume_receipt_sha256': migration['resume_receipt_sha256'],
                'original_pair_source_sha256': previous.source_hashes(),
                'sample_trace_prefix_reconstructed_from_frozen_schedule': True,
                'hardware_history': [
                    {'through_step': 10500, 'hardware': old_launch['hardware'], 'gpu_names': old_launch['gpu_names']},
                    {'through_step': 11000, 'hardware': hardware, 'gpu_names': gpu_names}],
                'strict_same_node_comparison': False,
                'optimizer_group_settings': group_settings(optimizer),
                'parent_receipt_sha256': cfg['parent_receipt_sha256'],
                'parent_weights_sha256': cfg['parent_weights_sha256'],
                'parent_optimizer_sha256': cfg['parent_optimizer_sha256'],
                'resume_initial_metrics': initial_metrics, 'history': history, 'final_metrics': final,
                'checkpoints': checkpoints, 'sample_trace_sha256_by_rank': [p['sample_trace_sha256'] for p in peers],
                'updates_by_rank': [p['changes'] for p in peers],
                'wall_hours': elapsed, 'allocated_gpu_hours': elapsed * 8,
                'median_step_seconds': float(np.median(times[1:])),
                'peak_memory_gib': max(p['peak_memory_gib'] for p in peers),
                'protected_pixels_accessed': False, 'automatic_extension': False}
            write_once(output / 'result.json', result)
            print('MOE_MIGRATION_END ' + json.dumps({k: result[k] for k in (
                'variant', 'steps', 'engineering_pass', 'wall_hours', 'allocated_gpu_hours',
                'median_step_seconds', 'peak_memory_gib')}), flush=True)
            if not engineering:
                raise RuntimeError('continuation engineering checks failed')
        parent.rank_zero_action(rank, finish)
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
