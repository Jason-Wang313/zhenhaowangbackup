#!/usr/bin/env python3
"""True 10000->11000 image-only MoE continuation; one Blur loss change only."""
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
    names = ['smoke/moe_blur_lpips_continuation_v1.py',
             'smoke/train_foundir_moe_blur_lpips_continuation_v1_ddp.py',
             'smoke/run_foundir_moe_blur_lpips_pair_v1_8gpu.sh',
             'smoke/start_foundir_moe_blur_lpips_pair_v1_tmux.sh',
             'smoke/compare_foundir_rain_blur_joint_vlm_moe_v2.py',
             'smoke/cure_nccl_rehearsal_v1.py', 'smoke/tmux_launcher_lib.sh',
             'third_party/flux2/src/flux2/util.py']
    return {**parent.source_hashes(), **{p: sha(ROOT / p) for p in names}}


def contract():
    config = read_config()
    if sha(ROOT / config['parent_config']) != config['parent_config_sha256']:
        raise RuntimeError('parent config changed')
    original, pc, base, train, dev, conditions, data = parent.read_contract(ROOT / config['parent_config'])
    verify_parent(config, parent.source_hashes(), verify_artifacts=False)
    return config, original, pc, base, train, dev, conditions, data


def preflight():
    config, original, pc, base, train, dev, conditions, data = contract()
    verify_parent(config, parent.source_hashes())
    # Metadata and frozen weights only. GPU execution checks development bytes;
    # training bytes are lazily checked by the original loader before decoding.
    for key in ('flow_weights', 'autoencoder_weights', 'prompt_context_cache'):
        if sha(ROOT / base['model'][key]) != base['model'][key + '_sha256']:
            raise RuntimeError('frozen model changed: ' + key)
    if sha(ROOT / pc['rain_lora']) != pc['rain_lora_sha256']:
        raise RuntimeError('frozen Rain architecture initialization changed')
    payload = {'status': 'moe_lpips_continuation_preflight_pass', 'config_sha256': CONFIG_SHA256,
               'data': data, 'parent_receipt_sha256': config['parent_receipt_sha256'],
               'source_sha256': source_hashes(), 'next_sample_position': 320000,
               'images_opened': 0, 'GPU_qualification_completed': False}
    print('MOE_LPIPS_PREFLIGHT ' + json.dumps(payload), flush=True)
    return payload


def group_settings(optimizer):
    return [{k: v for k, v in group.items() if k != 'params'} for group in optimizer.param_groups]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=VARIANTS, default='control')
    parser.add_argument('--run-id', default='pair_v1')
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    if args.preflight:
        preflight()
        return
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.run_id):
        raise ValueError('invalid fresh pair ID')
    cfg, original, pc, base, training, development, conditions, data = contract()
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
    pair = ROOT / cfg['runtime']['output_root'] / args.run_id
    output = pair / args.variant
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
            if args.variant == 'control':
                pair.mkdir(parents=True, exist_ok=False)
                write_once(pair / 'pair_binding.json', pair_binding)
            else:
                if json.loads((pair / 'pair_binding.json').read_text()) != pair_binding:
                    raise RuntimeError('treatment must use the SAME node/GPUs/code as control')
                control = json.loads((pair / 'control/result.json').read_text())
                if not (control['status'] == 'complete' and control['engineering_pass']
                        and control['steps'] == 11000):
                    raise RuntimeError('control must finish before treatment')
            output.mkdir(exist_ok=False)
            write_once(output / 'launch.json', {**pair_binding, 'config': cfg, 'variant': args.variant,
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
        # This is explicitly an authorized branch from a COMPLETED parent.
        # The legacy loader permits it only when total_steps > saved step;
        # it verifies every saved weight/Adam/RNG binding before restoration.
        resumed = load_state(model, optimizer, ROOT / cfg['parent_receipt'], arm='image_only',
                             config_sha=parent.CONFIG_SHA256, sources=parent.source_hashes(),
                             rank=rank, total_steps=cfg['training']['end_step'])
        verify_optimizer_groups(optimizer, expected_groups)
        if resumed['step'] != 10000 or resumed['next_sample_position'] != 320000:
            raise RuntimeError('wrong continuation offset')
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
        initial_metrics = assess(10000, False)
        dataset = JointDataset(ROOT, training, conditions=conditions, steps=11000,
                               rank=rank, seed=seed, start_step=10000)
        loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2,
            prefetch_factor=2, multiprocessing_context='spawn', pin_memory=True,
            generator=torch.Generator().manual_seed(seed + rank + 111))
        iterator = iter(loader)
        # LPIPS module construction/evaluation must not consume restored train RNG.
        restore_rng(initial_rng)
        trace = hashlib.sha256()
        checkpoints, history, times = [], [], []
        initial = {n: p.detach().flatten()[:64].cpu().clone() for n, p in model.named_parameters() if p.requires_grad}
        weight_error, max_adjustment = 0., 0.
        for step in range(10000, 11000):
            tick = time.monotonic()
            model.train(); optimizer.zero_grad(set_to_none=True)
            totals = {}
            for micro in range(4):
                batch = next(iterator)
                position = int(batch['position'][0])
                task_id = int(batch['task_id'][0])
                if position != step * 32 + micro * 8 + rank or task_id != micro % 2:
                    raise RuntimeError('parent balanced sample sequence changed')
                trace.update(json.dumps([position, task_id, batch['record_id'][0]], separators=(',', ':')).encode() + b'\n')
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
            if rank == 0 and (completed == 10001 or completed % 10 == 0 or completed == 10025):
                print('MOE_LPIPS_TRAIN ' + json.dumps({'variant': args.variant, 'step': completed,
                    'additional_steps': completed - 10000, 'max_steps': 11000,
                    'losses': dict(zip(totals, vector.tolist())), 'grad_norm': float(norm),
                    'seconds': times[-1], 'peak_memory_gib': torch.cuda.max_memory_allocated() / 1024**3}), flush=True)
            if completed in cfg['runtime']['save_steps']:
                states = [None] * 8
                dist.all_gather_object(states, rng_state())
                def save():
                    checkpoints.append(save_state(model, optimizer, output, step=completed, mode='formal',
                        arm=args.variant, config_sha=CONFIG_SHA256, sources=sources, rank_rng=states))
                parent.rank_zero_action(rank, save)
            if completed in cfg['evaluation']['full_milestones']:
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
                'engineering_pass': engineering, 'start_step': 10000, 'steps': 11000,
                'optimizer_steps_this_run': 1000, 'this_launch_presentations': 32000,
                'next_sample_position': 352000, 'config_sha256': CONFIG_SHA256,
                'source_sha256': sources, 'data': data, 'hardware': hardware, 'gpu_names': gpu_names,
                'optimizer_group_settings': group_settings(optimizer),
                'parent_receipt_sha256': cfg['parent_receipt_sha256'],
                'parent_weights_sha256': cfg['parent_weights_sha256'],
                'parent_optimizer_sha256': cfg['parent_optimizer_sha256'],
                'initial_metrics': initial_metrics, 'history': history, 'final_metrics': final,
                'checkpoints': checkpoints, 'sample_trace_sha256_by_rank': [p['sample_trace_sha256'] for p in peers],
                'updates_by_rank': [p['changes'] for p in peers],
                'wall_hours': elapsed, 'allocated_gpu_hours': elapsed * 8,
                'median_step_seconds': float(np.median(times[1:])),
                'peak_memory_gib': max(p['peak_memory_gib'] for p in peers),
                'protected_pixels_accessed': False, 'automatic_extension': False}
            write_once(output / 'result.json', result)
            print('MOE_LPIPS_END ' + json.dumps({k: result[k] for k in (
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
