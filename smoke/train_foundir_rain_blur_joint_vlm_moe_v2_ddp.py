#!/usr/bin/env python3
"""Matched image-only/VLM-diagnosis MoE with shared spatial adaptation."""
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
from safetensors.torch import load_file
from torch.nn.parallel import DistributedDataParallel as DDP

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'third_party/flux2/src'))
from smoke import train_foundir_rain_blur_image_moe_v1_ddp as previous
from smoke.train_foundir_blur_single_endpoint_v1_ddp import rank_zero_action
from lowlight_restoration.flux2_endpoint_precision import promote_lora_to_fp32, restore_endpoint, assert_fp32_optimizer
from lowlight_restoration.lora import inject_lora, load_lora_checkpoint
from lowlight_restoration.rain_blur_image_moe import image_moe_modules, warmup_fraction, router_classification_loss
from lowlight_restoration.joint_vlm_moe import (
    ARMS, join_conditions, install_joint_moe, enable_block_checkpointing,
    joint_conditioning, JointDataset, restoration_loss,
)
from lowlight_restoration.joint_moe_state import save_state, load_state, rng_state

old, write_json = previous.old, previous.write_json
CONFIG_SHA256 = '3bdd59b3fb3c010cc7a93ab299f3f67fe85cc5eb5a6f380f610d4338398baebe'


def source_hashes():
    names = ['lowlight_restoration/joint_vlm_moe.py', 'lowlight_restoration/joint_moe_state.py',
             'lowlight_restoration/blur_single_endpoint.py',
             'smoke/train_foundir_blur_single_endpoint_v1_ddp.py',
             'smoke/train_foundir_rain_blur_joint_vlm_moe_v2_ddp.py',
             'smoke/run_foundir_rain_blur_joint_vlm_moe_v2_8gpu.sh',
             'smoke/start_foundir_rain_blur_joint_vlm_moe_v2_tmux.sh',
             'third_party/flux2/src/flux2/model.py', 'third_party/flux2/src/flux2/sampling.py',
             'third_party/flux2/src/flux2/autoencoder.py']
    return {**previous.source_hashes(), **{name: old.sha256(ROOT / name) for name in names}}


def read_contract(path):
    if old.sha256(path) != CONFIG_SHA256:
        raise RuntimeError('joint protocol changed; do not mutate a launched contract')
    config = json.loads(path.read_text())
    if old.sha256(ROOT / config['parent_config']) != config['parent_config_sha256']:
        raise RuntimeError('parent data/init contract changed')
    parent_config, parent, training, development, metadata = previous.read_contract(ROOT / config['parent_config'])
    condition_path = ROOT / config['conditions']['path']
    if old.sha256(condition_path) != config['conditions']['sha256']:
        raise RuntimeError('VLM prediction cache changed')
    conditions = join_conditions(condition_path, training['rain'] + training['blur']
                                 + development['rain'] + development['blur_full'])
    data = {**metadata, 'conditions_sha256': config['conditions']['sha256'],
            'joined_predictions': len(conditions), 'joined_predictions_digest': previous.full.digest(conditions)}
    return config, parent_config, parent, training, development, conditions, data


def preflight(path):
    config, pc, parent, training, development, conditions, data = read_contract(path)
    previous.preflight(ROOT / config['parent_config'])
    cache = load_file(str(ROOT / parent['model']['prompt_context_cache']), device='cpu')
    if cache['cfg_context'].shape != (2, 512, 7680) or cache['cfg_ids'].shape != (2, 512, 4):
        raise RuntimeError('empty-context schema changed')
    if not bool(torch.isfinite(cache['cfg_context']).all()):
        raise RuntimeError('nonfinite prompt cache')
    payload = {'status': 'joint_vlm_moe_cpu_preflight_pass', 'config_sha256': CONFIG_SHA256,
               'data': data, 'condition_kind': 'predicted_five_levels_not_raw_tokens',
               'GPU_qualification_completed': False, 'protected_pixels_accessed': False}
    print('JOINT_MOE_PREFLIGHT ' + json.dumps(payload), flush=True)
    return payload


def parameter_groups(model, config):
    bins = {k: [] for k in ('shared_lora', 'experts', 'router', 'vlm')}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.dtype != torch.float32:
            raise RuntimeError('all trainable tensors must be FP32')
        if '.lora_A.' in name or '.lora_B.' in name:
            key = 'shared_lora'
        elif '.experts.' in name:
            key = 'experts'
        elif '.vlm_projector.' in name:
            key = 'vlm'
        elif '.router.' in name:
            key = 'router'
        else:
            raise RuntimeError(f'unexpected trainable tensor {name}')
        bins[key].append(p)
    keys = {'shared_lora': 'shared_lora_lr', 'experts': 'expert_lr', 'router': 'router_lr', 'vlm': 'router_lr'}
    return [{'params': params, 'lr': config['training'][keys[k]], 'name': k}
            for k, params in bins.items() if params]


def require_qualifications(paths, sources, data):
    if paths is None or len(paths) != 2:
        raise RuntimeError('formal requires BOTH matching arm engineering qualifications, not a quality gate')
    seen = set()
    for path in paths:
        q = json.loads(Path(path).read_text())
        if not (q['status'] == 'complete' and q['mode'] == 'qualification' and q['engineering_pass']
                and q['config_sha256'] == CONFIG_SHA256 and q['source_sha256'] == sources
                and q['data'] == data and q['steps'] == 25 and q['initial_insertion_parity_pass']
                and q['arm'] in ARMS and q['arm'] not in seen):
            raise RuntimeError('qualification receipt mismatch')
        checkpoint = q['checkpoints'][-1]
        for key in ('weights', 'optimizer'):
            if old.sha256(Path(path).parent / checkpoint[key]) != checkpoint[key + '_sha256']:
                raise RuntimeError('qualification checkpoint identity changed')
        seen.add(q['arm'])


def evaluate(model, ae, rows, conditions, context, ids, pack, metric, *, rank, world,
             device, output, split, step, disable_vlm=False, force_expert=None):
    model.eval()
    modules = image_moe_modules(model)
    for module in modules.values():
        if module.labels is not None or module.teacher_fraction:
            raise RuntimeError('training label state leaked into evaluation')
        module.force_expert = force_expert
    route = 'vlm_off' if disable_vlm else ('learned' if force_expert is None else f'constant_{force_expert}')
    local = []
    try:
        for index in range(rank, len(rows), world):
            row = rows[index]
            lq, gt = [x[None].to(device) for x in old.load_crop(row, 512)]
            condition = torch.tensor([conditions[row['record_id']]], device=device)
            with torch.inference_mode(), joint_conditioning(model, condition, disable_vlm=disable_vlm):
                result = restore_endpoint(model, ae, lq, context, ids, pack)['decoded']
                if not bool(torch.isfinite(result).all()):
                    raise RuntimeError('nonfinite restored image')
                quantized = ((result + 1) * 127.5).round().clamp(0, 255) / 127.5 - 1
                entry = {'record_id': row['record_id'], 'source_group': row['source_group'],
                         'input_psnr': old.psnr(gt, lq), 'output_psnr': old.psnr(gt, quantized),
                         'input_lpips': old.lpips_value(metric, gt, lq),
                         'output_lpips': old.lpips_value(metric, gt, quantized),
                         'expert_weights': {k: m.last_weights[0].cpu().tolist() for k, m in modules.items()},
                         'vlm_adjustment_max': max(float(m.last_adjustment.abs().max()) for m in modules.values())}
            local.append(entry)
            if index < 2 and route == 'learned':
                old.save_triptych(output / 'images' / split / f'{step:06d}_{index:03d}.png', row['record_id'], lq, quantized, gt)
        gathered = [None] * world
        dist.all_gather_object(gathered, local)
    finally:
        for module in modules.values():
            module.force_expert = None
    if rank != 0:
        return None
    records = sorted([r for shard in gathered for r in shard], key=lambda r: r['record_id'])
    if len(records) != len(rows) or {r['record_id'] for r in records} != {r['record_id'] for r in rows}:
        raise RuntimeError('evaluation identity mismatch')
    result = {'split': split, 'step': step, 'route': route, 'records': len(records), 'rows': records,
              'psnr': float(np.mean([r['output_psnr'] for r in records])),
              'lpips': float(np.mean([r['output_lpips'] for r in records])),
              'mean_expert_weights': {k: np.mean([r['expert_weights'][k] for r in records], axis=0).tolist()
                                      for k in modules}, 'GT_used_for_routing': False}
    write_json(output / 'metrics' / f'{split}_{step:06d}_{route}.json', result)
    print('JOINT_MOE_METRICS ' + json.dumps({k: v for k, v in result.items() if k != 'rows'}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/foundir_rain_blur_joint_vlm_moe_v2.json')
    parser.add_argument('--arm', choices=ARMS, default='image_only')
    parser.add_argument('--mode', choices=('qualification', 'formal'), default='qualification')
    parser.add_argument('--run-id', default='qualification_v1')
    parser.add_argument('--qualifications', type=Path, nargs=2)
    parser.add_argument('--resume-receipt', type=Path)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    if args.preflight:
        preflight(args.config)
        return
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.run_id) or (args.resume_receipt and args.mode != 'formal'):
        raise RuntimeError('invalid run ID or qualification resume')
    config, pc, parent, training, development, conditions, data = read_contract(args.config)
    sources = source_hashes()
    if args.mode == 'formal':
        require_qualifications(args.qualifications, sources, data)
    rank, world, local = int(os.environ['RANK']), int(os.environ['WORLD_SIZE']), int(os.environ['LOCAL_RANK'])
    if world != 8 or int(os.environ.get('LOCAL_WORLD_SIZE', 0)) != 8 or torch.cuda.device_count() != 8:
        raise RuntimeError('requires exactly one eight-GPU node')
    torch.cuda.set_device(local)
    device = torch.device('cuda', local)
    prop = torch.cuda.get_device_properties(device)
    if prop.total_memory < 75 * 1024**3 or not any(x in prop.name for x in ('H100', 'H200')):
        raise RuntimeError('requires H100/H200 with >=75 GiB')
    dist.init_process_group('nccl', device_id=device, timeout=timedelta(minutes=30))
    output = ROOT / config['runtime']['output_root'] / args.arm / args.run_id
    started = time.monotonic()
    try:
        gpu_names = [None] * world
        dist.all_gather_object(gpu_names, prop.name)
        if len(set(gpu_names)) != 1:
            raise RuntimeError('mixed GPU hardware')
        def prepare():
            if output.exists():
                raise RuntimeError('existing output; use a fresh run ID, including for resume')
            preflight(args.config)
            output.mkdir(parents=True)
            write_json(output / 'receipt.json', {'config': config, 'config_sha256': CONFIG_SHA256,
                       'source_sha256': sources, 'data': data, 'arm': args.arm, 'mode': args.mode})
        rank_zero_action(rank, prepare)
        seed = config['training']['seed']
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TORCH_HOME=str(ROOT / parent['runtime']['torch_home']),
            KLEIN_4B_BASE_MODEL_PATH=str(ROOT / parent['model']['flow_weights']), AE_MODEL_PATH=str(ROOT / parent['model']['autoencoder_weights']))
        from flux2.util import load_ae, load_flow_model
        from flux2.sampling import batched_prc_img
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
        cache = load_file(str(ROOT / parent['model']['prompt_context_cache']), device='cpu')
        context, ids = cache['cfg_context'][:1].to(device), cache['cfg_ids'][:1].to(device)
        del cache
        ae = load_ae(parent['model']['name'], device=device).eval().requires_grad_(False)
        model = load_flow_model(parent['model']['name'], device=device).eval()
        names = inject_lora(model, parent['lora']['target_patterns'], rank=32, alpha=32, dropout=0)
        if len(names) != 50:
            raise RuntimeError('shared LoRA target mismatch')
        promote_lora_to_fp32(model)
        load_lora_checkpoint(model, ROOT / pc['rain_lora'])
        # Compare insertion before any training on one allowed row per task/rank.
        probes, expected = [], []
        for split in ('rain', 'blur_preview'):
            row = development[split][rank]
            lq = old.load_crop(row, 512)[0][None].to(device)
            condition = torch.tensor([conditions[row['record_id']]], device=device)
            probes.append((lq, condition))
            with torch.inference_mode():
                expected.append(restore_endpoint(model, ae, lq, context, ids, batched_prc_img)['decoded'].cpu())
        modules = install_joint_moe(model, arm=args.arm, indices=config['architecture']['ffn_blocks'])
        # New modules are created in training mode; put the whole model in eval.
        model.eval()
        maximum_error = 0.0
        for (lq, condition), reference in zip(probes, expected):
            with torch.inference_mode(), joint_conditioning(model, condition):
                actual = restore_endpoint(model, ae, lq, context, ids, batched_prc_img)['decoded'].cpu()
            maximum_error = max(maximum_error, float((actual - reference).abs().max()))
        err = torch.tensor(maximum_error, device=device)
        dist.all_reduce(err, op=dist.ReduceOp.MAX)
        if not bool(torch.isfinite(err)) or float(err) > 1e-5:
            raise RuntimeError('MoE insertion changed initialized empty-context restoration')
        del probes, expected
        enable_block_checkpointing(model)
        groups = parameter_groups(model, config)
        params = [p for group in groups for p in group['params']]
        counts = {g['name']: sum(p.numel() for p in g['params']) for g in groups}
        if counts['shared_lora'] != 32440320 or counts['experts'] != 849346560:
            raise RuntimeError('shared/expert parameter count changed')
        optimizer = torch.optim.AdamW(groups, weight_decay=0, fused=True)
        stages = config['modes'][args.mode]
        resume = None
        if args.resume_receipt:
            resume = load_state(model, optimizer, args.resume_receipt, arm=args.arm, config_sha=CONFIG_SHA256,
                                sources=sources, rank=rank, total_steps=stages['steps'])
        start_step = resume['step'] if resume else 0
        wrapped = DDP(model, device_ids=[local], broadcast_buffers=False, gradient_as_bucket_view=True)
        metric = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True).to(device).eval().requires_grad_(False)
        evaluation_args = dict(rank=rank, world=world, device=device, output=output)
        history, checkpoints = [], []
        def assess(step, final=False):
            result = {}
            for split in ('rain', 'blur_full' if final and args.mode == 'formal' else 'blur_preview'):
                result[split] = evaluate(model, ae, development[split], conditions, context, ids, batched_prc_img,
                                        metric, split=split, step=step, **evaluation_args)
            if rank == 0:
                history.append(result)
            return result
        assess(start_step)
        dataset = JointDataset(ROOT, training, conditions=conditions, steps=stages['steps'], rank=rank,
                               seed=seed, start_step=start_step)
        loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2,
            prefetch_factor=2, multiprocessing_context='spawn', pin_memory=True,
            generator=torch.Generator().manual_seed(seed + rank + 111))
        iterator = iter(loader)
        initial = {n: p.detach().flatten()[:64].cpu().clone() for n, p in model.named_parameters() if p.requires_grad}
        times, seen = [], {'rain': set(), 'blur': set()}
        weight_error, adjustment_max = 0.0, 0.0
        for step in range(start_step, stages['steps']):
            tick = time.monotonic()
            model.train(); optimizer.zero_grad(set_to_none=True)
            totals = {}
            fraction = warmup_fraction(step, stages['hard_steps'], stages['transition_end'])
            for micro in range(4):
                batch = next(iterator)
                if int(batch['position'][0]) != step * 32 + micro * world + rank:
                    raise RuntimeError('sampling position mismatch')
                seen['rain' if int(batch['task_id'][0]) == 0 else 'blur'].add(batch['record_id'][0])
                lq, gt = batch['lq'].to(device), batch['gt'].to(device)
                labels, condition = batch['task_id'].to(device), batch['condition'].to(device)
                with torch.no_grad():
                    clean = ae.encode(gt.float()).float()
                with wrapped.no_sync() if micro < 3 else nullcontext():
                    # Keep routing state until checkpoint recomputation/backward finishes.
                    with joint_conditioning(model, condition, labels=labels, fraction=fraction):
                        result = restore_endpoint(wrapped, ae, lq, context, ids, batched_prc_img)
                        loss, terms = restoration_loss(result, clean, gt, metric, batched_prc_img, config['training']['losses'])
                        ce = router_classification_loss(model, labels)
                        objective = loss + config['training']['router_ce_weight'] * ce
                        if not bool(torch.isfinite(objective)):
                            raise RuntimeError('nonfinite training loss')
                        for module in modules.values():
                            weight_error = max(weight_error, float((module.last_weights.detach().sum(-1) - 1).abs().max()))
                            adjustment_max = max(adjustment_max, float(module.last_adjustment.detach().abs().max()))
                        (objective / 4).backward()
                for k, value in {'objective': objective, 'router_ce': ce, **terms}.items():
                    totals[k] = totals.get(k, 0.0) + float(value.detach()) / 4
            if any(p.grad is None for p in params):
                raise RuntimeError('trainable parameter disconnected from objective')
            norm = torch.nn.utils.clip_grad_norm_(params, config['training']['clip_grad_norm'])
            if not math.isfinite(float(norm)):
                raise RuntimeError('nonfinite gradients')
            optimizer.step(); assert_fp32_optimizer(optimizer)
            vector = torch.tensor(list(totals.values()), device=device, dtype=torch.float64)
            dist.all_reduce(vector); vector /= world
            torch.cuda.synchronize(); times.append(time.monotonic() - tick)
            completed = step + 1
            if rank == 0 and (completed == start_step + 1 or completed % 10 == 0 or completed in stages['milestones']):
                print('JOINT_MOE_TRAIN ' + json.dumps({'arm': args.arm, 'step': completed, 'max_steps': stages['steps'],
                    'losses': dict(zip(totals, vector.tolist())), 'grad_norm': float(norm), 'seconds': times[-1],
                    'peak_memory_gib': torch.cuda.max_memory_allocated() / 1024**3}), flush=True)
            if completed % config['runtime']['save_every'] == 0 or completed == 250 or completed in stages['milestones']:
                states = [None] * world
                dist.all_gather_object(states, rng_state())
                def save():
                    ck = save_state(model, optimizer, output, step=completed, mode=args.mode, arm=args.arm,
                                    config_sha=CONFIG_SHA256, sources=sources, rank_rng=states)
                    checkpoints.append(ck)
                rank_zero_action(rank, save)
            if completed in stages['milestones']:
                last = assess(completed, final=completed == stages['steps'])
        observed = {}
        for part in ('experts.0', 'experts.1', '.router.', '.lora_A.', '.lora_B.', '.vlm_projector.'):
            observed[part] = any(not torch.equal(p.detach().flatten()[:64].cpu(), initial[n])
                                for n, p in model.named_parameters() if p.requires_grad and part in n)
        diagnostics = {}
        if args.mode == 'formal':
            for split in ('rain', 'blur_preview'):
                for fixed in (0, 1):
                    diagnostics[f'{split}_constant_{fixed}'] = evaluate(model, ae, development[split], conditions,
                        context, ids, batched_prc_img, metric, split=split, step=stages['steps'], force_expert=fixed, **evaluation_args)
                if args.arm == 'vlm_assisted':
                    diagnostics[split + '_vlm_off'] = evaluate(model, ae, development[split], conditions,
                        context, ids, batched_prc_img, metric, split=split, step=stages['steps'], disable_vlm=True, **evaluation_args)
        gathered = [None] * world
        dist.all_gather_object(gathered, {'observed': observed, 'weight_error': weight_error,
            'adjustment_max': adjustment_max, 'peak_memory_gib': torch.cuda.max_memory_allocated() / 1024**3,
            'seen': seen})
        def finish():
            required = ['experts.0', 'experts.1', '.router.', '.lora_A.', '.lora_B.']
            if args.arm == 'vlm_assisted':
                required.append('.vlm_projector.')
            engineering = (all(all(r['observed'][k] for k in required) for r in gathered)
                           and max(r['weight_error'] for r in gathered) <= 1e-5
                           and (args.arm != 'image_only' or max(r['adjustment_max'] for r in gathered) == 0))
            elapsed = time.monotonic() - started
            result = {'status': 'complete', 'arm': args.arm, 'mode': args.mode, 'steps': stages['steps'],
                'engineering_pass': engineering, 'initial_insertion_parity_pass': True,
                'insertion_max_abs_error': float(err), 'config_sha256': CONFIG_SHA256,
                'source_sha256': sources, 'data': data, 'parameter_counts': counts,
                'history': history, 'checkpoints': checkpoints, 'diagnostics': diagnostics,
                'resume_parent': str(args.resume_receipt) if resume else None,
                'sample_presentations': stages['steps'] * 32, 'this_launch_presentations': (stages['steps'] - start_step) * 32,
                'unique_records_seen_this_launch': {k: len(set.union(*(r['seen'][k] for r in gathered))) for k in seen},
                'updates_by_rank': [r['observed'] for r in gathered],
                'max_weight_sum_error': max(r['weight_error'] for r in gathered),
                'max_vlm_adjustment': max(r['adjustment_max'] for r in gathered),
                'peak_memory_gib_all_ranks': max(r['peak_memory_gib'] for r in gathered),
                'median_training_step_seconds': float(np.median(times[1:] or times)),
                'wall_clock_hours': elapsed / 3600, 'allocated_gpu_hours': elapsed * world / 3600,
                'gpu_name': prop.name, 'protected_pixels_accessed': False,
                'absolute_quality_gate_pass': args.mode == 'formal' and last['rain']['psnr'] >= 34.2
                    and last['rain']['lpips'] <= .08 and last['blur_full']['psnr'] >= 25.9 and last['blur_full']['lpips'] <= .18,
                'VLM_advantage_requires_matched_other_arm': True, 'automatic_next_launch': False}
            write_json(output / 'result.json', result)
            print('JOINT_MOE_END ' + json.dumps({k: v for k, v in result.items()
                if k not in ('history', 'diagnostics', 'source_sha256', 'checkpoints', 'updates_by_rank')}), flush=True)
            if not engineering:
                raise RuntimeError('engineering checks failed; inspect complete result')
        rank_zero_action(rank, finish)
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
