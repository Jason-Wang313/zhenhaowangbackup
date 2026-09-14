#!/usr/bin/env python3
"""Controlled decoder-tail adaptation with an immutable endpoint student."""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
from safetensors.torch import load_file, save_file
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.checkpoint import checkpoint

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from smoke.train_foundir_blur_only_zimage_endpoint_distill_ddp import (
    EndpointStudent, load_settings, load_frozen_rows, load_aligned_pair,
    verify_model_artifacts, validate_gpu_inventory, sha256, write_json,
    gather, psnr, lpips_value, aggregate, save_triptych,
)


def settings_for(path):
    s = json.loads(path.read_text())
    for key in ('qwen_degradation_conditioning', 'mixed_degradations',
                'sealed_internal_accessed', 'official_foundir_test_accessed'):
        if s.get(key) is not False:
            raise ValueError(f'protected scope: {key}')
    if s.get('decoder_finetuning') is not True or s['steps'] != 400 or s['save_steps'] != [0, 200, 400]:
        raise ValueError('decoder scope/schedule changed')
    for key in ('learning_rate', 'lpips_weight', 'gt_anchor_weight'):
        if not math.isfinite(s[key]) or s[key] <= 0:
            raise ValueError(key)
    for field in ('endpoint_config', 'student_checkpoint', 'student_result'):
        if sha256(ROOT / s[field]) != s[field + '_sha256']:
            raise ValueError(f'identity mismatch: {field}')
    endpoint, config = load_settings(ROOT / s['endpoint_config'])
    return s, endpoint, config


def restore_student_fp32(model, path):
    """Parent constructor casts LoRA to BF16: restore original FP32 AFTER promotion."""
    model.promote_trainable()
    params = {n.removeprefix('pipe.dit.'): p for n, p in model.named_parameters() if p.requires_grad}
    saved = load_file(str(path))
    if set(params) != set(saved) or any(t.dtype != torch.float32 for t in saved.values()):
        raise ValueError('student checkpoint keys/precision mismatch')
    with torch.no_grad():
        for n, p in params.items():
            p.copy_(saved[n])
            if not torch.equal(p.cpu(), saved[n]):
                raise RuntimeError(f'inexact student restore: {n}')
    model.requires_grad_(False)


class DecoderTail(torch.nn.Module):
    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder.requires_grad_(False)
        for block in decoder.blocks[15:]:
            block.float().requires_grad_(True)
        decoder.conv_norm_out.float().requires_grad_(True)
        decoder.conv_out.float().requires_grad_(True)

    def decode(self, z):
        d = self.decoder
        with torch.no_grad():
            h = z.to(d.conv_in.weight.dtype) / d.scaling_factor + d.shift_factor
            h = d.conv_in(h)
            for block in d.blocks[:15]:
                h, _, _, _ = block(h, None, None, None)
        h = h.float()
        for block in d.blocks[15:]:
            def run(x, layer=block):
                return layer(x, None, None, None)[0]
            h = checkpoint(run, h, use_reentrant=False) if torch.is_grad_enabled() else run(h)
        return d.conv_out(d.conv_act(d.conv_norm_out(h)))

    def forward(self, predicted, gt_latent):
        return self.decode(predicted), self.decode(gt_latent)


def decoder_loss(output, anchor, gt, metric, s):
    pixel = (output.float() - gt.float()).square().mean()
    perceptual = metric((output.clamp(-1, 1) + 1) / 2, (gt.float() + 1) / 2).mean()
    reconstruction = (anchor.float() - gt.float()).square().mean()
    return pixel + s['lpips_weight'] * perceptual + s['gt_anchor_weight'] * reconstruction


def acceptance(snapshots):
    train = next(x['metrics'] for x in snapshots if x['step'] == 400 and x['role'] == 'tiny_overfit_train')
    dev = next(x for x in snapshots if x['step'] == 400 and x['role'] == 'visual_development')
    zero = next(x['metrics'] for x in snapshots if x['step'] == 0 and x['role'] == 'visual_development')
    m = dev['metrics']
    return (train['psnr_gain_db'] > 0 and train['lpips_improvement'] > 0
            and m['psnr_gain_db'] > 0 and m['lpips_improvement'] > 0
            and m['output_psnr'] > zero['output_psnr'] and m['output_lpips'] < zero['output_lpips']
            and dev['gt_roundtrip_psnr'] >= 35 and dev['gt_roundtrip_lpips'] <= .02)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--preflight', action='store_true')
    p.add_argument('--cpu-smoke', action='store_true')
    a = p.parse_args()
    s, endpoint, config = settings_for(a.config)
    train, dev = load_frozen_rows(config, verify_images=True)
    rank, world = int(os.getenv('RANK', 0)), int(os.getenv('WORLD_SIZE', 1))
    if a.preflight:
        verify_model_artifacts(config, large_hashes=True)
        for name, mem in [('NVIDIA H100 80GB HBM3', 81920), ('NVIDIA H200', 143000)]:
            validate_gpu_inventory([name] * 8, [mem] * 8, config)
        print('DECODER_PREFLIGHT_PASS', sha256(a.config), flush=True)
        return
    torch.set_num_threads(4)
    torch.manual_seed(s['seed'])
    if a.cpu_smoke:
        assert world == 1
        device = torch.device('cpu')
    else:
        if world != 8:
            raise ValueError('requires eight ranks')
        device = torch.device('cuda', int(os.environ['LOCAL_RANK']))
        torch.cuda.set_device(device)
        props = [torch.cuda.get_device_properties(i) for i in range(8)]
        validate_gpu_inventory([x.name for x in props], [x.total_memory // 2**20 for x in props], config)
        dist.init_process_group('nccl', device_id=device)
    verify_model_artifacts(config, large_hashes=rank == 0)
    if world > 1:
        dist.barrier()
    root = ROOT / s['output_dir']
    if not a.cpu_smoke:
        # Rank zero alone checks/creates output, avoiding a directory-creation race.
        if rank == 0:
            root.mkdir(parents=True, exist_ok=False)
        dist.barrier()
    model = EndpointStudent(config, device, ROOT / s['student_checkpoint'])
    restore_student_fp32(model, ROOT / s['student_checkpoint'])
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    metric = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True).to(device).eval().requires_grad_(False)
    cached = {}
    with torch.no_grad():
        prompt = model.pipe.units[1].process(model.pipe, config['model']['prompt'], None)['prompt_embeds']
        for role, rows in [('tiny_overfit_train', train), ('visual_development', dev)][:1 if a.cpu_smoke else 2]:
            items = []
            for row in rows[:1] if a.cpu_smoke else rows[rank::world]:
                lq, gt = load_aligned_pair(row, 64 if a.cpu_smoke else 512)
                rgb = model.pipe.preprocess_image(gt)
                z = model.pipe.vae_encoder(model.pipe.preprocess_image(lq))
                predicted, original = model({'lq_latent': z}, prompt)
                gt_latent = model.pipe.vae_encoder(rgb).detach()
                items.append(dict(record_id=row['record_id'], lq=lq, gt=gt, gt_rgb=rgb,
                                  predicted=predicted.detach(), gt_latent=gt_latent,
                                  original=original.detach(), original_anchor=model.pipe.vae_decoder(gt_latent).detach()))
            cached[role] = items
    tail = DecoderTail(model.pipe.vae_decoder)
    params = [p for p in tail.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=s['learning_rate'], weight_decay=0)
    def update(wrapper, item):
        optimizer.zero_grad(set_to_none=True)
        out, anchor = wrapper(item['predicted'], item['gt_latent'])
        loss = decoder_loss(out, anchor, item['gt_rgb'], metric, s)
        if not torch.isfinite(loss):
            raise RuntimeError('nonfinite decoder loss')
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
        optimizer.step()
        metric.reset()
        return loss.detach(), norm
    if a.cpu_smoke:
        before = params[-1].detach().clone()
        loss, norm = update(tail, cached['tiny_overfit_train'][0])
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
        assert not torch.equal(before, params[-1])
        assert all(optimizer.state[p]['exp_avg'].dtype == torch.float32 for p in params)
        assert all(p.grad is None for p in model.pipe.dit.parameters())
        assert all(p.grad is None for p in tail.decoder.blocks[:15].parameters())
        print('DECODER_REAL_CPU_SMOKE_PASS', float(loss), float(norm), sum(p.numel() for p in params), flush=True)
        return
    wrapper = DDP(tail, device_ids=[device.index])
    snapshots = []
    @torch.no_grad()
    def evaluate(step):
        for role, items in cached.items():
            local = []
            for item in items:
                decoded, anchor = tail(item['predicted'], item['gt_latent'])
                if step == -1:
                    decoded = item['original']
                    anchor = item['original_anchor']
                image = model.pipe.vae_output_to_image(decoded).convert('RGB')
                anchor_image = model.pipe.vae_output_to_image(anchor).convert('RGB')
                local.append(dict(record_id=item['record_id'], input_psnr=psnr(item['gt'], item['lq']),
                    input_lpips=lpips_value(metric, item['gt'], item['lq'], device), output_psnr=psnr(item['gt'], image),
                    output_lpips=lpips_value(metric, item['gt'], image, device),
                    gt_roundtrip_psnr=psnr(item['gt'], anchor_image), gt_roundtrip_lpips=lpips_value(metric, item['gt'], anchor_image, device)))
                folder = 'frozen_baseline' if step == -1 else f'step_{step:06d}'
                save_triptych(root / folder / role / (item['record_id'] + '.png'), item['record_id'], item['lq'], image, item['gt'])
            rows = sorted(gather(local, world), key=lambda x: x['record_id'])
            expected = train if role == 'tiny_overfit_train' else dev
            assert len(rows) == 16 and {r['record_id'] for r in rows} == {r['record_id'] for r in expected}
            assert all(math.isfinite(v) for r in rows for k,v in r.items() if k != 'record_id')
            if rank == 0:
                if step == -1:
                    prior = json.loads((ROOT / s['student_result']).read_text())
                    previous = next(x['rows'] for x in prior['snapshots'] if x['step'] == 400 and x['role'] == role)
                    by_id = {r['record_id']: r for r in previous}
                    for row in rows:
                        old = by_id[row['record_id']]
                        if abs(row['output_psnr'] - old['output_psnr']) > .02 or abs(row['output_lpips'] - old['output_lpips']) > .002:
                            raise RuntimeError('frozen student baseline differs from endpoint result')
                value = dict(step=step, role=role, rows=rows, metrics=aggregate(rows),
                    gt_roundtrip_psnr=sum(r['gt_roundtrip_psnr'] for r in rows)/16,
                    gt_roundtrip_lpips=sum(r['gt_roundtrip_lpips'] for r in rows)/16)
                write_json(root / folder / role / 'metrics.json', value)
                snapshots.append(value)
                print('DECODER_SNAPSHOT', json.dumps({k:v for k,v in value.items() if k != 'rows'}), flush=True)
        dist.barrier()
    evaluate(-1)
    evaluate(0)
    start = time.monotonic()
    for step in range(1, s['steps'] + 1):
        item = cached['tiny_overfit_train'][(step-1) % len(cached['tiny_overfit_train'])]
        loss, norm = update(wrapper, item)
        dist.all_reduce(loss)
        if rank == 0 and (step == 1 or step % 10 == 0):
            print('DECODER_TRAIN', json.dumps(dict(step=step, loss=float(loss)/world, grad_norm=float(norm), elapsed=time.monotonic()-start)), flush=True)
        if step in s['save_steps']:
            if rank == 0:
                state = {n:p.detach().cpu().contiguous() for n,p in tail.named_parameters() if p.requires_grad}
                path = root / f'step_{step:06d}.safetensors'
                save_file(state, path, metadata=dict(config_sha256=sha256(a.config), student_sha256=s['student_checkpoint_sha256']))
                loaded = load_file(str(path))
                assert set(loaded) == set(state) and all(torch.equal(loaded[n], state[n]) for n in state)
            dist.barrier()
            evaluate(step)
    if rank == 0:
        value = dict(status='decoder_tail_complete', config_sha256=sha256(a.config),
            checkpoint=str((root/'step_000400.safetensors').relative_to(ROOT)), checkpoint_sha256=sha256(root/'step_000400.safetensors'),
            student_checkpoint=s['student_checkpoint'], student_checkpoint_sha256=s['student_checkpoint_sha256'],
            snapshots=snapshots, automatic_metric_gate_pass=acceptance(snapshots), visual_review_required=True,
            decoder_finetuned=True, sealed_internal_accessed=False, official_foundir_test_accessed=False,
            trainable_parameters=sum(p.numel() for p in params), trainable_dtype='float32', steps=400, global_batch_size=8,
            world_size=world, hardware_name=torch.cuda.get_device_properties(device).name, wall_seconds=time.monotonic()-start)
        write_json(root/'result.json', value)
        print('DECODER_COMPLETE', json.dumps({k:v for k,v in value.items() if k != 'snapshots'}), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
