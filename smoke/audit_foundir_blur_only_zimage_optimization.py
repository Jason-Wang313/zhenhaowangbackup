#!/usr/bin/env python3
"""Eight independent training-only probes: latent/LoRA x RGB/mixed x two records."""
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
from safetensors.torch import save_file, load_file
from torch.utils.checkpoint import checkpoint

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from smoke.train_foundir_blur_only_zimage_endpoint_distill_ddp import (
    EndpointStudent, load_settings, load_frozen_rows, load_aligned_pair, sha256,
    verify_model_artifacts, validate_gpu_inventory, write_json, psnr, lpips_value,
    save_triptych, loss_terms,
)
from smoke.train_foundir_blur_only_zimage_decoder_tail_ddp import restore_student_fp32


def load_audit(path):
    s = json.loads(path.read_text())
    for key in ('decoder_finetuning', 'development_gt_optimized', 'qwen_degradation_conditioning',
                'mixed_degradations', 'sealed_internal_accessed', 'official_foundir_test_accessed'):
        if s.get(key) is not False:
            raise ValueError(f'forbidden scope: {key}')
    if s['arms'] != ['latent_rgb', 'latent_mixed', 'lora_rgb', 'lora_mixed']:
        raise ValueError('arm assignment changed')
    if s['record_ids'] != ['foundir_train_0042483', 'foundir_train_0091486']:
        raise ValueError('record assignment changed')
    if s['steps'] != 200 or s['snapshot_steps'] != [0, 1, 20, 100, 200]:
        raise ValueError('diagnostic schedule changed')
    for key in ('latent_learning_rate', 'lora_learning_rate'):
        if not math.isfinite(s[key]) or s[key] <= 0:
            raise ValueError(key)
    for field in ('endpoint_config', 'student_checkpoint', 'student_result', 'teacher_targets'):
        if sha256(ROOT / s[field]) != s[field + '_sha256']:
            raise ValueError(f'identity mismatch: {field}')
    e, c = load_settings(ROOT / s['endpoint_config'])
    teacher = json.loads((ROOT / s['teacher_targets']).read_text())
    rejected = {r['record_id'] for r in teacher['rows'] if r['role'] == 'tiny_overfit_train' and not r['teacher_usable']}
    if not set(s['record_ids']).issubset(rejected):
        raise ValueError('mixed arms require rejected teacher targets')
    return s, e, c


def assignment(rank, s):
    if not 0 <= rank < 8:
        raise ValueError('invalid probe rank')
    return s['arms'][rank // 2], s['record_ids'][rank % 2]


def objective(terms, arm, e):
    if arm.endswith('_rgb'):
        return terms['pixel_mse']
    return sum(e[k + '_weight'] * v for k,v in terms.items())


def gradient_summary(z, terms, e):
    gradients = {k: torch.autograd.grad(v, z, retain_graph=True)[0].detach().float().flatten()
                 for k,v in terms.items() if k != 'teacher_latent_mse'}
    norms = {k: float(g.norm()) for k,g in gradients.items()}
    pairs = {}
    for a in gradients:
        for b in gradients:
            if a < b:
                denom = norms[a] * norms[b]
                pairs[a + '__' + b] = float(torch.dot(gradients[a], gradients[b])) / denom if denom else None
    if not all(math.isfinite(v) for v in norms.values()):
        raise RuntimeError('nonfinite gradient probe')
    return dict(latent_gradient_norms=norms,
                weighted_latent_gradient_norms={k: v * e[k + '_weight'] for k,v in norms.items()},
                latent_gradient_cosines=pairs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--preflight', action='store_true')
    p.add_argument('--cpu-smoke', action='store_true')
    p.add_argument('--arm-index', type=int, default=4)
    a = p.parse_args()
    s, e, c = load_audit(a.config)
    train, _ = load_frozen_rows(c, verify_images=True)
    selected = {r['record_id']: r for r in train if r['record_id'] in s['record_ids']}
    assert set(selected) == set(s['record_ids'])
    if a.preflight:
        verify_model_artifacts(c, large_hashes=True)
        for name, mem in [('NVIDIA H100 80GB HBM3',81920), ('NVIDIA H200',143000)]:
            validate_gpu_inventory([name]*8, [mem]*8, c)
        print('OPT_AUDIT_PREFLIGHT_PASS', sha256(a.config), flush=True)
        return
    rank, world = int(os.getenv('RANK',0)), int(os.getenv('WORLD_SIZE',1))
    probe_rank = a.arm_index if a.cpu_smoke else rank
    arm, record_id = assignment(probe_rank, s)
    torch.set_num_threads(4)
    torch.manual_seed(s['seed'])
    if a.cpu_smoke:
        assert world == 1
        device = torch.device('cpu')
    else:
        if world != 8:
            raise ValueError('eight independent probe ranks required')
        device = torch.device('cuda', int(os.environ['LOCAL_RANK']))
        torch.cuda.set_device(device)
        props = [torch.cuda.get_device_properties(i) for i in range(8)]
        validate_gpu_inventory([x.name for x in props], [x.total_memory//2**20 for x in props], c)
        dist.init_process_group('nccl', device_id=device)
    verify_model_artifacts(c, large_hashes=rank == 0)
    root = ROOT / s['output_dir']
    if not a.cpu_smoke:
        if rank == 0:
            root.mkdir(parents=True, exist_ok=False)
        dist.barrier()
    folder = root / arm / record_id
    model = EndpointStudent(c, device, ROOT / s['student_checkpoint'])
    restore_student_fp32(model, ROOT / s['student_checkpoint'])
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    metric = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True).to(device).eval().requires_grad_(False)
    with torch.no_grad():
        prompt = model.pipe.units[1].process(model.pipe, c['model']['prompt'], None)['prompt_embeds']
        lq, gt = load_aligned_pair(selected[record_id], 64 if a.cpu_smoke else 512)
        gt_rgb = model.pipe.preprocess_image(gt)
        item = dict(lq_latent=model.pipe.vae_encoder(model.pipe.preprocess_image(lq)),
                    gt_rgb=gt_rgb, gt_latent=model.pipe.vae_encoder(gt_rgb), teacher_usable=False)
        item['teacher_latent'] = torch.zeros_like(item['gt_latent'])
        initial_z, initial_rgb = model(item, prompt)
        ceiling = model.pipe.vae_output_to_image(model.pipe.vae_decoder(item['gt_latent'])).convert('RGB')
    # Separate decoder/latent gradient probe, independent of DiT parameter gradients.
    probe_z = initial_z.detach().clone().requires_grad_(True)
    probe_rgb = model.pipe.vae_decoder(probe_z.to(model.pipe.torch_dtype))
    probe_loss, terms = loss_terms(probe_z, probe_rgb, item, metric, e)
    gradients = gradient_summary(probe_z, terms, e)
    metric.reset()
    del probe_rgb, probe_z, probe_loss, terms
    if arm.startswith('latent_'):
        free_z = torch.nn.Parameter(initial_z.detach().clone())
        params = [free_z]
        lr = s['latent_learning_rate']
        def forward(grad):
            rgb = checkpoint(model.pipe.vae_decoder, free_z.to(model.pipe.torch_dtype), use_reentrant=False) if grad else model.pipe.vae_decoder(free_z.to(model.pipe.torch_dtype))
            return free_z, rgb
    else:
        for name, param in model.pipe.dit.named_parameters():
            param.requires_grad_('lora_' in name)
        params = [p for p in model.parameters() if p.requires_grad]
        assert len(params) == 476 and all(p.dtype == torch.float32 for p in params)
        lr = s['lora_learning_rate']
        def forward(grad):
            return model(item, prompt)
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0)
    snapshots = []
    def evaluate(step):
        with torch.no_grad():
            z, rgb = forward(False)
            image = model.pipe.vae_output_to_image(rgb).convert('RGB')
            row = dict(step=step, psnr=psnr(gt,image), lpips=lpips_value(metric,gt,image,device),
                       raw_rgb_mse=float((rgb.float()-gt_rgb.float()).square().mean()),
                       gt_latent_mse=float((z.float()-item['gt_latent'].float()).square().mean()),
                       clipped_fraction=float((rgb.abs()>1).float().mean()))
            if not all(math.isfinite(v) for v in row.values()):
                raise RuntimeError('nonfinite probe metrics')
            if step == 0 and not a.cpu_smoke:
                previous = json.loads((ROOT/s['student_result']).read_text())
                baseline = next(r for x in previous['snapshots'] if x['step']==400 and x['role']=='tiny_overfit_train' for r in x['rows'] if r['record_id']==record_id)
                if abs(row['psnr']-baseline['output_psnr'])>.02 or abs(row['lpips']-baseline['output_lpips'])>.002:
                    raise RuntimeError('frozen endpoint reproduction failed')
            if not a.cpu_smoke:
                save_triptych(folder/f'step_{step:06d}.png', record_id + ' | ' + arm + ' TRAIN-GT DIAGNOSTIC', lq,image,gt)
                write_json(folder/f'step_{step:06d}.json', row)
            snapshots.append(row)
            print('OPT_AUDIT_SNAPSHOT', json.dumps(dict(rank=probe_rank,arm=arm,record_id=record_id,**row)),flush=True)
    evaluate(0)
    start = time.monotonic()
    first_update = None
    for step in range(1, (2 if a.cpu_smoke else s['steps'])+1):
        optimizer.zero_grad(set_to_none=True)
        z, rgb = forward(True)
        if arm.endswith('_rgb'):
            loss = (rgb.float()-gt_rgb.float()).square().mean()
        else:
            _, terms = loss_terms(z,rgb,item,metric,e)
            loss = objective(terms,arm,e)
        if not torch.isfinite(loss):
            raise RuntimeError('nonfinite probe loss')
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
        if step == 1:
            if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in params):
                raise RuntimeError('missing/nonfinite probe gradients')
            before = [p.detach().clone() for p in params]
        optimizer.step()
        metric.reset()
        if step == 1:
            changed = sum(int(torch.count_nonzero(p.detach()!=b)) for p,b in zip(params,before))
            delta_sq = sum(float((p.detach()-b).square().sum()) for p,b in zip(params,before))
            first_update = dict(changed_parameters=changed, total_parameters=sum(p.numel() for p in params), l2_update=math.sqrt(delta_sq))
            del before
            assert changed > 0 and all(optimizer.state[p]['exp_avg'].dtype==torch.float32 for p in params)
            assert all(p.grad is None for p in model.pipe.vae_decoder.parameters())
        if step == 1 or step % 20 == 0:
            print('OPT_AUDIT_TRAIN',json.dumps(dict(rank=probe_rank,arm=arm,record_id=record_id,step=step,loss=float(loss.detach()),grad_norm=float(norm),elapsed=time.monotonic()-start)),flush=True)
        if step in s['snapshot_steps']:
            evaluate(step)
    if a.cpu_smoke:
        print('OPT_AUDIT_REAL_CPU_SMOKE_PASS',arm,json.dumps(dict(first_update=first_update,gradients=gradients)),flush=True)
        return
    state = {'oracle_latent':params[0].detach().cpu().contiguous()} if arm.startswith('latent_') else {
        n.removeprefix('pipe.dit.'):p.detach().cpu().contiguous() for n,p in model.named_parameters() if p.requires_grad}
    path = folder/'final.safetensors'
    save_file(state,path,metadata=dict(config_sha256=sha256(a.config),arm=arm,record_id=record_id,diagnostic_only='true'))
    saved = load_file(str(path))
    assert set(saved)==set(state) and all(torch.equal(state[n],saved[n]) for n in state)
    result = dict(rank=rank,arm=arm,record_id=record_id,classification='training_GT_latent_oracle_not_deployable' if arm.startswith('latent_') else 'single_training_image_fit_not_generalization',
        checkpoint=str(path.relative_to(ROOT)),checkpoint_sha256=sha256(path),config_sha256=sha256(a.config),student_checkpoint_sha256=s['student_checkpoint_sha256'],
        input_psnr=psnr(gt,lq),input_lpips=lpips_value(metric,gt,lq,device),ceiling_psnr=psnr(gt,ceiling),ceiling_lpips=lpips_value(metric,gt,ceiling,device),
        gradients=gradients,first_update=first_update,snapshots=snapshots,learning_rate=lr,steps=s['steps'],batch_size=1,wall_seconds=time.monotonic()-start)
    write_json(folder/'result.json',result)
    all_results=[None]*world
    dist.all_gather_object(all_results,result)
    if rank==0:
        assert len(all_results)==8 and {(r['arm'],r['record_id']) for r in all_results}=={assignment(i,s) for i in range(8)}
        write_json(root/'result.json',dict(status='optimization_audit_complete',config_sha256=sha256(a.config),
            student_checkpoint=s['student_checkpoint'],student_checkpoint_sha256=s['student_checkpoint_sha256'],
            probes=all_results,diagnostic_only=True,independent_optimizers=True,world_size=8,hardware_name=torch.cuda.get_device_properties(device).name,
            sealed_internal_accessed=False,official_foundir_test_accessed=False,development_gt_optimized=False,decoder_finetuned=False))
        print('OPT_AUDIT_COMPLETE',sha256(root/'result.json'),flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__=='__main__':
    main()
