#!/usr/bin/env python3
"""GT-anchored endpoint student; frozen teacher targets, differentiable frozen VAE."""
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
from safetensors.torch import save_file
from torch.nn.parallel import DistributedDataParallel
from torch.utils.checkpoint import checkpoint

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party/diffsynth-studio"))
from lowlight_restoration.foundir_zimage_paired_flow import (
    load_frozen_rows, load_aligned_pair, sha256, verify_model_artifacts,
    validate_gpu_inventory, write_once,
)
from smoke.train_foundir_blur_only_zimage_teacher_paired_flow_v3_ddp import ZImagePairedFlowTeacherV3Module
from smoke.evaluate_foundir_blur_only_zimage_teacher_paired_flow_ddp import (
    psnr, lpips_value, aggregate, save_triptych,
)


def load_settings(path):
    settings = json.loads(path.read_text())
    if settings['status'] != 'authorized_blur_only_zimage_gt_anchored_endpoint_distillation':
        raise ValueError('unexpected endpoint experiment')
    for key in ('decoder_finetuning', 'qwen_degradation_conditioning', 'mixed_degradations',
                'sealed_internal_accessed', 'official_foundir_test_accessed'):
        if settings.get(key) is not False:
            raise ValueError(f'unsupported scope: {key}')
    if settings['student_steps'] != 1 or settings['teacher_steps'] != 50:
        raise ValueError('frozen teacher/student step count changed')
    if settings['steps'] != 400 or settings['save_steps'] != [0, 200, 400]:
        raise ValueError('bounded schedule changed')
    if settings['trainable_dtype'] != 'float32' or settings['teacher_filter'] != 'train_only_psnr_and_lpips_better_than_lq':
        raise ValueError('precision or teacher filter changed')
    for key in ('pixel_mse_weight', 'gt_latent_mse_weight', 'lpips_weight', 'teacher_latent_mse_weight', 'learning_rate'):
        if not math.isfinite(settings[key]) or settings[key] <= 0:
            raise ValueError(f'invalid loss/optimizer coefficient: {key}')
    parent = ROOT / settings['parent_config']
    if sha256(parent) != settings['parent_config_sha256']:
        raise ValueError('parent config identity changed')
    config = json.loads(parent.read_text())
    return settings, config


def teacher_is_usable(input_psnr, input_lpips, teacher_psnr, teacher_lpips):
    return teacher_psnr > input_psnr and teacher_lpips < input_lpips


def loss_terms(predicted_latent, decoded, item, metric, settings):
    pixel = torch.nn.functional.mse_loss(decoded.float(), item['gt_rgb'].float())
    latent = torch.nn.functional.mse_loss(predicted_latent.float(), item['gt_latent'].float())
    perceptual = metric((decoded.float().clamp(-1, 1) + 1) / 2,
                        (item['gt_rgb'].float() + 1) / 2).mean()
    teacher = torch.nn.functional.mse_loss(predicted_latent.float(), item['teacher_latent'].float())
    # The filter is computed exclusively from training GT before optimization.
    teacher = teacher * float(item['teacher_usable'])
    terms = dict(pixel_mse=pixel, gt_latent_mse=latent, lpips=perceptual, teacher_latent_mse=teacher)
    loss = sum(settings[name + '_weight'] * value for name, value in terms.items())
    return loss, terms


class EndpointStudent(ZImagePairedFlowTeacherV3Module):
    def promote_trainable(self):
        for param in self.parameters():
            if param.requires_grad:
                param.data = param.data.float()
        assert all(p.dtype == torch.float32 for p in self.parameters() if p.requires_grad)
        assert not any(p.requires_grad for p in self.pipe.vae_decoder.parameters())

    def velocity(self, latent, timestep, prompt, grad=False):
        return self.pipe.model_fn(dit=self.pipe.dit, controlnet=None, latents=latent,
                                 timestep=timestep, prompt_embeds=prompt,
                                 use_gradient_checkpointing=grad)

    def forward(self, item, prompt):
        lq = item['lq_latent']
        velocity = self.velocity(lq, lq.new_tensor([1000]), prompt, torch.is_grad_enabled())
        # FP32 subtraction preserves small endpoint corrections; decoder consumes BF16.
        latent = lq.float() - velocity.float()
        if torch.is_grad_enabled():
            decoded = checkpoint(self.pipe.vae_decoder, latent.to(self.pipe.torch_dtype), use_reentrant=False)
        else:
            decoded = self.pipe.vae_decoder(latent.to(self.pipe.torch_dtype))
        return latent, decoded


@torch.no_grad()
def prepare_item(model, row, prompt, metric, crop_size, teacher_steps, role):
    pipe, device = model.pipe, model.pipe.device
    lq, gt = load_aligned_pair(row, crop_size)
    lq_latent = pipe.vae_encoder(pipe.preprocess_image(lq)).detach()
    gt_rgb = pipe.preprocess_image(gt)
    gt_latent = pipe.vae_encoder(gt_rgb).detach()
    pipe.scheduler.set_timesteps(teacher_steps, shift=6.0)
    state = lq_latent.clone()
    for index, timestep in enumerate(pipe.scheduler.timesteps):
        v = model.velocity(state, timestep.reshape(1).to(device=device, dtype=pipe.torch_dtype), prompt)
        sigma = pipe.scheduler.sigmas[index].to(state)
        next_sigma = pipe.scheduler.sigmas[index + 1].to(state) if index + 1 < teacher_steps else state.new_tensor(0)
        state = state + v * (next_sigma - sigma)
    image = pipe.vae_output_to_image(pipe.vae_decoder(state)).convert('RGB')
    baseline = dict(record_id=row['record_id'], input_psnr=psnr(gt, lq), input_lpips=lpips_value(metric, gt, lq, device),
                    output_psnr=psnr(gt, image), output_lpips=lpips_value(metric, gt, image, device))
    usable = role == 'tiny_overfit_train' and teacher_is_usable(
        baseline['input_psnr'], baseline['input_lpips'], baseline['output_psnr'], baseline['output_lpips'])
    return dict(record_id=row['record_id'], lq=lq, gt=gt, lq_latent=lq_latent, gt_latent=gt_latent,
                gt_rgb=gt_rgb, teacher_latent=state.detach(), teacher_image=image,
                teacher_usable=usable, teacher_metrics=baseline)


def gather(local, world):
    if world == 1:
        return local
    parts = [None] * world
    dist.all_gather_object(parts, local)
    return [item for part in parts for item in part]


def write_json(path, value):
    write_once(path, (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--cpu-smoke', action='store_true')
    args = parser.parse_args()
    settings, config = load_settings(args.config)
    digest = sha256(args.config)
    teacher_path = ROOT / settings['teacher_checkpoint']
    rank, world = int(os.getenv('RANK', '0')), int(os.getenv('WORLD_SIZE', '1'))
    if rank == 0 and sha256(teacher_path) != settings['teacher_checkpoint_sha256']:
        raise RuntimeError('teacher checkpoint identity mismatch')
    train, dev = load_frozen_rows(config, verify_images=True)
    if args.preflight:
        verify_model_artifacts(config, large_hashes=True)
        for name, mem in [('NVIDIA H100 80GB HBM3', 81920), ('NVIDIA H200', 143000)]:
            validate_gpu_inventory([name] * 8, [mem] * 8, config)
        print('ENDPOINT_PREFLIGHT_PASS', digest, flush=True)
        return
    torch.set_num_threads(4)
    torch.manual_seed(settings['seed'])
    if args.cpu_smoke:
        device = torch.device('cpu')
        if world != 1:
            raise RuntimeError('CPU smoke is single process only')
    else:
        if world != 8:
            raise RuntimeError('exactly eight ranks required')
        device = torch.device('cuda', int(os.environ['LOCAL_RANK']))
        torch.cuda.set_device(device)
        validate_gpu_inventory([torch.cuda.get_device_properties(i).name for i in range(8)],
                               [torch.cuda.get_device_properties(i).total_memory // 2**20 for i in range(8)], config)
        dist.init_process_group('nccl', device_id=device)
    verify_model_artifacts(config, large_hashes=rank == 0)
    if world > 1:
        dist.barrier()
    output = ROOT / settings['output_dir']
    if not args.cpu_smoke:
        if output.exists() and any(output.iterdir()):
            raise RuntimeError('output exists; inspect before restarting')
        if rank == 0:
            output.mkdir(parents=True, exist_ok=True)
        dist.barrier()
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    metric = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True).to(device).eval()
    metric.requires_grad_(False)
    model = EndpointStudent(config, device, teacher_path)
    with torch.no_grad():
        prompt = model.pipe.units[1].process(model.pipe, config['model']['prompt'], None)['prompt_embeds']
    roles = [('tiny_overfit_train', train), ('visual_development', dev)]
    cached = {}
    for role, rows in roles[:1] if args.cpu_smoke else roles:
        selected = rows[:1] if args.cpu_smoke else rows[rank::world]
        cached[role] = [prepare_item(model, row, prompt, metric, 64 if args.cpu_smoke else 512,
                                    2 if args.cpu_smoke else 50, role) for row in selected]
        if not args.cpu_smoke:
            for item in cached[role]:
                save_triptych(output / 'teacher_reference' / role / (item['record_id'] + '.png'),
                              item['record_id'], item['lq'], item['teacher_image'], item['gt'])
    # Teacher targets were generated with the original BF16 LoRA before promotion.
    model.promote_trainable()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=settings['learning_rate'], weight_decay=0)
    if args.cpu_smoke:
        item = cached['tiny_overfit_train'][0]
        latent, decoded = model(item, prompt)
        loss, terms = loss_terms(latent, decoded, item, metric, settings)
        loss.backward()
        assert torch.isfinite(loss)
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
        assert all(p.grad is None for p in model.pipe.vae_decoder.parameters())
        before = params[1].detach().clone()
        optimizer.step()
        assert not torch.equal(before, params[1])
        assert all(optimizer.state[p]['exp_avg'].dtype == torch.float32 for p in params)
        print('ENDPOINT_REAL_CPU_SMOKE_PASS', json.dumps(dict(loss=float(loss.detach()),
              trainable_tensors=len(params), decoder_frozen=True, optimizer_dtype='float32')), flush=True)
        return
    training = DistributedDataParallel(model, device_ids=[device.index], find_unused_parameters=False)
    teacher_rows = gather([dict(role=role, **item['teacher_metrics'], teacher_usable=item['teacher_usable'])
                           for role, items in cached.items() for item in items], world)
    usable = sum(row['teacher_usable'] for row in teacher_rows)
    if rank == 0:
        write_json(output / 'teacher_targets.json', dict(config_sha256=digest, rows=teacher_rows,
                   teacher_usable_training_records=usable, teacher_checkpoint_sha256=settings['teacher_checkpoint_sha256']))
        print(f'ENDPOINT_TARGETS_READY teacher_usable={usable}/16', flush=True)
    snapshots = []
    def evaluate(step):
        with torch.no_grad():
            for role, items in cached.items():
                local = []
                for item in items:
                    _, decoded = model(item, prompt)
                    image = model.pipe.vae_output_to_image(decoded.to(model.pipe.torch_dtype)).convert('RGB')
                    local.append(dict(record_id=item['record_id'], input_psnr=psnr(item['gt'], item['lq']),
                        input_lpips=lpips_value(metric, item['gt'], item['lq'], device),
                        output_psnr=psnr(item['gt'], image), output_lpips=lpips_value(metric, item['gt'], image, device)))
                    save_triptych(output / f'step_{step:06d}' / role / (item['record_id'] + '.png'),
                                  item['record_id'], item['lq'], image, item['gt'])
                combined = sorted(gather(local, world), key=lambda r: r['record_id'])
                assert len(combined) == len({r['record_id'] for r in combined}) == 16
                if rank == 0:
                    payload = dict(step=step, role=role, metrics=aggregate(combined), rows=combined)
                    write_json(output / f'step_{step:06d}' / role / 'metrics.json', payload)
                    snapshots.append(payload)
                    print('ENDPOINT_SNAPSHOT', json.dumps({k:v for k,v in payload.items() if k != 'rows'}), flush=True)
        dist.barrier()
    evaluate(0)
    start = time.monotonic()
    for step in range(1, settings['steps'] + 1):
        # Two global batches visit the same sixteen frozen train crops once.
        item = cached['tiny_overfit_train'][(step - 1) % len(cached['tiny_overfit_train'])]
        optimizer.zero_grad(set_to_none=True)
        latent, decoded = training(item, prompt)
        loss, terms = loss_terms(latent, decoded, item, metric, settings)
        if not torch.isfinite(loss):
            raise RuntimeError('nonfinite endpoint loss')
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
        optimizer.step()
        metric.reset()
        values = torch.stack([loss.detach(), *[x.detach() for x in terms.values()]]).float()
        dist.all_reduce(values)
        values /= world
        if rank == 0 and (step == 1 or step % 10 == 0):
            entry = dict(step=step, loss=float(values[0]), terms=dict(zip(terms, values[1:].tolist())),
                         grad_norm=float(norm), elapsed_seconds=time.monotonic()-start)
            with (output / 'training_metrics.jsonl').open('a') as handle:
                handle.write(json.dumps(entry, allow_nan=False) + '\n')
            print('ENDPOINT_TRAIN', json.dumps(entry), flush=True)
        if step in settings['save_steps']:
            if rank == 0:
                state = {n.removeprefix('pipe.dit.'): p.detach().cpu().contiguous()
                         for n,p in model.named_parameters() if p.requires_grad}
                assert len(state) == 476 and all('lora_' in n for n in state)
                save_file(state, output / f'step_{step:06d}.safetensors',
                          metadata=dict(config_sha256=digest, step=str(step), trainable_dtype='float32'))
            dist.barrier()
            evaluate(step)
    if rank == 0:
        final_train = next(x for x in snapshots if x['step'] == settings['steps'] and x['role'] == 'tiny_overfit_train')
        result = dict(status='endpoint_student_complete', config_sha256=digest,
            checkpoint=str((output / 'step_000400.safetensors').relative_to(ROOT)),
            checkpoint_sha256=sha256(output / 'step_000400.safetensors'),
            teacher_checkpoint_sha256=settings['teacher_checkpoint_sha256'], teacher_usable_training_records=usable,
            classification='gt_anchored_filtered_distillation' if usable else 'supervised_single_step_recovery',
            snapshots=snapshots, automatic_metric_gate_pass=final_train['metrics']['psnr_gain_db'] > 0 and final_train['metrics']['lpips_improvement'] > 0,
            sealed_internal_accessed=False, official_foundir_test_accessed=False, decoder_finetuned=False,
            hardware_name=torch.cuda.get_device_properties(device).name,
            trainable_dtype='float32', frozen_weight_dtype='bfloat16',
            world_size=world, global_batch_size=8, steps=settings['steps'], wall_seconds=time.monotonic()-start)
        write_json(output / 'result.json', result)
        print('ENDPOINT_COMPLETE', json.dumps({k:v for k,v in result.items() if k!='snapshots'}), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
