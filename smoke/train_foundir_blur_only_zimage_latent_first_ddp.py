#!/usr/bin/env python3
"""Round-2 latent-first supervised endpoint correction, eight synchronized GPUs."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import torch
import torch.distributed as dist
from safetensors.torch import save_file,load_file
from torch.utils.checkpoint import checkpoint

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from lowlight_restoration.zimage_latent_first import average_gradients,joint_phase,loss_terms,metric_gate,render_report,validate_settings
from lowlight_restoration.zimage_repro_batch import seed_everything,install_flash_control,tensor_digest
from lowlight_restoration.zimage_padding_compat import build_padding_function,padding_operator_probe
from smoke.train_foundir_blur_only_zimage_endpoint_distill_ddp import (
    EndpointStudent,load_settings,load_frozen_rows,load_aligned_pair,verify_model_artifacts,
    validate_gpu_inventory,sha256,write_json,write_once,psnr,lpips_value,aggregate,save_triptych,gather,
)
from smoke.train_foundir_blur_only_zimage_decoder_tail_ddp import restore_student_fp32


class LatentFirstStudent(EndpointStudent):
    def forward(self,item,prompt,decode=True):
        lq=item['lq_latent']
        velocity=self.velocity(lq,lq.new_tensor([1000]),prompt,torch.is_grad_enabled())
        latent=lq.float()-velocity.float()
        if not decode:
            return latent,None
        rgb=(checkpoint(self.pipe.vae_decoder,latent.to(self.pipe.torch_dtype),use_reentrant=False)
             if torch.is_grad_enabled() else self.pipe.vae_decoder(latent.to(self.pipe.torch_dtype)))
        return latent,rgb


def load_config(path):
    s=json.loads(path.read_text())
    validate_settings(s)
    for field in ('parent_endpoint_config','initial_checkpoint','parent_result','repro_result'):
        if sha256(ROOT/s[field]) != s[field+'_sha256']:
            raise ValueError(f'round-2 parent identity mismatch: {field}')
    e,c=load_settings(ROOT/s['parent_endpoint_config'])
    repro=json.loads((ROOT/s['repro_result']).read_text())
    if repro['decision']['deterministic_same_gpu_control_exact'] is not True or repro['completed_fits'] != 24:
        raise ValueError('round-1 reproducibility not passed')
    if (ROOT/s['output_dir']).resolve() != ROOT/'checkpoints/foundir_blur_only_zimage_latent_first_v1':
        raise ValueError('unexpected output root')
    return s,e,c


def restore(model,named,initial,seed,metric):
    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        for name,p in named:
            p.copy_(initial[name].to(p.device))
    seed_everything(seed)
    metric.reset()


def update(model,params,item,prompt,metric,s,optimizer,joint,world):
    optimizer.zero_grad(set_to_none=True)
    latent,rgb=model(item,prompt,decode=joint)
    loss,terms=loss_terms(latent,rgb,item,metric,s,joint)
    if not torch.isfinite(loss):
        raise RuntimeError('nonfinite training loss')
    loss.backward()
    average_gradients(params,world)
    norm=torch.nn.utils.clip_grad_norm_(params,s['max_grad_norm'],error_if_nonfinite=True)
    # Rehearsal hashes these post-clipping averaged gradients.
    optimizer.step()
    metric.reset()
    return latent,rgb,loss,terms,norm


def distributed_rehearsal(model,named,initial,items,prompt,metric,s,world):
    params=[p for _,p in named]
    reference=[]
    initial_digest=tensor_digest(initial.items())
    for repeat in range(2):
        restore(model,named,initial,s['seed'],metric)
        optimizer=torch.optim.AdamW(params,lr=s['learning_rate'],weight_decay=0)
        rows=[]
        previous_digest=initial_digest
        for index,joint in enumerate((False,True)):
            item=items[index % len(items)]
            latent,rgb,loss,_,norm=update(model,params,item,prompt,metric,s,optimizer,joint,world)
            row=dict(joint=joint,loss=float(loss.detach()),grad_norm=float(norm),
                     output_digest=tensor_digest([('latent',latent)]+([] if rgb is None else [('rgb',rgb)])),
                     gradient_digest=tensor_digest([(n,p.grad) for n,p in named]),
                     updated_parameter_digest=tensor_digest(named))
            if row['updated_parameter_digest']==previous_digest:
                raise RuntimeError('rehearsal made no parameter update')
            previous_digest=row['updated_parameter_digest']
            if any(optimizer.state[p][key].dtype != torch.float32 for p in params for key in ('exp_avg','exp_avg_sq')):
                raise RuntimeError('non-FP32 optimizer state')
            if any(p.grad is not None for p in model.parameters() if not p.requires_grad):
                raise RuntimeError('frozen parameter received gradients')
            if world>1:
                digests=[None]*world
                dist.all_gather_object(digests,row['updated_parameter_digest'])
                if len(set(digests))!=1:
                    raise RuntimeError('data-parallel parameters differ across ranks')
            rows.append(row)
        if repeat and rows!=reference:
            raise RuntimeError('distributed two-stage rehearsal is not exactly repeatable')
        reference=rows
        del optimizer
    restore(model,named,initial,s['seed'],metric)
    if tensor_digest(named)!=initial_digest:
        raise RuntimeError('rehearsal failed to restore exact initial state')
    return dict(status='pass',two_repeats_exact=True,restored_initial_digest=initial_digest,updates=reference)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--preflight',action='store_true')
    parser.add_argument('--cpu-smoke',action='store_true')
    args=parser.parse_args()
    s,e,c=load_config(args.config)
    digest=sha256(args.config)
    train,dev=load_frozen_rows(c,verify_images=True)
    if len(train)!=16 or len(dev)!=16:
        raise ValueError('fixed 16/16 coverage changed')
    if args.preflight:
        verify_model_artifacts(c,large_hashes=True)
        for name,mem in [('NVIDIA H100 80GB HBM3',81920),('NVIDIA H200',143000)]:
            validate_gpu_inventory([name]*8,[mem]*8,c)
        from diffsynth.pipelines import z_image
        build_padding_function(z_image)
        print('LATENT_FIRST_PREFLIGHT_PASS',digest,flush=True)
        return
    rank,world=int(os.getenv('RANK','0')),int(os.getenv('WORLD_SIZE','1'))
    if args.cpu_smoke and world!=1:
        raise ValueError('CPU smoke is single process')
    if not args.cpu_smoke and world!=8:
        raise ValueError('eight synchronized ranks required')
    torch.set_num_threads(4)
    device=torch.device('cpu' if args.cpu_smoke else f'cuda:{int(os.environ["LOCAL_RANK"])}')
    if not args.cpu_smoke:
        for key,value in dict(CUBLAS_WORKSPACE_CONFIG=':4096:8',NCCL_ALGO='Ring',NCCL_PROTO='Simple').items():
            if os.environ.get(key)!=value:
                raise RuntimeError(f'deterministic runtime requires {key}={value}')
        torch.cuda.set_device(device)
        if torch.cuda.device_count()!=8:
            raise ValueError('exactly eight visible CUDA devices required')
        props=[torch.cuda.get_device_properties(i) for i in range(8)]
        validate_gpu_inventory([p.name for p in props],[p.total_memory//2**20 for p in props],c)
        dist.init_process_group('nccl',device_id=device)
    torch.use_deterministic_algorithms(True,warn_only=False)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    counters={} if args.cpu_smoke else install_flash_control(True)
    operator_gate=padding_operator_probe(device)
    from diffsynth.pipelines import z_image
    _,repaired,patch=build_padding_function(z_image)
    z_image.model_fn_z_image_turbo=repaired
    verify_model_artifacts(c,large_hashes=rank==0)
    if world>1: dist.barrier()
    root=ROOT/s['output_dir']
    if args.cpu_smoke:
        import tempfile
        root=Path(tempfile.mkdtemp(prefix='zimage_latent_first_cpu_',dir=ROOT/'logs'))
    else:
        if rank==0: root.mkdir(parents=True,exist_ok=False)
        dist.barrier()
    provenance=dict(git_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                    torch_version=str(torch.__version__),cuda_version=torch.version.cuda,
                    world=world,microbatch=1,gradient_accumulation=1,global_batch=world,
                    command=sys.argv,source_sha256=sha256(Path(__file__)),
                    helper_sha256=sha256(ROOT/'lowlight_restoration/zimage_latent_first.py'),
                    environment={key:os.environ.get(key) for key in
                                 ('CUBLAS_WORKSPACE_CONFIG','NCCL_ALGO','NCCL_PROTO','DIFFSYNTH_ATTENTION_IMPLEMENTATION')})
    if rank==0: write_json(root/'runtime.json',provenance)
    seed_everything(s['seed'])
    model=LatentFirstStudent(c,device,ROOT/s['initial_checkpoint'])
    restore_student_fp32(model,ROOT/s['initial_checkpoint'])
    for name,p in model.pipe.dit.named_parameters(): p.requires_grad_('lora_' in name)
    named=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
    if len(named)!=476 or any(p.dtype!=torch.float32 for _,p in named):
        raise RuntimeError('unexpected trainable parameters')
    params=[p for _,p in named]
    initial={n:p.detach().cpu().clone() for n,p in named}
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    metric=LearnedPerceptualImagePatchSimilarity(net_type='alex',normalize=True).to(device).eval().requires_grad_(False)
    cached={}
    with torch.no_grad():
        prompt=model.pipe.units[1].process(model.pipe,c['model']['prompt'],None)['prompt_embeds']
        for role,rows in [('tiny_overfit_train',train),('visual_development',dev)]:
            items=[]
            selected=rows[:2] if args.cpu_smoke else rows[rank::world]
            for row in selected:
                lq,gt=load_aligned_pair(row,64 if args.cpu_smoke else 512)
                gt_rgb=model.pipe.preprocess_image(gt)
                items.append(dict(record_id=row['record_id'],lq=lq,gt=gt,gt_rgb=gt_rgb,
                                  lq_latent=model.pipe.vae_encoder(model.pipe.preprocess_image(lq)),
                                  gt_latent=model.pipe.vae_encoder(gt_rgb)))
            cached[role]=items
    # No development item is ever supplied to rehearsal or optimizer updates.
    rehearsal=distributed_rehearsal(model,named,initial,cached['tiny_overfit_train'],prompt,metric,s,world)
    write_json(root/f'rehearsal_rank_{rank}.json',dict(config_sha256=digest,rank=rank,world=world,
                padding_patch=patch,operator_gate=operator_gate,**rehearsal))
    print('LATENT_FIRST_REHEARSAL_PASS',rank,flush=True)
    del initial
    if args.cpu_smoke:
        print('LATENT_FIRST_REAL_CPU_SMOKE_PASS',str(root.relative_to(ROOT)),flush=True)
        return
    previous=json.loads((ROOT/s['parent_result']).read_text())
    baselines={(x['role'],r['record_id']):r for x in previous['snapshots'] if x['step']==400 for r in x['rows']}
    snapshots=[]
    def evaluate(step):
        with torch.no_grad():
            for role,items in cached.items():
                local=[]
                for item in items:
                    latent,rgb=model(item,prompt,decode=True)
                    image=model.pipe.vae_output_to_image(rgb).convert('RGB')
                    row=dict(record_id=item['record_id'],input_psnr=psnr(item['gt'],item['lq']),
                             input_lpips=lpips_value(metric,item['gt'],item['lq'],device),
                             output_psnr=psnr(item['gt'],image),output_lpips=lpips_value(metric,item['gt'],image,device),
                             latent_mse=float((latent-item['gt_latent'].float()).square().mean()),
                             raw_rgb_mse=float((rgb.float()-item['gt_rgb'].float()).square().mean()),
                             clipped_fraction=float((rgb.abs()>1).float().mean()))
                    if not all(math.isfinite(v) for k,v in row.items() if k!='record_id'):
                        raise RuntimeError('nonfinite evaluation')
                    if step==0:
                        expected=baselines[(role,item['record_id'])]
                        if abs(row['output_psnr']-expected['output_psnr'])>.02 or abs(row['output_lpips']-expected['output_lpips'])>.002:
                            raise RuntimeError('inherited endpoint reproduction mismatch')
                    folder=root/f'step_{step:06d}'/role
                    save_triptych(folder/(item['record_id']+'.png'),item['record_id'],item['lq'],image,item['gt'])
                    write_json(folder/(item['record_id']+'.json'),dict(row,step=step,role=role,config_sha256=digest,
                               image_sha256=sha256(folder/(item['record_id']+'.png'))))
                    local.append(row)
                rows=sorted(gather(local,world),key=lambda r:r['record_id'])
                expected_ids={r['record_id'] for r in (train if role=='tiny_overfit_train' else dev)}
                if len(rows)!=16 or {r['record_id'] for r in rows}!=expected_ids:
                    raise RuntimeError('incomplete/duplicate distributed evaluation records')
                metrics=aggregate(rows)
                metrics.update({k:sum(r[k] for r in rows)/16 for k in ('latent_mse','raw_rgb_mse','clipped_fraction')})
                near=[r for r in rows if r['input_psnr']>=30]
                metrics['near_clean_records']=len(near)
                metrics['near_clean_psnr_gain']=sum(r['output_psnr']-r['input_psnr'] for r in near)/len(near) if near else None
                metrics['worst_psnr_gain']=min(r['output_psnr']-r['input_psnr'] for r in rows)
                if rank==0:
                    payload=dict(step=step,role=role,metrics=metrics,rows=rows,config_sha256=digest)
                    write_json(root/f'step_{step:06d}'/role/'metrics.json',payload)
                    snapshots.append(payload)
                    print('LATENT_FIRST_SNAPSHOT',json.dumps(dict(step=step,role=role,metrics=metrics)),flush=True)
                metric.reset()
        dist.barrier()
    evaluate(0)
    optimizer=torch.optim.AdamW(params,lr=s['learning_rate'],weight_decay=0)
    start=time.monotonic()
    saved=[]
    for step in range(1,s['steps']+1):
        joint=joint_phase(step,s)
        item=cached['tiny_overfit_train'][(step-1)%len(cached['tiny_overfit_train'])]
        _,_,loss,terms,norm=update(model,params,item,prompt,metric,s,optimizer,joint,world)
        values=torch.stack([loss.detach(),terms['latent_mse'].detach(),
                            terms.get('pixel_mse',loss.new_zeros(())).detach(),
                            terms.get('lpips',loss.new_zeros(())).detach()]).float()
        dist.all_reduce(values)
        values/=world
        if rank==0:
            row=dict(step=step,phase='joint' if joint else 'latent_only',loss=float(values[0]),
                     latent_mse=float(values[1]),pixel_mse=float(values[2]) if joint else None,
                     lpips=float(values[3]) if joint else None,preclip_grad_norm=float(norm),
                     elapsed_seconds=time.monotonic()-start)
            with (root/'training.jsonl').open('a') as handle: handle.write(json.dumps(row,allow_nan=False)+'\n')
            if step==1 or step==801 or step%20==0: print('LATENT_FIRST_TRAIN',json.dumps(row),flush=True)
        if step in s['snapshot_steps']:
            parameter_digests=[None]*world
            dist.all_gather_object(parameter_digests,tensor_digest(named))
            if len(set(parameter_digests))!=1:
                raise RuntimeError('snapshot parameters differ across ranks')
            if rank==0:
                state={n.removeprefix('pipe.dit.'):p.detach().cpu().contiguous() for n,p in named}
                path=root/f'step_{step:06d}.safetensors'
                save_file(state,path,metadata=dict(config_sha256=digest,step=str(step),classification='supervised_endpoint_repair'))
                loaded=load_file(str(path))
                if set(state)!=set(loaded) or any(loaded[n].dtype!=torch.float32 or not torch.equal(state[n],loaded[n]) for n in state):
                    raise RuntimeError('checkpoint roundtrip mismatch')
                saved.append(dict(step=step,path=str(path.relative_to(ROOT)),sha256=sha256(path)))
                del state,loaded
            dist.barrier()
            evaluate(step)
    if rank==0:
        final_dev=next(x['metrics'] for x in snapshots if x['step']==s['selection_step'] and x['role']=='visual_development')
        result=dict(status='latent_first_endpoint_complete',config_sha256=digest,classification='supervised_endpoint_repair_not_distillation',
                    checkpoint=saved[-1]['path'],checkpoint_sha256=saved[-1]['sha256'],checkpoints=saved,
                    initial_checkpoint_sha256=s['initial_checkpoint_sha256'],snapshots=snapshots,steps=s['steps'],
                    global_batch_size=8,world_size=8,hardware_name=torch.cuda.get_device_properties(device).name,
                    execution='strict_deterministic_fixed_order_gradient_allreduce',padding_patch=patch,
                    attention_calls_rank0=counters,training_log_sha256=sha256(root/'training.jsonl'),
                    runtime=provenance,runtime_sha256=sha256(root/'runtime.json'),
                    rehearsal_sha256={str(r):sha256(root/f'rehearsal_rank_{r}.json') for r in range(world)},
                    qwen_degradation_conditioning=False,mixed_degradations=False,teacher_imitation=False,
                    summary=dict(task_metrics=dict(blur=dict(psnr=final_dev['output_psnr'],lpips=final_dev['output_lpips'])),
                                 scope='16 reusable development crops, not official FoundIR acceptance'),
                    sealed_internal_accessed=False,official_foundir_test_accessed=False,development_gt_optimized=False,
                    decoder_finetuned=False,wall_seconds=time.monotonic()-start,**metric_gate(snapshots,s['selection_step']))
        write_json(root/'result.json',result)
        write_once(root/'report.md',render_report(result).encode())
        print('LATENT_FIRST_COMPLETE',json.dumps(dict(result_sha256=sha256(root/'result.json'),**metric_gate(snapshots,s['selection_step']))),flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__=='__main__':
    main()
