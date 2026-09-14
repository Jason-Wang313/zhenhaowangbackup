"""Frozen round-2 objective and admission checks, independent of model loading."""
import math

import torch
import torch.distributed as dist

from lowlight_restoration.zimage_repro_batch import PROTECTED


SNAPSHOTS = [0,200,400,600,800,1000,1200]


def average_gradients(params,world):
    params=list(params)
    if not params or world < 1:
        raise ValueError('empty parameters/invalid world size')
    if world > 1 and (not dist.is_initialized() or dist.get_world_size() != world):
        raise ValueError('gradient reduction world size mismatch')
    if any(p.grad is None or p.grad.dtype != torch.float32 for p in params):
        raise ValueError('missing/non-FP32 trainable gradient')
    if world == 1:
        return
    packed=torch.cat([p.grad.reshape(-1) for p in params])
    dist.all_reduce(packed)
    packed.div_(world)
    offset=0
    with torch.no_grad():
        for p in params:
            count=p.numel()
            p.grad.copy_(packed[offset:offset+count].view_as(p))
            offset+=count


def validate_settings(s):
    if s.get('status') != 'authorized_round2_latent_first_endpoint_repair':
        raise ValueError('unexpected round-2 experiment')
    if any(s.get(key) is not False for key in (*PROTECTED,'teacher_imitation')):
        raise ValueError('forbidden round-2 scope')
    expected = dict(steps=1200,latent_only_steps=800,snapshot_steps=SNAPSHOTS,
                    selection_step=1200,global_batch_size=8,learning_rate=5e-5,
                    max_grad_norm=1.,latent_weight=1.,joint_pixel_weight=1.,joint_lpips_weight=.05,
                    seed=2026091405)
    if any(s.get(key) != value for key,value in expected.items()):
        raise ValueError('frozen round-2 schedule/objective changed')


def joint_phase(step,s):
    if not 1 <= step <= s['steps']:
        raise ValueError('step outside frozen schedule')
    return step > s['latent_only_steps']


def loss_terms(latent,rgb,item,metric,s,joint):
    latent_mse = (latent.float()-item['gt_latent'].float()).square().mean()
    if not joint:
        return s['latent_weight']*latent_mse,dict(latent_mse=latent_mse)
    if rgb is None or metric is None:
        raise ValueError('joint stage requires differentiable decoded RGB and LPIPS')
    pixel = (rgb.float()-item['gt_rgb'].float()).square().mean()
    perceptual = metric((rgb.float().clamp(-1,1)+1)/2,(item['gt_rgb'].float()+1)/2).mean()
    return (s['latent_weight']*latent_mse+s['joint_pixel_weight']*pixel+s['joint_lpips_weight']*perceptual,
            dict(latent_mse=latent_mse,pixel_mse=pixel,lpips=perceptual))


def metric_gate(snapshots,final_step):
    by_key={(x['step'],x['role']):x for x in snapshots}
    if len(by_key) != len(snapshots):
        raise ValueError('duplicate evaluation snapshot')
    roles=('tiny_overfit_train','visual_development')
    if final_step != SNAPSHOTS[-1] or set(by_key) != {(step,role) for step in SNAPSHOTS for role in roles}:
        raise ValueError('incomplete/unexpected evaluation schedule')
    for snapshot in snapshots:
        if not all(math.isfinite(snapshot['metrics'][key]) for key in
                   ('psnr_gain_db','lpips_improvement','latent_mse')):
            raise ValueError('nonfinite gate metric')
    checks={}
    for role in roles:
        final=by_key[(final_step,role)]['metrics']
        values=[final['psnr_gain_db'],final['lpips_improvement'],final['latent_mse']]
        if not all(math.isfinite(v) for v in values):
            raise ValueError('nonfinite gate metric')
        checks[role+'_psnr']=final['psnr_gain_db']>0
        checks[role+'_lpips']=final['lpips_improvement']>0
    checks['training_latent_improved']=(by_key[(final_step,roles[0])]['metrics']['latent_mse']
                                      <by_key[(0,roles[0])]['metrics']['latent_mse'])
    return dict(checks=checks,automatic_metric_gate_pass=all(checks.values()),
                human_visual_review_required=True,human_visual_review_complete=False,
                quality_gate_pass=False,full_training_started=False)


def render_report(result):
    lines=['# Latent-first round-2 result','',
           'DIAGNOSTIC ONLY / 非主线. Supervised endpoint repair, NOT successful teacher distillation.',
           'Fixed 16 train / 16 reusable source-disjoint development crops; not official FoundIR acceptance.','',
           f"Config SHA-256: `{result['config_sha256']}`",
           f"Final checkpoint: `{result['checkpoint']}` (`{result['checkpoint_sha256']}`)",'',
           '| Step | Role | Input PSNR / LPIPS | Output PSNR / LPIPS | Latent MSE |',
           '|---:|---|---:|---:|---:|']
    for row in result['snapshots']:
        m=row['metrics']
        lines.append(f"| {row['step']} | {row['role']} | {m['input_psnr']:.4f} / {m['input_lpips']:.4f} | "
                     f"{m['output_psnr']:.4f} / {m['output_lpips']:.4f} | {m['latent_mse']:.6f} |")
    lines.extend(['',f"Automatic bounded metric gate: {result['automatic_metric_gate_pass']}.",
                  'Human visual review remains pending. Full training has NOT started.',
                  'Final step 1200 is predeclared; earlier snapshots are diagnostic only.',
                  'Original SOW Stage-2 Blur thresholds remain PSNR >=25.9 dB / LPIPS <=.18;',
                  'crop-subset comparisons do not establish contractual acceptance.',
                  'No decoder tuning, development-GT optimization, Qwen degradation conditioning,',
                  'mixed degradation, sealed internal or official FoundIR test access occurred.',
                  'Close the two-round decision budget: review metrics and fixed visuals; if this',
                  'recipe fails, reconsider formulation rather than adding indefinite sweeps.',''])
    return '\n'.join(lines)
