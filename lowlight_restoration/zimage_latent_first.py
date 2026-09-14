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
    if s.get('status') not in ('authorized_round2_latent_first_endpoint_repair',
                               'authorized_round2_latent_first_endpoint_repair_v2'):
        raise ValueError('unexpected round-2 experiment')
    if any(s.get(key) is not False for key in (*PROTECTED,'teacher_imitation')):
        raise ValueError('forbidden round-2 scope')
    expected = dict(steps=1200,latent_only_steps=800,snapshot_steps=SNAPSHOTS,
                    selection_step=1200,global_batch_size=8,learning_rate=5e-5,
                    max_grad_norm=1.,latent_weight=1.,joint_pixel_weight=1.,joint_lpips_weight=.05,
                    seed=2026091405)
    if any(s.get(key) != value for key,value in expected.items()):
        raise ValueError('frozen round-2 schedule/objective changed')
    if s.get('status').endswith('_v2'):
        replay=s.get('baseline_replay',{})
        expected_replay=dict(strict_repeats=2,require_exact_current_latent_rgb_metrics=True,
            historical_per_record_max_abs_psnr_delta_db=.05,
            historical_per_record_max_abs_lpips_delta=.005,
            historical_aggregate_max_abs_psnr_delta_db=.02,
            historical_aggregate_max_abs_lpips_delta=.002)
        if replay != expected_replay or s.get('authorization_hold_id') != 'zimage_latent_first_runtime-36be52b1efd75072':
            raise ValueError('frozen v2 baseline-recovery contract changed')


def baseline_replay_gate(current_a,current_b,historical,settings):
    """Require exact current replay and bounded, fully reported historical drift."""
    keys=('record_id','role')
    def index(rows):
        result={(row['record_id'],row['role']):row for row in rows}
        if len(result)!=len(rows):
            raise ValueError('duplicate baseline record')
        return result
    a,b,h=index(current_a),index(current_b),index(historical)
    if set(a)!=set(b) or set(a)!=set(h) or len(a)!=32:
        raise ValueError('baseline record coverage mismatch')
    exact_fields=('latent_digest','rgb_digest','image_sha256','input_psnr','input_lpips',
                  'output_psnr','output_lpips','latent_mse','raw_rgb_mse','clipped_fraction')
    rows=[]
    for key in sorted(a):
        if any(a[key][field]!=b[key][field] for field in exact_fields):
            raise ValueError(f'current deterministic baseline replay differs: {key}')
        delta_psnr=a[key]['output_psnr']-h[key]['output_psnr']
        delta_lpips=a[key]['output_lpips']-h[key]['output_lpips']
        rows.append(dict(record_id=key[0],role=key[1],historical_output_psnr=h[key]['output_psnr'],
                         current_output_psnr=a[key]['output_psnr'],psnr_delta_db=delta_psnr,
                         historical_output_lpips=h[key]['output_lpips'],current_output_lpips=a[key]['output_lpips'],
                         lpips_delta=delta_lpips,latent_digest=a[key]['latent_digest'],rgb_digest=a[key]['rgb_digest'],
                         image_sha256=a[key]['image_sha256']))
    per_psnr=max(abs(row['psnr_delta_db']) for row in rows)
    per_lpips=max(abs(row['lpips_delta']) for row in rows)
    aggregate={}
    for role in ('tiny_overfit_train','visual_development'):
        selected=[row for row in rows if row['role']==role]
        aggregate[role]=dict(psnr_delta_db=sum(row['psnr_delta_db'] for row in selected)/len(selected),
                             lpips_delta=sum(row['lpips_delta'] for row in selected)/len(selected))
    agg_psnr=max(abs(value['psnr_delta_db']) for value in aggregate.values())
    agg_lpips=max(abs(value['lpips_delta']) for value in aggregate.values())
    cfg=settings['baseline_replay']
    checks=dict(current_strict_replay_exact=True,
        historical_per_record_psnr_bounded=per_psnr<=cfg['historical_per_record_max_abs_psnr_delta_db'],
        historical_per_record_lpips_bounded=per_lpips<=cfg['historical_per_record_max_abs_lpips_delta'],
        historical_aggregate_psnr_bounded=agg_psnr<=cfg['historical_aggregate_max_abs_psnr_delta_db'],
        historical_aggregate_lpips_bounded=agg_lpips<=cfg['historical_aggregate_max_abs_lpips_delta'])
    return dict(status='pass' if all(checks.values()) else 'failure',checks=checks,
                maxima=dict(per_record_abs_psnr_delta_db=per_psnr,per_record_abs_lpips_delta=per_lpips,
                            aggregate_abs_psnr_delta_db=agg_psnr,aggregate_abs_lpips_delta=agg_lpips),
                aggregates=aggregate,rows=rows)


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
