"""Frozen experiment assignment and fail-closed numerical comparisons."""
from __future__ import annotations

import functools
import hashlib
import inspect
import math
import random

import numpy as np
import torch


PHASES = ['off_a', 'on', 'off_b']
SNAPSHOTS = [0, 1, 20, 40, 60, 80, 100, 120, 140, 160, 180, 190, 195, 200]
PROTECTED = ('decoder_finetuning', 'development_gt_optimized',
             'qwen_degradation_conditioning', 'mixed_degradations',
             'sealed_internal_accessed', 'official_foundir_test_accessed')


def validate_settings(settings):
    if any(settings.get(k) is not False for k in PROTECTED):
        raise ValueError('forbidden scope')
    if (settings['phases'] != PHASES or settings['snapshot_steps'] != SNAPSHOTS
            or settings['steps'] != 200 or settings['world_size'] != 8
            or settings['learning_rate'] != 5e-5
            or settings['psnr_tolerance'] != .05 or settings['lpips_tolerance'] != .005):
        raise ValueError('frozen diagnostic design changed')


def assignment(rank):
    if rank not in range(8):
        raise ValueError('rank must be 0..7')
    return dict(rank=rank, record_index=rank % 2,
                arm=['lora_rgb', 'lora_mixed'][rank % 2],
                deterministic=bool((rank // 2) % 2), replicate=rank // 4)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def tensor_digest(named_tensors):
    """Hash content, names, shape and dtype, independent of safetensors metadata."""
    digest = hashlib.sha256()
    for name, tensor in sorted(named_tensors):
        if tensor is None or not torch.isfinite(tensor).all():
            raise ValueError(f'missing/nonfinite tensor: {name}')
        value = tensor.detach().cpu().contiguous()
        digest.update(str((name, str(value.dtype), tuple(value.shape))).encode())
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def install_flash_control(deterministic):
    """Process-local override; never edit the ignored third-party checkout."""
    import flash_attn
    from diffsynth.core.attention import attention as backend
    if backend.ATTENTION_IMPLEMENTATION != 'flash_attention_2':
        raise RuntimeError('FlashAttention 2 must be explicitly selected')
    original = flash_attn.flash_attn_func
    if 'deterministic' not in inspect.signature(original).parameters:
        raise RuntimeError('installed FlashAttention lacks deterministic backward')
    counters = dict(flash_calls=0, sdpa_calls=0)

    @functools.wraps(original)
    def controlled(*args, **kwargs):
        kwargs['deterministic'] = deterministic
        counters['flash_calls'] += 1
        return original(*args, **kwargs)

    original_sdpa = backend.torch_sdpa

    @functools.wraps(original_sdpa)
    def counted_sdpa(*args, **kwargs):
        counters['sdpa_calls'] += 1
        return original_sdpa(*args, **kwargs)

    flash_attn.flash_attn_func = controlled
    backend.torch_sdpa = counted_sdpa
    return counters


def compare_runs(left, right, settings):
    if left['status'] != 'complete' or right['status'] != 'complete':
        return dict(comparable=False, reason='incomplete_run')
    if left['initial_parameter_digest'] != right['initial_parameter_digest']:
        return dict(comparable=False, reason='initial_weights_differ')
    a = {r['step']: r for r in left['snapshots']}
    b = {r['step']: r for r in right['snapshots']}
    if len(a) != len(left['snapshots']) or len(b) != len(right['snapshots']):
        raise ValueError('duplicate snapshot')
    if sorted(a) != settings['snapshot_steps'] or sorted(b) != settings['snapshot_steps']:
        return dict(comparable=False, reason='snapshot_schedule_differs')
    deltas = []
    for step in a:
        values = [a[step][k] for k in ('psnr', 'lpips')] + [b[step][k] for k in ('psnr', 'lpips')]
        if not all(math.isfinite(v) for v in values):
            raise ValueError('nonfinite comparison metrics')
        deltas.append(dict(step=step, psnr_delta=b[step]['psnr']-a[step]['psnr'],
                           lpips_delta=b[step]['lpips']-a[step]['lpips'],
                           latent_exact=a[step]['latent_digest']==b[step]['latent_digest']))
    max_psnr = max(abs(x['psnr_delta']) for x in deltas)
    max_lpips = max(abs(x['lpips_delta']) for x in deltas)
    return dict(comparable=True, trajectory=deltas,
                max_psnr_delta=max_psnr, max_lpips_delta=max_lpips,
                within_metric_tolerance=max_psnr <= settings['psnr_tolerance'] and max_lpips <= settings['lpips_tolerance'],
                first_gradient_exact=left['first_gradient_digest']==right['first_gradient_digest'],
                first_update_exact=left['first_updated_parameter_digest']==right['first_updated_parameter_digest'],
                final_parameters_exact=left['final_parameter_digest']==right['final_parameter_digest'],
                all_snapshot_latents_exact=all(x['latent_exact'] for x in deltas))


def summarize_comparisons(comparisons, complete_count):
    """Recommend a decision, never automatically admit full training."""
    det = [x for x in comparisons if x['kind'] in ('same_gpu_repeat', 'instrumentation') and x['deterministic']]
    expected = {(rank, kind) for rank in (2,3,6,7) for kind in ('same_gpu_repeat','instrumentation')}
    det_exact = len(det) == 8 and {(x.get('rank'),x['kind']) for x in det} == expected and all(x.get('comparable') and x.get('first_gradient_exact')
                                 and x.get('first_update_exact') and x.get('final_parameters_exact')
                                 and x.get('all_snapshot_latents_exact') for x in det)
    if complete_count != 24:
        decision = 'investigate_incomplete_controls_no_scaling'
    elif not det_exact:
        decision = 'deterministic_control_failed_isolate_remaining_numerics_no_scaling'
    else:
        decision = 'reproducible_execution_available_review_fit_and_design_round2'
    return dict(next_action=decision, deterministic_same_gpu_control_exact=det_exact,
                full_training_authorized_by_this_result=False,
                quality_pass=False, human_visual_review_required=True)
