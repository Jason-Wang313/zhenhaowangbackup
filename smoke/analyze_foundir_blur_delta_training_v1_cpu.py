#!/usr/bin/env python3
"""Describe completed training and representable LoRA updates; no image evaluation."""
import json
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from smoke.existing_flux_replay_contract_v1 import read, sha, write_new as write


def main():
    import torch
    from safetensors.torch import load_file
    begin = time.monotonic()
    run = ROOT/'checkpoints/foundir_blur_delta_velocity_v1'
    official = ROOT/'checkpoints/foundir_blur_delta_official_v1'
    parent = read(run/'receipt.json'); binding = read(official/'receipt.json')
    initial_spec = parent['lora']; final_spec = binding['weight']
    for spec in (initial_spec, final_spec):
        if sha(ROOT/spec['path']) != spec['sha256']: raise ValueError('LoRA SHA changed')
    initial = load_file(str(ROOT/initial_spec['path'])); final = load_file(str(ROOT/final_spec['path']))
    if set(initial) != set(final) or len(final) != 100: raise ValueError('LoRA parameter schema changed')
    parameters = {}; count = 0; changed_fp32 = 0; changed_bf16 = 0; norm0 = 0.; normdelta = 0.
    for name in sorted(final):
        a, b = initial[name].float(), final[name]
        if a.shape != b.shape or b.dtype != torch.float32 or not torch.isfinite(b).all():
            raise ValueError('invalid final parameter')
        delta = b-a; n = b.numel(); old_norm = float(a.double().square().sum())
        delta_norm = float(delta.double().square().sum())
        fp = int(torch.count_nonzero(delta)); bf = int(torch.count_nonzero(b.to(torch.bfloat16) != a.to(torch.bfloat16)))
        parameters[name] = {'numel': n, 'FP32_changed_fraction': fp/n, 'BF16_changed_fraction': bf/n,
                            'initial_L2': old_norm**.5, 'delta_L2': delta_norm**.5,
                            'relative_delta_L2': (delta_norm/old_norm)**.5 if old_norm else None}
        count += n; changed_fp32 += fp; changed_bf16 += bf; norm0 += old_norm; normdelta += delta_norm
    preflight = read(ROOT/'reports/foundir_blur_delta_official_v1_preflight_2026-11-06.json')
    history = preflight['training_history']
    if [r['step'] for r in history] != list(range(1,201)): raise ValueError('not the complete200 training history')
    epochs = []
    for offset in range(0,200,16):
        rows = history[offset:offset+16]
        epochs.append({'epoch': offset//16+1, 'steps': len(rows), 'complete128_source_epoch': len(rows)==16,
                       'range': [rows[0]['step'],rows[-1]['step']],
                       **{k: sum(r[k] for r in rows)/len(rows) for k in ('loss','mse','lpips','gradient_norm','max_rank_seconds')}})
    value = {'status': 'completed_training_description', 'initial_LoRA': initial_spec, 'final_LoRA': final_spec,
             'training_receipt_sha256': sha(run/'receipt.json'), 'official_receipt_sha256': sha(official/'receipt.json'),
             'parameters': parameters, 'parameter_count': count, 'FP32_changed_fraction': changed_fp32/count,
             'BF16_changed_fraction': changed_bf16/count, 'relative_delta_L2': (normdelta/norm0)**.5,
             'online_training_epochs': epochs, 'online_not_fixed_checkpoint_evaluation': True,
             'GT_or_LQ_pixels_read': False, 'no_extra_evaluation_population': True, 'gate': False,
             'interpretation_limit': 'Representable parameter updates do not prove quality improvement; only official150 decides acceptance.',
             'seconds': time.monotonic()-begin, 'exit_code': 0, 'GPU_hours': 0}
    write(ROOT/'reports/foundir_blur_delta_training_evidence_v1_CPU_2026-11-06.json', value)
    print('BLUR_DELTA_TRAINING_EVIDENCE '+json.dumps({k:v for k,v in value.items() if k not in ('parameters','online_training_epochs')}))


if __name__ == '__main__': main()
