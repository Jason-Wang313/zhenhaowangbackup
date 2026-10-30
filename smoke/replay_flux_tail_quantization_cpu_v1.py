#!/usr/bin/env python3
"""Read-only CPU replay of ONE hash-bound training-source diagnostic.

No FLUX/model/GT loading, CUDA initialization, cache writes, or training.
Print a small JSON report; --output may exclusively create a reports/*.json.
Alternative arithmetic is a diagnostic control, NEVER a deployment change.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DIAGNOSTIC = 'checkpoints/foundir_flux_tail_baseline_parity_v1/baseline_parity_2gpu_v1'
RESULT_SHA = '62945a5abf462f889aebd8fa667fd92b56f30deaa7798df76c48462b245b3228'
TRACE_SHA = '54998b694f4b0271b5efc0d55657e4b572276daa36e2e0de744f0df7bdaff847'
PARENT = 'checkpoints/foundir_flux_tail_energy_resumable_v2/qualification_8gpu_v2'
TRACE_KEYS = ('observation_normalized', 'raw_clamped', 'raw_uint8', 'proposal_uint8')
SCALAR_SOURCE = 'lowlight_restoration/scalar_residual.py'


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024**2), b''):
            digest.update(chunk)
    return digest.hexdigest()


def inside(root, relative):
    value = (root / relative).resolve()
    value.relative_to(root.resolve())
    return value


def bound(root, relative, expected):
    value = inside(root, relative)
    if sha(value) != expected:
        raise RuntimeError('SHA mismatch: ' + str(relative))
    return value


def validate_identity(identity):
    if (identity['index'] != 0 or identity['role'] != 'train'
            or identity['row']['split'] != 'train' or identity['row']['category'] != '05Noise'):
        raise RuntimeError('only original Noise training source index 0 is allowed')


def difference(left, right):
    import torch
    if left.shape != right.shape or left.dtype != right.dtype:
        raise RuntimeError('comparison shape/dtype mismatch')
    if left.device.type != 'cpu' or right.device.type != 'cpu':
        raise RuntimeError('CPU tensors only')
    if not (torch.isfinite(left).all() and torch.isfinite(right).all()):
        raise RuntimeError('nonfinite input')
    delta = left.double() - right.double()  # Never subtract uint8 directly.
    changed = left != right
    return dict(elements=left.numel(), different_elements=int(changed.sum()),
                different_rgb_pixels=int(changed.any(dim=1).sum()),
                max_abs=float(delta.abs().max()), mean_abs=float(delta.abs().mean()),
                rms=float(delta.square().mean().sqrt()), signed_mean=float(delta.mean()))


def rational_blend(lq, raw):
    """Exact round-to-nearest, ties-to-even of (7*lq+3*raw)/10."""
    import torch
    if lq.dtype != torch.uint8 or raw.dtype != torch.uint8 or lq.shape != raw.shape:
        raise ValueError('matching uint8 tensors required')
    n = 7 * lq.to(torch.int32) + 3 * raw.to(torch.int32)
    q, r = n // 10, n % 10
    result = q + ((r > 5) | ((r == 5) & (q % 2 == 1)))
    return result.to(torch.uint8), r == 5


def synthetic_audit():
    import torch
    from lowlight_restoration.scalar_residual import quantized_scalar_residual
    values = torch.arange(256, dtype=torch.uint8)
    lq = values[:, None].expand(256, 256).reshape(1, 1, 256, 256)
    raw = values[None, :].expand(256, 256).reshape_as(lq)
    floating, _ = quantized_scalar_residual(lq.float()/127.5-1, raw.float()/127.5-1, .3)
    exact, ties = rational_blend(lq, raw)
    return dict(uint8_pairs=65536, floating_vs_exact=difference(floating, exact),
                disagreements_at_exact_ties=int(((floating != exact) & ties).sum()),
                disagreements_off_exact_ties=int(((floating != exact) & ~ties).sum()),
                interpretation='Synthetic arithmetic sensitivity, NOT the historical-run root cause.')


def load_inputs(root):
    """Verify the evidence chain before decoding only four trace + two cache tensors."""
    from safetensors import safe_open
    result_file = bound(root, DIAGNOSTIC+'/result.json', RESULT_SHA)
    result = json.loads(result_file.read_text())
    receipt_file = bound(root, DIAGNOSTIC+'/preflight.json', result['preflight_sha256'])
    receipt = json.loads(receipt_file.read_text())
    validate_identity(receipt['identity'])
    if receipt['parent_run'] != PARENT:
        raise RuntimeError('unexpected parent run')
    parent_file = bound(root, PARENT+'/preflight.json', receipt['parent_preflight_sha256'])
    parent = json.loads(parent_file.read_text())
    if receipt['identity'] != parent['selected'][0]:
        raise RuntimeError('training identity differs from parent')
    meta_file = bound(root, PARENT+'/cache_000.json', receipt['cache_metadata_sha256'])
    meta = json.loads(meta_file.read_text())
    if (meta['identity'] != receipt['identity'] or meta['file'] != 'cache_000.safetensors'
            or meta['config_sha256'] != parent['config_sha256']
            or meta['source_sha256'] != parent['source_sha256']):
        raise RuntimeError('parent cache identity mismatch')
    cache_name = PARENT+'/'+meta['file']
    if receipt['assets'][cache_name] != meta['sha256']:
        raise RuntimeError('cache content binding mismatch')
    cache_file = bound(root, cache_name, meta['sha256'])
    scalar_file = bound(root, SCALAR_SOURCE, receipt['source_sha256'][SCALAR_SOURCE])
    # The imported function must be the same byte-exact producer as the saved run.
    if sha(ROOT/SCALAR_SOURCE) != sha(scalar_file):
        raise RuntimeError('local replay implementation differs from bound producer')
    row_name = 'gpu0_repeat0.json'
    row_file = bound(root, DIAGNOSTIC+'/'+row_name, result['artifact_sha256'][row_name])
    row = json.loads(row_file.read_text())
    if row['file'] != 'gpu0_repeat0.safetensors' or row['sha256'] != TRACE_SHA:
        raise RuntimeError('unexpected trace identity')
    if result['artifact_sha256'][row['file']] != TRACE_SHA:
        raise RuntimeError('trace binding differs from result')
    trace_file = bound(root, DIAGNOSTIC+'/'+row['file'], TRACE_SHA)
    with safe_open(str(trace_file), framework='pt', device='cpu') as handle:
        trace = {key: handle.get_tensor(key) for key in TRACE_KEYS}
    import torch
    for key, tensor in trace.items():
        actual = hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
        if actual != row['stage_sha256'][key]:
            raise RuntimeError('trace stage hash mismatch: '+key)
    with safe_open(str(cache_file), framework='pt', device='cpu') as handle:
        cache = {key: handle.get_tensor(key)[None] for key in ('observation', 'proposal')}
    evidence = dict(result_sha256=RESULT_SHA, trace_sha256=TRACE_SHA,
                    parent_preflight_sha256=receipt['parent_preflight_sha256'],
                    cache_sha256=meta['sha256'], scalar_source_sha256=sha(scalar_file),
                    selected_index=0, role='train', tensors_decoded=list(TRACE_KEYS)+list(cache),
                    other_three_traces_rehashed=False)
    return trace, cache, evidence


def replay(trace, cache):
    import torch
    from lowlight_restoration.scalar_residual import quantized_scalar_residual
    for key, tensor in (trace | cache).items():
        dtype = torch.float32 if key in ('observation_normalized', 'raw_clamped') else torch.uint8
        if tensor.shape != (1, 3, 512, 512) or tensor.dtype != dtype or tensor.device.type != 'cpu':
            raise RuntimeError('unexpected tensor geometry/dtype/device: '+key)
        if not bool(torch.isfinite(tensor).all()):
            raise RuntimeError('nonfinite tensor: '+key)
    obs, old = cache['observation'], cache['proposal']
    obs_cpu = obs.float()/127.5-1
    obs_saved, raw = trace['observation_normalized'], trace['raw_clamped']
    if not torch.equal(((obs_saved+1)*127.5).round().to(torch.uint8), obs):
        raise RuntimeError('saved observation is not the parent source')
    if bool((obs_saved.abs() > 1).any() or (raw.abs() > 1).any()):
        raise RuntimeError('normalized tensor out of range')
    raw_cpu, raw_norm = quantized_scalar_residual(obs_saved, raw, 1.)
    saved_raw = trace['raw_uint8']
    second_saved, _ = quantized_scalar_residual(obs_saved, saved_raw.float()/127.5-1, .3)
    second_cpu, _ = quantized_scalar_residual(obs_cpu, saved_raw.float()/127.5-1, .3)
    full, _ = quantized_scalar_residual(obs_saved, raw_norm, .3)
    exact, ties = rational_blend(obs, saved_raw)
    double = ((obs.double()/127.5-1) + .3 *
              ((saved_raw.double()/127.5-1) - (obs.double()/127.5-1)))
    double = ((double.clamp(-1, 1)+1)*127.5).round().clamp(0, 255).to(torch.uint8)
    actual = trace['proposal_uint8']
    outputs = dict(cpu_full_replay=full, cpu_second_saved_observation=second_saved,
                   cpu_second_cpu_observation=second_cpu,
                   diagnostic_float64_second_only=double, diagnostic_exact_rational_second_only=exact)
    gap = actual != old
    return dict(observation_normalization=difference(obs_cpu, obs_saved),
                first_quantization_cpu_vs_saved=difference(raw_cpu, saved_raw),
                saved_vs_historical=difference(actual, old),
                history_gap_elements_at_exact_ties=int((gap & ties).sum()),
                history_gap_elements_off_exact_ties=int((gap & ~ties).sum()),
                variants={name: dict(vs_saved_gpu=difference(value, actual),
                                    vs_historical=difference(value, old),
                                    historical_gap_matches=int((gap & (value == old)).sum()),
                                    new_mismatches_on_previously_equal=int((~gap & (value != old)).sum()))
                          for name, value in outputs.items()},
                interpretation='No old raw tensor exists. Matching one arithmetic control does not '
                               'alone identify the old generation path or authorize a baseline change.')


def write_report(destination, report):
    with destination.open('x') as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--synthetic-only', action='store_true')
    parser.add_argument('--output', help='optional fresh reports/*.json; never overwrites')
    args = parser.parse_args()
    root = args.root.resolve()
    destination = None
    if args.output:
        relative = Path(args.output)
        if len(relative.parts) != 2 or relative.parts[0] != 'reports' or relative.suffix != '.json':
            parser.error('--output must be a new reports/NAME.json')
        destination = inside(root, relative)
        if destination.exists():
            raise FileExistsError(destination)
    started = time.monotonic()
    # Detect unavailable remote input before importing torch or doing any work.
    if not args.synthetic_only and not inside(root, DIAGNOSTIC+'/result.json').is_file():
        raise FileNotFoundError('New-node diagnostic tensors are not mounted here: '+str(root/DIAGNOSTIC))
    import torch
    torch.set_num_threads(2)
    if torch.cuda.is_initialized():
        raise RuntimeError('CPU-only process required')
    report = dict(kind='synthetic_only' if args.synthetic_only else 'saved_tensor_cpu_replay',
                  synthetic=synthetic_audit(), training_steps=0, used_gpu_hours=0,
                  gt_tensor_decoded=False, protected_pixels_accessed=False,
                  baseline_changed=False, gpu_qualification_pass=False,
                  root_cause='not_established', torch=torch.__version__)
    if not args.synthetic_only:
        trace, cache, report['evidence'] = load_inputs(root)
        report['saved_tensor_replay'] = replay(trace, cache)
    report['cuda_initialized'] = torch.cuda.is_initialized()
    if report['cuda_initialized']:
        raise RuntimeError('unexpected CUDA initialization')
    report['elapsed_seconds'] = time.monotonic()-started
    report['script_sha256'] = sha(Path(__file__).resolve())
    if destination is not None:
        write_report(destination, report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == '__main__':
    sys.exit(main())
