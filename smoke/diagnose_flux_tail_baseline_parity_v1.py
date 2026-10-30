#!/usr/bin/env python3
"""One allowed training source, two independent GPUs, no learning or DDP.

Compare exact historical restore calls, preserving every diagnostic attempt.
Historical intermediate tensors do not exist: stage localization compares
current repeats/devices only, never pretends to locate historical divergence.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from smoke import flux_energy_qualification_contract_v1 as common

PREFLIGHT = 'checkpoints/foundir_flux_tail_relative_energy_v3_cluster/qualification_3x2_v1/preflight.json'
PREFLIGHT_SHA = '515b38e0e96b3db10071a2e882473e86bc12c5d54c220404ff4021f0e544119a'
OUTPUT_ROOT = 'checkpoints/foundir_flux_tail_baseline_parity_v1'
FILES = ['smoke/diagnose_flux_tail_baseline_parity_v1.py',
         'smoke/run_flux_tail_baseline_parity_2gpu_v1.sh']


def validate_identity(identity):
    if (identity['index'] != 0 or identity['role'] != 'train'
            or identity['row']['split'] != 'train'
            or identity['row']['category'] != '05Noise'):
        raise RuntimeError('diagnostic restricted to original training index 0')


def preflight():
    current = json.loads(common.bound(PREFLIGHT, PREFLIGHT_SHA).read_text())
    parent_path = common.path(current['spec']['parent_run'])
    parent = json.loads(common.bound(str(parent_path/'preflight.json'),
                                    current['parent_preflight_sha256']).read_text())
    for name, sha in current['source_sha256'].items():
        common.bound(name, sha)
    identity = current['selected'][0]
    validate_identity(identity)
    if identity != parent['selected'][0]:
        raise RuntimeError('parent training identity differs')
    meta_path = parent_path/'cache_000.json'
    meta = json.loads(meta_path.read_text())
    if (meta['identity'] != identity or meta['file'] != 'cache_000.safetensors'
            or meta['source_sha256'] != parent['source_sha256']
            or meta['config_sha256'] != parent['config_sha256']):
        raise RuntimeError('parent cache identity differs')
    prompt_root = common.path(parent['parent_cache'])
    prompt_meta = prompt_root/'prompts.json'
    prompt_info = json.loads(prompt_meta.read_text())
    if prompt_info['files']['prompt_noise.safetensors'] != '3c2c8ef6fbc1fc52eb35f75f1c3f717a8c87d921854c3b7b2acfe1dab53d74a4':
        raise RuntimeError('historical Noise prompt changed')
    # Bind only assets used here; do not decode GT/evaluation/protected images.
    assets = {str(parent_path/meta['file']): meta['sha256'],
              str(prompt_root/'prompt_noise.safetensors'): prompt_info['files']['prompt_noise.safetensors']}
    for field in ('flow', 'ae'):
        name = parent['base_spec'][field]
        assets[name] = parent['artifacts'][name]
    name = parent['plan']['baseline_checkpoint']
    assets[name] = parent['artifacts'][name]
    for name, sha in assets.items():
        common.bound(name, sha)
    receipt = dict(identity=identity, failed_preflight_sha256=PREFLIGHT_SHA,
                   parent_preflight_sha256=current['parent_preflight_sha256'],
                   parent_run=str(parent_path.relative_to(ROOT)),
                   cache_metadata_sha256=common.sha(meta_path),
                   prompt_metadata_sha256=common.sha(prompt_meta),
                   assets={str(common.path(k).relative_to(ROOT)): v for k, v in assets.items()},
                   source_sha256=current['source_sha256'] | {f: common.sha(ROOT/f) for f in FILES},
                   training_fit_only=True, training_steps=0, unique_sources=1,
                   repetitions_per_gpu=2, gpus=2, wall_cap_seconds=900,
                   historical_intermediate_tensors_available=False,
                   gt_tensor_decoded=False, protected_pixels_accessed=False)
    return receipt


def difference(left, right):
    import torch
    if left.shape != right.shape or left.dtype != right.dtype:
        raise RuntimeError('comparison shape/dtype differs')
    if not bool(torch.isfinite(left).all() and torch.isfinite(right).all()):
        raise RuntimeError('nonfinite diagnostic tensor')
    delta = left.double() - right.double()  # avoid uint8 subtraction wraparound
    changed = left != right
    return dict(equal=bool(torch.equal(left, right)), elements=left.numel(),
                different_elements=int(changed.sum()), different_fraction=float(changed.double().mean()),
                max_abs=float(delta.abs().max()), mean_abs=float(delta.abs().mean()),
                rms=float(delta.square().mean().sqrt()), signed_mean=float(delta.mean()))


def compare_traces(left, right, order):
    if set(left) != set(right) or set(left) != set(order) or len(set(order)) != len(order):
        raise RuntimeError('trace stages differ')
    stats = {key: difference(left[key], right[key]) for key in order}
    return dict(first_different_stage=next((k for k in order if not stats[k]['equal']), None), stages=stats)


def runtime(device):
    import torch
    p = torch.cuda.get_device_properties(device)
    driver = subprocess.run(['nvidia-smi', '--query-gpu=uuid,driver_version', '--format=csv,noheader'],
                            capture_output=True, text=True, timeout=10)
    if driver.returncode:
        raise RuntimeError('cannot record NVIDIA driver version')
    return dict(hostname=socket.gethostname(), device=device.index, name=p.name, uuid=str(p.uuid),
                memory_bytes=p.total_memory, capability=list(torch.cuda.get_device_capability(device)),
                python=sys.version, executable=sys.executable, torch=torch.__version__,
                cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                driver_versions=driver.stdout.strip().splitlines(),
                packages={n: importlib.metadata.version(n) for n in
                          ('numpy', 'safetensors', 'torchvision', 'transformers', 'torchmetrics', 'einops')},
                matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
                cudnn_tf32=torch.backends.cudnn.allow_tf32, cudnn_benchmark=torch.backends.cudnn.benchmark,
                cudnn_deterministic=torch.backends.cudnn.deterministic,
                deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                matmul_precision=torch.get_float32_matmul_precision(),
                flash_sdp=torch.backends.cuda.flash_sdp_enabled(),
                memory_efficient_sdp=torch.backends.cuda.mem_efficient_sdp_enabled(),
                math_sdp=torch.backends.cuda.math_sdp_enabled(),
                torch_config=torch.__config__.show(),
                environment={k: os.environ.get(k) for k in
                             ('CUDA_VISIBLE_DEVICES', 'CUBLAS_WORKSPACE_CONFIG', 'NVIDIA_TF32_OVERRIDE',
                              'OMP_NUM_THREADS', 'PYTHONPATH', 'TORCH_HOME')})


def traced_restore(flow, observation, seed):
    """Observe the original restore without changing its arithmetic/backends."""
    import torch
    from smoke.qualify_foundir_flux_energy_v1 import restore, deployed_proposal
    from lowlight_restoration.scalar_residual import quantized_scalar_residual
    tensors, order = {}, []

    def capture(key, tensor):
        if key in tensors:
            raise RuntimeError('duplicate trace stage: '+key)
        value = tensor.detach().cpu().contiguous().clone()
        if not bool(torch.isfinite(value).all()):
            raise RuntimeError('nonfinite stage: '+key)
        tensors[key] = value
        order.append(key)

    encode, decode = flow.ae.encode, flow.ae.decode
    count = [0]

    def enc(value):
        capture('observation_normalized', value)
        result = encode(value)
        capture('encoded_reference', result)
        return result

    def dec(value):
        capture('final_latent', value)
        result = decode(value)
        capture('decoded_raw_unclamped', result)
        return result

    def before(_module, args, kwargs):
        i = count[0]
        capture(f'step_{i:02d}_input', kwargs['x'])
        capture(f'step_{i:02d}_time', kwargs['timesteps'])
        if i == 0:
            for key in ('x_ids', 'ctx', 'ctx_ids'):
                capture('conditioning_'+key, kwargs[key])

    def after(_module, args, kwargs, output):
        capture(f'step_{count[0]:02d}_velocity', output)
        count[0] += 1

    flow.ae.encode, flow.ae.decode = enc, dec
    handles = [flow.model.register_forward_pre_hook(before, with_kwargs=True),
               flow.model.register_forward_hook(after, with_kwargs=True)]
    try:
        raw = restore(flow.model, flow.ae, observation, flow.context['cfg_context'], flow.context['cfg_ids'], seed)
        capture('raw_clamped', raw)
        raw_u8, _ = quantized_scalar_residual(observation, raw, 1.)
        capture('raw_uint8', raw_u8)
        proposal, _ = deployed_proposal(observation, raw)
        capture('proposal_uint8', proposal)
        if count[0] != 50:
            raise RuntimeError('expected exactly 50 FLUX forwards')
    finally:
        flow.ae.encode, flow.ae.decode = encode, decode
        for handle in handles:
            handle.remove()
    return tensors, order


def worker(output, index):
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    from smoke.flux_tail_energy_runtime_v1 import FrozenFlow
    receipt = json.loads((output/'preflight.json').read_text())
    validate_identity(receipt['identity'])
    for name, sha in receipt['source_sha256'].items():
        common.bound(name, sha)
    parent_path = common.path(receipt['parent_run'])
    parent = json.loads(common.bound(str(parent_path/'preflight.json'), receipt['parent_preflight_sha256']).read_text())
    device = torch.device('cuda', index)
    torch.cuda.set_device(device)
    common.write_new(output/f'gpu{index}_runtime.json', runtime(device))
    with safe_open(str(parent_path/'cache_000.safetensors'), framework='pt', device='cpu') as h:
        observation = h.get_tensor('observation')
        historical = h.get_tensor('proposal')[None]
    if observation.dtype != torch.uint8 or observation.shape != (3, 512, 512):
        raise RuntimeError('invalid cached observation')
    # No GT tensor is loaded; the old source's exact cached observation is reused.
    image = observation[None].to(device).float()/127.5-1
    flow = FrozenFlow(parent, device)
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats(device)
    rows = []
    for repetition in range(2):
        trace, order = traced_restore(flow, image, receipt['identity']['seed'])
        filename = f'gpu{index}_repeat{repetition}.safetensors'
        if (output/filename).exists():
            raise FileExistsError(filename)
        save_file(trace, str(output/filename))
        row = dict(device=index, repetition=repetition, file=filename, sha256=common.sha(output/filename),
                   order=order, historical_proposal_difference=difference(trace['proposal_uint8'], historical),
                   stage_sha256={k: hashlib.sha256(v.view(torch.uint8).numpy().tobytes()).hexdigest()
                                 for k, v in trace.items()})
        row['historical_proposal_difference']['different_rgb_pixels'] = int(
            (trace['proposal_uint8'] != historical).any(dim=1).sum())
        row['historical_proposal_difference']['different_rgb_pixel_fraction'] = float(
            (trace['proposal_uint8'] != historical).any(dim=1).float().mean())
        common.write_new(output/f'gpu{index}_repeat{repetition}.json', row)
        print('BASELINE_PARITY_SAMPLE '+json.dumps({k: row[k] for k in
              ('device', 'repetition', 'historical_proposal_difference')}), flush=True)
        rows.append(row)
        del trace
    flow.assert_frozen()
    common.write_new(output/f'gpu{index}_complete.json', dict(rows=rows, seconds=time.monotonic()-started,
                      peak_memory_gib=torch.cuda.max_memory_allocated(device)/1024**3))


def summarize(output):
    from safetensors.torch import load_file
    rows = [json.loads((output/f'gpu{i}_repeat{j}.json').read_text()) for i in range(2) for j in range(2)]
    for row in rows:
        common.bound(str(output/row['file']), row['sha256'])
    comparisons = {}
    for name, a, b in [('gpu0_repeat', 0, 1), ('gpu1_repeat', 2, 3), ('across_gpus_repeat0', 0, 2)]:
        left, right = [load_file(str(output/rows[k]['file'])) for k in (a, b)]
        if rows[a]['order'] != rows[b]['order']:
            raise RuntimeError('trace order differs')
        comparisons[name] = compare_traces(left, right, rows[a]['order'])
        del left, right
    parity = all(row['historical_proposal_difference']['equal'] for row in rows)
    result = dict(diagnostic_complete=True, historical_baseline_parity_pass=parity,
                  qualification_pass=False, training_steps=0, gt_tensor_decoded=False,
                  historical_intermediate_tensors_available=False,
                  historical_proposal_differences=[r['historical_proposal_difference'] for r in rows],
                  comparisons=comparisons, preflight_sha256=common.sha(output/'preflight.json'),
                  artifact_sha256={p.name: common.sha(p) for p in output.iterdir() if p.is_file()},
                  interpretation='Repeat/device stage differences localize current instability only; '
                                 'historical stage divergence and migration root cause remain unknown.')
    common.write_new(output/'result.json', result)
    print('BASELINE_PARITY_DIAGNOSTIC_COMPLETE historical_parity='+str(parity).lower(), flush=True)
    return 0 if parity else 2


def run(output):
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise RuntimeError('requires exactly two visible GPUs on one node')
    identities = [runtime(torch.device('cuda', i)) for i in range(2)]
    if len({v['uuid'] for v in identities}) != 2:
        raise RuntimeError('duplicate GPU identities')
    for i, p in enumerate(identities):
        if (not any(n in p['name'] for n in ('H100', 'H200'))
                or p['capability'] != [9, 0] or p['memory_bytes'] < 75*1024**3):
            raise RuntimeError('requires H100/H200 SM90 with >=75GiB')
        free, total = torch.cuda.mem_get_info(i)
        if total-free > 2*1024**3:
            raise RuntimeError(f'GPU {i} busy; no overlap/kill')
    if output.exists():
        raise FileExistsError('preserve old attempt; choose a fresh diagnostic ID')
    receipt = preflight()
    output.mkdir(parents=True, exist_ok=False)
    common.write_new(output/'preflight.json', receipt)
    common.write_new(output/'hardware.json', identities)
    processes = []
    try:
        for i in range(2):
            processes.append(subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()),
                                              'worker', '--run-id', output.name, '--device', str(i)]))
        while any(p.poll() is None for p in processes):
            if any(p.poll() not in (None, 0) for p in processes):
                raise RuntimeError('diagnostic worker failed; preserve partial artifacts')
            time.sleep(.5)
        if any(p.returncode != 0 for p in processes):
            raise RuntimeError('diagnostic worker failed')
    finally:
        for p in processes:
            if p.poll() is None:
                p.terminate()
        for p in processes:
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()
    return summarize(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['preflight', 'run', 'worker', 'summarize'])
    parser.add_argument('--run-id', default='baseline_parity_2gpu_v1')
    parser.add_argument('--device', type=int, choices=[0, 1])
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.run_id):
        parser.error('invalid diagnostic run ID')
    output = common.path(OUTPUT_ROOT)/args.run_id
    if args.phase == 'preflight':
        receipt = preflight()
        print('BASELINE_PARITY_PREFLIGHT_PASS '+json.dumps(receipt), flush=True)
        return 0
    if args.phase == 'worker':
        if args.device is None:
            parser.error('--device required')
        worker(output, args.device)
        return 0
    if args.phase == 'summarize':
        return summarize(output)
    return run(output)


if __name__ == '__main__':
    sys.exit(main())
