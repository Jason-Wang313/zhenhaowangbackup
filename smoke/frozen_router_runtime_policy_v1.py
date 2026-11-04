"""Cross-node Hopper drivers may differ; software and per-node receipts may not."""
import re


def execution_signature(value):
    name = value.get('gpu_name', '')
    if not any(f in name for f in ('H100', 'H200')):
        raise ValueError('only H100/H200 are supported')
    raw = value.get('nvidia_driver', '')
    # Both proprietary and open NVIDIA module descriptions occur on this cluster.
    releases = re.findall(r'^NVRM version:\s+NVIDIA UNIX (?:Open Kernel Module for x86_64|x86_64 Kernel Module)\s+(\d+\.\d+(?:\.\d+)?)(?:\s|$)', raw, re.MULTILINE)
    if len(releases) != 1:
        raise ValueError('missing or unrecognized NVIDIA driver description')
    signature = dict(value)
    signature['gpu_name'] = 'Hopper_H100_or_H200'
    signature.pop('nvidia_driver')
    return signature


def check_runtimes(values):
    if len(values) != 4:
        raise ValueError('four node runtime receipts required')
    signatures = [execution_signature(v) for v in values]
    if any(v != signatures[0] for v in signatures[1:]):
        raise ValueError('H100/H200 nodes must share exact CUDA/PyTorch/cuDNN/metric/memory-policy versions; driver releases may differ')
    return signatures[0]
