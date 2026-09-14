"""Pinned, process-local repair for PyTorch deterministic CUDA boolean assignment.

Only the two Base pipeline padding assignments change. The ignored upstream
checkout is never edited and ordinary historical workers remain unchanged.
"""
from __future__ import annotations

import hashlib
import inspect
from pathlib import Path

import torch

PIPELINE_SHA256 = 'ace84d674ea8356ffc0ebfeafefc4ceadba40590f52908071adb1e2e3a98a5e2'


def padding_where(features, mask, token):
    if features.ndim != 2 or mask.dtype != torch.bool or mask.shape != (features.shape[0],):
        raise ValueError('unexpected padding feature/mask shape')
    if token.requires_grad:
        raise ValueError('compatibility repair supports frozen padding tokens only')
    if token.shape != (1, features.shape[1]) or token.dtype != features.dtype or token.device != features.device:
        raise ValueError('unexpected padding token shape/precision/device')
    # Frozen pad tokens have no gradient; unmasked feature gradients pass
    # through exactly and masked gradients are zero, as with index assignment.
    return torch.where(mask[:, None], token, features)


def build_padding_function(module):
    path = Path(inspect.getsourcefile(module))
    if hashlib.sha256(path.read_bytes()).hexdigest() != PIPELINE_SHA256:
        raise RuntimeError('upstream Z-Image pipeline hash changed; re-audit repair')
    original = module.model_fn_z_image_turbo
    source = inspect.getsource(original)
    replacements = {
        'x[torch.cat(patch_metadata.get("x_pad_mask"))] = dit.x_pad_token.to(dtype=x.dtype, device=x.device)':
        'x = _padding_where(x, torch.cat(patch_metadata.get("x_pad_mask")), dit.x_pad_token.to(dtype=x.dtype, device=x.device))',
        'cap_feats[torch.cat(patch_metadata.get("cap_pad_mask"))] = dit.cap_pad_token.to(dtype=x.dtype, device=x.device)':
        'cap_feats = _padding_where(cap_feats, torch.cat(patch_metadata.get("cap_pad_mask")), dit.cap_pad_token.to(dtype=x.dtype, device=x.device))',
    }
    for before, after in replacements.items():
        if source.count(before) != 1:
            raise RuntimeError('expected exactly one occurrence of each pinned padding assignment')
        source = source.replace(before, after)
    namespace = dict(original.__globals__, _padding_where=padding_where)
    # Compile only a hash-pinned function already imported from local upstream.
    # No external source or user-controlled expression enters this transform.
    exec(compile(source, str(Path(__file__).resolve())+'::patched_pipeline', 'exec'), namespace)
    return original, namespace[original.__name__], dict(
        upstream_sha256=PIPELINE_SHA256, replacement_count=2,
        transformed_function_sha256=hashlib.sha256(source.encode()).hexdigest())


def padding_operator_probe(device):
    """Exact outputs/input-gradients for real width, masks and both precisions.

    Legacy uses ordinary execution; replacement must run with strict
    deterministic execution. This is called on CUDA before loading 6B weights.
    """
    previous = torch.are_deterministic_algorithms_enabled()
    previous_warn = torch.is_deterministic_algorithms_warn_only_enabled()
    rows = []
    try:
        for dtype in (torch.float32, torch.bfloat16):
            for kind in ('none', 'some', 'all'):
                torch.manual_seed(904)
                x = torch.randn(128,3840,device=device,dtype=dtype)
                token = torch.randn(1,3840,device=device,dtype=dtype)
                mask = torch.arange(128,device=device) >= {'none':128,'some':98,'all':0}[kind]
                weight = torch.randn_like(x)
                torch.use_deterministic_algorithms(False)
                a = x.clone().requires_grad_(True)
                legacy = a.clone()
                legacy[mask] = token
                (legacy.float()*weight.float()).sum().backward()
                torch.use_deterministic_algorithms(True, warn_only=False)
                b = x.clone().requires_grad_(True)
                repaired = padding_where(b,mask,token)
                (repaired.float()*weight.float()).sum().backward()
                if not torch.equal(legacy,repaired) or not torch.equal(a.grad,b.grad):
                    raise RuntimeError('padding operator forward/backward equivalence failed')
                rows.append(dict(dtype=str(dtype),mask=kind,forward_exact=True,gradient_exact=True))
    finally:
        torch.use_deterministic_algorithms(previous,warn_only=previous_warn)
    return dict(device=str(device),cases=rows,status='pass')


def validate_model_repair(model, item, prompt, metric, e, arm, seed, folder):
    """No optimizer updates; validate old/new + strict repeat at actual crop size."""
    from diffsynth.pipelines import z_image as module
    from lowlight_restoration.zimage_repro_batch import seed_everything,tensor_digest
    from smoke.audit_foundir_blur_only_zimage_optimization import objective,loss_terms,write_json
    original, repaired, metadata = build_padding_function(module)
    previous = torch.are_deterministic_algorithms_enabled()
    previous_warn = torch.is_deterministic_algorithms_warn_only_enabled()
    rows = []
    passed = False
    try:
        for name, function, strict in [('original',original,False),('repaired',repaired,False),
                                       ('strict_a',repaired,True),('strict_b',repaired,True)]:
            module.model_fn_z_image_turbo = function
            torch.use_deterministic_algorithms(strict,warn_only=False)
            seed_everything(seed)
            model.zero_grad(set_to_none=True)
            metric.reset()
            latent,rgb = model(item,prompt)
            if arm=='lora_rgb':
                loss = (rgb.float()-item['gt_rgb'].float()).square().mean()
            else:
                _,terms = loss_terms(latent,rgb,item,metric,e)
                loss = objective(terms,arm,e)
            loss.backward()
            rows.append(dict(name=name,loss=float(loss.detach()),
                             output_digest=tensor_digest([('latent',latent),('rgb',rgb)]),
                             gradient_digest=tensor_digest([(n,p.grad) for n,p in model.named_parameters() if p.requires_grad])))
        if len({r['output_digest'] for r in rows}) != 1 or len({r['gradient_digest'] for r in rows}) != 1:
            raise RuntimeError('full-model padding/strict-mode equivalence failed; do not train')
        passed = True
        return dict(status='pass',**metadata,probes=rows)
    finally:
        # Preserve failed evidence too, and never leave a failed patch activated.
        write_json(folder/'padding_model_gate.json',dict(status='pass' if passed else 'failed',**metadata,probes=rows))
        module.model_fn_z_image_turbo = repaired if passed else original
        torch.use_deterministic_algorithms(previous,warn_only=previous_warn)
        model.zero_grad(set_to_none=True)
        metric.reset()
