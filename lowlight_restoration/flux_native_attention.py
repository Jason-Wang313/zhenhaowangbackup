"""Rematerialize each native attention VJP without changing its arithmetic.

The original attention callable (including layouts, SDPA backend and autograd)
executes unchanged. Only the lifetime of saved derivative tensors changes.
This avoids the full-model derivative checkpoint reconstructing all layers'
quadratic derivative tensors at once. No query/head partition or manual VJP.
"""
from __future__ import annotations

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.checkpoint import checkpoint


def _native_vjp(q, k, v, grad_out, function, create_graph):
    with torch.enable_grad(), sdpa_kernel(SDPBackend.MATH):
        # Q/K/V share projection/view ancestors. autograd.grad targeting those
        # non-leaf edges can visit the caller's upstream graph while computing
        # this local VJP. Give each input a distinct identity edge. clone keeps
        # the higher-order connection (unlike detach), and its backward is the
        # identity. Constants alone get temporary grad-enabled leaves.
        inputs = tuple(x.clone() if x.requires_grad else x.detach().clone().requires_grad_(True)
                       for x in (q,k,v))
        output = function(*inputs)
        return torch.autograd.grad(output, inputs, grad_out, create_graph=create_graph)


class _AttentionWithCheckpointedVJP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, function):
        ctx.save_for_backward(q, k, v)
        ctx.function = function
        with sdpa_kernel(SDPBackend.MATH):
            return function(q, k, v)

    @staticmethod
    def backward(ctx, grad_out):
        q, k, v = ctx.saved_tensors
        higher_order = torch.is_grad_enabled()
        def calculate(q, k, v, gradient):
            return _native_vjp(q,k,v,gradient,ctx.function,higher_order)
        if higher_order:
            gradients = checkpoint(calculate,q,k,v,grad_out,use_reentrant=False)
        else:
            gradients = calculate(q,k,v,grad_out)
        return *gradients, None


def checkpointed_attention(q, k, v, function):
    """Run exactly function(q,k,v); checkpoint its native VJP for higher orders."""
    if (q.ndim != 4 or k.ndim != 4 or v.ndim != 4 or q.shape[:2] != k.shape[:2]
            or k.shape[:3] != v.shape[:3] or q.shape[-1] != k.shape[-1]
            or q.shape[-3] < 1 or q.shape[-2] < 1 or k.shape[-2] < 1
            or q.dtype != k.dtype or q.dtype != v.dtype):
        raise ValueError('expected matching nonempty B,H,N,D tensors')
    return _AttentionWithCheckpointedVJP.apply(q,k,v,function)
