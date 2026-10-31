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
        # Constants need temporary grad-enabled inputs for autograd.grad. A
        # live differentiable input is passed through unchanged, never detached.
        inputs = tuple(x if x.requires_grad else x.detach().requires_grad_(True) for x in (q,k,v))
        output = function(*inputs)
        # Q/K/V may share view ancestors in the caller's live graph. Inner
        # autograd must not release those buffers before outer backward uses
        # them. The temporary native output graph dies on return unless needed
        # by the requested higher-order graph.
        return torch.autograd.grad(output, inputs, grad_out, create_graph=create_graph, retain_graph=True)


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
