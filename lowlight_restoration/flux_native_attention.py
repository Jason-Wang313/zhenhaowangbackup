"""Full softmax attention with query-tiled, differentiable backward.

All keys are used by every query. This is not sparse/local attention. Forward
uses PyTorch math SDPA; backward recomputes probabilities in each query tile.
Checkpointing each tile VJP keeps the higher-order derivative graph from saving
all layers' quadratic attention matrices at once. Generation never uses this.
"""
from __future__ import annotations

import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.checkpoint import checkpoint

QUERY_CHUNK = 128


def _tile_vjp(q, k, v, grad_out):
    # Math SDPA accumulates FP16/BF16 inputs in FP32. Keep FP64 for numerical
    # reference tests. No inputs/gradients are detached, including grad_out.
    dtype = torch.float32 if q.dtype in (torch.float16, torch.bfloat16) else q.dtype
    qf, kf, vf, gf = (x.to(dtype) for x in (q, k, v, grad_out))
    scale = q.shape[-1] ** -.5
    # Split scaling matches PyTorch's math SDPA formulation.
    qscaled, kscaled = qf * scale**.5, kf * scale**.5
    probability = (qscaled @ kscaled.transpose(-2, -1)).softmax(dim=-1)
    grad_probability = gf @ vf.transpose(-2, -1)
    grad_score = probability * (grad_probability - (grad_probability * probability).sum(-1, keepdim=True))
    grad_q = (grad_score @ kscaled) * scale**.5
    grad_k = (grad_score.transpose(-2, -1) @ qscaled) * scale**.5
    grad_v = probability.transpose(-2, -1) @ gf
    return grad_q, grad_k, grad_v


class _TiledAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, chunk):
        ctx.save_for_backward(q, k, v)
        ctx.chunk = chunk
        pieces = []
        with sdpa_kernel(SDPBackend.MATH):
            for start in range(0, q.shape[-2], chunk):
                pieces.append(F.scaled_dot_product_attention(q[..., start:start+chunk, :], k, v,
                                                            dropout_p=0., is_causal=False))
        return torch.cat(pieces, dim=-2)

    @staticmethod
    def backward(ctx, grad_out):
        q, k, v = ctx.saved_tensors
        q_grads, grad_k, grad_v = [], None, None
        for start in range(0, q.shape[-2], ctx.chunk):
            args = (q[..., start:start+ctx.chunk, :], k, v,
                    grad_out[..., start:start+ctx.chunk, :])
            if torch.is_grad_enabled():
                dq, dk, dv = checkpoint(_tile_vjp, *args, use_reentrant=False)
            else:
                dq, dk, dv = _tile_vjp(*args)
            q_grads.append(dq)
            grad_k = dk if grad_k is None else grad_k + dk
            grad_v = dv if grad_v is None else grad_v + dv
        # Accumulate all query-tile K/V contributions in FP32 before the one
        # cast back to BF16; rounding every tile separately loses accuracy.
        return torch.cat(q_grads, dim=-2).to(q.dtype), grad_k.to(k.dtype), grad_v.to(v.dtype), None


def tiled_attention(q, k, v, chunk=QUERY_CHUNK):
    if (q.ndim != 4 or k.ndim != 4 or v.ndim != 4 or q.shape[:2] != k.shape[:2]
            or k.shape[:3] != v.shape[:3] or q.shape[-1] != k.shape[-1]
            or q.shape[-2] < 1 or k.shape[-2] < 1 or chunk < 1
            or q.dtype != k.dtype or q.dtype != v.dtype):
        raise ValueError('expected matching B,H,N,D tensors and a positive query chunk')
    return _TiledAttention.apply(q, k, v, int(chunk))
