# Z-Image Layer-1 deterministic bridge readiness — 2026-09-15

## Decision and scope

The user stopped the plateaued 20,000-step conditional noise-to-GT run
externally and explicitly authorized starting Layer-1. This is a **pure
Z-Image Base 6B teacher**, not an independent restoration/deblurring branch.
It uses isolated Blur training data only. Development pairs are descriptive
evaluation only and never enter optimization.

Distillation and decoder tuning remain downstream gates. Qwen degradation
conditioning, mixed degradation, sealed internal data, the official FoundIR
test, and development-GT optimization remain prohibited.

## Layer-1 formulation

- Encode aligned LQ and GT RGB crops with the original frozen Z-Image VAE.
- Train the deterministic bridge
  `x_sigma=(1-sigma)*GT_latent+sigma*LQ_latent` with velocity target
  `LQ_latent-GT_latent`.
- Begin inference at the exact LQ latent, so the zero-initialized model is an
  exact latent identity mapping instead of a random-noise generator.
- Feed a fixed 1x/2x/4x LQ-latent pyramid through zero-initialized projections
  at the image input and DiT depths 0/5/10/15/20/25.
- Preserve the Base DiT and add FP32 rank-64 LoRA plus a zero-initialized
  per-channel velocity gate.
- Optimize flow MSE, clean-latent Charbonnier, decoded-RGB Charbonnier, and
  differentiable LPIPS while keeping both VAE encoder and decoder frozen.

## Tiny gate

Train on 16 fixed Blur crops from 16 source groups for 1,000 optimizer steps,
global batch 8 on one homogeneous 8xH100/H200 host. Evaluate the same 16 train
crops and 16 source-disjoint descriptive development crops at steps
0/250/500/1000. Every snapshot stores per-image Input|Output|GT PNGs,
PSNR/LPIPS, latent error, metrics, parameter identity, and checkpoint hashes.

Teacher admission requires train PSNR >=26.4 dB, LPIPS <=0.16, both metrics
better than input, and human visual review. Completion does not automatically
start a larger run, distillation, or decoder tuning.

Focused tests pass (`4 passed`), full model/data hash preflight passes, and a
real Base-6B CPU smoke passes. The latter loaded the pinned DiT, text encoder,
and frozen VAE, verified exact zero-gate LQ-latent identity, and completed a
frozen-decoder RGB/LPIPS forward/backward with a finite nonzero velocity-gate
gradient. An expanded two-update smoke additionally verified FP32 AdamW state
and nonzero second-update gradients in both multi-depth projections
(`2.049e-3`) and DiT LoRA (`2.609e-5`). Evidence is under
`logs/zimage_bridge_cpu_23ufvxax/`.

## Launch and monitoring

```bash
cd /inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
smoke/launch_zimage_deterministic_bridge_tiny_8gpu.sh
```

The launcher accepts both H100 and H200, refuses mixed/partial GPU visibility,
uses tmux when available and nohup otherwise, refuses to overwrite evidence,
records five-second GPU telemetry, and emits an explicit terminal marker.
