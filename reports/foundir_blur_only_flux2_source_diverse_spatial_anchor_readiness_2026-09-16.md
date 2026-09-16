# FLUX.2 Blur-only source-diverse spatial-anchor pilot readiness — 2026-09-16

**DIAGNOSTIC ONLY / 非主线.** This bounded pilot is not a SOW result, does not
access a protected split, and cannot launch full scale automatically.

## Authorization and purpose

The completed 16-source trajectory-endpoint gate proved a causal PSNR gain over
its paired flow control, but both arms overfit and regressed on fixed
source-disjoint development. The user explicitly authorized the recommended
source-diverse pilot with `sure,execute`. The final resolved hash-bound design
event is `flux2_blur_source_diverse_spatial_anchor_design_v1-c8d0dfd7adb8e2b7`,
hold SHA-256
`dbf3866095d66b6eb5ebe07faade3b353be41d170767d51dff3471a1af003ce0`.
The recorded decision is USER authorization and does not represent or fabricate
an advisor reply.

## Frozen protocol

- exact FLUX.2 Blur step-10k parent SHA-256
  `e0c834d88fc69ba6916032ef7763cdf84047a60d2407eeef23eae8992359c439`;
- config SHA-256
  `ab5a41af64d6a8e39e228625a18b5b978ab616cc007308dcd2999242bb1a05be`;
- deterministic manifest SHA-256
  `ebd0319ab2f82e7fb629d5ed8fa09309100c9ef28ba80c46a38529288a2e31f8`;
- 512 isolated-Blur training source groups, one deterministic record and crop
  per group; zero overlap with the old 16 tiny-train or fixed development
  groups;
- paired `flow_control` and `trajectory_spatial_anchor` arms with independent
  fresh AdamW, identical record order, random-t flow noise/timestep, endpoint
  noise, parent, global batch, and checkpoint schedule;
- 400 optimizer steps per arm, global batch 8, checkpoints 200/400;
- exact official 50-step trajectory, transitions 1--46 detached and 47--50
  differentiable with non-reentrant activation checkpointing;
- candidate weights: flow MSE `1.0`, endpoint RGB Charbonnier `1.0`, LPIPS
  `0.02`, half/quarter-scale RGB Charbonnier total `0.5`, endpoint latent L1
  `0.25`, and average-pool-16 LQ identity `0.1`;
- fixed 16 pilot-train visuals at 0/200/400 and existing fixed 16
  source-disjoint development visuals at 0/400;
- development cannot tune arm, weights, or checkpoint.

The step-400 candidate must reach at least the parent fixed-development
`24.591062 / 0.150766` subject to the predeclared LPIPS allowance, beat control
by at least `0.15 dB` without more than `0.003` LPIPS regression, beat parent
PSNR on at least 10/16 records, and avoid any per-record PSNR regression above
2 dB. Human visual review is mandatory.

## Verification

- governance resolution passes;
- 28 focused tests pass, covering source uniqueness/exclusion, exact crop
  bounds, paired random streams, frozen weights/schedule, H100/H200 global
  batch equivalence, 46+4 trajectory coverage, official solver equality,
  detached prefix, checkpointed suffix output/gradient equality, Charbonnier
  gradients, protected flags, and the complete development gate;
- Python compilation, shell syntax, `git diff --check`, parent/model/manifest
  hashes, all 528 pair identities, prior VAE evidence, and launcher `--check`
  pass;
- before candidate optimizer step 1, every CUDA rank performs a full-resolution
  exact 50-step rehearsal, requires finite nonzero gradients on every trainable
  LoRA tensor, records peak memory, and rejects any frozen-VAE parameter
  gradient;
- Qwen degradation conditioning, mixed degradation, pixel refiner, decoder
  tuning, distillation, development-GT optimization, sealed internal, and the
  official FoundIR test remain disabled.

## Launch

```bash
cd /inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
smoke/launch_foundir_blur_only_flux2_source_diverse_spatial_anchor_v1_8gpu.sh
```

Monitor:

```bash
tail -f logs/foundir_blur_only_flux2_source_diverse_spatial_anchor_pilot_v1.log
```

No GPU job has been started from the CPU-only Codex shell.
