# Blur-only Z-Image solver-aligned paired-flow teacher v3 readiness — 2026-09-14

> **DIAGNOSTIC ONLY / 非主线。** This is a bounded Blur-only teacher
> learnability continuation, not a five-type FLUX.2 mainline or official
> FoundIR-test result.

## Decision and bounded change

V2 repaired the invalid v1 architecture and greatly improved output quality,
but its step-400 fixed-train result remained below the input by `1.4404 dB`
PSNR despite improving LPIPS by `0.0670`. The v2 trainer sampled a shifted
1,000-point grid and multiplied paired restoration loss by DiffSynth's
empirical text-to-image `training_weight`. That weight is zero at the exact
sigma-1 endpoint and has no established justification for this deterministic
LQ-to-GT flow.

V3 is an isolated continuation, not a model-family or data change:

- initialize exactly from immutable v2 step-400 LoRA SHA-256
  `2a3e9b3392dfea20f9e6e869febce1ff99bcaa751e317e247c908f683a4ecd39`;
- explicitly reset optimizer state;
- preserve Z-Image Base, frozen VAE/text encoder, DiT LoRA rank/alpha 32/32,
  no ControlNet, exact-LQ initialization, 50-step Euler, shift 6, and CFG 1;
- train only at the exact 50 timesteps used by deployed inference;
- use plain unweighted velocity MSE;
- schedule all 6,400 presentations so every one of 16 train records sees every
  one of 50 solver timesteps exactly 8 times;
- run 800 additional steps, saving cumulative steps 800 and 1200 under a new
  output root;
- evaluate parent step 400 and new steps 800/1200 on the unchanged fixed 16
  train plus 16 source-disjoint development crops.

Config:
`configs/foundir_blur_only_zimage_teacher_paired_flow_continuation_v3.json`,
SHA-256 `6909e7fcf2397fb4b911870076a74544b1f111edb3e0a895309490e804eaf438`.

## Verification

- The matching user continuation decision resolves hold
  `zimage_blur_teacher_paired_flow_tiny_overfit_v2-ea895f0c24c75257`;
  governance check passes.
- 26 focused v2/v3 tests passed.
- Python compilation, runtime imports, shell syntax, `git diff --check`, and
  the complete launcher `--check` path passed.
- Full SHA-256 verification passed for the six roughly 29 GB Base/VAE/text
  artifacts; no ControlNet artifact is resolved.
- The immutable parent checkpoint hash, metadata step 400, and all 476 LoRA
  tensor keys passed.
- A real 6B CPU constructor loaded all 476 parent tensors exactly. A 64x64
  forward/backward at the exact sigma-1 solver endpoint returned finite plain
  MSE `0.2126598209142685`; all 476 trainable tensors had finite gradients and
  all 238 LoRA-B tensors had nonzero gradients.
- The launcher detects homogeneous 8x H100/H200 at runtime, validates BF16
  attention on all GPUs, persists telemetry/log/terminal status, and falls
  back to `nohup + PID` when tmux is unavailable.

## Gate

Full teacher training remains prohibited until the final fixed-train result
improves both PSNR and LPIPS over input and the saved triptychs pass human
visual review. The fixed development results are reported as generalization
evidence. Distillation remains downstream of a valid teacher. Decoder tuning
remains conditional on an inadequate distilled result plus a measured
decoder/latent ceiling.

Qwen degradation conditioning, mixed degradation, sealed internal, and the
official FoundIR test are all disabled and prohibited.
