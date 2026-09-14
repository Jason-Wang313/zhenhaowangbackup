# Blur-only Z-Image paired-flow teacher v2 readiness — 2026-09-14

> **DIAGNOSTIC ONLY / 非主线。** This is a Blur-only teacher learnability gate,
> not the five-type FLUX.2 result and not an official FoundIR-test result.

## Current outcome

The first Z-Image teacher was scientifically invalid for restoration: its
frozen Turbo-only ControlNet was attached to undistilled Base, inference used
DiffSynth's default shift 3 instead of the published Base shift 6, and CFG 4
used a convention that differs from the native implementation. Its final
memorized-train output was only 9.3654 dB / 0.8408 LPIPS versus the input's
24.7232 dB / 0.3085 LPIPS. That immutable v1 run is preserved and is not being
restarted or overwritten.

The corrected v2 implementation is CPU-verified and ready for its bounded
8-GPU tiny-overfit. It does not yet claim image-quality success because this
shell exposes no CUDA device.

## Corrected architecture

- Foundation: undistilled `Tongyi-MAI/Z-Image` 6B Base, Apache-2.0.
- Frozen modules: VAE and native Qwen3 text encoder.
- Trainable module: DiT LoRA rank/alpha 32/32, 79,380,480 parameters.
- LQ condition: no ControlNet and no edit/image-to-LoRA path.
- Paired path: `x_sigma = (1-sigma) * GT_latent + sigma * LQ_latent`.
- Velocity target: `LQ_latent - GT_latent`.
- Inference: initialize with the exact LQ VAE latent at sigma 1 and integrate
  50 descending-sigma Euler steps to sigma 0.
- Scheduler: published Base static shift 6 for training and inference.
- CFG: 1.0, so no native/DiffSynth CFG convention ambiguity remains.
- Data: the same immutable 16 training crops and 16 source-disjoint visual
  development crops used by v1.

Config:
`configs/foundir_blur_only_zimage_teacher_paired_flow_overfit_v2.json`,
SHA-256 `d6f5946fcd75050a17d96c093f7590552f89d14caaf7dce0c659fe35fcb2fbb8`.

## Verification evidence

- Governance: hold `zimage_blur_teacher_tiny_overfit-a6becc1eff058fec` is
  resolved by the user's explicit continuation decision; decision SHA-256 is
  `48f62ed9fc5efb05063b86be8d15615c2b7d43cd848f92e2daee73005b7c036f`.
- Focused unit/contract tests: 14 passed. They cover path endpoints, exact
  Euler recovery, numerical agreement between the 50-step DiffSynth shift-6
  schedule and the official Base scheduler (within the NumPy/Torch linspace
  float32 rounding tolerance), exact-LQ evaluator initialization,
  forbidden scope, no-ControlNet artifacts, fixed crop identities,
  H100/H200 portability, and global-batch arithmetic.
- Complete artifact verification passed for both DiT shards, all three text
  encoder shards, and the VAE. Their hashes match the previously frozen
  identities. The model artifact set contains six files and no ControlNet.
- Real 6B CPU constructor/64x64 forward/backward passed with finite loss
  `3.0448391437530518`, finite gradients for all 476 trainable tensors, and
  nonzero gradients for the expected 238 LoRA-B tensors.
- Python compilation, runtime imports, launcher shell syntax, retained-tmux launcher
  `--check`, governance check, and `git diff --check` passed.
- This shell reports PyTorch 2.7.0+cu128 but CUDA unavailable/device count 0;
  therefore no v2 training or image-quality evaluation has been claimed here.

## Gate and next action

Run the new v2 launcher on one homogeneous 8xH100/H200 node. It writes to a
new output root and cannot overwrite v1. It must produce fixed
`Input | Z-Image Paired-Flow Output | Ground Truth` triptychs plus PSNR/LPIPS
for step 0 and step 400. Full teacher training remains gated on positive
memorized-train PSNR and LPIPS gains plus valid visuals. Only after that gate
passes may full teacher training and then few-step/single-step distillation
begin. Decoder fine-tuning remains conditional on a measured latent/decoder
ceiling after distillation.

Protected-data state: sealed internal `false`; official FoundIR test `false`.
