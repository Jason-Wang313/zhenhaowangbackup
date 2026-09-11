# FLUX.2 Blur-only code audit — 2026-09-11

Scope: advisor-authorized diagnostic only. No Qwen degradation prediction, mixed degradation,
pixel refiner, sealed internal split, official FoundIR test, or Z-Image is used.

## Confirmed findings

1. The old five-type tiny gate was not a Blur learnability gate. Its 32 evaluation images were
   aggregated across 11 isolated/mixed categories, so gains from other tasks could hide a Blur
   failure. The two isolated Blur examples in the historical gate changed PSNR from
   `19.78098 -> 19.38375 dB` and `19.88767 -> 20.03899 dB`; Blur was not demonstrated to learn.
2. The old v3 gate jointly trained DiT LoRA and a pixel refiner. It therefore could not establish
   that FLUX DiT LoRA itself learned restoration. The Blur-only sanity gate disables the refiner.
3. FLUX.2 Klein Base 4B is documented by the checked-in official implementation as a 50-step Base
   model. Official sampling uses `flux2.sampling.get_schedule()`, which applies a sequence-length-
   dependent time/SNR shift. The five-type evaluator instead used six equally spaced steps from
   `torch.linspace(1, 0, 7)`. This is a confirmed training/inference-contract mismatch.
4. Base inference officially uses an empty/prompt CFG pair. Guidance `1.0` is algebraically equal
   to the conditional branch used during LoRA training and is used here to avoid conflating the
   Blur learnability test with CFG strength. Checkpoint inference still goes through the official
   `denoise_cfg()` implementation.
5. FoundIR `01Blur` LQ/GT paths, dimensions, normalization, and aligned crop logic are consistent.
   All 105,057 train and 2,595 validation Blur pairs are at least 512 pixels on both sides. The
   existing source-group split has zero train/validation overlap.
6. FLUX.2 VAE `encode()` applies its own learned BatchNorm normalization and `decode()` applies the
   inverse. The project passes RGB in `[-1,1]` and does not add a second latent scale/shift. No VAE
   scale-direction bug was found.
7. The rectified-flow sign is internally consistent: training target is `source-clean`; reverse
   integration from `t=1` to `t=0` adds a negative time delta. No sign reversal was found.

## Remaining hypotheses, not yet proven bugs

- The custom degraded-anchored endpoint and six-step solver may have made optimization too hard or
  inaccurate for deblurring. The isolated test returns to the already proven, simpler
  pure-noise-to-clean reference-conditioned flow and official 50-step schedule.
- Rank-32 LoRA on the 50 existing attention/MLP targets may still lack adequate deblurring capacity.
  Only a Blur-only overfit can distinguish that from task interference.
- Uniform timestep sampling is project-defined because the official repository is inference-only.
  It is retained for the first controlled overfit and is not claimed to be an official recipe.

Decision rule: if the 16-pair DiT-LoRA-only test cannot materially improve both PSNR and LPIPS on
the same fixed Blur crops, stop at FLUX code/capacity diagnosis and report the fixed checkpoint
triptychs to the advisor. Only a passing automatic gate plus human visual review may authorize a
larger Blur-only training run.
