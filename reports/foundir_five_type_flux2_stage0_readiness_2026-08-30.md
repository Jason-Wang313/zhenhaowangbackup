# FoundIR five-type FLUX.2 stage-0 readiness — 2026-08-30

## Scope

This gate uses only official FoundIR **training** categories exactly expressible
by the frozen five-type Qwen schema: blur, noise, rain, haze, low-light, and
their admitted double/triple combinations. JPEG, Raindrop, and NightRain are
excluded. Protected FoundIR test pixels are not accessed.

## Implemented

- strict 685,495-pair category contract and official metadata hash;
- full LQ/GT readability, size, SHA-256, and GT pHash inspection;
- exact/perceptual GT-component grouped 96/2/2 split;
- frozen Joint Qwen2.5-VL condition export and canonical restoration text;
- balanced 44-target FLUX VAE ceiling over all 11 admitted categories;
- FLUX.2 native reference-image path with rank/alpha-32 image LoRA;
- degraded-anchored residual flow from `LQ latent + 0.05 noise` to GT latent;
- deterministic six-step inference and generic/Qwen/shuffled/type-oracle arms;
- exactly eight-GPU DDP on H100 80GB or H200, telemetry, duplicate-run lock,
  persistent logging, immutable outputs, and checkpoint resume;
- metadata-only preflight, compilation, shell syntax, and ten focused tests.

## Current external blocker

The official approximately 4.85 TB FoundIR training archive is absent from
`data/foundir/MillionIRData/Train`. The similarly named files under
`downloads/foundir_official/extracted_lq` and `extracted_gt` are protected test
data and are deliberately rejected as a substitute. The official repository
requires the author request form for the training-set download link.

No GPU command should be launched until the authorized archive is extracted at
the exact training root. The checked launcher fails before detaching if the
root is absent.

## Runtime estimate after data is present

- full 4.85 TB manifest/hash inspection: storage-dependent, roughly 1–6 hours;
- 128-image Joint-Qwen condition export: roughly 5–15 minutes on eight GPUs;
- balanced VAE ceiling: roughly 5–15 minutes;
- 200-step FLUX overfit plus four-arm evaluation: roughly 1–3 hours;
- total first gate: roughly 2–9 hours, dominated by storage hashing and FLUX.

Later pilot/ablation/full training remains a 1.5–4 day GPU program and is
authorized only after this stage-0 result is inspected.
