# Latent-first v2 startup recovery design

DIAGNOSTIC ONLY / 非主线. The v1 launch ended before optimizer step 1 because
two development ranks crossed its historical per-record metric tolerance.
The exact failed values were not persisted. This is a startup-evidence defect,
not a demonstrated training or image-quality failure. All v1 artifacts remain
immutable. The user explicitly said `execute`; this records continuation under
the already authorized isolated-Blur Z-Image route, not a new advisor reply.

## What remains frozen

The scientific recipe is unchanged: exact FP32 endpoint step-400 LoRA,
original frozen VAE/native text encoder, fixed prompt and same immutable 16
training plus 16 source-disjoint reusable development crops. Global batch 8,
LR 5e-5, AdamW weight decay 0 and gradient clip 1. Steps 1--800 use GT latent
MSE only; 801--1200 use latent MSE 1 + RGB MSE 1 + LPIPS .05. Candidate step
1200 and snapshots 0/200/400/600/800/1000/1200 remain predeclared. There is no
teacher imitation, decoder tuning, development-target optimization, Qwen
degradation condition, mixed degradation, sealed internal or official test.

Config: `configs/foundir_blur_only_zimage_latent_first_v2.json`, SHA-256
`6c6cfc24febcaae8d94b2ffbac188d6f09abff67421fd38a4b08d3723edc884d`.
New output root: `checkpoints/foundir_blur_only_zimage_latent_first_v2`.
The failed v1 log/runtime/hold and original recipe config are hash-pinned in
v2; v1 output is never deleted, overwritten or resumed in place.

## Complete baseline replay gate

After the same full CUDA two-stage distributed rehearsal and exact initial
weight reset, run the entire 16+16 baseline twice before optimization. Persist
all first-repeat triptychs/sidecars and both complete in-memory row sets. For
every record require exact equality across current repeats of predicted latent,
decoded RGB, quantized output image, input/output PSNR and LPIPS, latent/RGB
MSE and clipped fraction. Require the same initial parameter digest across all
ranks and equality to the post-rehearsal restored checkpoint digest.

Historical endpoint values are ordinary-execution reference evidence, not the
definition of the new strict-deterministic output. Report every per-record
delta and aggregate delta. Before observing the missing v1 rows, v2 freezes
the following maximum absolute drift limits:

- per-record: .05 dB PSNR and .005 LPIPS;
- each role aggregate: .02 dB PSNR and .002 LPIPS.

These are the existing reproducibility audit metric tolerances, combined with
the stricter prior aggregate limits. They do not replace exact current replay.
Any identity/repeat failure or drift beyond any limit writes complete baseline
evidence and stops before step 1. Passing means the exact immutable checkpoint
has a reproducible strict-execution baseline with bounded historical numerical
drift; it does not prove the old execution's unique drift cause or image
quality. Training step 0 uses this current baseline, never retrospectively
changes the acceptance gate, and does not compare final quality to a cherry-
picked historical value.

## Decision boundary

If startup passes, the same job continues the one already agreed round-2
quality correction. Automatic final admission still requires lower training
latent MSE and mean PSNR/LPIPS gains over input on both roles. Human review of
fixed triptychs remains mandatory, and no automatic full-data launch follows.
If startup fails, preserve its complete evidence and diagnose that exact
failure. If training finishes but quality fails, close the two-round decision
budget and reconsider the formulation rather than extending an LR/schedule
sweep. This recovery does not claim successful teacher distillation.

## CPU readiness

Sixty-one focused tests pass, including exact replay checks, rejection of a
single changed tensor digest, rejection of per-record historical drift beyond
the frozen bound, both loss phases and two-process global-batch gradient/update
equivalence. Full model/parent/v1-evidence hashes, both H100/H200 inventory
profiles, Python compilation, launcher syntax and governance resolution pass.

The real 6B 64x64 CPU rehearsal again passes both stages and two exact repeats;
its restored initial digest and both update/gradient digests match the prior
CPU evidence exactly. Evidence:

- `logs/zimage_latent_first_v2_tests.log`;
- `logs/zimage_latent_first_v2_preflight.log`;
- `logs/zimage_latent_first_v2_cpu_smoke.log`;
- `logs/zimage_latent_first_cpu_i1_du9il/rehearsal_rank_0.json`, SHA-256
  `c315b5feb4e33d078d0bcc67f6ce254cff37733c0a721a47c87a93f0a0b5bda9`.

This shell has no CUDA. CPU readiness does not establish full-resolution CUDA
baseline agreement or image quality; those are guarded inside the one v2 GPU
job.
