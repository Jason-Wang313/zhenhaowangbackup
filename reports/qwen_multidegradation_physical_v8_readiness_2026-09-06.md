# Qwen multi-degradation physical-v8 readiness — 2026-09-06

## Scope and reason

The frozen FoundIR-training-only diagnostic showed that the v7 per-type Qwen
adapter improved Blur recall from 28.26% to 63.04% and Noise recall from 0% to
54.35%, but type-set exact match remained 49.22%. Code-path, LQ/GT direction,
crop, label mapping, JSON parsing, and aggregation audits found no elementary
execution defect. The remaining mismatch was data morphology: v7 represented
Blur with one straight-motion operator and Noise with additive RGB Gaussian
noise, while real capture includes defocus, irregular shake, temporal-average
blur, signal-dependent noise, read noise, chroma effects, and banding.

## Bounded v8 correction

- Blur is exactly balanced over defocus, camera-shake, and temporal-average
  families, independently at L2--L5.
- Noise is exactly balanced over Poisson/read, heteroscedastic ISP, and
  chroma/banding families, independently at L2--L5.
- JPEG and resampling artifacts are bounded non-target nuisances. They are
  rotated between complete 12-record family-by-level blocks so they cannot
  reveal a target family or severity. Clean-only records remain literal clean
  crops.
- The Qwen decoder remains the decision maker. No classification head was
  added. A binary auxiliary loss uses the decoder's own L1 logit as the absent
  score and log-sum-exp of its L2--L5 logits as the present score.
- FoundIR contributes no Qwen training image and no Qwen training label.
- The v7 source partition is preserved exactly, so cumulative v7/v8 training
  sources remain disjoint from v8 validation sources.

## Frozen data evidence

- Data config SHA-256:
  `57a26adc873ed8c48ef54df661ae10845c288f5c993e45d012de026dd1d313e5`
- Manifest SHA-256:
  `777d0130bb0ff9590ad967e8bc848b13399de71df58bf72739f95be62720fb02`
- Generation result SHA-256:
  `3e7506151671d0ce207834a98d8e5a209c1864f82af829e8b6a0ba16835b157e`
- Records: 100,000 train, 10,000 validation, 10,000 protected test.
- Generation: 120,000/120,000 passed in 270.56 seconds with 64 CPU workers;
  733 preview files were hash-reused and 119,267 files were newly generated.
- Full image/evidence hashing passed for all 120,000 records.
- Protected test inference performed: false.
- FoundIR used for Qwen gradients: false.

## Training contract

- Training config SHA-256:
  `eef9fe05e21a2fd5b92543f17473e96cb1723b1d4f800ef13a2f6e1622168aeb`.
- Initialize the passing v7 per-type rank-64 adapter with a fresh optimizer.
- Visual and language LoRA rank/alpha: 64/64; train the visual merger.
- Ask Blur, Noise, Rain, Haze, and Low-light separately; decoder answers L1--L5.
- Replay all 100,000 training images and all five questions once: 500,000
  dialogue presentations, global batch 200, 2,500 optimizer steps.
- Use five deterministic, disjoint 20,000-source shards. Each shard saves a
  checkpoint after 500 steps, preserving one-epoch semantics while making an
  interrupted 8-GPU run resumable.
- Select and gate on synthetic validation only. Do not access the protected
  synthetic test during model selection.
- Hardware contract: exactly eight NVIDIA H100 or H200 GPUs, each accepted by
  the existing minimum-memory check.

## Verification

Focused unit tests cover legacy behavior, deterministic and monotonic physical
families, family/level balancing, nuisance semantics, source-shard coverage,
decoder presence-loss math and gradients, config drift rejection, and existing
trainer/preflight behavior. Python compilation, shell syntax, config parsing,
and `git diff --check` passed. The GPU launcher is detached with persistent
training and GPU telemetry logs and normalizes a scientific gate failure to
status 2 rather than reporting it as a crash.
