# Consolidated Z-Image reproducibility diagnostic (round 1 of 2)

DIAGNOSTIC ONLY / 非主线. User explicitly authorized execution after agreeing
to a consolidated diagnostic followed by one evidence-backed learnability gate.
No additional advisor round is required. This is not a successful teacher,
distilled model, development evaluation, or SOW result.

Exact user authorization for this batch: "sure, aiminng for the best quality,
don't afraid of extra work, execute now". This followed agreement to batch
independent controls and cap the next decision phase at two rounds; it does
not authorize protected-test access or silently changing the model family.

## Frozen matrix

Eight independent GPU workers: two existing training records/objectives x
two execution modes x two independent device replicates. Each worker runs
off_a -> on -> off_b on the SAME physical GPU, restoring exact FP32 endpoint
step-400 LoRA, fresh AdamW state and the same RNG seed before EVERY phase.
The only within-worker change is normalization/gradient/update instrumentation.
All phases use identical dense evaluation schedule, batch 1, LR 5e-5, clipping
1 and 200 steps. Total: 24 fits / 4,800 independent optimizer steps, not DDP.
No fit is warm-started from another fit. Base weights and decoder remain BF16
and frozen; teacher rejection makes mixed loss RGB + .1 latent + .05 LPIPS.

| Workers | Record / loss | Execution |
|---|---|---|
| 0,4 | 0042483 strong blur / RGB | ordinary FlashAttention 2 backward |
| 1,5 | 0091486 near-clean / mixed | ordinary FlashAttention 2 backward |
| 2,6 | 0042483 strong blur / RGB | deterministic execution bundle |
| 3,7 | 0091486 near-clean / mixed | deterministic execution bundle |

The deterministic bundle explicitly enables FlashAttention deterministic
backward, strict PyTorch deterministic algorithms and deterministic cuDNN;
cuDNN benchmarking and TF32 are disabled in BOTH modes. CUBLAS_WORKSPACE_CONFIG
is :4096:8 in BOTH modes before process startup. Record actual attention call
counts, hardware UUID, package versions and numerical flags. This contrasts
execution bundles, NOT a causal isolation of FlashAttention alone. Unsupported
deterministic operators produce a preserved branch failure, never silent
fallback, warn-only execution or a mislabeled deterministic result.

## Evidence and integrity

Record every pre-update loss/gradient norm and all 14 identical snapshots,
quantized PSNR/LPIPS, raw RGB/latent losses, clipping fractions and latent hashes.
Hash exact initial/first-updated/final trainable tensors and first full gradients
for every phase. These common checks also perturb timing, so the batch is a
controlled NEW reference, not guaranteed bitwise replay of either historical
run. Instrumentation is the only additional operation in on. Save final weights
for all 24 phases and 336 Input|Output|GT images. Hash all artifacts. Instrumented
phases record 49 layer traces; compare baseline activation scale as well as tail
gradients. A large pre-normalization activation is not by itself a failure.

Reset TorchMetrics state at phase boundaries and after each training update;
evaluation calls/schedule are identical across phases. Cache identical prompt,
LQ and GT tensors across same-worker phases and hash them. Frozen-module modes
are unchanged. No development GT optimization, Qwen degradation conditioning,
mixed degradation, sealed internal or official FoundIR test access.

Workers run concurrently and write separate persistent logs. The supervisor
waits for all branches and aggregates completed and failed arms into one report;
a numerical failure in one worker must not erase or stop independent controls.
Runtime results never overwrite existing evidence. Both H100 and H200 profiles
are checked; no GPU training occurs from the CPU coding shell.

## Predeclared interpretation / stopping rules

1. Different starting weights or cached tensors invalidate the comparison.
2. Compare off_a/off_b within the same GPU first, then off_a/on and cross-device
   replicates. Report exact tensor equality separately from .05 dB/.005 LPIPS
   trajectory tolerance. Report intermediate spikes, not only final recovery.
3. If deterministic same-GPU repeats and instrumentation pairs are exact but
   ordinary repeats diverge, a reproducible execution route exists. This does
   not prove every previous failure was due to FlashAttention or fix quality.
4. If deterministic controls still differ or cannot run, do not select a LR or
   scale training. Use first-gradient/first-update mismatch and module traces
   to isolate the remaining implementation/numerical issue. No automatic third
   sweep: report the specific unresolved cause and revise the implementation.
5. Even if reproducibility passes, compare both PSNR and LPIPS against input,
   inherited endpoint and training-GT latent oracle, and inspect fixed images.
   Near-clean preservation and strong-blur fidelity remain mandatory concerns.
6. Round 2 is ONE separately frozen correction/learnability gate on the existing
   16 train plus 16 source-disjoint development crops; it is not launched from
   this batch. Freeze its actual correction only after round-1 evidence. If
   it still cannot learn reliably, reconsider the training formulation instead
   of adding indefinite LR sweeps. Two rounds is a decision budget, not a
   promise of final restoration quality. Full teacher training, distillation
   and any later decoder intervention remain dependent stages of the open goal.
