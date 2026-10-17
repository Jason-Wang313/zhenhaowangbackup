# Complete energy candidate oracle v1 readiness — 2026-10-17

## Purpose

This no-training diagnostic tests whether the strong step-8,530 residual basis
now supplies a candidate pool capable of supporting the complete advisor-PPT
action-energy method. It is a GT-selecting upper bound and cannot be reported
as deployable model performance.

The prior seven-alpha local selector was stopped by the user. Its last logged
step is `3,910`; its last complete checkpoint/validation is step `3,412`.
Direct tmux termination left no normal launcher END marker. It must not resume.

## Candidate contract

Deployable candidates are:

1. unchanged proposal;
2. fidelity residual-basis output;
3. perceptual residual-basis output;
4. half-strength fidelity;
5. 1.25-strength fidelity;
6. 75/25 fidelity-perceptual blend;
7. 50/50 fidelity-perceptual blend.

The original real-denoising output, deraining wrong expert, blind Gaussian
denoiser detail-damaged output, and repeated-proposal over-restoration are
measured separately as contrastive negatives. They are never selectable
deployment actions in the oracle.

For 32x32 and 64x64 non-overlapping regions, GT selects the minimum-MSE
deployable candidate. The audit reports resulting whole-image mean PSNR/LPIPS,
candidate win shares, and winner stability after horizontally flipping the
input, rerunning all deployable models, and aligning the decision grids back.

## Frozen feasibility criteria

A patch protocol passes only if all hold:

- oracle PSNR is at least `+.08 dB` above frozen fidelity;
- oracle LPIPS is at most `.045362067760033824`;
- at least two candidates win at least 1% of regions;
- flip-aligned winner stability is at least `.45` for 32x32 or `.55` for
  64x64.

Passing only establishes that complete energy training is worth attempting.
It does not prove an inference-time selector can recover the oracle.

## Qualification completed

- JSON parse: pass.
- Python compilation: pass.
- Bash syntax: pass.
- Focused unit tests: 2/2 pass.
- Candidate names/order and blend arithmetic: pass.
- Patch oracle selects different candidates in different regions: pass.
- Diff whitespace check: pass.
- Output path is fresh: pass.
- Residual checkpoint remains hash-bound to
  `9ebc6b73e627e94f35ef52f511696215285c7efb9f800936d0b3d873b8aac95a`.
- Official, sealed, internal, parent-test and frozen-660 data remain excluded.

The Codex container has no GPU. Run the prepared launcher on one homogeneous
8xH100/H200 node only after the stopped seven-alpha job has released all GPUs.
Estimated cost is `.08–.25` wall-clock hours and `.64–2.0` allocated
GPU-hours; replace this estimate with measured runtime.
