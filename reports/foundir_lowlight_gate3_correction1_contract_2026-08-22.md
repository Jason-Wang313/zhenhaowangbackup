# FoundIR-Lowlight Gate 3 structured-Qwen correction 1 contract

Date: 2026-08-22 UTC

## Decision

The initial causal interface failed, but correction 1 is scientifically
justified because the cached Qwen expected severity retains positive
target-exposure correlation on all three outer families while the oracle arm
proves tone-interface capacity. This correction does not add spatial
restoration capacity and does not run Qwen again.

The compact restorer and completed measured adapter are frozen. The measured
and oracle output files are reused byte-for-byte from the initial causal
matrix. Only a 12-parameter structured calibration is trainable: six bounded
RGB gain/gamma slopes, five monotone posterior-knot adjustments, and one
non-negative measured-posterior residual scale. Offset residuals are fixed to
zero.

## Structured interface

- Inputs are the five canonical Qwen severity posteriors, normalized expected
  severity, normalized entropy, and top-posterior margin.
- The generic 32-dimensional projected visual summary is excluded because the
  preceding attribution found it unstable under leave-family transfer.
- A strongly regularized input-statistics ridge predicts the five posteriors;
  their low-dimensional residual is used without giving the correction a new
  measured-image restoration branch.
- Expected severity remains explicitly monotone: gain cannot decrease and
  gamma cannot increase as severity rises at fixed measured condition,
  uncertainty, and support.
- Entropy, margin, and nearest-training-support distance can only shrink the
  correction.
- Correct and within-family group-shuffled Qwen conditions are trained jointly.
  The shuffled output is tethered to the frozen measured output, and paired
  reconstruction-ranking terms require the correct condition to beat both the
  shuffled and measured controls. This prevents satisfying the contrast by
  intentionally damaging the negative arm.

## Preflight evidence

- 14 focused correction/causal/compact tests pass.
- Contract coverage is 429 training plus 245 outer-development records.
- The posterior-support radius is `0.2289806783`; training uses leave-group-out
  support distances rather than self-neighbour distances.
- Median outer support shrink is `0.8775` on LOL-v2, `0.9881` on SDSD, and
  `0.7758` on SICE. Unsupported tails are conservatively attenuated.
- Raw cached expected-severity versus target-derived first-channel log-gain
  correlation remains `0.3033 / 0.7893 / 0.7609` on outer LOL-v2/SDSD/SICE.
- Protected FoundIR-Lowlight IDs 1101--1150 were not accessed.

## Frozen identities and execution

- Config: `configs/foundir_lowlight_gate3_correction1_v1.json`
- Config SHA-256:
  `24023ae3fcdc9be8d784656967b5cc5c138dccd42019a948699d44237d6325df`
- Parent Qwen cache SHA-256:
  `a9e006f03949ecc348506f78a94b722079cb3a517c53dede50cdd8b8fb75e506`
- Parent measured adapter SHA-256:
  `3927884f2f812ea34710f797463c4cae3d14ee923b8cf3fd0332ac0d6c046e79`

Launch on four H100/H200 GPUs:

```bash
cd /inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang && CUDA_VISIBLE_DEVICES=0,1,2,3 bash smoke/launch_foundir_lowlight_gate3_correction1.sh
```

The same per-family causal thresholds remain frozen. If correction 1 fails,
the next action is residual attribution, not automatic correction 2 or Gate 4.
Correction 2 is used only if that attribution identifies a narrow remaining
interface defect with credible headroom.
