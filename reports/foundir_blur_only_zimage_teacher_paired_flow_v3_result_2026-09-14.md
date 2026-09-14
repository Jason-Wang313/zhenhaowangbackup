# Blur-only Z-Image paired-flow teacher v3 result

> **DIAGNOSTIC ONLY / 非主线。** This is not the five-type FLUX.2 mainline or
> an official FoundIR/SOW final result.

## Terminal status

- Result: scientific teacher-gate failure; runtime completed normally.
- Hardware: homogeneous 8x NVIDIA H100 80GB HBM3, BF16,
  `flash_attention_2`.
- Completion: 1200 cumulative optimizer steps at
  `2026-09-14T05:37:31Z`; v3 added 800 steps in `798.6471 s`.
- Scope: isolated Blur only; no Qwen degradation conditioning, mixed
  degradation, sealed internal, or official FoundIR test access.

## Exact identities

- Config:
  `configs/foundir_blur_only_zimage_teacher_paired_flow_continuation_v3.json`
  (`6909e7fcf2397fb4b911870076a74544b1f111edb3e0a895309490e804eaf438`).
- Parent v2 step-400 LoRA:
  `checkpoints/foundir_blur_only_zimage_teacher_paired_flow_overfit_v2/step_000400.safetensors`
  (`2a3e9b3392dfea20f9e6e869febce1ff99bcaa751e317e247c908f683a4ecd39`).
- Step-800 LoRA:
  `checkpoints/foundir_blur_only_zimage_teacher_paired_flow_continuation_v3/step_000800.safetensors`
  (`c37cc2211b7c80a610af6f2b1547f1aaa2055d852ea71f10aae77c753a5bbcb5`).
- Step-1200 LoRA:
  `checkpoints/foundir_blur_only_zimage_teacher_paired_flow_continuation_v3/step_001200.safetensors`
  (`e583e4871d8c498da655e4350e4931662f9310bc197d3d089f9c4241e6625323`).
- Training result:
  `checkpoints/foundir_blur_only_zimage_teacher_paired_flow_continuation_v3/training_result.json`
  (`17def3137a126e57b3acd79c6d2767d250a2b35d15ceecaf65ca59787be3fd34`).
- Evaluation result:
  `checkpoints/foundir_blur_only_zimage_teacher_paired_flow_continuation_v3/evaluation/result.json`
  (`393cc6427aa0b3d8aaf70db8dc90ca750e67388e6e9a2de475751dfebd414d12`).

## Protocol

V3 starts from the immutable v2 step-400 LoRA, resets the optimizer, uses no
ControlNet, and trains on only the exact deployed 50-step static-shift-6 grid
with plain unweighted velocity MSE. Inference starts from the exact LQ VAE
latent, uses 50 Euler steps, CFG 1, and the frozen Z-Image Base VAE/text
encoder. Evaluation uses the same fixed 16 training crops plus 16
source-group-disjoint development crops.

## PSNR/LPIPS curve

| Split | Step | Input PSNR | Output PSNR | Input LPIPS | Output LPIPS |
|---|---:|---:|---:|---:|---:|
| train | 400 | 24.7232 | 23.2828 | 0.3085 | 0.2415 |
| train | 800 | 24.7232 | 23.4878 | 0.3085 | 0.2223 |
| train | 1200 | 24.7232 | 23.5666 | 0.3085 | 0.2007 |
| development | 400 | 23.9045 | 22.6414 | 0.3263 | 0.2692 |
| development | 800 | 23.9045 | 22.5367 | 0.3263 | 0.2640 |
| development | 1200 | 23.9045 | 21.9002 | 0.3263 | 0.2536 |

At step 1200, train PSNR is `-1.1566 dB` versus input while LPIPS improves by
`0.1078`. Development PSNR is `-2.0043 dB` versus input while LPIPS improves
by `0.0727`. The automatic metric gate is false.

## Fixed visual review

The v1 content/noise collapse remains repaired: content and geometry are
preserved. However, v3 adds fine grain or web-like texture, overprocesses
near-clean inputs, and remains far from GT on strong blur. Near-clean samples
are harmed disproportionately, while a small number of lower-quality inputs
gain PSNR. Continuing the same loss increasingly optimizes perceptual
sharpness at the expense of pixel fidelity and development generalization.

## Authorized next action

Hold `zimage_blur_teacher_paired_flow_tiny_overfit_v3-b6567614820dbc4e`
(`616244388727264984cb45e3abd287d1946e546c1a9be07f7513985f97a67898`)
is resolved by the recorded user-after-advisor `resume_same_mainline`
decision. No additional advisor question is required. First measure the frozen
VAE ceiling and `[1,2,4,8,16,50]`-step solver curve; then proceed directly to
few-step/single-step distillation. If quality remains inadequate or the
VAE/latent ceiling is limiting, proceed to decoder tuning. The failed teacher
gate is evidence, not a permanent authorization deadlock.
