# Advisor failure report: Blur-only Z-Image teacher tiny-overfit

> **DIAGNOSTIC ONLY / 非主线：本结果不是五类型 FLUX.2 主线或 official
> FoundIR-test 结果，不得包装成 SOW final result。**

## Decision requested

The multi-step teacher tiny-overfit failed both its automatic restoration gate
and fixed-triptych visual gate. Full teacher training, few-step/single-step
distillation, and decoder tuning are stopped. Please decide whether to authorize
one bounded corrected tiny diagnostic that isolates the Turbo-ControlNet/Base
pairing, scheduler shift, CFG convention, and prediction trajectory.

## Exact experiment identity

- Advisor route: `换换Z Image吧。先训teacher，然后蒸馏一下，少步/单步。如果点还是不行，微调一下decoder。`
- Model: undistilled `Tongyi-MAI/Z-Image` 6B, 50-step inference, frozen
  `alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1` LQ condition, DiT LoRA
  rank/alpha 32/32.
- Config: `configs/foundir_blur_only_zimage_teacher_overfit_v1.json`, SHA-256
  `faba42a614a033b1b53a675555baec7962760ca8cce182083a219a71f59bf94c`.
- Final checkpoint:
  `checkpoints/foundir_blur_only_zimage_teacher_overfit_v1/step_000400.safetensors`,
  SHA-256
  `2505c79fd554e76297eaaaa6be0d6e5aabbd75a702a2c11666c19df87697177b`.
- Evaluation split: 16 memorized isolated-Blur training crops plus 16 frozen,
  source-group-disjoint isolated-Blur development crops; 512x512; fixed seeds;
  50 steps; CFG 4.
- Immutable result:
  `checkpoints/foundir_blur_only_zimage_teacher_overfit_v1/evaluation/result.json`,
  SHA-256
  `5e9b8b4a715382bb4d8d5a3e233314da9c34b7b3d7e4c4031663160dbea0e651`.
- Protected data: sealed internal `false`; official FoundIR test `false`.

## Result

Training completed 400/400 steps on 8x H100. Loss decreased from `1.770904`
at step 1 to `0.413092` at step 400, but this did not translate into restoration.

| Split / state | PSNR input -> output | LPIPS input -> output | Gate |
|---|---:|---:|---:|
| train / step 0 | 24.7232 -> 7.6775 | 0.3085 -> 0.8538 | fail |
| train / step 400 | 24.7232 -> 9.3654 | 0.3085 -> 0.8408 | fail |
| development / step 0 | 23.9045 -> 8.1023 | 0.3263 -> 0.8460 | fail |
| development / step 400 | 23.9045 -> 9.0932 | 0.3263 -> 0.8949 | fail |

Every row had 0/16 PSNR improvements and 0/16 LPIPS improvements. All 64
fixed `Input | Z-Image Output | Ground Truth` triptychs exist. Step 0 is
blue-grey noise; step 400 learns only coarse layout and has severe green/yellow
cast, content errors, low-frequency blocks, and texture collapse. The visual
gate is therefore `failure`.

For the required SOW comparison only, the source-disjoint step-400 diagnostic
values are PSNR `9.0932 < 25.9` (fail) and LPIPS `0.8949 > 0.18` (fail).
Because this is a selected FoundIR-training development subset and the official
test remains untouched, these are diagnostic comparisons, not contractual
FoundIR Blur scores.

## Root-cause status

Root cause remains uncertain. The read-only audit confirmed three concrete
incompatibility risks:

1. The frozen ControlNet was trained for Turbo but is attached to undistilled
   Base solely on tensor-shape compatibility.
2. The official Base scheduler config specifies static shift 6, while the
   current DiffSynth path used its default static shift 3.
3. Native and DiffSynth CFG formulas differ at scale 4.

Prediction sign and timestep scaling match after tracing both implementations,
so there is no evidence yet for a simple sign inversion. Step-0 collapse proves
that the failure precedes LoRA fine-tuning; step-400 coarse geometry proves only
that some condition information arrives. A new controlled diagnostic is needed
to isolate the candidates.

## Governance

- Active hold: `zimage_blur_teacher_tiny_overfit-a6becc1eff058fec`.
- Machine report:
  `reports/advisor_governance/reports/zimage_blur_teacher_tiny_overfit-a6becc1eff058fec.md`,
  SHA-256
  `6cb809a52f0411286f6da0be6849ce150b8997c994778b2562243fdc36ea05ff`.
- Until a matching advisor decision is recorded, allowed work is read-only
  audit and advisor-report preparation only. No new inference, training,
  distillation, decoder tuning, sealed-internal access, or official-test access
  is authorized.
