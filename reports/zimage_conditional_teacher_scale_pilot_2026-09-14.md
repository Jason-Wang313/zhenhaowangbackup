# User-authorized conditional teacher scale pilot

DIAGNOSTIC ONLY / 非主线. Not an official FoundIR/SOW acceptance result.

## Decision

After receiving the failed tiny metrics and visual findings, the user explicitly
requested: “最新训练曲线仍在改善(if so, why don't we just proceed, and don't
ovethink it, scale”. Both sampled train and development curves improved from
step400 to1200. That supports testing more diverse training; it does not prove
eventual convergence. No further advisor question is needed.

This is a prospective exception to tiny-to-pilot admission, NOT a retroactive
metric or visual pass. Preserve the tiny config/result/checkpoints and terminal
audit unchanged. The default pilot path still requires passing tiny review;
only the explicit `--scale-pilot` path admits the exact user-authorized run.

## Fixed experiment

- Same Z-Image Base 6B, rank32 FP32 LoRA + persistent aligned LQ projection.
- Initialize EXACT tiny step1200 weights; fresh FP32 AdamW and 100-step warmup.
- Same Gaussian-noise-to-GT flow, uniform sigma, LR5e-5 / condition LR1e-4,
  clip1, frozen original VAE/native fixed-prompt text encoder.
- 2048 training sources / 128 source-disjoint development sources, one pinned
  512px crop per source. Original16 train +16 development examples retained.
- 2000 additional optimizer steps, global batch32, eight homogeneous H100/H200,
  microbatch1/rank and accumulation4. This is 64,000 additional presentations
  (31.25 passes over the selected training sources), not full-data training.
- Paired D4 transforms before VAE encoding; source split and manifest unchanged.
- 50-step shift6 CFG1 inference at pilot steps0/1000/2000. Each snapshot saves
  all16 fixed training and128 development triptychs and PSNR/LPIPS sidecars.
- Evaluate predeclared final step2000. Existing pilot gate remains improvement
  over input in BOTH development PSNR and LPIPS, plus actual visual review.
  Training snapshots on16 examples do not represent all2048 training sources.
- No automatic full-training, distillation or decoder launch. No Qwen degradation
  conditioning, mixtures, sealed/internal/official-test access or development-GT
  optimization. No architecture or learning-rate sweep.

## Identity and operation

Config: `configs/foundir_blur_only_zimage_conditional_teacher_v1.json`, SHA256
`c8bbf5d7dc85c2436f554a7e01e38b9d44d3f554686e001566c248d278da94db`.
Authorization: `reports/zimage_conditional_teacher_scale_authorization_v1.json`,
SHA256 `21c6cc9b8f20e37fde16fad5799e4fa701118c4730ba5defc4fca6d1e6365460`.
Initial checkpoint SHA256:
`52a61cd9412f494c2141d5f34ffd9cbb2dc66911755e008f2828a63071ea93a7`.
Pilot manifest SHA256:
`19309079bd2a64e253527ff1b637fc3bd2a17522cde8c47fbd3a60283315e11f`.

The trainer checks authorization content hash, config/result/checkpoint identity,
failed tiny status and exact pilot budget before CUDA setup. Runtime and final
result embed admission provenance; reported quality is evaluated independently.
It rehearses/reset fresh Base, then loads the exact tiny checkpoint before
step0 evaluation and fresh optimizer creation. This preserves the established
CUDA equivalence checks without initializing formal pilot training from Base.

GPU launch, only after CPU preflight passes and prior output absence is checked:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 smoke/launch_zimage_conditional_teacher_8gpu.sh --scale-pilot
```

Log: `logs/foundir_blur_only_zimage_conditional_teacher_v1_pilot.log`.
Output: `checkpoints/foundir_blur_only_zimage_conditional_teacher_v1_pilot/`.
Expected:2000 training rows,2 checkpoints,6 aggregates,432 triptychs/sidecars,
eight CUDA rehearsal artifacts, result and terminal marker. No passing tiny
visual-review artifact is created. CPU checks are not a GPU quality result.

## Readiness verification

The exact launcher `--scale-pilot --check` passed with exit0:36 focused tests,
compilation/shell syntax, real tiny checkpoint/authorization binding, all six
model artifact hashes, both H100/H200 inventory profiles, and all2176 pilot
image pairs. Log:`logs/zimage_conditional_teacher_scale_preflight_v1.log`.
Executable paths verified; no previous pilot output/log/startup log exists.
The coding shell has no exposed GPU; launch remains the user's GPU-shell action.
