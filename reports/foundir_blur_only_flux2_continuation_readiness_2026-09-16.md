# FLUX.2 Blur-only step-10k→12k continuation readiness

Date: 2026-09-16 UTC

## Decision

Proceed with one 8×H100/H200 instance. Restore the verified step-10,000 LoRA
and AdamW optimizer exactly, train cumulatively to step 12,000, then evaluate
all 2,595 source-group-disjoint Blur development records. Do not repeat the
already successful 16-image overfit. Do not continue to step 14,000 until the
step-12,000 result is compared with step-10,000 (`25.734236 / 0.145406`).

## Frozen contract

- model: FLUX.2 Klein Base 4B, DiT LoRA rank/alpha 32/32, 50 modules;
- parent checkpoint: `step_010000.safetensors`, SHA-256
  `e0c834d88fc69ba6916032ef7763cdf84047a60d2407eeef23eae8992359c439`;
- exact parent AdamW state: `step_010000.optimizer.pt`;
- data: 105,057 train records / 7,339 train source groups; 2,595 development
  records / 165 development source groups; zero source-group overlap;
- objective and conditioning unchanged: pure noise-to-clean rectified flow
  with native LQ reference;
- cumulative data position starts at `10000 × 32 = 320000`;
- global batch 32 on both H100 (`2 × 2 × 8`) and H200 (`4 × 1 × 8`);
- linear continuation LR: 5e-6 at 10k, 3e-6 at 12k, 1e-6 at 14k;
- immutable candidates: 11k, 12k, 13k, 14k; first stop is 12k;
- forbidden: Qwen degradation conditioning, mixed degradation, pixel refiner,
  development-GT optimization, sealed internal, and official FoundIR test.

## Verification

- focused tests: 15 passed;
- Python compilation: passed;
- shell syntax: passed;
- advisor governance: hold resolved by the recorded user authorization;
- full preflight: passed, including parent/model hashes, optimizer metadata,
  path/cardinality checks, fixed visual set, and protected-data flags;
- training config SHA-256:
  `813f6e59630d5d37834f6d8ea745020e10139194996f1da7a0a7ab252ed3a589`.

The runner is crash-resumable at step 11k/12k, refuses checkpoint overwrite,
produces fixed 16-image triptychs at each candidate, and automatically starts
the full step-12k development evaluation after the staged training summary is
verified.

## Launch

```bash
cd /inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang && \
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 && \
smoke/launch_foundir_blur_only_flux2_continuation_v1_8gpu.sh
```

Monitor `logs/foundir_blur_only_flux2_continuation_v1.log`. A valid terminal
run ends with `FOUNDIR_BLUR_FLUX2_CONTINUATION_STEP12_COMPLETE` and then stops
at the explicit 12k→14k evidence gate.

## Conditional step-14k stage

The step-14k path is implemented but deliberately separate. Its executable
gate requires the immutable step-12k full-development result to satisfy both:

- PSNR strictly exceeds the step-10k baseline `25.734236`;
- LPIPS remains at or below the SOW limit `0.18`.

It additionally requires the step-12k checkpoint, optimizer state, and all 16
fixed visual snapshots. If any condition fails, it refuses to start. If all
conditions pass, the second launcher resumes exactly from step 12k, saves 13k
and 14k checkpoints/snapshots, and evaluates all 2,595 records at step 14k:

```bash
cd /inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang && \
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 && \
smoke/launch_foundir_blur_only_flux2_continuation_step14_v1_8gpu.sh
```

The step-14k launcher is never invoked automatically by the first stage.
