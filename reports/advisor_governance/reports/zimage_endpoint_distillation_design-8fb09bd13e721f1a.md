# Advisor 阶段报告：zimage_endpoint_distillation_design

> **DIAGNOSTIC ONLY / 非主线：本结果不是 FLUX.2 主线结果，不得替代或包装成 FLUX.2/SOW final result。**

- 结论：`information`
- Exact model：`Z-Image Base 6B GT-anchored single-step LoRA student with frozen VAE`
- 是否为 FLUX.2 主线结果：`false`
- Config：`configs/foundir_blur_only_zimage_endpoint_distill_v1.json` (`1d98854ec7dc40e3d678879d849955180d652cbfcd86ec77596912214c956f50`)
- Checkpoint：`checkpoints/foundir_blur_only_zimage_teacher_paired_flow_continuation_v3/step_001200.safetensors` (`e583e4871d8c498da655e4350e4931662f9310bc197d3d089f9c4241e6625323`)
- Evaluation split：`fixed_16_training_pairs_plus_16_source_disjoint_visual_development_pairs`
- Sealed internal 是否已访问：`false`
- Official FoundIR test 是否已访问：`false`
- Evidence：`reports/zimage_endpoint_distillation_proposal_2026-09-14.json` (`0e01bf25339f03d1a903d27122d22727797ec59325ce3728fe00bc92bb27f1a4`)

## Metrics 与 SOW 对照

| Task | PSNR / target | LPIPS / target | Checks |
|---|---:|---:|---:|
| n/a | n/a | n/a | informational gate |

## 结论与待决策项

- 原因/状态：Healthy GT VAE roundtrip; weak v3 teacher; BF16 optimization quantization and training/deployment trajectory mismatch identified as risks. Add FP32 LoRA optimization and endpoint RGB fidelity supervision.
- 请求 advisor 决策：Execute the already explicitly authorized single-step distillation/recovery under the user instruction to continue without another advisor request.

在 advisor/user 的明确、记录化决定前，不改变 model family、architecture、split、evaluation target 或 acceptance protocol。
