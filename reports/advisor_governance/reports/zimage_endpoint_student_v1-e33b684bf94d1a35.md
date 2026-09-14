# Advisor 阶段报告：zimage_endpoint_student_v1

> **DIAGNOSTIC ONLY / 非主线：本结果不是 FLUX.2 主线结果，不得替代或包装成 FLUX.2/SOW final result。**

- 结论：`pass`
- Exact model：`Z-Image Base 6B GT-anchored single-step student FP32 LoRA frozen VAE`
- 是否为 FLUX.2 主线结果：`false`
- Config：`configs/foundir_blur_only_zimage_endpoint_distill_v1.json` (`1d98854ec7dc40e3d678879d849955180d652cbfcd86ec77596912214c956f50`)
- Checkpoint：`checkpoints/foundir_blur_only_zimage_endpoint_distill_v1/step_000400.safetensors` (`d979af50f3013cc5e02d4c36943f32c4b512a75ee26c0aab20a37728b9c15d7e`)
- Evaluation split：`fixed_16_training_pairs_plus_16_source_disjoint_visual_development_pairs`
- Sealed internal 是否已访问：`false`
- Official FoundIR test 是否已访问：`false`
- Evidence：`checkpoints/foundir_blur_only_zimage_endpoint_distill_v1/result.json` (`b5934dc34bd4ac67ba030de67504af3fc0e517f66e650dcfc4e6ad9677adc680`)

## Metrics 与 SOW 对照

| Task | PSNR / target | LPIPS / target | Checks |
|---|---:|---:|---:|
| n/a | n/a | n/a | informational gate |

## 结论与待决策项

- 原因/状态：Bounded endpoint student finished; inspect fixed visuals and both metrics before scaling. Teacher acceptance filter uses only training pairs.
- 请求 advisor 决策：Continue within existing user authorization; no new advisor round requested.

在 advisor/user 的明确、记录化决定前，不改变 model family、architecture、split、evaluation target 或 acceptance protocol。
