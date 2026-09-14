# Advisor 阶段报告：zimage_blur_teacher_tiny_overfit

> **DIAGNOSTIC ONLY / 非主线：本结果不是 FLUX.2 主线结果，不得替代或包装成 FLUX.2/SOW final result。**

- 结论：`failure`
- Exact model：`Z-Image 6B undistilled 50-step Blur-only teacher with frozen LQ ControlNet and DiT LoRA rank32`
- 是否为 FLUX.2 主线结果：`false`
- Config：`configs/foundir_blur_only_zimage_teacher_overfit_v1.json` (`faba42a614a033b1b53a675555baec7962760ca8cce182083a219a71f59bf94c`)
- Checkpoint：`not_applicable` (`not_applicable`)
- Evaluation split：`fixed_16_training_pairs_plus_16_source_disjoint_visual_development_pairs`
- Sealed internal 是否已访问：`false`
- Official FoundIR test 是否已访问：`false`
- Evidence：`checkpoints/foundir_blur_only_zimage_teacher_overfit_v1/evaluation/result.json` (`5e9b8b4a715382bb4d8d5a3e233314da9c34b7b3d7e4c4031663160dbea0e651`)

## Metrics 与 SOW 对照

| Task | PSNR / target | LPIPS / target | Checks |
|---|---:|---:|---:|
| n/a | n/a | n/a | informational gate |

## 结论与待决策项

- 原因/状态：Z-Image teacher tiny-overfit automatic restoration gate failed; fixed visuals require inspection and the implementation/root cause remains uncertain.
- 请求 advisor 决策：Z-Image teacher tiny-overfit 未学会 Blur restoration。是否授权一个有界的 corrected tiny diagnostic，以隔离 ControlNet、scheduler/trajectory、CFG 或 prediction-sign 不兼容？

在 advisor/user 的明确、记录化决定前，不改变 model family、architecture、split、evaluation target 或 acceptance protocol。
