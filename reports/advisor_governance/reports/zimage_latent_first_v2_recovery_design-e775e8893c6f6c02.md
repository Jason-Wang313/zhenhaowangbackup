# Advisor 阶段报告：zimage_latent_first_v2_recovery_design

> **DIAGNOSTIC ONLY / 非主线：本结果不是 FLUX.2 主线结果，不得替代或包装成 FLUX.2/SOW final result。**

- 结论：`information`
- Exact model：`Z-Image Base latent-first supervised endpoint repair v2; exact endpoint step400 FP32 LoRA; original frozen VAE`
- 是否为 FLUX.2 主线结果：`false`
- Config：`configs/foundir_blur_only_zimage_latent_first_v2.json` (`6c6cfc24febcaae8d94b2ffbac188d6f09abff67421fd38a4b08d3723edc884d`)
- Checkpoint：`not_applicable` (`not_applicable`)
- Evaluation split：`fixed_16_train_16_source_disjoint_development_crops`
- Sealed internal 是否已访问：`false`
- Official FoundIR test 是否已访问：`false`
- Evidence：`configs/foundir_blur_only_zimage_latent_first_v2.json` (`6c6cfc24febcaae8d94b2ffbac188d6f09abff67421fd38a4b08d3723edc884d`)

## Metrics 与 SOW 对照

| Task | PSNR / target | LPIPS / target | Checks |
|---|---:|---:|---:|
| n/a | n/a | n/a | informational gate |

## 结论与待决策项

- 原因/状态：v1 stopped at step0 because incomplete historical baseline mismatch evidence. v2 freezes exact two-repeat strict baseline plus bounded fully reported historical drift before unchanged 1200-step recipe.
- 请求 advisor 决策：No new advisor response requested: user explicitly said execute under existing Blur-only Z-Image authority; preserve v1 and execute bounded recovery.

在 advisor/user 的明确、记录化决定前，不改变 model family、architecture、split、evaluation target 或 acceptance protocol。
