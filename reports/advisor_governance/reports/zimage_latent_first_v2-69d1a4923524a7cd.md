# Advisor 阶段报告：zimage_latent_first_v2

> **DIAGNOSTIC ONLY / 非主线：本结果不是 FLUX.2 主线结果，不得替代或包装成 FLUX.2/SOW final result。**

- 结论：`failure`
- Exact model：`Z-Image latent-first supervised endpoint repair v2, original frozen VAE`
- 是否为 FLUX.2 主线结果：`false`
- Config：`configs/foundir_blur_only_zimage_latent_first_v2.json` (`6c6cfc24febcaae8d94b2ffbac188d6f09abff67421fd38a4b08d3723edc884d`)
- Checkpoint：`checkpoints/foundir_blur_only_zimage_latent_first_v2/step_001200.safetensors` (`b90f5782e1ffa5db3c78ed53295472a5b3aac307d423ac03c067c3a00e23db92`)
- Evaluation split：`fixed_16_train_16_source_disjoint_development_crops`
- Sealed internal 是否已访问：`false`
- Official FoundIR test 是否已访问：`false`
- Evidence：`checkpoints/foundir_blur_only_zimage_latent_first_v2/result.json` (`f2924d157f9fe9c2814122ec96f1e1cb5fa543016df0b61cdbfa84f303b8800f`)

## Metrics 与 SOW 对照

| Task | PSNR / target | LPIPS / target | Checks |
|---|---:|---:|---:|
| blur | 21.4293 / >= 25.90 | 0.46346 / <= 0.18 | 0/2 |

## 结论与待决策项

- 原因/状态：Bounded round-2 recipe finished after exact strict baseline replay. Review both metrics and visuals; no successful teacher distillation or automatic full-data admission.
- 请求 advisor 决策：Close the agreed two-round decision budget after visual review; no additional advisor request was required for this authorized recovery.

在 advisor/user 的明确、记录化决定前，不改变 model family、architecture、split、evaluation target 或 acceptance protocol。
