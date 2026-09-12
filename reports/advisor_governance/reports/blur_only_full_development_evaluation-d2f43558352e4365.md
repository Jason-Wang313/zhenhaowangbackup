# Advisor 阶段报告：blur_only_full_development_evaluation

> **DIAGNOSTIC ONLY / 非主线：本结果不是 FLUX.2 主线结果，不得替代或包装成 FLUX.2/SOW final result。**

- 结论：`information`
- Exact model：`FLUX.2 Klein Base 4B isolated-Blur DiT LoRA rank/alpha 32/32`
- 是否为 FLUX.2 主线结果：`false`
- Config：`configs/foundir_blur_only_flux2_full_development_v1.json` (`42556934cfed99660916c11de14fb09508b5d628bef5dec7c8a070d8cc39bac9`)
- Checkpoint：`checkpoints/foundir_blur_only_flux2_full_v1/step_010000.safetensors` (`e0c834d88fc69ba6916032ef7763cdf84047a60d2407eeef23eae8992359c439`)
- Evaluation split：`source_group_disjoint_blur_development_2595`
- Sealed internal 是否已访问：`false`
- Official FoundIR test 是否已访问：`false`
- Evidence：`checkpoints/foundir_blur_only_flux2_full_v1/full_development_evaluation_v1/result.json` (`e08491f2def20de010231fa59b138efd965675bc00706362db61e55ddcaec464`)

## Metrics 与 SOW 对照

| Task | PSNR / target | LPIPS / target | Checks |
|---|---:|---:|---:|
| blur | 25.7342 / >= 25.90 | 0.14541 / <= 0.18 | 1/2 |

## 结论与待决策项

- 原因/状态：Five-checkpoint full Blur development curve completed; use the curve and fixed visual outputs to judge overfitting.
- 请求 advisor 决策：请根据全量 curve 与视觉结果判断：继续调整 Blur-only，还是先审计 hallucination/training-inference code？

在 advisor/user 的明确、记录化决定前，不改变 model family、architecture、split、evaluation target 或 acceptance protocol。
