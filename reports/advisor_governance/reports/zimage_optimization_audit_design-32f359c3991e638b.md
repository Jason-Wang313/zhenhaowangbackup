# Advisor 阶段报告：zimage_optimization_audit_design

> **DIAGNOSTIC ONLY / 非主线：本结果不是 FLUX.2 主线结果，不得替代或包装成 FLUX.2/SOW final result。**

- 结论：`information`
- Exact model：`Z-Image endpoint optimization diagnostic`
- 是否为 FLUX.2 主线结果：`false`
- Config：`configs/foundir_blur_only_zimage_decoder_tail_v1.json` (`d6f29a5ee41d60a02651a1e8a775551e5b913ebdd0f943fadaf033876df01984`)
- Checkpoint：`checkpoints/foundir_blur_only_zimage_decoder_tail_v1/step_000400.safetensors` (`b755b0834b802c85f8f2c6bb9719158c68ce1be0107a5dc8cda280cd26574ce2`)
- Evaluation split：`two_existing_training_crops_only`
- Sealed internal 是否已访问：`false`
- Official FoundIR test 是否已访问：`false`
- Evidence：`checkpoints/foundir_blur_only_zimage_decoder_tail_v1/result.json` (`cf6d7f0e6db7b72432fed03b76e1446f6d9bbbd771604a8b80a2cd65682ede46`)

## Metrics 与 SOW 对照

| Task | PSNR / target | LPIPS / target | Checks |
|---|---:|---:|---:|
| n/a | n/a | n/a | informational gate |

## 结论与待决策项

- 原因/状态：Root-cause audit and frozen four-arm design in reports/zimage_optimization_audit_design_2026-09-14.md.
- 请求 advisor 决策：Record existing user execute authorization for bounded Blur-only optimization audit; no new advisor round.

在 advisor/user 的明确、记录化决定前，不改变 model family、architecture、split、evaluation target 或 acceptance protocol。
