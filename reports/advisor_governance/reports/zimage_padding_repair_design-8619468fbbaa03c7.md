# Advisor 阶段报告：zimage_padding_repair_design

> **DIAGNOSTIC ONLY / 非主线：本结果不是 FLUX.2 主线结果，不得替代或包装成 FLUX.2/SOW final result。**

- 结论：`information`
- Exact model：`Z-Image deterministic padding compatibility repair; unchanged restoration model`
- 是否为 FLUX.2 主线结果：`false`
- Config：`configs/foundir_blur_only_zimage_repro_repair_v1.json` (`806f60d6d8a92363c5c9a5a29039e21b9cb895c70e8a1e1804c5219bd674b840`)
- Checkpoint：`not_applicable` (`not_applicable`)
- Evaluation split：`two_existing_training_crops_only`
- Sealed internal 是否已访问：`false`
- Official FoundIR test 是否已访问：`false`
- Evidence：`configs/foundir_blur_only_zimage_repro_repair_v1.json` (`806f60d6d8a92363c5c9a5a29039e21b9cb895c70e8a1e1804c5219bd674b840`)

## Metrics 与 SOW 对照

| Task | PSNR / target | LPIPS / target | Checks |
|---|---:|---:|---:|
| n/a | n/a | n/a | informational gate |

## 结论与待决策项

- 原因/状态：User explicitly authorized execution after deterministic CUDA indexing failure. Replace only two frozen-token padding assignments, require CUDA operator/full-model equivalence gates, reuse 12 completed ordinary fits and supplement only failed controls. See reports/zimage_padding_repair_design_2026-09-14.md.
- 请求 advisor 决策：Execute the existing authorized round-1 runtime repair; no new advisor request or quality-pass claim.

在 advisor/user 的明确、记录化决定前，不改变 model family、architecture、split、evaluation target 或 acceptance protocol。
