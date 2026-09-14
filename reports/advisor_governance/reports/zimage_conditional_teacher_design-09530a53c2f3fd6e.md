# Advisor 阶段报告：zimage_conditional_teacher_design

> **DIAGNOSTIC ONLY / 非主线：本结果不是 FLUX.2 主线结果，不得替代或包装成 FLUX.2/SOW final result。**

- 结论：`information`
- Exact model：`Z-Image Base 6B, persistent aligned LQ projection, native conditional flow, fresh FP32 LoRA`
- 是否为 FLUX.2 主线结果：`false`
- Config：`configs/foundir_blur_only_zimage_conditional_teacher_v1.json` (`c8bbf5d7dc85c2436f554a7e01e38b9d44d3f554686e001566c248d278da94db`)
- Checkpoint：`not_applicable` (`not_applicable`)
- Evaluation split：`fixed_train_and_source_disjoint_development`
- Sealed internal 是否已访问：`false`
- Official FoundIR test 是否已访问：`false`
- Evidence：`configs/foundir_blur_only_zimage_conditional_teacher_v1.json` (`c8bbf5d7dc85c2436f554a7e01e38b9d44d3f554686e001566c248d278da94db`)

## Metrics 与 SOW 对照

| Task | PSNR / target | LPIPS / target | Checks |
|---|---:|---:|---:|
| n/a | n/a | n/a | informational gate |

## 结论与待决策项

- 原因/状态：User-authorized first-principles redesign; native noise semantics and persistent LQ conditioning; tiny learnability then diverse pilot. Design at reports/zimage_conditional_teacher_design_2026-09-14.md; no GPU result yet.
- 请求 advisor 决策：Bind existing advisor route and latest explicit user execute to this design; no new advisor request.

在 advisor/user 的明确、记录化决定前，不改变 model family、architecture、split、evaluation target 或 acceptance protocol。
