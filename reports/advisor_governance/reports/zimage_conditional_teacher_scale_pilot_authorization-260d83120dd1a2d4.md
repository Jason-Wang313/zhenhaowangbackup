# Advisor 阶段报告：zimage_conditional_teacher_scale_pilot_authorization

> **DIAGNOSTIC ONLY / 非主线：本结果不是 FLUX.2 主线结果，不得替代或包装成 FLUX.2/SOW final result。**

- 结论：`information`
- Exact model：`Z-Image Base 6B conditional teacher, unchanged architecture`
- 是否为 FLUX.2 主线结果：`false`
- Config：`configs/foundir_blur_only_zimage_conditional_teacher_v1.json` (`c8bbf5d7dc85c2436f554a7e01e38b9d44d3f554686e001566c248d278da94db`)
- Checkpoint：`not_applicable` (`52a61cd9412f494c2141d5f34ffd9cbb2dc66911755e008f2828a63071ea93a7`)
- Evaluation split：`2048 training / 128 source-disjoint development, Blur training archive only`
- Sealed internal 是否已访问：`false`
- Official FoundIR test 是否已访问：`false`
- Evidence：`reports/zimage_conditional_teacher_scale_authorization_v1.json` (`21c6cc9b8f20e37fde16fad5799e4fa701118c4730ba5defc4fca6d1e6365460`)

## Metrics 与 SOW 对照

| Task | PSNR / target | LPIPS / target | Checks |
|---|---:|---:|---:|
| n/a | n/a | n/a | informational gate |

## 结论与待决策项

- 原因/状态：User explicitly requests scaling after disclosure of failed tiny metrics and visuals. Prospective admission exception for prepared pilot only; preserve failed tiny, unchanged pilot quality gate, exact step1200 initialization and fresh optimizer.
- 请求 advisor 决策：Record the latest explicit user scale instruction as authorization; no new advisor reply requested.

在 advisor/user 的明确、记录化决定前，不改变 model family、architecture、split、evaluation target 或 acceptance protocol。
