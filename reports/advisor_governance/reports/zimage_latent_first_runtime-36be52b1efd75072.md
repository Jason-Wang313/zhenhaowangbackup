# Advisor 阶段报告：zimage_latent_first_runtime

> **DIAGNOSTIC ONLY / 非主线：本结果不是 FLUX.2 主线结果，不得替代或包装成 FLUX.2/SOW final result。**

- 结论：`failure`
- Exact model：`Z-Image latent-first supervised endpoint repair`
- 是否为 FLUX.2 主线结果：`false`
- Config：`configs/foundir_blur_only_zimage_latent_first_v1.json` (`cc4cb5e26803fa20a4ea92fc91ae2b6185619685b1e04e0a071e908aa89da904`)
- Checkpoint：`not_applicable` (`not_applicable`)
- Evaluation split：`runtime_before_gate_completion`
- Sealed internal 是否已访问：`false`
- Official FoundIR test 是否已访问：`false`
- Evidence：`reports/advisor_governance/runtime_evidence/zimage_latent_first_runtime-2fe5180322be6e36.json` (`3064081492a0ec593f36356dc24e61f4388a45bd933764926d5941dd5c6f93e7`)

## Metrics 与 SOW 对照

| Task | PSNR / target | LPIPS / target | Checks |
|---|---:|---:|---:|
| n/a | n/a | n/a | informational gate |

## 结论与待决策项

- 原因/状态：workflow exit status 1; inspect logs/foundir_blur_only_zimage_latent_first_v1.log
- 请求 advisor 决策：请判断应修复并恢复同一 FLUX.2 mainline，还是停止/修改路线。

在 advisor/user 的明确、记录化决定前，不改变 model family、architecture、split、evaluation target 或 acceptance protocol。
