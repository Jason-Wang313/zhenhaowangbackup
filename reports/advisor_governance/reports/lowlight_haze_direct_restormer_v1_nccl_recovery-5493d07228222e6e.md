# Advisor 阶段报告：lowlight_haze_direct_restormer_v1_nccl_recovery

> **DIAGNOSTIC ONLY / 非主线：本结果不是 FLUX.2 主线结果，不得替代或包装成 FLUX.2/SOW final result。**

- 结论：`failure`
- Exact model：`project_direct_restormer_deraining_initialized`
- 是否为 FLUX.2 主线结果：`false`
- Config：`configs/foundir_lowlight_haze_restormer_v1.json` (`d60d67ba2a285fa0cb0691ac04f9630e2dc3129f1bf95d7563934ce777235cd3`)
- Checkpoint：`not_applicable` (`not_applicable`)
- Evaluation split：`runtime_before_gate_completion`
- Sealed internal 是否已访问：`false`
- Official FoundIR test 是否已访问：`false`
- Evidence：`reports/advisor_governance/runtime_evidence/lowlight_haze_direct_restormer_v1_nccl_recovery-b75de61585b6e4c0.json` (`abb38571cd123843e4f639de001c9b5897f788243cc98a5ac5d9435b378a53c1`)

## Metrics 与 SOW 对照

| Task | PSNR / target | LPIPS / target | Checks |
|---|---:|---:|---:|
| n/a | n/a | n/a | informational gate |

## 结论与待决策项

- 原因/状态：workflow exit status 1; inspect logs/foundir_lowlight_haze_nccl_recovery.log
- 请求 advisor 决策：请判断应修复并恢复同一 FLUX.2 mainline，还是停止/修改路线。

在 advisor/user 的明确、记录化决定前，不改变 model family、architecture、split、evaluation target 或 acceptance protocol。
