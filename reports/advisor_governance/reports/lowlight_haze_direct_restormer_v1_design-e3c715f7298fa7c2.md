# Advisor 阶段报告：lowlight_haze_direct_restormer_v1_design

> **DIAGNOSTIC ONLY / 非主线：本结果不是 FLUX.2 主线结果，不得替代或包装成 FLUX.2/SOW final result。**

- 结论：`information`
- Exact model：`project_direct_restormer_deraining_initialized`
- 是否为 FLUX.2 主线结果：`false`
- Config：`configs/foundir_lowlight_haze_restormer_v1.json` (`d60d67ba2a285fa0cb0691ac04f9630e2dc3129f1bf95d7563934ce777235cd3`)
- Checkpoint：`models/restormer-task-native/deraining.pth` (`225ed9acf80b3527d7080e5182918f642a23a78ef8593ad700b7e924bd660681`)
- Evaluation split：`training_only_source_disjoint_design`
- Sealed internal 是否已访问：`false`
- Official FoundIR test 是否已访问：`false`
- Evidence：`reports/advisor_governance/foundir_lowlight_haze_restormer_v1_design_evidence.json` (`d1f78596028292613aed759d937198a916a8bb9a5a4c73cff5d71714ee3e8a02`)

## Metrics 与 SOW 对照

| Task | PSNR / target | LPIPS / target | Checks |
|---|---:|---:|---:|
| n/a | n/a | n/a | informational gate |

## 结论与待决策项

- 原因/状态：Direct Restormer design frozen after stage attribution showed the second low-light stage is destructive for Low-light+Haze.
- 请求 advisor 决策：Approve one bounded project-owned direct Low-light+Haze Restormer run using no FoundIR weights?

在 advisor/user 的明确、记录化决定前，不改变 model family、architecture、split、evaluation target 或 acceptance protocol。
