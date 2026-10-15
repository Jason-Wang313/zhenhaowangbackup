# Advisor 阶段报告：foundir_compositional_ebt_lower_order_pilot_v1_design

> **DIAGNOSTIC ONLY / 非主线：本结果不是 FLUX.2 主线结果，不得替代或包装成 FLUX.2/SOW final result。**

- 结论：`information`
- Exact model：`matched Plain versus atomic/pairwise Compositional Energy-Based Transformer lower-order pilot`
- 是否为 FLUX.2 主线结果：`false`
- Config：`configs/foundir_compositional_ebt_lower_order_pilot_v1.json` (`a69e48b35fd1e58b03a864e8d1da15bfc5fdd1e3a8f5401c43b5ae67f6844d87`)
- Checkpoint：`not_applicable` (`not_applicable`)
- Evaluation split：`60 singleton/double train parents plus 30 source-disjoint singleton/double development parents`
- Sealed internal 是否已访问：`false`
- Official FoundIR test 是否已访问：`false`
- Evidence：`reports/advisor_governance/foundir_compositional_ebt_lower_order_pilot_v1_design_evidence.json` (`d6cafbe5866db4196934feb0986fc6b0eb8d24d637c7d2808e2b395b2ea0429e`)

## Metrics 与 SOW 对照

| Task | PSNR / target | LPIPS / target | Checks |
|---|---:|---:|---:|
| n/a | n/a | n/a | informational gate |

## 结论与待决策项

- 原因/状态：Phase-0 passed; bounded optimizer-update learnability pilot is CPU-ready and held-out triples remain prohibited
- 请求 advisor 决策：是否批准严格按 frozen config 执行 300+300 optimizer-step lower-order learnability pilot，并在结果后停止？

在 advisor/user 的明确、记录化决定前，不改变 model family、architecture、split、evaluation target 或 acceptance protocol。
