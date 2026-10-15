# Advisor 阶段报告：foundir_compositional_ebt_lower_order_scale_v2_design

> **DIAGNOSTIC ONLY / 非主线：本结果不是 FLUX.2 主线结果，不得替代或包装成 FLUX.2/SOW final result。**

- 结论：`information`
- Exact model：`matched Plain versus atomic/pairwise Compositional EBT 10k steps-only scale test`
- 是否为 FLUX.2 主线结果：`false`
- Config：`configs/foundir_compositional_ebt_lower_order_scale_v2.json` (`7d307614e00fdaa9ec6cb0f26bcc3ea2124543708ed03327eb2f409a201c44bd`)
- Checkpoint：`not_applicable` (`not_applicable`)
- Evaluation split：`60 singleton/double train plus 30 source-disjoint singleton/double development; no triple`
- Sealed internal 是否已访问：`false`
- Official FoundIR test 是否已访问：`false`
- Evidence：`reports/advisor_governance/foundir_compositional_ebt_lower_order_scale_v2_design_evidence.json` (`d2c1e05b1631c0161057e18c07ea4827d6a785514a8318e997687e53ded3542c`)

## Metrics 与 SOW 对照

| Task | PSNR / target | LPIPS / target | Checks |
|---|---:|---:|---:|
| n/a | n/a | n/a | informational gate |

## 结论与待决策项

- 原因/状态：v1 learned small positive development corrections but 300 steps were insufficient; v2 changes only optimizer steps to 10000
- 请求 advisor 决策：是否批准两条 matched arms 各自从同一 seed 重启至 10000 steps，并在 lower-order development gate 后停止？

在 advisor/user 的明确、记录化决定前，不改变 model family、architecture、split、evaluation target 或 acceptance protocol。
