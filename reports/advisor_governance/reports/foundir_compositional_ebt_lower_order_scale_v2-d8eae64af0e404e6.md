# Advisor 阶段报告：foundir_compositional_ebt_lower_order_scale_v2

> **DIAGNOSTIC ONLY / 非主线：本结果不是 FLUX.2 主线结果，不得替代或包装成 FLUX.2/SOW final result。**

- 结论：`pass`
- Exact model：`matched Plain versus atomic/pairwise Compositional EBT 10k steps-only scale`
- 是否为 FLUX.2 主线结果：`false`
- Config：`configs/foundir_compositional_ebt_lower_order_scale_v2.json` (`7d307614e00fdaa9ec6cb0f26bcc3ea2124543708ed03327eb2f409a201c44bd`)
- Checkpoint：`not_applicable` (`not_applicable`)
- Evaluation split：`source-disjoint singleton/double development only`
- Sealed internal 是否已访问：`false`
- Official FoundIR test 是否已访问：`false`
- Evidence：`checkpoints/foundir_compositional_ebt_lower_order_scale_v2/scale10k_v2/result.json` (`9a4e73d261d9bc03b7996133588f90f80e1c130a4c1e4f5f5ecf83a2e61dd3c0`)

## Metrics 与 SOW 对照

| Task | PSNR / target | LPIPS / target | Checks |
|---|---:|---:|---:|
| n/a | n/a | n/a | informational gate |

## 结论与待决策项

- 原因/状态：lower_order_learnability_pass after matched 10000-step arms; held-out triples untouched
- 请求 advisor 决策：请审阅 10k step-only result，并决定停止、扩大 model/data，或另行批准后续；不得自动进入 100k/triple。

在 advisor/user 的明确、记录化决定前，不改变 model family、architecture、split、evaluation target 或 acceptance protocol。
