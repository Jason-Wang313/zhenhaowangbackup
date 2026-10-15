# Advisor 阶段报告：foundir_compositional_ebt_lower_order_pilot_v1

> **DIAGNOSTIC ONLY / 非主线：本结果不是 FLUX.2 主线结果，不得替代或包装成 FLUX.2/SOW final result。**

- 结论：`failure`
- Exact model：`matched Plain versus atomic/pairwise Compositional Energy-Based Transformer`
- 是否为 FLUX.2 主线结果：`false`
- Config：`configs/foundir_compositional_ebt_lower_order_pilot_v1.json` (`a69e48b35fd1e58b03a864e8d1da15bfc5fdd1e3a8f5401c43b5ae67f6844d87`)
- Checkpoint：`not_applicable` (`not_applicable`)
- Evaluation split：`source-disjoint singleton/double development only`
- Sealed internal 是否已访问：`false`
- Official FoundIR test 是否已访问：`false`
- Evidence：`checkpoints/foundir_compositional_ebt_lower_order_pilot_v1/pilot_v1/result.json` (`0bfcc30805387e22b8ede15e91e1aa13b6c3d8335d924695d0d99d746ce4b78f`)

## Metrics 与 SOW 对照

| Task | PSNR / target | LPIPS / target | Checks |
|---|---:|---:|---:|
| n/a | n/a | n/a | informational gate |

## 结论与待决策项

- 原因/状态：lower_order_learnability_fail; held-out triples remained untouched
- 请求 advisor 决策：请审阅 lower-order learnability gate；只有 pass 后才考虑另行批准一次 frozen held-out-triple evaluation。

在 advisor/user 的明确、记录化决定前，不改变 model family、architecture、split、evaluation target 或 acceptance protocol。
