# Advisor 阶段报告：cure_two_track_reproduction_v1_design

> **DIAGNOSTIC ONLY / 非主线：本结果不是 FLUX.2 主线结果，不得替代或包装成 FLUX.2/SOW final result。**

- 结论：`information`
- Exact model：`official_CURE_OneRestore`
- 是否为 FLUX.2 主线结果：`false`
- Config：`configs/cure_two_track_reproduction_v1.json` (`01a428b7eced044cc26e5a6a939010d23eade49627d6e776d19938acef823119`)
- Checkpoint：`not_applicable` (`not_applicable`)
- Evaluation split：`CCDD11_train_development_only`
- Sealed internal 是否已访问：`false`
- Official FoundIR test 是否已访问：`false`
- Evidence：`reports/advisor_governance/cure_two_track_reproduction_v1_design_evidence.json` (`8ccc493059b012fe9f54bf1a32c5d0fd1960dd8d0392b519e984a832d47b9335`)

## Metrics 与 SOW 对照

| Task | PSNR / target | LPIPS / target | Checks |
|---|---:|---:|---:|
| n/a | n/a | n/a | informational gate |

## 结论与待决策项

- 原因/状态：R0 output collapse; CURE original-code exposure-control reproduction proposed
- 请求 advisor 决策：批准两条匹配的 CURE reproduction；仅 CCDD-11，triple 只用于冻结评价，不接触 FoundIR sealed test？

在 advisor/user 的明确、记录化决定前，不改变 model family、architecture、split、evaluation target 或 acceptance protocol。
