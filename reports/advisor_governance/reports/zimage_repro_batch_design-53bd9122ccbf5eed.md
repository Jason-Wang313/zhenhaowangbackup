# Advisor 阶段报告：zimage_repro_batch_design

> **DIAGNOSTIC ONLY / 非主线：本结果不是 FLUX.2 主线结果，不得替代或包装成 FLUX.2/SOW final result。**

- 结论：`information`
- Exact model：`Z-Image consolidated matched reproducibility controls; unchanged backbone and losses`
- 是否为 FLUX.2 主线结果：`false`
- Config：`configs/foundir_blur_only_zimage_repro_batch_v1.json` (`9aef9ebef30487e8657a6df506e0914ef55cd77765c37e39ff420211a95c7046`)
- Checkpoint：`not_applicable` (`not_applicable`)
- Evaluation split：`two_existing_training_crops_only`
- Sealed internal 是否已访问：`false`
- Official FoundIR test 是否已访问：`false`
- Evidence：`configs/foundir_blur_only_zimage_repro_batch_v1.json` (`9aef9ebef30487e8657a6df506e0914ef55cd77765c37e39ff420211a95c7046`)

## Metrics 与 SOW 对照

| Task | PSNR / target | LPIPS / target | Checks |
|---|---:|---:|---:|
| n/a | n/a | n/a | informational gate |

## 结论与待决策项

- 原因/状态：User explicitly authorized the consolidated two-round decision plan. Round 1 is 24 matched off/on/off fits; see reports/zimage_repro_batch_design_2026-09-14.md. CPU readiness is not GPU quality evidence.
- 请求 advisor 决策：Execute the already-authorized batch and inspect its numerical controls before freezing round 2; no further advisor request.

在 advisor/user 的明确、记录化决定前，不改变 model family、architecture、split、evaluation target 或 acceptance protocol。
