# Advisor 阶段报告：zimage_repro_repair_v1

> **DIAGNOSTIC ONLY / 非主线：本结果不是 FLUX.2 主线结果，不得替代或包装成 FLUX.2/SOW final result。**

- 结论：`information`
- Exact model：`Z-Image repaired deterministic controls with immutable ordinary-fit reuse`
- 是否为 FLUX.2 主线结果：`false`
- Config：`configs/foundir_blur_only_zimage_repro_repair_v1.json` (`806f60d6d8a92363c5c9a5a29039e21b9cb895c70e8a1e1804c5219bd674b840`)
- Checkpoint：`not_applicable` (`not_applicable`)
- Evaluation split：`two_existing_training_crops_only`
- Sealed internal 是否已访问：`false`
- Official FoundIR test 是否已访问：`false`
- Evidence：`checkpoints/foundir_blur_only_zimage_repro_repair_v1/result.json` (`9d85b92bc4cfc25d7843379b17e7221875e6b87ee9303a85e976609ff11438d3`)

## Metrics 与 SOW 对照

| Task | PSNR / target | LPIPS / target | Checks |
|---|---:|---:|---:|
| n/a | n/a | n/a | informational gate |

## 结论与待决策项

- 原因/状态：Deterministic padding repair supplement finished; inspect CUDA equivalence gates and combined controls. No automatic quality pass.
- 请求 advisor 决策：Continue within the existing user-authorized two-round decision budget; no additional advisor round.

在 advisor/user 的明确、记录化决定前，不改变 model family、architecture、split、evaluation target 或 acceptance protocol。
