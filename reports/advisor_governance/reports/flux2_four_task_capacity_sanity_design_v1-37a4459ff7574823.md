# Advisor 阶段报告：flux2_four_task_capacity_sanity_design_v1

> **DIAGNOSTIC ONLY / 非主线：本结果不是 FLUX.2 主线结果，不得替代或包装成 FLUX.2/SOW final result。**

- 结论：`information`
- Exact model：`FLUX.2 Klein Base 4B four independent single-degradation LoRA arms`
- 是否为 FLUX.2 主线结果：`false`
- Config：`configs/foundir_four_task_flux2_capacity_sanity_v1.json` (`995eae82efd24306139b2d91680beff9a5fa82076c34ec7f1b8821e74ffcd21c`)
- Checkpoint：`not_applicable` (`not_applicable`)
- Evaluation split：`FoundIR training archive: fixed 16 train plus 16 source-group-disjoint validation per task`
- Sealed internal 是否已访问：`false`
- Official FoundIR test 是否已访问：`false`
- Evidence：`reports/foundir_four_task_flux2_capacity_sanity_authorization_v1.json` (`06fa6acf761cacc648039ec6b45e0447e19bdd93aff9bd10f3ea62e9477eb87c`)

## Metrics 与 SOW 对照

| Task | PSNR / target | LPIPS / target | Checks |
|---|---:|---:|---:|
| n/a | n/a | n/a | informational gate |

## 结论与待决策项

- 原因/状态：Phase-1 design frozen: independent Noise/Haze/Rain/Low-light fail-fast capacity gates before any balanced five-task joint baseline; no Qwen degradation predictions or mixed degradation.
- 请求 advisor 决策：是否按用户已明确授权的 1→5 顺序执行本 hash-bound phase-1 design，并在四个结果共同审计前禁止 joint/Qwen/mixed 自动后续？

在 advisor/user 的明确、记录化决定前，不改变 model family、architecture、split、evaluation target 或 acceptance protocol。
