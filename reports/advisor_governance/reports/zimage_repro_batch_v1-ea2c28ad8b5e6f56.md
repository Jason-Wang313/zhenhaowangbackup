# Advisor 阶段报告：zimage_repro_batch_v1

> **DIAGNOSTIC ONLY / 非主线：本结果不是 FLUX.2 主线结果，不得替代或包装成 FLUX.2/SOW final result。**

- 结论：`information`
- Exact model：`Z-Image matched off/on/off ordinary/deterministic controls`
- 是否为 FLUX.2 主线结果：`false`
- Config：`configs/foundir_blur_only_zimage_repro_batch_v1.json` (`9aef9ebef30487e8657a6df506e0914ef55cd77765c37e39ff420211a95c7046`)
- Checkpoint：`not_applicable` (`not_applicable`)
- Evaluation split：`two_existing_training_crops_only`
- Sealed internal 是否已访问：`false`
- Official FoundIR test 是否已访问：`false`
- Evidence：`checkpoints/foundir_blur_only_zimage_repro_batch_v1/result.json` (`97c3d855f447d3e24667401466cc1bff985efe1ad4304f954ac171e78474ca23`)

## Metrics 与 SOW 对照

| Task | PSNR / target | LPIPS / target | Checks |
|---|---:|---:|---:|
| n/a | n/a | n/a | informational gate |

## 结论与待决策项

- 原因/状态：Consolidated batch finished; inspect complete and failed controls, numerical equality, trajectories and fixed visuals. No automatic quality pass.
- 请求 advisor 决策：Continue within the existing user-authorized two-round decision budget; no additional advisor round.

在 advisor/user 的明确、记录化决定前，不改变 model family、architecture、split、evaluation target 或 acceptance protocol。
