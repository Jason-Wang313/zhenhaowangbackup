# Advisor 阶段报告：flux2_blur_source_diverse_spatial_anchor_design_v1

> **DIAGNOSTIC ONLY / 非主线：本结果不是 FLUX.2 主线结果，不得替代或包装成 FLUX.2/SOW final result。**

- 结论：`information`
- Exact model：`FLUX.2 Klein Base 4B isolated-Blur step-10000 parent paired source-diverse flow-control/spatial-anchor pilot`
- 是否为 FLUX.2 主线结果：`false`
- Config：`configs/foundir_blur_only_flux2_source_diverse_spatial_anchor_pilot_v1.json` (`ab5a41af64d6a8e39e228625a18b5b978ab616cc007308dcd2999242bb1a05be`)
- Checkpoint：`not_applicable` (`not_applicable`)
- Evaluation split：`512 isolated-Blur training source groups with one deterministic record/crop each; excludes old tiny train and fixed development groups; fixed 16 source-disjoint development crops only at final gate`
- Sealed internal 是否已访问：`false`
- Official FoundIR test 是否已访问：`false`
- Evidence：`reports/flux2_blur_source_diverse_spatial_anchor_authorization_v1.json` (`970325682a1b9dc788434e8cf960dd07bed83f337afa8641fa8d5c5bbcc8649f`)

## Metrics 与 SOW 对照

| Task | PSNR / target | LPIPS / target | Checks |
|---|---:|---:|---:|
| n/a | n/a | n/a | informational gate |

## 结论与待决策项

- 原因/状态：Frozen source-diverse intermediate pilot addresses spatial-correspondence generalization with conservative endpoint latent/multiscale/low-frequency anchors; no independent restoration branch and no automatic full scale.
- 请求 advisor 决策：是否批准按冻结config执行512-source paired pilot，并仅在预声明step-400 development与visual gate通过后再考虑full scale？

在 advisor/user 的明确、记录化决定前，不改变 model family、architecture、split、evaluation target 或 acceptance protocol。
