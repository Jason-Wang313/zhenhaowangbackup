# Advisor 阶段报告：zimage_decoder_tail_design

> **DIAGNOSTIC ONLY / 非主线：本结果不是 FLUX.2 主线结果，不得替代或包装成 FLUX.2/SOW final result。**

- 结论：`information`
- Exact model：`Z-Image frozen endpoint student with trainable decoder tail`
- 是否为 FLUX.2 主线结果：`false`
- Config：`configs/foundir_blur_only_zimage_endpoint_distill_v1.json` (`1d98854ec7dc40e3d678879d849955180d652cbfcd86ec77596912214c956f50`)
- Checkpoint：`checkpoints/foundir_blur_only_zimage_endpoint_distill_v1/step_000400.safetensors` (`d979af50f3013cc5e02d4c36943f32c4b512a75ee26c0aab20a37728b9c15d7e`)
- Evaluation split：`fixed_16_train_16_development`
- Sealed internal 是否已访问：`false`
- Official FoundIR test 是否已访问：`false`
- Evidence：`checkpoints/foundir_blur_only_zimage_endpoint_distill_v1/result.json` (`b5934dc34bd4ac67ba030de67504af3fc0e517f66e650dcfc4e6ad9677adc680`)

## Metrics 与 SOW 对照

| Task | PSNR / target | LPIPS / target | Checks |
|---|---:|---:|---:|
| n/a | n/a | n/a | informational gate |

## 结论与待决策项

- 原因/状态：Design: reports/zimage_decoder_tail_design_2026-09-14.md. Endpoint quality remains inadequate; bounded decoder tail experiment under existing user authorization.
- 请求 advisor 决策：Record existing explicit user authorization for conditional decoder tuning; no new advisor round.

在 advisor/user 的明确、记录化决定前，不改变 model family、architecture、split、evaluation target 或 acceptance protocol。
