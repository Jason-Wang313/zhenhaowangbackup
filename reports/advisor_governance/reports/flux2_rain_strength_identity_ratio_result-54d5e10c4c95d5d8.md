# Advisor 阶段报告：flux2_rain_strength_identity_ratio_result

- 结论：`failure`
- Exact model：`FLUX.2 Klein Base 4B exact Rain step-2000 rank32 LoRA 3:2:1 ratio gate`
- 是否为 FLUX.2 主线结果：`true`
- Config：`configs/foundir_rain_flux2_strength_identity_ratio_gate_v1.json` (`00f0e17da13595a21c5ad1f054d08c073d5022aacbffb6c5c687cfc85b0e66d0`)
- Checkpoint：`checkpoints/foundir_rain_flux2_strength_identity_ratio_gate_v1/step_002250.safetensors` (`507a52f523f774f4de5a241dc7e78f57092085d1df76ae5cfa00c83c67a43cc0`)
- Evaluation split：`exact 1580 Rain parent training crops; frozen source-disjoint train-split development with 16 heavy, 16 moderate, 16 near-clean`
- Sealed internal 是否已访问：`false`
- Official FoundIR test 是否已访问：`false`
- Evidence：`checkpoints/foundir_rain_flux2_strength_identity_ratio_gate_v1/result.json` (`4514ae7d1129bc58c940600a7d519959009845fdbd2f3b5d1d694a049a32f534`)

## Metrics 与 SOW 对照

| Task | PSNR / target | LPIPS / target | Checks |
|---|---:|---:|---:|
| n/a | n/a | n/a | informational gate |

## 结论与待决策项

- 原因/状态：The one-change 3:2:1 Rain ratio gate completed; it decides whether reduced identity exposure retains near-clean gains while recovering heavy PSNR and LPIPS. There is no automatic continuation.
- 请求 advisor 决策：请检查五项 directional checks 与固定视觉；是否允许正式 Rain validation，或停止并重新诊断？

在 advisor/user 的明确、记录化决定前，不改变 model family、architecture、split、evaluation target 或 acceptance protocol。
