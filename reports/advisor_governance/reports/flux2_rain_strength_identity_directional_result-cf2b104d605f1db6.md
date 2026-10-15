# Advisor 阶段报告：flux2_rain_strength_identity_directional_result

- 结论：`failure`
- Exact model：`FLUX.2 Klein Base 4B exact Rain step-2000 rank32 LoRA strength/identity directional gate`
- 是否为 FLUX.2 主线结果：`true`
- Config：`configs/foundir_rain_flux2_strength_identity_directional_gate_v1.json` (`fa9e5fe097be5e18b46de2f904d94374771dd6fa3a686c5f3bf624459e2c8fc2`)
- Checkpoint：`checkpoints/foundir_rain_flux2_strength_identity_directional_gate_v1/step_002250.safetensors` (`fe2885d690eabdd4bf86865b7866c40a0e86302397ef821d1ae37a7f56a2b127`)
- Evaluation split：`exact 1580 Rain parent training crops and new source-disjoint train-split development: 16 heavy, 16 moderate, 16 near-clean`
- Sealed internal 是否已访问：`false`
- Official FoundIR test 是否已访问：`false`
- Evidence：`checkpoints/foundir_rain_flux2_strength_identity_directional_gate_v1/result.json` (`81199c746840e9363c3f7846068e3871b070a513337effe43e76e61413b6b6b9`)

## Metrics 与 SOW 对照

| Task | PSNR / target | LPIPS / target | Checks |
|---|---:|---:|---:|
| n/a | n/a | n/a | informational gate |

## 结论与待决策项

- 原因/状态：The bounded 250-step Rain strength/identity directional gate completed. It tests whether explicit near-clean identity supervision fixes overprocessing without sacrificing heavy/moderate restoration. There is no automatic continuation.
- 请求 advisor 决策：请检查五项 parent-vs-candidate directional checks 与固定视觉；是否允许后续更长 continuation，或停止该方向？

在 advisor/user 的明确、记录化决定前，不改变 model family、architecture、split、evaluation target 或 acceptance protocol。
