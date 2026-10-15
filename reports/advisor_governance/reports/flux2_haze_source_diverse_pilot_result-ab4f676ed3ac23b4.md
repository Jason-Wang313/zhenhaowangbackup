# Advisor 阶段报告：flux2_haze_source_diverse_pilot_result

- 结论：`failure`
- Exact model：`FLUX.2 Klein Base 4B fresh rank32 LoRA isolated-Haze source-diverse pilot`
- 是否为 FLUX.2 主线结果：`true`
- Config：`configs/foundir_haze_flux2_source_diverse_pilot_v1.json` (`1fde72e12149b8097fdd533bf73734b89a3c6ffb7278c5d483b2ced0f87d596d`)
- Checkpoint：`checkpoints/foundir_haze_flux2_source_diverse_pilot_v1/step_002000.safetensors` (`945b49c61fb2842bcd8fb96b4ea4a53c4c935f07729bfd5c3f17568060d780f0`)
- Evaluation split：`1580 isolated-Haze training source groups and frozen 128 source-group-disjoint validation records`
- Sealed internal 是否已访问：`false`
- Official FoundIR test 是否已访问：`false`
- Evidence：`checkpoints/foundir_haze_flux2_source_diverse_pilot_v1/result.json` (`c972fdeb43962b28644308737783729c5d74e22a5d4f1302229fedb5dba17d27`)

## Metrics 与 SOW 对照

| Task | PSNR / target | LPIPS / target | Checks |
|---|---:|---:|---:|
| n/a | n/a | n/a | informational gate |

## 结论与待决策项

- 原因/状态：The pilot completed normally. Haze validation reached 21.969477 PSNR / 0.111869 LPIPS: PSNR narrowly missed the 22.1 floor by 0.130523 dB (0.5906%), while LPIPS passed with 41.1218% safety. The fixed curve improved through step 2000; exact checkpoint and optimizer continuation is the bounded next action.
- 请求 advisor 决策：是否允许保持相同 1,580 sources、模型、objective 和 optimizer，从 exact step-2000 只续训 1,000 steps 到 step-3000，再做冻结 Haze validation？

在 advisor/user 的明确、记录化决定前，不改变 model family、architecture、split、evaluation target 或 acceptance protocol。
