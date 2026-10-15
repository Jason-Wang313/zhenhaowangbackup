# Advisor 阶段报告：haze_source_diverse_pilot_runtime

- 结论：`failure`
- Exact model：`FLUX.2 Klein Base 4B fresh rank32 LoRA isolated-Haze source-diverse pilot`
- 是否为 FLUX.2 主线结果：`true`
- Config：`configs/foundir_haze_flux2_source_diverse_pilot_v1.json` (`1fde72e12149b8097fdd533bf73734b89a3c6ffb7278c5d483b2ced0f87d596d`)
- Checkpoint：`not_applicable` (`not_applicable`)
- Evaluation split：`runtime_before_gate_completion`
- Sealed internal 是否已访问：`false`
- Official FoundIR test 是否已访问：`false`
- Evidence：`reports/advisor_governance/runtime_evidence/haze_source_diverse_pilot_runtime-cab0c4f19e919ba7.json` (`cf3efa379bf5bba782cd092e291a6f423bfcccde6526f135d97248e5a520f0d3`)

## Metrics 与 SOW 对照

| Task | PSNR / target | LPIPS / target | Checks |
|---|---:|---:|---:|
| n/a | n/a | n/a | informational gate |

## 结论与待决策项

- 原因/状态：workflow exit status 1; inspect /inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang/logs/foundir_haze_flux2_source_diverse_pilot_v1.log
- 请求 advisor 决策：请判断应修复并恢复同一 FLUX.2 mainline，还是停止/修改路线。

在 advisor/user 的明确、记录化决定前，不改变 model family、architecture、split、evaluation target 或 acceptance protocol。
