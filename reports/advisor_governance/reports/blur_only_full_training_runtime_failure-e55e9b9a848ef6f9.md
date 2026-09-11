# Advisor 阶段报告：blur_only_full_training_runtime_failure

- 结论：`failure`
- Exact model：`FLUX.2 Klein Base 4B DiT LoRA rank32 isolated-Blur-only`
- 是否为 FLUX.2 主线结果：`true`
- Config：`configs/foundir_blur_only_flux2_full_v1.json` (`27b3a9d37d4501dff62ddef2cbf08d87bed4fc537d2104bb3b5fe5955a57b1e2`)
- Checkpoint：`not_applicable` (`not_applicable`)
- Evaluation split：`runtime_before_gate_completion`
- Sealed internal 是否已访问：`false`
- Official FoundIR test 是否已访问：`false`
- Evidence：`reports/advisor_governance/runtime_evidence/blur_only_full_training_runtime_failure-570a30bef20d33a6.json` (`9ffce1e41c53d4912f397b2ec3fac652a3752083495556a696e78dce2dd210d2`)

## Metrics 与 SOW 对照

| Task | PSNR / target | LPIPS / target | Checks |
|---|---:|---:|---:|
| n/a | n/a | n/a | informational gate |

## 结论与待决策项

- 原因/状态：workflow exit status 1; inspect /inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang/logs/foundir_blur_only_flux2_full_v1.log
- 请求 advisor 决策：请判断应修复并恢复同一 FLUX.2 mainline，还是停止/修改路线。

在 advisor/user 的明确、记录化决定前，不改变 model family、architecture、split、evaluation target 或 acceptance protocol。
