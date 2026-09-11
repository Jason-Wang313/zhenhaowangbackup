# Advisor 阶段报告：blur_only_full_training_runtime_failure

- 结论：`failure`
- Exact model：`FLUX.2 Klein Base 4B DiT LoRA rank32 isolated-Blur-only`
- 是否为 FLUX.2 主线结果：`true`
- Config：`configs/foundir_blur_only_flux2_full_v1.json` (`01b00df153f3dd9279f63c039857d0989ca88f331c32b1e02e9c734d895bd0ae`)
- Checkpoint：`not_applicable` (`not_applicable`)
- Evaluation split：`runtime_before_gate_completion`
- Sealed internal 是否已访问：`false`
- Official FoundIR test 是否已访问：`false`
- Evidence：`reports/advisor_governance/runtime_evidence/blur_only_full_training_runtime_failure-2c5a0acf8b5757d5.json` (`cc3e57976c60697ba3ef7aa72d74fc023672ab093fc54f3047cd7ff993d9429f`)

## Metrics 与 SOW 对照

| Task | PSNR / target | LPIPS / target | Checks |
|---|---:|---:|---:|
| n/a | n/a | n/a | informational gate |

## 结论与待决策项

- 原因/状态：workflow exit status 1; inspect /inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang/logs/foundir_blur_only_flux2_full_v1.log
- 请求 advisor 决策：请判断应修复并恢复同一 FLUX.2 mainline，还是停止/修改路线。

在 advisor/user 的明确、记录化决定前，不改变 model family、architecture、split、evaluation target 或 acceptance protocol。
