# Advisor 阶段报告：foundir_compositional_ebt_phase0_v1_design

> **DIAGNOSTIC ONLY / 非主线：本结果不是 FLUX.2 主线结果，不得替代或包装成 FLUX.2/SOW final result。**

- 结论：`information`
- Exact model：`VLM-activated atomic/pairwise compositional Energy-Based Transformer (Phase-0 only)`
- 是否为 FLUX.2 主线结果：`false`
- Config：`configs/foundir_compositional_ebt_phase0_v1.json` (`b79d3cfec14f55a26f8563e9488fa8f6ed57c1f55ac15467ea5642caabaf0cea`)
- Checkpoint：`not_applicable` (`not_applicable`)
- Evaluation split：`none_design_and_zero_update_smoke_only`
- Sealed internal 是否已访问：`false`
- Official FoundIR test 是否已访问：`false`
- Evidence：`reports/advisor_governance/foundir_compositional_ebt_phase0_v1_design_evidence.json` (`298e3030203e0b11bf2f05eadb205a770f539fafb38183fb811f968a7f138182`)

## Metrics 与 SOW 对照

| Task | PSNR / target | LPIPS / target | Checks |
|---|---:|---:|---:|
| n/a | n/a | n/a | informational gate |

## 结论与待决策项

- 原因/状态：Lower-order FLUX.2/Restormer success coexists with a 16.736791 dB unseen-triple deficit; correctly routed stage evidence isolates inter-expert composition failure. Phase-0 tests whether the official EBT formulation can be instantiated safely before any training.
- 请求 advisor 决策：是否批准仅执行 pinned official EBT audit、minimal implementation、CPU tests 与 zero-update CUDA smoke，并继续禁止训练、triple endpoint 与 official FoundIR test access？

在 advisor/user 的明确、记录化决定前，不改变 model family、architecture、split、evaluation target 或 acceptance protocol。
