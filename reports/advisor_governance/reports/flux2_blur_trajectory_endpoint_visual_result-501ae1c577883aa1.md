# Advisor 阶段报告：flux2_blur_trajectory_endpoint_visual_result

> **DIAGNOSTIC ONLY / 非主线：本结果不是 FLUX.2 主线结果，不得替代或包装成 FLUX.2/SOW final result。**

- 结论：`failure`
- Exact model：`FLUX.2 Klein Base 4B isolated-Blur step-10000 parent trajectory-endpoint tiny gate`
- 是否为 FLUX.2 主线结果：`false`
- Config：`configs/foundir_blur_only_flux2_trajectory_endpoint_gate_v1.json` (`d1824e219a5f46629e6d8cc3e964e5b676512e2bb8037c41c5bf88a01f9f7b23`)
- Checkpoint：`checkpoints/foundir_blur_only_flux2_trajectory_endpoint_gate_v1/arms/trajectory_endpoint/step_000200.safetensors` (`67a0dc81fc7361de2b5f00a7abcfd0b97ab872e73e60a8662f637a4de5278abc`)
- Evaluation split：`16 frozen training crops plus 16 source-group-disjoint descriptive development crops; visual review only, no protected split`
- Sealed internal 是否已访问：`false`
- Official FoundIR test 是否已访问：`false`
- Evidence：`reports/foundir_blur_only_flux2_trajectory_endpoint_visual_review_v1.json` (`17e48c633813b4138c61fe037a9aef2740683612b5cd8b3b3ae2bb9f9f6d4d62`)

## Metrics 与 SOW 对照

| Task | PSNR / target | LPIPS / target | Checks |
|---|---:|---:|---:|
| n/a | n/a | n/a | informational gate |

## 结论与待决策项

- 原因/状态：Automatic train checks passed, but required visual/full-scale admission failed: source-disjoint candidate regressed 1.914361 dB and 0.035996 LPIPS versus parent, with sharp but spatially mismatched edges/text on hard motion examples.
- 请求 advisor 决策：请决定是否授权一个专门解决source-disjoint spatial correspondence的新设计；当前checkpoint不得full-scale、distill或decoder-tune。

在 advisor/user 的明确、记录化决定前，不改变 model family、architecture、split、evaluation target 或 acceptance protocol。
