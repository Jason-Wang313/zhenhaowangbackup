# Advisor 阶段报告：flux2_blur_source_diverse_spatial_anchor_pilot_result

- 结论：`failure`
- Exact model：`FLUX.2 Klein Base 4B isolated-Blur step-10000 parent paired source-diverse spatial-anchor pilot`
- 是否为 FLUX.2 主线结果：`true`
- Config：`configs/foundir_blur_only_flux2_source_diverse_spatial_anchor_pilot_v1.json` (`ab5a41af64d6a8e39e228625a18b5b978ab616cc007308dcd2999242bb1a05be`)
- Checkpoint：`checkpoints/foundir_blur_only_flux2_source_diverse_spatial_anchor_pilot_v1/arms/trajectory_spatial_anchor/step_000400.safetensors` (`a1be1e983d33de71faa6ea0da3be1ff161a73c5bc791e602cb7be0a5fff3d29d`)
- Evaluation split：`512 unique training source groups plus fixed 16 source-disjoint development crops; no protected split`
- Sealed internal 是否已访问：`false`
- Official FoundIR test 是否已访问：`false`
- Evidence：`checkpoints/foundir_blur_only_flux2_source_diverse_spatial_anchor_pilot_v1/result.json` (`47d72b0d4efb887f9e684d09e83854dcd3e9a00b233ee5c707292fefab674de7`)

## Metrics 与 SOW 对照

| Task | PSNR / target | LPIPS / target | Checks |
|---|---:|---:|---:|
| n/a | n/a | n/a | informational gate |

## 结论与待决策项

- 原因/状态：Source-diverse paired pilot finished; inspect predeclared development gate and all fixed Input|Output|GT visuals. No automatic full-scale action.
- 请求 advisor 决策：若step-400 development与visual gate均通过，是否授权扩到完整105057-record isolated-Blur training？

在 advisor/user 的明确、记录化决定前，不改变 model family、architecture、split、evaluation target 或 acceptance protocol。
