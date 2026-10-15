# Advisor 阶段报告：flux2_lowlight_scalar_full_development_result

- 结论：`information`
- Exact model：`FLUX.2 Klein Base 4B isolated-Low-light step-400 plus frozen alpha=.925 scalar residual`
- 是否为 FLUX.2 主线结果：`true`
- Config：`configs/foundir_lowlight_flux2_scalar_residual_full_development_v1.json` (`fea0227d13b6d572897d683a8833e2312047cf28be2d4cf12462ddf95092856e`)
- Checkpoint：`checkpoints/foundir_four_task_flux2_capacity_sanity_v1/lowlight/step_000400.safetensors` (`592e49ca2606565c5aeec8aadeb5466cf0c34b30024d5c02f0f2292ad28c90d8`)
- Evaluation split：`all 1400 source-group-disjoint isolated-Low-light development records; no protected split`
- Sealed internal 是否已访问：`false`
- Official FoundIR test 是否已访问：`false`
- Evidence：`checkpoints/foundir_lowlight_flux2_scalar_residual_full_development_v1/result.json` (`2673ace6eac831b06684826cd732fbd14c57974eef81eb86e0260cf931bd93ca`)

## Metrics 与 SOW 对照

| Task | PSNR / target | LPIPS / target | Checks |
|---|---:|---:|---:|
| low_light | 19.7791 / >= 18.90 | 0.15406 / <= 0.28 | 2/2 |

## 结论与待决策项

- 原因/状态：Frozen-alpha Low-light full-development confirmation completed. SOW mean PSNR/LPIPS are formal; tail statistics are descriptive per explicit user acceptance scope. Human review remains required.
- 请求 advisor 决策：根据完整 Low-light development SOW metrics 与固定视觉审查，是否接受该 Low-light correction 并继续 Haze？

在 advisor/user 的明确、记录化决定前，不改变 model family、architecture、split、evaluation target 或 acceptance protocol。
