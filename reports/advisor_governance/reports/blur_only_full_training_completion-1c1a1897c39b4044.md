# Advisor 阶段报告：blur_only_full_training_completion

- 结论：`information`
- Exact model：`FLUX.2 Klein Base 4B DiT LoRA rank32 isolated-Blur-only`
- 是否为 FLUX.2 主线结果：`true`
- Config：`configs/foundir_blur_only_flux2_full_v1.json` (`ba5e2975ba36c9300d454c573882312ddda7415a3de370efcef707899512b61c`)
- Checkpoint：`checkpoints/foundir_blur_only_flux2_full_v1/step_010000.safetensors` (`e0c834d88fc69ba6916032ef7763cdf84047a60d2407eeef23eae8992359c439`)
- Evaluation split：`source_group_disjoint_development_fixed_visual_subset`
- Sealed internal 是否已访问：`false`
- Official FoundIR test 是否已访问：`false`
- Evidence：`checkpoints/foundir_blur_only_flux2_full_v1/training_summary.json` (`c5f93d191b219174ee80d0751687d47f188b7d0efdacc4ef22e0947c6735d62e`)

## Metrics 与 SOW 对照

| Task | PSNR / target | LPIPS / target | Checks |
|---|---:|---:|---:|
| n/a | n/a | n/a | informational gate |

## 结论与待决策项

- 原因/状态：Blur-only full training completed; every candidate has fixed development triptychs; checkpoint selection is intentionally paused before metrics for human visual screening.
- 请求 advisor 决策：请先肉眼检查每个 candidate 的固定 Input|Output|GT；确认 visually valid candidates 后，是否授权仅在这些 candidates 上比较完整 source-group-disjoint Blur development PSNR/LPIPS？

在 advisor/user 的明确、记录化决定前，不改变 model family、architecture、split、evaluation target 或 acceptance protocol。
