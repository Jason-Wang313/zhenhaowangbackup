# Advisor 阶段报告：zimage_blur_vae_ceiling_solver_sweep_v1

> **DIAGNOSTIC ONLY / 非主线：本结果不是 FLUX.2 主线结果，不得替代或包装成 FLUX.2/SOW final result。**

- 结论：`information`
- Exact model：`Frozen Z-Image v3 Blur teacher VAE-ceiling and 1/2/4/8/16/50-step diagnostic`
- 是否为 FLUX.2 主线结果：`false`
- Config：`configs/foundir_blur_only_zimage_vae_ceiling_v1.json` (`062d697b9a0e29728affd983212c27fcff2325a6764099bfca7b4f36b9953292`)
- Checkpoint：`not_applicable` (`not_applicable`)
- Evaluation split：`fixed_16_training_pairs_plus_16_source_disjoint_visual_development_pairs`
- Sealed internal 是否已访问：`false`
- Official FoundIR test 是否已访问：`false`
- Evidence：`checkpoints/foundir_blur_only_zimage_vae_ceiling_v1/result.json` (`b33eac791bd03a61783e0ad5ca288de75ed7e273fd832865de0c423b8a3cc431`)

## Metrics 与 SOW 对照

| Task | PSNR / target | LPIPS / target | Checks |
|---|---:|---:|---:|
| n/a | n/a | n/a | informational gate |

## 结论与待决策项

- 原因/状态：Frozen v3 teacher VAE ceiling and solver-step sweep completed; use the measured ceiling and best step count to choose distillation and decoder fine-tuning.
- 请求 advisor 决策：No new advisor decision is required; teacher distillation and conditional decoder tuning were already explicitly authorized by the advisor and user.

在 advisor/user 的明确、记录化决定前，不改变 model family、architecture、split、evaluation target 或 acceptance protocol。
