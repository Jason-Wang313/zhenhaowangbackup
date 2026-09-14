# Advisor 阶段报告：zimage_conditional_teacher_pilot

> **DIAGNOSTIC ONLY / 非主线：本结果不是 FLUX.2 主线结果，不得替代或包装成 FLUX.2/SOW final result。**

- 结论：`failure`
- Exact model：`Z-Image Base 6B conditional noise-to-GT teacher, persistent LQ, FP32 LoRA`
- 是否为 FLUX.2 主线结果：`false`
- Config：`configs/foundir_blur_only_zimage_conditional_teacher_v1.json` (`c8bbf5d7dc85c2436f554a7e01e38b9d44d3f554686e001566c248d278da94db`)
- Checkpoint：`checkpoints/foundir_blur_only_zimage_conditional_teacher_v1_pilot/step_002000.safetensors` (`cf4a9b0f710f4026816df2898e10d299624139aa5472eec0610297540b465c54`)
- Evaluation split：`pilot_training_archive_source_disjoint_development`
- Sealed internal 是否已访问：`false`
- Official FoundIR test 是否已访问：`false`
- Evidence：`checkpoints/foundir_blur_only_zimage_conditional_teacher_v1_pilot/result.json` (`45b74d23629f6fe7a5f3d73d6f28891f29f044e86461ee9df220f5a49a984097`)

## Metrics 与 SOW 对照

| Task | PSNR / target | LPIPS / target | Checks |
|---|---:|---:|---:|
| n/a | n/a | n/a | informational gate |

## 结论与待决策项

- 原因/状态：Conditional teacher stage finished. Review fixed images and metrics; tiny measures learnability, diverse pilot measures generalization.
- 请求 advisor 决策：Complete visual review before the next authorized stage; no new advisor request.

在 advisor/user 的明确、记录化决定前，不改变 model family、architecture、split、evaluation target 或 acceptance protocol。
