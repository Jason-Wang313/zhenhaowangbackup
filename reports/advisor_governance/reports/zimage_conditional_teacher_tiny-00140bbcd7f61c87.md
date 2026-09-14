# Advisor 阶段报告：zimage_conditional_teacher_tiny

> **DIAGNOSTIC ONLY / 非主线：本结果不是 FLUX.2 主线结果，不得替代或包装成 FLUX.2/SOW final result。**

- 结论：`failure`
- Exact model：`Z-Image Base 6B conditional noise-to-GT teacher, persistent LQ, FP32 LoRA`
- 是否为 FLUX.2 主线结果：`false`
- Config：`configs/foundir_blur_only_zimage_conditional_teacher_v1.json` (`c8bbf5d7dc85c2436f554a7e01e38b9d44d3f554686e001566c248d278da94db`)
- Checkpoint：`checkpoints/foundir_blur_only_zimage_conditional_teacher_v1_tiny/step_001200.safetensors` (`52a61cd9412f494c2141d5f34ffd9cbb2dc66911755e008f2828a63071ea93a7`)
- Evaluation split：`tiny_training_archive_source_disjoint_development`
- Sealed internal 是否已访问：`false`
- Official FoundIR test 是否已访问：`false`
- Evidence：`checkpoints/foundir_blur_only_zimage_conditional_teacher_v1_tiny/result.json` (`ed945a012bd39ac6d08cc3d08f5f9ce0ce4e662f229405ecc21fe06d4281359d`)

## Metrics 与 SOW 对照

| Task | PSNR / target | LPIPS / target | Checks |
|---|---:|---:|---:|
| n/a | n/a | n/a | informational gate |

## 结论与待决策项

- 原因/状态：Conditional teacher stage finished. Review fixed images and metrics; tiny measures learnability, diverse pilot measures generalization.
- 请求 advisor 决策：Complete visual review before the next authorized stage; no new advisor request.

在 advisor/user 的明确、记录化决定前，不改变 model family、architecture、split、evaluation target 或 acceptance protocol。
