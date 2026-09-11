# Advisor 阶段报告：blur_only_tiny_overfit

> **DIAGNOSTIC ONLY / 非主线：本结果不是 FLUX.2 主线结果，不得替代或包装成 FLUX.2/SOW final result。**

- 结论：`failure`
- Exact model：`FLUX.2 Klein Base 4B DiT LoRA rank32 blur-only`
- 是否为 FLUX.2 主线结果：`false`
- Config：`configs/foundir_blur_only_flux2_sanity_v1.json` (`89a89dbc9303c8443760c9b999187370d85ff93b8d93b2d663cceedf5003cd36`)
- Checkpoint：`checkpoints/foundir_blur_only_flux2_sanity_v1/step_000400.safetensors` (`694b7bb89d9b9a9c3e7c4902d79f2800ffad54b9f04d98336121d2955c105d5c`)
- Evaluation split：`foundir_training_archive_same_crop_tiny_overfit_16; frozen_validation_not_accessed`
- Sealed internal 是否已访问：`false`
- Official FoundIR test 是否已访问：`false`
- Evidence：`checkpoints/foundir_blur_only_flux2_sanity_v1/result.json` (`9e7acbe57618bd207dbdce8bd6846a5a534ae59a395a9fcce8dd6714686f5e9d`)

## Metrics 与 SOW 对照

| Task | PSNR / target | LPIPS / target | Checks |
|---|---:|---:|---:|
| n/a | n/a | n/a | informational gate |

## 结论与待决策项

- 原因/状态：Restoration learned: PSNR +3.6892 dB and LPIPS -0.22332 on 16 training crops; formal preregistered gate failed only fixed flow-loss ratio 0.75556 > 0.35; 3/16 easy crops regressed in PSNR.
- 请求 advisor 决策：请决定是否接受 restoration+visual evidence 已证明 Blur learnability并授权 Blur-only full training，或因 flow-loss ratio 未过而继续 FLUX code/optimization diagnosis。

在 advisor/user 的明确、记录化决定前，不改变 model family、architecture、split、evaluation target 或 acceptance protocol。
