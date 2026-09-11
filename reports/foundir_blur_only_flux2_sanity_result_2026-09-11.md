# FLUX.2 Blur-only tiny-overfit result — 2026-09-11

## Advisor-ready conclusion

按照 advisor 要求，我们先排除了 Qwen degradation prediction 和 mixed degradation，
只用 isolated Blur 做了 FLUX.2 DiT-LoRA tiny overfit。结果证明当前 FLUX.2 pipeline
**能够学会 deblurring mapping**：16 个 training crops 的平均 PSNR 从
`24.7232` 提升到 `28.4124 dB`（`+3.6892 dB`），LPIPS 从 `0.30852` 降到
`0.08520`；PSNR 改善 `13/16`，LPIPS 改善 `16/16`。固定样本的 checkpoint
triptychs 也显示强 motion blur 的边缘与纹理被明显恢复，内容和 geometry 基本保持，
未见明显 structural hallucination。

但是，预注册 automatic gate 的正式结论仍是 **failure (4/5 checks passed)**：
fixed flow-loss probe 仅从 `0.59511` 降到 `0.44964`，ratio 为 `0.75556`，没有达到
预注册的 `<=0.35`。我们没有降低 threshold，也没有把 formal failure 改称 pass。

## Exact experiment

- Scope：advisor-authorized diagnostic；仅 isolated Blur tiny overfit。
- Model：FLUX.2 Klein Base 4B；frozen VAE；frozen native Qwen3 text encoder；
  trainable DiT LoRA rank/alpha `32/32`，共 `50` 个 target modules。
- Disabled：Qwen degradation prediction、mixed degradation、pixel refiner。
- Objective：pure-noise-to-clean reference-conditioned rectified flow。
- Inference：official 50-step shifted schedule，CFG guidance `1.0`。
- Data：FoundIR training archive 中冻结的 `16` 个 Blur pairs，来自 `16` 个独立
  source groups；crop size `512`。另冻结了 `16` 个 source-group-disjoint validation
  pairs，但本次没有读取或推理这些 validation pairs。
- Training：8×H200，global batch `8`，`400` optimizer steps，LR `1e-4`，
  bfloat16；wall time `244.58 s`。
- Config：`configs/foundir_blur_only_flux2_sanity_v1.json`
  (`89a89dbc9303c8443760c9b999187370d85ff93b8d93b2d663cceedf5003cd36`)
- Frozen manifest：`manifests/foundir_blur_only_sanity_v1.csv`
  (`7114dab36b3978f28657f366e9a961462c3dbfb06927c9262754cced0d8c8e6f`)
- Checkpoint：`checkpoints/foundir_blur_only_flux2_sanity_v1/step_000400.safetensors`
  (`694b7bb89d9b9a9c3e7c4902d79f2800ffad54b9f04d98336121d2955c105d5c`)
- Result：`checkpoints/foundir_blur_only_flux2_sanity_v1/result.json`
  (`9e7acbe57618bd207dbdce8bd6846a5a534ae59a395a9fcce8dd6714686f5e9d`)
- Code：commit `91e953cd5f13b012796f827ec8303e9f0dbc507f`，与 `origin/main` 一致。
- Protected data：sealed internal 未访问；official FoundIR test 未访问。

## Checkpoint progression on the same fixed 8 training crops

| Step | PSNR | LPIPS |
|---:|---:|---:|
| 0 | 16.5435 | 0.4262 |
| 50 | 20.2117 | 0.2554 |
| 100 | 19.4589 | 0.2296 |
| 200 | 24.0201 | 0.1262 |
| 400 | 27.8000 | 0.0752 |

## Limits and visual finding

- 这是 same-training-crop learnability test，不是 validation、sealed evaluation 或 SOW
  final result；其绝对 PSNR/LPIPS 不得与 SOW threshold 比较。
- `3/16` 较轻、原本已经较清晰的 crops 出现 PSNR regression（约 `-1.03`、
  `-1.26`、`-2.89 dB`），提示 easy cases 可能存在 over-restoration；full training
  前需保留 identity/easy-case control 和固定 validation visual review。
- Codex 已对 step-400 的全部 16 张 `Input | Output | GT` 做 preliminary visual
  review；仍需 user/advisor 肉眼确认，不能用自动指标替代。

## Decision requested

请 advisor 明确决定二选一：

1. 是否接受 restoration metrics + checkpoint visual evidence 已证明 Blur learnability，
   授权进入 source-group-disjoint validation、固定 triptych visual review 的 Blur-only
   full training；或
2. 是否因为预注册 flow-loss ratio 未过，要求继续停在 FLUX code/optimization diagnosis。

在 advisor 明确回复并记录前，不启动 Blur-only full training、其他 GPU experiment、
Z-Image、sealed internal 或 official FoundIR test。
