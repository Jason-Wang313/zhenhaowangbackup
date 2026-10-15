# Advisor 阶段报告：sealed_internal_final

- 结论：`failure`
- Exact model：`VLM-guided FLUX.2 Klein 4B; frozen physical-v8 Qwen; DiT LoRA + pixel refiner`
- 是否为 FLUX.2 主线结果：`true`
- Config：`configs/foundir_five_type_flux2_dit_v3.json` (`5145e97b2f170b617f34180783d8e7934a77b5ef82f2e6c5cae7756c89763810`)
- Checkpoint：`checkpoints/foundir_five_type_flux2_dit_v3/full_curriculum/step_042500.safetensors` (`8e7b30710fa78d297c27a78c96063052561d4200d31806daac4b0c8a3a867766`)
- Evaluation split：`sealed_internal`
- Sealed internal 是否已访问：`true`
- Official FoundIR test 是否已访问：`false`
- Evidence：`checkpoints/foundir_five_type_flux2_dit_v3/final_internal_freeze.json` (`e4a4332e7b3f7e83bf447be9d94435330c0500cdd0b3baef8d4257a2cf582d7d`)

## Metrics 与 SOW 对照

| Task | PSNR / target | LPIPS / target | Checks |
|---|---:|---:|---:|
| blur | 20.2746 / >= 25.90 | 0.24732 / <= 0.18 | 0/2 |
| noise | 21.8466 / >= 38.90 | 0.14974 / <= 0.10 | 0/2 |
| haze | 17.9196 / >= 22.10 | 0.25581 / <= 0.19 | 0/2 |
| rain | 21.2095 / >= 34.20 | 0.25095 / <= 0.08 | 0/2 |
| low_light | 18.1965 / >= 18.90 | 0.25837 / <= 0.28 | 1/2 |

## Training setup

- Backbone：FLUX.2 Klein 4B；VAE 与 native Qwen3 text encoder frozen；DiT LoRA rank/alpha `32/32`；另含 `33,251`-parameter bounded pixel refiner。
- Data：FoundIR training archive 中五类及其 single/double/triple mixtures，共 `685,495` records；不含 JPEG；official test 未参与。
- Optimization：8 GPUs，global batch `32`，AdamW，full curriculum 共 `1,800,000` presentations / `56,250` optimizer steps，learning rate `5e-5`，BF16。
- Selection/evaluation：仅用 source-group-disjoint development 在 6 个 candidate checkpoints 中按五类 PSNR/LPIPS joint margin 选择；随后只运行一次 sealed internal。

## 结论与待决策项

- 原因/状态：FLUX.2 sealed internal 未达到 frozen SOW gate（仅 `1/10` checks 通过）。现有证据说明 restoration capacity/optimization 存在明显缺口，尤其是 Noise 与 Rain PSNR；目前尚不能声称已确认唯一深层原因。
- 请求 advisor 决策：请决定继续 FLUX.2 并批准具体 architecture/capacity correction，或明确批准改变 model family/route。

在 advisor/user 的明确、记录化决定前，不改变 model family、architecture、split、evaluation target 或 acceptance protocol。
