# FLUX.2 Blur-only 全量 development 结果

> DIAGNOSTIC ONLY / 非五类主线 final result。未访问 sealed internal 或 official FoundIR test。

- 数据：全部 2595 张 source-group-disjoint isolated Blur validation images（165 个 source groups）；每张固定一个 512×512 aligned crop。
- 模型：FLUX.2 Klein Base 4B；Blur-only；DiT LoRA rank/alpha 32/32；无 Qwen degradation conditioning、无 mixed degradation、无 pixel refiner。
- 推理：official shifted schedule，50 steps，guidance 1.0；所有 checkpoints 共用相同 crop 和 per-record seed。
- Evaluation config：`configs/foundir_blur_only_flux2_full_development_v1.json` (`42556934cfed99660916c11de14fb09508b5d628bef5dec7c8a070d8cc39bac9`)。
- Candidate checkpoint bundle：`reports/foundir_blur_only_flux2_candidate_bundle_v1.json` (`243ea85048b8c7f8bf3ad29212e497f97fbbb3043ac6b36974fce047991ebf19`)。

| Checkpoint | SHA-256 | PSNR | LPIPS | SOW PSNR >=25.9 | SOW LPIPS <=0.18 |
|---:|---|---:|---:|:---:|:---:|
| `checkpoints/foundir_blur_only_flux2_full_v1/step_002000.safetensors` | `0dac2f8f68949a8f9d017815e3aadf2d7826599c3447254530b36d0ce0ed8f53` | 24.9627 | 0.15685 | fail | pass |
| `checkpoints/foundir_blur_only_flux2_full_v1/step_004000.safetensors` | `013d30ccfbec62301b307276ed7858e2a546d2235e55d4ed68229f1979726c41` | 25.4778 | 0.14859 | fail | pass |
| `checkpoints/foundir_blur_only_flux2_full_v1/step_006000.safetensors` | `bfb9ef17f905dc62e20fd03964b2e8c264d8c8c843c21cb725611b266c2c5c18` | 25.3495 | 0.14785 | fail | pass |
| `checkpoints/foundir_blur_only_flux2_full_v1/step_008000.safetensors` | `f29c1b4a8fe683db7ffaec6cd906820776dddbf4cae49ef1f3c96486422c3822` | 25.6358 | 0.14484 | fail | pass |
| `checkpoints/foundir_blur_only_flux2_full_v1/step_010000.safetensors` | `e0c834d88fc69ba6916032ef7763cdf84047a60d2407eeef23eae8992359c439` | 25.7342 | 0.14541 | fail | pass |

- 明确 Pareto overfitting：否；定义为存在 PSNR、LPIPS 同时优于 step 10000 的 earlier checkpoint。
- 任一指标出现后期退化：是。
- PSNR 最佳：step 10000；LPIPS 最佳：step 8000。
- 请求 advisor 判断：根据全量 curve 和已发送的视觉结果，下一步是继续 Blur-only 调整，还是先审计 hallucination/training-inference code。
