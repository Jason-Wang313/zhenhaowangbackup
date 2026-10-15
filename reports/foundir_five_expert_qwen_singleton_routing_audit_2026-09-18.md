# Qwen → 五专家 singleton routing 审计

日期：2026-09-18

## 结论

现有 physical-v8 Qwen condition cache 不能直接接入五专家 router。

审计范围为五个完整、source-disjoint、isolated development split，共
8,269 records。严格规则为：Qwen level > 1 表示 active；只有恰好一个 active
label 才允许进入确定性五专家 router。Ground-truth category 只用于事后聚合，
不参与 route。

缓存身份：

- conditions CSV：
  `manifests/foundir_five_type_all_qwen_conditions_v3.csv`
- CSV SHA-256：
  `e07e16234d4d1eb674b5d667732600628448b215e9fab67e6e6732f99c1ea2f7`
- metadata SHA-256：
  `8e39ce6c0a1078b0af8c510ab7acb06b4ed53392ae21fa3e722de5e01a78e96e`
- decoding：修复后的 left padding；不是已作废的 right-padding incident
- adapter：
  `checkpoints/qwen_multidegradation_100k_rank64_per_type_physical_v8/merger_trainable/checkpoints/V8_physical_shard5/adapter.safetensors`
- adapter SHA-256：
  `5416311d7cd0aa0308e206084c13c7c950f9dacfc96cd558466642804ccbee3c`

## 严格 singleton 路由结果

| True type | Records / groups | Correct singleton | Wrong singleton | Safe refusal |
|---|---:|---:|---:|---:|
| Blur | 2,595 / 165 | 2,356 (90.7900%) | 10 (.3854%) | 229 (8.8247%) |
| Noise | 1,859 / 34 | 145 (7.7999%) | 273 (14.6853%) | 1,441 (77.5148%) |
| Haze | 1,580 / 158 | 1,263 (79.9367%) | 49 (3.1013%) | 268 (16.9620%) |
| Rain | 835 / 117 | 611 (73.1737%) | 28 (3.3533%) | 196 (23.4731%) |
| Low-light | 1,400 / 21 | 1,290 (92.1429%) | 2 (.1429%) | 108 (7.7143%) |
| **Overall** | **8,269 / 495** | **5,665 (68.5089%)** | **362 (4.3778%)** | **2,242 (27.1133%)** |

Safe refusals consist of 687 multi-label predictions and 1,555 empty
predictions. They reduce coverage but do not silently select a wrong expert.
The 362 wrong singleton predictions are deployment-blocking.

Wrong-singleton confusion:

- Blur → Haze: 2
- Blur → Low-light: 8
- Noise → Blur: 17
- Noise → Haze: 10
- Noise → Low-light: 246
- Haze → Blur: 14
- Haze → Low-light: 35
- Rain → Blur: 27
- Rain → Haze: 1
- Low-light → Noise: 2

## Root cause and decision

The current Qwen objective predicts five independent degradation severities and
was trained primarily on synthetic physical-v8 mixtures. The immediate
deployment problem is narrower: choose one of five isolated experts. Independent
thresholds create empty and multi-label outputs, while the synthetic-to-FoundIR
domain gap is especially severe for Noise.

Do not deploy this cache and do not tune routing thresholds on development.
The bounded next gate is a source-balanced, FoundIR-training-only, five-way
Qwen visual classification objective with a confidence-based abstention path.
It must be selected without development leakage and must report:

1. per-class and macro accuracy;
2. wrong-singleton rate;
3. abstention/coverage;
4. source-group-balanced confidence intervals;
5. the downstream route confusion matrix.

Mixed-degradation training or routing remains out of scope until this singleton
Qwen gate passes.
