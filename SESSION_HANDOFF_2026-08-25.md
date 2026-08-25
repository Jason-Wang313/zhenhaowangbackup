# Session handoff — 2026-08-25

## Mandatory startup

Work only in `/inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang` and begin
by telling the user exactly: `I am working in zhenhaowang, which is the correct
workspace.` Read `agents.md` and this handoff before acting. Preserve all
unrelated tracked and untracked user files. Verify `HEAD`, `origin/main`, and
the machine-readable artifacts directly rather than relying on chat history.

## Protected-data rules

- Do not access the FoundIR official test.
- Do not use FoundIR training data for this Qwen stage.
- The synthetic fixed test has not been accessed by Correction 4 and remains
  protected for exactly one evaluation only after all frozen validation gates
  pass.
- Do not lower gates or tune from a fixed-test result.

## Advisor-approved objective

Qwen must identify every present degradation and each independent strength:
blur, noise, rain, haze, and low-light, with L1 absent and L2--L5 increasing
severity. The downstream interface remains:

`Qwen JSON -> deterministic canonical restoration text -> FLUX text encoder -> FLUX attention`

The eventual restoration model is one FoundIR-trained all-in-one model. The
synthetic AgenticIR corpus is for Qwen supervision, not a substitute FoundIR
restoration dataset.

## Correction history

All four full validation variants failed at least one frozen gate and stopped
before fixed-test access:

| Variant | Macro-F1 | Set exact | Combined exact | Typed severity accuracy | Typed severity MAE | Triple set exact |
|---|---:|---:|---:|---:|---:|---:|
| Correction 1 | 0.9285 | 0.7842 | 0.6258 | 0.7074 | 0.5398 | 0.5500 |
| Correction 2 | 0.9314 | 0.7958 | 0.6333 | 0.7130 | 0.5296 | 0.5750 |
| Correction 3 | 0.9282 | 0.7900 | 0.6333 | 0.7162 | 0.5278 | 0.5806 |
| Correction 4 | 0.9287 | 0.7875 | 0.6075 | 0.6801 | 0.5694 | 0.5750 |

Correction 4 replaced variable-length JSON with five fixed slots:

`{"blur":1,"noise":1,"rain":1,"haze":1,"low_light":1}`

The full implementation was committed and pushed at
`9e1a838aba33b34b594fd8e5c4f666e04a4c3918`. Its frozen config is
`configs/qwen_multidegradation_lora_correction4_v1.json`, SHA-256
`b954bd3822b1250ee0571bec91d1a4c7d1e4503bb72017e952db28ebfc82768a`.

## Correction-4 GPU result

The user launched the run on two NVIDIA H100 80 GB GPUs. It ran from
`2026-08-25T04:14:58Z` to `2026-08-25T04:49:52Z` and completed all 600 steps.
Training took `1910.61` seconds; the final reported loss was `0.0582643`.
Gradient scope passed on both ranks. The runner exited with intentional status
2 after frozen validation failed; this was not a crash.

Validation passed structure (`0.9967`), macro-F1 (`0.9287`), micro-F1
(`0.9291`), and gradient scope. It failed:

- degradation-set exact: `0.7875 < 0.80`;
- combined exact: `0.6075 < 0.65`;
- typed severity accuracy: `0.6801 < 0.75`;
- typed severity MAE: `0.5694 > 0.50`;
- triple set exact: `0.5750 < 0.65`.

Micro precision was high (`0.9623`) but recall was only `0.8981`. Across the
five slots, 220 active types were missed versus 76 false positives. Fixed slots
therefore did not solve compound omission and regressed most decision metrics
relative to Correction 2.

Authoritative artifacts:

- result:
  `checkpoints/qwen_multidegradation_lora_correction4_v1/merger_trainable/validation_result.json`,
  SHA-256 `8972c5d9d4dd201b123286bd7eff4a9ca16e6ad5d3eb5b7aab635dd826e5029d`;
- predictions:
  `checkpoints/qwen_multidegradation_lora_correction4_v1/merger_trainable/validation_predictions.json`,
  SHA-256 `2b09e832d8fdf0c2f5e13a94f2968600c5c8c89fa4b6c3080329bf88a3dccd80`;
- adapter:
  `checkpoints/qwen_multidegradation_lora_correction4_v1/merger_trainable/checkpoints/G_fixed_slot_independent_decisions/adapter.safetensors`,
  SHA-256 `f5b2fd7d303142e8655c11457f943efb4c3bc01e2205eda8279b61f00b84afcf`;
- report: `reports/qwen_multidegradation_validation_correction4_2026-08-25.md`;
- log: `logs/qwen_multidegradation_lora_correction4_v1.log`.

No Correction-4 `fixed_test_result.json` or `fixed_test_predictions.json`
exists. Selection, real-sanity inference, and rendering did not run. FoundIR
was not accessed.

## Next decision

Do not launch Correction 4 again and do not start a fifth validation-tuned
continuation. The repeated plateau is now evidence that schema and loss-level
patches are not changing the information learned for compound recognition.
The next step requires a genuinely different, advisor-reviewed scientific
choice, such as stronger compound-identifiability supervision or a direct
five-head discriminative readout whose predictions still serialize to the
approved Qwen JSON/canonical-text interface. The current adapter is negative
evidence and is not authorized to guide FLUX.
