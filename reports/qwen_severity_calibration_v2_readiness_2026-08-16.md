# Qwen severity calibration v2 readiness

Date: 2026-08-16 UTC
Scope: Lowlight validation readiness and generic five-task safety interlocks

## Why v2 is required

The first frozen-Qwen Lowlight smoke used a subjective English zero-shot
rubric. Its JSON validity and degradation-type accuracy were both 100%, but
severity exact accuracy was only 36% and all five L5 inputs were predicted as
L4. Image paths, labels, hashes, Qwen visual inputs, and preprocessing were
audited and were correct. The failure was the untested assumption that a
general frozen VLM would automatically align to the project's synthetic L1-L5
scale.

The failed v1 outputs under `checkpoints/qwen_lowlight_smoke25/` remain
unchanged as diagnostic evidence.

## v2 protocol

The new bounded experiment keeps Qwen frozen and changes only severity
calibration and validation:

- calibration uses three train-only same-source visual sequences, each with
  complete labeled L1-L5 examples;
- validation uses 25 distinct, source-disjoint validation images, exactly five
  per level and no repeated source;
- the internal test and official FoundIR split are protected and are not used;
- decoding is deterministic with `CUBLAS_WORKSPACE_CONFIG=:4096:8`;
- JSON must be strictly valid for all 25 outputs;
- all five levels must be predicted;
- every level must achieve recall of at least 0.4;
- overall exact accuracy must be at least 0.6;
- Qwen must match or beat the train-fitted mean-luminance centroid baseline in
  both accuracy and MAE.

The frozen validation baseline is 64% exact accuracy and 0.40 MAE. Therefore,
Qwen cannot pass merely by producing plausible JSON or avoiding L5 collapse.
If this visual few-shot route fails, it does not authorize test evaluation or
restoration conditioning; the next bounded fallback is an ordinal head on
frozen Qwen visual embeddings, followed by Qwen LoRA only if necessary.

## Cross-task safeguards

The generic assessment protocol defines the canonical names `low_light`,
`noise`, `blur`, `haze`, and `rain`. For every task it rejects zero-shot,
rubric-only, or absent calibration; requires disjoint calibration and
validation splits; requires a comparison baseline; and applies structured
output, predicted-class coverage, per-level recall, accuracy, and
baseline-relative gates.

Only Lowlight is configured for the v2 validation experiment. Noise, Blur,
Haze, and Rain are explicitly marked `blocked_no_task_calibration`, so the
same uncalibrated mistake cannot silently advance those tasks. This is an
execution safeguard, not evidence that those four task datasets or assessors
have been run.

## CPU verification

The complete preflight passed with:

- config SHA-256:
  `25a1ad4085e13b4fffcef9e5f5c13238fa5188af4665d5eb13f7f68ab0d0d3d1`
- Qwen revision:
  `cc594898137f460bfe9f0759e9844b3ce807cfb5`
- 16 exact model files verified;
- calibration records: 15;
- calibration bundle SHA-256:
  `f9b05225cb2f68a7e4227eafa05479b993aae2e02a49c41431db66d852fd4e24`
- validation records: 25;
- validation bundle SHA-256:
  `c46a9e473137f9ee0dfd12288b1c6536c2761d4ff64cf191a0f93aac7d87855a`
- processor images per query: 16;
- total visual tokens per query: 7,721;
- rendered prompt tokens: 8,176;
- 35 repository unit tests passed;
- Python compilation, shell syntax, and `git diff --check` passed.

## Offline H100 command

```bash
CUDA_VISIBLE_DEVICES=0 smoke/run_qwen_lowlight_fewshot_validation25.sh
```

The launcher runs the fail-closed preflight again, uses the workspace-local
environment and model, preserves per-image JSON evidence, and exits nonzero if
any gate fails.
