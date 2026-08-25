# Qwen multi-degradation validation correction 1

Date: 2026-08-25 UTC

## Boundary

- FoundIR official test was not accessed.
- FoundIR training data was not used.
- The fixed synthetic test remains unaccessed.
- The advisor-approved Qwen architecture, JSON target, source-disjoint corpus,
  frozen decoder/tokenizer, LoRA scope, merger scope, and validation gates are
  unchanged.

## Completed ablation evidence

The 200-step overfit gate passed every structured, type, and severity metric at
1.0, proving that the trainable scope can represent the task. Both full
variants then completed all 630 optimizer steps and were evaluated on the same
1,200-image synthetic validation split.

| metric | merger frozen | merger trainable |
|---|---:|---:|
| valid structured output | 0.9850 | 0.9883 |
| type macro-F1 | 0.9120 | 0.9159 |
| type micro-F1 | 0.9123 | 0.9164 |
| degradation-set exact match | 0.7558 | 0.7617 |
| combined type+severity exact match | 0.4992 | 0.4983 |
| typed severity accuracy | 0.5847 | 0.6005 |
| typed severity MAE | 0.7403 | 0.7167 |
| triple set exact match | 0.4694 | 0.4833 |

Neither variant passed the unchanged validation gate. Selection therefore
returned `no_variant_passed`, and the fixed test was not run.

## Root-cause audit

Trainable merger improved type F1, set accuracy, severity accuracy, and
severity MAE, so it is the evidence-backed continuation base. Error
decomposition showed a sharp composition-size discontinuity:

| composition | set exact match | combined exact match |
|---|---:|---:|
| clean | 0.9750 | 0.9750 |
| single | 0.8889 | 0.6750 |
| double | 0.8417 | 0.4778 |
| triple | 0.4833 | 0.1833 |

The curriculum exposed clean and single samples in A, B, and C; double samples
in B and C; and triple samples only in C. In addition, the one global cosine
schedule was already decaying when triples first appeared. Triple images thus
received both the fewest exposures and the weakest learning-rate interval.
This matches the observed failure: triple samples predominantly omit one true
degradation, while single and double type recognition are already strong.

The 14 invalid outputs were not truncations. They were valid JSON predictions
of four or five active types, rejected by the corpus maximum of three. Raising
`max_new_tokens` would therefore not address the core failure.

## Bounded correction

Correction 1 starts from the hash-pinned trainable-merger C checkpoint, creates
a fresh optimizer, and trains two shuffled epochs over the complete 9,600-image
training mixture. This adds the missing full-mixture exposure and restarts a
bounded cosine schedule without changing data, labels, architecture, or gates.

- LoRA learning rate: 5e-5
- merger learning rate: 5e-6
- optimizer steps: 600
- effective global batch: 32 on both 2xH100 and 4xH100
- validation: the same synthetic validation split
- fixed test: automatically accessed exactly once only if every validation
  gate passes

All initialization, base-validation, overfit, manifest, generation, model, and
real-sanity artifacts are verified by SHA-256 before GPU training. The
correction preflight also fails if the base fixed-test artifact exists.
