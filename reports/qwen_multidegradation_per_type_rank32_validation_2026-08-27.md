# Qwen per-type rank-32 validation result

## Outcome

The advisor-directed Qwen2.5-VL experiment asked the same image five separate
questions, one each for blur, noise, rain, haze, and low-light. It used rank-32
visual/language LoRA, a trainable visual merger, no classifier heads, global
batch 100, and three complete epochs.

Training completed all 1,440 optimizer steps. Epoch mean losses were
`0.0774334`, `0.0405060`, and `0.0297576`; each epoch recorded
`loss_still_descending=true`. The 32-image overfit gate achieved 1.0 on every
type, level, and composition metric.

Frozen validation completed all 6,000 type-specific decoder queries for 1,200
source-disjoint synthetic images. The automatic validation gate failed, so
selection, protected fixed-test inference, and real-image sanity inference did
not run.

## Validation metrics

Passed checks:

- valid response and valid structured output: `1.0000`;
- per-query five-class level accuracy: `0.8665 >= 0.80`;
- type macro-F1: `0.93282 >= 0.90`;
- type micro-F1: `0.93333 >= 0.92`;
- type precision: `0.97193`;
- type recall: `0.89769`;
- gradient scope: passed.

Failed checks:

- degradation-set exact: `0.7900 < 0.80`;
- combined type-and-level exact: `0.50833 < 0.65`;
- active-level accuracy: `0.65509 < 0.75`;
- active-level MAE with missing penalty: `0.58796 > 0.50`;
- triple degradation-set exact: `0.5500 < 0.65`.

The overall 86.65% per-query accuracy is partly driven by absent L1 decisions:
absent-level accuracy was 98.54%, while active L2--L5 exact accuracy was
65.51%. Per-type five-class accuracies were blur 82.92%, noise 90.42%, rain
88.83%, haze 87.50%, and low-light 83.58%.

For the 360 triple-degradation images, 198 (55.0%) had all three types detected
with no false positive, 4 detected all three plus one false positive, 142
detected two, and 16 detected only one. Triple active-component recall was
83.89%, but all-three detection was only 56.11% and exact-set accuracy was
55.0%. Triple active-level exact accuracy was 57.04%.

## Evaluation recovery

The first validation attempt completed generation but exposed an evaluation
bug when four/five active predictions were passed through the ground-truth
maximum-three validator. Commit `ec78ba14e3e4c3fad2a62f12aa6bc19f79e9cecf`
fixed the prediction/ground-truth boundary and added hash-checked
validation-only recovery. Four active predictions occurred in only four of
1,200 validation records and were correctly retained as false positives.

One recovery process was externally terminated without a Python traceback or
runner exit marker. A later recovery completed normally. Neither event changed
the adapter or validation data.

## Frozen artifacts

- config SHA-256:
  `43af6977452d4d62a5845d18c3d55cbcd30c56d9e9c1e9f8019b76c3983d5f58`;
- H3 adapter SHA-256:
  `5c7df484672088806cca56c83947cc3a3c7fa695b608bba79ed3a9ee11d2399b`;
- validation result SHA-256:
  `c3939afab873a34b744aeab961699732345eb0da573c9641d232a1ddd4099051`;
- validation predictions SHA-256:
  `9ee926d50d81b7e7339c2b0aac09f5de0309370c9f9bf5be5c4cf7a924f2be1`.

## Scientific stop condition

The experiment meets the advisor's stated expectation of 80%+ overall
per-question accuracy and demonstrates strong type recognition. It does not
meet the frozen project's stronger requirements for exact active severity and
all-three compound recognition. Do not change gates after seeing validation,
access the protected fixed test, or authorize this checkpoint to guide FLUX
until the advisor clarifies which objective is authoritative.
