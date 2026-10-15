# FoundIR Qwen five-class conservative v2 terminal result

Status: completed; the individually frozen v2 gate failed narrowly.

## Frozen v2 policy

- Qwen adapter and v1 linear head remained frozen.
- Confidence threshold was tightened once from `0.925` to `0.95`.
- No retraining and no threshold sweep were performed.
- Holdout: 300 FoundIR training records, 60 per class.
- Every v2 source was globally excluded from all v1 fit, calibration, and holdout sources.
- Development and internal-test data were not accessed.

## v2 result

- Correct accepted: 288/300 (96.0000%).
- Wrong accepted: 1/300 (0.3333%).
- Refused: 11/300 (3.6667%).
- Coverage: 96.3333%.
- Accepted accuracy: 99.6540%.

| Class | Correct | Wrong | Refused | Coverage | Wrong rate |
|---|---:|---:|---:|---:|---:|
| Blur | 56 | 1 | 3 | 95.0000% | 1.6667% |
| Noise | 59 | 0 | 1 | 98.3333% | 0% |
| Haze | 57 | 0 | 3 | 95.0000% | 0% |
| Rain | 57 | 0 | 3 | 95.0000% | 0% |
| Low-light | 59 | 0 | 1 | 98.3333% | 0% |

The sole error was `foundir_train_0035610`, labeled Blur and predicted Noise with confidence `0.9805519`. It is a high-confidence error, so another small confidence-threshold sweep is not justified.

## Frozen gate decision

- Overall wrong rate <=0.5%: pass (`0.3333%`).
- Every-class wrong rate <=1%: fail for Blur (`1/60 = 1.6667%`).
- Overall coverage >=65%: pass (`96.3333%`).
- Every-class coverage >=50%: pass (minimum `95%`).
- Final v2 gate: fail.

## Cumulative evidence (descriptive, not a retroactive gate)

The v1 and v2 terminal holdouts are source-disjoint. Their descriptive aggregate is:

- Records: 750 (150 per class).
- Correct accepted: 721 (96.1333%).
- Wrong accepted: 3 (0.4000%).
- Refused: 26 (3.4667%).
- Coverage: 96.5333%.
- Accepted accuracy: 99.5856%.
- Per-class wrong rates: Blur `0.6667%`, Noise `0.6667%`, Low-light `0.6667%`, Haze `0%`, Rain `0%`.

This aggregate satisfies the numerical intent of the frozen limits, but it was computed after both terminal results were known and must not be mislabeled as either preregistered gate passing.

## Runtime and artifacts

- Wall time from launcher PID publication to result: approximately 42.28 seconds.
- Allocation: approximately 0.094 GPU-hours on 8 GPUs.
- Result: `checkpoints/foundir_qwen_five_class_fresh_holdout_v2/result.json`.
- Result SHA-256: `dff4c6d14a754f278f0ea005b7d9eefa50b3fc5ab2a6b7f89b7f95c86185faeb`.

## Recommendation

Do not run another confidence-threshold sweep. The remaining error is high confidence, while cumulative source-disjoint evidence already shows a 0.4% overall wrong-route rate and at most 0.6667% per class. The next decision should be explicit: either accept this cumulative safety evidence and freeze the router for one development evaluation, or require a new classifier intervention with a genuinely fresh terminal dataset. Do not blur those two choices.
