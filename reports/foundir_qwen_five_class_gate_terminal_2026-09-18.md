# FoundIR Qwen five-class singleton gate terminal result

Status: completed; strict training-only holdout gate failed narrowly.

## Frozen method

- Frozen Qwen physical-v8 step-2500 visual representation.
- One standardized five-way multinomial logistic-regression head.
- Source-balanced FoundIR isolated training records only.
- Fit/calibration/holdout source groups are disjoint.
- Selected on calibration: `C=0.01`, temperature `0.75`, confidence threshold `0.925`.
- Development and internal-test data were not accessed.

## Calibration selection

- Records: 450.
- Correct accepted: 437 (97.1111%).
- Wrong accepted: 0 (0%).
- Refused: 13 (2.8889%).
- Accepted accuracy: 100%.

## Independent training-only holdout

- Records: 450 (90 per class).
- Correct accepted: 433 (96.2222%).
- Wrong accepted: 2 (0.4444%).
- Refused: 15 (3.3333%).
- Coverage: 96.6667%.
- Accepted accuracy: 99.5402%.

Per class:

| Class | Correct | Wrong | Refused | Coverage | Wrong rate |
|---|---:|---:|---:|---:|---:|
| Blur | 87 | 0 | 3 | 96.6667% | 0% |
| Noise | 87 | 1 | 2 | 97.7778% | 1.1111% |
| Haze | 87 | 0 | 3 | 96.6667% | 0% |
| Rain | 85 | 0 | 5 | 94.4444% | 0% |
| Low-light | 87 | 1 | 2 | 97.7778% | 1.1111% |

## Frozen gate decision

- Overall wrong rate <=0.5%: pass (`0.4444%`).
- Every-class wrong rate <=1%: fail for Noise and Low-light (`1/90 = 1.1111%` each).
- Overall coverage >=65%: pass (`96.6667%`).
- Every-class coverage >=50%: pass (minimum `94.4444%`).
- Final gate: fail.

The per-class gate effectively required zero errors because each class had 90 records. The result must not be relabeled as a pass after seeing the holdout.

## Runtime and artifacts

- Successful run wall time: 101.91 seconds.
- Successful-run allocation: approximately 0.226 GPU-hours on 8 GPUs.
- Result: `checkpoints/foundir_qwen_five_class_gate_v1/result.json`.
- Result SHA-256: `ecc942d55cd160186a11befe01a4e012d31a63393b7db9c3c4de7a18eb6ccd9b`.
- Head SHA-256: `0ae6fe27a48d1fd99bfa86e23fea14129c4567494eb38d92543389be0984bdab`.

## Bounded next step

Do not retrain Qwen and do not access development yet. Use a more conservative confidence policy selected without reusing this holdout, then evaluate once on a fresh, previously unused training-source holdout. Mixed degradation remains blocked until the singleton gate passes.
