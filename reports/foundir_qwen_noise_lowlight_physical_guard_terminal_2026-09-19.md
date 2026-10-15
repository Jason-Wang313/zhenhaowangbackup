# Noise / Noise+Low-light physical guard terminal result

Date: 2026-09-19

## Scope and policy

- The rule was frozen from 90 Noise and 90 Noise+Low-light calibration records.
- It uses only robust image exposure and dark-region high-frequency energy.
- It is applied only when the frozen dual Qwen router proposes `noise`.
- No model was trained and no restoration expert was changed.
- FoundIR development internal and official test data were not accessed.

## Frozen rule

- Calibration: 0 wrong, 47.78% standalone physical-rule coverage.
- Rule SHA-256: `029098ef90228f7b6a55a6166b2d776a71481d7bac246097570ee1a893b5f7d8`

## Prospective training-only holdout

Only six untouched paired source groups remained, producing 12 records.

- Correct: 9/12
- Wrong: 0/12
- Refusal: 3/12
- Coverage: 75.00%
- Noise coverage: 66.67%
- Noise+Low-light coverage: 83.33%
- Gate: pass, with an explicit effective-sample-size limitation
- Runtime: 127.21 seconds on 8 GPUs
- Result SHA-256: `fceb6f0f6310fbd5717dd9008effe7c455b6f83be6149ea134ea346ff42817d3`

## Frozen full-development integration replay

The exact frozen rule was replayed without adjustment over the existing 16,461-record development predictions. It inspected the 1,051 records previously routed as `noise`.

- Overall wrong rate: 0.2734%
- Overall coverage: 81.6050%
- Noise wrong rate: 1.0758% (fails the 1% per-class maximum)
- Noise coverage: 37.3319% (fails the 50% per-class minimum)
- Noise+Low-light wrong rate: 0.6311%
- Noise+Low-light coverage: 67.6420%
- Gate: **fail**
- GPU hours: 0
- Predictions SHA-256: `fb3ff1eef8b5acaf9cadc6cc973735c28b430144fc3da69d73430270d4d6633e`

## Terminal decision

The physical guard is rejected and must not be deployed. Dark pure-Noise images overlap the Noise+Low-light exposure range, so brightness plus dark-region noise does not provide a stable discriminant.

Do not tune these thresholds against either exposed holdout or the full development set, and do not add another head using the same evidence. The five-class singleton router remains the frozen deployable router. Mixed routing remains unresolved.

If mixed support continues, the next bounded direction should remove this routing boundary instead of tuning it: first test whether one existing restoration route is safe for both Noise and Noise+Low-light; only if neither is cross-compatible should a single shared Noise-family restoration expert be trained on both categories.
