# FoundIR mixed consensus router development result

Status: failed; not deployable.

## Frozen evaluation

- Frozen Qwen physical-v8 representation.
- Frozen five-class head at temperature `0.75`, confidence threshold `0.925`.
- Frozen eleven-class head at temperature `2.0`, mixed confidence threshold `0.95`.
- Mixed proposals were accepted only above the eleven-class threshold.
- Singleton proposals required agreement with the five-class head above its threshold.
- No model, Qwen adapter, or restoration expert was trained.

The evaluation covered the complete development data available to this route:

- 8,269 singleton records;
- 8,192 mixed records;
- 16,461 records total.

No FoundIR internal test or official test data was accessed.

## Result

- Correct accepted: 14,226 (86.4225%).
- Wrong accepted: 201 (1.2211%).
- Safe refusals: 2,034 (12.3565%).
- Coverage: 87.6435%.
- Accepted accuracy: 98.6068%.
- Gate: **fail** because overall wrong rate exceeded 0.5%, and two mixed classes exceeded the 1% per-class wrong limit.

| Mixed class | Records | Correct | Wrong | Refused | Coverage | Wrong rate |
|---|---:|---:|---:|---:|---:|---:|
| Blur+Noise | 579 | 512 | 3 | 64 | 88.9465% | 0.5181% |
| Haze+Low-light | 1,580 | 1,497 | 0 | 83 | 94.7468% | 0% |
| Rain+Haze | 1,580 | 1,537 | 10 | 33 | 97.9114% | 0.6329% |
| Blur+Low-light | 2,131 | 1,716 | 47 | 368 | 82.7311% | 2.2055% |
| Noise+Low-light | 1,743 | 743 | 138 | 862 | 50.5450% | 7.9174% |
| Blur+Noise+Low-light | 579 | 424 | 2 | 153 | 73.5751% | 0.3454% |

The dominant confusion was compositional collapse:

- 136 of 138 `Noise+Low-light` errors were routed to `Low-light`.
- 26 of 47 `Blur+Low-light` errors were routed to `Low-light`.
- The other 21 `Blur+Low-light` errors were routed to `Blur+Noise+Low-light`.

This shows that a mutually exclusive eleven-way head does not reliably preserve each active degradation component. Threshold sweeping the same head is not the next step: `Noise+Low-light` is already only barely above the 50% coverage floor.

## Bounded next direction

Keep Qwen and all restoration experts frozen. Replace the mutually exclusive eleven-way head with a compositional linear head containing five independent present/absent outputs: Blur, Noise, Rain, Haze, and Low-light. Accept only an exact supported component set with calibrated safety margins; otherwise abstain. This directly represents mixed inputs without another expensive model or restoration training round.

## Artifacts and runtime

- Result: `checkpoints/foundir_qwen_mixed_consensus_development_v3/result.json`
- Result SHA-256: `e4d7168c48b4fe0ffbb82bf9e14819001a30781989a03d1b8112a3e82c4ca3f2`
- Predictions: `checkpoints/foundir_qwen_mixed_consensus_development_v3/predictions.csv`
- Predictions SHA-256: `7fbd2a92ce659ecb6222b3c8971f182aa9a03006d3f1489a98a89d89030957a0`
- Launcher-to-result runtime: 430.65 seconds (7 minutes 10.65 seconds).
- Allocation: approximately 0.957 GPU-hours on eight GPUs.
