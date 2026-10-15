# FoundIR Qwen eleven-class training-only terminal result

## Decision

The preregistered eleven-class recognition gate did not pass. Mixed-degradation
development evaluation and expert-composition experiments remain blocked. The
already-passed five-class singleton fail-closed router remains the deployable
routing path; mixed or uncertain inputs must continue to abstain.

No FoundIR development or internal-test record was used in either attempt.

## V1 frozen gate

- Holdout: 990 source-disjoint training-only records, 90 per class.
- Correct accepted: 801 (80.9091%).
- Wrong accepted: 2 (0.2020%).
- Safe refusals: 187 (18.8889%).
- Coverage: 81.1111%; accepted accuracy: 99.7509%.
- Failure: `blur+low_light` and `low_light` each had one wrong route, or
  1.1111% per class versus the frozen maximum of 1%.

## Bounded V2 retry

V2 was limited to one simple change selected only on the V1 calibration split:
`C=0.5`, temperature `2.0`, and threshold `0.95`. It used a fresh holdout of 440
training-only records (40 per class) whose source groups were absent from the
complete V1 partition. V1 holdout records were quarantined.

- Correct accepted: 359 (81.5909%).
- Wrong accepted: 1 (0.2273%).
- Safe refusals: 80 (18.1818%).
- Coverage: 81.8182%; accepted accuracy: 99.7222%.
- Ten of eleven classes had zero wrong routes.
- All class coverages were at least 65%.
- Failure: Rain had one wrong route among 40 records (2.5%), exceeding the
  frozen maximum of 1%.
- The failure was `foundir_train_0396679`: truth `rain`, accepted prediction
  `noise` at 0.955864 confidence.

## Terminal interpretation

The frozen Qwen representation plus a linear head is highly accurate, but it
has not demonstrated the required per-class fail-closed safety for mixed
routing. Raising a threshold after observing this holdout would be test-set
tuning and is not permitted. Additional C/threshold sweeps would repeat the
optimization loop, so this bounded line is stopped.

## Official-test prohibition

The FoundIR official test must never be accessed unless the advisor explicitly
authorizes official-test access in a written, auditable decision. User intent,
generic permission to continue, final-validation language, existing archives,
or an unrelated advisor approval do not qualify. No official-test data was
accessed here. The repository-wide binding policy is
`FOUNDir_OFFICIAL_TEST_ACCESS_POLICY.md`.

## Frozen artifacts

- V2 result SHA-256:
  `b8b0ba57641a9c53bf6a28a400c07fec4ab08c1c26e0b435074ad3bbbdde192c`
- V2 head SHA-256:
  `8cf7fb6183c78e6804698ab838d71858d8c4903d2d9a5eb9ce2b6a080c715f0a`
- V2 manifest SHA-256:
  `b58c08a63ff42d06cf2f3f54b9a3838cadaa94359c4f6c5cca7e030e3d012ecc`
- V2 config SHA-256:
  `ff4e43102cc1e533e425d6792dfc8b936fa05db3e39091da4d1cd44b1246e838`
