# Frozen-Qwen ridge Lowlight validation failure

Date: 2026-08-16 UTC  
Scope: Lowlight degradation-severity assessment only

## Result

The scalar ridge head passed its 300-record train-source selection gate but
failed the fixed 25-source validation gate. It is not approved for restoration
conditioning.

- selection accuracy / MAE: `0.74 / 0.2633333`
- selection recall L1-L5: `0.90, 0.65, 0.70, 0.5667, 0.8833`
- fixed-validation accuracy / MAE: `0.48 / 0.52`
- fixed-validation baseline accuracy / MAE: `0.64 / 0.40`
- fixed-validation recall L1-L5: `0.80, 0.40, 0.60, 0.00, 0.60`
- all five output levels were represented, but none of the five true L4
  records was classified as L4: four were classified as L3 and one as L5
- internal test used: `false`
- official FoundIR used: `false`

Authoritative artifacts:

- metrics: `checkpoints/qwen_lowlight_ridge_head/metrics.json`
- metrics SHA-256:
  `4baf465aab731a845249a3bdbbdbec7e642439bae892686a0a7f5b09f6c5b0fd`
- selection metrics SHA-256:
  `c9c90778dae9af1e6e746b477fc8562417b4dc7d06052b8ab97b56dfd5863821`
- checkpoint SHA-256:
  `643ba5ab047b7930c3a1eaef002e1d62300ac66da69f8fdfbc534f2570ac3a93`
- log SHA-256:
  `a3cf05eb47f442524acbf99593f06eeec220942e9f7094d9e3f854f5c05813db`
- code revision:
  `60485559d543852376a8aa405e1ec9ecae303790`

## Audit and diagnosis

No label-path, image-hash, source-isolation, cache-provenance, or output-parser
fault was found. The cached features are bound to the degraded-image SHA-256,
the frozen Qwen revision, the extractor hash, and the pooling configuration.
The model produced valid structured output for every record and predicted all
five levels. This failure is therefore different from the earlier zero-shot
L5-to-L4 collapse.

The dominant problem is high-dimensional overfit and an optimistic single
selection split:

- fitted examples: `1,200` from `240` complete train sources
- frozen-Qwen feature dimension: `7,168`
- fitted-head accuracy: `0.9392`
- five-fold source-grouped accuracy over the `300` cached train sources:
  `0.6767 +/- 0.0287`
- five-fold source-grouped MAE: `0.3333`
- matching luminance-centroid cross-validation accuracy / MAE:
  `0.6467 / 0.4187`
- mean grouped-CV recall L1-L5:
  `0.843, 0.603, 0.650, 0.497, 0.790`

The original `0.74` selection result was materially above the multi-fold
estimate. The fixed validation set is also small, with five records per level,
so one difficult source changes a per-level recall by 20 percentage points.
This does not excuse the failure: the predeclared gate failed and remains
failed.

The validation L4 raw ridge scores were `3.2070`, `3.2176`, `5.1621`,
`2.6647`, and `2.9019`. With nearest-integer decoding, four fall below the
L3/L4 boundary and one exceeds the L4/L5 boundary. Train-only nested threshold
calibration improved grouped-CV accuracy only slightly (`0.6767` to about
`0.6833`), so changing rounding thresholds is not a sufficient repair.

Dataset composition did not reveal a path-selection error:

- fitted sources: LSDIR `139`, DIV2K `88`, MiO100 `13`
- selection sources: LSDIR `25`, DIV2K `28`, MiO100 `7`
- fixed-validation sources: LSDIR `9`, DIV2K `15`, MiO100 `1`

## Leakage status and next decision

The 25 fixed-validation source IDs in this artifact are now exposed. They may
be used for diagnosis, but they must not be reused as an unbiased final gate
for a revised head.

The next bounded attempt should retain frozen Qwen and the same pooling schema,
expand feature extraction to all `1,600` Lowlight train sources (`8,000`
records), choose regularization using only source-grouped train folds, and then
evaluate once on a newly frozen validation-source subset that explicitly
excludes these 25 exposed sources. The internal test and FoundIR must remain
untouched. If the larger frozen-feature head still does not generalize, the
next escalation is a bounded Qwen vision-LoRA experiment rather than further
post-hoc tuning on validation.

Noise, Blur, Haze, and Rain remain `blocked_no_task_calibration`; this result
does not run or validate those tasks.
