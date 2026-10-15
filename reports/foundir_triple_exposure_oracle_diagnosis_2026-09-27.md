# FoundIR triple scalar-exposure oracle diagnosis — 2026-09-27

**DIAGNOSTIC ONLY / 非主线。** The scalar uses GT and is therefore non-deployable. It is an upper-bound diagnosis, not a restoration result. Existing full-resolution development-validation artifacts only; no inference, training, or official-test access.

| Stage | Original PSNR | Mean luma-match α | Luma-match PSNR / gain | Mean least-squares α | Least-squares PSNR / gain |
|---|---:|---:|---:|---:|---:|
| after_low_light | 16.987595 | 0.792585 | 20.364603 / +3.377008 | 0.824301 | 20.631794 / +3.644199 |
| after_blur | 16.715288 | 0.788380 | 20.229229 / +3.513942 | 0.818143 | 20.477888 / +3.762601 |

Interpretation: a large oracle gain means exposure overshoot is an important correctable component. The remaining gap after the least-squares scalar is a lower bound on non-scalar color/structure/detail error under this restricted correction family; it must not be called an exact causal decomposition.

Machine-readable evidence: `reports/foundir_triple_exposure_oracle_diagnosis_2026-09-27.json`.
