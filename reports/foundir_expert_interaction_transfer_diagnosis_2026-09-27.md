# FoundIR expert-interaction transfer diagnosis — 2026-09-27

**DIAGNOSTIC ONLY / 非主线。** Existing full-resolution development-validation artifacts only; no inference, training, or official-test access.

## Same transition on double compositions

| Transition/domain | N | Before PSNR | After PSNR | Mean ΔPSNR | Improved / worsened | Luma bias before → after | Δ gradient error | Δ Laplacian error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Noise → Low-light on Low-light+Noise | 43 | 8.502571 | 18.402203 | +9.899631 | 43 / 0 | -91.074 → +0.057 | -0.447 | -1.139 |
| Low-light → Blur on Low-light+Blur | 52 | 16.566712 | 16.691903 | +0.125191 | 32 / 20 | +23.066 → +23.387 | -0.087 | +0.056 |

## Double-to-triple transfer

| Transition | Double mean ΔPSNR | Triple mean ΔPSNR | Triple minus double | Double improved | Triple improved |
|---|---:|---:|---:|---:|---:|
| Noise → Low-light | +9.899631 | +6.807510 | -3.092122 | 43/43 | 50/53 |
| Low-light → Blur | +0.125191 | -0.272308 | -0.397499 | 32/52 | 4/53 |

## Interpretation rule

A large double-to-triple deterioration supports composition-domain interaction failure. Similar behavior in both domains instead means the expert transition itself is weak and should not be framed as a triple-specific generalization failure. This comparison is observational because the image sets differ; it is not a paired causal estimate.

Machine-readable evidence: `reports/foundir_expert_interaction_transfer_diagnosis_2026-09-27.json`.
