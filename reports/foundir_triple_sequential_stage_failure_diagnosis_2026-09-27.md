# FoundIR triple sequential-stage failure diagnosis — 2026-09-27

**DIAGNOSTIC ONLY / 非主线。** This report uses only the frozen 660-image development-validation artifacts. It did not run inference, update model weights, or access the official FoundIR test.

## Scope

- Category: `18Lowlight_Blur_Noise` (`Blur + Noise + Low-light`).
- Records: 53 correctly routed full-resolution images; the seven Low-light fallback records are excluded.
- Frozen project sequence: `Noise → Low-light → Blur`.
- PSNR is recomputed on full-image RGB uint8 pixels and is checked exactly against the frozen final sidecars.
- Stage LPIPS was not recomputed; the table reuses existing LPIPS only where frozen sidecars already exist.

## Stage means

| Stage | PSNR | Existing LPIPS | Luma mean | Luma bias vs GT | Luma MAE vs GT | Gradient error vs GT | Laplacian error vs GT | Dark clip | Bright clip |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Degraded input | 10.110227 | — | 40.779 | -74.054 | 74.055 | 3.264 | 7.230 | 1.738% | 0.000% |
| After Noise expert | 10.180086 | — | 41.470 | -73.364 | 73.364 | 3.286 | 7.084 | 1.171% | 0.000% |
| After Low-light expert | 16.987595 | — | 144.337 | 29.503 | 32.920 | 2.834 | 7.288 | 0.000% | 0.000% |
| After Blur expert (project final) | 16.715288 | 0.181321 | 145.136 | 30.302 | 33.634 | 3.055 | 7.790 | 0.000% | 0.000% |
| Released FoundIR Generalist | 24.986867 | 0.038079 | 99.400 | -15.434 | 15.473 | 1.416 | 3.921 | 0.031% | 0.000% |
| Full FoundIR Generalist → Low-light specialist | 33.238410 | 0.032058 | 117.962 | 3.129 | 8.034 | 1.255 | 3.768 | 0.119% | 0.191% |

## Sequential transition diagnosis

| Transition | Mean ΔPSNR | Median ΔPSNR | Improved / worsened / unchanged | Δ luma MAE | Δ gradient error | Δ Laplacian error |
|---|---:|---:|---:|---:|---:|---:|
| Degraded input → After Noise expert | 0.069859 | 0.071220 | 52 / 1 / 0 | -0.690 | 0.022 | -0.145 |
| After Noise expert → After Low-light expert | 6.807510 | 6.933211 | 50 / 3 / 0 | -40.444 | -0.452 | 0.204 |
| After Low-light expert → After Blur expert (project final) | -0.272308 | -0.304675 | 4 / 49 / 0 | 0.714 | 0.221 | 0.502 |

## Evidence-based conclusion

1. The Noise expert is nearly inert on this triple domain: it adds only `+0.069859 dB`. It slightly reduces Laplacian error but increases gradient error.
2. The Low-light expert supplies almost all project recovery (`+6.807510 dB`), but it overshoots GT exposure by `+29.503` mean luma. Its output remains `-7.999272 dB` below the Released FoundIR Generalist.
3. The final Blur expert is consistently destructive: `49/53` records lose PSNR, the mean change is `-0.272308 dB`, and luma, gradient, and Laplacian errors all increase.
4. Skipping Blur cannot solve the failure: the post-Low-light mean is only `16.987595 dB`. The final project output trails the Released Generalist by `-8.271580 dB` and Full FoundIR by `-16.523122 dB` on the same 53 correctly routed records.

Therefore routing error is ruled out for this subset, while expert ordering is only a secondary problem. The dominant limitation is that independently trained singleton experts do not perform joint composite restoration on the out-of-domain triple input/intermediate states; their small gains and errors accumulate across the cascade. This evidence supports testing an interaction-aware or joint-composition mechanism, not merely another router threshold or a reordered singleton cascade.

The closest successful reproduction is already present in these artifacts: FoundIR's Generalist jointly restores the composite and its Low-light specialist then corrects the residual (`24.986867 → 33.238410 dB`). The next project experiment should isolate whether a compatibility mechanism learned on double compositions can make frozen singleton experts compose on a held-out triple. A direct triple-trained specialist may improve metrics, but it would not by itself constitute a credible novelty claim.

## Artifacts

- JSON: `reports/foundir_triple_sequential_stage_failure_diagnosis_2026-09-27.json`
- Per-record CSV: `reports/foundir_triple_sequential_stage_failure_diagnosis_2026-09-27.csv`
