# FoundIR composition integrated Restormer v2 — terminal result

**DIAGNOSTIC ONLY / 非主线.** This is a full-resolution, 660-record validation
comparison for the paper table. It does not replace the terminal official
singleton SOW600 result and does not access the official FoundIR test.

## Contract

- Manifest: 660 validation records, 60 in each of 11 categories, SHA-256
  `a94128195ffd219f7e94b44c1ec98291c75e727f17e2c3af88cf1c2f9b6282a9`.
- New deployment change: when the frozen router accepts final route
  `haze+low_light`, use the project-trained Low-light+Haze Restormer step 6000.
- Frozen router outcomes inside `09Lowlight_Haze`: 53 accepted correct routes,
  5 `low_light_fallback` routes, and 2 abstentions. Only the 53 accepted routes
  use the new direct expert. The fallback and abstention behavior remains
  fail-closed.
- All other 600 records retain their prior frozen routed project outputs.
- Baselines are the immutable Released FoundIR Generalist and complete Released
  FoundIR Generalist -> Specialist outputs from the prior terminal comparison.
- Metrics: full RGB uint8-image PSNR and area-weighted non-overlapping 512-tile
  LPIPS.
- No training or model-weight update occurred in this run.

This execution is a hash-verified artifact replay, not fresh inference for the
607 unchanged records. Every source sidecar and project output is bound by
SHA-256 in the new per-record artifact. This is mathematically identical to
rerunning deterministic unchanged stages, while avoiding approximately 10.75
hours of redundant 8-GPU inference. A paper must disclose this provenance; it
must not describe the run as fresh inference of all 660 images.

## Overall result

| 660-image overall | PSNR | LPIPS |
|---|---:|---:|
| Project, route-aware integrated Restormer v2 | 25.647200 | .168062 |
| Released FoundIR Generalist | 24.757710 | .176648 |
| Full FoundIR Generalist -> Specialist | 27.169232 | .167938 |
| Project minus Generalist | +.889490 dB | -.008586 |
| Project minus Full FoundIR | -1.522032 dB | +.000124 |

The integrated project clearly beats the Released Generalist on both metrics.
Against Full FoundIR, LPIPS is effectively tied but remains worse by
`.000124`; PSNR remains lower by `1.522032 dB`.

## Composition-size result

| Composition | Project PSNR / LPIPS | Generalist PSNR / LPIPS | Full FoundIR PSNR / LPIPS | Project minus Full FoundIR |
|---|---:|---:|---:|---:|
| Single, 300 | 28.714078 / .141957 | 27.232394 / .174412 | 28.852611 / .168653 | -.138532 dB / -.026696 |
| Double, 300 | 24.321537 / .192313 | 22.245454 / .206662 | 24.184117 / .194418 | +.137420 dB / -.002105 |
| Triple, 60 | 16.941122 / .177332 | 24.945571 / .037753 | 33.677913 / .031967 | -16.736791 dB / +.145365 |

The route-aware double aggregate now beats Full FoundIR on both metrics. The
triple category is the dominant remaining failure and is unchanged by this
experiment.

## Per-category result

| Category | Project PSNR / LPIPS | Generalist PSNR / LPIPS | Full FoundIR PSNR / LPIPS |
|---|---:|---:|---:|
| Blur | 23.359381 / .250501 | 23.836034 / .322237 | 23.836034 / .322237 |
| Blur + Noise | 44.446832 / .014591 | 44.733364 / .013190 | 44.733364 / .013190 |
| Noise | 40.133486 / .096746 | 42.655305 / .067910 | 42.655305 / .067910 |
| Haze | 24.214152 / .090172 | 19.363806 / .139754 | 19.799497 / .150095 |
| Low-light + Haze | 21.168830 / .199704 | 15.763173 / .202570 | 20.939309 / .159126 |
| Rain | 37.983646 / .075819 | 36.284786 / .070384 | 36.284786 / .070384 |
| Rain + Haze | 21.181264 / .158861 | 18.739283 / .198891 | 19.192118 / .210446 |
| Low-light | 17.879726 / .196546 | 14.022038 / .271774 | 21.687430 / .232637 |
| Low-light + Blur | 16.768149 / .304552 | 17.299545 / .288214 | 21.363892 / .258885 |
| Low-light + Noise | 18.042611 / .283859 | 14.691902 / .330445 | 14.691902 / .330445 |
| Low-light + Blur + Noise | 16.941122 / .177332 | 24.945571 / .037753 | 33.677913 / .031967 |

The deployed Low-light+Haze result is `21.168830/.199704`, rather than the
capacity-only all-60 Restormer number `22.215445/.182860`, because five frozen
fallbacks and two abstentions are preserved. It beats the Generalist on both
metrics and Full FoundIR on PSNR, but not Full FoundIR LPIPS.

## Terminal identity and audit

- Config:
  `configs/foundir_composition_integrated_restormer_v2.json`, SHA-256
  `c30581a701f6b08b6c66503b7b681bcc1f386ca43799a74d387367ab280ad6ad`.
- Result:
  `checkpoints/foundir_composition_integrated_restormer_v2/result.json`,
  SHA-256 `511940015fb3e3898f7da4ce7f31958dfb7f0bf9175062002de801cd2fcf7c1f`.
- Exactly 660 per-record sidecars; all have `status=complete`, the same config
  SHA-256, `official_foundir_test_accessed=false`, and
  `fresh_inference=false`.
- Route counts: 53 replacement, 5 fallback, 2 abstain, 600 unchanged other.
- Recomputed per-record means match the terminal result.
- Temporary files: zero.
- Start/end: `2026-09-26T10:40:24Z` to `10:41:02Z`, 38 seconds wall time.
- GPU wall time and GPU-hours: zero.
- Terminal marker:
  `FOUNDIR_COMPOSITION_INTEGRATED_RESTORMER_V2_END status=0`.

## Research decision

Freeze the double aggregate: it now beats Full FoundIR on both metrics. Do not
continue adding category-specific double specialists merely to improve the
margin. The next paper-novelty investigation should use the unchanged triple
`Low-light+Blur+Noise` failure as its primary diagnostic target, beginning with
literature review and controlled degradation-order/failure analysis rather
than an automatic new training run.
