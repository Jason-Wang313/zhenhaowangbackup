# DIAGNOSTIC ONLY / 非主线 — FoundIR full-pipeline composition result

Date: 2026-09-24

This is a development-only comparison. It is not the project mainline result,
not an official FoundIR test result, and not an SOW acceptance result. The
already-terminal official SOW600 v8 result remains unchanged.

## Scope and identities

- Split: frozen 660-image full-resolution composition validation set, 60
  records in each of 11 categories.
- Project system: the frozen routed project pipeline from the original
  comparison.
- FoundIR baseline: released Generalist output followed by the documented
  degradation-aware specialist where applicable.
- Weather specialist: 120 records in `08Haze` and `13Rain_Haze`.
- Low-light specialist: 240 records in `09Lowlight_Haze`, `14Lowlight`,
  `15Lowlight_Blur`, and `18Lowlight_Blur_Noise`.
- The remaining 300 records retain the released Generalist output.
- Model-weight updates: none.
- Official FoundIR test access: false.

Frozen artifacts:

- Config: `configs/foundir_composition_whole_image_full_pipeline_v1.json`
- Config SHA-256: `c1dc876671622fd4d0541b52ca28a6cf95c036f181249d275ba370c98d835760`
- Result: `checkpoints/foundir_composition_whole_image_full_pipeline_v1/result.json`
- Result SHA-256: `ab4a1b0330c732432072917dca4ba74012bf7efe166e347e43aee4461a1e09e6`
- Weather checkpoint SHA-256:
  `25fa90003de8fa6965d861fc66159fb0c188654723acc15e8c49ec8359ca8b93`
- Low-light checkpoint SHA-256:
  `f834248e145d020d5a02f257b7ceb383b579add2522932938b3e1e0bd460abe6`

## Terminal result

| 660-image overall | Project | Released FoundIR Generalist | Full FoundIR Generalist → Specialist | Project minus Full FoundIR |
|---|---:|---:|---:|---:|
| PSNR ↑ | 24.914822 | 24.757710 | 27.169232 | -2.254410 dB |
| LPIPS ↓ | .172793 | .176648 | .167938 | +.004855 |

The earlier Generalist-only aggregate had the project ahead by `+.157112 dB`
PSNR and `-.003854` LPIPS. After applying the documented specialists, Full
FoundIR leads the project overall by `2.254410 dB` PSNR and `.004855` LPIPS.
Therefore the Generalist-only result must not be used to claim that the project
beats the complete FoundIR paper pipeline.

## Composition-size result

| Composition | Project PSNR / LPIPS | Full FoundIR PSNR / LPIPS | Conclusion |
|---|---:|---:|---|
| Single, 300 | 28.714078 / .141957 | 28.852611 / .168653 | Project LPIPS better; Full FoundIR PSNR +.138532 dB |
| Double, 300 | 22.710305 / .202722 | 24.184117 / .194418 | Full FoundIR better on both metrics |
| Triple, 60 | 16.941122 / .177332 | 33.677913 / .031967 | Full FoundIR substantially better on both metrics |

## Per-category result

| Category | Project PSNR / LPIPS | Full FoundIR PSNR / LPIPS | Conclusion |
|---|---:|---:|---|
| Blur | 23.359381 / .250501 | 23.836034 / .322237 | Project LPIPS better; Full FoundIR PSNR better |
| Blur + Noise | 44.446832 / .014591 | 44.733364 / .013190 | Full FoundIR better on both |
| Noise | 40.133486 / .096746 | 42.655305 / .067910 | Full FoundIR better on both |
| Haze | 24.214152 / .090172 | 19.799497 / .150095 | Project better on both |
| Low-light + Haze | 13.112670 / .251747 | 20.939309 / .159126 | Full FoundIR better on both |
| Rain | 37.983646 / .075819 | 36.284786 / .070384 | Project PSNR better; Full FoundIR LPIPS better |
| Rain + Haze | 21.181264 / .158861 | 19.192118 / .210446 | Project better on both |
| Low-light | 17.879726 / .196546 | 21.687430 / .232637 | Project LPIPS better; Full FoundIR PSNR better |
| Low-light + Blur | 16.768149 / .304552 | 21.363892 / .258885 | Full FoundIR better on both |
| Low-light + Noise | 18.042611 / .283859 | 14.691902 / .330445 | Project better on both |
| Low-light + Blur + Noise | 16.941122 / .177332 | 33.677913 / .031967 | Full FoundIR substantially better on both |

Across the 11 category means, the project wins both metrics in three
categories, Full FoundIR wins both in five, and three categories split PSNR
and LPIPS.

## Runtime and audit

- Hardware: one homogeneous 8x NVIDIA H200 node.
- Start: `2026-09-24T23:11:09Z`.
- End: `2026-09-24T23:16:22Z`.
- End-to-end wall time: 5 minutes 13 seconds.
- Aggregate GPU time: approximately 0.70 GPU-hours.
- Artifacts: 360 specialist PNGs, 360 specialist metadata sidecars, and 660
  final metric sidecars.
- Independent audit rehashed all 360 specialist outputs against metadata,
  checked all 660 metric sidecars for finite metrics and protected-test flags,
  and returned `FULL_PIPELINE_INDEPENDENT_AUDIT_PASS`.
- A second launch attempt was rejected by the process lock at
  `2026-09-24T23:11:40Z`; it did not affect the healthy primary run, which
  ended with status 0.
