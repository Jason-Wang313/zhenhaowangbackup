# DIAGNOSTIC ONLY / 非主线 — Official SOW600 Released FoundIR comparison

Completed at `2026-09-25T01:38:23Z` (`2026-09-25 09:38:23` Shanghai time).
This comparison does not replace the terminal project v8 result. It evaluates
the released FoundIR baseline on the exact same 600 official singleton pairs
and official metric protocol.

## Terminal result

| Category | Records | Project PSNR / LPIPS | FoundIR Generalist PSNR / LPIPS | Full FoundIR PSNR / LPIPS | Project minus Full FoundIR | Conclusion |
|---|---:|---:|---:|---:|---:|---|
| Blur | 150 | 28.092635 / .144089 | 25.577556 / .180644 | 25.577556 / .180644 | +2.515080 dB / -.036555 | Project wins both |
| Noise | 100 | 40.048323 / .056601 | 39.807950 / .064671 | 39.807950 / .064671 | +.240373 dB / -.008070 | Project wins both |
| Haze | 200 | 22.677420 / .110437 | 17.842509 / .227461 | 19.084373 / .224938 | +3.593046 dB / -.114500 | Project wins both |
| Rain | 100 | 37.809175 / .052901 | 34.293876 / .079429 | 34.293876 / .079429 | +3.515298 dB / -.026529 | Project wins both |
| Low-light | 50 | 19.322459 / .148799 | 13.548912 / .291033 | 18.240826 / .238644 | +1.081634 dB / -.089846 | Project wins both |
| 600-image overall | 600 | 29.168753 / .103485 | 25.821272 / .169251 | 26.626220 / .164044 | +2.542533 dB / -.060559 | Project wins both |

The project wins both PSNR and LPIPS against Full FoundIR in all five official
singleton categories. The smallest PSNR lead is Noise at `+.240373 dB`; the
smallest LPIPS lead is also Noise at `.008070`. The user's singleton objective
is therefore already satisfied by the frozen terminal v8 project system. No
additional singleton training is justified merely to beat released FoundIR.

## FoundIR routing and protocol

- Generalist ran on all 600 official LQ images.
- Weather Specialist ran after Generalist on all 200 Haze images.
- Low-light Specialist ran after Generalist on all 50 Low-light images.
- Blur, Noise, and Rain have no documented routed specialist in this protocol
  and retain Generalist output.
- Metrics: original-size RGB, no resize/crop/border shave, `pyiqa` 0.1.16,
  unweighted arithmetic mean per image within each category.
- Project values are the hash-bound terminal v8 official metrics from canonical
  commit `5003e1e51bf9373fbdb352cc0420d712d396f114`.
- No training, checkpoint selection, or model-weight update occurred.

## Provenance and audit

- Config:
  `configs/foundir_official_sow600_released_full_pipeline_v1.json`, SHA-256
  `3dff5bb4341d3f97c15470fe2d8bfaab7c9b22af051c13171d38fa986224802d`.
- Result:
  `checkpoints/foundir_official_sow600_released_full_pipeline_v1/result.json`,
  SHA-256 `afdef408521989bc449497c637403642612a9687e5004878f4687a765532b9d6`.
- Generalist checkpoint SHA-256:
  `2603a0be47d0844d4c9cea542ecca039e0e4aaa39d45d41064bfbdc13c0830c9`.
- Weather checkpoint SHA-256:
  `25fa90003de8fa6965d861fc66159fb0c188654723acc15e8c49ec8359ca8b93`.
- Low-light checkpoint SHA-256:
  `f834248e145d020d5a02f257b7ceb383b579add2522932938b3e1e0bd460abe6`.
- Artifacts: 600 Generalist PNGs and metadata sidecars, 250 Specialist PNGs
  and metadata sidecars, and 600 metric sidecars.
- Independent audit rehashed all 850 outputs, checked dimensions and specialist
  ancestry, verified all finite metrics, independently recomputed every
  category aggregate, and returned `INDEPENDENT_AUDIT_PASS`.

## Runtime

- Hardware: one homogeneous 8x NVIDIA H200 node.
- Start: `2026-09-25T01:19:59Z`.
- End: `2026-09-25T01:38:23Z`.
- End-to-end wall time: 18 minutes 24 seconds.
- Aggregate GPU time: approximately 2.45 GPU-hours.
- Terminal status: 0.

The next research stage is double-composition development. Official sealed
singleton data and metrics must not be used as training examples, gradients,
or iterative checkpoint-selection data.
